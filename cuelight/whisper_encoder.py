"""CueLight-Base, a Whisper encoder made frame-causal (left-padded convs and causal attention) with per-frame
end-of-turn heads. It streams one 20 ms frame at a time with a key/value cache per layer.

Long streams use windows of `chunk` frames. When one fills up, the last `context` frames are re-encoded as the start
of the next one (positions restart), which is exactly how the model was evaluated. The streaming detector's reset at each new user
turn normally starts a fresh window long before that.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cuelight.features import STACK


class _Attention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d, bias=False)
        self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d)


class _Layer(nn.Module):
    def __init__(self, d: int, ffn: int):
        super().__init__()
        self.self_attn = _Attention(d)
        self.self_attn_layer_norm, self.final_layer_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ffn), nn.Linear(ffn, d)


class WhisperTurnModel(nn.Module):
    def __init__(self, n_mels: int = 80, d: int = 768, layers: int = 12, heads: int = 12, ffn: int = 3072,
                 max_positions: int = 1500, n_out: int = 6, chunk: int = 1500, context: int = 750):
        super().__init__()
        self.n_mels, self.d, self.n_heads = n_mels, d, heads
        self.chunk, self.context = min(chunk, max_positions), context
        self.conv1 = nn.Conv1d(n_mels, d, 3)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2)
        self.embed_positions = nn.Embedding(max_positions, d)
        self.layers = nn.ModuleList(_Layer(d, ffn) for _ in range(layers))
        self.layer_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def _window(self, x):
        """x (B, T, 2 * n_mels): encode a window from its start -> (final states, streaming state at its end)."""
        b, t, _ = x.shape
        m = x.reshape(b, t * STACK, self.n_mels).transpose(1, 2)
        c1 = F.gelu(F.conv1d(F.pad(m, (2, 0)), self.conv1.weight, self.conv1.bias))
        h = F.gelu(F.conv1d(F.pad(c1, (1, 0)), self.conv2.weight, self.conv2.bias, stride=2))
        h = h.transpose(1, 2) + self.embed_positions.weight[:t]
        kv = []
        for layer in self.layers:
            a = layer.self_attn
            y = layer.self_attn_layer_norm(h)
            q, k, v = (p(y).view(b, t, self.n_heads, -1).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            kv.append((k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + a.out_proj(o.transpose(1, 2).reshape(b, t, self.d))
            h = h + layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        return self.layer_norm(h), {"x": x, "mel": m[..., -2:], "c1": c1[..., -1:], "kv": kv}

    def forward(self, x, state=None):
        """x (B, T, 2 * n_mels) -> (logits (B, T, n_out), None), whole sequence at once."""
        t = x.shape[1]
        outs, s, done = [], 0, 0
        while done < t:
            e = min(s + self.chunk, t)
            outs.append(self._window(x[:, s:e])[0][:, done - s:])
            done, s = e, e - self.context
        return self.head(torch.cat(outs, 1)), None

    def init_state(self, b: int = 1) -> dict:
        return {"x": None}

    def step(self, f, state):
        """f (B, 2 * n_mels) one frame -> ((B, n_out) logits for this frame, new state)."""
        if state["x"] is None:
            h, st = self._window(f[:, None])
            return self.head(h[:, -1]), st
        n = state["x"].shape[1]
        if n == self.chunk:
            _, state = self._window(state["x"][:, n - self.context:])
            n = self.context
        b = f.shape[0]
        mel = f.view(b, STACK, self.n_mels).transpose(1, 2)
        c1 = F.gelu(F.conv1d(torch.cat([state["mel"], mel], -1), self.conv1.weight, self.conv1.bias))
        h = F.gelu(F.conv1d(torch.cat([state["c1"], c1], -1), self.conv2.weight, self.conv2.bias, stride=2))
        h = h.transpose(1, 2) + self.embed_positions.weight[n:n + 1]
        kv = []
        for layer, (pk, pv) in zip(self.layers, state["kv"]):
            a = layer.self_attn
            y = layer.self_attn_layer_norm(h)
            q, k, v = (p(y).view(b, 1, self.n_heads, -1).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            kv.append((k, v))
            o = F.scaled_dot_product_attention(q, k, v)
            h = h + a.out_proj(o.transpose(1, 2).reshape(b, 1, self.d))
            h = h + layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        logits = self.head(self.layer_norm(h)[:, 0])
        return logits, {"x": torch.cat([state["x"], f[:, None]], 1), "mel": mel, "c1": c1[..., -1:], "kv": kv}
