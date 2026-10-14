"""CueLight-Nano models: a causal depthwise-conv front end, recurrent blocks, and per-frame heads.

    init_state(batch)  -> state                   fixed size, independent of how long the stream has run
    step(x_t, state)   -> (logits_t, new_state)   x_t (B, F) one 20 ms feature frame; logits_t (B, 2 + ...)
    forward(x)         -> (logits (B, T, ...), _) the same outputs for a whole sequence at once

logits[..., 0] is "the user is speaking", logits[..., 1] is "the user has finished their turn" (apply a sigmoid).
"""
from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F


class CausalDSConv(nn.Module):
    """Depthwise-separable 1-D conv with left padding only. Streaming keeps the last k-1 inputs."""

    def __init__(self, d_in: int, d_out: int, k: int = 3):
        super().__init__()
        self.k = k
        self.dw = nn.Conv1d(d_in, d_in, k, groups=d_in)
        self.pw = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):  # (B, T, C)
        y = self.dw(F.pad(x.transpose(1, 2), (self.k - 1, 0)))
        return F.gelu(self.pw(y)).transpose(1, 2)

    def init_state(self, b, d_in, device=None):
        return torch.zeros(b, d_in, self.k - 1, device=device)

    def step(self, x, buf):  # x (B, C), buf (B, C, k-1)
        win = torch.cat([buf, x.unsqueeze(-1)], dim=-1)
        y = F.gelu(self.pw(self.dw(win))).squeeze(-1)
        return y, win[:, :, 1:]


class GRUCore(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rnn = nn.GRU(d, d, batch_first=True)

    def forward(self, x, h0=None):
        y, h = self.rnn(x, None if h0 is None else h0.unsqueeze(0))
        return y, h.squeeze(0)

    def init_state(self, b, device=None):
        return torch.zeros(b, self.d, device=device)

    def step(self, x, h):
        # Same weights as the sequence path, written out so it exports as plain matmuls.
        w_ih, w_hh = self.rnn.weight_ih_l0, self.rnn.weight_hh_l0
        b_ih, b_hh = self.rnn.bias_ih_l0, self.rnn.bias_hh_l0
        gi = x @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = gi.chunk(3, -1)
        h_r, h_z, h_n = gh.chunk(3, -1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        return h, h



class Mamba3Core(nn.Module):
    """Mamba-3-style selective SSM, SISO, in plain tensor ops (no custom kernels).

    Following Mamba-3 (arXiv 2603.15569): exponential-trapezoidal discretisation
        h_t = a_t h_{t-1} + (1 - lam_t) dt_t a_t B_{t-1} x_{t-1} + lam_t dt_t B_t x_t,   a_t = exp(dt_t (A_t + i theta_t))
    with complex state (data-dependent rotation, here as an explicit complex diagonal in re/im pairs),
    all of dt, lam, A, theta, B, C input-dependent, RMS-normed B/C plus learned biases, no short conv,
    and an output gate. y_t = Re(C_t^T h_t) + D x_t, per head of size P; B, C shared across heads.
    State per head: N/2 complex x P, plus the previous step's (B x) term for the trapezoid.
    """

    def __init__(self, d: int, expand: int = 2, heads: int = 4, n: int = 32, chunk: int = 16, ckpt: bool = True):
        super().__init__()
        self.d, self.di, self.h, self.n2, self.chunk, self.ckpt = d, expand * d, heads, n // 2, chunk, ckpt
        self.p = self.di // heads
        self.in_proj = nn.Linear(d, 2 * self.di + 2 * n + 3 * heads + heads * self.n2)
        self.b_bias = nn.Parameter(torch.zeros(heads, n))
        self.c_bias = nn.Parameter(torch.zeros(heads, n))
        self.D = nn.Parameter(torch.ones(heads, self.p))
        self.out_proj = nn.Linear(self.di, d)
        with torch.no_grad():  # dt ~ 20 ms-4 s timescales at 20 ms frames, mild decay, small rotations
            bias = self.in_proj.bias
            o = 2 * self.di + 2 * n
            dt = torch.exp(torch.empty(heads).uniform_(math.log(0.01), math.log(0.5)))
            bias[o:o + heads] = dt + torch.log(-torch.expm1(-dt))  # softplus^-1
            bias[o + heads:o + 2 * heads] = 0.0                     # lam = 0.5 (trapezoid)
            bias[o + 2 * heads:o + 3 * heads] = 0.5                 # A = -softplus(.)

    def _proj(self, u):
        di, n, h, n2 = self.di, 2 * self.n2, self.h, self.n2
        x, z, Bm, Cm, dt, lam, A, th = torch.split(self.in_proj(u), [di, di, n, n, h, h, h, h * n2], -1)
        shp = u.shape[:-1]
        x = x.reshape(*shp, h, self.p)
        Bm = F.rms_norm(Bm, (n,)).unsqueeze(-2) + self.b_bias  # (..., h, n)
        Cm = F.rms_norm(Cm, (n,)).unsqueeze(-2) + self.c_bias
        dt = F.softplus(dt)                                    # (..., h)
        lam = torch.sigmoid(lam)
        log_r = -dt * F.softplus(A)                            # Re(dt * A) < 0
        ang = dt.unsqueeze(-1) * th.reshape(*shp, h, n2)       # dt * theta, (..., h, n2)
        return x, z, Bm, Cm, dt, lam, log_r, ang

    @staticmethod
    def _bx(Bm, x, n2):
        # B (.., h, 2*n2) as n2 complex numbers times x (.., h, p) -> (.., h, n2, p) re/im
        br, bi = Bm[..., :n2], Bm[..., n2:]
        return br.unsqueeze(-1) * x.unsqueeze(-2), bi.unsqueeze(-1) * x.unsqueeze(-2)

    def init_state(self, b, device=None):
        # [h_re, h_im] (h * n2 * p each) + previous (B x) re/im + previous dt*(1-lam) is folded in at use time
        return torch.zeros(b, 4 * self.h * self.n2 * self.p, device=device)

    def _unpack(self, s):
        k = self.h * self.n2 * self.p
        sh = (-1, self.h, self.n2, self.p)
        return [s[:, i * k:(i + 1) * k].reshape(sh) for i in range(4)]

    def step(self, u, s):
        x, z, Bm, Cm, dt, lam, log_r, ang = self._proj(u)
        h_re, h_im, pb_re, pb_im = self._unpack(s)
        a_re = (torch.exp(log_r).unsqueeze(-1) * torch.cos(ang)).unsqueeze(-1)  # (b, h, n2, 1)
        a_im = (torch.exp(log_r).unsqueeze(-1) * torch.sin(ang)).unsqueeze(-1)
        bx_re, bx_im = self._bx(Bm, x, self.n2)
        k1 = ((1 - lam) * dt)[..., None, None]
        k2 = (lam * dt)[..., None, None]
        # a * (h + k1 * prev_bx) + k2 * bx
        t_re, t_im = h_re + k1 * pb_re, h_im + k1 * pb_im
        n_re = a_re * t_re - a_im * t_im + k2 * bx_re
        n_im = a_re * t_im + a_im * t_re + k2 * bx_im
        y = self._out(n_re, n_im, Cm, x, z)
        k = self.h * self.n2 * self.p
        return y, torch.cat([n_re.reshape(-1, k), n_im.reshape(-1, k), bx_re.reshape(-1, k), bx_im.reshape(-1, k)], -1)

    def _out(self, h_re, h_im, Cm, x, z):
        n2 = self.n2
        cr, ci = Cm[..., :n2].unsqueeze(-1), Cm[..., n2:].unsqueeze(-1)
        y = (cr * h_re - ci * h_im).sum(-2) + self.D * x  # Re(c h) summed over state, (.., h, p)
        y = y.reshape(*y.shape[:-2], self.di) * F.silu(z)
        return self.out_proj(y)

    def forward(self, u, s0=None):
        """Chunked parallel form in the 'attention' (SSD-like) layout: per chunk only (t, s, h, n2) and
        (h, n2, p) tensors are built, never a (time, h, n2, p) state per frame."""
        b, t, _ = u.shape
        n2 = self.n2
        x, z, Bm, Cm, dt, lam, log_r, ang = self._proj(u)
        k1, k2 = (1 - lam) * dt, lam * dt                                   # (b, t, h)
        s0 = self.init_state(b, u.device) if s0 is None else s0
        h_re, h_im, pb_re, pb_im = self._unpack(s0)                         # (b, h, n2, p)
        lr = log_r.unsqueeze(-1).expand(*log_r.shape, n2)                   # (b, t, h, n2)
        br, bi, cr_, ci_ = Bm[..., :n2], Bm[..., n2:], Cm[..., :n2], Cm[..., n2:]
        ys, c = [], self.chunk

        def body(st, h_re, h_im):
            sl = slice(st, st + c)
            m = lr[:, sl].shape[1]
            Lr, La = torch.cumsum(lr[:, sl], 1), torch.cumsum(ang[:, sl], 1)    # (b, m, h, n2)
            mask = torch.tril(torch.ones(m, m, device=u.device, dtype=u.dtype))[None, :, :, None, None]
            mag = torch.exp((Lr.unsqueeze(2) - Lr.unsqueeze(1)).clamp(max=0.0)) * mask   # M[t,s] = a_{s+1..t}
            da = La.unsqueeze(2) - La.unsqueeze(1)
            M_re, M_im = mag * torch.cos(da), mag * torch.sin(da)                # (b, t, s, h, n2)
            cr, ci = cr_[:, sl].unsqueeze(2), ci_[:, sl].unsqueeze(2)            # c_t
            cM_re, cM_im = cr * M_re - ci * M_im, cr * M_im + ci * M_re          # c_t M[t,s]
            xs, k1s, k2s = x[:, sl], k1[:, sl], k2[:, sl]
            bsr, bsi = br[:, sl], bi[:, sl]                                      # b_s
            # a_s and the previous step's b (b_{s-1}). At s = 0 it uses the carried state instead.
            ar = torch.exp(lr[:, sl]) * torch.cos(ang[:, sl])
            ai = torch.exp(lr[:, sl]) * torch.sin(ang[:, sl])
            bpr = torch.cat([torch.zeros_like(bsr[:, :1]), bsr[:, :-1]], 1) if st == 0 else torch.cat([br[:, st - 1:st], bsr[:, :-1]], 1)
            bpi = torch.cat([torch.zeros_like(bsi[:, :1]), bsi[:, :-1]], 1) if st == 0 else torch.cat([bi[:, st - 1:st], bsi[:, :-1]], 1)
            xp = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, :-1]], 1) if st == 0 else torch.cat([x[:, st - 1:st], xs[:, :-1]], 1)
            abr, abi = ar * bpr - ai * bpi, ar * bpi + ai * bpr                  # a_s b_{s-1}
            # scores S[t, s] = Re sum_n c_t M[t,s] (.)_s
            S1 = (cM_re * bsr.unsqueeze(1) - cM_im * bsi.unsqueeze(1)).sum(-1)    # (b, t, s, h)
            S2 = (cM_re * abr.unsqueeze(1) - cM_im * abi.unsqueeze(1)).sum(-1)
            if st == 0:
                S2 = S2.clone()
                S2[:, :, 0] = 0  # no previous step before the stream starts (handled via pb, which is 0)
            y = torch.einsum("btsh,bsh,bshp->bthp", S1, k2s, xs) + torch.einsum("btsh,bsh,bshp->bthp", S2, k1s, xp)
            if st == 0:  # carried previous (B x) from s0 enters at s = 0 through a_0
                q_re = cM_re[:, :, 0] * ar[:, :1] - cM_im[:, :, 0] * ai[:, :1]   # Re/Im of c_t M[t,0] a_0
                q_im = cM_re[:, :, 0] * ai[:, :1] + cM_im[:, :, 0] * ar[:, :1]
                y = y + k1s[:, :1].unsqueeze(-1) * (torch.einsum("bthn,bhnp->bthp", q_re, pb_re)
                                                    - torch.einsum("bthn,bhnp->bthp", q_im, pb_im))
            # carry-in from the previous chunk's state: Re sum_n c_t G_t h0, G_t = a_{0..t}
            g = torch.exp(Lr)
            g_re, g_im = g * torch.cos(La), g * torch.sin(La)
            cg_re, cg_im = cr_[:, sl] * g_re - ci_[:, sl] * g_im, cr_[:, sl] * g_im + ci_[:, sl] * g_re
            y = y + torch.einsum("bthn,bhnp->bthp", cg_re, h_re) - torch.einsum("bthn,bhnp->bthp", cg_im, h_im)
            y = y + self.D * xs
            # state at the chunk end: h = sum_s M[last, s] w_s + G_last h0
            Ml_re, Ml_im = M_re[:, -1], M_im[:, -1]                             # (b, s, h, n2)
            w1r, w1i = k2s.unsqueeze(-1) * bsr, k2s.unsqueeze(-1) * bsi          # coefficient of x_s
            w2r, w2i = k1s.unsqueeze(-1) * abr, k1s.unsqueeze(-1) * abi          # coefficient of x_{s-1}
            if st == 0:
                w2r, w2i = w2r.clone(), w2i.clone()
                w2r[:, 0] = 0
                w2i[:, 0] = 0
            e1r, e1i = Ml_re * w1r - Ml_im * w1i, Ml_re * w1i + Ml_im * w1r
            e2r, e2i = Ml_re * w2r - Ml_im * w2i, Ml_re * w2i + Ml_im * w2r
            n_re = torch.einsum("bshn,bshp->bhnp", e1r, xs) + torch.einsum("bshn,bshp->bhnp", e2r, xp)
            n_im = torch.einsum("bshn,bshp->bhnp", e1i, xs) + torch.einsum("bshn,bshp->bhnp", e2i, xp)
            gl_re, gl_im = g_re[:, -1].unsqueeze(-1), g_im[:, -1].unsqueeze(-1)
            n_re = n_re + gl_re * h_re - gl_im * h_im
            n_im = n_im + gl_re * h_im + gl_im * h_re
            if st == 0:  # pb enters at s = 0 with coefficient M[last,0] k1_0 a_0
                f_re = Ml_re[:, 0] * ar[:, 0] - Ml_im[:, 0] * ai[:, 0]
                f_im = Ml_re[:, 0] * ai[:, 0] + Ml_im[:, 0] * ar[:, 0]
                kk = k1s[:, 0].unsqueeze(-1)
                f_re, f_im = (kk * f_re).unsqueeze(-1), (kk * f_im).unsqueeze(-1)
                n_re = n_re + f_re * pb_re - f_im * pb_im
                n_im = n_im + f_re * pb_im + f_im * pb_re
            return self.out_proj(y.reshape(b, m, self.di) * F.silu(z[:, sl])), n_re, n_im

        for st in range(0, t, c):
            if self.ckpt and torch.is_grad_enabled() and self.training:  # recompute each chunk in backward
                y_c, h_re, h_im = torch.utils.checkpoint.checkpoint(body, st, h_re, h_im, use_reentrant=False)
            else:
                y_c, h_re, h_im = body(st, h_re, h_im)
            ys.append(y_c)
        kdim = self.h * n2 * self.p
        last_bx_re = br[:, -1].unsqueeze(-1) * x[:, -1].unsqueeze(-2)
        last_bx_im = bi[:, -1].unsqueeze(-1) * x[:, -1].unsqueeze(-2)
        s_out = torch.cat([h_re.reshape(b, kdim), h_im.reshape(b, kdim), last_bx_re.reshape(b, kdim),
                           last_bx_im.reshape(b, kdim)], -1)
        return torch.cat(ys, 1), s_out



class ConvSSMCore(nn.Module):
    """A short causal depthwise conv (k=4) + SiLU in front of an SSM core (B/C biases initialised at one)."""

    def __init__(self, d: int, inner: nn.Module, k: int = 4):
        super().__init__()
        self.d, self.k, self.inner = d, k, inner
        self.conv = nn.Conv1d(d, d, k, groups=d)
        for nm in ("b_bias", "c_bias"):
            if hasattr(inner, nm):
                nn.init.ones_(getattr(inner, nm))

    def init_state(self, b, device=None):
        return torch.cat([torch.zeros(b, self.d * (self.k - 1), device=device), self.inner.init_state(b, device)], -1)

    def _split(self, s):
        n = self.d * (self.k - 1)
        return s[:, :n].reshape(-1, self.d, self.k - 1), s[:, n:]

    def step(self, x, s):
        buf, si = self._split(s)
        win = torch.cat([buf, x.unsqueeze(-1)], -1)
        u = F.silu((win * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias)
        y, si = self.inner.step(u, si)
        return y, torch.cat([win[:, :, 1:].reshape(x.shape[0], -1), si], -1)

    def forward(self, x, s0=None):
        b = x.shape[0]
        s0 = self.init_state(b, x.device) if s0 is None else s0
        buf, si = self._split(s0)
        v = torch.cat([buf, x.transpose(1, 2)], -1)
        y, si = self.inner(F.silu(self.conv(v)).transpose(1, 2), si)
        return y, torch.cat([v[:, :, -(self.k - 1):].reshape(b, -1), si], -1)



class Block(nn.Module):
    def __init__(self, d: int, core: nn.Module, mlp_mult: int = 2):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.core = core
        self.up = nn.Linear(d, 2 * mlp_mult * d)
        self.down = nn.Linear(mlp_mult * d, d)

    def _mlp(self, x):
        a, g = self.up(self.n2(x)).chunk(2, -1)
        return self.down(a * F.silu(g))

    def forward(self, x, h0=None):
        y, h = self.core(self.n1(x), h0)
        x = x + y
        return x + self._mlp(x), h

    def step(self, x, h):
        y, h = self.core.step(self.n1(x), h)
        x = x + y
        return x + self._mlp(x), h


class TurnEndModel(nn.Module):
    def __init__(self, d_in: int = 128, d: int = 224, n_blocks: int = 3, core: str = "gru",
                 ssm_n: int = 320, mlp_mult: int = 2, n_future: int = 0, feat_dim: int = 0,
                 attn_window: int = 50, attn_heads: int = 4):
        super().__init__()
        self.d_in, self.d = d_in, d
        self.register_buffer("feat_mean", torch.zeros(d_in))
        self.register_buffer("feat_std", torch.ones(d_in))
        self.conv1 = CausalDSConv(d_in, d)
        self.conv2 = CausalDSConv(d, d)
        make = {"gru": lambda: GRUCore(d), "mamba3c": lambda: ConvSSMCore(d, Mamba3Core(d, chunk=64, ckpt=False))}
        # core: one name for every block, or a comma-separated list with one per block
        cores = core.split(",") if "," in core else [core] * n_blocks
        self.blocks = nn.ModuleList(Block(d, make[c](), mlp_mult) for c in cores)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 + n_future)  # [speech, complete, future voice-activity bins...]
        # training-only feature adapter (not in the released weights)
        self.feat = nn.Linear(d, feat_dim) if feat_dim else None

    def _heads(self, x):
        h = self.norm(x)
        out = self.head(h)
        return out if self.feat is None else torch.cat([out, self.feat(h)], -1)

    def _norm_in(self, x):
        return (x - self.feat_mean) / self.feat_std

    def forward(self, x, state=None, return_hidden: bool = False):
        """x (B, T, d_in) -> logits (B, T, 2), final recurrent states. Conv context starts at zero.

        `state` here carries only the recurrent cores (used for chunked truncated BPTT); the conv
        buffers are rebuilt from the previous chunk by the caller via `context` frames if needed.
        return_hidden: also return the last block's per-frame features (B, T, d), for pre-training heads.
        """
        x = self.conv2(self.conv1(self._norm_in(x)))
        hs = []
        for i, blk in enumerate(self.blocks):
            x, h = blk(x, None if state is None else state[i])
            hs.append(h)
        if return_hidden:
            return self._heads(x), hs, x
        return self._heads(x), hs

    def init_state(self, b: int = 1, device=None) -> list[torch.Tensor]:
        return [self.conv1.init_state(b, self.d_in, device), self.conv2.init_state(b, self.d, device)] + [
            blk.core.init_state(b, device) for blk in self.blocks]

    def step(self, x, state):
        c1, c2, *hs = state
        x = self._norm_in(x)
        x, c1 = self.conv1.step(x, c1)
        x, c2 = self.conv2.step(x, c2)
        new = [c1, c2]
        for blk, h in zip(self.blocks, hs):
            x, h = blk.step(x, h)
            new.append(h)
        return self._heads(x), new


