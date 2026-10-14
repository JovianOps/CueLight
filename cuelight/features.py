"""Causal log-mel front ends. Each 10 ms hop emits a window that *ends* at that hop (no lookahead), and two hops are
stacked into one 20 ms model frame, so frame i covers audio up to (i + 1) * 320 samples.

The mel filter banks ship with each model's weights, so features are bit-for-bit those the model was trained on.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SR = 16000
HOP = 160  # 10 ms
STACK = 2
FRAME = HOP * STACK  # 320 samples = 20 ms


class LogMel(torch.nn.Module):
    """CueLight-Nano features: 64 log-mel bins, 25 ms Hann window, 512-point FFT."""

    WIN, N_FFT, EPS = 400, 512, 1e-6

    def __init__(self, fb: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hann_window(self.WIN, periodic=False), persistent=False)
        self.register_buffer("fb", fb.float(), persistent=False)  # (N_FFT // 2 + 1, n_mels)

    @property
    def dim(self) -> int:
        return self.fb.shape[1] * STACK

    def _frames(self, padded):
        win = padded.unfold(-1, self.WIN, HOP) * self.window
        return torch.log(torch.fft.rfft(win, n=self.N_FFT).abs().pow(2) @ self.fb + self.EPS)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T), T a multiple of 320 -> (B, T // 320, dim)."""
        assert wav.shape[-1] % FRAME == 0, "pad audio to a multiple of 320 samples"
        mel = self._frames(F.pad(wav.float(), (self.WIN - HOP, 0)))
        b, n, m = mel.shape
        return mel.reshape(b, n // STACK, STACK * m)

    def init_state(self, b: int = 1) -> torch.Tensor:
        return torch.zeros(b, self.WIN - HOP, device=self.fb.device)

    def step(self, chunk: torch.Tensor, state: torch.Tensor):
        """chunk (B, 320) new samples -> ((B, dim) features of this frame, new state)."""
        padded = torch.cat([state, chunk.float()], -1)
        mel = self._frames(padded)
        return mel.reshape(mel.shape[0], -1), padded[:, -(self.WIN - HOP):]


class WhisperLogMel(torch.nn.Module):
    """CueLight-Base features: Whisper's log-mel (25 ms window, 400-point FFT), with a fixed floor in place of
    Whisper's per-utterance max normalisation, so every frame is causal."""

    N_FFT, FLOOR = 400, -7.0

    def __init__(self, fb: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hann_window(self.N_FFT), persistent=False)
        self.register_buffer("fb", fb.float(), persistent=False)  # (N_FFT // 2 + 1, n_mels)

    @property
    def dim(self) -> int:
        return self.fb.shape[1] * STACK

    def _frames(self, padded):
        win = padded.unfold(-1, self.N_FFT, HOP) * self.window
        mel = torch.fft.rfft(win, n=self.N_FFT).abs().pow(2) @ self.fb
        return (torch.log10(mel.clamp(min=1e-10)).clamp(min=self.FLOOR) + 4.0) / 4.0

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.shape[-1] % FRAME == 0, "pad audio to a multiple of 320 samples"
        x = self._frames(F.pad(wav.float(), (self.N_FFT - HOP, 0)))
        b, n, m = x.shape
        return x.reshape(b, n // STACK, STACK * m)

    def init_state(self, b: int = 1) -> torch.Tensor:
        return torch.zeros(b, self.N_FFT - HOP, device=self.fb.device)

    def step(self, chunk: torch.Tensor, state: torch.Tensor):
        padded = torch.cat([state, chunk.float()], -1)
        f = self._frames(padded)
        return f.reshape(f.shape[0], -1), padded[:, -(self.N_FFT - HOP):]
