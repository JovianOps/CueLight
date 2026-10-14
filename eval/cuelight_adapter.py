"""eot-bench adapters for CueLight (https://github.com/livekit/eot-bench).

    pip install -e path/to/eot-bench && pip install -e path/to/CueLight
    eot-harness predict --path livekit/eot-bench-data --name all --split validation \
        --adapter eval.cuelight_adapter:CueLightNanoAdapter --output-dir output
    eot-harness compute-metrics --predictions output/<span set>/en/<run>/predictions.parquet --output-dir ...

The harness asks for p(end of turn) at every 0.1 s grid point of each pause, giving the audio of the turn up to that
point. CueLight is a streaming model, so instead of re-running on each prefix the adapter keeps the stream's state and
only processes the new audio since the previous point of the same turn. The scores are identical and the cost is
linear. Each turn starts from a fresh state, and `score_point = None` tells the harness to score the model
continuously through each pause.
"""
from __future__ import annotations

import numpy as np
import torch

import cuelight

FRAME = 320


class _CueLightAdapter:
    model_name = "cuelight-nano"
    display_name = "CueLight-Nano"
    score_point = None

    def __init__(self, model: str | None = None, device: str = "cpu"):
        self.model, self.fe = cuelight.load(model or self.model_name, device)
        self.device = device
        self.adapter_id = f"cuelight/{model or self.model_name}"
        self._prefix, self._n, self._state, self._fe_state, self._p = None, 0, None, None, 0.5

    def supports_language(self, lang_code: str) -> bool:
        return str(lang_code).lower().startswith("en")  # English-only models

    @torch.no_grad()
    def _score(self, arr: np.ndarray) -> float:
        n = len(arr) // FRAME * FRAME
        same_turn = self._prefix is not None and n >= self._n and np.array_equal(arr[:self._n], self._prefix)
        if not same_turn:  # a new turn (or an earlier point): restart from a fresh state
            self._n, self._state, self._fe_state = 0, self.model.init_state(1), self.fe.init_state(1)
        for i in range(self._n, n, FRAME):
            chunk = torch.from_numpy(arr[i:i + FRAME])[None].to(self.device)
            f, self._fe_state = self.fe.step(chunk, self._fe_state)
            logits, self._state = self.model.step(f, self._state)
            self._p = float(torch.sigmoid(logits[0, 1]))
        self._n, self._prefix = n, arr[:n].copy()
        return self._p

    def predict_batch(self, batch):
        out = []
        for item in batch:
            audio = item.get("audio") or {}
            arr = np.asarray(audio["array"], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            if int(audio["sampling_rate"]) != 16000:
                raise ValueError("CueLight expects 16 kHz audio")
            out.append(self._score(np.ascontiguousarray(arr)))
        return out


class CueLightNanoAdapter(_CueLightAdapter):
    model_name, display_name = "cuelight-nano", "CueLight-Nano"


class CueLightNanoSSMAdapter(_CueLightAdapter):
    model_name, display_name = "cuelight-nano-ssm", "CueLight-Nano-SSM"


class CueLightBaseAdapter(_CueLightAdapter):
    model_name, display_name = "cuelight-base", "CueLight-Base"
