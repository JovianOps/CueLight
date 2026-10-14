"""Loading released models: a directory (or Hugging Face repo) with `config.json` and `model.safetensors`."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cuelight.features import LogMel, WhisperLogMel

# Short names -> Hugging Face model repos.
MODELS = {
    "cuelight-nano": "sshah03/cuelight-nano",
    "cuelight-nano-ssm": "sshah03/cuelight-nano-ssm",
    "cuelight-base": "sshah03/cuelight-base",
}


def _resolve(name_or_path: str) -> Path:
    src = MODELS.get(name_or_path, name_or_path)
    if Path(src).is_dir():
        return Path(src)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(src), allow_patterns=["config.json", "model.safetensors"]))


def load(name_or_path: str, device: str = "cpu"):
    """-> (model, frontend), in eval mode. name_or_path: a short name above, a local directory, or a HF repo id."""
    from safetensors.torch import load_file
    d = _resolve(name_or_path)
    cfg = json.loads((d / "config.json").read_text())
    sd = load_file(str(d / "model.safetensors"))
    fb = sd.pop("frontend.fb")
    if cfg["architecture"] == "whisper-causal":
        from cuelight.whisper_encoder import WhisperTurnModel
        model = WhisperTurnModel(**cfg["model"])
        frontend = WhisperLogMel(fb)
    else:
        from cuelight.model import TurnEndModel
        model = TurnEndModel(**cfg["model"])
        frontend = LogMel(fb)
    model.load_state_dict(sd)
    model.cuelight_config = cfg
    return model.to(device).eval(), frontend.to(device)
