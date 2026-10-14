from pathlib import Path

import numpy as np
import pytest
import torch

import cuelight

LOCAL = Path(__file__).resolve().parents[1] / "hf_upload"  # local copies if they exist, otherwise Hugging Face


def src(name):
    return str(LOCAL / name) if (LOCAL / name).is_dir() else name


@pytest.mark.parametrize("name", ["cuelight-nano", "cuelight-nano-ssm"])
@torch.no_grad()
def test_streaming_matches_whole_sequence(name):
    model, fe = cuelight.load(src(name))
    wav = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.05
    whole = torch.sigmoid(model(fe(torch.from_numpy(wav)[None]))[0][0, :, 1]).numpy()
    det = cuelight.TurnDetector(model, fe, reset_on_turn_start=False)
    stream = np.array([det.push(wav[i:i + 320]) for i in range(0, len(wav), 320)])
    assert np.abs(stream - whole).max() < 1e-4
    assert ((0 <= stream) & (stream <= 1)).all()


def test_push_needs_20ms_chunks():
    det = cuelight.TurnDetector.from_pretrained(src("cuelight-nano"))
    with pytest.raises(AssertionError):
        det.push(np.zeros(160, np.float32))
