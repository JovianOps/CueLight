"""Score CueLight (and Smart Turn v3.2) on the English part of Smart Turn's test set
(pipecat-ai/smart-turn-data-v3.2-test, 7,820 clips). Each model hears the clip followed by 0.2 s of silence; Smart Turn
is also scored on the clip as given (its own protocol). Shards (~0.5 GB each) are downloaded and deleted one at a time.

    pip install -e ".[eval]"
    python eval/smart_turn_test.py --models cuelight-nano cuelight-base --smart-turn --out results/smart_turn_test
"""
import argparse
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

import cuelight

REPO, TAIL = "pipecat-ai/smart-turn-data-v3.2-test", 3200


def smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    sess = ort.InferenceSession(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"))
    fx, n = WhisperFeatureExtractor(chunk_length=8), 8 * 16000

    def predict(wav):
        wav = np.pad(wav[-n:], (max(0, n - len(wav)), 0))
        x = fx(wav, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n, truncation=True,
               do_normalize=True).input_features.astype(np.float32)
        return float(sess.run(None, {"input_features": x})[0].reshape(-1)[0])
    return predict


@torch.no_grad()
def cuelight_predict(name, device):
    model, fe = cuelight.load(name, device)

    def predict(wav):
        x = np.concatenate([np.zeros((-(len(wav) + TAIL)) % 320, np.float32), wav, np.zeros(TAIL, np.float32)])
        logits, _ = model(fe(torch.from_numpy(x)[None].to(device)))
        return float(torch.sigmoid(logits[0, -1, 1]))
    return predict


def metrics(y, p):
    yh = p > 0.5
    tp, fp, fn = (yh & y).sum(), (yh & ~y).sum(), (~yh & y).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"accuracy": round(100 * float((yh == y).mean()), 2), "precision": round(100 * prec, 2),
            "recall": round(100 * rec, 2), "f1": round(200 * prec * rec / max(prec + rec, 1e-9), 2),
            "auc": round(float(roc_auc_score(y, p)), 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cuelight-nano"])
    ap.add_argument("--smart-turn", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/smart_turn_test")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fns = {m: cuelight_predict(m, a.device) for m in a.models}
    if a.smart_turn:
        st = smart_turn()
        fns["smart-turn-v3.2 (+0.2 s silence)"] = lambda w: st(np.concatenate([w, np.zeros(TAIL, np.float32)]))
        fns["smart-turn-v3.2 (clip as given)"] = st
    for k in range(10):
        f_out = out / f"shard{k}.parquet"
        if f_out.exists():
            continue
        cache = tempfile.mkdtemp()
        f = hf_hub_download(REPO, f"data/train-{k:05d}-of-00010.parquet", repo_type="dataset", cache_dir=cache)
        rows = pq.read_table(f, filters=[("language", "=", "eng")]).to_pylist()
        shutil.rmtree(cache, ignore_errors=True)
        recs = []
        for r in rows:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            rec = {"id": r["id"], "label": bool(r["endpoint_bool"]), "synthetic": bool(r["synthetic"])}
            rec.update({m: fn(wav) for m, fn in fns.items()})
            recs.append(rec)
        pd.DataFrame(recs).to_parquet(f_out)
        print(f"shard {k}: {len(recs)} clips", flush=True)
    d = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("shard*.parquet"))], ignore_index=True)
    res = {m: metrics(d.label.to_numpy(), d[m].to_numpy()) for m in fns}
    print(json.dumps(res, indent=1))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
