"""Stream a 16 kHz mono WAV file through CueLight and print where it thinks each turn ends.

    python examples/stream_wav.py path/to/audio.wav [--model cuelight-nano] [--preset snappy|careful]
"""
import argparse

import numpy as np
import soundfile as sf

from cuelight import Endpointer

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--model", default="cuelight-nano")
ap.add_argument("--preset", default="snappy", choices=["snappy", "careful"])
a = ap.parse_args()

audio, sr = sf.read(a.wav, dtype="float32", always_2d=True)
assert sr == 16000, "resample to 16 kHz first"
audio = audio.mean(1)
ep = Endpointer.from_pretrained(a.model, preset=a.preset)
for i in range(0, len(audio) - 319, 320):
    if ep.push(audio[i:i + 320]):
        print(f"{(i + 320) / 16000:7.2f} s  end of turn (p_end = {ep.p_end:.2f})")
