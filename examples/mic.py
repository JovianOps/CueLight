"""Live microphone demo: prints when CueLight thinks you've finished talking.

    pip install sounddevice
    python examples/mic.py [--model cuelight-nano]
"""
import argparse
import queue

import numpy as np
import sounddevice as sd

from cuelight import Endpointer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cuelight-nano")
ap.add_argument("--preset", default="snappy", choices=["snappy", "careful"])
a = ap.parse_args()

ep = Endpointer.from_pretrained(a.model, preset=a.preset)
q: queue.Queue = queue.Queue()
with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=320,
                    callback=lambda data, *_: q.put(data[:, 0].copy())):
    print("Listening... speak, then pause. Ctrl+C to stop.")
    while True:
        ended = ep.push(q.get())
        bar = "#" * int(30 * ep.p_end)
        print(f"\rspeech {ep.det.p_speech:4.2f}  end-of-turn {ep.p_end:4.2f} {bar:<30}", end="", flush=True)
        if ended:
            print("\n-> your turn is over")
