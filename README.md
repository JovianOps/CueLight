# CueLight

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/sshah03)

An always-listening turn detector for voice agents.

Every voice agent has to keep deciding whether the user is done talking. Most turn detectors wait for a pause and
then make that call once. CueLight listens the whole time instead, and gives a new end-of-turn probability for every
20 ms of audio, including while the user is still speaking. The smallest model is 1.9 MB (I was able to run it on 2.5% of one CPU core) so it can stay on all the time, even on a phone or a small device.

It works from audio alone (no transcript needed), so it hears tone and pace as well as words. The code and weights are Apache-2.0 and don't depend on any particular voice framework. It's English only for now.

**[Try CueLight-Nano in your browser](https://sshah03.github.io/CueLight/)**. It runs locally in WebAssembly, so no audio leaves your device.

This is a first iteration. It will continue to improve. To keep it small, no promises on supporting other languages in the immediate future.

## Why CueLight?

[Smart Turn](https://github.com/pipecat-ai/smart-turn) is a really good open turn detector, and CueLight is built on
its open training data. So why make another one?

I wanted something small enough to run continuously on every 20 ms of audio, anywhere from a server down to a phone, a wearable or an embedded chip, and that keeps updating its answer while the user is talking.

**CueLight-Nano** is that model. It's 0.5M parameters, and it matches Smart Turn's accuracy (English only) on both Smart Turn's and LiveKit's benchmarks with 1/16 of the parameters. Effectively trading support for many languages for size.

Once streaming worked, the obvious next question was how much more accuracy a bit more compute would buy. That's
**CueLight-Base**, a Whisper small encoder made causal so it can stream. On real conversations it waits about 0.2 s
less than Smart Turn and LiveKit's local model at the same interruption rate, and it still runs continuously on a
quarter of one CPU core. It's meant for servers and desktops where accuracy matters more than size. Both models have
the same interface, so switching between them is one argument. However, it is significantly larger than SmartTurn. It's also a bit disingenous to compare it to LiveKit mini, since their larger models would outperform CueLight-Base.

## Models

| model | parameters | file size | CPU cost (1 thread) | best for |
| --- | ---: | ---: | --- | --- |
| **CueLight-Base** | 88M | 336 MB | 5.0 ms per 20 ms of audio | servers and desktops, where you want the most accurate model |
| **CueLight-Nano** | 0.5M | 1.9 MB | 0.5 ms per 20 ms of audio | phones, wearables and embedded devices, always on at almost no cost |
| CueLight-Nano 8-bit (ONNX) | 0.5M | 1.5 MB | about the same as Nano | the same model at a smaller download size |
| CueLight-Nano-SSM (experimental) | 0.5M | 1.9 MB | 0.7 ms per 20 ms of audio | Nano with a Mamba-3 core instead of a GRU, for comparison |

CPU cost was measured on an Apple M4 Pro (MacBook Pro) with fp32 weights on one thread, feature extraction included.
5 ms per 20 ms of audio means the model keeps a quarter of one core busy while it listens.

## How well does it work?

I compared CueLight with the two turn detectors you can run locally,
[Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) and
[LiveKit Turn Detector v1-mini](https://docs.livekit.io/agents/logic/turns/turn-detector/), on each of their own
benchmarks. All the numbers are for English.

### LiveKit's benchmark (eot-bench)

[eot-bench](https://github.com/livekit/eot-bench) has 400 real English user turns from conversations with voice
agents, with 1,250 pauses in them. The last pause in each turn is where the user actually finished, and the others are
mid-turn hesitations. A good turn detector stays quiet through the hesitations and responds quickly at the real end.
There's always a trade-off, though. Tune any detector to interrupt less and it waits longer.

**How long does your user wait after they finish talking?** Here every model is tuned to interrupt on at most 1 in 20
pauses.

| model | average wait after the user stops talking |
| --- | ---: |
| **CueLight-Base** | **0.88 s** |
| CueLight-Nano | 0.99 s |
| CueLight-Nano-SSM | 1.02 s |
| Smart Turn v3.2 | 1.05 s |
| LiveKit v1-mini | 1.07 s |

**How often does it cut your user off?** Here every model is tuned to respond within 0.6 s of the user finishing.

| model | interrupts the user on |
| --- | ---: |
| **CueLight-Base** | **10 in 100 pauses** |
| LiveKit v1-mini | 12 in 100 pauses |
| Smart Turn v3.2 | 14 in 100 pauses |
| CueLight-Nano-SSM | 14 in 100 pauses |
| CueLight-Nano | 14 to 15 in 100 pauses |

**How many real turn endings does it catch?** Out of 100 times the user really finishes, this is how many each model
recognizes, while firing on no more than 1 in 20 mid-turn pauses.

| model | turn endings caught |
| --- | ---: |
| **CueLight-Base** | **69 of 100** |
| Smart Turn v3.2 | 51 of 100 |
| CueLight-Nano-SSM | 48 of 100 |
| CueLight-Nano | 48 of 100 |
| LiveKit v1-mini | 45 of 100 |

CueLight-Base waits 0.17 to 0.19 s less than either of the other two when they're all held to 1 in 20 interruptions,
interrupts less often, and catches more turn endings. Its lead over Smart Turn is statistically significant at most
settings, and over LiveKit v1-mini when interruptions are kept to 1 in 20. CueLight-Nano is about as accurate as both,
with 1/16 of Smart Turn's parameters.

The waits above are dead air while the detector decides, not computation time. Computation adds the CPU cost from the
models table on top.

### Smart Turn's benchmark

[Smart Turn's test set](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test) has 7,820 English
clips, each one either a finished or an unfinished utterance. Each model hears the clip plus 0.2 s of silence and
decides whether the speaker is done.

| model | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| **CueLight-Base** | **95.4%** | 93.7% | **97.1%** | **95.4%** |
| CueLight-Nano | 95.0% | **94.2%** | 95.6% | 94.9% |
| CueLight-Nano-SSM | 95.0% | 94.1% | 95.8% | 94.9% |
| Smart Turn v3.2 | 94.7% | 94.0% | 95.3% | 94.6% |

CueLight-Base is slightly more accurate than Smart Turn here (0.7 points, with a 95% interval of 0.2
to 1.1), and Nano and Nano-SSM are level with it. This set comes from the same source as the training data for both
Smart Turn and CueLight, so everything scores high on it. Smart Turn is shown the way its own benchmark runs it (the
fp32 model on the clip as given), which is its best score. With the 0.2 s of silence added it gets 94.2%. LiveKit
v1-mini isn't in this table because its license only allows it to be used inside the LiveKit Agents framework.

## Quick start

```bash
pip install git+https://github.com/sshah03/CueLight
```

The simplest way to use it is to let it tell you when the turn ends:

```python
from cuelight import Endpointer

ep = Endpointer.from_pretrained("cuelight-nano")   # or "cuelight-base"
for chunk in mic_audio_chunks():                    # 16 kHz mono float32, 320 samples (20 ms) each
    if ep.push(chunk):                              # True once, on the frame where the user's turn ends
        respond()
```

There are two presets. `snappy` is the default.

| preset | interrupts the user on | average wait, Nano | average wait, Base |
| --- | --- | ---: | ---: |
| `snappy` | about 1 in 10 pauses | about 0.73 s | about 0.60 s |
| `careful` | about 1 in 20 pauses | about 1.0 s | about 0.88 s |

`snappy` works well when your agent stops talking as soon as the user starts again, because then an occasional early
response doesn't cost much. If being cut off is a bigger deal in your app, use `careful`
(`Endpointer.from_pretrained("cuelight-nano", preset="careful")`). If you want faster responses without more
interruptions, switching to Base does more than a more aggressive setting. Both presets were tuned on half of
eot-bench, so the real rates will depend on your users and audio. You can also set your own with
`EndpointPolicy(threshold, min_silence_s, timeout_s)`.

If you'd rather work with the probability directly:

```python
from cuelight import TurnDetector

det = TurnDetector.from_pretrained("cuelight-nano")
p_end = det.push(chunk)        # probability the user has finished their turn, updated every 20 ms
det.p_speech                   # probability the user is speaking, for the same frame
```

The detector resets itself whenever a new user turn starts (speech after at least 2 s of silence), which keeps long
calls as accurate as short clips. If your pipeline already restarts the detector at each turn, pass
`reset_on_turn_start=False`.

There are two examples. `examples/mic.py` is a live microphone demo (it needs `pip install sounddevice`), and
`examples/stream_wav.py` runs on a file.

The weights download from Hugging Face the first time you use them:
[sshah03/cuelight-nano](https://huggingface.co/sshah03/cuelight-nano) (1.9 MB),
[sshah03/cuelight-nano-ssm](https://huggingface.co/sshah03/cuelight-nano-ssm) (1.9 MB) and
[sshah03/cuelight-base](https://huggingface.co/sshah03/cuelight-base) (336 MB). `from_pretrained` also takes a local
folder with `config.json` and `model.safetensors` in it.

## How it works

Nano is a small causal GRU over log-mel features. Base is the Whisper small.en encoder, changed so each frame only
sees audio up to that point, with the same output head as Nano. Both were trained on real and synthetic conversational
speech, using soft labels from bigger teacher models, with noise and reverb added. When streaming, each 20 ms frame
just updates a state (a GRU hidden state for Nano, a key/value cache for Base), so nothing gets recomputed.

## Limitations

It's English only. Smart Turn covers 23 languages.

It only hears audio and doesn't see the text of the conversation. Detectors that do can sometimes do better on
ambiguous pauses.

CueLight-Base keeps about 25% of a CPU core busy while it listens, which on average is more than a detector that only
runs at pauses. If cost matters most, use Nano.

Only Nano and Nano-SSM have been exported to ONNX so far. Each file in `docs/models/` is one streaming step (20 ms of
audio and the state go in, the probabilities and the new state come out), and that's what the browser demo runs.

Nano also comes in an 8-bit version (`docs/models/cuelight-nano-int8.onnx`, 1.5 MB, with the model's own weights at
about 0.5 MB). On eot-bench it waits 1004 and 746 ms at 5% and 10% cut-offs, against 990 and 751 ms for the full
precision Nano, so it's essentially the same model. It's smaller, but not faster, and it's ONNX only for now, so the Python
package still uses the full precision weights.

## License and credits

The code and weights are Apache-2.0. CueLight was trained on, and builds on, other people's work. The full notices
are in [NOTICE](NOTICE) and [LICENSES/](LICENSES/).

Training data:

- [Smart Turn v3.2 data](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) by Pipecat / Daily,
  with audio from Liva AI, Midcentury and MundoAI (CC BY 4.0)
- People's Speech by MLCommons (CC BY)
- Harper Valley Bank by Gridspace and Stanford (CC BY 4.0)
- Open Yap by The Agentic Data Company (CC BY 4.0)
- eot-bench data by LiveKit (CC BY 4.0)

Models:

- Smart Turn v3.2 (BSD-2-Clause, Copyright (c) 2024-2025, Daily), used as a teacher
- Whisper by OpenAI (MIT and Apache-2.0)

Some of the training data comes with conditions that carry over, so please don't use CueLight to identify speakers or
to clone voices.
