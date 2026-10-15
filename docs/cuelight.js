// CueLight in the browser: the streaming step (features + model) as one ONNX graph, run with onnxruntime-web.
// One call per 20 ms of 16 kHz audio (320 samples): (chunk, fe_state, s0..s4) -> (p[speech, end], states').
// Same logic as the Python package: a state reset at the start of each new user turn, and an end-of-turn policy.

const FRAME = 320;
const HOP_S = FRAME / 16000;

const RESET = { minSilenceS: 2.0, onsetFrames: 3, lookbackS: 0.3, speechThr: 0.5 };

export class CueLight {
  // name is "cuelight-nano", "cuelight-nano-int8" or "cuelight-nano-ssm". Reads models/<name>.json (state shapes and presets) and the ONNX file.
  static async load(name, base = "models/") {
    const v = "?v=5"; // bump with each model / page update so browsers never mix old and new files
    const meta = await (await fetch(`${base}${name}.json${v}`)).json();
    const session = await ort.InferenceSession.create(base + meta.model.split("/").pop() + v, { executionProviders: ["wasm"] });
    return new CueLight(session, meta);
  }

  constructor(session, meta) {
    this.session = session;
    this.shapes = meta.states;
    this.presets = meta.presets;
    this.stateNames = session.inputNames.filter((n) => n !== "chunk");
    this.replay = Math.round(RESET.lookbackS / HOP_S) + RESET.onsetFrames;
    this.preset = meta.presets.snappy; // snappy by default (responds sooner, interrupts about 1 in 10 pauses)
    this.clear();
  }

  setPreset(name) { this.preset = this.presets[name]; }

  zeros(name) {
    const shape = this.shapes[name];
    return new ort.Tensor("float32", new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }

  clear() {
    this.feState = this.zeros("fe_state");
    this.modelState = this.stateNames.filter((n) => n !== "fe_state").map((n) => this.zeros(n));
    this.buf = []; // [chunk, feState before it] for the replay after a reset
    this.silentFrames = 0; this.armed = false; this.run = 0; // turn-start tracker
    this.epSilent = 0; this.fired = true; // end-of-turn policy, with nothing to end until the user speaks
    this.frames = 0; this.resets = 0;
  }

  async step(chunk, feState, modelState) {
    const feeds = { chunk: new ort.Tensor("float32", chunk, [1, FRAME]), fe_state: feState };
    this.stateNames.filter((n) => n !== "fe_state").forEach((n, i) => { feeds[n] = modelState[i]; });
    const out = await this.session.run(feeds);
    const p = out.p.data;
    return { pSpeech: p[0], pEnd: p[1], feState: out.fe_state_out,
             modelState: this.stateNames.filter((n) => n !== "fe_state").map((n) => out[n + "_out"]) };
  }

  // Turn-start reset: fires once when speech lasts onsetFrames after >= minSilenceS of silence.
  onset(pSpeech) {
    if (pSpeech < RESET.speechThr) { this.silentFrames += 1; this.run = 0; return false; }
    if (this.run === 0) { this.armed = this.silentFrames >= Math.round(RESET.minSilenceS / HOP_S); this.silentFrames = 0; }
    this.run += 1;
    if (this.armed && this.run === RESET.onsetFrames) { this.armed = false; return true; }
    return false;
  }

  // 320 samples at 16 kHz -> { pSpeech, pEnd, ended, silenceS }
  async push(chunk) {
    this.buf.push([chunk, this.feState]);
    if (this.buf.length > this.replay) this.buf.shift();
    let r = await this.step(chunk, this.feState, this.modelState);
    this.feState = r.feState; this.modelState = r.modelState;
    const pSpeech = r.pSpeech;
    if (this.onset(pSpeech)) { // new user turn: rerun the last few frames from a fresh model state
      let ms = this.stateNames.filter((n) => n !== "fe_state").map((n) => this.zeros(n));
      for (const [c, fs] of this.buf) { r = await this.step(c, fs, ms); ms = r.modelState; }
      this.modelState = ms; this.resets += 1;
    }
    const pEnd = r.pEnd;
    this.frames += 1;
    let ended = false;
    if (pSpeech >= 0.5) { this.epSilent = 0; this.fired = false; }
    else {
      this.epSilent += 1;
      const s = this.epSilent * HOP_S;
      if (!this.fired && ((s >= this.preset.minSilenceS && pEnd >= this.preset.threshold) || s >= this.preset.timeoutS)) {
        this.fired = true; ended = true;
      }
    }
    return { pSpeech, pEnd, ended, silenceS: this.epSilent * HOP_S };
  }
}
