// Microphone -> 16 kHz mono, in 320-sample (20 ms) chunks. Low-pass (windowed sinc) at the device rate, then
// fractional resampling by linear interpolation.
class To16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    const cutoff = 7600 / sampleRate, taps = 63, m = (taps - 1) / 2;
    this.h = new Float32Array(taps);
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const x = i - m;
      const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
      this.h[i] = sinc * w; sum += this.h[i];
    }
    for (let i = 0; i < taps; i++) this.h[i] /= sum;
    this.hist = new Float32Array(taps);
    this.hpos = 0;
    this.filtered = []; // low-passed samples at the device rate, not yet resampled
    this.t = 0;         // fractional read position in `filtered`
    this.out = new Float32Array(320);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    const taps = this.h.length;
    for (let i = 0; i < ch.length; i++) {
      this.hist[this.hpos] = ch[i];
      let y = 0;
      for (let k = 0; k < taps; k++) y += this.h[k] * this.hist[(this.hpos - k + taps) % taps];
      this.hpos = (this.hpos + 1) % taps;
      this.filtered.push(y);
    }
    while (this.t + 1 < this.filtered.length) {
      const i = Math.floor(this.t), f = this.t - i;
      this.out[this.n++] = this.filtered[i] * (1 - f) + this.filtered[i + 1] * f;
      if (this.n === 320) { this.port.postMessage(this.out.slice(0)); this.n = 0; }
      this.t += this.ratio;
    }
    const drop = Math.floor(this.t);
    this.filtered = this.filtered.slice(drop);
    this.t -= drop;
    return true;
  }
}
registerProcessor("to16k", To16k);
