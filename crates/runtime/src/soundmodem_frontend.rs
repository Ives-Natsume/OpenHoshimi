//! Streaming IQ front-end for the vendored soundmodem C decoder.
//!
//! The C core ([`openhoshimi_soundmodem`]) demodulates 48 kHz complex baseband
//! near 0 Hz with a narrow FLL pull-in range. Real recordings arrive as complex
//! baseband at arbitrary rates and, on low passes, sweep tens of kHz of Doppler.
//! [`StreamDecoder`] bridges the two as a causal push/pull state machine so the
//! same path serves both offline files and a live SDR/soundcard source:
//!
//!   * input is consumed in ~1 s causal blocks (no whole-file look-ahead);
//!   * a SEARCH -> LOCKED -> COAST tracker follows one emitter's `s^2` line,
//!     confirming a new carrier over several blocks before re-anchoring so FM
//!     splatter, dropouts and mid-recording retunes are ordinary transitions;
//!   * each block is mixed to 0 Hz by a phase-continuous NCO and resampled to
//!     48 kHz by a stateful polyphase resampler, then fed to one C decoder that
//!     is reset only when the carrier re-anchors far away (a retune).
//!
//! This is the ASRTU-1 path only; every other satellite keeps the pure-Rust
//! pipeline.

use std::collections::HashSet;

use openhoshimi_core::IqSample;
use openhoshimi_soundmodem::{Decoder, Frame, SAMPLE_RATE};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

/// Trajectory / tracking block length, in seconds.
const BLOCK_SEC: usize = 1;

/// Half of the ±near-DC guard, in Hz, masked out of the `s^2` peak search.
/// Kept tight so a near-baseband carrier is still found while true LO leakage
/// (a spike at exactly 0) is rejected.
const DC_GUARD_HZ: f32 = 500.0;

/// Number of distinct `s^2` peaks kept per block for the tracker to choose from.
const TOP_K: usize = 8;

/// Max carrier slew per block the tracker will follow (Hz). Wider than any LEO
/// Doppler step yet far tighter than the gap to a competing emitter.
const TRACK_SEARCH_HZ: f32 = 4000.0;

/// Half-width, in output taps, of the resampler's anti-alias prototype.
const RESAMPLE_HALF_SPAN: usize = 10;

/// Carrier step (Hz) above which a new lock is treated as a retune, resetting
/// the decoder rather than forcing its loops to survive a large jump.
const SEGMENT_JUMP_HZ: f32 = 8000.0;

/// `s^2` margin a peak must clear to keep the lock. Below this the block is
/// treated as a fade and the tracker coasts on the last carrier rather than
/// chasing an FM/noise peak (margin ~10-15) off the true Doppler line.
const HOLD_MARGIN: f32 = 20.0;

/// `s^2` margin a candidate must clear to acquire the lock. Real emitters run in
/// the hundreds; FM splatter false peaks sit around 10-15, so this rejects them.
const ACQUIRE_MARGIN: f32 = 20.0;

/// Consecutive blocks a candidate line must persist before the tracker locks
/// onto it, so a one-off spike cannot pull the carrier.
const ACQUIRE_CONFIRM: usize = 3;

/// `s^2` margin so high the line is unambiguously the signal (FM splatter tops
/// out ~15); the tracker locks on it in one block, not losing the leading
/// frames to the confirmation delay.
const FAST_LOCK_MARGIN: f32 = 100.0;

/// Blocks the tracker feeds a faded lock (holding the last carrier) before
/// giving up and returning to acquisition. Generous, to ride through long
/// fades the way a batch tracker would; a retune is caught sooner by the
/// far-line re-anchor rather than by exhausting this.
const COAST_MAX: usize = 30;

/// IIR weight damping per-block `s^2` jitter on the tracked carrier into phase
/// noise; a fresh lock sets the carrier directly, bypassing this.
const CARRIER_SMOOTH: f32 = 0.5;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Find the top [`TOP_K`] distinct `s^2` carrier peaks of one block, each as
/// `(carrier_hz, peak_to_mean_margin)`, strongest first. `buf`/`mag` are scratch
/// of length `seg.len()`; results are written to `out`.
fn block_peaks(
    seg: &[Complex32],
    fs: u32,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    buf: &mut [Complex32],
    mag: &mut [f32],
    out: &mut Vec<(f32, f32)>,
) {
    let n = seg.len();
    let mean: Complex32 = seg.iter().sum::<Complex32>() / n as f32;
    for (o, &s) in buf.iter_mut().zip(seg) {
        let z = s - mean;
        *o = z * z; // BPSK doubling
    }
    fft.process(buf);

    let mask = (DC_GUARD_HZ / fs as f32 * n as f32) as usize;
    let supp = (500.0 / fs as f32 * n as f32) as usize; // ±500 Hz peak spacing
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for (k, c) in buf.iter().enumerate() {
        if k < mask || k > n - mask {
            mag[k] = 0.0;
        } else {
            mag[k] = c.norm_sqr();
            sum += mag[k];
            cnt += 1;
        }
    }
    let mean_mag = (sum / cnt.max(1) as f32).max(1e-20);

    out.clear();
    for _ in 0..TOP_K {
        let mut peak = 0.0f32;
        let mut peak_k = 0usize;
        for (k, &m) in mag.iter().enumerate() {
            if m > peak {
                peak = m;
                peak_k = k;
            }
        }
        if peak <= 0.0 {
            break;
        }
        let signed = if peak_k <= n / 2 {
            peak_k as f32
        } else {
            peak_k as f32 - n as f32
        };
        out.push((signed * fs as f32 / n as f32 / 2.0, peak / mean_mag));
        let lo = peak_k.saturating_sub(supp);
        let hi = (peak_k + supp).min(n - 1);
        mag[lo..=hi].iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Strongest peak within the slew limit of `pred` that still clears
/// [`HOLD_MARGIN`]. A genuine (even faded) BPSK line keeps a concentrated `s^2`
/// peak, so this holds real signal while a flat-noise block yields nothing and
/// the tracker coasts on the last carrier instead of chasing noise.
fn hold_peak(peaks: &[(f32, f32)], pred: f32) -> Option<f32> {
    peaks
        .iter()
        .filter(|(f, m)| (f - pred).abs() <= TRACK_SEARCH_HZ && *m >= HOLD_MARGIN)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|&(f, _)| f)
}

/// Stateful rational polyphase resampler (upfirdn) for a streaming complex
/// signal: `out_rate = in_rate * l / m`. Carries filter history across calls so
/// block boundaries introduce no transient.
struct StreamResampler {
    l: usize,
    m: usize,
    taps: Vec<f32>,
    ntaps: usize,
    buf: Vec<Complex32>,
    base: usize,
    in_total: usize,
    out_up: usize,
}

impl StreamResampler {
    fn new(in_fs: u32, out_fs: u32) -> Self {
        let g = gcd(in_fs as usize, out_fs as usize);
        let l = out_fs as usize / g;
        let m = in_fs as usize / g;
        let half = RESAMPLE_HALF_SPAN * l.max(m);
        let ntaps = 2 * half + 1;
        let fc = 0.5 / l.max(m) as f32;
        let mut taps = vec![0.0f32; ntaps];
        for (i, tap) in taps.iter_mut().enumerate() {
            let nn = i as f32 - half as f32;
            let sinc = if nn == 0.0 {
                2.0 * fc
            } else {
                (2.0 * std::f32::consts::PI * fc * nn).sin() / (std::f32::consts::PI * nn)
            };
            let w =
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (ntaps as f32 - 1.0)).cos();
            *tap = sinc * w * l as f32;
        }
        Self {
            l,
            m,
            taps,
            ntaps,
            buf: Vec::new(),
            base: 0,
            in_total: 0,
            out_up: 0,
        }
    }

    fn process(&mut self, block: &[Complex32], out: &mut Vec<Complex32>) {
        if self.l == 1 && self.m == 1 {
            out.extend_from_slice(block);
            return;
        }
        self.buf.extend_from_slice(block);
        self.in_total += block.len();
        while self.out_up / self.l < self.in_total {
            let t = self.out_up;
            let mut acc = Complex32::new(0.0, 0.0);
            let mut k = t % self.l;
            while k < self.ntaps {
                let xi = (t - k) / self.l;
                if xi >= self.base {
                    let idx = xi - self.base;
                    if idx < self.buf.len() {
                        acc += self.buf[idx] * self.taps[k];
                    }
                }
                k += self.l;
            }
            out.push(acc);
            self.out_up += self.m;
        }
        let keep = self.out_up.saturating_sub(self.ntaps) / self.l;
        if keep > self.base {
            let drop = (keep - self.base).min(self.buf.len());
            self.buf.drain(0..drop);
            self.base += drop;
        }
    }
}

/// Causal push/pull ASRTU-1 decoder over the soundmodem C backend. Feed complex
/// baseband at `fs` with [`StreamDecoder::push`]; drain frames with
/// [`StreamDecoder::take_frames`]. The same object drives offline and live use.
pub struct StreamDecoder {
    fs: u32,
    block_len: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    fbuf: Vec<Complex32>,
    mag: Vec<f32>,
    peaks: Vec<(f32, f32)>,
    accum: Vec<Complex32>,

    locked: bool,
    carrier: f32,
    coast: usize,
    cand: Option<(f32, usize)>,
    last_lock: f32,
    ever: bool,

    mix_phase: f64,
    resampler: StreamResampler,
    dec: Option<Decoder>,

    mixed: Vec<Complex32>,
    resout: Vec<Complex32>,
    ilv: Vec<f32>,

    frames: Vec<Frame>,
    seen: HashSet<Vec<u8>>,
}

impl StreamDecoder {
    /// Create a decoder for complex baseband at `fs` Hz.
    pub fn new(fs: u32) -> Self {
        let block_len = fs as usize * BLOCK_SEC;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(block_len.max(1));
        Self {
            fs,
            block_len,
            fft,
            fbuf: vec![Complex32::new(0.0, 0.0); block_len.max(1)],
            mag: vec![0.0; block_len.max(1)],
            peaks: Vec::new(),
            accum: Vec::with_capacity(block_len),
            locked: false,
            carrier: 0.0,
            coast: 0,
            cand: None,
            last_lock: 0.0,
            ever: false,
            mix_phase: 0.0,
            resampler: StreamResampler::new(fs, SAMPLE_RATE),
            dec: None,
            mixed: Vec::new(),
            resout: Vec::new(),
            ilv: Vec::new(),
            frames: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Push complex baseband samples. Decoded frames accumulate internally.
    pub fn push(&mut self, cplx: &[Complex32]) {
        let mut off = 0;
        while off < cplx.len() {
            let space = self.block_len - self.accum.len();
            let take = (cplx.len() - off).min(space);
            self.accum.extend_from_slice(&cplx[off..off + take]);
            off += take;
            if self.accum.len() == self.block_len {
                let block = std::mem::take(&mut self.accum);
                self.process_block(&block);
                self.accum = block;
                self.accum.clear();
            }
        }
    }

    /// Flush the trailing partial block at end of stream.
    pub fn finish(&mut self) {
        if !self.accum.is_empty() {
            let block = std::mem::take(&mut self.accum);
            self.process_block(&block);
            self.accum = block;
            self.accum.clear();
        }
        if let Some(dec) = &mut self.dec {
            dec.flush();
            while let Some(f) = dec.poll() {
                if self.seen.insert(f.data.clone()) {
                    self.frames.push(f);
                }
            }
        }
    }

    /// Take all frames decoded so far (de-duplicated by payload).
    pub fn take_frames(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.frames)
    }

    fn process_block(&mut self, block: &[Complex32]) {
        if block.len() == self.block_len {
            self.update_tracking(block);
        }
        if self.locked {
            self.mix_and_feed(block);
        }
    }

    /// Advance the SEARCH -> LOCKED -> COAST carrier tracker on a full block.
    fn update_tracking(&mut self, block: &[Complex32]) {
        block_peaks(
            block,
            self.fs,
            &self.fft,
            &mut self.fbuf,
            &mut self.mag,
            &mut self.peaks,
        );

        if self.locked {
            if let Some(f) = hold_peak(&self.peaks, self.carrier) {
                self.carrier = CARRIER_SMOOTH * self.carrier + (1.0 - CARRIER_SMOOTH) * f;
                self.coast = 0;
                self.cand = None;
                return;
            }
            self.coast += 1;
            // Catch a retune fast: a far, strong line confirming over a few
            // blocks re-anchors without waiting for the coast to run out.
            if let Some((f, m)) = self.peaks.first().copied() {
                if m >= ACQUIRE_MARGIN && (f - self.carrier).abs() > TRACK_SEARCH_HZ {
                    let cnt = match self.cand {
                        Some((cf, c)) if (f - cf).abs() <= TRACK_SEARCH_HZ => c + 1,
                        _ => 1,
                    };
                    if cnt >= ACQUIRE_CONFIRM || m >= FAST_LOCK_MARGIN {
                        if (f - self.last_lock).abs() > SEGMENT_JUMP_HZ {
                            self.dec = None;
                        }
                        self.carrier = f;
                        self.last_lock = f;
                        self.coast = 0;
                        self.cand = None;
                    } else {
                        self.cand = Some((f, cnt));
                    }
                } else {
                    self.cand = None;
                }
            }
            if self.coast > COAST_MAX {
                self.locked = false;
                self.cand = None;
            }
            return;
        }

        // SEARCH: confirm the strongest line over several blocks before locking.
        let strongest = self.peaks.first().copied();
        match strongest {
            Some((f, m)) if m >= ACQUIRE_MARGIN => {
                let cnt = match self.cand {
                    Some((cf, c)) if (f - cf).abs() <= TRACK_SEARCH_HZ => c + 1,
                    _ => 1,
                };
                if cnt >= ACQUIRE_CONFIRM || m >= FAST_LOCK_MARGIN {
                    if self.ever && (f - self.last_lock).abs() > SEGMENT_JUMP_HZ {
                        self.dec = None; // retune: fresh decoder
                    }
                    self.carrier = f;
                    self.last_lock = f;
                    self.ever = true;
                    self.locked = true;
                    self.coast = 0;
                    self.cand = None;
                } else {
                    self.cand = Some((f, cnt));
                }
            }
            _ => self.cand = None,
        }
    }

    /// Mix the block to 0 Hz (phase-continuous), resample to 48 kHz and feed the
    /// C decoder, collecting any frames it emits.
    fn mix_and_feed(&mut self, block: &[Complex32]) {
        let two_pi = 2.0 * std::f64::consts::PI;
        let w = -two_pi * self.carrier as f64 / self.fs as f64;
        self.mixed.clear();
        self.mixed.reserve(block.len());
        for &z in block {
            self.mix_phase = (self.mix_phase + w).rem_euclid(two_pi);
            let (s, c) = self.mix_phase.sin_cos();
            let (s, c) = (s as f32, c as f32);
            self.mixed
                .push(Complex32::new(z.re * c - z.im * s, z.re * s + z.im * c));
        }

        self.resout.clear();
        self.resampler.process(&self.mixed, &mut self.resout);

        self.ilv.clear();
        self.ilv.reserve(self.resout.len() * 2);
        for c in &self.resout {
            self.ilv.push(c.re);
            self.ilv.push(c.im);
        }

        let dec = self
            .dec
            .get_or_insert_with(|| Decoder::new().expect("decoder alloc"));
        dec.feed_iq(&self.ilv);
        while let Some(f) = dec.poll() {
            if self.seen.insert(f.data.clone()) {
                self.frames.push(f);
            }
        }
    }
}

/// Decode ASRTU-1 CCSDS frames from full-pass IQ via the soundmodem C backend.
///
/// A thin offline wrapper over [`StreamDecoder`]: the whole recording is pushed
/// through the same causal pipeline a live source would drive.
pub fn decode_iq(iq: &[IqSample], fs: u32) -> Vec<Frame> {
    if iq.is_empty() {
        return Vec::new();
    }
    let mut sd = StreamDecoder::new(fs);
    let mut buf: Vec<Complex32> = Vec::with_capacity(fs as usize);
    for s in iq {
        buf.push(Complex32::new(s.i, s.q));
        if buf.len() == fs as usize {
            sd.push(&buf);
            buf.clear();
        }
    }
    sd.push(&buf);
    sd.finish();
    sd.take_frames()
}
