//! IQ front-end for the vendored soundmodem C decoder.
//!
//! The C core ([`openhoshimi_soundmodem`]) demodulates 48 kHz complex baseband
//! near 0 Hz with a narrow FLL pull-in range. Real recordings arrive as complex
//! baseband at arbitrary rates and, on low passes, sweep tens of kHz of Doppler.
//! [`decode_iq`] bridges the two with a survey / fit / replay pipeline that uses
//! the RS code, not any acquisition threshold, as the judge of a real frame:
//!
//!   * survey the strongest `s^2` carrier line once per second (no gating);
//!   * fit smooth Doppler tracks through the confident detections by RANSAC,
//!     the collinearity and slew constraints rejecting the scattered noise
//!     peaks that a scalar margin threshold never cleanly could;
//!   * replay each track through its own decoder, mixing along the fitted
//!     carrier curve so the Doppler is de-rotated and the signal energy stays
//!     coherent across a whole burst.
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
/// Half-width, in output taps, of the resampler's anti-alias prototype.
const RESAMPLE_HALF_SPAN: usize = 10;
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

// ==== offline replay decoder: survey -> RANSAC Doppler tracks -> replay ====

/// Minimal `s^2` peak-to-mean ratio for a block to seed track fitting. Weak but
/// generous: physics (collinearity + bounded slew) rejects noise, not this.
const ANCHOR_MARGIN: f32 = 15.0;

/// Max carrier slope (Hz/s) a Doppler track may have; well above LEO Doppler at
/// 435 MHz, far below the jump to an unrelated noise peak.
const MAX_SLEW_HZ_S: f32 = 250.0;

/// Frequency tolerance (Hz) for a detection to count as lying on a track.
const INLIER_TOL_HZ: f32 = 1500.0;

/// Seconds padded before/after a track when replaying, so the burst preamble
/// and trailing frames fall inside the decoded window.
const REPLAY_PREROLL_S: f32 = 2.0;
const REPLAY_POSTROLL_S: f32 = 15.0;

/// Min anchors on a track for it to be worth replaying. Two is enough: a weak
/// burst may seed only two anchors yet still decode once de-rotated, so the RS
/// code, not this, is the real filter.
const MIN_INLIERS: usize = 2;

/// Max Doppler tracks fit (compute budget); the strongest rank first.
const MAX_TRACKS: usize = 12;

/// A fitted Doppler track: carrier(t) = a + b*t (Hz), backed by anchors at the
/// times `ats` (seconds). Replay only visits windows around those anchors, so
/// the quiet gaps between bursts on one Doppler curve are not re-decoded.
struct Track {
    a: f32,
    b: f32,
    ats: Vec<f32>,
}

/// Per-1 s block, the strongest `s^2` carrier peak (Hz) and its margin.
fn survey(cplx: &[Complex32], fs: u32) -> Vec<(f32, f32)> {
    let block = fs as usize * BLOCK_SEC;
    let nblocks = cplx.len() / block;
    if nblocks == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(block);
    let mut buf = vec![Complex32::new(0.0, 0.0); block];
    let mut mag = vec![0.0f32; block];
    let mut peaks = Vec::new();
    let mut out = Vec::with_capacity(nblocks);
    for b in 0..nblocks {
        block_peaks(
            &cplx[b * block..(b + 1) * block],
            fs,
            &fft,
            &mut buf,
            &mut mag,
            &mut peaks,
        );
        out.push(peaks.first().copied().unwrap_or((0.0, 0.0)));
    }
    out
}

/// Least-squares line fit `freq = a + b*t` through `(t, f)` points.
fn fit_line(pts: &[(f32, f32)]) -> (f32, f32) {
    let n = pts.len() as f32;
    let mt = pts.iter().map(|p| p.0).sum::<f32>() / n;
    let mf = pts.iter().map(|p| p.1).sum::<f32>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for &(t, f) in pts {
        num += (t - mt) * (f - mf);
        den += (t - mt) * (t - mt);
    }
    let b = if den > 1e-6 { num / den } else { 0.0 };
    (mf - b * mt, b)
}

/// RANSAC-fit smooth Doppler tracks through anchor detections `(t_s, f)`, peeling
/// off the best line each round; the collinearity + slew constraints reject the
/// scattered noise peaks that no scalar margin threshold could.
fn ransac_tracks(anchors: &[(f32, f32)]) -> Vec<Track> {
    let on_line = |a: f32, b: f32, t: f32, f: f32| (f - (a + b * t)).abs() <= INLIER_TOL_HZ;
    let mut rem: Vec<(f32, f32)> = anchors.to_vec();
    let mut tracks = Vec::new();
    while tracks.len() < MAX_TRACKS && rem.len() >= 2 {
        let mut best: Option<(usize, f32, f32)> = None;
        for i in 0..rem.len() {
            for j in (i + 1)..rem.len() {
                let (t1, f1) = rem[i];
                let (t2, f2) = rem[j];
                if (t2 - t1).abs() < 0.5 {
                    continue;
                }
                let b = (f2 - f1) / (t2 - t1);
                if b.abs() > MAX_SLEW_HZ_S {
                    continue;
                }
                let a = f1 - b * t1;
                let cnt = rem.iter().filter(|&&(t, f)| on_line(a, b, t, f)).count();
                if best.map_or(true, |(bc, _, _)| cnt > bc) {
                    best = Some((cnt, a, b));
                }
            }
        }
        let Some((cnt, a, b)) = best else { break };
        if cnt < 2 {
            break;
        }
        let inl: Vec<(f32, f32)> = rem
            .iter()
            .copied()
            .filter(|&(t, f)| on_line(a, b, t, f))
            .collect();
        rem.retain(|&(t, f)| !on_line(a, b, t, f));
        if cnt < MIN_INLIERS {
            continue;
        }
        let (a2, b2) = fit_line(&inl);
        tracks.push(Track {
            a: a2,
            b: b2,
            ats: inl.iter().map(|p| p.0).collect(),
        });
    }
    tracks
}

/// Sample spans to decode for a track: touching-merged windows around each
/// anchor (a fresh decoder per burst re-acquires after a quiet gap) plus, when
/// there are gaps, the full extent as one continuous span. Their union never
/// loses frames versus either alone. Window width (hence merging) is fixed by
/// the roll constants, so there is no accuracy knob here.
fn track_spans(tr: &Track, fs: u32, len: usize) -> Vec<(usize, usize)> {
    let mut wins: Vec<(usize, usize)> = tr
        .ats
        .iter()
        .map(|&a| {
            let s0 = ((a - REPLAY_PREROLL_S).max(0.0) * fs as f32) as usize;
            let s1 = (((a + 1.0 + REPLAY_POSTROLL_S) * fs as f32) as usize).min(len);
            (s0, s1)
        })
        .collect();
    wins.sort_unstable();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (s0, s1) in wins {
        match spans.last_mut() {
            Some(last) if s0 <= last.1 => last.1 = last.1.max(s1),
            _ => spans.push((s0, s1)),
        }
    }
    // Also decode the full continuous extent, so a signal that fades in and out
    // across gaps is still decoded whole. Union with the per-burst windows never
    // loses frames versus either alone; it only costs the extra continuous decode
    // — the price of maximising decode rate without any accuracy knob.
    if spans.len() > 1 {
        let (lo, hi) = (spans[0].0, spans.last().unwrap().1);
        spans.push((lo, hi));
    }
    spans
}

/// Decode one span: mix to 0 Hz along carrier(t) = a + b*t, resample to 48 kHz
/// and run the C decoder. RS is the sole arbiter.
fn decode_span(cplx: &[Complex32], fs: u32, s0: usize, s1: usize, a: f32, b: f32) -> Vec<Frame> {
    if s1 <= s0 {
        return Vec::new();
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut phase = 0.0f64;
    let mixed: Vec<Complex32> = cplx[s0..s1]
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let t = (s0 + i) as f32 / fs as f32;
            let fc = a + b * t;
            phase = (phase - two_pi * fc as f64 / fs as f64).rem_euclid(two_pi);
            let (s, c) = phase.sin_cos();
            let (s, c) = (s as f32, c as f32);
            Complex32::new(z.re * c - z.im * s, z.re * s + z.im * c)
        })
        .collect();

    let mut rs = StreamResampler::new(fs, SAMPLE_RATE);
    let mut res = Vec::new();
    rs.process(&mixed, &mut res);
    let mut ilv = Vec::with_capacity(res.len() * 2);
    for c in &res {
        ilv.push(c.re);
        ilv.push(c.im);
    }
    let mut frames = Vec::new();
    if let Some(mut dec) = Decoder::new() {
        dec.feed_iq(&ilv);
        dec.flush();
        while let Some(f) = dec.poll() {
            frames.push(f);
        }
    }
    frames
}

/// Decode ASRTU-1 CCSDS frames from full-pass IQ via the soundmodem C backend.
///
/// Offline three-stage pipeline: survey the `s^2` carrier every second, fit
/// smooth Doppler tracks through the confident detections (RANSAC, constrained
/// by physics), then replay each track through its own decoder in parallel.
/// There is no acquisition threshold — the RS code alone judges a real frame.
pub fn decode_iq(iq: &[IqSample], fs: u32) -> Vec<Frame> {
    use rayon::prelude::*;
    if iq.is_empty() {
        return Vec::new();
    }
    let cplx: Vec<Complex32> = iq.iter().map(|s| Complex32::new(s.i, s.q)).collect();
    let sv = survey(&cplx, fs);
    let anchors: Vec<(f32, f32)> = sv
        .iter()
        .enumerate()
        .filter(|(_, &(_, m))| m >= ANCHOR_MARGIN)
        .map(|(b, &(f, _))| (b as f32, f))
        .collect();
    let tracks = ransac_tracks(&anchors);

    // Flatten every track's spans into one job list and decode them all in
    // parallel, so a single dominant track's spans still use every core.
    let jobs: Vec<(usize, usize, f32, f32)> = tracks
        .iter()
        .flat_map(|tr| {
            let (a, b) = (tr.a, tr.b);
            track_spans(tr, fs, cplx.len())
                .into_iter()
                .map(move |(s0, s1)| (s0, s1, a, b))
        })
        .collect();

    let mut all: Vec<Frame> = jobs
        .par_iter()
        .flat_map(|&(s0, s1, a, b)| decode_span(&cplx, fs, s0, s1, a, b))
        .collect();

    let mut seen = HashSet::new();
    all.retain(|f| seen.insert(f.data.clone()));
    all
}

/// De-rotate the strongest Doppler track over the whole recording and resample
/// to 48 kHz, returning the baseband IQ the streaming decoder expects.
///
/// This is the same survey/fit/de-rotate front-end [`decode_iq`] feeds to the
/// C decoder, stopped one step short of decoding. Exporting its output gives a
/// known-good 48 kHz baseband stream for exercising the live path from a file
/// when no radio is on hand. Returns empty if no track is found.
pub fn derotate_iq_48k(iq: &[IqSample], fs: u32) -> Vec<IqSample> {
    if iq.is_empty() {
        return Vec::new();
    }
    let cplx: Vec<Complex32> = iq.iter().map(|s| Complex32::new(s.i, s.q)).collect();
    let sv = survey(&cplx, fs);
    let anchors: Vec<(f32, f32)> = sv
        .iter()
        .enumerate()
        .filter(|(_, &(_, m))| m >= ANCHOR_MARGIN)
        .map(|(b, &(f, _))| (b as f32, f))
        .collect();
    let tracks = ransac_tracks(&anchors);
    // The strongest track carries the most supporting anchors.
    let Some(best) = tracks.iter().max_by_key(|t| t.ats.len()) else {
        return Vec::new();
    };
    let (a, b) = (best.a, best.b);

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut phase = 0.0f64;
    let mixed: Vec<Complex32> = cplx
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let t = i as f32 / fs as f32;
            let fc = a + b * t;
            phase = (phase - two_pi * fc as f64 / fs as f64).rem_euclid(two_pi);
            let (s, c) = phase.sin_cos();
            let (s, c) = (s as f32, c as f32);
            Complex32::new(z.re * c - z.im * s, z.re * s + z.im * c)
        })
        .collect();

    let mut rs = StreamResampler::new(fs, SAMPLE_RATE);
    let mut res = Vec::new();
    rs.process(&mixed, &mut res);
    res.into_iter()
        .map(|c| IqSample { i: c.re, q: c.im })
        .collect()
}
