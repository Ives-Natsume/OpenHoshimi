//! Band-edge frequency-locked loop for wide carrier acquisition.
//!
//! Our linear BPSK demodulator tracks the residual carrier with a Costas
//! phase-locked loop and, optionally, a Kay squared-symbol FLL. Both have a
//! narrow pull-in on the order of their loop bandwidth, so a recording tuned
//! more than roughly that far off the true carrier never acquires and decodes
//! zero frames while the reference tool (Lilacsat ASRTU-1 soundmodem / GNU
//! Radio) gets several.
//!
//! [`BandEdgeFll`] is a port of GNU Radio's `digital_fll_band_edge_cc`. It
//! estimates frequency error from the signal's spectral band edges rather than
//! from phase, which gives a wide pull-in of about `±2 * baudrate` — for a
//! 9k6 downlink that is roughly `±19.2 kHz`, independent of the sample rate.
//! Placed ahead of the Costas loop it performs coarse frequency acquisition;
//! the Costas loop then does fine phase tracking.
//!
//! The block is complex-in / complex-out: it returns each input sample
//! derotated by its current NCO estimate.

use std::f32::consts::PI;

use openhoshimi_core::IqSample;

/// Cardinal sine, `sin(pi*x) / (pi*x)`, with the removable singularity at the
/// origin evaluated to its limit of `1`.
fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        let arg = PI * x;
        arg.sin() / arg
    }
}

/// A single complex FIR filter running over a circular buffer. Used for the two
/// band-edge filters; kept private to this module.
#[derive(Debug, Clone)]
struct ComplexFir {
    /// Filter taps (complex).
    taps: Vec<IqSample>,
    /// Circular sample history, same length as `taps`.
    history: Vec<IqSample>,
    /// Index of the most recently written history slot.
    index: usize,
}

impl ComplexFir {
    fn new(taps: Vec<IqSample>) -> Self {
        let len = taps.len();
        Self {
            taps,
            history: vec![IqSample::default(); len],
            index: 0,
        }
    }

    /// Push one sample and return the filter output. Matches the reference
    /// convolution: newest sample multiplies `taps[0]`, walking the history
    /// backwards in time against increasing tap index.
    fn process(&mut self, input: IqSample) -> IqSample {
        let len = self.taps.len();
        self.history[self.index] = input;

        let mut acc = IqSample::default();
        let mut buf_idx = self.index;
        for tap in &self.taps {
            let s = self.history[buf_idx];
            acc.i += s.i * tap.i - s.q * tap.q;
            acc.q += s.i * tap.q + s.q * tap.i;
            buf_idx = if buf_idx == 0 { len - 1 } else { buf_idx - 1 };
        }

        self.index += 1;
        if self.index >= len {
            self.index = 0;
        }
        acc
    }
}

/// Band-edge frequency-locked loop (GNU Radio `digital_fll_band_edge_cc`).
///
/// Feed complex baseband samples one at a time with [`process`](Self::process);
/// each call returns the sample derotated by the loop's current frequency
/// estimate. [`frequency`](Self::frequency) exposes the per-sample phase
/// increment (radians/sample) for tests and diagnostics.
#[derive(Debug, Clone)]
pub struct BandEdgeFll {
    lower: ComplexFir,
    upper: ComplexFir,
    /// Proportional loop gain.
    alpha: f32,
    /// Integral loop gain.
    beta: f32,
    /// NCO phase accumulator, radians.
    phase: f32,
    /// NCO frequency estimate, radians/sample.
    freq: f32,
    /// Symmetric clamp on `freq`, radians/sample (`2*pi*(2/sps)`).
    freq_limit: f32,
}

impl BandEdgeFll {
    /// Build a band-edge FLL.
    ///
    /// * `samples_per_symbol` — sample rate divided by baud rate. May be
    ///   fractional (e.g. `26.04` at 250 kHz / 9600 baud).
    /// * `rolloff` — the transmit pulse-shaping roll-off (RRC). ASRTU-1 uses
    ///   `0.35`. Controls where the band edges sit.
    /// * `filter_size` — number of taps in each band-edge filter. Roughly
    ///   `20 * samples_per_symbol` spans ~20 symbols like the reference; the
    ///   caller caps it at a sane maximum.
    /// * `loop_bw` — normalized loop bandwidth (e.g. `0.01`). Wider acquires
    ///   faster but tracks noisier.
    ///
    /// Non-finite or non-positive inputs are clamped to safe values so the
    /// constructor never panics; a `filter_size` below `1` is raised to `1`.
    pub fn new(samples_per_symbol: f32, rolloff: f32, filter_size: usize, loop_bw: f32) -> Self {
        let sps = if samples_per_symbol.is_finite() && samples_per_symbol > 1.0 {
            samples_per_symbol
        } else {
            2.0
        };
        let ro = if rolloff.is_finite() && rolloff > 0.0 {
            rolloff
        } else {
            0.35
        };
        let bw = if loop_bw.is_finite() && loop_bw > 0.0 {
            loop_bw
        } else {
            0.0
        };
        let fsize = filter_size.max(1);

        // Second-order loop constants (critically damped, GNU Radio form).
        let damping = std::f32::consts::FRAC_1_SQRT_2;
        let denom = 1.0 + 2.0 * damping * bw + bw * bw;
        let alpha = 4.0 * damping * bw / denom;
        let beta = 4.0 * bw * bw / denom;

        let freq_limit = 2.0 * PI * (2.0 / sps);

        // Baseband prototype: two sincs straddling the symbol rate, sampled
        // over ~2*M symbols. `M = round(fsize/sps)`.
        let m = (fsize as f32 / sps).round();
        let step = 2.0 / sps;
        let mut bb = Vec::with_capacity(fsize);
        let mut power = 0.0f32;
        for i in 0..fsize {
            let k = -m + i as f32 * step;
            let pos = ro * k;
            let tap = sinc(pos - 0.5) + sinc(pos + 0.5);
            power += tap * tap;
            bb.push(tap);
        }
        let inv_power = if power > 0.0 { 1.0 / power } else { 1.0 };

        // Spin the normalized prototype up and down to the band edges at
        // `(1 + ro)` times the half-symbol rate. Taps are stored reversed so
        // the FIR's newest-sample-first convolution matches the reference.
        let n = (fsize as f32 - 1.0) / 2.0;
        let half_over_sps = 0.5 / sps;
        let mut lower_taps = vec![IqSample::default(); fsize];
        let mut upper_taps = vec![IqSample::default(); fsize];
        for (i, &raw) in bb.iter().enumerate() {
            let tap = raw * inv_power;
            let k = (i as f32 - n) * half_over_sps;
            let arg = 2.0 * PI * (1.0 + ro) * k;
            let (sin_a, cos_a) = arg.sin_cos();
            let index = fsize - 1 - i;
            // Lower edge: spin down (multiply by exp(-j*arg)).
            lower_taps[index] = IqSample {
                i: tap * cos_a,
                q: -tap * sin_a,
            };
            // Upper edge: spin up (multiply by exp(+j*arg)).
            upper_taps[index] = IqSample {
                i: tap * cos_a,
                q: tap * sin_a,
            };
        }

        Self {
            lower: ComplexFir::new(lower_taps),
            upper: ComplexFir::new(upper_taps),
            alpha,
            beta,
            phase: 0.0,
            freq: 0.0,
            freq_limit,
        }
    }

    /// Current NCO frequency estimate in radians per sample. Multiply by
    /// `sample_rate / (2*pi)` to read it in Hz.
    pub fn frequency(&self) -> f32 {
        self.freq
    }

    /// Derotate one sample and advance the loop.
    ///
    /// The sample is mixed with `exp(+j*phase)`, run through both band-edge
    /// filters, and the squared-magnitude difference of the two outputs drives
    /// a PI update of the NCO. The returned sample is the derotated input.
    pub fn process(&mut self, input: IqSample) -> IqSample {
        let (sin_p, cos_p) = self.phase.sin_cos();
        // nco = exp(+j*phase); y = input * nco.
        let y = IqSample {
            i: input.i * cos_p - input.q * sin_p,
            q: input.i * sin_p + input.q * cos_p,
        };

        let o_lo = self.lower.process(y);
        let o_hi = self.upper.process(y);
        let mag_hi = o_hi.i * o_hi.i + o_hi.q * o_hi.q;
        let mag_lo = o_lo.i * o_lo.i + o_lo.q * o_lo.q;
        // Discriminator: upper-edge power minus lower-edge power. When the
        // carrier sits high, the upper edge carries more energy so the error
        // is positive and the loop drives `phase` up, mixing with exp(+j*phase)
        // to rotate the spectrum back down toward centre. This matches GNU
        // Radio's `digital_fll_band_edge_cc` and is validated by the
        // acquisition unit test (a sign flip diverges).
        let error = mag_hi - mag_lo;

        self.freq += self.beta * error;
        self.freq = self.freq.clamp(-self.freq_limit, self.freq_limit);
        self.phase += self.freq + self.alpha * error;
        // Wrap phase to keep the sin/cos arguments small.
        while self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        while self.phase < -2.0 * PI {
            self.phase += 2.0 * PI;
        }

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Feed a synthetic BPSK signal with a known carrier offset and assert the
    /// FLL converges to that offset AND derotates the output. This is the
    /// SIGN-correctness guard: if the discriminator sign were wrong the loop
    /// would run to the frequency clamp instead of the true offset.
    #[test]
    fn acquires_known_bpsk_offset() {
        let sample_rate = 50_000.0f32;
        let baudrate = 9_600.0f32;
        let sps = sample_rate / baudrate; // ~5.208
        let offset_hz = 8_000.0f32;

        // Random ±1 BPSK, upsampled, then spun by the carrier offset.
        let mut lcg: u32 = 0x1234_5678;
        let n_symbols = 20_000usize;
        let sps_i = sps.round() as usize;
        let increment = TAU * offset_hz / sample_rate;
        let mut carrier = 0.0f32;
        let mut samples = Vec::with_capacity(n_symbols * sps_i);
        for _ in 0..n_symbols {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bit = (lcg >> 16) & 1;
            let sym = if bit == 1 { 1.0 } else { -1.0 };
            for _ in 0..sps_i {
                let (s, c) = carrier.sin_cos();
                samples.push(IqSample {
                    i: sym * c,
                    q: sym * s,
                });
                carrier += increment;
                if carrier >= TAU {
                    carrier -= TAU;
                }
            }
        }

        let fsize = (20.0 * sps).round() as usize;
        let mut fll = BandEdgeFll::new(sps, 0.35, fsize, 0.02);

        let mut out = Vec::with_capacity(samples.len());
        for &s in &samples {
            out.push(fll.process(s));
        }

        // Converged frequency (rad/sample) -> Hz. The signal sits at +offset;
        // the NCO mixes with exp(+j*phase), so to null a +offset carrier the
        // loop must settle its estimate at -offset. Acquisition therefore means
        // `est_hz ≈ -offset_hz`. If the discriminator sign were wrong the loop
        // would run to +offset (positive feedback) or to the frequency clamp
        // instead, so this is the sign-correctness guard.
        let est_hz = fll.frequency() * sample_rate / TAU;
        assert!(
            (est_hz + offset_hz).abs() < 500.0,
            "FLL did not acquire: estimate {est_hz:.1} Hz, expected ~{:.1} Hz to \
             null a +{offset_hz:.1} Hz carrier (freq={:.5} rad/sample). A large \
             miss or the wrong sign means the discriminator sign is wrong.",
            -offset_hz,
            fll.frequency()
        );

        // The derotated tail should be a clean ±1 BPSK constellation: energy
        // collapsed onto one axis. Measure the fraction of |q|/|i| over the
        // last 10% of samples; a derotated signal keeps most power in-phase.
        let tail = &out[out.len() * 9 / 10..];
        let mut in_phase_power = 0.0f32;
        let mut quad_power = 0.0f32;
        for s in tail {
            in_phase_power += s.i * s.i;
            quad_power += s.q * s.q;
        }
        assert!(
            quad_power < 0.5 * in_phase_power,
            "output not derotated: quad power {quad_power:.1} vs in-phase {in_phase_power:.1}"
        );
    }

    /// With zero offset the loop must stay put (no self-driving drift).
    #[test]
    fn stays_locked_on_zero_offset() {
        let sps = 5.208f32;
        let fsize = (20.0 * sps).round() as usize;
        let mut fll = BandEdgeFll::new(sps, 0.35, fsize, 0.02);

        let mut lcg: u32 = 0xdead_beef;
        for _ in 0..40_000 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sym = if (lcg >> 16) & 1 == 1 { 1.0 } else { -1.0 };
            fll.process(IqSample { i: sym, q: 0.0 });
        }

        let est_hz = fll.frequency() * 50_000.0 / TAU;
        assert!(
            est_hz.abs() < 300.0,
            "FLL drifted on zero offset: {est_hz:.1} Hz"
        );
    }
}
