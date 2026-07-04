//! Decimating channel filter for wideband IQ recordings.
//!
//! A recording captured at a rate far above the symbol rate (for example a
//! 250 kHz SDR baseband carrying a 9k6 signal) wastes the demodulator's loops
//! on out-of-band noise and runs them at an inefficient samples-per-symbol.
//! [`ChannelFilter`] pulls a single narrow channel down to baseband and
//! decimates it to a working rate near the demodulator's sweet spot:
//!
//! 1. Derotate by `-offset_hz` with a per-sample NCO so the target carrier
//!    lands at DC.
//! 2. Low-pass with a Hamming-windowed sinc whose cutoff sits just below the
//!    post-decimation Nyquist frequency. This both anti-aliases the following
//!    decimation and rejects interferers/noise outside the channel, which is
//!    a direct SNR gain on the retained signal.
//! 3. Keep every `decimation`th filtered sample.
//!
//! The filter is intentionally simple and stateful: feed it chunks with
//! [`ChannelFilter::push`] and it carries its NCO phase, FIR history, and
//! decimation phase across calls.

use std::f32::consts::TAU;

use openhoshimi_core::IqSample;

use crate::cpm::sinc_lowpass_taps;

/// Complex FIR low-pass with independent history on the I and Q rails.
///
/// Kept private to this module; [`ChannelFilter`] is the public surface.
#[derive(Debug, Clone)]
struct ComplexLowpass {
    taps: Vec<f32>,
    hist_i: Vec<f32>,
    hist_q: Vec<f32>,
    head: usize,
}

impl ComplexLowpass {
    fn new(taps: Vec<f32>) -> Self {
        let len = taps.len().max(1);
        Self {
            taps,
            hist_i: vec![0.0; len],
            hist_q: vec![0.0; len],
            head: 0,
        }
    }

    fn push(&mut self, sample: IqSample) -> IqSample {
        if self.taps.is_empty() {
            return sample;
        }
        let len = self.hist_i.len();
        self.hist_i[self.head] = sample.i;
        self.hist_q[self.head] = sample.q;
        let mut acc_i = 0.0f32;
        let mut acc_q = 0.0f32;
        for (offset, tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            acc_i += self.hist_i[index] * tap;
            acc_q += self.hist_q[index] * tap;
        }
        self.head = (self.head + 1) % len;
        IqSample { i: acc_i, q: acc_q }
    }
}

/// Derotating, low-pass, decimating channel selector for complex IQ.
#[derive(Debug, Clone)]
pub struct ChannelFilter {
    /// Radians per input sample of the derotation NCO (`-2*pi*offset/fs`).
    phase_increment: f32,
    /// Current NCO phase, wrapped to keep it bounded.
    phase: f32,
    lowpass: ComplexLowpass,
    decimation: u32,
    /// Counts input samples modulo `decimation` so output cadence survives
    /// across `push` calls.
    decim_phase: u32,
}

impl ChannelFilter {
    /// Build a channel filter that shifts `offset_hz` to DC and decimates by
    /// `decimation`.
    ///
    /// The low-pass cutoff is placed at `0.45 / decimation` cycles per input
    /// sample, i.e. 90% of the post-decimation Nyquist frequency. At a 5x
    /// decimation of a 250 kHz recording that is a ~22.5 kHz cutoff on the
    /// 50 kHz output rail — wide enough to pass a 9k6 BPSK channel plus LEO
    /// Doppler, narrow enough to reject everything else in the 250 kHz span.
    ///
    /// `decimation` is clamped to at least 1; a value of 1 still derotates and
    /// low-passes (useful to reject out-of-band energy without changing rate).
    pub fn new(sample_rate: u32, offset_hz: f32, decimation: u32) -> Self {
        let decimation = decimation.max(1);
        let decim = decimation as f32;
        // Cutoff and tap count mirror the well-tested anti-alias design used by
        // the CPM decimation stage: a wider decimation needs a proportionally
        // longer FIR to hold the transition band.
        let taps = if decimation > 1 {
            let cutoff = 0.45f32 / decim;
            let span = (8.0 * decim).ceil() as usize;
            let len = span | 1;
            sinc_lowpass_taps(cutoff, len)
        } else {
            Vec::new()
        };
        let phase_increment = if sample_rate > 0 {
            -TAU * offset_hz / sample_rate as f32
        } else {
            0.0
        };
        Self {
            phase_increment,
            phase: 0.0,
            lowpass: ComplexLowpass::new(taps),
            decimation,
            decim_phase: 0,
        }
    }

    /// Integer decimation factor applied by this filter.
    pub fn decimation(&self) -> u32 {
        self.decimation
    }

    /// Push a chunk of input samples and return the decimated channel output.
    pub fn push(&mut self, samples: &[IqSample]) -> Vec<IqSample> {
        let mut out = Vec::with_capacity(samples.len() / self.decimation as usize + 1);
        for &sample in samples {
            // Derotate: multiply by exp(j*phase) where phase = -2*pi*offset*t.
            let (sin, cos) = self.phase.sin_cos();
            let rotated = IqSample {
                i: sample.i * cos - sample.q * sin,
                q: sample.i * sin + sample.q * cos,
            };
            self.phase += self.phase_increment;
            if self.phase > TAU {
                self.phase -= TAU;
            } else if self.phase < -TAU {
                self.phase += TAU;
            }

            let filtered = self.lowpass.push(rotated);

            self.decim_phase += 1;
            if self.decim_phase >= self.decimation {
                self.decim_phase = 0;
                out.push(filtered);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sample_rate: u32, freq_hz: f32, n: usize) -> Vec<IqSample> {
        (0..n)
            .map(|k| {
                let phase = TAU * freq_hz * k as f32 / sample_rate as f32;
                IqSample {
                    i: phase.cos(),
                    q: phase.sin(),
                }
            })
            .collect()
    }

    fn mean_power(samples: &[IqSample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = samples.iter().map(|s| s.i * s.i + s.q * s.q).sum();
        sum / samples.len() as f32
    }

    #[test]
    fn decimates_by_the_requested_factor() {
        let mut filter = ChannelFilter::new(250_000, 0.0, 5);
        let input = tone(250_000, 0.0, 5000);
        let out = filter.push(&input);
        assert_eq!(out.len(), 1000);
        assert_eq!(filter.decimation(), 5);
    }

    #[test]
    fn derotation_moves_the_target_carrier_to_dc() {
        // A tone at +40 kHz derotated by +40 kHz should land at DC: a constant
        // (up to filter warm-up), so its post-filter phase advance is ~0.
        let sample_rate = 250_000u32;
        let mut filter = ChannelFilter::new(sample_rate, 40_000.0, 5);
        let input = tone(sample_rate, 40_000.0, 20_000);
        let out = filter.push(&input);
        let tail = &out[out.len() / 2..];
        // Consecutive-sample dot/cross gives the residual rotation rate. At DC
        // the cross term (imaginary part) averages to ~0.
        let mut cross = 0.0f32;
        let mut dot = 0.0f32;
        for w in tail.windows(2) {
            dot += w[0].i * w[1].i + w[0].q * w[1].q;
            cross += w[0].i * w[1].q - w[0].q * w[1].i;
        }
        let residual = cross.atan2(dot).abs();
        assert!(residual < 0.05, "residual rotation too high: {residual}");
    }

    #[test]
    fn rejects_out_of_band_energy() {
        // A tone at +90 kHz is well outside the ~22.5 kHz channel around DC and
        // must be attenuated hard relative to an in-band DC tone.
        let sample_rate = 250_000u32;
        let n = 40_000;

        let mut pass = ChannelFilter::new(sample_rate, 0.0, 5);
        let in_band = pass.push(&tone(sample_rate, 0.0, n));

        let mut stop = ChannelFilter::new(sample_rate, 0.0, 5);
        let out_band = stop.push(&tone(sample_rate, 90_000.0, n));

        let pass_power = mean_power(&in_band[in_band.len() / 2..]);
        let stop_power = mean_power(&out_band[out_band.len() / 2..]);
        assert!(
            stop_power < pass_power * 1e-3,
            "out-of-band leakage too high: pass={pass_power} stop={stop_power}"
        );
    }
}
