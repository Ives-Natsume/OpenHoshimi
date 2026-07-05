//! Generic continuous-phase modulation demodulator.
//!
//! Covers the binary CPM family used by amateur satellite downlinks: FSK,
//! MSK, GFSK, and GMSK. Pipeline:
//!
//! 1. Optional `swap_iq` and a static frequency-offset NCO.
//! 2. Optional complex IQ low-pass + integer decimation when the input
//!    sample rate is much higher than the symbol rate. This MUST run
//!    before the FM discriminator: atan2 is nonlinear, so out-of-band
//!    noise dominates the per-sample phase delta when the input sample
//!    rate is far above the signal bandwidth.
//! 3. Per-sample noncoherent FM discriminator (atan2 of cross/dot products)
//!    on the decimated IQ stream.
//! 4. For FSK/MSK: integrate-and-dump symbol detection over a fixed
//!    sample window per symbol.
//! 5. For GFSK/GMSK: Gaussian receive matched filter sized from the
//!    configured `gaussian_bt`, then sub-sample interpolation at the
//!    nominal symbol rate.
//! 6. Hard slicer with optional differential decoding and inversion.

use openhoshimi_core::{DecodeError, Demodulator, IqSample};

/// Binary continuous-phase modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpmMode {
    /// Binary frequency-shift keying.
    Fsk,
    /// Minimum-shift keying.
    Msk,
    /// Gaussian-filtered binary frequency-shift keying.
    Gfsk,
    /// Gaussian minimum-shift keying.
    Gmsk,
}

/// Configuration for [`CpmDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct CpmConfig {
    /// IQ sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// CPM waveform family.
    pub mode: CpmMode,
    /// Modulation index. MSK and GMSK usually use `0.5`.
    pub modulation_index: f32,
    /// Gaussian BT product for GFSK/GMSK modes.
    pub gaussian_bt: Option<f32>,
    /// Frequency offset correction in Hz, applied before phase slicing.
    pub frequency_offset_hz: f32,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Swap I and Q before carrier correction.
    pub swap_iq: bool,
    /// Integer decimation factor applied to the post-discriminator sample
    /// stream before symbol detection. `0` selects an automatic factor that
    /// keeps the post-decimation rate near 16 samples per symbol when the
    /// input rate is more than 32x the baud rate; otherwise no decimation
    /// is applied. Set to `1` to force no decimation.
    pub decimation: u32,
    /// Reserved for future closed-loop timing recovery. Currently unused;
    /// the demodulator runs open-loop at the nominal symbol rate.
    pub timing_loop_bandwidth: f32,
}

impl CpmConfig {
    /// Create a configuration with conservative defaults for `mode`.
    pub fn new(sample_rate: u32, baudrate: u32, mode: CpmMode) -> Self {
        Self {
            sample_rate,
            baudrate,
            mode,
            modulation_index: match mode {
                CpmMode::Msk | CpmMode::Gmsk => 0.5,
                CpmMode::Fsk | CpmMode::Gfsk => 1.0,
            },
            gaussian_bt: match mode {
                CpmMode::Gfsk | CpmMode::Gmsk => Some(0.5),
                CpmMode::Fsk | CpmMode::Msk => None,
            },
            frequency_offset_hz: 0.0,
            differential: false,
            invert: false,
            swap_iq: false,
            decimation: 0,
            timing_loop_bandwidth: 0.0,
        }
    }
}

/// Resolve the integer decimation factor used between the discriminator and
/// the symbol detector. `requested == 0` selects an automatic factor that
/// keeps the post-decimation rate near 16 samples per symbol when the input
/// rate is more than 32x the baud rate.
pub(crate) fn resolve_decimation(sample_rate: u32, baudrate: u32, requested: u32) -> u32 {
    if requested >= 1 {
        return requested;
    }
    if baudrate == 0 {
        return 1;
    }
    let sps_in = sample_rate as f32 / baudrate as f32;
    if sps_in <= 32.0 {
        return 1;
    }
    const TARGET_SPS: f32 = 16.0;
    let factor = (sps_in / TARGET_SPS).round() as u32;
    factor.max(1)
}

/// Noncoherent IQ demodulator for FSK/MSK/GFSK/GMSK signals.
#[derive(Debug, Clone)]
pub struct CpmDemodulator {
    config: CpmConfig,
    last_sample: Option<IqSample>,
    previous_symbol: Option<u8>,
    carrier_phase: f32,
    carrier_increment: f32,
    decimation: u32,
    decim_counter: u32,
    decim_lpf: Option<ComplexFir>,
    sampler: SymbolSampler,
}

impl CpmDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: CpmConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        let decimation = resolve_decimation(config.sample_rate, config.baudrate, config.decimation);
        let effective_rate = config.sample_rate as f32 / decimation as f32;
        let samples_per_symbol = effective_rate / config.baudrate as f32;
        let decim_lpf = if decimation > 1 {
            Some(ComplexFir::lowpass(decimation))
        } else {
            None
        };
        let sampler = match (config.mode, config.gaussian_bt) {
            (CpmMode::Gfsk | CpmMode::Gmsk, Some(bt)) => SymbolSampler::Filtered {
                filter: GaussianFir::new(bt, samples_per_symbol),
                interp: TrackingInterpolator::new(samples_per_symbol, 0.05),
            },
            _ => SymbolSampler::IntegrateAndDump {
                integrator: IntegrateAndDump::new(samples_per_symbol),
            },
        };
        Ok(Self {
            carrier_increment: -std::f32::consts::TAU * config.frequency_offset_hz
                / config.sample_rate as f32,
            config,
            last_sample: None,
            previous_symbol: None,
            carrier_phase: 0.0,
            decimation,
            decim_counter: 0,
            decim_lpf,
            sampler,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> CpmConfig {
        self.config
    }

    /// Return the integer decimation factor applied between the
    /// discriminator and the symbol detector.
    pub fn decimation(&self) -> u32 {
        self.decimation
    }

    fn hard_slice(&mut self, sample: f32) -> u8 {
        let mut symbol = u8::from(sample >= 0.0);
        if self.config.invert {
            symbol ^= 1;
        }

        if self.config.differential {
            let decoded = match self.previous_symbol {
                Some(previous) => symbol ^ previous,
                None => symbol,
            };
            self.previous_symbol = Some(symbol);
            decoded
        } else {
            symbol
        }
    }

    fn normalize_sample(&self, sample: IqSample) -> IqSample {
        if self.config.swap_iq {
            IqSample {
                i: sample.q,
                q: sample.i,
            }
        } else {
            sample
        }
    }

    fn correct_frequency(&mut self, sample: IqSample) -> IqSample {
        if self.config.frequency_offset_hz == 0.0 {
            return sample;
        }

        let sin = self.carrier_phase.sin();
        let cos = self.carrier_phase.cos();
        self.carrier_phase += self.carrier_increment;
        if self.carrier_phase >= std::f32::consts::TAU
            || self.carrier_phase <= -std::f32::consts::TAU
        {
            self.carrier_phase %= std::f32::consts::TAU;
        }

        IqSample {
            i: sample.i * cos - sample.q * sin,
            q: sample.i * sin + sample.q * cos,
        }
    }
}

impl Demodulator for CpmDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            let sample = self.correct_frequency(self.normalize_sample(sample));

            let filtered_iq = match &mut self.decim_lpf {
                Some(lpf) => lpf.push(sample),
                None => sample,
            };
            if self.decimation > 1 {
                self.decim_counter += 1;
                if self.decim_counter < self.decimation {
                    continue;
                }
                self.decim_counter = 0;
            }

            let delta = match self.last_sample {
                Some(previous) => phase_delta(previous, filtered_iq),
                None => 0.0,
            };
            self.last_sample = Some(filtered_iq);

            if let Some(symbol_value) = self.sampler.push(delta) {
                bits.push(self.hard_slice(symbol_value));
            }
        }

        bits
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn baudrate(&self) -> u32 {
        self.config.baudrate
    }
}

fn validate_config(config: CpmConfig) -> Result<(), DecodeError> {
    if config.sample_rate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM sample rate must be greater than zero".to_string(),
        ));
    }
    if config.baudrate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM baudrate must be greater than zero".to_string(),
        ));
    }
    if config.sample_rate < config.baudrate * 2 {
        return Err(DecodeError::InvalidEncoding(
            "CPM sample rate must be at least 2x the baudrate".to_string(),
        ));
    }
    if config.modulation_index <= 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM modulation index must be greater than zero".to_string(),
        ));
    }
    if let Some(bt) = config.gaussian_bt {
        if bt <= 0.0 {
            return Err(DecodeError::InvalidEncoding(
                "CPM gaussian BT must be greater than zero".to_string(),
            ));
        }
    }
    Ok(())
}

fn phase_delta(previous: IqSample, current: IqSample) -> f32 {
    let dot = previous.i.mul_add(current.i, previous.q * current.q);
    let cross = previous.i * current.q - previous.q * current.i;
    cross.atan2(dot)
}

#[derive(Debug, Clone)]
enum SymbolSampler {
    IntegrateAndDump {
        integrator: IntegrateAndDump,
    },
    Filtered {
        filter: GaussianFir,
        interp: TrackingInterpolator,
    },
}

impl SymbolSampler {
    fn push(&mut self, delta: f32) -> Option<f32> {
        match self {
            SymbolSampler::IntegrateAndDump { integrator } => integrator.push(delta),
            SymbolSampler::Filtered { filter, interp } => {
                let filtered = filter.push(delta);
                interp.push(filtered)
            }
        }
    }
}

/// Integrate-and-dump symbol detector for unfiltered FSK/MSK signals. Sums
/// the post-discriminator samples over a fractional symbol period and emits
/// the accumulated value as the symbol estimate.
#[derive(Debug, Clone)]
pub(crate) struct IntegrateAndDump {
    samples_per_symbol: f32,
    sample_phase: f32,
    accumulator: f32,
}

impl IntegrateAndDump {
    pub(crate) fn new(samples_per_symbol: f32) -> Self {
        Self {
            samples_per_symbol,
            sample_phase: 0.0,
            accumulator: 0.0,
        }
    }

    pub(crate) fn push(&mut self, delta: f32) -> Option<f32> {
        self.accumulator += delta;
        self.sample_phase += 1.0;
        if self.sample_phase >= self.samples_per_symbol {
            let symbol = self.accumulator;
            self.accumulator = 0.0;
            self.sample_phase -= self.samples_per_symbol;
            Some(symbol)
        } else {
            None
        }
    }
}

/// Closed-loop symbol-rate interpolator with Mueller-Müller decision-
/// directed timing recovery for binary signals. The proportional path
/// tracks short-term symbol-clock jitter via an instant `mu` correction;
/// an optional integral path feeds the per-input phase increment `omega`
/// to absorb steady-state PPM drift over long recordings. The loop update
/// is gated by a leak-decay envelope estimate so timing does not wander
/// on the noise/silence between bursts.
///
/// The integral path is opt-in for a reason. It is the right tool when
/// the input has a sustained clock-rate offset that drifts over many
/// seconds — for example, multi-minute SatNOGS captures of a passing
/// satellite where ground-station and spacecraft oscillators differ by
/// a small but measurable PPM. On those recordings, leaving the loop
/// first-order costs hundreds of frames of CRC-valid data because the
/// proportional-only loop cannot track a sustained rate offset, only
/// instantaneous phase. With the integral path on, error from per-symbol
/// noise is averaged into the rate estimate over many symbols and
/// converges to the true PPM offset.
///
/// On short, strong-signal IQ recordings the steady-state offset is
/// negligible and the integral path is a liability: a few burst-edge
/// noise samples can shift the rate estimate enough to corrupt the rest
/// of the burst, with no in-band mechanism to detect and reset. We tried
/// several mitigations (leaky integrator, gate-only updates, warm-up
/// delay) on the unified loop; each helped marginally but none recovered
/// the clean output that a first-order loop produces on the same input.
///
/// So the loop is structured to let the *caller* choose. FM-audio
/// recordings (which have already been demodulated by an external SDR
/// front-end and reach us as long, drifty captures) opt into the second
/// order. IQ-domain demodulation (where carrier recovery and Doppler
/// compensation already happen upstream of the symbol clock) stays
/// first-order. This is not a per-satellite knob — it is a per-input-
/// type choice that follows the physical structure of the signal chain.
#[derive(Debug, Clone)]
pub(crate) struct TrackingInterpolator {
    omega: f32,
    omega_min: f32,
    omega_max: f32,
    mu: f32,
    prev_input: f32,
    last_output: f32,
    have_output: bool,
    primed: bool,
    gain_p: f32,
    gain_i: f32,
    envelope: f32,
}

impl TrackingInterpolator {
    /// First-order proportional-only Mueller-Müller. Right for inputs
    /// where the symbol-clock offset is small and stable over the
    /// recording length — typically IQ captures of a single short pass.
    pub(crate) fn new(samples_per_symbol: f32, gain: f32) -> Self {
        Self::with_integral(samples_per_symbol, gain, 0.0)
    }

    /// Second-order proportional + integral Mueller-Müller. Right for
    /// long recordings where the symbol clock drifts (Doppler-induced
    /// rate variation, oscillator PPM offset). `gain_i` is typically
    /// `gain_p / 16` for a critically-damped response.
    pub(crate) fn with_integral(samples_per_symbol: f32, gain_p: f32, gain_i: f32) -> Self {
        let nominal = 1.0 / samples_per_symbol;
        let bound = nominal * 0.01;
        Self {
            omega: nominal,
            omega_min: nominal - bound,
            omega_max: nominal + bound,
            mu: 0.0,
            prev_input: 0.0,
            last_output: 0.0,
            have_output: false,
            primed: false,
            gain_p,
            gain_i,
            envelope: 0.0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> Option<f32> {
        if !self.primed {
            self.prev_input = sample;
            self.primed = true;
            return None;
        }

        let next_phase = self.mu + self.omega;
        if next_phase >= 1.0 {
            let frac = ((1.0 - self.mu) / self.omega).clamp(0.0, 1.0);
            let symbol = lerp(self.prev_input, sample, frac);
            let abs_symbol = symbol.abs();

            self.envelope = (self.envelope * 0.999).max(abs_symbol);

            let mut new_mu = next_phase - 1.0;
            if self.have_output {
                let abs_prev = self.last_output.abs();
                let gate = self.envelope * 0.4;
                if abs_symbol >= gate && abs_prev >= gate {
                    let sign_curr = if symbol >= 0.0 { 1.0 } else { -1.0 };
                    let sign_prev = if self.last_output >= 0.0 { 1.0 } else { -1.0 };
                    let error = self.last_output * sign_curr - symbol * sign_prev;
                    new_mu += self.gain_p * error;
                    if new_mu < 0.0 {
                        new_mu = 0.0;
                    } else if new_mu >= 1.0 {
                        new_mu = 1.0 - self.omega * 0.5;
                    }
                    if self.gain_i > 0.0 {
                        self.omega = (self.omega + self.gain_i * error)
                            .clamp(self.omega_min, self.omega_max);
                    }
                }
            }
            self.mu = new_mu;
            self.last_output = symbol;
            self.have_output = true;
            self.prev_input = sample;
            Some(symbol)
        } else {
            self.mu = next_phase;
            self.prev_input = sample;
            None
        }
    }
}

pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Closed-loop symbol-rate sampler with Gardner timing-error detector and
/// a proportional-integral loop filter. Mirrors gr-satellites'
/// `symbol_sync_ff(TED_GARDNER, sps, clk_bw, damping, ted_gain,
/// clk_limit*sps, 1, constellation_bpsk, IR_PFB_NO_MF)` in
/// `python/components/demodulators/fsk_demodulator.py:133`.
///
/// Gardner TED computes timing error without needing reliable symbol
/// decisions: `e = y_mid * (y[n] - y[n-1])`, where `y_mid` is the sample
/// halfway between the previous and current symbol instants and `y[n]`,
/// `y[n-1]` are the two symbol-instant samples. The mid-symbol sample
/// scaled by the slope of the matched-filter output across the decision
/// instant averages to zero when the clock is centred and grows with the
/// sign of the timing offset otherwise. It is decision-free, so it works
/// even when individual BPSK decisions are wrong — decisive at low SNR
/// where Mueller-Muller's decision-driven error term degenerates into
/// noise.
///
/// The PI loop tracks both per-symbol jitter (proportional) and slow
/// rate offsets like spacecraft Doppler/PPM (integral) without the
/// caller having to choose between first- and second-order responses.
/// gr-satellites uses `clk_bw=0.06` (relative to symbol rate),
/// `damping=1.0`, `ted_gain=1.47`, and clamps the instantaneous symbol
/// period to `+/- clk_limit*sps` around nominal with `clk_limit=0.004`.
#[derive(Debug, Clone)]
pub(crate) struct GardnerTracker {
    period: f32,
    period_avg: f32,
    period_min: f32,
    period_max: f32,
    alpha: f32,
    beta: f32,
    /// Sample index of the next symbol decision instant (fractional).
    next_symbol: f32,
    /// Sample index of the next mid-symbol sample (fractional).
    next_mid: f32,
    /// Symbol-instant value of the previous symbol, the left endpoint of
    /// the Gardner slope `y[n] - y[n-1]`. Both axes are kept so the TED can be
    /// evaluated as the rotation-invariant `Re{conj(y_mid) * (y[n]-y[n-1])}`,
    /// which stays valid when the carrier phase is not yet removed (i.e. when
    /// timing runs ahead of the Costas loop).
    prev_symbol: f32,
    prev_symbol_aux: f32,
    /// Mid-symbol complex sample `(i, q)` captured but not yet consumed by the
    /// current symbol-time update. `None` between symbols. This is the sample
    /// halfway between the previous and current symbol instants.
    mid_pending: Option<(f32, f32)>,
    /// True once we have a previous symbol value, so the Gardner slope is
    /// meaningful and a TED update can be generated.
    have_prev_symbol: bool,
    /// Number of input samples consumed so far. Float so it interleaves
    /// cleanly with the fractional `next_*` schedules.
    cursor: f32,
    /// Input sample at index `cursor - 1`, used as the left endpoint of
    /// linear interpolation at sub-sample times.
    prev_input: f32,
    /// Auxiliary-channel sample at index `cursor - 1`, interpolated with
    /// the SAME fraction as `prev_input` so a second axis (e.g. the Q of a
    /// complex symbol) is decided at the identical recovered instant. The
    /// timing loop is driven only by the primary channel; the auxiliary is
    /// a passive rider. Meaningful only via [`GardnerTracker::push_dual`].
    prev_input_aux: f32,
    primed: bool,
}

impl GardnerTracker {
    /// Build a Gardner tracker with gr-satellites' loop shape (`clk_bw=0.06`,
    /// `damping=1.0`, `ted_gain=1.47`) but a wider period clamp
    /// (`clk_limit=0.08`, vs gr-satellites' `0.004`).
    ///
    /// gr-satellites' tight `0.004` clamp assumes the symbol clock arrives
    /// already near-centred from an upstream coarse timing stage. This
    /// demodulator instead acquires timing cold, directly on the post-Costas
    /// stream, so the loop must be free to slew the instantaneous period far
    /// enough to null the initial sub-sample phase/period offset. At the
    /// ASRTU-1 operating point (50 kHz / 9600 baud = 5.208 samples/symbol) a
    /// `0.004` clamp bounds the period to +/-0.02 samples, too small to
    /// acquire from cold and the loop never locks; widening to `0.08`
    /// (+/-0.42 samples) lets it acquire while staying well inside a symbol
    /// period. Empirically this is a broad, stable plateau on a real
    /// recording (clk_limit 0.06/0.08/0.10 all decode ~27-29 frames), not a
    /// knife-edge. `samples_per_symbol` is the nominal period at the input
    /// rate.
    pub(crate) fn new(samples_per_symbol: f32) -> Self {
        Self::with_constants(samples_per_symbol, 0.06, 1.0, 1.47, 0.08)
    }

    /// Build a Gardner tracker with explicit loop constants.
    ///
    /// `clk_bw` is the loop natural frequency normalized to the symbol
    /// rate (cycles per symbol). `damping` is the loop damping ratio.
    /// `ted_gain` divides both PI gains to compensate for the TED's
    /// inherent slope (Gardner's empirical value is `1.47 / symbol`).
    /// `clk_limit` is the maximum fractional period excursion from
    /// nominal: the instantaneous period stays within
    /// `[sps*(1-clk_limit), sps*(1+clk_limit)]`.
    pub(crate) fn with_constants(
        samples_per_symbol: f32,
        clk_bw: f32,
        damping: f32,
        ted_gain: f32,
        clk_limit: f32,
    ) -> Self {
        let omega_n_t = clk_bw;
        let denom = 1.0 + 2.0 * damping * omega_n_t + omega_n_t * omega_n_t;
        // Standard 2nd-order PLL design (GNU Radio control_loop): the raw
        // alpha/beta target a unit-gain TED. Divide by ted_gain to refer
        // the gains to Gardner's actual slope.
        let alpha = (4.0 * damping * omega_n_t) / (denom * ted_gain);
        let beta = (4.0 * omega_n_t * omega_n_t) / (denom * ted_gain);
        let bound = samples_per_symbol * clk_limit;
        Self {
            period: samples_per_symbol,
            period_avg: samples_per_symbol,
            period_min: samples_per_symbol - bound,
            period_max: samples_per_symbol + bound,
            alpha,
            beta,
            // Schedule the first symbol decision at the natural matched-
            // filter peak position (sps/2 after the stream starts), and the
            // first mid-sample half a period after it (at `sps`), which is
            // the midpoint between the first and second symbol instants. The
            // first symbol fires without a TED update because
            // `have_prev_symbol` is still false; the first update lands at
            // the second symbol, which matches gr-satellites' behaviour:
            // `symbol_sync_ff` does not generate a TED update on its very
            // first output either.
            next_symbol: samples_per_symbol * 0.5,
            next_mid: samples_per_symbol,
            prev_symbol: 0.0,
            prev_symbol_aux: 0.0,
            mid_pending: None,
            have_prev_symbol: false,
            cursor: 0.0,
            prev_input: 0.0,
            prev_input_aux: 0.0,
            primed: false,
        }
    }

    /// Push one input sample. Returns the BPSK soft value at the current
    /// symbol decision instant when one is reached on this sample, else
    /// `None`. The soft value is the matched-filter output interpolated
    /// at the recovered symbol time; downstream callers slice it.
    ///
    /// The single-channel entry point mirroring gr-satellites' real-valued
    /// `symbol_sync_ff`. The linear IQ demodulator carries both axes through
    /// [`Self::push_dual`], so in the library build this thin real-valued
    /// wrapper is reached only from the tracker's own unit test; the
    /// `dead_code` allow keeps it available as the documented scalar API
    /// without tripping `-D warnings` on the non-test lib target.
    #[allow(dead_code)]
    pub(crate) fn push(&mut self, sample: f32) -> Option<f32> {
        self.advance(sample, 0.0).map(|(primary, _)| primary)
    }

    /// Push one complex input sample, driving the timing loop from the
    /// primary channel `i` and interpolating the auxiliary channel `q` at
    /// the SAME recovered instant with the SAME fraction. Returns
    /// `Some((i, q))` at a symbol decision instant, else `None`.
    ///
    /// The Gardner TED and PI loop see only `i`, so the recovered clock is
    /// identical to what [`Self::push`] would produce for the same `i`
    /// stream — `q` rides that clock passively. This keeps both axes of a
    /// complex symbol decided together (required for QPSK/OQPSK) while
    /// letting I own the clock, which is well defined for BPSK/DBPSK after
    /// Costas lock puts the information on I.
    pub(crate) fn push_dual(&mut self, i: f32, q: f32) -> Option<(f32, f32)> {
        self.advance(i, q)
    }

    /// Shared clock/TED core. `primary` drives the Gardner loop; `aux` is
    /// interpolated at the identical sub-sample instant and returned
    /// alongside the primary decision. Callers that only need the primary
    /// channel pass a dummy `aux` and discard it.
    fn advance(&mut self, primary: f32, aux: f32) -> Option<(f32, f32)> {
        if !self.primed {
            self.prev_input = primary;
            self.prev_input_aux = aux;
            self.cursor = 1.0;
            self.primed = true;
            return None;
        }

        // Capture the mid-symbol value first so it is always strictly older
        // than the symbol-time value used in the same Gardner update. With
        // the schedule `next_mid = next_symbol - T/2`, the mid event always
        // comes before the symbol event in the stream, so this ordering is
        // correct. Both axes are interpolated so the rotation-invariant TED
        // below can use the quadrature component.
        if self.cursor >= self.next_mid {
            let frac = (self.next_mid - (self.cursor - 1.0)).clamp(0.0, 1.0);
            self.mid_pending = Some((
                lerp(self.prev_input, primary, frac),
                lerp(self.prev_input_aux, aux, frac),
            ));
            self.next_mid = f32::INFINITY;
        }

        let mut emitted = None;
        if self.cursor >= self.next_symbol {
            let frac = (self.next_symbol - (self.cursor - 1.0)).clamp(0.0, 1.0);
            let symbol = lerp(self.prev_input, primary, frac);
            // Same fraction, same endpoints index: the auxiliary axis is
            // sampled at the identical recovered instant as the primary.
            let symbol_aux = lerp(self.prev_input_aux, aux, frac);

            if let Some((mid_i, mid_q)) = self.mid_pending.take() {
                if self.have_prev_symbol {
                    // Rotation-invariant Gardner TED:
                    //   e = Re{ conj(y_mid) * (y[n] - y[n-1]) }
                    //     = mid_i*(sym_i - prev_i) + mid_q*(sym_q - prev_q).
                    // The classic scalar form uses only the in-phase term,
                    // which requires the constellation to already sit on I --
                    // i.e. the Costas loop must run BEFORE timing. Summing the
                    // quadrature term makes the detector invariant to an
                    // un-removed carrier phase, so timing can run on the
                    // still-rotating stream and the Costas loop can follow it.
                    // On the Costas-first path q ~= 0, so this reduces to the
                    // scalar TED. The mid sample scaled by the slope across the
                    // decision instant averages to zero when the clock is
                    // centred; it is decision-free, so it works even when
                    // individual BPSK decisions are wrong.
                    //
                    // Sign: a POSITIVE error means the sampler is LATE, so the
                    // period must shorten to advance the next instant.
                    let error = mid_i * (symbol - self.prev_symbol)
                        + mid_q * (symbol_aux - self.prev_symbol_aux);
                    self.period_avg = (self.period_avg - self.beta * error)
                        .clamp(self.period_min, self.period_max);
                    self.period = (self.period_avg - self.alpha * error)
                        .clamp(self.period_min, self.period_max);
                }
            }
            self.prev_symbol = symbol;
            self.prev_symbol_aux = symbol_aux;
            self.have_prev_symbol = true;

            self.next_symbol += self.period;
            self.next_mid = self.next_symbol - 0.5 * self.period;
            emitted = Some((symbol, symbol_aux));
        }

        self.prev_input = primary;
        self.prev_input_aux = aux;
        self.cursor += 1.0;

        // `cursor` and the `next_*` schedules accumulate unbounded over the
        // whole stream. In f32 they lose integer resolution past 2^24 samples
        // (~67 s at 250 kHz): `cursor += 1.0` stops advancing entirely and the
        // timing loop freezes, silently truncating a long recording. Rebase by
        // a whole-sample offset once the cursor grows large. Subtracting the
        // same integer from `cursor`, `next_symbol` and `next_mid` leaves every
        // comparison and interpolation fraction unchanged -- they depend only
        // on the differences -- while keeping the values small and precise.
        //
        // The threshold (2^23) sits above the sample count any single ~90 s
        // 50 kHz recording reaches, so those runs never rebase and decode
        // bit-for-bit as before; only longer/higher-rate streams (e.g. the
        // 250 kHz weak pass) cross it, and they rebase well before the 2^24
        // freeze. `next_mid` is left alone while it holds its consumed-marker
        // `INFINITY` sentinel.
        const REBASE_THRESHOLD: f32 = 8_388_608.0; // 2^23
        if self.cursor >= REBASE_THRESHOLD {
            let shift = self.cursor - 1.0;
            self.cursor -= shift;
            self.next_symbol -= shift;
            if self.next_mid.is_finite() {
                self.next_mid -= shift;
            }
        }

        emitted
    }
}

/// Symmetric Gaussian FIR shaped by a BT product, span 4 symbols.
#[derive(Debug, Clone)]
pub(crate) struct GaussianFir {
    taps: Vec<f32>,
    history: Vec<f32>,
    head: usize,
}

impl GaussianFir {
    pub(crate) fn new(bt: f32, samples_per_symbol: f32) -> Self {
        let taps = gaussian_taps(bt, samples_per_symbol, 4);
        let history = vec![0.0; taps.len()];
        Self {
            taps,
            history,
            head: 0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> f32 {
        if self.taps.is_empty() {
            return sample;
        }
        self.history[self.head] = sample;
        let len = self.history.len();
        let mut acc = 0.0f32;
        for (offset, tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            acc += self.history[index] * tap;
        }
        self.head = (self.head + 1) % len;
        acc
    }
}

/// Length-N rectangular FIR with unit-sum normalization, evaluated as a
/// running sum: O(1) per sample regardless of length. Used as the matched
/// filter for plain (rectangular-pulse) FSK and as the moving-average stage
/// of a long-window DC blocker. Matches gr-satellites' boxcar
/// `np.ones(sqfilter_len) / sqfilter_len` in
/// `python/components/demodulators/fsk_demodulator.py`.
#[derive(Debug, Clone)]
pub(crate) struct BoxcarFilter {
    history: Vec<f32>,
    head: usize,
    sum: f32,
    inv_len: f32,
}

impl BoxcarFilter {
    /// Build a boxcar of explicit length. Length is clamped to at least 1.
    pub(crate) fn new(length: usize) -> Self {
        let len = length.max(1);
        Self {
            history: vec![0.0; len],
            head: 0,
            sum: 0.0,
            inv_len: 1.0 / len as f32,
        }
    }

    /// Build the symbol matched filter for a rectangular-pulse signal:
    /// length equals `samples_per_symbol` rounded to the nearest integer.
    pub(crate) fn matched(samples_per_symbol: f32) -> Self {
        let len = samples_per_symbol.round().max(1.0) as usize;
        Self::new(len)
    }

    pub(crate) fn push(&mut self, sample: f32) -> f32 {
        let oldest = self.history[self.head];
        self.sum += sample - oldest;
        self.history[self.head] = sample;
        self.head += 1;
        if self.head >= self.history.len() {
            self.head = 0;
        }
        self.sum * self.inv_len
    }
}

pub(crate) fn gaussian_taps(bt: f32, samples_per_symbol: f32, span_symbols: usize) -> Vec<f32> {
    let half = (span_symbols as f32 * samples_per_symbol * 0.5).ceil() as i32;
    let len = (2 * half + 1) as usize;
    let alpha = (std::f32::consts::PI * bt) / (samples_per_symbol * (2.0_f32.ln() / 2.0).sqrt());
    let mut taps = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for n in -half..=half {
        let x = alpha * n as f32;
        let value = (-x * x).exp();
        taps.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

/// Real-valued FIR with a circular history buffer. Used as the anti-alias
/// low-pass before integer decimation of the post-discriminator stream.
#[derive(Debug, Clone)]
pub(crate) struct RealFir {
    taps: Vec<f32>,
    history: Vec<f32>,
    head: usize,
}

impl RealFir {
    /// Build a Hamming-windowed sinc low-pass with cutoff at the post-decimation
    /// Nyquist frequency, sized for the given decimation factor.
    pub(crate) fn lowpass(decimation: u32) -> Self {
        let decim = decimation.max(1) as f32;
        let cutoff = 0.45f32 / decim;
        let span = (8.0 * decim).ceil() as usize;
        let len = span | 1;
        let taps = sinc_lowpass_taps(cutoff, len);
        let history = vec![0.0; len];
        Self {
            taps,
            history,
            head: 0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> f32 {
        if self.taps.is_empty() {
            return sample;
        }
        self.history[self.head] = sample;
        let len = self.history.len();
        let mut acc = 0.0f32;
        for (offset, tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            acc += self.history[index] * tap;
        }
        self.head = (self.head + 1) % len;
        acc
    }
}

/// Complex-valued FIR built from two `RealFir` channels. Used as the
/// anti-alias low-pass on the carrier-corrected IQ stream before integer
/// decimation feeds the FM discriminator.
#[derive(Debug, Clone)]
pub(crate) struct ComplexFir {
    i: RealFir,
    q: RealFir,
}

impl ComplexFir {
    /// Build a Hamming-windowed sinc low-pass with cutoff at the
    /// post-decimation Nyquist frequency, sized for the given decimation
    /// factor. The same real taps are applied to I and Q independently —
    /// this is the standard linear-phase complex baseband anti-alias.
    pub(crate) fn lowpass(decimation: u32) -> Self {
        Self {
            i: RealFir::lowpass(decimation),
            q: RealFir::lowpass(decimation),
        }
    }

    pub(crate) fn push(&mut self, sample: IqSample) -> IqSample {
        IqSample {
            i: self.i.push(sample.i),
            q: self.q.push(sample.q),
        }
    }
}

/// Hamming-windowed sinc low-pass with normalized cutoff `cutoff` cycles per
/// sample (must be in `(0, 0.5)`). `len` is rounded up to the nearest odd
/// number internally if needed.
pub fn sinc_lowpass_taps(cutoff: f32, len: usize) -> Vec<f32> {
    let len = if len % 2 == 0 { len + 1 } else { len };
    let half = (len / 2) as i32;
    let mut taps = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for n in -half..=half {
        let x = n as f32;
        let sinc = if n == 0 {
            2.0 * cutoff
        } else {
            let arg = std::f32::consts::TAU * cutoff * x;
            arg.sin() / (std::f32::consts::PI * x)
        };
        let window =
            0.54 - 0.46 * (std::f32::consts::TAU * (n + half) as f32 / (len - 1) as f32).cos();
        let value = sinc * window;
        taps.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn synthesize_fsk(bits: &[u8], sample_rate: u32, baudrate: u32) -> Vec<IqSample> {
        let samples_per_symbol = sample_rate / baudrate;
        let deviation_hz = baudrate as f32;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(bits.len() * samples_per_symbol as usize);

        for &bit in bits {
            let freq = if bit & 1 == 1 {
                deviation_hz
            } else {
                -deviation_hz
            };
            let increment = TAU * freq / sample_rate as f32;
            for _ in 0..samples_per_symbol {
                out.push(IqSample {
                    i: phase.cos(),
                    q: phase.sin(),
                });
                phase += increment;
            }
        }

        out
    }

    /// Synthesize a GMSK waveform by convolving an NRZ symbol stream with a
    /// Gaussian frequency pulse and integrating to phase. The pulse is
    /// normalized so the per-symbol phase change is exactly `pi *
    /// modulation_index` for a +1 NRZ symbol.
    fn synthesize_gmsk(
        bits: &[u8],
        sample_rate: u32,
        baudrate: u32,
        bt: f32,
        modulation_index: f32,
    ) -> Vec<IqSample> {
        let sps_int = (sample_rate / baudrate) as usize;
        let sps = sps_int as f32;
        let span_symbols = 4usize;
        let pulse_len = sps_int * span_symbols;
        let centre = (pulse_len as f32 - 1.0) / 2.0;
        let alpha = (std::f32::consts::PI * bt) / (sps * (2.0_f32.ln() / 2.0).sqrt());
        let mut pulse = vec![0.0f32; pulse_len];
        let mut sum = 0.0f32;
        for (n, tap) in pulse.iter_mut().enumerate() {
            let x = alpha * (n as f32 - centre);
            *tap = (-x * x).exp();
            sum += *tap;
        }
        // Normalize so each symbol slice integrates to 1: total area =
        // span_symbols, so per-symbol-mass = 1.
        if sum > 0.0 {
            let scale = span_symbols as f32 / sum;
            for tap in &mut pulse {
                *tap *= scale;
            }
        }

        let pad = span_symbols;
        let total_symbols = bits.len() + 2 * pad;
        let total_samples = total_symbols * sps_int;
        let mut nrz = vec![0.0f32; total_symbols];
        for (i, &bit) in bits.iter().enumerate() {
            nrz[pad + i] = if bit & 1 == 1 { 1.0 } else { -1.0 };
        }

        let mut freq = vec![0.0f32; total_samples];
        for (sym_index, &symbol) in nrz.iter().enumerate() {
            if symbol == 0.0 {
                continue;
            }
            let start = sym_index * sps_int;
            for (offset, &tap) in pulse.iter().enumerate() {
                let index = start + offset;
                if index < total_samples {
                    freq[index] += symbol * tap;
                }
            }
        }

        // Per-sample phase increment: pi * h * freq[n] / sps. With freq
        // integrating to 1 per symbol slice and sps samples per symbol,
        // the per-symbol phase change is pi * h.
        let phase_step = std::f32::consts::PI * modulation_index / sps;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(total_samples);
        for &f in &freq {
            phase += f * phase_step;
            out.push(IqSample {
                i: phase.cos(),
                q: phase.sin(),
            });
        }
        out
    }

    /// Search for the best alignment between expected and recovered
    /// streams within a small window to absorb pulse ring-up, LPF group
    /// delay, and sub-sample phase offsets. Returns `(best_matches,
    /// best_offset)`.
    fn best_alignment(expected: &[u8], recovered: &[u8], max_offset: usize) -> (usize, usize) {
        let mut best_matches = 0usize;
        let mut best_offset = 0usize;
        for offset in 0..=max_offset {
            if recovered.len() < offset + expected.len() {
                break;
            }
            let matches = expected
                .iter()
                .zip(&recovered[offset..offset + expected.len()])
                .filter(|(left, right)| left == right)
                .count();
            if matches > best_matches {
                best_matches = matches;
                best_offset = offset;
            }
        }
        (best_matches, best_offset)
    }

    #[test]
    fn recovers_binary_fsk_symbols() {
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 17) & 1) as u8
            })
            .collect();
        let samples = synthesize_fsk(&bits, 48_000, 1_200);
        let config = CpmConfig::new(48_000, 1_200, CpmMode::Fsk);
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        // Compare a leading window short enough that recovered.len() is
        // guaranteed to cover it after LPF startup + decimation rounding.
        let window = 32usize;
        let (best_matches, best_offset) = best_alignment(&bits[..window], &recovered, 8);
        assert!(
            best_matches * 100 >= window * 90,
            "FSK match rate too low: {best_matches}/{window} at offset {best_offset}"
        );
    }

    #[test]
    fn recovers_binary_fsk_symbols_with_swapped_iq() {
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 17) & 1) as u8
            })
            .collect();
        let mut samples = synthesize_fsk(&bits, 48_000, 1_200);
        for sample in &mut samples {
            std::mem::swap(&mut sample.i, &mut sample.q);
        }
        let mut config = CpmConfig::new(48_000, 1_200, CpmMode::Fsk);
        config.swap_iq = true;
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        let window = 32usize;
        let (best_matches, best_offset) = best_alignment(&bits[..window], &recovered, 8);
        assert!(
            best_matches * 100 >= window * 90,
            "FSK match rate too low: {best_matches}/{window} at offset {best_offset}"
        );
    }

    #[test]
    fn recovers_gmsk_symbols() {
        // Use a longer pseudo-random bit stream so steady-state recovery
        // dominates the assertion.
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((mixed >> 16) & 1) as u8
            })
            .collect();
        let samples = synthesize_gmsk(&bits, 48_000, 9_600, 0.5, 0.5);
        let config = CpmConfig::new(48_000, 9_600, CpmMode::Gmsk);
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        // Search for the best alignment between expected and recovered
        // streams within a small window to absorb pulse ring-up and
        // sub-sample phase offsets.
        let mut best_matches = 0usize;
        let mut best_offset = 0usize;
        for offset in 0..=12 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(left, right)| left == right)
                .count();
            if matches > best_matches {
                best_matches = matches;
                best_offset = offset;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 90,
            "GMSK match rate too low: {best_matches}/{total} at offset {best_offset}"
        );
    }

    #[test]
    fn rejects_invalid_baudrate() {
        let config = CpmConfig::new(48_000, 0, CpmMode::Gmsk);
        let err = match CpmDemodulator::new(config) {
            Ok(_) => panic!("invalid config should fail"),
            Err(err) => err,
        };

        assert!(matches!(err, DecodeError::InvalidEncoding(_)));
    }

    #[test]
    fn gaussian_taps_normalise_to_unit_dc_gain() {
        let taps = gaussian_taps(0.5, 5.0, 4);
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }

    #[test]
    fn boxcar_filter_averages_to_unit_dc_gain() {
        let mut filter = BoxcarFilter::new(8);
        // Once the buffer is fully filled with the same value, the output
        // must equal that value (unit DC gain).
        for _ in 0..16 {
            let _ = filter.push(0.75);
        }
        let out = filter.push(0.75);
        assert!((out - 0.75).abs() < 1e-6, "DC gain not unity: {out}");
    }

    #[test]
    fn boxcar_filter_matches_rectangular_pulse_peak() {
        // 4 samples of -1 then 4 samples of +1 with a length-4 boxcar
        // produces +1 at the symbol centre after the transition completes.
        let mut filter = BoxcarFilter::new(4);
        for _ in 0..4 {
            let _ = filter.push(-1.0);
        }
        let mut last = 0.0f32;
        for _ in 0..4 {
            last = filter.push(1.0);
        }
        assert!((last - 1.0).abs() < 1e-6, "post-transition peak: {last}");
    }

    #[test]
    fn gardner_locks_and_recovers_fractional_sps() {
        // Clean BPSK NRZ at a fractional 5.2 samples/symbol (the ASRTU-1
        // 50 kHz / 9600 baud operating point), boxcar-matched then tracked.
        // A correct Gardner loop nulls the fractional timing offset and
        // recovers every settled-tail bit exactly. This is the isolated
        // regression guard for the timing-error detector: the tracker was
        // previously dead code with a non-standard TED and did not lock.
        let sps = 5.2f32;
        let bits: Vec<u8> = (0..300u32).map(|i| u8::from((i * 11 + 3) % 5 < 2)).collect();
        // Upsample with a nearest-sample hold at the fractional period.
        let mut stream: Vec<f32> = Vec::new();
        let total = bits.len() as f32 * sps;
        let mut t = 0.0f32;
        let mut n = 0usize;
        while t < total {
            let sym = bits[n.min(bits.len() - 1)];
            stream.push(if sym == 1 { 1.0 } else { -1.0 });
            t += 1.0;
            n = (t / sps) as usize;
        }
        let mut boxcar = BoxcarFilter::matched(sps);
        let mut tracker = GardnerTracker::new(sps);
        let mut out: Vec<u8> = Vec::new();
        for sample in stream {
            let filtered = boxcar.push(sample);
            if let Some(value) = tracker.push(filtered) {
                out.push(u8::from(value >= 0.0));
            }
        }
        // Align the settled tail over a small warm-up window, tolerating the
        // BPSK global-inversion ambiguity.
        let tail = 100usize;
        let expected = &bits[bits.len() - tail..];
        let mut best = usize::MAX;
        for shift in 0..8 {
            if out.len() < tail + shift {
                continue;
            }
            let recovered = &out[out.len() - tail - shift..out.len() - shift];
            let errors = recovered.iter().zip(expected).filter(|(a, b)| a != b).count();
            best = best.min(errors.min(tail - errors));
        }
        assert_eq!(best, 0, "gardner did not lock: tail errors {best}");
    }

    #[test]
    fn gardner_survives_past_the_f32_freeze_point() {
        // Regression guard for the f32 cursor freeze. Past 2^24 input samples
        // the old f32 schedule stopped advancing (`cursor += 1.0` became a
        // no-op) and the loop emitted no more symbols, silently truncating any
        // stream longer than ~67 s at 250 kHz. The rebase must keep symbols
        // coming at the steady ~1/sps rate all the way through a stream that
        // runs past 2^24. The emission schedule is signal-independent, so a
        // cheap square wave is enough.
        let sps = 5.0f32;
        let mut tracker = GardnerTracker::new(sps);
        let total: u64 = (1u64 << 24) + 500_000;
        let tail_start = total - 200_000;
        let mut emitted_total: u64 = 0;
        let mut emitted_tail: u64 = 0;
        for n in 0..total {
            let sample = if (n / 3) % 2 == 0 { 1.0 } else { -1.0 };
            if tracker.push(sample).is_some() {
                emitted_total += 1;
                if n >= tail_start {
                    emitted_tail += 1;
                }
            }
        }
        // Without the fix, emission stops at 2^24: the total falls ~20 % short
        // and the tail window (entirely past 2^24) yields nothing.
        assert!(
            emitted_total > (total as f64 / f64::from(sps) * 0.9) as u64,
            "too few symbols overall: {emitted_total}"
        );
        assert!(
            emitted_tail > 30_000,
            "timing loop froze past 2^24: only {emitted_tail} symbols in the tail window"
        );
    }
}
