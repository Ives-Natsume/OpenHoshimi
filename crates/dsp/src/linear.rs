//! Generic linear IQ demodulator.
//!
//! This module provides a hard-decision demodulator for BPSK/DBPSK/QPSK/
//! OQPSK streams. A fixed frequency offset can be corrected before slicing,
//! and a small decision-directed carrier tracker keeps the phase from
//! drifting away on real recordings.

use std::f32::consts::TAU;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use openhoshimi_core::{DecodeError, Demodulator, IqSample};

use crate::cpm::{BoxcarFilter, GardnerTracker};
use crate::fll::BandEdgeFll;

/// Process-wide debug sink for post-Costas symbol decisions. Each line is
/// `i,q\n` of the IQ value handed to the slicer. Enabled by
/// [`open_symbol_dump`]; intended for offline SNR / lock analysis on real
/// recordings.
static SYMBOL_DUMP: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Open (or replace) the process-wide symbol dump file used by
/// [`LinearDemodulator`]. Returns the IO error from [`File::create`] if the
/// path cannot be opened. Pass an empty path to disable.
pub fn open_symbol_dump(path: &Path) -> io::Result<()> {
    let mut guard = SYMBOL_DUMP
        .lock()
        .map_err(|_| io::Error::other("symbol dump mutex poisoned"))?;
    if path.as_os_str().is_empty() {
        *guard = None;
        return Ok(());
    }
    let file = File::create(path)?;
    *guard = Some(BufWriter::new(file));
    Ok(())
}

fn write_symbol_dump(sample: IqSample) {
    let Ok(mut guard) = SYMBOL_DUMP.lock() else {
        return;
    };
    if let Some(writer) = guard.as_mut() {
        let _ = writeln!(writer, "{:.6},{:.6}", sample.i, sample.q);
    }
}

/// Linear modulation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMode {
    /// Binary phase-shift keying.
    Bpsk,
    /// Differential binary phase-shift keying.
    Dbpsk,
    /// Quadrature phase-shift keying.
    Qpsk,
    /// Offset quadrature phase-shift keying.
    Oqpsk,
}

/// Configuration for [`LinearDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    /// IQ sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// Linear modulation mode.
    pub mode: LinearMode,
    /// Frequency offset correction in Hz, applied before symbol slicing.
    pub frequency_offset_hz: f32,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Swap I and Q before carrier correction.
    pub swap_iq: bool,
    /// Closed-loop carrier tracker bandwidth, normalized to the symbol rate.
    /// `0.0` disables the tracker and keeps a static frequency correction.
    pub carrier_loop_bandwidth: f32,
    /// Frequency-locked loop bandwidth normalized to the symbol rate. When
    /// `> 0.0` a Kay-style cross-product discriminator on squared symbols
    /// drives the NCO integrator, providing wide pull-in over Doppler
    /// before the phase-locked Costas tracker takes over. `0.0` keeps only
    /// the Costas loop active.
    pub frequency_loop_bandwidth: f32,
    /// Maximum absolute NCO offset from the static frequency correction, in
    /// Hz. `0.0` selects a conservative default of one symbol rate. Increase
    /// for LEO passes whose Doppler swing exceeds the default.
    pub nco_max_offset_hz: f32,
    /// Root-raised-cosine matched filter roll-off factor in the range
    /// `(0.0, 1.0]`. `0.0` disables matched filtering and falls back to the
    /// boxcar integrate-and-dump symbol filter, preserving behaviour of
    /// downlinks that do not assume RRC pulse shaping.
    pub matched_filter_rolloff: f32,
    /// Matched filter span in symbol periods. Used only when
    /// [`Self::matched_filter_rolloff`] is greater than zero. `0` selects a
    /// reasonable default span (six symbols).
    pub matched_filter_span_symbols: usize,
    /// Band-edge FLL loop bandwidth, normalized to the symbol rate. When
    /// `> 0.0` a GNU-Radio-style `digital_fll_band_edge_cc` runs per sample
    /// ahead of the carrier correction and Costas loop, giving a wide
    /// (`±~2*baudrate`) frequency acquisition that the phase loops cannot.
    /// `0.0` disables it. It cascades with, rather than replaces, the Kay
    /// squared-symbol FLL ([`Self::frequency_loop_bandwidth`]) and the Costas
    /// loop: the band-edge FLL takes the coarse wide lock on the raw samples,
    /// then the Costas/Kay loops clean up residual phase and Doppler.
    pub fll_band_edge_bw: f32,
    /// Band-edge FLL RRC roll-off used to design the band-edge filters. Only
    /// consulted when [`Self::fll_band_edge_bw`] is greater than zero. This is
    /// independent of [`Self::matched_filter_rolloff`]'s use for the RRC
    /// decision-path matched filter; if this is `0.0` while the FLL is on, a
    /// default of `0.35` is used.
    pub fll_band_edge_rolloff: f32,
}

impl LinearConfig {
    /// Create a configuration with mode-specific defaults.
    pub fn new(sample_rate: u32, baudrate: u32, mode: LinearMode) -> Self {
        Self {
            sample_rate,
            baudrate,
            mode,
            frequency_offset_hz: 0.0,
            differential: mode == LinearMode::Dbpsk,
            invert: false,
            swap_iq: false,
            carrier_loop_bandwidth: 0.0,
            frequency_loop_bandwidth: 0.0,
            nco_max_offset_hz: 0.0,
            matched_filter_rolloff: 0.0,
            matched_filter_span_symbols: 0,
            fll_band_edge_bw: 0.0,
            fll_band_edge_rolloff: 0.0,
        }
    }
}

/// Hard-decision IQ demodulator for BPSK/DBPSK/QPSK/OQPSK signals.
#[derive(Debug, Clone)]
pub struct LinearDemodulator {
    config: LinearConfig,
    carrier_phase: f32,
    carrier_increment: f32,
    nco_fll_offset: f32,
    /// Costas phase accumulator, applied AFTER symbol-timing recovery, once
    /// per recovered symbol. The pre-timing `correct_frequency` derotates only
    /// by the static carrier increment and the Kay FLL offset, leaving the
    /// constellation still spinning at the residual phase; the Gardner TED is
    /// rotation invariant so timing locks on that spinning stream. This
    /// accumulator then pulls each recovered symbol back onto the in-phase
    /// axis, matching the official topology (timing first, Costas second) used
    /// by both gr-satellites' GRC flow and the in-house soundmodem decoder.
    costas_phase: f32,
    /// Costas NCO frequency (integral term), advanced once per symbol.
    costas_freq: f32,
    nco_limit_radians: f32,
    loop_alpha: f32,
    loop_beta: f32,
    freq_loop_alpha: f32,
    fll_prev_squared: Option<IqSample>,
    agc_gain: f32,
    agc_alpha: f32,
    dc_i: f32,
    dc_q: f32,
    dc_alpha: f32,
    /// Boxcar (rectangular) matched filter, length = round(samples/symbol),
    /// applied per axis before the Gardner tracker when no RRC matched
    /// filter is configured. This mirrors gr-satellites'
    /// `fsk_demodulator.py`, which runs `np.ones(sps)/sps` ahead of
    /// `symbol_sync_ff`. It restores the integrate-and-dump processing gain
    /// (~sqrt(sps)) that the former per-symbol accumulator provided, so the
    /// Gardner loop interpolates a smoothed, matched stream rather than a
    /// single noisy instantaneous sample. `None` when an RRC matched filter
    /// is configured, since that already band-limits and matches the pulse.
    pre_boxcar: Option<(BoxcarFilter, BoxcarFilter)>,
    /// Closed-loop symbol-timing recovery. The Gardner TED and PI loop are
    /// driven by the in-phase (I) channel — where the information sits after
    /// Costas lock for BPSK/DBPSK — and the quadrature (Q) channel is
    /// interpolated at the identical recovered instant via
    /// [`GardnerTracker::push_dual`], so both axes of a complex symbol are
    /// decided together for the QPSK/OQPSK paths. This replaces the former
    /// integrate-and-dump / `sample_phase` / `peak_symbol_sample` mechanism,
    /// which could not null a sub-sample static timing offset at the
    /// fractional 5.2 samples/symbol of the 9k6 ASRTU-1 downlink.
    timing: GardnerTracker,
    previous_binary_symbol: Option<IqSample>,
    matched_taps: Option<Vec<f32>>,
    matched_buffer_i: Vec<f32>,
    matched_buffer_q: Vec<f32>,
    matched_index: usize,
    /// Soft-decision (i8) value for each bit emitted by the most recent
    /// `push_samples` call, sign-aligned with the emitted hard bit (bit `0`
    /// -> positive, bit `1` -> negative, magnitude = reliability). Populated
    /// for binary (BPSK/DBPSK) modes so the streaming Viterbi can run soft.
    last_soft_i8: Vec<i8>,
    /// Complex post-timing decision symbol `(i, q)` for each bit emitted by
    /// the most recent `push_samples` call, aligned 1:1 with `last_soft_i8`.
    /// Retains the quadrature component (dropped by the real-valued soft) so an
    /// offline block decoder can re-estimate carrier phase over a whole frame.
    /// Populated for binary (BPSK/DBPSK) modes only.
    last_symbol_iq: Vec<(f32, f32)>,
    /// Band-edge FLL for wide frequency acquisition, applied per sample after
    /// AGC/DC-block and before the carrier correction / Costas loop. `None`
    /// when `fll_band_edge_bw <= 0.0`.
    band_edge_fll: Option<BandEdgeFll>,
}

impl LinearDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: LinearConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        let samples_per_symbol = config.sample_rate as f32 / config.baudrate as f32;
        let (loop_alpha, loop_beta) = loop_filter_gains(config.carrier_loop_bandwidth);
        // Acquisition cascade: the band-edge FLL (below) derotates the raw
        // samples for a wide coarse frequency lock, THEN correct_frequency runs
        // the Costas phase loop and the Kay squared-symbol FLL on the result
        // for fine residual/Doppler tracking. The two FLLs drive independent
        // NCOs in series (band-edge on the raw stream, Kay inside
        // correct_frequency), so they do not fight -- and both are needed:
        // dropping the Kay FLL here collapses ASRTU-1 decode to zero because
        // the band-edge lock alone leaves too much residual for the Gardner
        // timing loop, whereas keeping both recovers 52/52 contiguous frames.
        let band_edge_active = config.fll_band_edge_bw > 0.0;
        let freq_loop_alpha = if config.frequency_loop_bandwidth > 0.0 {
            config.frequency_loop_bandwidth
        } else {
            0.0
        };
        let band_edge_fll = if band_edge_active {
            let rolloff = if config.fll_band_edge_rolloff > 0.0 {
                config.fll_band_edge_rolloff
            } else {
                0.35
            };
            // Span ~20 symbols like the reference, capped so the per-sample
            // FIRs stay affordable at high sample rates.
            let fsize = ((20.0 * samples_per_symbol).round() as usize).clamp(8, 1024);
            Some(BandEdgeFll::new(
                samples_per_symbol,
                rolloff,
                fsize,
                config.fll_band_edge_bw,
            ))
        } else {
            None
        };
        let nco_limit_radians = if config.nco_max_offset_hz > 0.0 {
            TAU * config.nco_max_offset_hz / config.sample_rate as f32
        } else {
            TAU * config.baudrate as f32 / config.sample_rate as f32
        };
        // The RRC decision-path matched filter and the band-edge FLL's
        // band-edge design both read a roll-off value, but from DIFFERENT
        // config fields so callers can feed the FLL a roll-off without turning
        // on the RRC decision path (prior A/B showed the RRC midpoint-decision
        // path regresses ASRTU-1). The RRC matched filter is driven solely by
        // `matched_filter_rolloff`; the FLL roll-off is `fll_band_edge_rolloff`
        // (see above). When the band-edge FLL is active we additionally force
        // the RRC decision path off as a belt-and-braces guard, since the
        // pipeline feeds `matched_filter_rolloff` into the FLL roll-off.
        let matched_taps = if config.matched_filter_rolloff > 0.0 && !band_edge_active {
            let span = if config.matched_filter_span_symbols == 0 {
                6
            } else {
                config.matched_filter_span_symbols
            };
            Some(rrc_taps(
                samples_per_symbol,
                config.matched_filter_rolloff,
                span,
            ))
        } else {
            None
        };
        let buffer_len = matched_taps.as_ref().map_or(0, Vec::len);
        Ok(Self {
            carrier_increment: -TAU * config.frequency_offset_hz / config.sample_rate as f32,
            costas_phase: 0.0,
            costas_freq: 0.0,
            nco_fll_offset: 0.0,
            nco_limit_radians,
            loop_alpha,
            loop_beta,
            freq_loop_alpha,
            fll_prev_squared: None,
            agc_gain: 1.0,
            agc_alpha: 2.0e-2 / samples_per_symbol,
            dc_i: 0.0,
            dc_q: 0.0,
            dc_alpha: 2.0e-3 / samples_per_symbol,
            pre_boxcar: if matched_taps.is_some() {
                None
            } else {
                Some((
                    BoxcarFilter::matched(samples_per_symbol),
                    BoxcarFilter::matched(samples_per_symbol),
                ))
            },
            timing: GardnerTracker::new(samples_per_symbol),
            config,
            carrier_phase: 0.0,
            previous_binary_symbol: None,
            matched_taps,
            matched_buffer_i: vec![0.0; buffer_len],
            matched_buffer_q: vec![0.0; buffer_len],
            matched_index: 0,
            last_soft_i8: Vec::new(),
            last_symbol_iq: Vec::new(),
            band_edge_fll,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> LinearConfig {
        self.config
    }

    fn slice_symbol(&mut self, current: IqSample) -> Vec<u8> {
        match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => vec![self.slice_binary(current)],
            LinearMode::Qpsk | LinearMode::Oqpsk => self.slice_quadrature(current),
        }
    }

    fn slice_symbol_soft(&mut self, current: IqSample) -> Vec<i8> {
        match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => vec![self.slice_binary_soft(current)],
            LinearMode::Qpsk | LinearMode::Oqpsk => self.slice_quadrature_soft(current),
        }
    }

    /// Slice one binary symbol into its hard bit AND a sign-aligned soft
    /// value in a single pass, so any differential state advances exactly
    /// once. The hard bit is identical to [`slice_binary`](Self::slice_binary);
    /// the soft sign matches it under the streaming-Viterbi convention
    /// (bit `0` -> positive, bit `1` -> negative) with magnitude carrying
    /// per-symbol reliability.
    fn slice_binary_paired(&mut self, current: IqSample) -> (u8, i8) {
        let (mut bit, mut conf) = if self.config.differential {
            let (bit, cos) = match self.previous_binary_symbol {
                Some(previous) => {
                    let dot = previous.i.mul_add(current.i, previous.q * current.q);
                    let mag_prev = previous
                        .i
                        .mul_add(previous.i, previous.q * previous.q)
                        .sqrt();
                    let mag_curr = current.i.mul_add(current.i, current.q * current.q).sqrt();
                    let denom = mag_prev * mag_curr;
                    let cos = if denom > f32::EPSILON {
                        (dot / denom).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    // bit = (dot < 0); cos has the same sign as dot, so cos is
                    // already in the Viterbi convention (bit 0 -> positive).
                    (u8::from(dot < 0.0), cos)
                }
                None => (0u8, 0.0f32),
            };
            self.previous_binary_symbol = Some(current);
            (bit, cos)
        } else {
            // Hard bit matches `slice_binary`: i >= 0 -> 1. Encoding bit 0 as
            // positive then requires the negated in-phase projection.
            //
            // normalize_in_phase() divides out the symbol magnitude, so it
            // reports cos(theta) alone -- a near-noise symbol whose tiny i,q
            // happen to align on-axis still claims full confidence, throwing
            // away the fade/SNR information the amplitude carries. Re-attach an
            // amplitude reliability bounded to 1: a confident-amplitude symbol
            // (|s| >= ref) keeps exactly its previous soft value (the soft
            // Viterbi metric is calibrated to +/-SOFT_CENTER and must never be
            // scaled up), while a faded symbol is deweighted toward zero.
            let cos = -normalize_in_phase(current);
            let amp = current.i.hypot(current.q);
            let reliability = (amp / SOFT_DECISION_AMP_REF).min(1.0);
            (u8::from(current.i >= 0.0), cos * reliability)
        };

        if self.config.invert {
            bit ^= 1;
            conf = -conf;
        }
        (bit, soft_to_i8(conf))
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

    fn slice_binary(&mut self, current: IqSample) -> u8 {
        if self.config.differential {
            return self.slice_differential_binary(current);
        }

        let mut symbol = u8::from(current.i >= 0.0);
        if self.config.invert {
            symbol ^= 1;
        }
        symbol
    }

    fn slice_binary_soft(&mut self, current: IqSample) -> i8 {
        let value = if self.config.differential {
            self.slice_differential_binary_soft(current)
        } else {
            normalize_in_phase(current)
        };
        let value = if self.config.invert { -value } else { value };
        soft_to_i8(value)
    }

    fn slice_differential_binary(&mut self, current: IqSample) -> u8 {
        let mut bit = match self.previous_binary_symbol {
            Some(previous) => {
                let dot = previous.i.mul_add(current.i, previous.q * current.q);
                u8::from(dot < 0.0)
            }
            None => 0,
        };
        self.previous_binary_symbol = Some(current);

        if self.config.invert {
            bit ^= 1;
        }
        bit
    }

    fn slice_differential_binary_soft(&mut self, current: IqSample) -> f32 {
        let value = match self.previous_binary_symbol {
            Some(previous) => {
                let dot = previous.i.mul_add(current.i, previous.q * current.q);
                let mag_prev = previous
                    .i
                    .mul_add(previous.i, previous.q * previous.q)
                    .sqrt();
                let mag_curr = current.i.mul_add(current.i, current.q * current.q).sqrt();
                let denom = mag_prev * mag_curr;
                if denom > f32::EPSILON {
                    (dot / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        self.previous_binary_symbol = Some(current);
        value
    }

    fn slice_quadrature(&self, current: IqSample) -> Vec<u8> {
        let mut i_bit = u8::from(current.i >= 0.0);
        let mut q_bit = u8::from(current.q >= 0.0);
        if self.config.invert {
            i_bit ^= 1;
            q_bit ^= 1;
        }
        vec![i_bit, q_bit]
    }

    fn slice_quadrature_soft(&self, current: IqSample) -> Vec<i8> {
        let mag = current.i.mul_add(current.i, current.q * current.q).sqrt();
        let (mut i_value, mut q_value) = if mag > f32::EPSILON {
            (current.i / mag, current.q / mag)
        } else {
            (0.0, 0.0)
        };
        if self.config.invert {
            i_value = -i_value;
            q_value = -q_value;
        }
        vec![soft_to_i8(i_value), soft_to_i8(q_value)]
    }

    fn apply_dc_block_and_agc(&mut self, sample: IqSample) -> IqSample {
        self.dc_i += self.dc_alpha * (sample.i - self.dc_i);
        self.dc_q += self.dc_alpha * (sample.q - self.dc_q);

        let corrected = IqSample {
            i: sample.i - self.dc_i,
            q: sample.q - self.dc_q,
        };

        let power = corrected.i.mul_add(corrected.i, corrected.q * corrected.q);
        if power > 0.0 && power.is_finite() {
            let target_gain = 1.0 / power.sqrt();
            self.agc_gain += self.agc_alpha * (target_gain - self.agc_gain);
            self.agc_gain = self.agc_gain.clamp(0.1, 64.0);
        }

        IqSample {
            i: corrected.i * self.agc_gain,
            q: corrected.q * self.agc_gain,
        }
    }

    /// Apply the band-edge FLL if configured, returning the derotated sample.
    /// A no-op when the FLL is disabled.
    fn apply_band_edge_fll(&mut self, sample: IqSample) -> IqSample {
        match self.band_edge_fll.as_mut() {
            Some(fll) => fll.process(sample),
            None => sample,
        }
    }

    fn correct_frequency(&mut self, sample: IqSample) -> IqSample {
        let phase = self.carrier_phase;
        let sin = phase.sin();
        let cos = phase.cos();
        // Pre-timing frequency correction only: the static carrier increment
        // plus the Kay FLL offset. The Costas PHASE loop no longer contributes
        // here -- it moved after timing recovery (see `costas_derotate`), so
        // the Gardner loop runs on a frequency-corrected but still phase-
        // rotating stream and its rotation-invariant TED locks the clock there.
        self.carrier_phase += self.carrier_increment + self.nco_fll_offset;
        if self.carrier_phase >= TAU || self.carrier_phase <= -TAU {
            self.carrier_phase %= TAU;
        }

        IqSample {
            i: sample.i * cos - sample.q * sin,
            q: sample.i * sin + sample.q * cos,
        }
    }

    /// Kay-style squared-symbol FLL, measured on the PRE-Costas recovered
    /// symbol (the still-rotating constellation straight out of the Gardner
    /// loop). It feeds `nco_fll_offset` back into the per-sample
    /// `correct_frequency` ahead of timing, so the wide Doppler pull-in stays
    /// in front of the symbol clock exactly as before; only the measurement
    /// point moved from post- to pre-Costas, which is fine because the
    /// squaring nonlinearity removes the BPSK phase anyway.
    ///
    /// Dropping this loop collapses ASRTU-1 decode to zero (the band-edge FLL
    /// alone leaves too much residual Doppler for the timing loop), so it is
    /// kept verbatim; only the Costas PHASE loop relocated after timing.
    fn update_fll(&mut self, symbol: IqSample) {
        if self.freq_loop_alpha <= 0.0 {
            return;
        }
        let magnitude_squared = symbol.i.mul_add(symbol.i, symbol.q * symbol.q);
        if magnitude_squared <= f32::EPSILON {
            return;
        }
        let scale = magnitude_squared.sqrt();
        let i = symbol.i / scale;
        let q = symbol.q / scale;

        // imag(s²_n * conj(s²_{n-1})) collapses BPSK 180° flips so the
        // discriminator sees only the residual carrier rotation. Leaky
        // integrator keeps the FLL from sticking at the clamp during fades.
        let s2 = IqSample {
            i: i * i - q * q,
            q: 2.0 * i * q,
        };
        if let Some(prev) = self.fll_prev_squared {
            let mag_curr_sq = s2.i.mul_add(s2.i, s2.q * s2.q);
            let mag_prev_sq = prev.i.mul_add(prev.i, prev.q * prev.q);
            let denom = (mag_curr_sq * mag_prev_sq).sqrt();
            if denom > f32::EPSILON {
                let cross = s2.q * prev.i - s2.i * prev.q;
                let fll_error = (cross / denom).clamp(-1.0, 1.0);
                let leak = self.freq_loop_alpha;
                // Same `+carrier_phase` de-rotation convention: SUBTRACT the
                // discriminator, otherwise the FLL runs to the wrong NCO rail.
                self.nco_fll_offset =
                    (1.0 - leak) * self.nco_fll_offset - self.freq_loop_alpha * fll_error;
                self.nco_fll_offset = self
                    .nco_fll_offset
                    .clamp(-self.nco_limit_radians, self.nco_limit_radians);
            }
        }
        self.fll_prev_squared = Some(s2);
    }

    /// Costas phase loop, run ONCE PER RECOVERED SYMBOL after timing recovery.
    /// Derotates the pre-Costas symbol by the current `costas_phase`, forms the
    /// decision-directed error, advances the PI loop, and returns the
    /// derotated symbol used for slicing. This is the official topology
    /// (timing first, Costas second) and matches the in-house soundmodem
    /// decoder, whose Costas also updates per symbol.
    fn costas_derotate(&mut self, pre: IqSample) -> IqSample {
        if self.loop_alpha == 0.0 && self.loop_beta == 0.0 {
            return pre;
        }

        // Derotate by `+costas_phase`, the same rotation sense as
        // `correct_frequency`, so a symbol sitting at residual angle `+theta`
        // is pulled toward the in-phase axis when the loop drives
        // `costas_phase` toward `-theta`.
        let sin = self.costas_phase.sin();
        let cos = self.costas_phase.cos();
        let current = IqSample {
            i: pre.i * cos - pre.q * sin,
            q: pre.i * sin + pre.q * cos,
        };

        let magnitude_squared = current.i.mul_add(current.i, current.q * current.q);
        if magnitude_squared <= f32::EPSILON {
            return current;
        }
        let scale = magnitude_squared.sqrt();
        let i = current.i / scale;
        let q = current.q / scale;

        let error = match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => i * q,
            LinearMode::Qpsk | LinearMode::Oqpsk => {
                let sign_i = if i >= 0.0 { 1.0 } else { -1.0 };
                let sign_q = if q >= 0.0 { 1.0 } else { -1.0 };
                sign_i * q - sign_q * i
            }
        };
        // Detector `i*q` is positive for a `+theta` residual; both PI terms
        // must SUBTRACT it to pull the constellation back to I (adding it is
        // positive feedback onto the quadrature axis). Negating once flips both
        // consistently.
        let error = -error.clamp(-1.0, 1.0);

        // Per-symbol update: unlike the old per-sample Costas there is no
        // 1/samples_per_symbol scaling on the integral term, because the loop
        // now advances exactly once per symbol -- the rate `loop_beta` and
        // `loop_alpha` were designed for.
        self.costas_freq += self.loop_beta * error;
        self.costas_freq = self
            .costas_freq
            .clamp(-self.nco_limit_radians, self.nco_limit_radians);
        self.costas_phase += self.costas_freq + self.loop_alpha * error;
        if self.costas_phase >= TAU || self.costas_phase <= -TAU {
            self.costas_phase %= TAU;
        }

        current
    }

    fn matched_filter(&mut self, sample: IqSample) -> IqSample {
        let Some(taps) = self.matched_taps.as_ref() else {
            return sample;
        };
        let len = taps.len();
        self.matched_buffer_i[self.matched_index] = sample.i;
        self.matched_buffer_q[self.matched_index] = sample.q;
        self.matched_index = (self.matched_index + 1) % len;

        let mut acc_i = 0.0f32;
        let mut acc_q = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for k in 0..len {
            let buf_index = (self.matched_index + k) % len;
            let coeff = taps[k];
            acc_i += self.matched_buffer_i[buf_index] * coeff;
            acc_q += self.matched_buffer_q[buf_index] * coeff;
        }
        IqSample { i: acc_i, q: acc_q }
    }

    /// Apply the rectangular-pulse boxcar matched filter to both axes when
    /// no RRC filter is configured. Returns the sample unchanged otherwise,
    /// so the RRC path (which already matches the pulse) is not double
    /// filtered. This is the integrate-and-dump matched filter, run
    /// per-sample so the Gardner tracker downstream sees a smoothed stream.
    fn symbol_matched_filter(&mut self, sample: IqSample) -> IqSample {
        match self.pre_boxcar.as_mut() {
            Some((box_i, box_q)) => IqSample {
                i: box_i.push(sample.i),
                q: box_q.push(sample.q),
            },
            None => sample,
        }
    }

    /// Process a block of IQ samples and emit soft-decision symbols.
    ///
    /// Each output element is a signed correlation value where positive
    /// means the channel bit was received as `0` and negative means `1`;
    /// magnitude is the per-bit reliability. The same DC blocker, AGC,
    /// matched filter, carrier tracker and timing recovery used by
    /// [`Demodulator::push_samples`] are applied here, only the slicing
    /// step differs.
    pub fn push_samples_soft(&mut self, samples: &[IqSample]) -> Vec<i8> {
        let mut soft_bits = Vec::new();

        for &sample in samples {
            let sample = self.normalize_sample(sample);
            let sample = self.apply_dc_block_and_agc(sample);
            // Band-edge FLL: wide coarse frequency acquisition ahead of the
            // narrow Costas/Kay phase loops. No-op unless configured.
            let sample = self.apply_band_edge_fll(sample);
            let sample = self.correct_frequency(sample);
            let filtered = self.matched_filter(sample);
            let filtered = self.symbol_matched_filter(filtered);

            // The Gardner loop tracks the symbol clock on I and interpolates
            // Q at the identical recovered instant; a decision fires only when
            // the loop reaches a symbol time, not on a fixed sample count.
            if let Some((i, q)) = self.timing.push_dual(filtered.i, filtered.q) {
                let pre = IqSample { i, q };
                // Kay FLL is measured on the pre-Costas (still-rotating) symbol
                // and feeds `correct_frequency` ahead of timing; the Costas
                // phase loop then pulls this symbol onto the in-phase axis.
                self.update_fll(pre);
                let current = self.costas_derotate(pre);
                write_symbol_dump(current);
                soft_bits.extend_from_slice(&self.slice_symbol_soft(current));
            }
        }

        soft_bits
    }
}

impl Demodulator for LinearDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();
        self.last_soft_i8.clear();
        self.last_symbol_iq.clear();

        for &sample in samples {
            let sample = self.normalize_sample(sample);
            let sample = self.apply_dc_block_and_agc(sample);
            // Band-edge FLL: wide coarse frequency acquisition ahead of the
            // narrow Costas/Kay phase loops. No-op unless configured.
            let sample = self.apply_band_edge_fll(sample);
            let sample = self.correct_frequency(sample);
            let filtered = self.matched_filter(sample);
            let filtered = self.symbol_matched_filter(filtered);

            // The Gardner loop tracks the symbol clock on I and interpolates
            // Q at the identical recovered instant; a decision fires only when
            // the loop reaches a symbol time, not on a fixed sample count.
            if let Some((i, q)) = self.timing.push_dual(filtered.i, filtered.q) {
                let pre = IqSample { i, q };
                // Kay FLL is measured on the pre-Costas (still-rotating) symbol
                // and feeds `correct_frequency` ahead of timing; the Costas
                // phase loop then pulls this symbol onto the in-phase axis.
                self.update_fll(pre);
                let current = self.costas_derotate(pre);
                write_symbol_dump(current);
                if matches!(self.config.mode, LinearMode::Bpsk | LinearMode::Dbpsk) {
                    // Binary modes also produce a sign-aligned soft value so a
                    // downstream streaming Viterbi can decode soft (~2 dB gain).
                    let (bit, soft) = self.slice_binary_paired(current);
                    bits.push(bit);
                    self.last_soft_i8.push(soft);
                    self.last_symbol_iq.push((current.i, current.q));
                } else {
                    bits.extend_from_slice(&self.slice_symbol(current));
                }
            }
        }

        bits
    }

    fn last_soft_symbols(&self) -> &[i8] {
        &self.last_soft_i8
    }

    fn last_symbol_iq(&self) -> &[(f32, f32)] {
        &self.last_symbol_iq
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn baudrate(&self) -> u32 {
        self.config.baudrate
    }
}

fn validate_config(config: LinearConfig) -> Result<(), DecodeError> {
    if config.sample_rate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "linear sample rate must be greater than zero".to_string(),
        ));
    }
    if config.baudrate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "linear baudrate must be greater than zero".to_string(),
        ));
    }
    if !config.frequency_offset_hz.is_finite() {
        return Err(DecodeError::InvalidEncoding(
            "linear frequency offset must be finite".to_string(),
        ));
    }
    if !config.carrier_loop_bandwidth.is_finite() || config.carrier_loop_bandwidth < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear carrier loop bandwidth must be a non-negative finite value".to_string(),
        ));
    }
    if !config.frequency_loop_bandwidth.is_finite() || config.frequency_loop_bandwidth < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear frequency loop bandwidth must be a non-negative finite value".to_string(),
        ));
    }
    if !config.nco_max_offset_hz.is_finite() || config.nco_max_offset_hz < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear NCO max offset must be a non-negative finite value in Hz".to_string(),
        ));
    }
    if !config.matched_filter_rolloff.is_finite() || config.matched_filter_rolloff < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear matched filter rolloff must be a non-negative finite value".to_string(),
        ));
    }
    if config.matched_filter_rolloff > 1.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear matched filter rolloff must be <= 1.0".to_string(),
        ));
    }
    Ok(())
}

fn rrc_taps(samples_per_symbol: f32, rolloff: f32, span_symbols: usize) -> Vec<f32> {
    let length = (samples_per_symbol * span_symbols as f32).round() as usize;
    let length = length.max(1) | 1;
    let center = (length - 1) as f32 / 2.0;
    let mut taps = Vec::with_capacity(length);
    let mut energy = 0.0f32;
    for k in 0..length {
        let t_in_symbols = (k as f32 - center) / samples_per_symbol;
        let value = rrc_impulse(t_in_symbols, rolloff);
        taps.push(value);
        energy += value * value;
    }
    if energy > 0.0 {
        let scale = 1.0 / energy.sqrt();
        for tap in taps.iter_mut() {
            *tap *= scale;
        }
    }
    taps
}

fn rrc_impulse(t: f32, rolloff: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if t.abs() < 1.0e-6 {
        return 1.0 - rolloff + 4.0 * rolloff / pi;
    }
    if rolloff > 0.0 {
        let singular = 1.0 / (4.0 * rolloff);
        if (t.abs() - singular).abs() < 1.0e-4 {
            let term1 = (1.0 + 2.0 / pi) * (pi / (4.0 * rolloff)).sin();
            let term2 = (1.0 - 2.0 / pi) * (pi / (4.0 * rolloff)).cos();
            return rolloff / std::f32::consts::SQRT_2 * (term1 + term2);
        }
    }
    let pi_t = pi * t;
    let num = (pi_t * (1.0 - rolloff)).sin() + 4.0 * rolloff * t * (pi_t * (1.0 + rolloff)).cos();
    let denom = pi_t * (1.0 - (4.0 * rolloff * t).powi(2));
    num / denom
}

fn loop_filter_gains(bandwidth: f32) -> (f32, f32) {
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return (0.0, 0.0);
    }
    let damping = std::f32::consts::FRAC_1_SQRT_2;
    let denom = 1.0 + 2.0 * damping * bandwidth + bandwidth * bandwidth;
    let alpha = 4.0 * damping * bandwidth / denom;
    let beta = 4.0 * bandwidth * bandwidth / denom;
    (alpha, beta)
}

/// Project a complex symbol onto the in-phase axis with magnitude normalization.
///
/// For non-differential BPSK the channel bit is encoded by the sign of `i`,
/// but the magnitude depends on signal strength and integration gain. Returning
/// `i / |sample|` keeps the soft value in `[-1.0, 1.0]` so downstream Viterbi
/// metrics receive comparable confidences across recordings.
fn normalize_in_phase(sample: IqSample) -> f32 {
    let mag = sample.i.mul_add(sample.i, sample.q * sample.q).sqrt();
    if mag > f32::EPSILON {
        (sample.i / mag).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Reference symbol amplitude at which a soft decision is considered fully
/// reliable. The AGC (see `apply_dc_block_and_agc`) drives the sample stream
/// to unit power, so a well-formed symbol lands near |s| = 1; symbols weaker
/// than this are faded/noisy and get their soft value scaled down in
/// proportion to |s| (see `slice_binary_paired`). Chosen at the AGC target so
/// the whole confident cluster (|s| >= 1) keeps its previous soft value
/// unchanged; a sweep on a real pass showed 1.0-1.25 is the knee (1.25 already
/// deweights ~84% of symbols and starts dropping frames).
const SOFT_DECISION_AMP_REF: f32 = 1.0;

/// Map a signed correlation value to a soft-decision symbol.
///
/// Positive `value` means bit `0`, negative means bit `1`; magnitude carries
/// per-bit reliability. Output is clamped to the i8 range so downstream soft
/// Viterbi metrics stay representable.
fn soft_to_i8(value: f32) -> i8 {
    const SOFT_CENTER: f32 = 64.0;
    if !value.is_finite() {
        return 0;
    }
    let scaled = (value * SOFT_CENTER).clamp(i8::MIN as f32 + 1.0, i8::MAX as f32);
    scaled.round() as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_symbols(symbols: &[IqSample], samples_per_symbol: usize) -> Vec<IqSample> {
        let mut out = Vec::with_capacity(symbols.len() * samples_per_symbol);
        for &symbol in symbols {
            for _ in 0..samples_per_symbol {
                out.push(symbol);
            }
        }
        out
    }

    fn add_frequency_offset(
        samples: &[IqSample],
        sample_rate: u32,
        offset_hz: f32,
    ) -> Vec<IqSample> {
        let mut phase = 0.0f32;
        let increment = TAU * offset_hz / sample_rate as f32;
        let mut out = Vec::with_capacity(samples.len());

        for &sample in samples {
            let sin = phase.sin();
            let cos = phase.cos();
            out.push(IqSample {
                i: sample.i * cos - sample.q * sin,
                q: sample.i * sin + sample.q * cos,
            });
            phase += increment;
            if phase >= TAU || phase <= -TAU {
                phase %= TAU;
            }
        }

        out
    }

    #[test]
    fn recovers_bpsk_symbols() {
        let expected = [1, 0, 1, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: if *bit == 1 { 1.0 } else { -1.0 },
                q: 0.0,
            })
            .collect();
        let samples = repeat_symbols(&symbols, 4);
        let config = LinearConfig::new(4_800, 1_200, LinearMode::Bpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn recovers_bpsk_with_swapped_iq() {
        let expected = [1, 0, 1, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: 0.0,
                q: if *bit == 1 { 1.0 } else { -1.0 },
            })
            .collect();
        let samples = repeat_symbols(&symbols, 4);
        let mut config = LinearConfig::new(4_800, 1_200, LinearMode::Bpsk);
        config.swap_iq = true;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn corrects_fixed_bpsk_frequency_offset() {
        let expected = [1, 0, 1, 1, 0, 0, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: if *bit == 1 { 1.0 } else { -1.0 },
                q: 0.0,
            })
            .collect();
        let clean = repeat_symbols(&symbols, 80);
        let offset_samples = add_frequency_offset(&clean, 96_000, 100.0);
        let mut config = LinearConfig::new(96_000, 1_200, LinearMode::Bpsk);
        config.frequency_offset_hz = 100.0;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&offset_samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn carrier_loop_acquires_uncorrected_offset() {
        // Regression guard for the Costas carrier loop. Every other test
        // leaves `carrier_loop_bandwidth = 0`, so `costas_derotate`
        // short-circuits and the loop never runs -- which is how a
        // positive-feedback sign error and a per-symbol/per-sample integral
        // mis-scaling both shipped while 56 tests passed. Here we force the
        // loop to acquire and track a frequency offset that is NOT corrected
        // statically (frequency_offset_hz = 0), at a realistic 40 samples per
        // symbol, and require bit-exact recovery up to the BPSK 180-degree
        // phase ambiguity.
        let bits: Vec<u8> = (0..240u32).map(|i| u8::from((i * 7 + 3) % 5 == 0)).collect();
        let symbols: Vec<IqSample> = bits
            .iter()
            .map(|b| IqSample {
                i: if *b == 1 { 1.0 } else { -1.0 },
                q: 0.0,
            })
            .collect();
        let clean = repeat_symbols(&symbols, 40); // 48000 Hz / 1200 baud = 40 sps
        let offset = add_frequency_offset(&clean, 48_000, 40.0);

        let mut config = LinearConfig::new(48_000, 1_200, LinearMode::Bpsk);
        config.frequency_offset_hz = 0.0; // loop must acquire it, not static correction
        config.carrier_loop_bandwidth = 0.05;
        let mut demodulator = LinearDemodulator::new(config).expect("valid config");

        let recovered = demodulator.push_samples(&offset);

        // Skip the acquisition transient, then the loop must be locked: the
        // tail matches the transmitted bits exactly, modulo a global inversion.
        //
        // The skip window widened from 60 to 150 symbols when symbol timing
        // moved to the joint Gardner + boxcar path: the timing loop and the
        // Costas loop now acquire together from a cold start, so the settled
        // region begins later than it did under the fixed-phase
        // integrate-and-dump sampler. This is a warm-up shift, not a weaker
        // lock — beyond the transient the tail error rate is 0.000 here (the
        // 0.2 ceiling is a generous unlocked/random guard, unchanged).
        let skip = 150usize;
        let n = recovered.len().min(bits.len());
        assert!(n > skip + 80, "too few symbols recovered: {n}");
        let direct = (skip..n).filter(|&k| recovered[k] != bits[k]).count();
        let inverted = (skip..n).filter(|&k| recovered[k] == bits[k]).count();
        let errors = direct.min(inverted);
        let rate = errors as f32 / (n - skip) as f32;
        // A locked loop drives the settled-tail error rate far below the 0.5
        // of an unlocked/random output. Before the sign fix the loop locked
        // onto the quadrature axis and this rate sat near 0.5; after the fix
        // it is well under 0.2.
        assert!(
            rate < 0.2,
            "carrier loop did not lock: error rate {rate:.3} (errors={errors} direct={direct} inverted={inverted} of {})",
            n - skip
        );
    }

    #[test]
    fn recovers_dbpsk_phase_changes_with_carrier_phase_offset() {
        let expected = [0, 1, 0, 1, 1, 0, 0, 1];
        let mut symbol = IqSample { i: 1.0, q: 0.0 };
        let rotation = IqSample { i: 0.0, q: 1.0 };
        let mut symbols = Vec::new();

        for bit in expected {
            if bit == 1 {
                symbol.i = -symbol.i;
                symbol.q = -symbol.q;
            }
            symbols.push(IqSample {
                i: symbol.i * rotation.i - symbol.q * rotation.q,
                q: symbol.i * rotation.q + symbol.q * rotation.i,
            });
        }

        let samples = repeat_symbols(&symbols, 8);
        let config = LinearConfig::new(9_600, 1_200, LinearMode::Dbpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn recovers_qpsk_symbols_as_iq_bit_pairs() {
        // Each quadrant maps to a fixed 2-bit pair; check that both axes are
        // sliced together at the recovered symbol instant. The Gardner timing
        // loop's TED is now rotation invariant (it consumes both I and Q), so
        // like the other timing-sensitive tests in this module the stream is
        // repeated long enough to clear the loop's cold-start warm-up and the
        // settled tail is what gets asserted, rather than a 3-symbol toy burst
        // where the last decision straddles the end-of-stream boundary.
        let pattern = [
            IqSample { i: 1.0, q: 1.0 },
            IqSample { i: -1.0, q: 1.0 },
            IqSample { i: -1.0, q: -1.0 },
            IqSample { i: 1.0, q: -1.0 },
        ];
        // slice_quadrature maps each axis independently: bit = (axis >= 0).
        let expected_pairs = [[1u8, 1], [0, 1], [0, 0], [1, 0]];
        let symbols: Vec<IqSample> = (0..40).flat_map(|_| pattern).collect();
        let samples = repeat_symbols(&symbols, 4);
        let config = LinearConfig::new(4_800, 1_200, LinearMode::Qpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        // Skip the warm-up, then every emitted bit pair must match the fixed
        // quadrant mapping of the repeating pattern. Align the settled tail to
        // a pattern boundary so the 4-symbol cycle indexes cleanly.
        let skip_pairs = 20usize;
        let pairs: Vec<&[u8]> = recovered.chunks(2).skip(skip_pairs).collect();
        assert!(pairs.len() > 40, "too few QPSK pairs recovered: {}", pairs.len());
        // Find the phase of the first settled pair within the 4-symbol cycle.
        let phase = (0..4)
            .find(|&p| {
                pairs
                    .iter()
                    .take(4)
                    .enumerate()
                    .all(|(k, pair)| pair == &expected_pairs[(p + k) % 4])
            })
            .expect("settled QPSK pairs do not match any cycle phase");
        for (k, pair) in pairs.iter().enumerate() {
            assert_eq!(
                *pair,
                &expected_pairs[(phase + k) % 4],
                "QPSK pair {k} mismatch"
            );
        }
    }

    #[test]
    fn rrc_taps_have_unit_energy_and_symmetric_shape() {
        let taps = rrc_taps(8.0, 0.5, 6);
        let energy: f32 = taps.iter().map(|t| t * t).sum();
        assert!((energy - 1.0).abs() < 1.0e-4, "energy={energy}");
        let len = taps.len();
        for k in 0..len / 2 {
            let lhs = taps[k];
            let rhs = taps[len - 1 - k];
            assert!(
                (lhs - rhs).abs() < 1.0e-4,
                "tap[{k}]={lhs} tap[{}]={rhs}",
                len - 1 - k
            );
        }
    }

    #[test]
    fn matched_filter_recovers_dbpsk_with_pulse_shaping() {
        let sample_rate = 9_600u32;
        let baudrate = 1_200u32;
        let sps = (sample_rate / baudrate) as usize;
        let pulse = rrc_taps(sps as f32, 0.5, 6);

        // The demodulator now recovers symbol timing with a closed Gardner
        // loop instead of a fixed-phase integrate-and-dump sampler, so the
        // signal must be long enough for the loop to acquire before the
        // settled region is inspected. The former 16-symbol burst was
        // shorter than the joint RRC + timing warm-up. 200 DBPSK symbols
        // leaves an ample locked tail.
        let bits: Vec<u8> = (0..200u32).map(|i| u8::from((i * 13 + 5) % 7 < 3)).collect();
        let mut symbol = 1.0f32;
        let mut symbol_stream: Vec<f32> = Vec::new();
        for &bit in &bits {
            if bit == 1 {
                symbol = -symbol;
            }
            symbol_stream.push(symbol);
        }
        let upsampled_len = symbol_stream.len() * sps + pulse.len();
        let mut shaped: Vec<IqSample> = vec![IqSample::default(); upsampled_len];
        for (n, sym) in symbol_stream.iter().enumerate() {
            for (k, tap) in pulse.iter().enumerate() {
                shaped[n * sps + k].i += *sym * *tap;
            }
        }

        let mut config = LinearConfig::new(sample_rate, baudrate, LinearMode::Dbpsk);
        config.matched_filter_rolloff = 0.5;
        config.matched_filter_span_symbols = 6;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };
        let recovered = demodulator.push_samples(&shaped);

        // Compare the settled tail against the source over a small alignment
        // window: the Gardner loop's warm-up can offset the recovered stream
        // by a few symbols relative to the source (same rationale as the BER
        // testbench's `min_errors_after_alignment`). A locked loop drives the
        // aligned tail to zero errors.
        let tail = 80usize;
        assert!(
            recovered.len() >= tail + 8,
            "need at least {} recovered bits, got {}",
            tail + 8,
            recovered.len()
        );
        let expected_tail = &bits[bits.len() - tail..];
        let mut best = usize::MAX;
        for shift in 0..8 {
            if recovered.len() < tail + shift {
                continue;
            }
            let recovered_tail = &recovered[recovered.len() - tail - shift..recovered.len() - shift];
            let mismatches = recovered_tail
                .iter()
                .zip(expected_tail)
                .filter(|(a, b)| a != b)
                .count();
            best = best.min(mismatches.min(tail - mismatches));
        }
        assert!(best <= 1, "settled-tail mismatches after alignment: {best}");
    }
}
