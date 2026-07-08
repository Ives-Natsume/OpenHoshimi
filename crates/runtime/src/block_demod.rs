//! Block (non-streaming) CCSDS frame recovery via ASM correlation.
//!
//! The streaming pipeline can only lock frames its carrier/timing loops track
//! in real time, so on a weak pass it misses frames the Viterbi never syncs.
//! This offline stage takes the whole run's complex decision symbols and does
//! what a streaming decoder structurally cannot:
//!
//! 1. Correlate the channel-symbol stream against the *coded* ASM pattern.
//!    The CCSDS ASM (0x1ACFFC1D) is prepended before the convolutional encoder,
//!    so in the channel domain it is `conv_encode(nrzs(ASM))`. Correlating the
//!    52 channel bits that depend only on the ASM (the first ~6 input bits
//!    depend on the unknown preceding state) gives ~17 dB of processing gain --
//!    enough to detect a frame start well below the RS decode threshold. A
//!    coherent complex correlation makes the detector invariant to the
//!    (unknown, roughly constant) carrier phase over the marker.
//!
//! 2. Use each detected marker as a pilot: the correlation's argument is the
//!    residual carrier phase, so the frame is de-rotated by it -- a cleaner
//!    reference than the streaming Costas loop achieves on a weak pass -- then
//!    Viterbi-decoded, NRZ-S decoded, and handed to the outer RS stage as a
//!    255-byte codeword (with per-bit reliabilities for erasure decoding).
//!
//! The recovered codewords flow through the same [`openhoshimi_codec`] RS path
//! as the streaming frames, including the erasure-forcing sanity gate, so a
//! false correlation peak on noise fails RS exactly like a streaming false
//! sync and is dropped.

use std::time::SystemTime;

use openhoshimi_codec::{StreamingViterbi, CCSDS_ASM};
use openhoshimi_core::{Frame, FrameType};

/// Channel bits produced by one 32-bit ASM through the rate-1/2 encoder.
const ASM_CHANNEL_BITS: usize = 32 * 2;
/// Number of leading ASM channel bits whose value depends on the unknown
/// convolutional-encoder state entering the marker (first 6 input bits x 2).
const ASM_UNCERTAIN_BITS: usize = 12;
/// Length of the correlation template: the ASM channel bits that depend only
/// on the ASM itself.
const TEMPLATE_LEN: usize = ASM_CHANNEL_BITS - ASM_UNCERTAIN_BITS; // 52
/// One RS(255,223) codeword as channel bits at rate 1/2.
const CODEWORD_CHANNEL_BITS: usize = 255 * 8 * 2; // 4080
/// RS codeword size in bytes.
const CODEWORD_BYTES: usize = 255;
/// RS codeword size in bits.
const CODEWORD_BITS: usize = CODEWORD_BYTES * 8; // 2040
/// ASM pattern length in decoded bits.
const ASM_BITS: usize = 32;

/// CCSDS K=7 rate-1/2 convolutional encoder, matching the decoder convention
/// in `openhoshimi_codec::fec` (G1 = 0x4f, G2 = 0x6d with the second output
/// inverted). Returns two channel bits per input bit, starting from state 0.
fn conv_encode(bits: &[u8]) -> Vec<u8> {
    const G1: u8 = 0x4f;
    const G2: u8 = 0x6d;
    let mut state = 0u8;
    let mut out = Vec::with_capacity(bits.len() * 2);
    for &bit in bits {
        let reg = ((state << 1) | (bit & 1)) & 0x7f;
        out.push((reg & G1).count_ones() as u8 & 1);
        out.push(((reg & G2).count_ones() as u8 & 1) ^ 1);
        state = ((state << 1) | (bit & 1)) & 0x3f;
    }
    out
}

/// NRZ-S (differential) encode: `c[k] = c[k-1] XOR m[k]`, starting from the
/// given reference bit. The transmit chain applies this after the encoder, so
/// the channel-domain ASM is `nrzs(conv_encode(...))`... but the pipeline
/// Viterbi-decodes then NRZ-S decodes, i.e. it treats the channel symbols as a
/// plain convolutional code whose *input* was NRZ-S precoded. The channel ASM
/// pattern is therefore `conv_encode(nrzs(ASM))`.
fn nrzs_encode(bits: &[u8], reference: u8) -> Vec<u8> {
    let mut prev = reference & 1;
    bits.iter()
        .map(|&b| {
            prev ^= b & 1;
            prev
        })
        .collect()
}

/// The two polarity-variant correlation templates (`+1.0` for channel bit 0,
/// `-1.0` for bit 1), one per NRZ-S reference. The reference is the unknown
/// bit before the ASM; its two values give complementary precodings, and the
/// coherent correlation picks whichever matches.
fn asm_templates() -> [[f32; TEMPLATE_LEN]; 2] {
    let asm_bits: Vec<u8> = (0..ASM_BITS)
        .map(|i| ((CCSDS_ASM >> (31 - i)) & 1) as u8)
        .collect();
    let mut templates = [[0.0f32; TEMPLATE_LEN]; 2];
    for (reference, template) in templates.iter_mut().enumerate() {
        let channel = conv_encode(&nrzs_encode(&asm_bits, reference as u8));
        for (k, slot) in template.iter_mut().enumerate() {
            *slot = if channel[ASM_UNCERTAIN_BITS + k] == 0 {
                1.0
            } else {
                -1.0
            };
        }
    }
    templates
}

/// A detected ASM: channel index of the first template tap, the residual
/// carrier phase to de-rotate by, and the normalised correlation magnitude.
struct AsmHit {
    /// Channel-symbol index the template's first tap aligns to.
    template_pos: usize,
    /// Residual carrier phase (radians) from the coherent correlation argument.
    phase: f32,
    /// Correlation magnitude normalised by the mean symbol amplitude, so a
    /// perfect match approaches [`TEMPLATE_LEN`].
    score: f32,
}

/// Slide the coherent complex correlator over the symbol stream and return the
/// local-maximum ASM hits above `threshold`, de-duplicated within `min_gap`.
fn correlate_asms(symbols: &[(f32, f32)], threshold: f32, min_gap: usize) -> Vec<AsmHit> {
    if symbols.len() < TEMPLATE_LEN {
        return Vec::new();
    }
    let templates = asm_templates();
    // Mean symbol amplitude normalises the score to ~TEMPLATE_LEN at a perfect
    // match regardless of AGC level.
    let mut amp_sum = 0.0f64;
    for &(i, q) in symbols {
        amp_sum += (i * i + q * q).sqrt() as f64;
    }
    let mean_amp = (amp_sum / symbols.len() as f64).max(1e-6) as f32;

    let windows = symbols.len() - TEMPLATE_LEN + 1;
    let mut hits: Vec<AsmHit> = Vec::new();
    let mut best_in_run: Option<AsmHit> = None;
    let mut last_pos: usize = 0;
    for pos in 0..windows {
        // Coherent correlation for both polarity variants; keep the stronger.
        let mut best_score = 0.0f32;
        let mut best_phase = 0.0f32;
        for template in &templates {
            let mut acc_i = 0.0f32;
            let mut acc_q = 0.0f32;
            for (k, &t) in template.iter().enumerate() {
                let (si, sq) = symbols[pos + k];
                acc_i += si * t;
                acc_q += sq * t;
            }
            let mag = (acc_i * acc_i + acc_q * acc_q).sqrt() / mean_amp;
            if mag > best_score {
                best_score = mag;
                best_phase = acc_q.atan2(acc_i);
            }
        }
        if best_score < threshold {
            continue;
        }
        let hit = AsmHit {
            template_pos: pos,
            phase: best_phase,
            score: best_score,
        };
        match best_in_run.as_ref() {
            Some(prev) if pos - last_pos <= min_gap => {
                if hit.score > prev.score {
                    best_in_run = Some(hit);
                    last_pos = pos;
                } else {
                    last_pos = pos;
                }
            }
            _ => {
                if let Some(prev) = best_in_run.take() {
                    hits.push(prev);
                }
                last_pos = pos;
                best_in_run = Some(hit);
            }
        }
    }
    if let Some(prev) = best_in_run.take() {
        hits.push(prev);
    }
    hits
}

/// Quantise a de-rotated real decision value to the soft-Viterbi i8 convention
/// (bit 0 -> positive, magnitude ~64 for a confident symbol).
fn quantize_soft(value: f32) -> i8 {
    let scaled = (value * 64.0).clamp(i8::MIN as f32 + 1.0, i8::MAX as f32);
    scaled.round() as i8
}

/// Find the ASM (0x1ACFFC1D) in a decoded bit stream, allowing up to
/// `max_hamming` bit errors. Returns the bit index of the ASM's first bit.
fn find_asm(bits: &[u8], max_hamming: u32) -> Option<usize> {
    if bits.len() < ASM_BITS {
        return None;
    }
    let mut best: Option<(usize, u32)> = None;
    for start in 0..=bits.len() - ASM_BITS {
        let mut dist = 0u32;
        for i in 0..ASM_BITS {
            let expected = ((CCSDS_ASM >> (31 - i)) & 1) as u8;
            if bits[start + i] != expected {
                dist += 1;
                if dist > max_hamming {
                    break;
                }
            }
        }
        if dist <= max_hamming && best.map_or(true, |(_, d)| dist < d) {
            best = Some((start, dist));
            if dist == 0 {
                break;
            }
        }
    }
    best.map(|(pos, _)| pos)
}

/// Attempt to recover one frame from a detected ASM. Returns a CCSDS codeword
/// [`Frame`] (255 bytes + per-bit reliabilities) for the outer RS stage, or
/// `None` if the window is too short or the ASM does not re-appear after
/// Viterbi decoding.
fn decode_hit(symbols: &[(f32, f32)], hit: &AsmHit) -> Option<Frame> {
    // The template's first tap sits ASM_UNCERTAIN_BITS into the ASM, so the
    // marker (and thus the frame) starts a little earlier.
    let asm_start = hit.template_pos.checked_sub(ASM_UNCERTAIN_BITS)?;
    // Decode from a little before the ASM (Viterbi warm-up / traceback) through
    // the ASM plus one full codeword, with a margin for the traceback delay.
    const WARMUP: usize = 64;
    const TAIL_MARGIN: usize = 96;
    let start = asm_start.saturating_sub(WARMUP);
    let needed = ASM_CHANNEL_BITS + CODEWORD_CHANNEL_BITS + TAIL_MARGIN;
    let end = (asm_start + needed).min(symbols.len());
    if end < asm_start + ASM_CHANNEL_BITS + CODEWORD_CHANNEL_BITS {
        return None;
    }
    let window = &symbols[start..end];
    // Window index of the ASM's first symbol (the phase-ramp reference point).
    let asm_index = asm_start - start;

    let mut amp_sum = 0.0f64;
    for &(i, q) in window {
        amp_sum += (i * i + q * q).sqrt() as f64;
    }
    let scale = (amp_sum / window.len() as f64).max(1e-6) as f32;

    // A weak pass leaves a residual carrier frequency the streaming loops did
    // not remove; the coherent ASM correlation is blind to it (it only spans
    // ~52 symbols) but it rotates the frame body through many turns, so a
    // constant-phase de-rotation decodes to noise. Search a per-frame frequency
    // (`dphi`, radians/symbol) around the ASM phase, referencing the ramp at the
    // ASM so the marker stays aligned, and keep the estimate that yields the
    // cleanest decode. Coarse-to-fine: localise the ASM, then refine for the
    // body (the body needs `dphi` an order of magnitude tighter than the ASM).
    let evaluate = |dphi: f32| -> Option<(u64, Vec<u8>, Vec<u32>)> {
        let soft: Vec<i8> = window
            .iter()
            .enumerate()
            .map(|(k, &(i, q))| {
                let a = -(hit.phase + dphi * (k as f32 - asm_index as f32));
                let (c, s) = (a.cos(), a.sin());
                // Soft convention: bit 0 -> positive, so negate the in-phase axis.
                quantize_soft(-(i * c - q * s) / scale)
            })
            .collect();
        let mut viterbi = StreamingViterbi::new();
        let decoded = viterbi.push_soft_symbols(&soft);
        let reliabilities = viterbi.last_reliabilities();
        if decoded.len() != reliabilities.len() || decoded.len() < ASM_BITS + CODEWORD_BITS {
            return None;
        }
        // NRZ-S decode: d[k] = out[k] ^ out[k-1]. Invariant to global inversion,
        // so the 180-degree residual the phase estimate cannot resolve drops out.
        let mut nrzs_bits = Vec::with_capacity(decoded.len());
        let mut nrzs_rel = Vec::with_capacity(decoded.len());
        let mut prev_bit = 0u8;
        let mut prev_rel = 0u32;
        for (idx, (&bit, &rel)) in decoded.iter().zip(reliabilities.iter()).enumerate() {
            if idx == 0 {
                nrzs_bits.push(0);
                nrzs_rel.push(0);
            } else {
                nrzs_bits.push((bit & 1) ^ prev_bit);
                nrzs_rel.push(rel.min(prev_rel));
            }
            prev_bit = bit & 1;
            prev_rel = rel;
        }
        let asm_pos = find_asm(&nrzs_bits, 6)?;
        let payload_start = asm_pos + ASM_BITS;
        if payload_start + CODEWORD_BITS > nrzs_bits.len() {
            return None;
        }
        // Quality = summed reliability of the codeword bits; the correct
        // frequency de-rotates the whole body coherently, maximising it.
        let quality: u64 = nrzs_rel[payload_start..payload_start + CODEWORD_BITS]
            .iter()
            .map(|&r| u64::from(r))
            .sum();
        let raw_bits = nrzs_bits[payload_start..payload_start + CODEWORD_BITS].to_vec();
        let raw_rel = nrzs_rel[payload_start..payload_start + CODEWORD_BITS].to_vec();
        Some((quality, raw_bits, raw_rel))
    };

    // Coarse frequency sweep (+/- ~0.025 rad/sym ~= +/-38 Hz at 9600 baud).
    let mut best: Option<(f32, u64, Vec<u8>, Vec<u32>)> = None;
    let consider = |dphi: f32, best: &mut Option<(f32, u64, Vec<u8>, Vec<u32>)>| {
        if let Some((q, bits, rel)) = evaluate(dphi) {
            if best.as_ref().map_or(true, |(_, bq, _, _)| q > *bq) {
                *best = Some((dphi, q, bits, rel));
            }
        }
    };
    let mut dphi = -0.025;
    while dphi <= 0.025 {
        consider(dphi, &mut best);
        dphi += 0.0025;
    }
    // Fine sweep around the coarse best for the body (tighter by ~10x).
    if let Some((coarse, _, _, _)) = best.clone() {
        let mut d = coarse - 0.0025;
        while d <= coarse + 0.0025 {
            consider(d, &mut best);
            d += 0.00025;
        }
    }
    let (best_dphi, best_quality, mut codeword_bits, mut codeword_rel) = best?;

    // Iterative decision-directed phase refinement. The coarse+fine `dphi`
    // sweep only fits a *constant* residual frequency; a weak pass also carries
    // slower non-linear phase wander that a linear ramp cannot follow, and that
    // wander is what keeps the body above the Viterbi threshold. Re-encode the
    // tentative Viterbi output to a channel reference, estimate a smoothed
    // per-symbol carrier phase against it (a wrong reference bit only adds noise
    // the smoothing averages out, so it still converges from a majority-correct
    // decode), de-rotate, and re-decode. Opt-in: the default block path is the
    // proven constant-frequency decode.
    if std::env::var_os("OPENHOSHIMI_BLOCK_ITER").is_some() {
        if let Some((q, bits, rel)) = refine_phase(window, scale, hit.phase, best_dphi, asm_index) {
            if q > best_quality {
                codeword_bits = bits;
                codeword_rel = rel;
            }
        }
    }

    // Pack MSB-first into the 255-byte codeword and carry per-bit reliabilities.
    let mut raw = vec![0u8; CODEWORD_BYTES];
    for (i, chunk) in codeword_bits.chunks_exact(8).enumerate() {
        let mut byte = 0u8;
        for &b in chunk {
            byte = (byte << 1) | (b & 1);
        }
        raw[i] = byte;
    }
    let soft_bits: Vec<f32> = codeword_rel.iter().map(|&r| r as f32).collect();

    Some(Frame {
        satellite_id: 0,
        timestamp: SystemTime::now(),
        rssi_dbm: None,
        raw,
        frame_type: FrameType::Ccsds,
        soft_bits: Some(soft_bits),
    })
}

/// Iterative decision-directed phase refinement for one frame window. Seeds
/// from the constant-frequency (`phase0` + `dphi0` ramp) decode, then repeatedly
/// re-encodes the tentative Viterbi output to a channel reference, fits a
/// smoothed per-symbol carrier phase against the raw symbols, de-rotates, and
/// re-decodes. A wrong reference bit only injects noise the smoothing averages
/// out, so it converges from a majority-correct decode and follows the slow
/// non-linear phase wander a constant-frequency estimate leaves behind. Returns
/// the best codeword bits + reliabilities seen across the iterations (by summed
/// reliability), or `None` if no iteration locates the ASM.
fn refine_phase(
    window: &[(f32, f32)],
    scale: f32,
    phase0: f32,
    dphi0: f32,
    asm_index: usize,
) -> Option<(u64, Vec<u8>, Vec<u32>)> {
    const ITERS: usize = 4;
    const SMOOTH: usize = 15;

    let mut soft: Vec<i8> = window
        .iter()
        .enumerate()
        .map(|(k, &(i, q))| {
            let a = -(phase0 + dphi0 * (k as f32 - asm_index as f32));
            quantize_soft(-(i * a.cos() - q * a.sin()) / scale)
        })
        .collect();

    let mut result: Option<(u64, Vec<u8>, Vec<u32>)> = None;
    for _ in 0..ITERS {
        let mut viterbi = StreamingViterbi::new();
        let decoded = viterbi.push_soft_symbols(&soft);
        let reliabilities = viterbi.last_reliabilities();
        if decoded.len() != reliabilities.len() || decoded.len() < ASM_BITS + CODEWORD_BITS {
            return result;
        }

        // Extract the codeword for the current decode; keep the best so far.
        let mut nrzs_bits = Vec::with_capacity(decoded.len());
        let mut nrzs_rel = Vec::with_capacity(decoded.len());
        let mut prev_bit = 0u8;
        let mut prev_rel = 0u32;
        for (idx, (&bit, &rel)) in decoded.iter().zip(reliabilities.iter()).enumerate() {
            if idx == 0 {
                nrzs_bits.push(0);
                nrzs_rel.push(0);
            } else {
                nrzs_bits.push((bit & 1) ^ prev_bit);
                nrzs_rel.push(rel.min(prev_rel));
            }
            prev_bit = bit & 1;
            prev_rel = rel;
        }
        if let Some(asm_pos) = find_asm(&nrzs_bits, 8) {
            let ps = asm_pos + ASM_BITS;
            if ps + CODEWORD_BITS <= nrzs_bits.len() {
                let q: u64 = nrzs_rel[ps..ps + CODEWORD_BITS]
                    .iter()
                    .map(|&r| u64::from(r))
                    .sum();
                if result.as_ref().map_or(true, |(bq, _, _)| q > *bq) {
                    result = Some((
                        q,
                        nrzs_bits[ps..ps + CODEWORD_BITS].to_vec(),
                        nrzs_rel[ps..ps + CODEWORD_BITS].to_vec(),
                    ));
                }
            }
        }

        // Re-encode the tentative decode to a channel reference, aligned 1:1
        // with the window (decoded bit j -> channel symbols 2j, 2j+1).
        let ref_channel = conv_encode(&decoded);
        let n = window.len().min(ref_channel.len());
        if n == 0 {
            return result;
        }
        // Raw per-symbol carrier phase: angle of the symbol rotated onto the
        // reference's expected axis (bit 0 -> -1, bit 1 -> +1).
        let mut phase = vec![0.0f32; n];
        for (k, slot) in phase.iter_mut().enumerate() {
            let sign = if ref_channel[k] == 0 { -1.0 } else { 1.0 };
            let (i, q) = window[k];
            *slot = (sign * q).atan2(sign * i);
        }
        // Unwrap so the moving average does not straddle +/-pi wraps.
        for k in 1..n {
            while phase[k] - phase[k - 1] > std::f32::consts::PI {
                phase[k] -= std::f32::consts::TAU;
            }
            while phase[k] - phase[k - 1] < -std::f32::consts::PI {
                phase[k] += std::f32::consts::TAU;
            }
        }
        // Moving-average smooth: rejects the noise from wrong reference bits.
        let half = SMOOTH / 2;
        let mut smoothed = vec![0.0f32; n];
        for (k, slot) in smoothed.iter_mut().enumerate() {
            let lo = k.saturating_sub(half);
            let hi = (k + half + 1).min(n);
            *slot = phase[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        }
        // Re-derotate with the estimated phase (extrapolate the tail).
        soft = window
            .iter()
            .enumerate()
            .map(|(k, &(i, q))| {
                let ph = smoothed[k.min(n - 1)];
                quantize_soft(-(i * ph.cos() + q * ph.sin()) / scale)
            })
            .collect();
    }
    result
}

/// Recover CCSDS codewords from a whole run of complex decision symbols by ASM
/// correlation. `threshold` is the minimum normalised correlation score; peaks
/// closer than `min_gap` channel symbols are collapsed to their strongest.
///
/// The returned frames are 255-byte codewords for the outer RS stage; false
/// correlation peaks yield codewords that fail RS downstream, exactly like a
/// streaming false sync.
pub fn recover_frames(symbols: &[(f32, f32)], threshold: f32, min_gap: usize) -> Vec<Frame> {
    correlate_asms(symbols, threshold, min_gap)
        .iter()
        .filter_map(|hit| decode_hit(symbols, hit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_is_52_bipolar_taps() {
        let t = asm_templates();
        for template in &t {
            assert_eq!(template.len(), TEMPLATE_LEN);
            for &tap in template.iter() {
                assert!(tap == 1.0 || tap == -1.0);
            }
        }
        // The two references give complementary NRZ-S, so the two templates
        // must differ (a real polarity ambiguity, not a duplicate).
        assert_ne!(t[0], t[1]);
    }

    #[test]
    fn correlates_perfectly_on_clean_encoded_asm() {
        // Build a clean channel-symbol stream: encode nrzs(ASM) and lay the 52
        // template taps as unit-amplitude symbols. The correlator must peak at
        // TEMPLATE_LEN with near-zero phase.
        let templates = asm_templates();
        let symbols: Vec<(f32, f32)> = templates[0].iter().map(|&t| (t, 0.0)).collect();
        let hits = correlate_asms(&symbols, 40.0, 100);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 50.0, "score {}", hits[0].score);
        assert!(hits[0].phase.abs() < 0.05, "phase {}", hits[0].phase);
    }
}
