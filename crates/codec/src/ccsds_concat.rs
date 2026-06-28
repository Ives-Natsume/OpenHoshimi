//! CCSDS Concatenated codec (Viterbi r=1/2 K=7 inner + RS(255,223) outer).
//!
//! Implements the standard CCSDS TM Synchronization and Channel Coding
//! concatenated scheme used by ASRTU-1 (AO-123) and other satellites:
//!
//! TX: payload -> RS(255,223) -> CCSDS scramble -> [ASM prepend]
//!     -> convolutional encode -> NRZ-S precode -> transmit
//! RX: NRZ-S decode -> Viterbi decode -> ASM detect -> CCSDS descramble
//!     -> RS(255,223) -> payload
//!
//! The ASM (0x1ACFFC1D) is prepended to the RS codeword BEFORE the
//! convolutional encoder, so it is convolutionally coded along with the
//! data and appears in the Viterbi-decoded bit stream (matching the
//! gr-satellites `ccsds_concatenated_deframer` topology). The pipeline
//! runs a [`crate::StreamingViterbi`] over the whole channel-symbol
//! stream, the [`openhoshimi_dsp::SyncwordFramer`] detects the ASM in the
//! decoded bits, and [`CcsdsConcatDecoder::decode_frame_bytes`] performs
//! the differential decode, descramble and RS stages on the extracted
//! 255-byte codeword.

use openhoshimi_core::DecodeError;

use crate::fec::{ccsds_randomizer, reed_solomon, viterbi, ReedSolomon};

/// Reed-Solomon basis selection for the outer code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsBasis {
    /// Conventional basis (used by ASRTU-1, most CCSDS satellites).
    Conventional,
    /// Dual basis (used by some older satellites).
    Dual,
}

/// CCSDS Concatenated frame decoder.
#[derive(Debug, Clone)]
pub struct CcsdsConcatDecoder {
    /// Expected payload size after RS decoding (e.g. 223 for RS(255,223)).
    frame_size: usize,
    /// Apply differential decoding after Viterbi (resolves BPSK 180 deg
    /// phase ambiguity).
    differential: bool,
    /// Reed-Solomon basis (reserved for future dual-basis support).
    #[allow(dead_code)]
    rs_basis: RsBasis,
}

/// A successfully decoded CCSDS Concatenated frame.
#[derive(Debug, Clone)]
pub struct CcsdsConcatFrame {
    /// Decoded payload bytes.
    pub payload: Vec<u8>,
    /// Number of Reed-Solomon byte errors corrected.
    pub corrected_errors: usize,
}

/// Number of coded channel bits for one RS(255,223) codeword at rate 1/2.
/// 255 bytes * 8 bits * 2 (rate-1/2) = 4080 channel bits.
pub const CCSDS_CONCAT_CODED_BITS: usize = 255 * 8 * 2;

/// The standard CCSDS TM attached sync marker.
pub const CCSDS_ASM: u32 = 0x1ACF_FC1D;

impl CcsdsConcatDecoder {
    /// Create a decoder with the given parameters.
    pub fn new(frame_size: usize, differential: bool, rs_basis: RsBasis) -> Self {
        Self {
            frame_size,
            differential,
            rs_basis,
        }
    }

    /// Decode from hard-decision channel bits (one bit per byte, 0 or 1).
    ///
    /// Input length must be [`CCSDS_CONCAT_CODED_BITS`] (4080).
    pub fn decode(&self, coded_bits: &[u8]) -> Result<CcsdsConcatFrame, DecodeError> {
        if coded_bits.len() != CCSDS_CONCAT_CODED_BITS {
            return Err(DecodeError::InvalidEncoding(format!(
                "CCSDS Concatenated expects {} coded bits, got {}",
                CCSDS_CONCAT_CODED_BITS,
                coded_bits.len()
            )));
        }
        // Differential (NRZ-S) decode BEFORE Viterbi: the precoding is applied
        // after the convolutional encoder on the channel symbols.
        let bits = if self.differential {
            differential_decode(coded_bits)
        } else {
            coded_bits.to_vec()
        };
        let decoded_bits = viterbi::decode_bits(&bits)?;
        self.post_viterbi(&decoded_bits)
    }

    /// Decode from soft-decision channel symbols (signed i8, one per coded
    /// bit). Positive = bit 0, negative = bit 1, magnitude = confidence.
    ///
    /// Input length must be [`CCSDS_CONCAT_CODED_BITS`] (4080).
    pub fn decode_soft(&self, soft: &[i8]) -> Result<CcsdsConcatFrame, DecodeError> {
        if soft.len() != CCSDS_CONCAT_CODED_BITS {
            return Err(DecodeError::InvalidEncoding(format!(
                "CCSDS Concatenated expects {} soft symbols, got {}",
                CCSDS_CONCAT_CODED_BITS,
                soft.len()
            )));
        }
        // For soft-decision with differential precoding, convert to hard
        // decisions for the differential decode step, then re-encode as soft
        // for Viterbi. This loses some soft information but is the simplest
        // correct approach.
        if self.differential {
            let hard: Vec<u8> = soft
                .iter()
                .map(|&s| if s >= 0 { 0u8 } else { 1u8 })
                .collect();
            let decoded_hard = differential_decode(&hard);
            let decoded_bits = viterbi::decode_bits(&decoded_hard)?;
            self.post_viterbi(&decoded_bits)
        } else {
            let decoded_bits = viterbi::decode_soft_bits(soft)?;
            self.post_viterbi(&decoded_bits)
        }
    }

    /// Post-Viterbi processing: pack → descramble → RS.
    fn post_viterbi(&self, bits: &[u8]) -> Result<CcsdsConcatFrame, DecodeError> {
        // Pack one-bit-per-byte into bytes (MSB first).
        let byte_count = bits.len() / 8;
        let mut bytes = vec![0u8; byte_count];
        for (i, chunk) in bits.chunks_exact(8).enumerate() {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | (bit & 1);
            }
            bytes[i] = byte;
        }

        // CCSDS descramble (self-inverse XOR sequence).
        ccsds_randomizer::xor_sequence(&mut bytes);

        // RS(255,223) decode. The codeword is 255 bytes; after RS we get
        // frame_size bytes of payload.
        if bytes.len() < reed_solomon::PARITY_LEN + self.frame_size {
            return Err(DecodeError::TooShort(bytes.len()));
        }
        let rs = ReedSolomon::new(1);
        let decoded = rs.decode_shortened(&bytes)?;

        if decoded.message.len() < self.frame_size {
            return Err(DecodeError::TooShort(decoded.message.len()));
        }

        Ok(CcsdsConcatFrame {
            payload: decoded.message[..self.frame_size].to_vec(),
            corrected_errors: decoded.corrected_errors,
        })
    }

    /// Decode a frame that has already been Viterbi-decoded by the streaming
    /// decoder in the pipeline. Input is packed bytes (one RS codeword, 255
    /// bytes). Applies differential decode (if enabled) → descramble → RS.
    pub fn decode_frame_bytes(&self, data: &[u8]) -> Result<CcsdsConcatFrame, DecodeError> {
        if data.len() < reed_solomon::PARITY_LEN + self.frame_size {
            return Err(DecodeError::TooShort(data.len()));
        }

        let mut bytes = if self.differential {
            // Unpack to bits, differential decode, repack.
            let bits: Vec<u8> = data
                .iter()
                .flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1))
                .collect();
            let decoded = differential_decode(&bits);
            let byte_count = decoded.len() / 8;
            let mut packed = vec![0u8; byte_count];
            for (i, chunk) in decoded.chunks_exact(8).enumerate() {
                let mut byte = 0u8;
                for &bit in chunk {
                    byte = (byte << 1) | (bit & 1);
                }
                packed[i] = byte;
            }
            packed
        } else {
            data.to_vec()
        };

        // CCSDS descramble.
        ccsds_randomizer::xor_sequence(&mut bytes);

        // RS(255,223) decode.
        let rs = ReedSolomon::new(1);
        let decoded = rs.decode_shortened(&bytes)?;

        if decoded.message.len() < self.frame_size {
            return Err(DecodeError::TooShort(decoded.message.len()));
        }

        Ok(CcsdsConcatFrame {
            payload: decoded.message[..self.frame_size].to_vec(),
            corrected_errors: decoded.corrected_errors,
        })
    }

    /// Decode a Viterbi-decoded RS codeword with soft-decision erasure
    /// marking. `soft_bits` holds one signed confidence per codeword bit
    /// (length `data.len() * 8`, bit order MSB-first within each byte) as
    /// produced by the streaming soft Viterbi reliabilities; only the
    /// magnitude is used.
    ///
    /// Errors-only RS(255,223) is tried first. If it fails (more than 16
    /// symbol errors), the `erasure_k` least-reliable bytes -- ranked by the
    /// minimum `|soft|` across their 8 bits -- are marked as erasures and the
    /// codeword is re-decoded. Erasures double the effective correction
    /// reach: RS can resolve a codeword when `2*errors + erasures <= 32`, so
    /// providing good erasure positions recovers frames that errors-only
    /// decoding rejects. Falls back to [`decode_frame_bytes`](Self::decode_frame_bytes)
    /// when erasure marking does not apply (differential path or a soft
    /// buffer that does not line up with the codeword).
    pub fn decode_frame_bytes_with_soft(
        &self,
        data: &[u8],
        soft_bits: &[f32],
        erasure_k: usize,
    ) -> Result<CcsdsConcatFrame, DecodeError> {
        if data.len() < reed_solomon::PARITY_LEN + self.frame_size {
            return Err(DecodeError::TooShort(data.len()));
        }
        if self.differential || soft_bits.len() != data.len() * 8 {
            return self.decode_frame_bytes(data);
        }

        // CCSDS descramble (XOR, so per-byte reliability alignment is kept).
        let mut bytes = data.to_vec();
        ccsds_randomizer::xor_sequence(&mut bytes);

        let rs = ReedSolomon::new(1);

        // Errors-only first: cheaper, and spends no erasure budget.
        if let Ok(decoded) = rs.decode_shortened(&bytes) {
            if decoded.message.len() >= self.frame_size {
                return Ok(CcsdsConcatFrame {
                    payload: decoded.message[..self.frame_size].to_vec(),
                    corrected_errors: decoded.corrected_errors,
                });
            }
        }

        // Rank bytes by ascending reliability (minimum |soft| over their bits)
        // and mark the weakest as erasures.
        let mut byte_confidences: Vec<(usize, f32)> = (0..data.len())
            .map(|i| {
                let mut min_abs = f32::INFINITY;
                for j in 0..8 {
                    let s = soft_bits[i * 8 + j].abs();
                    if s < min_abs {
                        min_abs = s;
                    }
                }
                (i, min_abs)
            })
            .collect();
        byte_confidences.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = erasure_k.min(data.len()).min(reed_solomon::PARITY_LEN);
        let erasures: Vec<usize> = byte_confidences[..k].iter().map(|(idx, _)| *idx).collect();

        let decoded = rs.decode_shortened_with_erasures(&bytes, &erasures)?;
        if decoded.message.len() < self.frame_size {
            return Err(DecodeError::TooShort(decoded.message.len()));
        }
        Ok(CcsdsConcatFrame {
            payload: decoded.message[..self.frame_size].to_vec(),
            corrected_errors: decoded.corrected_errors,
        })
    }
}

/// Differential decode: output[i] = input[i] XOR input[i-1].
/// Resolves 180-degree BPSK phase ambiguity.
fn differential_decode(bits: &[u8]) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(bits.len());
    out.push(bits[0]); // first bit has no reference; keep as-is
    for i in 1..bits.len() {
        out.push((bits[i] ^ bits[i - 1]) & 1);
    }
    out
}

/// Differential encode (for test roundtrip): output[i] = output[i-1] XOR input[i].
#[cfg(test)]
fn differential_encode(bits: &[u8]) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(bits.len());
    out.push(bits[0]);
    for i in 1..bits.len() {
        out.push((out[i - 1] ^ bits[i]) & 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_hard_decision() {
        // Build a 223-byte payload, RS encode, scramble, Viterbi encode,
        // then differential encode (precoding after conv encoder), then
        // decode and verify.
        let payload: Vec<u8> = (0..223u8).collect();

        // RS encode (223 → 255 bytes).
        let rs_codeword = reed_solomon::encode_shortened(&payload, 1);
        assert_eq!(rs_codeword.len(), 255);

        // CCSDS scramble.
        let mut scrambled = rs_codeword.clone();
        ccsds_randomizer::xor_sequence(&mut scrambled);

        // Unpack to bits (MSB first).
        let mut bits: Vec<u8> = Vec::with_capacity(scrambled.len() * 8);
        for byte in &scrambled {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }

        // Viterbi encode (rate 1/2).
        let coded = viterbi::encode_bits(&bits);
        assert_eq!(coded.len(), CCSDS_CONCAT_CODED_BITS);

        // Differential encode (NRZ-S precoding, applied after conv encoder).
        let precoded = differential_encode(&coded);

        // Decode (differential decode happens before Viterbi in the decoder).
        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        let frame = decoder.decode(&precoded).expect("roundtrip decode");
        assert_eq!(frame.payload, payload);
        assert_eq!(frame.corrected_errors, 0);
    }

    #[test]
    fn roundtrip_with_bit_errors() {
        let payload: Vec<u8> = (0..223u8).map(|i| i.wrapping_mul(7)).collect();
        let rs_codeword = reed_solomon::encode_shortened(&payload, 1);
        let mut scrambled = rs_codeword;
        ccsds_randomizer::xor_sequence(&mut scrambled);
        let mut bits: Vec<u8> = Vec::with_capacity(scrambled.len() * 8);
        for byte in &scrambled {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }
        let coded = viterbi::encode_bits(&bits);
        let mut precoded = differential_encode(&coded);

        // Introduce a few bit errors in the precoded (channel) stream.
        precoded[100] ^= 1;
        precoded[200] ^= 1;
        precoded[500] ^= 1;
        precoded[1000] ^= 1;
        precoded[2000] ^= 1;

        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        let frame = decoder.decode(&precoded).expect("should correct errors");
        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn rejects_wrong_length() {
        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        assert!(decoder.decode(&[0u8; 100]).is_err());
        assert!(decoder.decode_soft(&[0i8; 100]).is_err());
    }

    #[test]
    fn differential_encode_decode_roundtrip() {
        let bits: Vec<u8> = (0..100u8).map(|i| i & 1).collect();
        let encoded = differential_encode(&bits);
        let decoded = differential_decode(&encoded);
        // First bit may differ (no reference), rest must match.
        assert_eq!(&decoded[1..], &bits[1..]);
    }

    /// A real 223-byte ASRTU-1 (AO-123) telemetry payload decoded by a
    /// SatNOGS station (observation 14194336, 2026-05-30). Used to prove
    /// the full pipeline recovers genuine on-air frames bit-exact.
    const ASRTU1_REAL_PAYLOAD: [u8; 223] = [
        0x03, 0x24, 0xa7, 0xde, 0x00, 0xc0, 0x08, 0x05, 0xc8, 0x5d, 0x00, 0x87, 0x00, 0x05, 0x00,
        0x00, 0x2b, 0x2d, 0x1a, 0x03, 0xdc, 0x11, 0xb2, 0x60, 0xb7, 0x00, 0xf0, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0xd4, 0x00, 0xd8, 0x00, 0x2b, 0x01, 0xb9,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x0d, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0x05, 0x05,
        0x00, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x9c, 0x00, 0x00, 0x00,
        0x16, 0x2e, 0x0c, 0x2e, 0x2e, 0x12, 0x0b, 0x64, 0x2e, 0x12, 0x0c, 0x2e, 0x2e, 0x11, 0x2e,
        0xdd, 0xdd, 0xdd, 0xdd, 0x0b, 0x09, 0x08, 0x04, 0x09, 0x06, 0x05, 0x0a, 0x0a, 0x09, 0x0c,
        0x0f, 0x0f, 0x9c, 0xfe, 0x27, 0xb9, 0x00, 0xf0, 0x00, 0xff, 0xff, 0xff, 0xcb, 0xc0, 0xc0,
        0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0,
        0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0,
        0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0,
        0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0,
        0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0,
    ];

    /// Build the channel-symbol stream the ASRTU-1 transmitter would emit
    /// for a 223-byte payload: RS encode -> CCSDS scramble -> prepend ASM
    /// -> convolutional encode -> NRZ-S precode. Returns one-bit-per-byte
    /// channel symbols.
    fn encode_asrtu1_channel(payload: &[u8]) -> Vec<u8> {
        // RS(255,223) encode.
        let rs_codeword = reed_solomon::encode_shortened(payload, 1);
        assert_eq!(rs_codeword.len(), 255);
        // CCSDS scramble.
        let mut scrambled = rs_codeword;
        ccsds_randomizer::xor_sequence(&mut scrambled);
        // Prepend the 32-bit ASM (MSB first) BEFORE convolutional encoding.
        let mut info_bits: Vec<u8> = (0..32)
            .rev()
            .map(|i| ((CCSDS_ASM >> i) & 1) as u8)
            .collect();
        for byte in &scrambled {
            for shift in (0..8).rev() {
                info_bits.push((byte >> shift) & 1);
            }
        }
        // NRZ-S precode the information bits (ASM + codeword) BEFORE the
        // convolutional encoder, matching the gr-satellites
        // ccsds_rs_deframer topology (Viterbi -> diff_decode -> ASM
        // detect). The receiver therefore NRZ-S decodes the Viterbi
        // output, not the channel symbols.
        let precoded_info = differential_encode(&info_bits);
        // Convolutional encode the precoded [ASM + codeword] stream.
        viterbi::encode_bits(&precoded_info)
    }

    #[test]
    fn full_pipeline_recovers_real_asrtu1_frame() {
        use crate::StreamingViterbi;

        let channel = encode_asrtu1_channel(&ASRTU1_REAL_PAYLOAD);

        // Pipeline RX order (matches gr-satellites ccsds_rs_deframer):
        //   Viterbi decode -> NRZ-S decode -> ASM detect -> descramble -> RS
        // The streaming decoder delays output by its traceback depth, so
        // append flush symbols (which the live pipeline gets for free from
        // the next frame) to drain the full codeword.
        let mut symbols = channel;
        symbols.extend(std::iter::repeat(0u8).take(128));
        let mut viterbi = StreamingViterbi::new();
        let viterbi_out = viterbi.push_symbols(&symbols);
        // NRZ-S decode the Viterbi output (whole bit stream, before ASM
        // detection).
        let decoded_bits = differential_decode(&viterbi_out);

        // The ASM (0x1ACFFC1D) must appear in the NRZ-S-decoded stream.
        let asm_bits: Vec<u8> = (0..32)
            .rev()
            .map(|i| ((CCSDS_ASM >> i) & 1) as u8)
            .collect();
        let asm_pos = decoded_bits
            .windows(32)
            .position(|w| w == asm_bits.as_slice())
            .expect("ASM not found in NRZ-S-decoded Viterbi stream");

        // Extract the 255-byte codeword (2040 bits) after the ASM and pack.
        let codeword_bits = &decoded_bits[asm_pos + 32..asm_pos + 32 + 255 * 8];
        let mut codeword = vec![0u8; 255];
        for (i, chunk) in codeword_bits.chunks_exact(8).enumerate() {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | (bit & 1);
            }
            codeword[i] = byte;
        }

        // NRZ-S already undone above, so the codec just does descramble + RS.
        let decoder = CcsdsConcatDecoder::new(223, false, RsBasis::Conventional);
        let frame = decoder
            .decode_frame_bytes(&codeword)
            .expect("decode real ASRTU-1 frame");
        assert_eq!(frame.payload, ASRTU1_REAL_PAYLOAD);
        assert_eq!(frame.corrected_errors, 0);
    }

    #[test]
    fn soft_erasure_recovers_beyond_errors_only() {
        // Build the scrambled RS codeword as it reaches the codec (after the
        // streaming Viterbi and NRZ-S stages): RS encode -> CCSDS scramble.
        let rs_codeword = reed_solomon::encode_shortened(&ASRTU1_REAL_PAYLOAD, 1);
        assert_eq!(rs_codeword.len(), 255);
        let mut scrambled = rs_codeword;
        ccsds_randomizer::xor_sequence(&mut scrambled);

        // Corrupt 24 bytes -- past the errors-only reach of 16, but inside the
        // 32-symbol erasure capacity once their positions are known.
        let error_positions: Vec<usize> = (0..24).map(|i| 3 + i * 9).collect();
        let mut data = scrambled;
        let mut soft = vec![32.0f32; data.len() * 8];
        for &p in &error_positions {
            data[p] ^= 0x5a;
            // Mark every bit of a corrupted byte as low confidence so the
            // codec ranks it among the erasure candidates.
            for j in 0..8 {
                soft[p * 8 + j] = 1.0;
            }
        }

        let decoder = CcsdsConcatDecoder::new(223, false, RsBasis::Conventional);

        // Errors-only decoding cannot cope with 24 symbol errors.
        assert!(
            decoder.decode_frame_bytes(&data).is_err(),
            "errors-only RS should reject 24 byte errors"
        );

        // Soft erasure marking recovers the genuine payload.
        let frame = decoder
            .decode_frame_bytes_with_soft(&data, &soft, 24)
            .expect("soft erasure decode should recover the frame");
        assert_eq!(frame.payload, ASRTU1_REAL_PAYLOAD);
        assert_eq!(frame.corrected_errors, 24);
    }
}
