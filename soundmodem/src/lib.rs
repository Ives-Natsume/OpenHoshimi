//! Safe Rust wrapper over the vendored HIT LilacSat soundmodem C decoder.
//!
//! The C core ([`csrc/sm_api.h`]) demodulates ASRTU-1 (AO-123) 9k6 BPSK from
//! 48 kHz mono 12 kHz-IF `i16` audio and yields RS(255,223) CCSDS frames. This
//! module owns the opaque decoder handle and presents a small safe API; the
//! front-end that converts arbitrary IQ/FM input into the required 48 kHz IF
//! stream lives in the runtime crate.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::os::raw::c_int;

/// Payload length of one decoded CCSDS frame (RS(255,223) message).
pub const FRAME_LEN: usize = 223;

/// Input sample rate the decoder expects (Hz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Intermediate frequency the decoder expects the signal to sit at (Hz).
pub const IF_FREQ: u32 = 12_000;

/// Opaque C handle; never dereferenced on the Rust side.
#[repr(C)]
struct SmDecoder {
    _private: [u8; 0],
}

extern "C" {
    fn sm_decoder_new() -> *mut SmDecoder;
    fn sm_decoder_free(dec: *mut SmDecoder);
    fn sm_decoder_feed(dec: *mut SmDecoder, samples: *const i16, n: usize);
    fn sm_decoder_feed_iq(dec: *mut SmDecoder, iq: *const f32, n: usize);
    fn sm_decoder_flush(dec: *mut SmDecoder);
    fn sm_decoder_poll(dec: *mut SmDecoder, out: *mut u8, corrected: *mut c_int) -> c_int;
}

/// A decoded CCSDS frame and its Reed-Solomon correction count.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The 223-byte RS-decoded payload.
    pub data: Vec<u8>,
    /// Number of bytes the outer RS code corrected (0 = clean).
    pub corrected: i32,
}

/// Owning handle to a soundmodem decoder instance.
pub struct Decoder {
    raw: *mut SmDecoder,
}

impl Decoder {
    /// Allocate a decoder, or `None` if the C allocation failed.
    pub fn new() -> Option<Self> {
        // SAFETY: sm_decoder_new returns an owned pointer or null.
        let raw = unsafe { sm_decoder_new() };
        if raw.is_null() {
            None
        } else {
            Some(Self { raw })
        }
    }

    /// Feed `i16` samples at [`SAMPLE_RATE`] (mono, [`IF_FREQ`] IF). Decoded
    /// frames queue internally; drain them with [`Decoder::poll`].
    pub fn feed(&mut self, samples: &[i16]) {
        // SAFETY: raw is a valid handle; samples/len describe a live slice.
        unsafe { sm_decoder_feed(self.raw, samples.as_ptr(), samples.len()) }
    }

    /// Pop one decoded frame, or `None` when the internal queue is empty.
    pub fn poll(&mut self) -> Option<Frame> {
        let mut buf = [0u8; FRAME_LEN];
        let mut corrected: c_int = 0;
        // SAFETY: raw is valid; buf is FRAME_LEN bytes as the C side expects.
        let got = unsafe { sm_decoder_poll(self.raw, buf.as_mut_ptr(), &mut corrected) };
        if got == 1 {
            Some(Frame {
                data: buf.to_vec(),
                corrected: corrected as i32,
            })
        } else {
            None
        }
    }

    /// Feed complex baseband samples at [`SAMPLE_RATE`] as interleaved `(re, im)`
    /// float pairs (so `iq.len()` is `2 * sample_count`). The signal must sit
    /// near 0 Hz. This is the lossless IQ path: no real projection, no int16
    /// quantisation. Decoded frames queue internally.
    pub fn feed_iq(&mut self, iq: &[f32]) {
        // SAFETY: raw is valid; iq/len describe a live interleaved slice.
        unsafe { sm_decoder_feed_iq(self.raw, iq.as_ptr(), iq.len() / 2) }
    }

    /// Feed samples and collect every frame that becomes available.
    pub fn feed_collect(&mut self, samples: &[i16]) -> Vec<Frame> {
        self.feed(samples);
        self.drain()
    }

    /// Feed interleaved IQ and collect every frame that becomes available.
    pub fn feed_iq_collect(&mut self, iq: &[f32]) -> Vec<Frame> {
        self.feed_iq(iq);
        self.drain()
    }

    /// Process the buffered trailing partial block at end of stream.
    pub fn flush(&mut self) {
        // SAFETY: raw is a valid decoder handle.
        unsafe { sm_decoder_flush(self.raw) }
    }

    fn drain(&mut self) -> Vec<Frame> {
        let mut frames = Vec::new();
        while let Some(f) = self.poll() {
            frames.push(f);
        }
        frames
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: raw came from sm_decoder_new and is freed exactly once.
        unsafe { sm_decoder_free(self.raw) }
    }
}

// The decoder owns all its state behind the handle and shares nothing, so it
// is safe to move between threads (but not to share without synchronisation).
unsafe impl Send for Decoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_and_drops() {
        let dec = Decoder::new().expect("decoder alloc");
        drop(dec);
    }

    #[test]
    fn silence_decodes_nothing() {
        let mut dec = Decoder::new().expect("decoder alloc");
        let silence = vec![0i16; SAMPLE_RATE as usize];
        let frames = dec.feed_collect(&silence);
        assert!(frames.is_empty());
    }
}
