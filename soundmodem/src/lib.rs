//! Safe Rust wrapper over the vendored HIT LilacSat soundmodem C decoder.
//!
//! The C core ([`csrc/sm_api.h`]) demodulates ASRTU-1 (AO-123) 9k6 BPSK from
//! 48 kHz mono 12 kHz-IF `i16` audio and yields RS(255,223) CCSDS frames. This
//! module owns the opaque decoder handle and presents a small safe API; the
//! front-end that converts arbitrary IQ/FM input into the required 48 kHz IF
//! stream lives in the runtime crate.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::os::raw::{c_char, c_int};

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
    fn tm_decode_frames(
        frames: *const u8,
        nframes: usize,
        out: *mut c_char,
        cap: usize,
        color: c_int,
        full: c_int,
    ) -> c_int;
    fn tm_decoder_open() -> *mut TmDecoder;
    fn tm_decoder_close(dec: *mut TmDecoder);
    fn tm_decoder_feed(
        dec: *mut TmDecoder,
        frame: *const u8,
        out: *mut c_char,
        cap: usize,
        color: c_int,
        full: c_int,
    ) -> c_int;
    fn tm_decoder_feed_fields(
        dec: *mut TmDecoder,
        frame: *const u8,
        out: *mut TmFieldOut,
        cap: usize,
    ) -> c_int;
}

/// C ABI mirror of `struct tm_field_out`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TmFieldOut {
    packet: *const c_char,
    cn: *const c_char,
    en: *const c_char,
    unit: *const c_char,
    value: [c_char; 64],
    kind: u8,
}

/// The value class of a [`DecodedField`], hinting how to type its `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// A scaled or raw number (`value` parses as a decimal).
    Number,
    /// An enumeration description or other textual value.
    Enum,
    /// A raw hex string (e.g. `0xDEADBEEF`).
    Hex,
}

/// One decoded telemetry field extracted from a frame.
#[derive(Debug, Clone)]
pub struct DecodedField {
    /// Packet code the field belongs to, e.g. `"SAT-S0"`.
    pub packet: String,
    /// Field name (Chinese).
    pub cn: String,
    /// Field name (English), may be empty.
    pub en: String,
    /// Engineering unit, empty when unit-less.
    pub unit: String,
    /// Formatted value (scaled number, enum description or hex string).
    pub value: String,
    /// How to interpret `value`.
    pub kind: FieldKind,
}

/// Opaque handle to a stateful telemetry decoder; never dereferenced in Rust.
#[repr(C)]
struct TmDecoder {
    _private: [u8; 0],
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

/// Decode ASRTU-1 telemetry packets from a run of decoded CCSDS frames into a
/// human-readable report. Frames must be in receive order: the 223-byte frames
/// carry a KISS-framed packet stream, and a packet body may straddle two
/// frames. Frames whose length is not [`FRAME_LEN`] are skipped. When `color`
/// is true the report carries ANSI colour (field-coloured hex dump and value
/// highlighting); pass false when the sink is not a terminal. When `full` is
/// true each packet is rendered in detail (hex dump + fields); when false only
/// a one-line summary per packet is produced.
pub fn decode_telemetry(frames: &[Frame], color: bool, full: bool) -> String {
    let mut flat = Vec::with_capacity(frames.len() * FRAME_LEN);
    for f in frames {
        if f.data.len() == FRAME_LEN {
            flat.extend_from_slice(&f.data);
        }
    }
    let nframes = flat.len() / FRAME_LEN;
    if nframes == 0 {
        return String::new();
    }
    // The C side renders a bounded amount of text per field plus a coloured hex
    // dump; size the output buffer generously per frame and let C truncate.
    let cap = nframes * 48 * 1024 + 8192;
    let mut out = vec![0u8; cap];
    // SAFETY: flat holds nframes*FRAME_LEN bytes; out has cap bytes; the C side
    // writes a NUL-terminated string within cap.
    unsafe {
        tm_decode_frames(
            flat.as_ptr(),
            nframes,
            out.as_mut_ptr() as *mut c_char,
            cap,
            c_int::from(color),
            c_int::from(full),
        );
    }
    let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
    String::from_utf8_lossy(&out[..end]).into_owned()
}

/// Stateful telemetry decoder: feed frames one at a time (in receive order)
/// and render the telemetry each frame completes, so a caller printing frames
/// can show the decode directly beneath each one. KISS reassembly state (for
/// packets that straddle frames) is carried across feeds.
pub struct TelemetryDecoder {
    raw: *mut TmDecoder,
}

impl TelemetryDecoder {
    /// Allocate a decoder, or `None` on allocation failure.
    pub fn new() -> Option<Self> {
        // SAFETY: tm_decoder_open returns an owned pointer or null.
        let raw = unsafe { tm_decoder_open() };
        if raw.is_null() {
            None
        } else {
            Some(Self { raw })
        }
    }

    /// Feed one 223-byte frame and return the rendered telemetry for the
    /// packets that completed within it (empty when none). `color` emits ANSI;
    /// `full` selects the detailed render over the one-line summary. Frames not
    /// exactly [`FRAME_LEN`] bytes are ignored.
    pub fn feed(&mut self, frame: &[u8], color: bool, full: bool) -> String {
        if frame.len() != FRAME_LEN {
            return String::new();
        }
        // One frame completes at most a few packets; size the buffer for the
        // detailed render and let C truncate.
        let mut out = vec![0u8; 256 * 1024];
        let cap = out.len();
        // SAFETY: raw is a valid handle; frame is FRAME_LEN bytes; out has cap
        // bytes and the C side writes a NUL-terminated string within it.
        unsafe {
            tm_decoder_feed(
                self.raw,
                frame.as_ptr(),
                out.as_mut_ptr() as *mut c_char,
                cap,
                c_int::from(color),
                c_int::from(full),
            );
        }
        let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
        String::from_utf8_lossy(&out[..end]).into_owned()
    }

    /// Feed one 223-byte frame and return the structured fields of every packet
    /// completed within it (empty when none). Frames not exactly [`FRAME_LEN`]
    /// bytes are ignored.
    pub fn feed_fields(&mut self, frame: &[u8]) -> Vec<DecodedField> {
        if frame.len() != FRAME_LEN {
            return Vec::new();
        }
        // A frame completes at most a couple of packets (≤144 fields each).
        const CAP: usize = 512;
        // SAFETY: TmFieldOut is plain data; a zeroed instance (null pointers,
        // empty value) is a valid placeholder the C side overwrites.
        let mut out = vec![unsafe { std::mem::zeroed::<TmFieldOut>() }; CAP];
        // SAFETY: raw is valid; frame is FRAME_LEN bytes; out has CAP entries.
        let n = unsafe { tm_decoder_feed_fields(self.raw, frame.as_ptr(), out.as_mut_ptr(), CAP) };
        let n = usize::try_from(n).unwrap_or(0).min(CAP);
        out[..n]
            .iter()
            .map(|f| DecodedField {
                packet: cstr(f.packet),
                cn: cstr(f.cn),
                en: cstr(f.en),
                unit: cstr(f.unit),
                value: cstr_array(&f.value),
                kind: match f.kind {
                    1 => FieldKind::Enum,
                    2 => FieldKind::Hex,
                    _ => FieldKind::Number,
                },
            })
            .collect()
    }
}

/// Read a NUL-terminated C string pointer into an owned `String` (empty on
/// null).
fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the pointer references a NUL-terminated static string in the C
    // tables, valid for the program's lifetime.
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// Read a fixed C `char` array up to its first NUL into an owned `String`.
fn cstr_array(a: &[c_char]) -> String {
    let bytes: &[u8] = unsafe { &*(a as *const [c_char] as *const [u8]) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl Drop for TelemetryDecoder {
    fn drop(&mut self) {
        // SAFETY: raw came from tm_decoder_open and is freed exactly once.
        unsafe { tm_decoder_close(self.raw) }
    }
}

// The decoder owns its reassembly state behind the handle and shares nothing.
unsafe impl Send for TelemetryDecoder {}

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
