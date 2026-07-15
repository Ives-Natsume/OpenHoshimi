//! Decode a 48 kHz mono 16-bit PCM WAV (12 kHz IF) through the C soundmodem
//! backend. Dev tool used to confirm the Rust FFI reproduces the standalone
//! C decode; not a shipped binary.
//!
//! Usage: cargo run --example decode_wav -- in48k_mono.wav

use std::env;
use std::fs;

use openhoshimi_soundmodem::Decoder;

fn rd_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn rd_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: decode_wav <in48k_mono.wav>");
    let bytes = fs::read(&path).expect("read wav");
    assert!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV"
    );

    let mut pos = 12;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = rd_u32(&bytes[pos + 4..]) as usize;
        let body = pos + 8;
        match id {
            b"fmt " => {
                channels = rd_u16(&bytes[body + 2..]);
                rate = rd_u32(&bytes[body + 4..]);
                bits = rd_u16(&bytes[body + 14..]);
            }
            b"data" => data = &bytes[body..(body + sz).min(bytes.len())],
            _ => {}
        }
        pos = body + sz + (sz & 1);
    }
    assert!(
        channels == 1 && bits == 16,
        "need mono 16-bit (got {channels}ch {bits}b)"
    );
    eprintln!("wav: {rate} Hz mono 16-bit, {} samples", data.len() / 2);

    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut dec = Decoder::new().expect("decoder");
    let mut total = 0usize;
    for block in samples.chunks(1920) {
        for f in dec.feed_collect(block) {
            total += 1;
            eprintln!(
                "#{total:03} corrected={:3}  {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                f.corrected, f.data[0], f.data[1], f.data[2], f.data[3], f.data[4], f.data[5],
                f.data[6], f.data[7],
            );
        }
    }
    eprintln!("TOTAL FRAMES: {total}");
}
