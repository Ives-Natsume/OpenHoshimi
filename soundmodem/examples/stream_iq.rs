//! Stream a 48 kHz stereo 16-bit IQ WAV (I = left, Q = right) through the C
//! decoder's streaming IQ entry, in small chunks, exactly as the live decoder
//! does. Dev tool to confirm a file decodes via `feed_iq` (not just the offline
//! batch front-end); not a shipped binary.
//!
//! Usage: cargo run --example stream_iq -- in48k_iq.wav

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
    let path = env::args().nth(1).expect("usage: stream_iq <in48k_iq.wav>");
    let bytes = fs::read(&path).expect("read wav");
    assert!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV"
    );

    let mut pos = 12;
    let (mut channels, mut rate, mut bits) = (0u16, 0u32, 0u16);
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
        channels == 2 && bits == 16,
        "need stereo 16-bit IQ (got {channels}ch {bits}b)"
    );
    eprintln!(
        "wav: {rate} Hz stereo 16-bit, {} IQ samples",
        data.len() / 4
    );

    let mut dec = Decoder::new().expect("decoder alloc");
    let mut frames = 0usize;
    // Feed in 4096-sample chunks to mirror the live capture cadence.
    for chunk in data.chunks(4096 * 4) {
        let ilv: Vec<f32> = chunk
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect();
        frames += dec.feed_iq_collect(&ilv).len();
    }
    dec.flush();
    while dec.poll().is_some() {
        frames += 1;
    }
    println!("streaming feed_iq decoded {frames} frame(s)");
}
