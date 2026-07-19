//! Offline file decoding tool for ClosedHoshimi.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::{
    build_image_reassembler, Ax100Mode, Ax25Frame, GeoscanFrame, ImageDecoder, ImageReassembler,
};
use openhoshimi_core::satellite::load_satellite;
use openhoshimi_core::satellite::ModemDef;
use openhoshimi_core::{Frame, InputSource, IoError, IqSample, IqSource};
use openhoshimi_dsp::estimate_audio_carrier;
use openhoshimi_dsp::linear::open_symbol_dump;
use openhoshimi_dsp::ChannelFilter;
use openhoshimi_io::{
    detect_audio_mode_auto, open_audio_source, read_audio_prefix, read_iq_prefix,
    AudioMode as IoAudioMode, MonoIqSource, WavIqSource,
};
use openhoshimi_runtime::pipeline::{
    blind_linear_iq_channel_setup, estimate_cpm_iq_frequency_offset_hz,
    estimate_iq_frequency_offset_hz, format_call, format_hex, format_timestamp, frame_type_label,
    infer_tuning_offset_hz, is_ao40_fec_downlink, prepare_linear_iq_setup_scored, select_downlink,
    BitPipeline, DecodedFrame, InputKind, SoftAo40Pipeline,
};
use openhoshimi_runtime::soundmodem_frontend::StreamResampler;
use openhoshimi_telemetry::SchemaParser;
use rustfft::num_complex::Complex;

mod live;

const READ_CHUNK: usize = 4096;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Satellite TOML. Required except in `--live` mode.
    satellite_toml: Option<PathBuf>,
    /// Input WAV. Required except in `--live` mode.
    input_wav: Option<PathBuf>,
    /// How to interpret the input WAV: `auto` (sniff the file, default),
    /// `iq` (force stereo I/Q), `fm` (FM-discriminator mono audio), or
    /// `ssb` (single-sideband mono audio with `--audio-carrier-hz`
    /// giving the signal centre frequency inside the audio band).
    #[arg(long, value_enum, default_value_t = AudioMode::Auto)]
    audio_mode: AudioMode,
    /// Audio-band carrier frequency in Hz for `--audio-mode ssb`.
    /// Ignored in any other mode.
    #[arg(long)]
    audio_carrier_hz: Option<f32>,
    /// If set, run every decoded payload through the satellite's
    /// image reassembler (when one is configured in the TOML) and
    /// write each completed image to `<DIR>/openhoshimi_image_NN.{jpg,bin}`
    /// when decoding finishes. Useful for offline regression-testing
    /// the image pipeline without launching the GUI.
    #[arg(long)]
    dump_images: Option<PathBuf>,
    /// Decode ASRTU-1 (AO-123) through the vendored soundmodem C backend
    /// instead of the pure-Rust pipeline. Reads the input as stereo IQ,
    /// re-tunes per window to follow Doppler, and feeds the C decoder.
    #[arg(long)]
    soundmodem: bool,
    /// Feed a 48 kHz mono 12 kHz-IF WAV straight into the soundmodem C decoder,
    /// bypassing the IQ front-end. This is the official soundmodem input format
    /// (e.g. produced by SDR#); useful as a reference and to isolate front-end
    /// versus decoder behaviour.
    #[arg(long)]
    soundmodem_raw: bool,
    /// Where to write the full ASRTU-1 telemetry report (every packet, hex
    /// dump plus all fields). Defaults to `<input>.telemetry.txt`. The console
    /// shows only a one-line-per-packet summary to stay readable.
    #[arg(long)]
    tlm_out: Option<PathBuf>,
    /// Live-decode ASRTU-1 from a soundcard through the streaming soundmodem
    /// decoder, with an interactive terminal UI (device picker, spectrum and
    /// live decode status). Takes no input file.
    #[arg(long)]
    live: bool,
    /// Soundcard input device name for `--live`. When omitted, `--live` shows
    /// an interactive picker (arrow keys, Enter to select).
    #[arg(long)]
    soundcard_device: Option<String>,
    /// Feed `--live` from a 48 kHz IQ WAV instead of a soundcard, so the live
    /// path can be exercised without a radio. Pair with a file produced by
    /// `--export-iq`.
    #[arg(long)]
    live_file: Option<PathBuf>,
    /// With `--soundmodem`, write the de-rotated 48 kHz baseband IQ (the signal
    /// the decoder actually sees) to this WAV instead of decoding. The result
    /// is a known-good input for `--live --live-file`.
    #[arg(long)]
    export_iq: Option<PathBuf>,
    /// Treat the `--live` soundcard as a mono FM-audio feed rather than stereo
    /// IQ (e.g. SDR# FM output through a virtual cable).
    #[arg(long)]
    live_mono: bool,
    /// Stop `--live` automatically after this many seconds. Omit to run until
    /// the user quits (q / Esc).
    #[arg(long)]
    live_seconds: Option<u64>,
    /// Playback speed multiplier for `--live-file` (e.g. `4` to play a 5-minute
    /// recording in ~75 s). Ignored for a soundcard, which is always real time.
    #[arg(long, default_value_t = 1.0)]
    live_speed: f32,
}

/// How the input file should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AudioMode {
    /// Detect IQ vs mono audio from the file and the filename hint.
    Auto,
    /// Force stereo I/Q interpretation (errors if the file is mono).
    Iq,
    /// Force FM-discriminator mono audio (instantaneous-frequency waveform).
    Fm,
    /// Force SSB mono audio. Requires `--audio-carrier-hz`.
    Ssb,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("decode_file: {err}");
            ExitCode::from(1)
        }
    }
}

/// Decode ASRTU-1 via the vendored soundmodem C backend: slurp the input as
/// stereo IQ and hand it to the windowed front-end, printing each frame.
fn run_soundmodem(
    input: &Path,
    tlm_out: Option<&Path>,
    export_iq: Option<&Path>,
) -> Result<(), String> {
    let mut source =
        WavIqSource::open(input).map_err(|err| format!("failed to open IQ WAV: {err}"))?;
    let fs = source.sample_rate();
    eprintln!("decode_file: soundmodem backend, IQ input at {fs} Hz");

    let mut iq: Vec<IqSample> = Vec::new();
    let mut buf = vec![IqSample { i: 0.0, q: 0.0 }; 65536];
    let src: &mut dyn IqSource = &mut source;
    loop {
        match src.read_samples(&mut buf) {
            Ok(0) => break,
            Ok(n) => iq.extend_from_slice(&buf[..n]),
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("IQ read error: {err}")),
        }
    }
    eprintln!(
        "decode_file: read {} IQ samples ({:.1}s)",
        iq.len(),
        iq.len() as f32 / fs as f32
    );

    if let Some(path) = export_iq {
        let baseband = openhoshimi_runtime::soundmodem_frontend::derotate_iq_48k(&iq, fs);
        if baseband.is_empty() {
            return Err("export-iq: no Doppler track found to de-rotate".to_string());
        }
        write_iq_wav(path, &baseband, openhoshimi_soundmodem::SAMPLE_RATE)?;
        eprintln!(
            "decode_file: wrote {} de-rotated IQ samples ({:.1}s @ {} Hz) to {}",
            baseband.len(),
            baseband.len() as f32 / openhoshimi_soundmodem::SAMPLE_RATE as f32,
            openhoshimi_soundmodem::SAMPLE_RATE,
            path.display()
        );
        return Ok(());
    }

    let frames = openhoshimi_runtime::soundmodem_frontend::decode_iq(&iq, fs);
    let tlm_path = tlm_out
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("telemetry.txt"));
    print_sm_frames(&frames, &tlm_path);
    Ok(())
}

/// Write interleaved IQ as a 16-bit stereo WAV (I = left, Q = right), peak
/// normalised to avoid clipping. A minimal writer: the canonical 44-byte PCM
/// header followed by the samples.
fn write_iq_wav(path: &Path, iq: &[IqSample], sample_rate: u32) -> Result<(), String> {
    let peak = iq
        .iter()
        .flat_map(|s| [s.i.abs(), s.q.abs()])
        .fold(0.0f32, f32::max)
        .max(1e-9);
    let scale = 30000.0 / peak;
    let mut pcm: Vec<u8> = Vec::with_capacity(iq.len() * 4);
    for s in iq {
        pcm.extend_from_slice(&((s.i * scale) as i16).to_le_bytes());
        pcm.extend_from_slice(&((s.q * scale) as i16).to_le_bytes());
    }
    let data_len = pcm.len() as u32;
    let byte_rate = sample_rate * 4;
    let mut out: Vec<u8> = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes()); // stereo
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&pcm);
    std::fs::write(path, out).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

/// Print soundmodem frames: a header line per frame followed by a classic
/// `hexdump -C`-style body (offset, two 8-byte hex groups, ASCII sidebar),
/// coloured when stdout is a terminal and left plain when piped.
pub(crate) fn print_sm_frames(frames: &[openhoshimi_soundmodem::Frame], tlm_path: &Path) {
    let color = std::io::stdout().is_terminal();
    // Stateful decoder so each frame's telemetry prints directly beneath it
    // (packets straddle frames, so it carries reassembly state across feeds).
    let mut tlm = openhoshimi_soundmodem::TelemetryDecoder::new();
    for (i, f) in frames.iter().enumerate() {
        let seq = f.data.get(2).copied().unwrap_or(0);
        println!(
            "#{:03}  seq={seq:02X}  corrected={}  {} bytes",
            i + 1,
            f.corrected,
            f.data.len()
        );
        hexdump(&f.data, color);
        if let Some(dec) = tlm.as_mut() {
            print!("{}", dec.feed(&f.data, color, false));
        }
    }
    println!("decode_file: decoded {} frame(s)", frames.len());

    // The full report (every packet, hex dump + all fields) goes to a file;
    // the console kept only the per-frame summaries above.
    let full = openhoshimi_soundmodem::decode_telemetry(frames, false, true);
    if full.is_empty() {
        return;
    }
    match std::fs::write(tlm_path, full) {
        Ok(()) => println!(
            "decode_file: full telemetry saved to {}",
            tlm_path.display()
        ),
        Err(err) => eprintln!("decode_file: could not write {}: {err}", tlm_path.display()),
    }
}

/// Render `data` as a coloured `hexdump -C`-style block. Zero bytes are dim,
/// printable ASCII green, other bytes cyan; the ASCII column mirrors this.
fn hexdump(data: &[u8], color: bool) {
    const DIM: &str = "\x1b[38;5;242m";
    const GRN: &str = "\x1b[32m";
    const CYN: &str = "\x1b[36m";
    const RST: &str = "\x1b[0m";
    let printable = |b: u8| (0x20..=0x7e).contains(&b);

    for (row, chunk) in data.chunks(16).enumerate() {
        let mut line = String::new();
        let off = format!("{:04x}", row * 16);
        if color {
            line.push_str(&format!("  {DIM}{off}{RST}  "));
        } else {
            line.push_str(&format!("  {off}  "));
        }
        for j in 0..16 {
            if j == 8 {
                line.push(' ');
            }
            match chunk.get(j) {
                Some(&b) => {
                    let cell = format!("{b:02x} ");
                    if !color {
                        line.push_str(&cell);
                    } else if b == 0 {
                        line.push_str(&format!("{DIM}{cell}{RST}"));
                    } else if printable(b) {
                        line.push_str(&format!("{GRN}{cell}{RST}"));
                    } else {
                        line.push_str(&format!("{CYN}{cell}{RST}"));
                    }
                }
                None => line.push_str("   "),
            }
        }
        line.push_str(" |");
        for &b in chunk {
            if printable(b) {
                let ch = (b as char).to_string();
                if color {
                    line.push_str(&format!("{GRN}{ch}{RST}"));
                } else {
                    line.push_str(&ch);
                }
            } else if color {
                line.push_str(&format!("{DIM}.{RST}"));
            } else {
                line.push('.');
            }
        }
        line.push('|');
        println!("{line}");
    }
}

/// Feed a 48 kHz mono 12 kHz-IF WAV straight into the soundmodem C decoder,
/// bypassing the IQ front-end (the official soundmodem input format).
fn run_soundmodem_raw(input: &Path, tlm_out: Option<&Path>) -> Result<(), String> {
    let mut source = open_audio_source(input)?;
    let fs = source.sample_rate();
    // The decoder's symbol timing is fixed at 48 kHz; resample any other rate
    // so recordings from soundcards/SDRs at 44.1 kHz, 96 kHz, etc. still decode.
    let mut resampler = (fs != openhoshimi_soundmodem::SAMPLE_RATE)
        .then(|| StreamResampler::new(fs, openhoshimi_soundmodem::SAMPLE_RATE));
    if resampler.is_some() {
        eprintln!(
            "decode_file: resampling {fs} Hz -> {} Hz for the decoder",
            openhoshimi_soundmodem::SAMPLE_RATE
        );
    }
    let mut dec = openhoshimi_soundmodem::Decoder::new().ok_or("decoder alloc failed")?;
    let mut buf = [0.0f32; 4096];
    let mut frames = Vec::new();
    loop {
        match source.read_samples(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let pcm: Vec<i16> = match resampler.as_mut() {
                    Some(rs) => {
                        let cin: Vec<Complex<f32>> =
                            buf[..n].iter().map(|&x| Complex::new(x, 0.0)).collect();
                        let mut out = Vec::new();
                        rs.process(&cin, &mut out);
                        out.iter().map(|c| (c.re * 32767.0) as i16).collect()
                    }
                    None => buf[..n].iter().map(|&x| (x * 32767.0) as i16).collect(),
                };
                frames.extend(dec.feed_collect(&pcm));
            }
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("audio read error: {err}")),
        }
    }
    dec.flush();
    while let Some(f) = dec.poll() {
        frames.push(f);
    }
    let tlm_path = tlm_out
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("telemetry.txt"));
    print_sm_frames(&frames, &tlm_path);
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.live {
        return live::run_live(live::LiveOpts {
            device: args.soundcard_device.clone(),
            mono: args.live_mono,
            seconds: args.live_seconds,
            tlm_out: args.tlm_out.clone(),
            file: args.live_file.clone(),
            speed: args.live_speed,
        });
    }
    let input_wav = args
        .input_wav
        .as_deref()
        .ok_or("an input WAV path is required (or use --live)")?;
    if args.soundmodem_raw {
        return run_soundmodem_raw(input_wav, args.tlm_out.as_deref());
    }
    if args.soundmodem {
        return run_soundmodem(
            input_wav,
            args.tlm_out.as_deref(),
            args.export_iq.as_deref(),
        );
    }
    if let Some(path) = std::env::var_os("OPENHOSHIMI_SYMBOL_DUMP") {
        open_symbol_dump(Path::new(&path))
            .map_err(|err| format!("failed to open symbol dump file: {err}"))?;
    }
    let satellite_toml = args
        .satellite_toml
        .as_deref()
        .ok_or("a satellite TOML path is required")?;
    let satellite = load_satellite(satellite_toml)
        .map_err(|err| format!("failed to load satellite definition: {err}"))?;
    let selected = select_downlink(&satellite)?;
    let telemetry = selected
        .downlink
        .telemetry_schema
        .as_ref()
        .and_then(|name| satellite.telemetry.get(name))
        .map(SchemaParser::new);
    let mut image_reassembler: Option<Box<dyn ImageReassembler>> = if args.dump_images.is_some() {
        match selected.downlink.image.as_ref() {
            Some(def) => match build_image_reassembler(def) {
                Ok(r) => Some(r),
                Err(err) => {
                    eprintln!(
                        "decode_file: --dump-images requested but image config is invalid: {err}"
                    );
                    None
                }
            },
            None => {
                eprintln!(
                        "decode_file: --dump-images requested but the active downlink has no [downlink.image] block"
                    );
                None
            }
        }
    } else {
        None
    };

    let tuning_offset_hz =
        infer_tuning_offset_hz(input_wav, selected.downlink.freq_hz).unwrap_or(0) as f32;
    let mut runner = match selected.input_kind {
        InputKind::Audio => {
            let source: Box<dyn InputSource> = open_audio_source(input_wav)?;
            let sample_rate = source.sample_rate();
            let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
            pipeline.configure_demodulator(selected.downlink, sample_rate, 0.0)?;
            PipelineRunner::Audio { source, pipeline }
        }
        InputKind::FmAudio => {
            let source: Box<dyn InputSource> = open_audio_source(input_wav)?;
            let sample_rate = source.sample_rate();
            let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
            pipeline.configure_fm_audio_demodulator(selected.downlink, sample_rate)?;
            PipelineRunner::FmAudio { source, pipeline }
        }
        InputKind::Iq => {
            let resolved_mode: IoAudioMode = match args.audio_mode {
                AudioMode::Auto => detect_audio_mode_auto(input_wav)?,
                AudioMode::Iq => IoAudioMode::Iq,
                AudioMode::Fm => IoAudioMode::Fm,
                AudioMode::Ssb => IoAudioMode::Ssb,
            };
            match resolved_mode {
                IoAudioMode::Fm => {
                    eprintln!("decode_file: treating mono input as FM-discriminator audio");
                    let source: Box<dyn InputSource> = open_audio_source(input_wav)?;
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
                    pipeline.configure_fm_audio_demodulator(selected.downlink, sample_rate)?;
                    PipelineRunner::FmAudio { source, pipeline }
                }
                IoAudioMode::Ssb => {
                    if is_ao40_fec_downlink(selected.downlink) {
                        return Err(
                            "SSB audio mode does not support AO-40 FEC downlinks".to_string()
                        );
                    }
                    let mut mono: Box<dyn InputSource> = open_audio_source(input_wav)?;
                    let sample_rate = mono.sample_rate();
                    let (carrier_hz, prefix) = match args.audio_carrier_hz {
                        Some(value) => {
                            if !value.is_finite() {
                                return Err(format!(
                                    "--audio-carrier-hz must be finite, got {value}"
                                ));
                            }
                            eprintln!(
                                "decode_file: treating mono input as SSB audio, mixing carrier {value:.1} Hz to baseband (user-supplied)"
                            );
                            (value, Vec::new())
                        }
                        None => {
                            let nyquist = sample_rate as f32 / 2.0;
                            if nyquist <= 400.0 {
                                return Err(format!(
                                    "SSB audio carrier auto-estimate: sample rate {sample_rate} Hz is too low"
                                ));
                            }
                            let prefix_len = sample_rate as usize;
                            let prefix = read_audio_prefix(&mut *mono, prefix_len)?;
                            let estimate = estimate_audio_carrier(
                                &prefix,
                                sample_rate,
                                200.0,
                                nyquist - 200.0,
                            )
                            .ok_or_else(|| {
                                "SSB audio carrier auto-estimate failed: no peak found in audio band; pass --audio-carrier-hz <hz>".to_string()
                            })?;
                            let prefix_samples = prefix.len();
                            eprintln!(
                                "decode_file: treating mono input as SSB audio, mixing carrier {estimate:.1} Hz to baseband (auto-estimated from {prefix_samples} samples)"
                            );
                            (estimate, prefix)
                        }
                    };
                    let source: Box<dyn IqSource> =
                        Box::new(MonoIqSource::with_prefix(mono, prefix));
                    let mut pipeline = BitPipeline::<IqSample>::new(selected.downlink)?;
                    pipeline.configure_demodulator(selected.downlink, sample_rate, carrier_hz)?;
                    PipelineRunner::Iq {
                        source,
                        pipeline,
                        pending: Vec::new(),
                        channel: None,
                        working_sample_rate: sample_rate,
                    }
                }
                IoAudioMode::Iq => {
                    let source = WavIqSource::open(input_wav)
                        .map_err(|err| format!("failed to open WAV as IQ: {err}"))?;
                    build_iq_runner(source, input_wav, selected.downlink, tuning_offset_hz)?
                }
                IoAudioMode::Auto => unreachable!("IoAudioMode::Auto is resolved above"),
            }
        }
    };
    eprintln!(
        "decode_file: using {} ({})",
        selected.downlink.label,
        runner.description()
    );

    // Opt-in offline block decoder: recover CCSDS frames by ASM correlation
    // after the stream ends, catching frames the streaming loops never locked.
    let block_demod = std::env::var_os("OPENHOSHIMI_BLOCK_DEMOD").is_some();
    if block_demod {
        runner.enable_block_demod();
    }

    let mut frame_count = 0usize;
    loop {
        let batch = runner.read_next_batch()?;
        let batch = match batch {
            Some(batch) => batch,
            None => break,
        };

        for entry in batch {
            frame_count += 1;
            let timestamp = runner.timestamp();
            match entry {
                BatchEntry::Decoded { decoded, raw } => {
                    print_decoded_frame(
                        frame_count,
                        timestamp,
                        decoded,
                        &raw,
                        telemetry.as_ref(),
                        &mut image_reassembler,
                    );
                }
                BatchEntry::Pending(mut frame) => {
                    frame.satellite_id = satellite.satellite.norad_id;
                    match runner.decode_frame(&frame) {
                        Ok(decoded) => print_decoded_frame(
                            frame_count,
                            timestamp,
                            decoded,
                            &frame.raw,
                            telemetry.as_ref(),
                            &mut image_reassembler,
                        ),
                        Err(err) => {
                            eprintln!(
                                "decode_file: failed to decode frame #{frame_count:03}: {err}"
                            );
                            let preview: Vec<String> = frame
                                .raw
                                .iter()
                                .take(16)
                                .map(|b| format!("{b:02X}"))
                                .collect();
                            eprintln!(
                                "decode_file:   first {} bytes: {}",
                                preview.len(),
                                preview.join(" ")
                            );
                        }
                    }
                }
                BatchEntry::DecodeError(err) => {
                    eprintln!("decode_file: failed to decode frame #{frame_count:03}: {err}");
                }
            }
        }
    }

    // Offline block decoder pass: recover frames by ASM correlation over the
    // whole run, decoded through the same codec (RS + erasures + sanity gate)
    // as the streaming frames, so false correlation peaks are rejected.
    if block_demod {
        let block_frames = runner.finalize();
        eprintln!(
            "decode_file: block decoder recovered {} candidate codeword(s)",
            block_frames.len()
        );
        for mut frame in block_frames {
            frame.satellite_id = satellite.satellite.norad_id;
            frame_count += 1;
            let timestamp = runner.timestamp();
            match runner.decode_frame(&frame) {
                Ok(decoded) => print_decoded_frame(
                    frame_count,
                    timestamp,
                    decoded,
                    &frame.raw,
                    telemetry.as_ref(),
                    &mut image_reassembler,
                ),
                Err(err) => {
                    eprintln!("decode_file: block frame #{frame_count:03} failed: {err}");
                }
            }
        }
    }

    if let Some(err) = runner.last_framer_error() {
        eprintln!("decode_file: last discarded HDLC frame: {err}");
        if let Some(bytes) = runner.last_failed_bytes() {
            eprintln!(
                "decode_file: last discarded HDLC bytes ({} bytes): {}",
                bytes.len(),
                bytes
                    .iter()
                    .map(|b| format!("{b:02X}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        if let Some(longest) = runner.longest_failed_bytes() {
            let same_as_last = runner
                .last_failed_bytes()
                .map(|last| last == longest)
                .unwrap_or(false);
            if !same_as_last {
                eprintln!(
                    "decode_file: longest discarded HDLC bytes ({} bytes): {}",
                    longest.len(),
                    longest
                        .iter()
                        .map(|b| format!("{b:02X}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }
    }
    if let Some(distance) = runner.best_sync_distance() {
        eprintln!("decode_file: run summary: frames={frame_count} best_sync_distance={distance}");
    } else if frame_count == 0 {
        eprintln!("decode_file: no frames decoded");
    }
    if let Some(history) = runner.distance_history() {
        let formatted: String = history
            .iter()
            .map(|entry| match entry {
                Some(distance) => format!("{distance:2}"),
                None => " -".to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "decode_file: ao40 sync distance per second ({} entries): {formatted}",
            history.len()
        );
    }

    if let (Some(dir), Some(r)) = (args.dump_images.as_ref(), image_reassembler.as_ref()) {
        let snap = r.snapshot();
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let ext = match snap.decoder {
            ImageDecoder::Jpeg => "jpg",
            ImageDecoder::Raw => "bin",
        };
        for stream in &snap.images {
            let path = dir.join(format!("openhoshimi_image_{:02}.{ext}", stream.image_idx));
            std::fs::write(&path, &stream.bytes)
                .map_err(|e| format!("write {}: {e}", path.display()))?;
            eprintln!(
                "decode_file: wrote {} ({} bytes, {}/{} chunks)",
                path.display(),
                stream.bytes.len(),
                stream.received_count,
                stream.total_chunks,
            );
        }
        eprintln!(
            "decode_file: dumped {} image(s) to {}",
            snap.images.len(),
            dir.display()
        );
    }

    Ok(())
}

enum PipelineRunner {
    Audio {
        source: Box<dyn InputSource>,
        pipeline: BitPipeline<f32>,
    },
    FmAudio {
        source: Box<dyn InputSource>,
        pipeline: BitPipeline<f32>,
    },
    Iq {
        source: Box<dyn IqSource>,
        pipeline: BitPipeline<IqSample>,
        pending: Vec<IqSample>,
        /// Decimating channel filter applied to every raw batch before the
        /// demodulator when the blind coarse search selected a far-offset
        /// carrier. `None` for the native-rate (filename/near-baseband) path,
        /// which leaves the sample stream untouched.
        channel: Option<ChannelFilter>,
        /// Sample rate the pipeline actually runs at (input rate divided by the
        /// channel decimation, or the input rate when no channel filter is in
        /// place). Used to convert the demodulator's sample count to a
        /// wall-clock timestamp.
        working_sample_rate: u32,
    },
    IqSoftAo40 {
        source: WavIqSource,
        pipeline: Box<SoftAo40Pipeline>,
        pending: Vec<IqSample>,
    },
}

enum BatchEntry {
    Decoded {
        decoded: DecodedFrame,
        raw: Vec<u8>,
    },
    Pending(Frame),
    #[allow(dead_code)]
    DecodeError(String),
}

impl PipelineRunner {
    fn description(&self) -> &str {
        match self {
            Self::Audio { source, .. } => source.description(),
            Self::FmAudio { source, .. } => source.description(),
            Self::Iq { source, .. } => source.description(),
            Self::IqSoftAo40 { source, .. } => source.description(),
        }
    }

    fn timestamp(&self) -> Duration {
        match self {
            Self::Audio { source, pipeline } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
            Self::FmAudio { source, pipeline } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
            Self::Iq {
                pipeline,
                working_sample_rate,
                ..
            } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / *working_sample_rate as f64,
            ),
            Self::IqSoftAo40 {
                source, pipeline, ..
            } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
        }
    }

    fn read_next_batch(&mut self) -> Result<Option<Vec<BatchEntry>>, String> {
        match self {
            Self::Audio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read WAV samples: {err}")),
                };
                let frames = pipeline.push_samples(&samples[..read]);
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::FmAudio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read audio samples: {err}")),
                };
                let frames = pipeline.push_samples(&samples[..read]);
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::Iq {
                source,
                pipeline,
                pending,
                channel,
                ..
            } => {
                // The `pending` prefix is already at the working rate (decimated
                // by the channel filter when one is engaged), so it is pushed
                // straight through. Freshly-read source batches are raw input
                // samples and must pass through the channel filter first.
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    let frames = pipeline.push_samples(&samples);
                    return Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()));
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read IQ WAV samples: {err}")),
                };
                let frames = match channel {
                    Some(filter) => {
                        let decimated = filter.push(&samples[..read]);
                        pipeline.push_samples(&decimated)
                    }
                    None => pipeline.push_samples(&samples[..read]),
                };
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::IqSoftAo40 {
                source,
                pipeline,
                pending,
            } => {
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    let decoded = pipeline.push_samples(&samples);
                    return Ok(Some(
                        decoded
                            .into_iter()
                            .map(|decoded| BatchEntry::Decoded {
                                decoded,
                                raw: Vec::new(),
                            })
                            .collect(),
                    ));
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read IQ WAV samples: {err}")),
                };
                let decoded = pipeline.push_samples(&samples[..read]);
                Ok(Some(
                    decoded
                        .into_iter()
                        .map(|decoded| BatchEntry::Decoded {
                            decoded,
                            raw: Vec::new(),
                        })
                        .collect(),
                ))
            }
        }
    }

    fn decode_frame(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.decode_frame(frame),
            Self::FmAudio { pipeline, .. } => pipeline.decode_frame(frame),
            Self::Iq { pipeline, .. } => pipeline.decode_frame(frame),
            Self::IqSoftAo40 { .. } => {
                Err("IqSoftAo40 pipeline decodes inline; pending frames are not used".to_string())
            }
        }
    }

    /// Enable offline ASM-correlation block decoding on the underlying pipeline.
    fn enable_block_demod(&mut self) {
        match self {
            Self::Audio { pipeline, .. } => pipeline.enable_block_demod(),
            Self::FmAudio { pipeline, .. } => pipeline.enable_block_demod(),
            Self::Iq { pipeline, .. } => pipeline.enable_block_demod(),
            Self::IqSoftAo40 { .. } => {}
        }
    }

    /// Recover frames from the block decoder after the stream ends.
    fn finalize(&self) -> Vec<Frame> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.finalize(),
            Self::FmAudio { pipeline, .. } => pipeline.finalize(),
            Self::Iq { pipeline, .. } => pipeline.finalize(),
            Self::IqSoftAo40 { .. } => Vec::new(),
        }
    }

    fn last_framer_error(&self) -> Option<&openhoshimi_core::DecodeError> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.last_framer_error(),
            Self::FmAudio { pipeline, .. } => pipeline.last_framer_error(),
            Self::Iq { pipeline, .. } => pipeline.last_framer_error(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn last_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::FmAudio { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::Iq { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn longest_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::FmAudio { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::Iq { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn best_sync_distance(&self) -> Option<usize> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.best_sync_distance(),
            Self::FmAudio { pipeline, .. } => pipeline.best_sync_distance(),
            Self::Iq { pipeline, .. } => pipeline.best_sync_distance(),
            Self::IqSoftAo40 { pipeline, .. } => pipeline.best_sync_distance(),
        }
    }

    fn distance_history(&self) -> Option<&[Option<usize>]> {
        match self {
            Self::Audio { .. } | Self::Iq { .. } | Self::FmAudio { .. } => None,
            Self::IqSoftAo40 { pipeline, .. } => Some(pipeline.distance_history()),
        }
    }
}

fn print_decoded_frame(
    index: usize,
    timestamp: Duration,
    frame: DecodedFrame,
    raw: &[u8],
    telemetry: Option<&SchemaParser>,
    image_reassembler: &mut Option<Box<dyn ImageReassembler>>,
) {
    match frame {
        DecodedFrame::Ax25(ax25) => {
            print_ax25_frame(index, timestamp, &ax25, raw);
            print_telemetry_fields(telemetry, &ax25.info);
        }
        DecodedFrame::Ao40 {
            payload,
            corrected_errors,
        } => {
            println!(
                "#{index:03}  {}  AO40-FEC             [AO-40 FEC]  {} bytes  corrected={corrected_errors}",
                format_timestamp(timestamp),
                payload.len()
            );
            if !raw.is_empty() {
                println!("      raw: {}", format_hex(raw));
            }
            print_telemetry_fields(telemetry, &payload);
        }
        DecodedFrame::Ax100 {
            mode,
            payload,
            corrected_errors,
            crc_ok,
        } => {
            let status = match crc_ok {
                Some(true) => "  [CRC ok]",
                Some(false) => "  [CRC FAIL]",
                None => "",
            };
            println!(
                "#{index:03}  {}  AX100/{:<11}  [GOMspace AX100]  {} bytes  corrected={corrected_errors}{status}",
                format_timestamp(timestamp),
                ax100_mode_label(mode),
                payload.len()
            );
            println!("      payload: {}", format_hex(&payload));
            let ascii: String = payload
                .iter()
                .map(|&b| {
                    if (0x20..0x7f).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("      ascii:   {ascii}");
            // Only surface telemetry for frames whose integrity is
            // confirmed; CRC-failed payloads carry residual bit errors and
            // must not be interpreted as housekeeping values.
            if crc_ok != Some(false) {
                print_telemetry_fields(telemetry, &payload);
            }
        }
        DecodedFrame::Geoscan(geoscan) => {
            print_geoscan_frame(index, timestamp, &geoscan, raw);
            print_telemetry_fields(telemetry, &geoscan.payload);
            if let Some(r) = image_reassembler.as_mut() {
                if geoscan.crc_ok {
                    if let Some(update) = r.ingest(&geoscan.payload) {
                        if update.started_new_image {
                            eprintln!(
                                "decode_file: image #{} starts at frame {} (offset 0)",
                                update.image_idx, index
                            );
                        }
                    }
                }
            }
        }
        DecodedFrame::Ssdv(packet) => {
            let crc = if packet.crc_ok { "CRC ok" } else { "CRC FAIL" };
            let kind = match packet.kind {
                openhoshimi_codec::SsdvPacketKind::WithFec => "SSDV+FEC",
                openhoshimi_codec::SsdvPacketKind::NoFec => "SSDV    ",
            };
            let rs = match packet.rs_errors {
                Some(n) => format!("rs={}", n),
                None => "rs=-".to_string(),
            };
            println!(
                "#{index:03}  {}  {kind}             [{crc}]  cs={:<6}  img={:03}  pkt={:05}  {}x{}  {rs}",
                format_timestamp(timestamp),
                packet.callsign,
                packet.image_id,
                packet.packet_id,
                packet.width,
                packet.height,
            );
            if let Some(r) = image_reassembler.as_mut() {
                if packet.crc_ok {
                    if let Some(update) = r.ingest(&packet.raw) {
                        if update.started_new_image {
                            eprintln!(
                                "decode_file: SSDV image #{} (id {}) starts at frame {}",
                                update.image_idx, packet.image_id, index
                            );
                        }
                    }
                }
            }
        }
        DecodedFrame::Raw {
            frame_type,
            raw_len,
        } => {
            println!(
                "#{index:03}  {}  RAW/{:<12}  [raw]  {raw_len} bytes",
                format_timestamp(timestamp),
                frame_type_label(frame_type)
            );
            println!("      raw: {}", format_hex(raw));
            print_telemetry_fields(telemetry, raw);
        }
    }
}

fn print_geoscan_frame(index: usize, timestamp: Duration, frame: &GeoscanFrame, raw: &[u8]) {
    let crc_status = if frame.crc_ok { "CRC ok" } else { "CRC FAIL" };
    println!(
        "#{index:03}  {}  GEOSCAN              [{crc_status}]  {} bytes  crc_rx={:#06x}  crc_calc={:#06x}",
        format_timestamp(timestamp),
        frame.payload.len(),
        frame.crc_received,
        frame.crc_expected,
    );
    if !raw.is_empty() {
        println!("      raw:     {}", format_hex(raw));
    }
    println!("      payload: {}", format_hex(&frame.payload));
    println!("      ascii:   {}", format_ascii(&frame.payload));
}

fn format_ascii(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| {
            if (0x20..=0x7e).contains(b) {
                *b as char
            } else {
                '.'
            }
        })
        .collect()
}

fn print_ax25_frame(index: usize, timestamp: Duration, frame: &Ax25Frame, raw: &[u8]) {
    let source = format_call(&frame.source);
    let destination = format_call(&frame.destination);
    let frame_kind = if frame.control == 0x03 && frame.pid == Some(0xf0) {
        "AX.25 UI"
    } else {
        "AX.25"
    };

    println!(
        "#{index:03}  {}  {source:<9} >  {destination:<9}  [{frame_kind}]  {} bytes",
        format_timestamp(timestamp),
        raw.len()
    );
    println!("      raw: {}", format_hex(raw));
}

fn ax100_mode_label(mode: Ax100Mode) -> &'static str {
    match mode {
        Ax100Mode::ReedSolomon => "rs",
        Ax100Mode::AsmGolay => "asm-golay",
        Ax100Mode::AsmGolayCrc => "asm-golay-crc",
    }
}

fn print_telemetry_fields(telemetry: Option<&SchemaParser>, bytes: &[u8]) {
    let Some(telemetry) = telemetry else {
        return;
    };

    for field in telemetry.parse_bytes(bytes) {
        println!(
            "      {}.{} = {}",
            field.group,
            field.key,
            format_telemetry_value(&field)
        );
    }
}

fn format_telemetry_value(field: &openhoshimi_core::TelemetryField) -> String {
    let value = match &field.value {
        openhoshimi_core::TelemetryValue::Float(value) => format_float(*value),
        openhoshimi_core::TelemetryValue::Int(value) => value.to_string(),
        openhoshimi_core::TelemetryValue::Bool(value) => value.to_string(),
        openhoshimi_core::TelemetryValue::Text(text) => text.clone(),
        openhoshimi_core::TelemetryValue::Bytes(bytes) => format_hex(bytes),
    };
    match &field.unit {
        Some(unit) if !unit.is_empty() => format!("{value} {unit}"),
        _ => value,
    }
}

fn format_float(value: f64) -> String {
    let mut text = format!("{value:.3}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    text
}

/// Seconds of prefix the blind coarse sweep scores. Longer than the native
/// 8 s window so the FLL/timing loops converge at a coarse-grid residual and
/// accumulate enough RS-validated frames to out-score noise offsets.
const BLIND_PREFIX_SECONDS: usize = 30;

/// Read the blind-sweep prefix from a freshly-reopened copy of the input WAV,
/// leaving the streaming decode source untouched.
fn read_blind_prefix(input_path: &Path, sample_rate: u32) -> Result<Vec<IqSample>, String> {
    let mut probe = WavIqSource::open(input_path)
        .map_err(|err| format!("failed to reopen WAV for blind coarse search: {err}"))?;
    // Diagnostic: score the blind sweep on a window starting this many seconds
    // into the pass, not just the start. On passes whose SNR peaks mid-pass the
    // decodable signal is absent from the first window (frequency x time search).
    let start_sec = std::env::var("OPENHOSHIMI_BLIND_START_SEC")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let skip = sample_rate as usize * start_sec;
    let len = sample_rate as usize * BLIND_PREFIX_SECONDS;
    let all = read_iq_prefix(&mut probe, skip + len)?;
    Ok(all.into_iter().skip(skip).collect())
}

/// Build the IQ-mode `PipelineRunner` for a real stereo IQ WAV source.
///
/// Owns the prefix scan, polarity/skew alignment, and CPM carrier
/// estimation that the IQ path uses to lock onto a recording.
fn build_iq_runner(
    mut source: WavIqSource,
    input_path: &Path,
    downlink_in: &openhoshimi_core::satellite::DownlinkDef,
    tuning_offset_hz: f32,
) -> Result<PipelineRunner, String> {
    let sample_rate = source.sample_rate();
    // Native scan: the established 8 s prefix, read from the streaming source —
    // its behaviour on already-working recordings stays byte-for-byte unchanged.
    let prefix_len = sample_rate as usize * 8;
    let prefix = read_iq_prefix(&mut source, prefix_len)?;
    let scored =
        prepare_linear_iq_setup_scored(downlink_in, sample_rate, &prefix, tuning_offset_hz);
    let native_frames = scored.as_ref().map_or(0, |(_, frames)| *frames);

    // Blind coarse frequency acquisition: sweep candidate carrier offsets across
    // the whole recording band through a decimating channel filter, scoring each
    // by RS-validated prefix frames. This isolates a single narrow channel —
    // rejecting the other in-band signals that otherwise dominate the AGC/loops
    // — and runs the demodulator at ~5 samples/symbol.
    //
    // The sweep reads its own longer prefix from a freshly-reopened source so it
    // does not disturb the streaming `source` used for decode. It needs the
    // longer look to let the FLL/timing loops converge at a coarse residual and
    // accumulate enough RS-validated frames to out-score noise.
    //
    // The native scan already tries the filename tuning, the FFT carrier peaks,
    // and the average offset within the band-edge FLL's pull-in, so it is kept
    // whenever it decodes. The blind result is adopted only when it decimates a
    // wideband recording (decimation > 1 — a rate the native chain leaves
    // unfiltered at ~26 sps) AND recovers strictly more RS-validated prefix
    // frames than the native scan. At decimation == 1 (the 50 kHz ASRTU
    // recording) the blind chain is functionally the native chain, so it is not
    // allowed to override a working native decode. AO-40 FEC keeps its dedicated
    // soft path. A diagnostic env override forces the channel path.
    let force_channel = std::env::var_os("OPENHOSHIMI_FORCE_CHANNEL_OFFSET").is_some();
    if !is_ao40_fec_downlink(downlink_in) {
        let blind_prefix = read_blind_prefix(input_path, sample_rate)?;
        if let Some(channelized) =
            blind_linear_iq_channel_setup(downlink_in, sample_rate, &blind_prefix, tuning_offset_hz)
        {
            let adopt = channelized.decimation > 1 && channelized.prefix_frames > native_frames;
            if force_channel || adopt {
                // The channelized runner re-derives its own prefix from the raw
                // stream `source` (already positioned past the native 8 s
                // prefix), so `prefix` here is only the native window.
                return build_channelized_iq_runner(source, prefix, sample_rate, channelized);
            }
        }
    }

    let setup = scored.as_ref().map(|(setup, _)| setup.clone());
    let (downlink, tuning_offset_hz, pending) = if let Some(setup) = setup {
        let pending = if setup.sample_skip < prefix.len() {
            prefix.into_iter().skip(setup.sample_skip).collect()
        } else {
            Vec::new()
        };
        eprintln!(
            "decode_file: prefix setup: tuning={:.1} Hz skip={} prefix_frames={}",
            setup.tuning_offset_hz,
            setup.sample_skip,
            scored.as_ref().map_or(0, |(_, frames)| *frames),
        );
        if let Some(openhoshimi_core::satellite::ModemDef::Linear {
            differential,
            invert,
            swap_iq,
            ..
        }) = setup.downlink.modem.as_ref()
        {
            eprintln!(
                "decode_file: prefix polarity: differential={differential} invert={invert} swap_iq={swap_iq}",
            );
        }
        (setup.downlink, setup.tuning_offset_hz, pending)
    } else {
        let downlink = downlink_in.clone();
        let mut tuning = tuning_offset_hz;
        if matches!(downlink.modem, Some(ModemDef::Cpm { .. })) {
            let estimate = estimate_cpm_iq_frequency_offset_hz(&prefix, sample_rate)
                .or_else(|| estimate_iq_frequency_offset_hz(&prefix, sample_rate))
                .filter(|v| v.is_finite());
            if let Some(estimate) = estimate {
                eprintln!(
                    "decode_file: cpm carrier estimate from prefix: {estimate:.1} Hz (filename hint: {tuning:.1} Hz)",
                );
                tuning = estimate;
            }
        }
        (downlink, tuning, prefix)
    };
    if is_ao40_fec_downlink(&downlink) {
        eprintln!("decode_file: using soft-decision Viterbi path for AO-40 FEC");
        let pipeline = SoftAo40Pipeline::new(&downlink, sample_rate, tuning_offset_hz)?;
        Ok(PipelineRunner::IqSoftAo40 {
            source,
            pipeline: Box::new(pipeline),
            pending,
        })
    } else {
        let mut pipeline = BitPipeline::<IqSample>::new(&downlink)?;
        pipeline.configure_demodulator(&downlink, sample_rate, tuning_offset_hz)?;
        Ok(PipelineRunner::Iq {
            source: Box::new(source),
            pipeline,
            pending,
            channel: None,
            working_sample_rate: sample_rate,
        })
    }
}

/// Build the IQ runner for a carrier acquired by the blind coarse frequency
/// search. Installs a decimating channel filter that shifts the chosen offset
/// to DC and drops the sample rate to the working rate, then runs the linear
/// pipeline on the decimated stream. The prefix already read from the source is
/// decimated once here so it is not re-read.
fn build_channelized_iq_runner(
    source: WavIqSource,
    prefix: Vec<IqSample>,
    input_sample_rate: u32,
    channelized: openhoshimi_runtime::pipeline::ChannelizedIqSetup,
) -> Result<PipelineRunner, String> {
    let openhoshimi_runtime::pipeline::ChannelizedIqSetup {
        channel_offset_hz,
        decimation,
        working_sample_rate,
        setup,
        prefix_frames,
        prefix_sync_locks,
    } = channelized;

    eprintln!(
        "decode_file: blind coarse acquisition: offset={channel_offset_hz:+.1} Hz decimation={decimation} working_rate={working_sample_rate} Hz prefix_frames={prefix_frames} prefix_sync_locks={prefix_sync_locks}",
    );
    if let Some(ModemDef::Linear {
        differential,
        invert,
        swap_iq,
        ..
    }) = setup.downlink.modem.as_ref()
    {
        eprintln!(
            "decode_file: blind polarity: differential={differential} invert={invert} swap_iq={swap_iq}",
        );
    }

    // Decimate the already-read prefix through a filter identical to the one
    // that will process the live stream, then drop the resolved sample skip in
    // the decimated domain so demodulation starts at the best symbol boundary.
    let mut prefix_filter = ChannelFilter::new(input_sample_rate, channel_offset_hz, decimation);
    let decimated_prefix = prefix_filter.push(&prefix);
    let pending: Vec<IqSample> = if setup.sample_skip < decimated_prefix.len() {
        decimated_prefix
            .into_iter()
            .skip(setup.sample_skip)
            .collect()
    } else {
        Vec::new()
    };

    let mut pipeline = BitPipeline::<IqSample>::new(&setup.downlink)?;
    pipeline.configure_demodulator(&setup.downlink, working_sample_rate, setup.tuning_offset_hz)?;

    // The live-stream channel filter is a fresh instance: its NCO/FIR/decimation
    // phase must start from the sample right after the prefix. The source has
    // already advanced past the prefix (read_iq_prefix consumed it), and the
    // decimation cadence realigns within a few output samples, so a fresh
    // filter is correct here.
    let channel = ChannelFilter::new(input_sample_rate, channel_offset_hz, decimation);

    Ok(PipelineRunner::Iq {
        source: Box::new(source),
        pipeline,
        pending,
        channel: Some(channel),
        working_sample_rate,
    })
}
