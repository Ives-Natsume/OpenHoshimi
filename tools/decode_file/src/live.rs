//! Interactive live ASRTU-1 decode from a soundcard (or a file, for testing).
//!
//! `decode_file --live` opens a terminal UI: an arrow-key device picker, then a
//! live view with a spectrum, running decode status and the latest telemetry.
//! Audio capture and the streaming soundmodem decoder run on a worker thread;
//! the main thread only renders and reads keys. The demod/FEC/telemetry is our
//! own `openhoshimi_soundmodem` decoder. The UI is written from scratch; only
//! three ideas are borrowed from the upstream CLI's approach: capture at a
//! fixed 48 kHz, accumulate a fixed-size block before the FFT, and draw the
//! spectrum on a fixed dB scale (never a per-frame min/max, which turns noise
//! into a full-height mess).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use openhoshimi_core::{InputSource, IoError, IqSample, IqSource};
use openhoshimi_io::{
    enumerate_input_devices, open_audio_source, SoundcardIqSource, SoundcardSource, WavIqSource,
};
use openhoshimi_runtime::soundmodem_frontend::StreamResampler;
use openhoshimi_soundmodem::{DecodedField, Decoder, Frame, TelemetryDecoder, SAMPLE_RATE};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Sparkline};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// FFT size for the live spectrum.
const FFT_LEN: usize = 1024;
/// Samples read per capture iteration.
const CHUNK: usize = 4096;
/// Recent telemetry lines kept for the live view.
const LOG_LINES: usize = 200;
/// Spectrum dB window: bars fill between these two levels (fixed, not adaptive,
/// so a bar's height reflects an absolute level and the display stays steady).
const FLOOR_DB: f32 = -100.0;
const TOP_DB: f32 = -20.0;
/// Exponential smoothing applied to the spectrum, per FFT.
const SPEC_ALPHA: f32 = 0.4;
/// Minimum spacing between spectrum FFTs.
const FFT_PERIOD: Duration = Duration::from_millis(66);
/// After this long with no fresh samples, the view flags "waiting for input".
const STALL: Duration = Duration::from_secs(2);
/// How many seconds of raw IQ to retain for the full decode pass on exit. Caps
/// memory on a long session while covering a whole LEO pass.
const ACCUM_SECS: usize = 900;

type Cf = Complex<f32>;

/// Options for a live session.
pub struct LiveOpts {
    pub device: Option<String>,
    pub mono: bool,
    pub seconds: Option<u64>,
    pub tlm_out: Option<PathBuf>,
    pub file: Option<PathBuf>,
    pub speed: f32,
}

/// State shared between the capture/decode worker and the render loop.
struct Shared {
    /// Raw per-bin spectrum in dB, DC-centred for IQ (0..fs/2 for mono).
    spectrum_db: Vec<f32>,
    peak_hz: f32,
    level_dbfs: f32,
    frames: usize,
    packets: usize,
    corrected: i64,
    /// Compact "packet  field=value ..." lines, newest last.
    log: VecDeque<String>,
    total_samples: u64,
    /// When the last non-empty block arrived (for the stall indicator).
    last_data: Option<Instant>,
    /// The file source reached its end.
    finished: bool,
    stop: bool,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            spectrum_db: Vec::new(),
            peak_hz: 0.0,
            level_dbfs: -120.0,
            frames: 0,
            packets: 0,
            corrected: 0,
            log: VecDeque::new(),
            total_samples: 0,
            last_data: None,
            finished: false,
            stop: false,
        }
    }
}

/// Either capture path, so the worker handles a stereo IQ device/file or a mono
/// FM-audio device/file through one loop.
enum Source {
    Iq(Box<dyn IqSource>),
    Mono(Box<dyn InputSource>),
}

impl Source {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::Iq(s) => s.sample_rate(),
            Self::Mono(s) => s.sample_rate(),
        }
    }

    fn description(&self) -> String {
        match self {
            Self::Iq(s) => s.description().to_string(),
            Self::Mono(s) => s.description().to_string(),
        }
    }

    fn is_mono(&self) -> bool {
        matches!(self, Self::Mono(_))
    }
}

/// Result handed back from the interactive session for the final decode.
struct LiveResult {
    /// Capture sample rate.
    rate: u32,
    /// Raw captured IQ (for a stereo source) to run the full offline pass over.
    iq: Option<Vec<IqSample>>,
    /// Frames the streaming path decoded live (used for a mono source, where
    /// the offline IQ front-end does not apply).
    streamed: Vec<Frame>,
}

/// Entry point for `--live`. Runs the interactive session; then, with the
/// terminal restored, decodes the captured IQ with the full offline
/// survey/replay pass (so the final yield matches `--soundmodem`, not just the
/// causal live path) and prints the frames as a coloured hex dump plus the
/// telemetry report, exactly like the batch path.
pub fn run_live(opts: LiveOpts) -> Result<(), String> {
    let mut terminal = ratatui::try_init().map_err(|err| format!("terminal init failed: {err}"))?;
    let outcome = live_session(&mut terminal, &opts);
    ratatui::restore();
    let result = outcome?;

    let frames = match &result.iq {
        Some(iq) if !iq.is_empty() => {
            eprintln!(
                "decode_file: full decode pass over {:.0}s of captured IQ...",
                iq.len() as f32 / result.rate as f32
            );
            openhoshimi_runtime::soundmodem_frontend::decode_iq(iq, result.rate)
        }
        _ => result.streamed,
    };
    let tlm_path = opts
        .tlm_out
        .clone()
        .unwrap_or_else(|| PathBuf::from("live.telemetry.txt"));
    crate::print_sm_frames(&frames, &tlm_path);
    Ok(())
}

fn live_session(
    terminal: &mut ratatui::DefaultTerminal,
    opts: &LiveOpts,
) -> Result<LiveResult, String> {
    // A file source is paced to (scaled) real time so it plays like a pass; a
    // soundcard already arrives in real time.
    let (source, pace) = match &opts.file {
        Some(path) => (open_from_file(path, opts.mono)?, true),
        None => {
            let device = match opts.device.clone() {
                Some(name) => name,
                None => match pick_device(terminal)? {
                    Some(name) => name,
                    // User cancelled the picker: nothing captured.
                    None => {
                        return Ok(LiveResult {
                            rate: 0,
                            iq: None,
                            streamed: Vec::new(),
                        })
                    }
                },
            };
            (open_source(&device, opts.mono)?, false)
        }
    };
    let sample_rate = source.sample_rate();
    let description = source.description();
    let speed = if opts.speed > 0.0 { opts.speed } else { 1.0 };

    let shared = Arc::new(Mutex::new(Shared::default()));
    let frames = Arc::new(Mutex::new(Vec::<Frame>::new()));
    let iq_log = Arc::new(Mutex::new(VecDeque::<IqSample>::new()));
    let cap = (sample_rate as usize).saturating_mul(ACCUM_SECS);
    let pace_speed = if pace { Some(speed) } else { None };
    let worker = spawn_worker(
        source,
        pace_speed,
        cap,
        Arc::clone(&shared),
        Arc::clone(&frames),
        Arc::clone(&iq_log),
    );

    let started = Instant::now();
    let deadline = opts.seconds.map(|s| started + Duration::from_secs(s));
    loop {
        {
            let snap = shared.lock().map_err(|_| "state lock poisoned")?;
            terminal
                .draw(|f| draw_live(f, &snap, &description, sample_rate, started.elapsed()))
                .map_err(|err| format!("draw failed: {err}"))?;
        }
        if event::poll(Duration::from_millis(100)).map_err(|err| format!("input poll: {err}"))? {
            if let Event::Key(key) = event::read().map_err(|err| format!("input read: {err}"))? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                {
                    break;
                }
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
    }

    // Signal the worker to stop and wait for it.
    if let Ok(mut s) = shared.lock() {
        s.stop = true;
    }
    let _ = worker.join();

    let streamed = frames.lock().map_err(|_| "frames lock poisoned")?.clone();
    let iq = if opts.mono {
        None
    } else {
        Some(
            iq_log
                .lock()
                .map(|log| log.iter().copied().collect::<Vec<_>>())
                .unwrap_or_default(),
        )
    };
    Ok(LiveResult {
        rate: sample_rate,
        iq,
        streamed,
    })
}

/// Interactive device picker. Returns the chosen device name, or `None` if the
/// user pressed Esc / q.
fn pick_device(terminal: &mut ratatui::DefaultTerminal) -> Result<Option<String>, String> {
    let devices = enumerate_input_devices();
    if devices.is_empty() {
        return Err("no soundcard input devices found".to_string());
    }
    let mut state = ListState::default();
    state.select(Some(devices.iter().position(|d| d.is_default).unwrap_or(0)));

    loop {
        terminal
            .draw(|f| {
                let items: Vec<ListItem> = devices
                    .iter()
                    .map(|d| {
                        let tag = if d.is_default { "  (default)" } else { "" };
                        ListItem::new(format!("{}{tag}", d.name))
                    })
                    .collect();
                let list = List::new(items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" Select input device  (up/down move, Enter select, q cancel) "),
                    )
                    .highlight_style(
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                    .highlight_symbol("> ");
                f.render_stateful_widget(list, f.area(), &mut state);
            })
            .map_err(|err| format!("draw failed: {err}"))?;

        if let Event::Key(key) = event::read().map_err(|err| format!("input read: {err}"))? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Up => {
                    let i = state.selected().unwrap_or(0);
                    state.select(Some(i.checked_sub(1).unwrap_or(devices.len() - 1)));
                }
                KeyCode::Down => {
                    let i = state.selected().unwrap_or(0);
                    state.select(Some((i + 1) % devices.len()));
                }
                KeyCode::Enter => {
                    let i = state.selected().unwrap_or(0);
                    return Ok(Some(devices[i].name.clone()));
                }
                KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                _ => {}
            }
        }
    }
}

fn open_source(device: &str, mono: bool) -> Result<Source, String> {
    if mono {
        SoundcardSource::open_by_name(device)
            .map(|s| Source::Mono(Box::new(s)))
            .map_err(|err| format!("open soundcard (mono) failed: {err}"))
    } else {
        SoundcardIqSource::open_by_name(device)
            .map(|s| Source::Iq(Box::new(s)))
            .map_err(|err| format!("open soundcard (IQ) failed: {err}"))
    }
}

fn open_from_file(path: &Path, mono: bool) -> Result<Source, String> {
    if mono {
        open_audio_source(path).map(Source::Mono)
    } else {
        WavIqSource::open(path)
            .map(|s| Source::Iq(Box::new(s)))
            .map_err(|err| format!("open IQ WAV failed: {err}"))
    }
}

/// Spawn the capture + decode worker. It reads the source, accumulates a
/// fixed-size window for the spectrum, feeds the streaming decoder and records
/// every frame. `pace` is `Some(speed)` for a file (played at `speed` times
/// real time) and `None` for a soundcard (already real time).
fn spawn_worker(
    source: Source,
    pace: Option<f32>,
    iq_cap: usize,
    shared: Arc<Mutex<Shared>>,
    frames: Arc<Mutex<Vec<Frame>>>,
    iq_log: Arc<Mutex<VecDeque<IqSample>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_LEN);
        let mut decoder = Decoder::new();
        let mut telemetry = TelemetryDecoder::new();
        let sample_rate = source.sample_rate();
        let mono = source.is_mono();
        let mut source = source;
        // The decoder's timing is fixed at 48 kHz; resample anything else (e.g.
        // a 44.1 kHz soundcard) before feeding it, or symbol timing never locks.
        let mut resampler =
            (sample_rate != SAMPLE_RATE).then(|| StreamResampler::new(sample_rate, SAMPLE_RATE));

        let mut window: VecDeque<Cf> = VecDeque::with_capacity(FFT_LEN);
        let mut smoothed: Vec<f32> = Vec::new();
        let mut last_fft = Instant::now() - FFT_PERIOD;

        loop {
            if shared.lock().map_or(true, |s| s.stop) {
                break;
            }
            let block = match read_block(&mut source) {
                Ok(Some(block)) => block,
                Ok(None) => continue,
                // End of a file source: flag it and leave the view up.
                Err(_) => {
                    if let Ok(mut s) = shared.lock() {
                        s.finished = true;
                    }
                    break;
                }
            };
            let n = block.len();
            if let Some(speed) = pace {
                let secs = n as f32 / (sample_rate as f32 * speed.max(0.01));
                thread::sleep(Duration::from_secs_f32(secs));
            }

            // Retain raw IQ (bounded) so the exit pass can run the offline
            // survey/replay decoder over the whole session.
            if let Chunk::Iq(v) = &block {
                if let Ok(mut log) = iq_log.lock() {
                    log.extend(v.iter().map(|c| IqSample { i: c.re, q: c.im }));
                    while log.len() > iq_cap {
                        log.pop_front();
                    }
                }
            }

            push_window(&mut window, &block);
            let rms = block_rms(&block);
            let mut spectrum: Option<(Vec<f32>, f32)> = None;
            if window.len() == FFT_LEN && last_fft.elapsed() >= FFT_PERIOD {
                spectrum = Some(compute_spectrum(
                    &fft,
                    &window,
                    &mut smoothed,
                    mono,
                    sample_rate,
                ));
                last_fft = Instant::now();
            }

            let new = decode_block(&mut decoder, &mut resampler, &block);

            if let Ok(mut s) = shared.lock() {
                s.total_samples += n as u64;
                s.last_data = Some(Instant::now());
                s.level_dbfs = 20.0 * rms.max(1e-9).log10();
                if let Some((db, peak_hz)) = spectrum {
                    s.spectrum_db = db;
                    s.peak_hz = peak_hz;
                }
            }
            if !new.is_empty() {
                record_frames(&shared, &frames, telemetry.as_mut(), &new);
            }
        }
    })
}

/// One captured block of audio: complex IQ or mono PCM (already `i16`).
enum Chunk {
    Iq(Vec<Cf>),
    Mono(Vec<i16>),
}

impl Chunk {
    fn len(&self) -> usize {
        match self {
            Self::Iq(v) => v.len(),
            Self::Mono(v) => v.len(),
        }
    }
}

fn read_block(source: &mut Source) -> Result<Option<Chunk>, IoError> {
    match source {
        Source::Iq(src) => {
            let mut buf = vec![IqSample::default(); CHUNK];
            let n = src.read_samples(&mut buf)?;
            if n == 0 {
                return Ok(None);
            }
            Ok(Some(Chunk::Iq(
                buf[..n].iter().map(|s| Cf::new(s.i, s.q)).collect(),
            )))
        }
        Source::Mono(src) => {
            let mut buf = [0.0f32; CHUNK];
            let n = match src.read_samples(&mut buf) {
                Ok(0) => return Ok(None),
                Ok(n) => n,
                Err(err) => return Err(err),
            };
            Ok(Some(Chunk::Mono(
                buf[..n].iter().map(|&x| (x * 32767.0) as i16).collect(),
            )))
        }
    }
}

/// Append a block to the rolling FFT window, keeping only the newest FFT_LEN.
fn push_window(window: &mut VecDeque<Cf>, block: &Chunk) {
    match block {
        Chunk::Iq(v) => window.extend(v.iter().copied()),
        Chunk::Mono(v) => window.extend(v.iter().map(|&s| Cf::new(f32::from(s) / 32768.0, 0.0))),
    }
    while window.len() > FFT_LEN {
        window.pop_front();
    }
}

fn block_rms(block: &Chunk) -> f32 {
    match block {
        Chunk::Iq(v) => {
            (v.iter().map(|c| c.norm_sqr()).sum::<f32>() / v.len().max(1) as f32).sqrt()
        }
        Chunk::Mono(v) => {
            let s: f32 = v
                .iter()
                .map(|&x| {
                    let f = f32::from(x) / 32768.0;
                    f * f
                })
                .sum();
            (s / v.len().max(1) as f32).sqrt()
        }
    }
}

fn decode_block(
    decoder: &mut Option<Decoder>,
    resampler: &mut Option<StreamResampler>,
    block: &Chunk,
) -> Vec<Frame> {
    let Some(dec) = decoder.as_mut() else {
        return Vec::new();
    };
    match block {
        Chunk::Iq(v) => {
            let resampled;
            let samples: &[Cf] = match resampler {
                Some(rs) => {
                    let mut out = Vec::new();
                    rs.process(v, &mut out);
                    resampled = out;
                    &resampled
                }
                None => v,
            };
            let mut ilv = Vec::with_capacity(samples.len() * 2);
            for c in samples {
                ilv.push(c.re);
                ilv.push(c.im);
            }
            dec.feed_iq_collect(&ilv)
        }
        Chunk::Mono(pcm) => match resampler {
            Some(rs) => {
                let cin: Vec<Cf> = pcm.iter().map(|&s| Cf::new(f32::from(s), 0.0)).collect();
                let mut out = Vec::new();
                rs.process(&cin, &mut out);
                let res: Vec<i16> = out.iter().map(|c| c.re as i16).collect();
                dec.feed_collect(&res)
            }
            None => dec.feed_collect(pcm),
        },
    }
}

/// Windowed FFT of the current window, returning per-bin dB (DC-centred for IQ)
/// after exponential smoothing, plus the peak-bin frequency offset in Hz.
fn compute_spectrum(
    fft: &Arc<dyn Fft<f32>>,
    window: &VecDeque<Cf>,
    smoothed: &mut Vec<f32>,
    mono: bool,
    fs: u32,
) -> (Vec<f32>, f32) {
    let mut buf: Vec<Cf> = window.iter().copied().collect();
    for (i, c) in buf.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FFT_LEN as f32 - 1.0)).cos();
        *c *= w;
    }
    fft.process(&mut buf);

    let norm = 1.0 / FFT_LEN as f32;
    let db = |c: &Cf| 10.0 * (c.norm_sqr() * norm * norm + 1e-12).log10();
    let raw: Vec<f32> = if mono {
        buf[..FFT_LEN / 2].iter().map(db).collect()
    } else {
        // fftshift so DC sits in the middle.
        let mut v = Vec::with_capacity(FFT_LEN);
        v.extend(buf[FFT_LEN / 2..].iter().map(db));
        v.extend(buf[..FFT_LEN / 2].iter().map(db));
        v
    };

    if smoothed.len() != raw.len() {
        *smoothed = raw.clone();
    } else {
        for (s, &r) in smoothed.iter_mut().zip(&raw) {
            *s = SPEC_ALPHA * r + (1.0 - SPEC_ALPHA) * *s;
        }
    }

    let peak_idx = smoothed
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    let peak_hz = if mono {
        peak_idx as f32 / raw.len() as f32 * (fs as f32 / 2.0)
    } else {
        (peak_idx as f32 / raw.len() as f32 - 0.5) * fs as f32
    };
    (smoothed.clone(), peak_hz)
}

/// Fold newly decoded frames into the shared status and telemetry log.
fn record_frames(
    shared: &Mutex<Shared>,
    frames: &Mutex<Vec<Frame>>,
    telemetry: Option<&mut TelemetryDecoder>,
    new: &[Frame],
) {
    let mut lines: Vec<String> = Vec::new();
    let mut packets = 0usize;
    let mut tlm = telemetry;
    for frame in new {
        if let Some(dec) = tlm.as_deref_mut() {
            let fields = dec.feed_fields(&frame.data);
            if let Some(line) = summarise(&fields) {
                packets += 1;
                lines.push(line);
            }
        }
    }
    if let Ok(mut s) = shared.lock() {
        s.frames += new.len();
        s.packets += packets;
        for f in new {
            s.corrected += i64::from(f.corrected);
        }
        for line in lines {
            if s.log.len() == LOG_LINES {
                s.log.pop_front();
            }
            s.log.push_back(line);
        }
    }
    if let Ok(mut store) = frames.lock() {
        store.extend_from_slice(new);
    }
}

/// One compact status line for a decoded packet: its code plus a few fields.
fn summarise(fields: &[DecodedField]) -> Option<String> {
    let first = fields.first()?;
    let mut line = format!("{:<7}", first.packet);
    for f in fields.iter().take(6) {
        let unit = if f.unit.is_empty() { "" } else { &f.unit };
        line.push_str(&format!("  {}={}{}", f.cn, f.value, unit));
    }
    Some(line)
}

/// Bin a per-bin dB spectrum down to `width` columns on the fixed dB scale.
fn spectrum_heights(db: &[f32], width: usize) -> Vec<u64> {
    if db.is_empty() || width == 0 {
        return Vec::new();
    }
    let per = db.len().div_ceil(width).max(1);
    db.chunks(per)
        .map(|c| {
            let avg = c.iter().copied().sum::<f32>() / c.len() as f32;
            (((avg - FLOOR_DB) / (TOP_DB - FLOOR_DB)) * 100.0).clamp(0.0, 100.0) as u64
        })
        .collect()
}

fn draw_live(
    frame: &mut ratatui::Frame,
    s: &Shared,
    description: &str,
    sample_rate: u32,
    elapsed: Duration,
) {
    let areas = Layout::vertical([
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Min(3),
    ])
    .split(frame.area());

    let width = usize::from(areas[0].width.saturating_sub(2));
    let heights = spectrum_heights(&s.spectrum_db, width);
    let spectrum = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Spectrum  (peak {:+.1} kHz, {FLOOR_DB:.0}..{TOP_DB:.0} dB) ",
            s.peak_hz / 1000.0
        )))
        .data(&heights)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(spectrum, areas[0]);

    // Health line: a clear cue for the three states the user cares about.
    let stalled = !s.finished && s.last_data.map_or(true, |t| t.elapsed() > STALL);
    let health = if s.finished {
        Span::styled(
            "file done - press q to quit",
            Style::default().fg(Color::Yellow),
        )
    } else if stalled {
        Span::styled(
            "waiting for input (no samples)...",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled("capturing", Style::default().fg(Color::Green))
    };
    let rate_note = if sample_rate == SAMPLE_RATE {
        Span::raw(String::new())
    } else {
        Span::styled(
            format!("   WARNING device is {sample_rate} Hz, decoder needs {SAMPLE_RATE} Hz"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    };

    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("device ", Style::default().fg(Color::DarkGray)),
            Span::raw(description.to_string()),
        ]),
        Line::from(vec![
            Span::raw(format!(
                "rate {sample_rate} Hz   elapsed {:>3}s   level {:>6.1} dBFS   ",
                elapsed.as_secs(),
                s.level_dbfs
            )),
            health,
            rate_note,
        ]),
        Line::from(vec![
            Span::styled("frames ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                s.frames.to_string(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   packets ", Style::default().fg(Color::DarkGray)),
            Span::styled(s.packets.to_string(), Style::default().fg(Color::Green)),
            Span::styled("   RS corrected ", Style::default().fg(Color::DarkGray)),
            Span::raw(s.corrected.to_string()),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Decode status  (q / Esc to quit) "),
    );
    frame.render_widget(status, areas[1]);

    let rows = usize::from(areas[2].height.saturating_sub(2));
    let log: Vec<Line> = s
        .log
        .iter()
        .rev()
        .take(rows)
        .rev()
        .map(|l| Line::from(l.clone()))
        .collect();
    let telemetry = Paragraph::new(log).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Telemetry (live) "),
    );
    frame.render_widget(telemetry, areas[2]);
}
