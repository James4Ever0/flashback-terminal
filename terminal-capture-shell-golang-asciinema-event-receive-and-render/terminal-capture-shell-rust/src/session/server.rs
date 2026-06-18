use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use nix::pty::Winsize;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, Duration};
use tracing::{debug, info, trace, warn};

use crate::capture::buffer::Buffer;
use crate::capture::engine::Engine;
use crate::config::{Config, default_device_id};
use crate::protocol::{CaptureResponse, Request, Response, StatusResponse};
use crate::server::client::Client;
use crate::pty;
use crate::tty::TtySize;
use crate::vt_capture::VtCapture;

pub struct Server {
    session_id: String,
    socket_path: PathBuf,
    config: Arc<Config>,
    command: Vec<String>,
    cols: u16,
    rows: u16,
}

#[derive(Debug)]
enum ControlMessage {
    Resize(u16, u16),
    Kill,
}

impl Server {
    pub fn new(
        session_id: String,
        socket_path: PathBuf,
        config: Config,
        command: Vec<String>,
        cols: u16,
        rows: u16,
    ) -> Self {
        Self {
            session_id,
            socket_path,
            config: Arc::new(config),
            command,
            cols,
            rows,
        }
    }

    pub async fn run(self) -> Result<()> {
        info!(
            "starting server session={} socket={} command={:?}",
            self.session_id,
            self.socket_path.display(),
            self.command
        );

        let winsize: Winsize = TtySize(self.cols, self.rows).into();
        let mut extra_env = HashMap::<String, String>::new();
        extra_env.insert("FLASHBACK_SHELL".to_owned(), "1".to_owned());
        info!("spawning pty command={:?} env={:?}", self.command, extra_env);
        let pty = Arc::new(pty::spawn(
            &self.command,
            winsize,
            &extra_env,
        )?);
        info!("pty spawned");

        let vt = Arc::new(std::sync::Mutex::new(VtCapture::new(
            self.cols as usize,
            self.rows as usize,
            self.config.scrollback_lines,
        )));

        // Ensure socket parent dir exists and remove stale socket.
        if let Some(dir) = self.socket_path.parent() {
            let _ = tokio::fs::create_dir_all(dir).await;
        }
        let _ = tokio::fs::remove_file(&self.socket_path).await;
        let listener = UnixListener::bind(&self.socket_path)
            .with_context(|| format!("cannot bind socket {}", self.socket_path.display()))?;

        let (pty_input_tx, mut pty_input_rx) = mpsc::channel::<Vec<u8>>(256);
        let (pty_output_tx, _pty_output_rx) = broadcast::channel::<Vec<u8>>(256);
        let (control_tx, mut control_rx) = mpsc::channel::<ControlMessage>(16);

        // PTY reader: tee output to attached clients and to VT.
        let pty_reader = {
            let pty = pty.clone();
            let pty_output_tx = pty_output_tx.clone();
            let vt = vt.clone();
            let mut decoder = Utf8Decoder::new();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let mut total = 0usize;
                let mut first = true;
                info!("pty reader task started");
                loop {
                    match pty.read(&mut buf).await {
                        Ok(0) => {
                            info!("pty read returned 0 after {} bytes; shell exited?", total);
                            break;
                        }
                        Ok(n) => {
                            total += n;
                            if first {
                                info!("pty read first {} bytes", n);
                                first = false;
                            } else {
                                trace!("pty read {} bytes", n);
                            }
                            let data = buf[..n].to_vec();
                            if pty_output_tx.send(data.clone()).is_err() {
                                trace!("no attached clients for pty output");
                            }
                            if let Some(text) = decoder.feed(&data) {
                                if let Ok(mut vt) = vt.lock() {
                                    vt.feed(&text);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("pty read error: {e}");
                            break;
                        }
                    }
                }
                info!("pty reader task finished");
            })
        };

        // PTY writer: write input from attached clients.
        let pty_writer = {
            let pty = pty.clone();
            tokio::spawn(async move {
                info!("pty writer task started");
                while let Some(data) = pty_input_rx.recv().await {
                    trace!("pty writer received {} bytes", data.len());
                    if let Err(e) = pty.write(&data).await {
                        warn!("pty write error: {e}");
                        break;
                    }
                }
                info!("pty writer task finished");
            })
        };

        // Background capture loop.
        let capture_task = {
            let config = self.config.clone();
            let session_id = self.session_id.clone();
            let vt = vt.clone();
            tokio::spawn(async move {
                if config.capture_interval == 0 || config.disable_capture {
                    return;
                }

                let mut engine = Engine::new(config.state_dir());
                engine.diff_only = config.diff_only;
                engine.diff_mode = config.diff_mode;
                engine.text_only = config.text_only;

                let buffer = Buffer::new(config.buffer_dir(), config.buffer_size);
                let client = Client::new(
                    config.server_url.clone(),
                    config.device_id.clone().unwrap_or_else(default_device_id),
                );

                // Retry buffered captures from previous failed uploads.
                let buffered = match buffer.drain() {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("failed to drain capture buffer: {e}");
                        Vec::new()
                    }
                };
                if !buffered.is_empty() {
                    info!("retrying {} buffered captures", buffered.len());
                    match client.upload(&buffered).await {
                        Ok(()) => {
                            if let Err(e) = engine.save_hashes(&buffered) {
                                warn!("failed to save hashes after retry: {e}");
                            }
                        }
                        Err(e) => {
                            warn!("buffer retry upload failed: {e}");
                            if let Err(e) = buffer.add(&buffered) {
                                warn!("failed to re-buffer captures: {e}");
                            }
                        }
                    }
                }

                info!(
                    "background capture started (first_capture_delay={}s, interval={}s)",
                    config.first_capture_delay, config.capture_interval
                );

                if config.first_capture_delay > 0 {
                    tokio::time::sleep(Duration::from_secs(config.first_capture_delay)).await;
                }

                let mut ticker = interval(Duration::from_secs(config.capture_interval));
                loop {
                    ticker.tick().await;
                    let snapshot = build_capture(&session_id, &vt, config.text_only);
                    if let Some(capture) = engine.process(
                        &session_id,
                        &format!("{session_id}:0.0"),
                        &format!("{session_id}:0.0"),
                        snapshot.ansi,
                        snapshot.text,
                        snapshot.cols,
                        snapshot.rows,
                    ) {
                        match client.upload(std::slice::from_ref(&capture)).await {
                            Ok(()) => {
                                if let Err(e) = engine.save_hashes(std::slice::from_ref(&capture)) {
                                    warn!("failed to save hash: {e}");
                                }
                            }
                            Err(e) => {
                                warn!("upload failed: {e}; buffering capture");
                                if let Err(e) = buffer.add(std::slice::from_ref(&capture)) {
                                    warn!("failed to buffer capture: {e}");
                                }
                            }
                        }
                    }
                }
            })
        };

        // Main accept loop.
        info!("session server started: {}", self.session_id);
        let mut sigchld = signal(SignalKind::child())?;
        let result: Result<()> = async {
            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        let (stream, _) = accept?;
                        let pty_input_tx = pty_input_tx.clone();
                        let pty_output_rx = pty_output_tx.subscribe();
                        let control_tx = control_tx.clone();
                        let vt = vt.clone();
                        let session_id = self.session_id.clone();
                        let text_only = self.config.text_only;
                        tokio::spawn(async move {
                            if let Err(e) = handle_client(
                                stream,
                                session_id,
                                vt,
                                pty_input_tx,
                                pty_output_rx,
                                control_tx,
                                text_only,
                            )
                            .await
                            {
                                warn!("client handler error: {e}");
                            }
                        });
                    }

                    _ = sigchld.recv() => {
                        debug!("SIGCHLD received");
                        break;
                    }

                    msg = control_rx.recv() => {
                        match msg {
                            Some(ControlMessage::Resize(cols, rows)) => {
                                pty.resize(Winsize {
                                    ws_col: cols,
                                    ws_row: rows,
                                    ws_xpixel: 0,
                                    ws_ypixel: 0,
                                });
                                if let Ok(mut vt) = vt.lock() {
                                    vt.resize(cols as usize, rows as usize);
                                }
                                info!("resized to {cols}x{rows}");
                            }
                            Some(ControlMessage::Kill) => {
                                info!("kill requested");
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }
            Ok(())
        }
        .await;

        // Cleanup.
        let _ = tokio::fs::remove_file(&self.socket_path).await;
        pty_reader.abort();
        pty_writer.abort();
        capture_task.abort();
        info!("session server stopped: {}", self.session_id);
        result
    }
}

async fn handle_client(
    stream: UnixStream,
    session_id: String,
    vt: Arc<std::sync::Mutex<VtCapture>>,
    pty_input_tx: mpsc::Sender<Vec<u8>>,
    mut pty_output_rx: broadcast::Receiver<Vec<u8>>,
    control_tx: mpsc::Sender<ControlMessage>,
    text_only: bool,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();
    info!("client connected");

    while let Some(line) = lines.next_line().await? {
        trace!("received request line: {} bytes", line.len());
        let request: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                warn!("invalid request: {e}");
                let resp = Response::Error(format!("invalid request: {e}"));
                send_response(&mut writer, &resp).await?;
                continue;
            }
        };
        info!("handling request: {:?}", request);

        match request {
            Request::Attach => {
                info!("building attach snapshot");
                // Send current screen snapshot as initial frame.
                let snapshot = build_capture(&session_id, &vt, text_only);
                info!(
                    "attach snapshot: ansi={} text={}",
                    snapshot.ansi.len(),
                    snapshot.text.len()
                );
                send_response(&mut writer, &Response::Capture(snapshot)).await?;
                info!("attach snapshot sent");

                // Preserve any bytes already buffered by BufReader so the bridges
                // do not lose early PTY output sent right after the snapshot.
                let mut reader = lines.into_inner();

                // Bridge PTY output -> client.
                let writer = Arc::new(tokio::sync::Mutex::new(writer));
                let writer2 = writer.clone();
                let output_bridge = tokio::spawn(async move {
                    let mut total = 0usize;
                    let mut first = true;
                    info!("output bridge started");
                    while let Ok(data) = pty_output_rx.recv().await {
                        total += data.len();
                        if first {
                            info!("output bridge: first {} bytes", data.len());
                            first = false;
                        } else {
                            trace!("output bridge: {} bytes (total {})", data.len(), total);
                        }
                        if writer2.lock().await.write_all(&data).await.is_err() {
                            info!("output bridge: client write failed");
                            break;
                        }
                    }
                    info!("output bridge finished after {} bytes", total);
                });

                // Bridge client input -> PTY as raw bytes.
                let mut buf = [0u8; 4096];
                let mut total = 0usize;
                let mut first = true;
                info!("input bridge started");
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) => {
                            info!("input bridge: client disconnected after {} bytes", total);
                            break;
                        }
                        Ok(n) => {
                            total += n;
                            if first {
                                info!("input bridge: first {} bytes", n);
                                first = false;
                            } else {
                                trace!("input bridge: {} bytes (total {})", n, total);
                            }
                            if pty_input_tx.send(buf[..n].to_vec()).await.is_err() {
                                info!("input bridge: pty input channel closed");
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("input bridge: read error: {e}");
                            break;
                        }
                    }
                }

                output_bridge.abort();
                info!("client attach session ended");
                return Ok(());
            }

            Request::Resize { cols, rows } => {
                control_tx
                    .send(ControlMessage::Resize(cols, rows))
                    .await?;
                send_response(&mut writer, &Response::Ok).await?;
            }

            Request::Capture { text_only: req_text_only } => {
                let text_only = req_text_only || text_only;
                let capture = build_capture(&session_id, &vt, text_only);
                trace!(
                    "capture response: ansi={} text={}",
                    capture.ansi.len(),
                    capture.text.len()
                );
                send_response(&mut writer, &Response::Capture(capture)).await?;
            }

            Request::Status => {
                let (cols, rows) = if let Ok(vt) = vt.lock() {
                    vt.size()
                } else {
                    (80, 24)
                };
                send_response(
                    &mut writer,
                    &Response::Status(StatusResponse {
                        session_id: session_id.clone(),
                        socket_path: String::new(),
                        cols: cols as u16,
                        rows: rows as u16,
                    }),
                )
                .await?;
            }

            Request::Kill => {
                control_tx.send(ControlMessage::Kill).await?;
                send_response(&mut writer, &Response::Ok).await?;
                return Ok(());
            }
        }
    }

    info!("client disconnected without attach");
    Ok(())
}

async fn send_response(writer: &mut tokio::net::unix::OwnedWriteHalf, resp: &Response) -> Result<()> {
    let mut line = serde_json::to_vec(resp)?;
    line.push(b'\n');
    writer.write_all(&line).await?;
    Ok(())
}

fn build_capture(
    session_id: &str,
    vt: &Arc<std::sync::Mutex<VtCapture>>,
    text_only: bool,
) -> CaptureResponse {
    let (text, ansi, cols, rows) = if let Ok(vt) = vt.lock() {
        let text = vt.capture_text().join("\n");
        let ansi = if text_only {
            String::new()
        } else {
            vt.capture_ansi()
        };
        let (cols, rows) = vt.size();
        (text, ansi, cols, rows)
    } else {
        (String::new(), String::new(), 80, 24)
    };

    let content = if text_only { &text } else { &ansi };
    let hash = format!("{:x}", md5::compute(content.as_bytes()));

    CaptureResponse {
        session_id: session_id.to_owned(),
        pane_id: format!("{session_id}:0.0"),
        target: format!("{session_id}:0.0"),
        ansi,
        text,
        hash,
        cols,
        rows,
        timestamp: Utc::now(),
        metadata: {
            let mut m = HashMap::new();
            m.insert("ansi".to_owned(), (!text_only).to_string());
            m
        },
    }
}

/// Minimal UTF-8 decoder that buffers partial sequences at chunk boundaries.
struct Utf8Decoder {
    buf: Vec<u8>,
}

impl Utf8Decoder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn feed(&mut self, bytes: &[u8]) -> Option<String> {
        self.buf.extend_from_slice(bytes);
        let valid_up_to = match std::str::from_utf8(&self.buf) {
            Ok(_) => self.buf.len(),
            Err(e) => e.valid_up_to(),
        };

        if valid_up_to == 0 {
            if self.buf.len() > 4 {
                self.buf.clear();
            }
            return None;
        }

        let mut valid = self.buf.split_off(valid_up_to);
        std::mem::swap(&mut self.buf, &mut valid);
        let text = String::from_utf8_lossy(&valid).to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}
