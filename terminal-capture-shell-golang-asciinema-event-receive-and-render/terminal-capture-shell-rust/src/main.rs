mod capture;
mod cli;
mod config;
mod protocol;
mod pty;
mod server;
mod session;
mod tty;
mod vt;
mod vt_capture;

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{info, trace, warn};

use crate::capture::buffer::Buffer;
use crate::capture::engine::Engine;
use crate::cli::{Cli, Commands};
use crate::config::{Config, default_device_id};
use crate::protocol::{Request, Response};
use crate::server::Client;
use crate::session::Server;
use crate::tty::DevTty;

const ENV_PREFIX: &str = "FLASHBACK_SHELL_RUST_";

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config_path: Option<PathBuf> = cli
        .config
        .clone()
        .or_else(Config::default_config_path);
    let mut config = Config::load(config_path.as_deref(), ENV_PREFIX)?;
    if cli.no_capture {
        config.disable_capture = true;
    }
    if cli.allow_nested {
        config.allow_nested = true;
    }

    init_logging(cli.verbose, cli.log_file.as_deref())?;

    match cli.command {
        Commands::New { args } => cmd_new(config, config_path, cli.verbose, cli.log_file, args).await,
        Commands::Capture => cmd_capture(config).await,
        Commands::List => cmd_list(config).await,
        Commands::Kill { id } => cmd_kill(config, &id).await,
        Commands::Check => cmd_check(config, config_path).await,
        Commands::Server {
            session_id,
            socket_path,
            cols,
            rows,
        } => {
            let args: Vec<String> = std::env::var("FLASHBACK_SHELL_RUST_COMMAND_ARGS")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let command = config.shell_command(&args);
            let server = Server::new(session_id, socket_path.into(), config, command, cols, rows);
            server.run().await
        }
    }
}

enum LogWriter {
    Stdout(std::io::Stdout),
    Stderr(std::io::Stderr),
    File(std::io::LineWriter<std::fs::File>),
}

impl std::io::Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            LogWriter::Stdout(w) => w.write(buf),
            LogWriter::Stderr(w) => w.write(buf),
            LogWriter::File(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            LogWriter::Stdout(w) => w.flush(),
            LogWriter::Stderr(w) => w.flush(),
            LogWriter::File(w) => w.flush(),
        }
    }
}

#[derive(Clone)]
struct SharedWriter(std::sync::Arc<std::sync::Mutex<LogWriter>>);

impl std::io::Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).flush()
    }
}

fn init_logging(verbose: u8, log_file: Option<&Path>) -> Result<()> {
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;

    let level = match verbose {
        0 => tracing::Level::INFO,
        1 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    let filter = tracing_subscriber::filter::LevelFilter::from_level(level);

    if let Some(path) = log_file {
        let inner: LogWriter = if path.as_os_str() == std::ffi::OsStr::new("-") {
            LogWriter::Stdout(std::io::stdout())
        } else {
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => LogWriter::File(std::io::LineWriter::new(file)),
                Err(e) => {
                    eprintln!(
                        "warning: cannot open log file {}: {e}; logging to stderr",
                        path.display()
                    );
                    LogWriter::Stderr(std::io::stderr())
                }
            }
        };

        let shared = SharedWriter(std::sync::Arc::new(std::sync::Mutex::new(inner)));
        let shared2 = shared.clone();
        let file_layer = fmt::layer()
            .with_ansi(false)
            .with_writer(move || shared2.clone())
            .with_filter(filter);
        tracing_subscriber::registry().with(file_layer).init();
    } else {
        let fmt_layer = fmt::layer()
            .with_writer(std::io::stderr)
            .with_filter(filter);
        tracing_subscriber::registry().with(fmt_layer).init();
    }

    Ok(())
}

struct SocketGuard {
    path: PathBuf,
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn cmd_new(
    config: Config,
    config_path: Option<PathBuf>,
    verbose: u8,
    log_file: Option<PathBuf>,
    args: Vec<String>,
) -> Result<()> {
    // Refuse nested sessions unless explicitly allowed.
    if std::env::var("FLASHBACK_SHELL").is_ok() && !config.allow_nested {
        anyhow::bail!(
            "refusing to start a nested capture session (set --allow-nested or FLASHBACK_SHELL_RUST_ALLOW_NESTED=1 to override)"
        );
    }

    info!("starting new session");
    let mut session_id = format!(
        "{:x}",
        md5::compute(format!(
            "{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_millis()
        ))
    );
    session_id.truncate(12);
    info!("session_id={}", session_id);

    fs::create_dir_all(&config.socket_dir).await?;
    let socket_path = config.socket_dir.join(format!("{}.sock", session_id));
    let _socket_guard = SocketGuard { path: socket_path.clone() };
    let state_dir = config.socket_dir.parent().unwrap().join("state");
    fs::create_dir_all(&state_dir).await?;

    // Always capture server logs to a file so startup/PTY errors are visible.
    let log_dir = config.socket_dir.parent().unwrap().join("logs");
    fs::create_dir_all(&log_dir).await?;
    let server_log_file = log_file.clone().or_else(|| Some(log_dir.join(format!("{}.log", session_id))));
    info!("socket_path={}", socket_path.display());
    info!("server_log_file={}", server_log_file.as_ref().map(|p| p.display().to_string()).unwrap_or_default());

    // Determine terminal size.
    let (cols, rows) = if let Ok(tty) = DevTty::open().await {
        let winsize = tty.get_size();
        if winsize.ws_col > 0 && winsize.ws_row > 0 {
            info!("tty size={}x{}", winsize.ws_col, winsize.ws_row);
            (winsize.ws_col, winsize.ws_row)
        } else {
            info!("tty reported 0x0, using default size 80x24");
            (80, 24)
        }
    } else {
        info!("no /dev/tty, using default size 80x24");
        (80, 24)
    };

    // Spawn background server process.
    let server_exe = std::env::current_exe()?;
    let args_json = serde_json::to_string(&args)?;
    let mut cmd = Command::new(server_exe);
    cmd.env("FLASHBACK_SHELL_RUST_COMMAND_ARGS", args_json)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if verbose > 0 {
        for _ in 0..verbose {
            cmd.arg("-v");
        }
    }
    if let Some(ref path) = server_log_file {
        cmd.arg("-l").arg(path);
        if path.as_os_str() == "-" {
            // Let server logs go to stdout so they are visible on the terminal.
            cmd.stdout(Stdio::inherit());
        }
    }
    if let Some(ref path) = config_path {
        cmd.arg("--config").arg(path);
    }
    let mut child = cmd
        .arg("server")
        .arg(&session_id)
        .arg(socket_path.to_string_lossy().as_ref())
        .arg(cols.to_string())
        .arg(rows.to_string())
        .spawn()
        .context("cannot spawn session server")?;
    info!("spawned server child pid={:?}", child.id());

    // Wait for socket.
    info!("waiting for server socket");
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);
    let mut checked = 0;
    while !socket_path.exists() {
        checked += 1;
        if checked % 10 == 0 {
            info!("still waiting for server socket (check {})", checked);
        }
        if tokio::time::Instant::now() > deadline {
            info!("server socket deadline reached; killing child");
            let _ = child.kill().await;
            let status = child.wait().await;
            anyhow::bail!(
                "session server did not start (child status: {:?})",
                status
            );
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }
    info!("server socket ready after {} checks", checked);

    // Connect and attach.
    info!("connecting to server socket");
    let stream = UnixStream::connect(&socket_path)
        .await
        .with_context(|| format!("cannot connect to {}", socket_path.display()))?;
    let (reader, mut writer) = stream.into_split();
    info!("connected to server socket");

    let attach = Request::Attach;
    let mut line = serde_json::to_vec(&attach)?;
    line.push(b'\n');
    writer.write_all(&line).await?;
    info!("sent Attach request");

    // Open /dev/tty for raw I/O.
    info!("opening /dev/tty");
    let tty_reader = DevTty::open().await?;
    let tty_writer = tty_reader.try_clone()?;
    let tty_size = tty_reader.get_size();
    info!("opened /dev/tty size={}x{}", tty_size.ws_col, tty_size.ws_row);

    // Drain initial snapshot response (with timeout to detect a hung server).
    let mut lines = BufReader::new(reader).lines();
    let snapshot_timeout = tokio::time::Duration::from_secs(3);
    match tokio::time::timeout(snapshot_timeout, lines.next_line()).await {
        Ok(Ok(Some(line))) => {
            info!("received initial snapshot: {} bytes", line.len());
        }
        Ok(Ok(None)) => {
            anyhow::bail!("server closed connection before snapshot");
        }
        Ok(Err(e)) => {
            anyhow::bail!("error reading initial snapshot: {e}");
        }
        Err(_) => {
            anyhow::bail!("timed out waiting for initial snapshot from server");
        }
    }

    // Preserve any bytes already buffered by BufReader so the output bridge
    // does not lose early PTY output sent right after the snapshot.
    let reader = lines.into_inner();
    info!("starting tty bridges");

    // Bridge PTY output -> TTY.
    let mut output_bridge = tokio::spawn(async move {
        let mut reader = reader;
        let mut buf = [0u8; 4096];
        let mut total = 0usize;
        let mut first = true;
        loop {
            match reader.read(&mut buf).await {
                Ok(0) => {
                    info!("output bridge: reader closed after {} bytes", total);
                    break;
                }
                Ok(n) => {
                    total += n;
                    if first {
                        info!("output bridge: first {} bytes from server", n);
                        first = false;
                    } else {
                        trace!("output bridge: {} bytes (total {})", n, total);
                    }
                    if tty_writer.write_all(&buf[..n]).await.is_err() {
                        info!("output bridge: tty write error");
                        break;
                    }
                }
                Err(e) => {
                    info!("output bridge: reader error: {}", e);
                    break;
                }
            }
        }
    });

    // Bridge TTY input -> PTY.
    let mut input_bridge = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        let mut total = 0usize;
        let mut first = true;
        loop {
            match tty_reader.read(&mut buf).await {
                Ok(0) => {
                    info!("input bridge: tty closed after {} bytes", total);
                    break;
                }
                Ok(n) => {
                    total += n;
                    if first {
                        info!("input bridge: first {} bytes from tty", n);
                        first = false;
                    } else {
                        trace!("input bridge: {} bytes (total {})", n, total);
                    }
                    let data = &buf[..n];
                    if writer.write_all(data).await.is_err() {
                        info!("input bridge: socket write error");
                        break;
                    }
                }
                Err(e) => {
                    info!("input bridge: tty read error: {}", e);
                    break;
                }
            }
        }
    });

    // Monitor SIGWINCH and send resize requests.
    info!("entering interactive loop");
    let mut sigwinch = signal(SignalKind::window_change())?;
    let resize_socket_path = socket_path.clone();
    let resize_task = tokio::spawn(async move {
        loop {
            sigwinch.recv().await;
            if let Ok(tty) = DevTty::open().await {
                let winsize = tty.get_size();
                info!("SIGWINCH resize to {}x{}", winsize.ws_col, winsize.ws_row);
                let _ = send_request(
                    &resize_socket_path,
                    Request::Resize {
                        cols: winsize.ws_col,
                        rows: winsize.ws_row,
                    },
                )
                .await;
            }
        }
    });

    // Wait for either bridge to finish.
    tokio::select! {
        _ = &mut output_bridge => info!("output bridge finished"),
        _ = &mut input_bridge => info!("input bridge finished"),
    }
    output_bridge.abort();
    input_bridge.abort();

    resize_task.abort();
    info!("sending Kill and exiting");
    let _ = send_request(
        &socket_path,
        Request::Kill,
    )
    .await;
    Ok(())
}

async fn cmd_capture(config: Config) -> Result<()> {
    let mut engine = Engine::new(config.state_dir());
    engine.diff_only = config.diff_only;
    engine.diff_mode = config.diff_mode;
    engine.text_only = config.text_only;

    let client = Client::new(
        config.server_url.clone(),
        config.device_id.clone().unwrap_or_else(default_device_id),
    );
    let buffer = Buffer::new(config.buffer_dir(), config.buffer_size);

    // First, retry buffered captures.
    let buffered = buffer.drain()?;
    if !buffered.is_empty() {
        info!("retrying {} buffered captures", buffered.len());
        if client.upload(&buffered).await.is_err() {
            buffer.add(&buffered)?;
        }
    }

    let sessions = discover_sessions(&config.socket_dir).await?;
    let mut captures = Vec::new();

    for (_session_id, socket_path) in sessions {
        let resp = send_request(&socket_path,
            Request::Capture {
                text_only: config.text_only,
            },
        )
        .await?;

        if let Response::Capture(c) = resp {
            if let Some(capture) = engine.process(
                &c.session_id,
                &c.pane_id,
                &c.target,
                c.ansi,
                c.text,
                c.cols,
                c.rows,
            ) {
                captures.push(capture);
            }
        }
    }

    if captures.is_empty() {
        info!("no changed captures");
        return Ok(());
    }

    match client.upload(&captures).await {
        Ok(()) => {
            engine.save_hashes(&captures)?;
            info!("uploaded {} captures", captures.len());
        }
        Err(e) => {
            warn!("upload failed: {e}; buffering {} captures", captures.len());
            buffer.add(&captures)?;
        }
    }

    Ok(())
}

async fn cmd_list(config: Config) -> Result<()> {
    let sessions = discover_sessions(&config.socket_dir).await?;
    if sessions.is_empty() {
        println!("no sessions");
        return Ok(());
    }
    for (id, path) in sessions {
        println!("{}\t{}", id, path.display());
    }
    Ok(())
}

async fn cmd_kill(config: Config, id: &str) -> Result<()> {
    let socket_path = config.socket_dir.join(format!("{}.sock", id));
    send_request(&socket_path, Request::Kill).await?;
    println!("killed {}", id);
    Ok(())
}

async fn cmd_check(config: Config, config_path: Option<PathBuf>) -> Result<()> {
    println!(
        "config path: {}",
        config_path
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".to_owned())
    );
    println!("server_url: {}", config.server_url);
    println!("socket_dir: {}", config.socket_dir.display());
    println!("device_id: {}", config.device_id.as_deref().unwrap_or("(auto)"));
    println!("capture_interval: {}", config.capture_interval);
    println!("diff_only: {}", config.diff_only);
    println!("diff_mode: {:?}", config.diff_mode);
    println!("text_only: {}", config.text_only);
    println!("shell: {}", config.shell_command(&[]).join(" "));

    // Check for musl target.
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .await?;
    let targets = String::from_utf8_lossy(&output.stdout);
    if targets.contains("x86_64-unknown-linux-musl") {
        println!("musl target: installed");
    } else {
        println!("musl target: not installed (run: rustup target add x86_64-unknown-linux-musl)");
    }

    // Verify Cargo.lock has no image crates (only works after first build).
    let lock_path = std::env::current_exe()?
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("Cargo.lock");
    if lock_path.exists() {
        let lock = std::fs::read_to_string(lock_path)?;
        let bad = ["name = \"png\"", "name = \"image\"", "name = \"libpng\""]
            .iter()
            .any(|k| lock.contains(k));
        println!("image crates in lock: {}", if bad { "yes" } else { "no" });
    }

    Ok(())
}

async fn send_request(socket_path: &Path, request: Request) -> Result<Response> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("cannot connect to {}", socket_path.display()))?;

    let mut line = serde_json::to_vec(&request)?;
    line.push(b'\n');
    stream.write_all(&line).await?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    let resp: Response = serde_json::from_str(&line)
        .with_context(|| format!("invalid response: {line}"))?;
    Ok(resp)
}

async fn discover_sessions(socket_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut sessions = Vec::new();
    if !socket_dir.exists() {
        return Ok(sessions);
    }

    let mut entries = fs::read_dir(socket_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("sock") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                sessions.push((stem.to_owned(), path));
            }
        }
    }
    sessions.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(sessions)
}

