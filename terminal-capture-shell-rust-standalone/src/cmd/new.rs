use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use nix::pty::Winsize;
use nix::sys::wait::WaitStatus;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Mutex;
use tokio::time::{interval, sleep};

use crate::capture::buffer::{Buffer, MemoryBuffer, RetryBuffer};
use crate::capture::engine::Engine;
use crate::cli::NewArgs;
use crate::config::{Config, ConfigSources};
use crate::log::Logger;
use crate::pty;
use crate::server::Client;
use crate::tty::DevTty;
use crate::vt_capture::VtCapture;

pub async fn run(
    config: Config,
    _sources: ConfigSources,
    args: NewArgs,
    logger: Logger,
) -> ExitCode {
    if !config.allow_nested && std::env::var("FLASHBACK_SHELL").is_ok() {
        logger.error(
            "refusing to start a nested capture session (FLASHBACK_SHELL is set); use --allow-nested or set allow_nested: true to override",
        );
        return ExitCode::from(1);
    }

    logger.info(&format!(
        "effective config: server_url={} shell={} buffer_size={} buffer_mode={} buffer_dir={} device_id={} capture_interval={} first_capture_delay={} disable_capture={} diff_only={} diff_mode={:?} text_only={} scrollback_lines={} allow_nested={}",
        config.server_url,
        config.shell_binary(),
        config.buffer_size,
        config.buffer_mode,
        config.buffer_dir.display(),
        config.device_id,
        config.capture_interval,
        config.first_capture_delay,
        config.disable_capture,
        config.diff_only,
        config.diff_mode,
        config.text_only,
        config.scrollback_lines,
        config.allow_nested,
    ));

    let shell_bin = config.shell_binary();
    let shell_args = if args.shell_args.is_empty() {
        vec!["-l".to_owned()]
    } else {
        args.shell_args
    };
    let command = vec![shell_bin]
        .into_iter()
        .chain(shell_args.into_iter())
        .collect::<Vec<_>>();

    // Open /dev/tty for raw I/O and size.
    let tty_reader = match DevTty::open().await {
        Ok(t) => t,
        Err(e) => {
            logger.error(&format!("failed to open /dev/tty: {e}"));
            return ExitCode::from(1);
        }
    };
    let tty_writer = match tty_reader.try_clone() {
        Ok(t) => t,
        Err(e) => {
            logger.error(&format!("failed to clone /dev/tty: {e}"));
            return ExitCode::from(1);
        }
    };
    let tty_resizer = match tty_reader.try_clone() {
        Ok(t) => t,
        Err(e) => {
            logger.error(&format!("failed to clone /dev/tty: {e}"));
            return ExitCode::from(1);
        }
    };
    let size = tty_reader.get_size();
    let (cols, rows) = if size.ws_col > 0 && size.ws_row > 0 {
        (size.ws_col, size.ws_row)
    } else {
        logger.warn("tty reported 0x0, using default size 80x24");
        (80, 24)
    };

    // Spawn PTY.
    let winsize = Winsize {
        ws_col: cols,
        ws_row: rows,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let mut extra_env = HashMap::<String, String>::new();
    extra_env.insert("FLASHBACK_SHELL".to_owned(), "1".to_owned());
    if std::env::var("TERM").is_err() {
        extra_env.insert("TERM".to_owned(), "xterm-256color".to_owned());
    }

    let pty = match pty::spawn(&command, winsize, &extra_env) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            logger.error(&format!("failed to start pty: {e}"));
            return ExitCode::from(1);
        }
    };

    // VT emulator.
    let vt = Arc::new(Mutex::new(VtCapture::new(
        cols as usize,
        rows as usize,
        config.scrollback_lines,
    )));

    // Broadcast channel for background-task shutdown.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    // VT log task.
    let vt_log_task = if let Some(vt_log_path) = args.vt_log_file {
        let vt = vt.clone();
        let interval_duration = args.vt_log_interval;
        let logger = logger.clone();
        let mut shutdown = shutdown_tx.subscribe();
        Some(tokio::spawn(async move {
            let mut ticker = interval(interval_duration);
            loop {
                tokio::select! {
                    _ = shutdown.recv() => break,
                    _ = ticker.tick() => {}
                }
                let (ansi, cols, rows) = {
                    let vt = vt.lock().await;
                    let (cols, rows) = vt.size();
                    (vt.capture_ansi(), cols as u16, rows as u16)
                };
                let line = format!(
                    "--- {} {}x{} ---\n{}\n",
                    chrono::Local::now().to_rfc3339(),
                    cols,
                    rows,
                    ansi
                );
                if let Err(e) = tokio::fs::write(&vt_log_path, line).await {
                    logger.warn(&format!("failed to write vt log: {e}"));
                }
            }
        }))
    } else {
        None
    };

    // Background capture.
    let (retry_buffer, retry_cleanup) = new_retry_buffer(&config);
    let capture_task = if !config.disable_capture && config.capture_interval > 0 {
        let config = config.clone();
        let vt = vt.clone();
        let logger = logger.clone();
        let shutdown = shutdown_tx.subscribe();
        let client = Client::new(
            config.server_url.clone(),
            config.device_id.clone(),
            retry_buffer,
            logger.clone(),
        );
        Some(tokio::spawn(async move {
            background_capture(config, vt, logger, shutdown, client).await;
        }))
    } else {
        None
    };

    // Wait for child exit in a blocking task.
    let pty_for_wait = pty.clone();
    let (exit_tx, mut exit_rx) = tokio::sync::watch::channel::<Option<WaitStatus>>(None);
    tokio::spawn(async move {
        let status = match pty_for_wait.wait(None).await {
            Ok(status) => Some(status),
            Err(_) => None,
        };
        let _ = exit_tx.send(status);
    });

    // Signals.
    let mut sigint = match signal(SignalKind::interrupt()) {
        Ok(s) => s,
        Err(e) => {
            logger.error(&format!("failed to register SIGINT handler: {e}"));
            return ExitCode::from(1);
        }
    };
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            logger.error(&format!("failed to register SIGTERM handler: {e}"));
            return ExitCode::from(1);
        }
    };
    let mut sighup = match signal(SignalKind::hangup()) {
        Ok(s) => s,
        Err(e) => {
            logger.error(&format!("failed to register SIGHUP handler: {e}"));
            return ExitCode::from(1);
        }
    };
    let mut sigwinch = match signal(SignalKind::window_change()) {
        Ok(s) => s,
        Err(e) => {
            logger.error(&format!("failed to register SIGWINCH handler: {e}"));
            return ExitCode::from(1);
        }
    };

    // Input bridge: /dev/tty -> PTY.
    let logger_input = logger.clone();
    let pty_for_input = pty.clone();
    let input_bridge = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        let mut total = 0usize;
        loop {
            match tty_reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    total += n;
                    if let Err(e) = pty_for_input.write(&buf[..n]).await {
                        logger_input.debug(&format!("pty write error: {e}"));
                        break;
                    }
                }
                Err(e) => {
                    logger_input.debug(&format!("tty read error: {e}"));
                    break;
                }
            }
        }
        logger_input.debug(&format!("input bridge finished after {total} bytes"));
    });

    // Output bridge: PTY -> /dev/tty + VT.
    let logger_output = logger.clone();
    let pty_for_output = pty.clone();
    let vt_for_output = vt.clone();
    let output_bridge = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        let mut total = 0usize;
        loop {
            match pty_for_output.read(&mut buf).await {
                Ok(0) => {
                    logger_output.debug("pty read returned 0; shell exited");
                    break;
                }
                Ok(n) => {
                    total += n;
                    if let Err(e) = tty_writer.write_all(&buf[..n]).await {
                        logger_output.debug(&format!("tty write error: {e}"));
                        break;
                    }
                    let text = String::from_utf8_lossy(&buf[..n]);
                    vt_for_output.lock().await.feed(&text);
                }
                Err(e) => {
                    logger_output.debug(&format!("pty read error: {e}"));
                    break;
                }
            }
        }
        logger_output.debug(&format!("output bridge finished after {total} bytes"));
    });

    // Resize handler.
    let pty_for_resize = pty.clone();
    let vt_for_resize = vt.clone();
    let logger_resize = logger.clone();
    let resize_task = tokio::spawn(async move {
        loop {
            sigwinch.recv().await;
            let size = tty_resizer.get_size();
            if size.ws_col > 0 && size.ws_row > 0 {
                let winsize = Winsize {
                    ws_col: size.ws_col,
                    ws_row: size.ws_row,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };
                pty_for_resize.resize(winsize);
                vt_for_resize
                    .lock()
                    .await
                    .resize(size.ws_col as usize, size.ws_row as usize);
                logger_resize.debug(&format!(
                    "resized to {}x{}",
                    size.ws_col, size.ws_row
                ));
            }
        }
    });

    // Wait for exit or signal.
    let exit_status = tokio::select! {
        _ = sigint.recv() => {
            logger.info("received SIGINT, shutting down");
            pty.kill();
            None
        }
        _ = sigterm.recv() => {
            logger.info("received SIGTERM, shutting down");
            pty.kill();
            None
        }
        _ = sighup.recv() => {
            logger.info("received SIGHUP, shutting down");
            pty.kill();
            None
        }
        _ = exit_rx.changed() => {
            logger.info("shell process exited");
            *exit_rx.borrow()
        }
    };

    // If a signal triggered shutdown, wait briefly for the child to exit.
    let exit_status = match exit_status {
        Some(s) => Some(s),
        None => match tokio::time::timeout(Duration::from_secs(5), exit_rx.changed()).await {
            Ok(Ok(())) => *exit_rx.borrow(),
            _ => {
                logger.warn("timed out waiting for shell to exit after signal");
                None
            }
        },
    };

    let exit_code = exit_status.map(exit_code_for_status).unwrap_or(ExitCode::SUCCESS);

    // Let output bridge drain.
    let _ = tokio::time::timeout(Duration::from_secs(1), output_bridge).await;
    input_bridge.abort();
    resize_task.abort();

    // Cancel background capture and VT log.
    let _ = shutdown_tx.send(());
    if let Some(t) = capture_task {
        let _ = tokio::time::timeout(Duration::from_secs(2), t).await;
    }
    if let Some(cleanup) = retry_cleanup {
        cleanup();
    }
    if let Some(t) = vt_log_task {
        let _ = tokio::time::timeout(Duration::from_secs(1), t).await;
    }

    // Final capture writes.
    let final_ansi = {
        let vt = vt.lock().await;
        vt.capture_ansi()
    };
    let final_text = {
        let vt = vt.lock().await;
        vt.capture_text().join("\n")
    };

    if let Some(path) = args.capture_file {
        if let Err(e) = tokio::fs::write(&path, &final_ansi).await {
            logger.warn(&format!("failed to write ansi capture: {e}"));
        }
    }
    if let Some(path) = args.capture_text_file {
        if let Err(e) = tokio::fs::write(&path, &final_text).await {
            logger.warn(&format!("failed to write text capture: {e}"));
        }
    }

    exit_code
}

fn exit_code_for_status(status: WaitStatus) -> ExitCode {
    match status {
        WaitStatus::Exited(_, code) => ExitCode::from(code as u8),
        WaitStatus::Signaled(_, sig, _) => ExitCode::from(128 + sig as u8),
        _ => ExitCode::SUCCESS,
    }
}

async fn background_capture(
    config: Config,
    vt: Arc<Mutex<VtCapture>>,
    logger: Logger,
    mut shutdown: tokio::sync::broadcast::Receiver<()>,
    client: Client,
) {
    if let Err(e) = client.flush_retries().await {
        logger.debug(&format!("flush retries: {e}"));
    }

    let mut engine = Engine::new(PathBuf::new());
    engine.diff_only = config.diff_only;
    engine.diff_mode = config.diff_mode;
    engine.text_only = config.text_only;

    logger.info(&format!(
        "background capture started (first_capture_delay={}s, interval={}s)",
        config.first_capture_delay, config.capture_interval
    ));

    if config.first_capture_delay > 0 {
        tokio::select! {
            _ = shutdown.recv() => {
                logger.info("background capture stopped");
                return;
            }
            _ = sleep(Duration::from_secs(config.first_capture_delay)) => {}
        }
    }

    do_capture(
        &mut engine,
        &client,
        &vt,
        &logger,
    )
    .await;

    let mut ticker = interval(Duration::from_secs(config.capture_interval));
    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                logger.info("background capture stopped");
                return;
            }
            _ = ticker.tick() => {
                do_capture(
                    &mut engine,
                    &client,
                    &vt,
                    &logger,
                )
                .await;
            }
        }
    }
}

async fn do_capture(
    engine: &mut Engine,
    client: &Client,
    vt: &Arc<Mutex<VtCapture>>,
    logger: &Logger,
) {
    let (ansi, text, cols, rows) = {
        let vt = vt.lock().await;
        (
            vt.capture_ansi(),
            vt.capture_text().join("\n"),
            vt.size().0,
            vt.size().1,
        )
    };

    if let Some(capture) = engine.process("main", "main", "main", ansi, text, cols, rows) {
        logger.info("background capture: changed");
        match client.upload(std::slice::from_ref(&capture)).await {
            Ok(()) => {
                if let Err(e) = engine.save_hashes(std::slice::from_ref(&capture)) {
                    logger.warn(&format!("failed to save hash: {e}"));
                }
            }
            Err(e) => {
                logger.debug(&format!("background upload: {e}"));
            }
        }
    }
}

fn new_retry_buffer(
    config: &Config,
) -> (Option<RetryBuffer>, Option<Box<dyn FnOnce()>>) {
    if config.buffer_mode.to_lowercase() == "disk" {
        let parent = &config.buffer_dir;
        let dir = match std::fs::create_dir_all(parent)
            .and_then(|_| tempfile::tempdir_in(parent))
        {
            Ok(d) => {
                let path = d.path().to_path_buf();
                std::mem::forget(d);
                path
            }
            Err(e) => {
                eprintln!(
                    "warning: cannot create disk buffer dir in {}: {e}; using memory",
                    parent.display()
                );
                return (
                    Some(RetryBuffer::Memory(MemoryBuffer::new(config.buffer_size))),
                    None,
                );
            }
        };
        let cleanup: Box<dyn FnOnce()> = {
            let dir = dir.clone();
            Box::new(move || {
                let _ = std::fs::remove_dir_all(&dir);
            })
        };
        (
            Some(RetryBuffer::Disk(Buffer::new(dir, config.buffer_size))),
            Some(cleanup),
        )
    } else {
        (
            Some(RetryBuffer::Memory(MemoryBuffer::new(config.buffer_size))),
            None,
        )
    }
}
