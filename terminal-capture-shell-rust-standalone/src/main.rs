mod capture;
mod cli;
mod cmd;
mod config;
mod log;
mod pty;
mod server;
mod tty;
mod vt;
mod vt_capture;

use std::path::PathBuf;
use std::process::ExitCode;

use crate::cli::NewArgs;
use crate::config::Config;
use crate::log::{Level, Logger};

const ENV_PREFIX: &str = "FLASHBACK_SHELL_PTY_";

#[tokio::main]
async fn main() -> ExitCode {
    if should_print_help() {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let (verbose, config_path, log_file, no_capture, allow_nested, subcommand, sub_args) =
        match parse_global_args(std::env::args().skip(1).collect()) {
            Ok(parsed) => parsed,
            Err(e) => {
                eprintln!("invalid arguments: {e}");
                print_usage();
                return ExitCode::from(1);
            }
        };

    let config_path = match config_path {
        Some(p) => Some(p),
        None => match Config::ensure_default_config() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("warning: cannot create default config: {e}");
                None
            }
        },
    };

    let (mut config, mut sources) = match Config::load(config_path.as_deref(), ENV_PREFIX) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load config: {e}");
            return ExitCode::from(1);
        }
    };

    if no_capture {
        config.disable_capture = true;
        sources.disable_capture = config_source("true", "cli");
    }

    if let Some(allow_nested) = allow_nested {
        config.allow_nested = allow_nested;
        sources.allow_nested = config_source(&allow_nested.to_string(),
            "cli",
        );
    }

    let (writer, log_path) = Logger::open_writer(log_file.as_deref());
    let logger = Logger::new(verbosity_to_level(verbose), writer);
    logger.debug(&format!("using config file: {}", config_path.as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(none)".to_owned())));
    logger.debug(&format!("using log destination: {log_path}"));

    let result = match subcommand.as_deref() {
        Some("new") => match NewArgs::parse(sub_args) {
            Ok(args) => cmd::new::run(config, sources, args, logger).await,
            Err(e) => {
                logger.error(&format!("failed to parse new args: {e}"));
                ExitCode::from(1)
            }
        },
        Some("check") => cmd::check::run(config, sources, logger).await,
        Some(other) => {
            logger.error(&format!("unknown command: {other}"));
            print_usage();
            ExitCode::from(1)
        }
        None => {
            logger.error("no command specified");
            print_usage();
            ExitCode::from(1)
        }
    };

    result
}

fn should_print_help() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    args.is_empty()
        || args.iter().any(|a| a == "help" || a == "-h" || a == "--help")
}

fn parse_global_args(args: Vec<String>) -> anyhow::Result<(
    u8,
    Option<PathBuf>,
    Option<String>,
    bool,
    Option<bool>,
    Option<String>,
    Vec<String>,
)> {
    let (verbose, mut rest) = cli::extract_verbose(args);

    let mut config: Option<PathBuf> = None;
    let mut log_file: Option<String> = None;
    let mut no_capture = false;
    let mut allow_nested: Option<bool> = None;

    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        match arg.as_str() {
            "-c" => {
                if i + 1 >= rest.len() {
                    anyhow::bail!("missing value for -c");
                }
                config = Some(PathBuf::from(rest[i + 1].clone()));
                i += 2;
            }
            "--config" => {
                if i + 1 >= rest.len() {
                    anyhow::bail!("missing value for --config");
                }
                config = Some(PathBuf::from(rest[i + 1].clone()));
                i += 2;
            }
            "-l" => {
                if i + 1 >= rest.len() {
                    anyhow::bail!("missing value for -l");
                }
                log_file = Some(rest[i + 1].clone());
                i += 2;
            }
            "--no-capture" => {
                no_capture = true;
                i += 1;
            }
            "--allow-nested" => {
                allow_nested = Some(true);
                i += 1;
            }
            _ if arg.starts_with('-') => {
                anyhow::bail!("unknown global flag: {arg}");
            }
            _ => break,
        }
    }

    let subcommand = if i < rest.len() {
        Some(rest[i].clone())
    } else {
        None
    };
    let sub_args = if i + 1 < rest.len() {
        rest.split_off(i + 1)
    } else {
        Vec::new()
    };

    Ok((
        verbose,
        config,
        log_file,
        no_capture,
        allow_nested,
        subcommand,
        sub_args,
    ))
}

fn verbosity_to_level(verbose: u8) -> Level {
    match verbose {
        0 => Level::Error,
        1 => Level::Warn,
        2 => Level::Info,
        _ => Level::Debug,
    }
}

pub fn config_source(value: &str, origin: &str) -> config::Source {
    config::Source {
        value: value.to_owned(),
        origin: origin.to_owned(),
    }
}

fn print_usage() {
    println!("terminal-capture-shell-rust-standalone - Terminal session capture tool (PTY/VT)");
    println!();
    println!("Usage:");
    println!("  terminal-capture-shell-rust-standalone [global flags] <command> [command args]");
    println!();
    println!("Commands:");
    println!("  new [args...]              Start a new shell inside a PTY");
    println!("  check                      Validate dependencies and show effective config values");
    println!();
    println!("Global flags:");
    println!("  -c <path>                Config file path (default: ~/.config/terminal-capture-shell-pty.yaml)");
    println!("  -v, -vv, -vvv            Verbosity: warn, info, debug");
    println!("  -l <path>                Log output file (default: ~/.flashback-shell-pty/log/flashback-shell-pty.log)");
    println!("  -l -                     Log to stderr");
    println!("  -l /dev/stdout           Log to stdout (may corrupt PTY attach output)");
    println!("  --no-capture             Disable background capture for this session");
    println!("  --allow-nested           Allow starting a session inside an existing capture session");
    println!();
    println!("Environment variables (override config file values):");
    println!("  FLASHBACK_SHELL_PTY_SERVER_URL");
    println!("  FLASHBACK_SHELL_PTY_SHELL");
    println!("  FLASHBACK_SHELL_PTY_BUFFER_SIZE");
    println!("  FLASHBACK_SHELL_PTY_BUFFER_MODE");
    println!("  FLASHBACK_SHELL_PTY_BUFFER_DIR");
    println!("  FLASHBACK_SHELL_PTY_DEVICE_ID");
    println!("  FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL");
    println!("  FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY");
    println!("  FLASHBACK_SHELL_PTY_DISABLE_CAPTURE");
    println!("  FLASHBACK_SHELL_PTY_DIFF_ONLY");
    println!("  FLASHBACK_SHELL_PTY_DIFF_MODE");
    println!("  FLASHBACK_SHELL_PTY_TEXT_ONLY");
    println!("  FLASHBACK_SHELL_PTY_SCROLLBACK_LINES");
    println!("  FLASHBACK_SHELL_PTY_ALLOW_NESTED");
    println!();
    println!("Config file options (default: ~/.config/terminal-capture-shell-pty.yaml):");
    println!("  server_url, shell, buffer_size, buffer_mode, buffer_dir, device_id,");
    println!("  capture_interval, first_capture_delay, disable_capture, diff_only,");
    println!("  diff_mode, text_only, scrollback_lines, allow_nested");
    println!();
    println!("Config precedence: cli flags > env vars > config file > defaults");
}
