use std::process::ExitCode;

use crate::config::{Config, ConfigSources};
use crate::log::Logger;
use crate::pty;

pub async fn run(
    config: Config,
    sources: ConfigSources,
    logger: Logger,
) -> ExitCode {
    println!("terminal-capture-shell-rust-standalone configuration:");
    print_source("server_url", &sources.server_url);
    print_source("shell", &sources.shell);
    print_source("buffer_size", &sources.buffer_size);
    print_source("buffer_mode", &sources.buffer_mode);
    print_source("buffer_dir", &sources.buffer_dir);
    print_source("device_id", &sources.device_id);
    print_source("capture_interval", &sources.capture_interval);
    print_source("first_capture_delay", &sources.first_capture_delay);
    print_source("disable_capture", &sources.disable_capture);
    print_source("diff_only", &sources.diff_only);
    print_source("diff_mode", &sources.diff_mode);
    print_source("text_only", &sources.text_only);
    print_source("scrollback_lines", &sources.scrollback_lines);
    print_source("allow_nested", &sources.allow_nested);

    println!();
    println!("Validation:");

    let shell_bin = config.shell_binary();
    if !std::path::Path::new(&shell_bin).exists() {
        println!("  shell:             FAIL ({shell_bin} not found)");
        return ExitCode::from(1);
    }
    if std::env::split_paths(&std::env::var("PATH").unwrap_or_default())
        .map(|p| p.join(&shell_bin))
        .any(|p| p.exists())
        || std::path::Path::new(&shell_bin).is_absolute()
    {
        // ok
    } else {
        println!("  shell:             FAIL ({shell_bin} not executable from PATH)");
        return ExitCode::from(1);
    }
    println!("  shell:             OK ({shell_bin})");

    // Try to open a PTY with a trivial command.
    let command = vec![shell_bin, "-c".to_owned(), "exit 0".to_owned()];
    let winsize = nix::pty::Winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let extra_env = std::collections::HashMap::new();
    match pty::spawn(&command, winsize, &extra_env) {
        Ok(pty) => {
            let _ = pty.wait(None).await;
            println!("  pty:               OK");
        }
        Err(e) => {
            println!("  pty:               FAIL ({e})");
            return ExitCode::from(1);
        }
    }

    logger.info("check completed successfully");
    ExitCode::SUCCESS
}

fn print_source(name: &str, source: &crate::config::Source) {
    println!("  {name:19} {} ({})", source.value, source.origin);
}
