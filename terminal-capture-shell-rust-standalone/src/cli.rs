use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct NewArgs {
    pub capture_file: Option<String>,
    pub capture_text_file: Option<String>,
    pub vt_log_file: Option<String>,
    pub vt_log_interval: Duration,
    pub shell_args: Vec<String>,
}

impl NewArgs {
    pub fn parse(mut args: Vec<String>) -> anyhow::Result<Self> {
        let mut result = Self {
            vt_log_interval: Duration::from_secs(2),
            ..Default::default()
        };

        let mut i = 0;
        'parse: while i < args.len() {
            let arg = &args[i];

            if arg == "--" {
                i += 1;
                break;
            }

            if !arg.starts_with('-') {
                break;
            }

            if !arg.starts_with("--") {
                // First short-looking argument is treated as a shell argument.
                break;
            }

            match arg.as_str() {
                "--capture" => {
                    if i + 1 >= args.len() {
                        anyhow::bail!("missing value for --capture");
                    }
                    result.capture_file = Some(args[i + 1].clone());
                    i += 2;
                }
                _ if arg.starts_with("--capture=") => {
                    result.capture_file = Some(arg.strip_prefix("--capture=").unwrap().to_owned());
                    i += 1;
                }
                "--capture-text" => {
                    if i + 1 >= args.len() {
                        anyhow::bail!("missing value for --capture-text");
                    }
                    result.capture_text_file = Some(args[i + 1].clone());
                    i += 2;
                }
                _ if arg.starts_with("--capture-text=") => {
                    result.capture_text_file =
                        Some(arg.strip_prefix("--capture-text=").unwrap().to_owned());
                    i += 1;
                }
                "--vt-log" => {
                    if i + 1 >= args.len() {
                        anyhow::bail!("missing value for --vt-log");
                    }
                    result.vt_log_file = Some(args[i + 1].clone());
                    i += 2;
                }
                _ if arg.starts_with("--vt-log=") => {
                    result.vt_log_file = Some(arg.strip_prefix("--vt-log=").unwrap().to_owned());
                    i += 1;
                }
                "--vt-log-interval" => {
                    if i + 1 >= args.len() {
                        anyhow::bail!("missing value for --vt-log-interval");
                    }
                    result.vt_log_interval = humantime::parse_duration(&args[i + 1])
                        .map_err(|e| anyhow::anyhow!("invalid --vt-log-interval: {e}"))?;
                    i += 2;
                }
                _ if arg.starts_with("--vt-log-interval=") => {
                    let value = arg.strip_prefix("--vt-log-interval=").unwrap();
                    result.vt_log_interval = humantime::parse_duration(value)
                        .map_err(|e| anyhow::anyhow!("invalid --vt-log-interval: {e}"))?;
                    i += 1;
                }
                _ => {
                    // Unknown long flag: treat it and the rest as shell args.
                    break 'parse;
                }
            }
        }

        result.shell_args = args.split_off(i);

        if let Some(first) = result.shell_args.first() {
            if first.trim_start_matches('-') == "no-capture" {
                anyhow::bail!("--no-capture is a global flag and must appear before the subcommand (e.g. terminal-capture-shell-rust-standalone --no-capture new)");
            }
        }

        Ok(result)
    }
}

/// Parse -v/-vv/-vvv out of args and return (count, remaining).
pub fn extract_verbose(args: Vec<String>) -> (u8, Vec<String>) {
    let mut count = 0u8;
    let mut rest = Vec::new();
    for arg in args {
        match arg.as_str() {
            "-v" => count = count.saturating_add(1),
            "-vv" => count = count.saturating_add(2),
            "-vvv" => count = count.saturating_add(3),
            _ => rest.push(arg),
        }
    }
    (count, rest)
}
