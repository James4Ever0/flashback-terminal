use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_SERVER_URL: &str = "";
const DEFAULT_CAPTURE_INTERVAL: u64 = 30;
const DEFAULT_FIRST_CAPTURE_DELAY: u64 = 5;
const DEFAULT_BUFFER_SIZE: usize = 100;
const DEFAULT_BUFFER_MODE: &str = "memory";
const DEFAULT_BUFFER_DIR: &str = "/tmp";
const DEFAULT_SCROLLBACK_LINES: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffMode {
    Suffix,
    Index,
}

impl Default for DiffMode {
    fn default() -> Self {
        DiffMode::Suffix
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_server_url")]
    pub server_url: String,
    #[serde(default)]
    pub shell: String,
    #[serde(default = "default_device_id")]
    pub device_id: String,
    #[serde(default = "default_capture_interval")]
    pub capture_interval: u64,
    #[serde(default = "default_first_capture_delay")]
    pub first_capture_delay: u64,
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    #[serde(default = "default_buffer_mode")]
    pub buffer_mode: String,
    #[serde(default = "default_buffer_dir")]
    pub buffer_dir: PathBuf,
    #[serde(default = "default_diff_only")]
    pub diff_only: bool,
    #[serde(default)]
    pub diff_mode: DiffMode,
    #[serde(default = "default_text_only")]
    pub text_only: bool,
    #[serde(default = "default_scrollback_lines")]
    pub scrollback_lines: usize,
    #[serde(default = "default_disable_capture")]
    pub disable_capture: bool,
    #[serde(default = "default_allow_nested")]
    pub allow_nested: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: default_server_url(),
            shell: String::new(),
            device_id: default_device_id(),
            capture_interval: default_capture_interval(),
            first_capture_delay: default_first_capture_delay(),
            buffer_size: default_buffer_size(),
            buffer_mode: default_buffer_mode(),
            buffer_dir: default_buffer_dir(),
            diff_only: default_diff_only(),
            diff_mode: DiffMode::default(),
            text_only: default_text_only(),
            scrollback_lines: default_scrollback_lines(),
            disable_capture: default_disable_capture(),
            allow_nested: default_allow_nested(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct FileConfig {
    #[serde(default)]
    server_url: Option<String>,
    #[serde(default)]
    shell: Option<String>,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    capture_interval: Option<u64>,
    #[serde(default)]
    first_capture_delay: Option<u64>,
    #[serde(default)]
    buffer_size: Option<usize>,
    #[serde(default)]
    buffer_mode: Option<String>,
    #[serde(default)]
    buffer_dir: Option<PathBuf>,
    #[serde(default)]
    diff_only: Option<bool>,
    #[serde(default)]
    diff_mode: Option<DiffMode>,
    #[serde(default)]
    text_only: Option<bool>,
    #[serde(default)]
    scrollback_lines: Option<usize>,
    #[serde(default)]
    disable_capture: Option<bool>,
    #[serde(default)]
    allow_nested: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub value: String,
    pub origin: String,
}

#[derive(Debug, Clone)]
pub struct ConfigSources {
    pub server_url: Source,
    pub shell: Source,
    pub device_id: Source,
    pub capture_interval: Source,
    pub first_capture_delay: Source,
    pub buffer_size: Source,
    pub buffer_mode: Source,
    pub buffer_dir: Source,
    pub diff_only: Source,
    pub diff_mode: Source,
    pub text_only: Source,
    pub scrollback_lines: Source,
    pub disable_capture: Source,
    pub allow_nested: Source,
}

fn default_server_url() -> String {
    DEFAULT_SERVER_URL.to_owned()
}

fn default_capture_interval() -> u64 {
    DEFAULT_CAPTURE_INTERVAL
}

fn default_first_capture_delay() -> u64 {
    DEFAULT_FIRST_CAPTURE_DELAY
}

fn default_buffer_size() -> usize {
    DEFAULT_BUFFER_SIZE
}

fn default_buffer_mode() -> String {
    DEFAULT_BUFFER_MODE.to_owned()
}

fn default_buffer_dir() -> PathBuf {
    PathBuf::from(DEFAULT_BUFFER_DIR)
}

fn default_diff_only() -> bool {
    false
}

fn default_text_only() -> bool {
    false
}

fn default_scrollback_lines() -> usize {
    DEFAULT_SCROLLBACK_LINES
}

fn default_disable_capture() -> bool {
    false
}

fn default_allow_nested() -> bool {
    false
}

pub fn default_device_id() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| format!("device-{}", std::process::id()))
}

impl Config {
    pub fn default_config_path() -> PathBuf {
        expand_tilde("~/.config/terminal-capture-shell-pty.yaml")
            .unwrap_or_else(|| PathBuf::from("/tmp/terminal-capture-shell-pty.yaml"))
    }

    pub fn load(path: Option<&Path>, env_prefix: &str) -> Result<(Self, ConfigSources)> {
        let (mut cfg, mut src) = default_config_with_sources();

        if let Some(path) = path {
            if path.exists() {
                let content = std::fs::read_to_string(path)
                    .with_context(|| format!("cannot read config file {}", path.display()))?;
                let file_cfg: FileConfig = serde_yaml::from_str(&content)
                    .with_context(|| format!("cannot parse config file {}", path.display()))?;
                apply_yaml(&mut cfg, &mut src, &file_cfg);
            }
        }

        apply_env_overrides(&mut cfg, &mut src, env_prefix);
        cfg.buffer_dir = expand_tilde(&cfg.buffer_dir.to_string_lossy())
            .unwrap_or_else(|| cfg.buffer_dir.clone());

        Ok((cfg, src))
    }

    pub fn ensure_default_config() -> Result<PathBuf> {
        let path = Self::default_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            std::fs::write(&path, DEFAULT_CONFIG_TEMPLATE)?;
        }
        Ok(path)
    }

    pub fn shell_binary(&self) -> String {
        let shell = self.shell.trim();
        if !shell.is_empty() {
            return shell.to_owned();
        }
        std::env::var("SHELL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "/bin/bash".to_owned())
    }
}

fn default_config_with_sources() -> (Config, ConfigSources) {
    let cfg = Config::default();
    let src = ConfigSources {
        server_url: Source {
            value: cfg.server_url.clone(),
            origin: "default".to_owned(),
        },
        shell: Source {
            value: cfg.shell.clone(),
            origin: "default".to_owned(),
        },
        device_id: Source {
            value: cfg.device_id.clone(),
            origin: "default".to_owned(),
        },
        capture_interval: Source {
            value: cfg.capture_interval.to_string(),
            origin: "default".to_owned(),
        },
        first_capture_delay: Source {
            value: cfg.first_capture_delay.to_string(),
            origin: "default".to_owned(),
        },
        buffer_size: Source {
            value: cfg.buffer_size.to_string(),
            origin: "default".to_owned(),
        },
        buffer_mode: Source {
            value: cfg.buffer_mode.clone(),
            origin: "default".to_owned(),
        },
        buffer_dir: Source {
            value: cfg.buffer_dir.display().to_string(),
            origin: "default".to_owned(),
        },
        diff_only: Source {
            value: cfg.diff_only.to_string(),
            origin: "default".to_owned(),
        },
        diff_mode: Source {
            value: format!("{:?}", cfg.diff_mode).to_lowercase(),
            origin: "default".to_owned(),
        },
        text_only: Source {
            value: cfg.text_only.to_string(),
            origin: "default".to_owned(),
        },
        scrollback_lines: Source {
            value: cfg.scrollback_lines.to_string(),
            origin: "default".to_owned(),
        },
        disable_capture: Source {
            value: cfg.disable_capture.to_string(),
            origin: "default".to_owned(),
        },
        allow_nested: Source {
            value: cfg.allow_nested.to_string(),
            origin: "default".to_owned(),
        },
    };
    (cfg, src)
}

fn apply_yaml(cfg: &mut Config, src: &mut ConfigSources, file: &FileConfig) {
    if let Some(v) = &file.server_url {
        cfg.server_url = v.clone();
        src.server_url = source(v.clone(), "config");
    }
    if let Some(v) = &file.shell {
        cfg.shell = v.clone();
        src.shell = source(v.clone(), "config");
    }
    if let Some(v) = &file.device_id {
        cfg.device_id = v.clone();
        src.device_id = source(v.clone(), "config");
    }
    if let Some(v) = file.capture_interval {
        cfg.capture_interval = v;
        src.capture_interval = source(v.to_string(), "config");
    }
    if let Some(v) = file.first_capture_delay {
        cfg.first_capture_delay = v;
        src.first_capture_delay = source(v.to_string(), "config");
    }
    if let Some(v) = file.buffer_size {
        cfg.buffer_size = v;
        src.buffer_size = source(v.to_string(), "config");
    }
    if let Some(v) = &file.buffer_mode {
        cfg.buffer_mode = v.clone();
        src.buffer_mode = source(v.clone(), "config");
    }
    if let Some(v) = &file.buffer_dir {
        cfg.buffer_dir = v.clone();
        src.buffer_dir = source(v.display().to_string(), "config");
    }
    if let Some(v) = file.diff_only {
        cfg.diff_only = v;
        src.diff_only = source(v.to_string(), "config");
    }
    if let Some(v) = file.diff_mode {
        cfg.diff_mode = v;
        src.diff_mode = source(format!("{:?}", v).to_lowercase(), "config");
    }
    if let Some(v) = file.text_only {
        cfg.text_only = v;
        src.text_only = source(v.to_string(), "config");
    }
    if let Some(v) = file.scrollback_lines {
        cfg.scrollback_lines = v;
        src.scrollback_lines = source(v.to_string(), "config");
    }
    if let Some(v) = file.disable_capture {
        cfg.disable_capture = v;
        src.disable_capture = source(v.to_string(), "config");
    }
    if let Some(v) = file.allow_nested {
        cfg.allow_nested = v;
        src.allow_nested = source(v.to_string(), "config");
    }
}

fn apply_env_overrides(cfg: &mut Config, src: &mut ConfigSources, prefix: &str) {
    if let Ok(v) = std::env::var(format!("{prefix}SERVER_URL")) {
        cfg.server_url = v.clone();
        src.server_url = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}SHELL")) {
        cfg.shell = v.clone();
        src.shell = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}DEVICE_ID")) {
        cfg.device_id = v.clone();
        src.device_id = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}CAPTURE_INTERVAL")) {
        if let Ok(n) = v.parse::<u64>() {
            cfg.capture_interval = n;
            src.capture_interval = source(v, "env");
        }
    }
    if let Ok(v) = std::env::var(format!("{prefix}FIRST_CAPTURE_DELAY")) {
        if let Ok(n) = v.parse::<u64>() {
            cfg.first_capture_delay = n;
            src.first_capture_delay = source(v, "env");
        }
    }
    if let Ok(v) = std::env::var(format!("{prefix}BUFFER_SIZE")) {
        if let Ok(n) = v.parse::<usize>() {
            cfg.buffer_size = n;
            src.buffer_size = source(v, "env");
        }
    }
    if let Ok(v) = std::env::var(format!("{prefix}BUFFER_MODE")) {
        cfg.buffer_mode = v.clone();
        src.buffer_mode = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}BUFFER_DIR")) {
        cfg.buffer_dir = expand_tilde(&v).unwrap_or_else(|| PathBuf::from(v.clone()));
        src.buffer_dir = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}DIFF_ONLY")) {
        let b = parse_bool(&v);
        cfg.diff_only = b;
        src.diff_only = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}DIFF_MODE")) {
        cfg.diff_mode = match v.to_lowercase().as_str() {
            "index" => DiffMode::Index,
            _ => DiffMode::Suffix,
        };
        src.diff_mode = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}TEXT_ONLY")) {
        let b = parse_bool(&v);
        cfg.text_only = b;
        src.text_only = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}SCROLLBACK_LINES")) {
        if let Ok(n) = v.parse::<usize>() {
            cfg.scrollback_lines = n;
            src.scrollback_lines = source(v, "env");
        }
    }
    if let Ok(v) = std::env::var(format!("{prefix}DISABLE_CAPTURE")) {
        let b = parse_bool(&v);
        cfg.disable_capture = b;
        src.disable_capture = source(v, "env");
    }
    if let Ok(v) = std::env::var(format!("{prefix}ALLOW_NESTED")) {
        let b = parse_bool(&v);
        cfg.allow_nested = b;
        src.allow_nested = source(v, "env");
    }
}

fn source(value: String, origin: &str) -> Source {
    Source {
        value,
        origin: origin.to_owned(),
    }
}

fn parse_bool(s: &str) -> bool {
    matches!(
        s.to_lowercase().trim(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn expand_tilde(path: &str) -> Option<PathBuf> {
    if path.starts_with("~/") {
        dirs::home_dir().map(|h| h.join(&path[2..]))
    } else if path == "~" {
        dirs::home_dir()
    } else {
        Some(PathBuf::from(path))
    }
}

const DEFAULT_CONFIG_TEMPLATE: &str = r#"# terminal-capture-shell-pty configuration
# Environment variables override values in this file.
# Available env vars (names match the config file keys):
#   FLASHBACK_SHELL_PTY_SERVER_URL
#   FLASHBACK_SHELL_PTY_SHELL
#   FLASHBACK_SHELL_PTY_BUFFER_SIZE
#   FLASHBACK_SHELL_PTY_BUFFER_MODE
#   FLASHBACK_SHELL_PTY_BUFFER_DIR
#   FLASHBACK_SHELL_PTY_DEVICE_ID
#   FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL
#   FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY
#   FLASHBACK_SHELL_PTY_DISABLE_CAPTURE
#   FLASHBACK_SHELL_PTY_DIFF_ONLY
#   FLASHBACK_SHELL_PTY_DIFF_MODE
#   FLASHBACK_SHELL_PTY_TEXT_ONLY
#   FLASHBACK_SHELL_PTY_SCROLLBACK_LINES
#   FLASHBACK_SHELL_PTY_ALLOW_NESTED

# Remote server URL for uploading captures (empty = local only)
server_url: ""

# Shell binary inside sessions (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Retry buffer backend: "memory" (default) or "disk"
buffer_mode: memory

# Parent directory for the disk retry buffer (only used when buffer_mode: disk).
# Default is /tmp, which is cleared on reboot.
buffer_dir: "/tmp"

# Device identifier sent with uploads (empty = hostname)
device_id: ""

# Seconds between background captures for 'new' command (0 = disable)
capture_interval: 30

# Seconds to wait before the first background capture after 'new' starts.
first_capture_delay: 5

# Disable background capture entirely for 'new' command
disable_capture: false

# Only capture lines that newly appeared since the previous capture.
# First capture always returns the full buffer/screen.
diff_only: false

# Diff algorithm when diff_only is true: "suffix" (default) or "index".
diff_mode: suffix

# Send only plain-text captures; omit ANSI escape codes from uploads.
text_only: false

# Maximum scrollback lines kept by the VT emulator.
scrollback_lines: 1000

# Allow starting a capture session inside another capture session.
# The shell inside a session sets FLASHBACK_SHELL=1; by default launching
# `new` while that variable is present is refused.
allow_nested: false
"#;
