use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_SERVER_URL: &str = "http://localhost:8080";
const DEFAULT_CAPTURE_INTERVAL: u64 = 30;
const DEFAULT_FIRST_CAPTURE_DELAY: u64 = 5;
const DEFAULT_BUFFER_SIZE: usize = 100;
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
    #[serde(default = "default_socket_dir")]
    pub socket_dir: PathBuf,
    pub shell: Option<String>,
    pub device_id: Option<String>,
    #[serde(default = "default_capture_interval")]
    pub capture_interval: u64,
    #[serde(default = "default_first_capture_delay")]
    pub first_capture_delay: u64,
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
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
            socket_dir: default_socket_dir(),
            shell: None,
            device_id: None,
            capture_interval: default_capture_interval(),
            first_capture_delay: default_first_capture_delay(),
            buffer_size: default_buffer_size(),
            diff_only: default_diff_only(),
            diff_mode: DiffMode::default(),
            text_only: default_text_only(),
            scrollback_lines: default_scrollback_lines(),
            disable_capture: default_disable_capture(),
            allow_nested: default_allow_nested(),
        }
    }
}

fn default_server_url() -> String {
    DEFAULT_SERVER_URL.to_owned()
}

fn default_socket_dir() -> PathBuf {
    expand_tilde(
        "~/.flashback-shell-rust/sockets"
    ).unwrap_or_else(|| PathBuf::from("/tmp/flashback-shell-rust/sockets"))
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

fn default_diff_only() -> bool {
    true
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

impl Config {
    pub fn load(path: Option<&Path>, env_prefix: &str) -> Result<Self> {
        let mut cfg = if let Some(path) = path {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read config file {}", path.display()))?;
            serde_yaml::from_str(&content)
                .with_context(|| format!("cannot parse config file {}", path.display()))?
        } else {
            Config::default()
        };

        cfg.socket_dir = expand_tilde(&cfg.socket_dir.to_string_lossy())
            .unwrap_or_else(|| cfg.socket_dir.clone());

        // Apply environment variables with the given prefix.
        if let Ok(v) = std::env::var(format!("{env_prefix}SERVER_URL")) {
            cfg.server_url = v;
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}SOCKET_DIR")) {
            cfg.socket_dir = expand_tilde(&v).unwrap_or_else(|| PathBuf::from(v));
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}SHELL")) {
            cfg.shell = Some(v);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}DEVICE_ID")) {
            cfg.device_id = Some(v);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}CAPTURE_INTERVAL")) {
            cfg.capture_interval = v.parse().unwrap_or(DEFAULT_CAPTURE_INTERVAL);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}FIRST_CAPTURE_DELAY")) {
            cfg.first_capture_delay = v.parse().unwrap_or(DEFAULT_FIRST_CAPTURE_DELAY);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}BUFFER_SIZE")) {
            cfg.buffer_size = v.parse().unwrap_or(DEFAULT_BUFFER_SIZE);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}DIFF_ONLY")) {
            cfg.diff_only = parse_bool(&v);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}DIFF_MODE")) {
            cfg.diff_mode = match v.to_lowercase().as_str() {
                "index" => DiffMode::Index,
                _ => DiffMode::Suffix,
            };
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}TEXT_ONLY")) {
            cfg.text_only = parse_bool(&v);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}SCROLLBACK_LINES")) {
            cfg.scrollback_lines = v.parse().unwrap_or(DEFAULT_SCROLLBACK_LINES);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}DISABLE_CAPTURE")) {
            cfg.disable_capture = parse_bool(&v);
        }
        if let Ok(v) = std::env::var(format!("{env_prefix}ALLOW_NESTED")) {
            cfg.allow_nested = parse_bool(&v);
        }

        Ok(cfg)
    }

    pub fn default_config_path() -> Option<PathBuf> {
        expand_tilde("~/.config/terminal-capture-shell.yaml")
    }

    pub fn shell_command(&self, args: &[String],
    ) -> Vec<String> {
        let shell = self
            .shell
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("SHELL").ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "/bin/bash".to_owned());

        let mut cmd = vec![shell];
        cmd.extend(args.iter().cloned());
        cmd
    }

    pub fn state_dir(&self) -> PathBuf {
        self.socket_dir
            .parent()
            .map(|p| p.join("state"))
            .unwrap_or_else(|| PathBuf::from("/tmp/flashback-shell-rust/state"))
    }

    pub fn buffer_dir(&self) -> PathBuf {
        self.socket_dir
            .parent()
            .map(|p| p.join("buffer"))
            .unwrap_or_else(|| PathBuf::from("/tmp/flashback-shell-rust/buffer"))
    }
}

pub fn default_device_id() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| format!("device-{}", std::process::id()))
}

fn parse_bool(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

fn expand_tilde(path: &str) -> Option<PathBuf> {
    if path.starts_with("~/") {
        dirs::home_dir().map(|h| h.join(&path[2..]))
    } else if path == "~" {
        dirs::home_dir()
    } else {
        Some(PathBuf::from(path))
    }
}
