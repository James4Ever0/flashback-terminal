use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "terminal-capture-shell-rust")]
#[command(about = "Pure-Rust terminal capture tool with internal VT rendering")]
#[command(version)]
pub struct Cli {
    #[arg(short, long, global = true, help = "Path to config file")]
    pub config: Option<PathBuf>,

    #[arg(short, long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity")]
    pub verbose: u8,

    #[arg(short, long, global = true, help = "Path to log file")]
    pub log_file: Option<PathBuf>,

    #[arg(long, global = true, help = "Disable background capture")]
    pub no_capture: bool,

    #[arg(long, global = true, help = "Allow nested capture sessions")]
    pub allow_nested: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start a new shell session.
    New {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Capture all sessions and upload changes.
    Capture,

    /// List running sessions.
    List,

    /// Kill a specific session.
    Kill {
        id: String,
    },

    /// Validate dependencies and show effective config.
    Check,

    /// Internal: run background session server.
    #[command(hide = true)]
    Server {
        session_id: String,
        socket_path: String,
        cols: u16,
        rows: u16,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DiffModeArg {
    Suffix,
    Index,
}
