use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Request {
    Attach,
    Resize { cols: u16, rows: u16 },
    Capture { text_only: bool },
    Kill,
    Status,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub session_id: String,
    pub socket_path: String,
    pub cols: u16,
    pub rows: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureResponse {
    pub session_id: String,
    pub pane_id: String,
    pub target: String,
    pub ansi: String,
    pub text: String,
    pub hash: String,
    pub cols: usize,
    pub rows: usize,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Response {
    Ok,
    Status(StatusResponse),
    Capture(CaptureResponse),
    Error(String),
}
