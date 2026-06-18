use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest;
use serde::Serialize;

use crate::capture::engine::Capture;

pub struct Client {
    http: reqwest::Client,
    server_url: String,
    device_id: String,
}

#[derive(Serialize)]
struct UploadPayload<'a> {
    device_id: String,
    timestamp: DateTime<Utc>,
    captures: &'a [Capture],
}

impl Client {
    pub fn new(server_url: String, device_id: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            server_url,
            device_id,
        }
    }

    pub async fn upload(&self,
        captures: &[Capture],
    ) -> Result<()> {
        if captures.is_empty() {
            return Ok(());
        }

        let payload = UploadPayload {
            device_id: self.device_id.clone(),
            timestamp: Utc::now(),
            captures,
        };

        let url = format!("{}/api/captures", self.server_url.trim_end_matches('/'));

        let mut last_err = None;
        for attempt in 0..3 {
            match self
                .http
                .post(&url)
                .json(&payload)
                .timeout(Duration::from_secs(30))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(());
                    }
                    last_err = Some(anyhow::anyhow!(
                        "upload failed with status {status}"
                    ));
                }
                Err(e) => {
                    last_err = Some(e.into());
                }
            }
            tokio::time::sleep(Duration::from_millis(500 * (attempt + 1))).await;
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("upload failed")))
    }
}
