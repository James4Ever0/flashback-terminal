use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::capture::{Capture, RetryBuffer};
use crate::log::Logger;

pub struct Client {
    http: reqwest::Client,
    server_url: String,
    device_id: String,
    buffer: Option<RetryBuffer>,
    logger: Logger,
}

#[derive(Serialize)]
struct UploadPayload<'a> {
    device_id: String,
    timestamp: DateTime<Utc>,
    captures: &'a [Capture],
}

impl Client {
    pub fn new(
        server_url: String,
        device_id: String,
        buffer: Option<RetryBuffer>,
        logger: Logger,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            server_url,
            device_id,
            buffer,
            logger,
        }
    }

    pub async fn upload(
        &self,
        captures: &[Capture],
    ) -> Result<()> {
        if captures.is_empty() {
            return Ok(());
        }

        if self.server_url.is_empty() {
            let body_len = serde_json::to_string(&UploadPayload {
                device_id: self.device_id.clone(),
                timestamp: Utc::now(),
                captures,
            })
            .map(|s| s.len())
            .unwrap_or(0);
            self.logger.info(&format!(
                "declining upload of {} capture(s) ({} bytes): no server URL configured",
                captures.len(),
                body_len
            ));
            return Err(anyhow::anyhow!("no server URL configured"));
        }

        let payload = UploadPayload {
            device_id: self.device_id.clone(),
            timestamp: Utc::now(),
            captures,
        };

        let url = format!("{}/api/captures", self.server_url.trim_end_matches('/'));
        let body = serde_json::to_vec(&payload)?;

        self.logger.info(&format!(
            "POST {url}: sending {} capture(s), {} bytes",
            captures.len(),
            body.len()
        ));

        let mut last_err = None;
        for attempt in 0..3 {
            if attempt > 0 {
                self.logger.info(&format!(
                    "retrying upload to {url} (attempt {}/3)",
                    attempt + 1
                ));
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }

            match self
                .http
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body.clone())
                .timeout(Duration::from_secs(30))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        self.logger.info(&format!(
                            "upload to {url} succeeded: {status} ({} bytes, {} capture(s))",
                            body.len(),
                            captures.len()
                        ));
                        return Ok(());
                    }
                    last_err = Some(anyhow::anyhow!("upload failed with status {status}"));
                    self.logger.warn(&format!(
                        "upload attempt {}/3 to {url} returned {status}",
                        attempt + 1
                    ));
                }
                Err(e) => {
                    self.logger.warn(&format!("upload attempt {}/3 to {url} failed: {e}", attempt + 1));
                    last_err = Some(e.into());
                }
            }
        }

        self.logger.error(&format!("upload to {url} failed after 3 attempts: {last_err:?}"));

        if let Some(buffer) = &self.buffer {
            if let Err(e) = buffer.add(captures) {
                self.logger.error(&format!("failed to buffer captures for retry: {e}"));
            } else {
                self.logger.info(&format!("buffered {} capture(s) for retry", captures.len()));
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("upload failed")))
    }

    pub async fn flush_retries(&self) -> Result<()> {
        let buffer = match &self.buffer {
            Some(b) => b,
            None => return Ok(()),
        };

        let buffered = buffer.drain()?;
        if buffered.is_empty() {
            self.logger.debug("no buffered captures to flush");
            return Ok(());
        }

        self.logger.info(&format!("flushing {} buffered capture(s)", buffered.len()));
        match self.upload(&buffered).await {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = buffer.add(&buffered);
                Err(e)
            }
        }
    }
}
