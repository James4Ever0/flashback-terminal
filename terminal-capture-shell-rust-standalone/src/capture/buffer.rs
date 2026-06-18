use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Utc;
use serde_json;

use super::engine::Capture;

pub struct Buffer {
    dir: PathBuf,
    max_size: usize,
}

impl Buffer {
    pub fn new(dir: impl AsRef<Path>, max_size: usize) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            max_size,
        }
    }

    pub fn add(&self,
        captures: &[Capture],
    ) -> Result<()> {
        if captures.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{}.jsonl", Utc::now().timestamp_millis()));
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        for c in captures {
            let line = serde_json::to_vec(c)?;
            file.write_all(&line)?;
            file.write_all(b"\n")?;
        }
        self.trim()?;
        Ok(())
    }

    /// Returns all buffered captures and removes the buffer files.
    pub fn drain(&self,
    ) -> Result<Vec<Capture>> {
        let mut captures = Vec::new();
        let entries = match fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return Ok(captures),
        };

        let mut files: Vec<PathBuf> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }
        files.sort();

        for path in files {
            let file = fs::File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(c) = serde_json::from_str::<Capture>(&line) {
                    captures.push(c);
                }
            }
            let _ = fs::remove_file(&path);
        }

        Ok(captures)
    }

    fn trim(&self,
    ) -> Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };

        let mut files: Vec<(std::fs::DirEntry, PathBuf)> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                files.push((entry, path));
            }
        }

        if files.len() <= self.max_size {
            return Ok(());
        }

        files.sort_by(|a, b| {
            let ma = a.0.metadata().and_then(|m| m.modified()).ok();
            let mb = b.0.metadata().and_then(|m| m.modified()).ok();
            ma.cmp(&mb)
        });

        let to_remove = files.len().saturating_sub(self.max_size);
        for (entry, _) in files.into_iter().take(to_remove) {
            let _ = fs::remove_file(entry.path());
        }
        Ok(())
    }
}

/// In-memory bounded retry buffer. Oldest entries are dropped when the bound is
/// exceeded.
pub struct MemoryBuffer {
    max_size: usize,
    entries: std::sync::Mutex<VecDeque<Capture>>,
}

impl MemoryBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: std::sync::Mutex::new(VecDeque::new()),
        }
    }

    pub fn add(&self, captures: &[Capture]) -> Result<()> {
        if captures.is_empty() {
            return Ok(());
        }
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        for c in captures {
            entries.push_back(c.clone());
        }
        while entries.len() > self.max_size {
            entries.pop_front();
        }
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<Capture>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        Ok(std::mem::take(&mut *entries).into())
    }
}

/// Unified retry buffer used by the standalone client.
pub enum RetryBuffer {
    Memory(MemoryBuffer),
    Disk(Buffer),
}

impl RetryBuffer {
    pub fn add(&self, captures: &[Capture]) -> Result<()> {
        match self {
            RetryBuffer::Memory(b) => b.add(captures),
            RetryBuffer::Disk(b) => b.add(captures),
        }
    }

    pub fn drain(&self) -> Result<Vec<Capture>> {
        match self {
            RetryBuffer::Memory(b) => b.drain(),
            RetryBuffer::Disk(b) => b.drain(),
        }
    }
}
