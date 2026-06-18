use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::DiffMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capture {
    pub session_id: String,
    pub pane_id: String,
    pub target: String,
    pub ansi: String,
    pub text: String,
    pub hash: String,
    pub cols: usize,
    pub rows: usize,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

pub struct Engine {
    state_dir: Option<PathBuf>,
    pub diff_only: bool,
    pub diff_mode: DiffMode,
    pub text_only: bool,
    last_hash: String,
    last_text: String,
}

impl Engine {
    pub fn new(state_dir: impl AsRef<Path>) -> Self {
        let path = state_dir.as_ref();
        let state_dir = if path.as_os_str().is_empty() {
            None
        } else {
            Some(path.to_path_buf())
        };
        Self {
            state_dir,
            diff_only: true,
            diff_mode: DiffMode::Suffix,
            text_only: false,
            last_hash: String::new(),
            last_text: String::new(),
        }
    }

    pub fn process(
        &mut self,
        session_id: &str,
        pane_id: &str,
        target: &str,
        ansi: String,
        text: String,
        cols: usize,
        rows: usize,
    ) -> Option<Capture> {
        let hash_input = if self.text_only { &text } else { &ansi };
        let hash = format!("{:x}", md5::compute(hash_input.as_bytes()));

        if self.is_duplicate(session_id, pane_id, &hash) {
            return None;
        }

        let (capture_ansi, capture_text) = if self.diff_only {
            let prev_text = if self.last_text.is_empty() {
                self.load_prev(session_id, pane_id).unwrap_or_default()
            } else {
                self.last_text.clone()
            };
            if !prev_text.is_empty() {
                let prev_lines: Vec<&str> = prev_text.split('\n').collect();
                let curr_lines: Vec<&str> = text.split('\n').collect();

                let (diff, indices) = match self.diff_mode {
                    DiffMode::Suffix => diff_lines(&prev_lines, &curr_lines),
                    DiffMode::Index => diff_lines_index(&prev_lines, &curr_lines),
                };

                if diff.is_empty() {
                    let _ = self.save_prev(session_id, pane_id, &text);
                    return None;
                }

                let ansi_lines: Vec<&str> = ansi.split('\n').collect();
                let selected_ansi = if ansi_lines.len() == curr_lines.len() {
                    indices.iter().map(|i| ansi_lines[*i]).collect::<Vec<_>>().join("\n")
                } else {
                    ansi.clone()
                };

                let _ = self.save_prev(session_id, pane_id, &text);
                (selected_ansi, diff.join("\n"))
            } else {
                let _ = self.save_prev(session_id, pane_id, &text);
                (ansi, text)
            }
        } else {
            (ansi, text)
        };

        let metadata = if self.text_only {
            let mut m = HashMap::new();
            m.insert("ansi".to_owned(), "false".to_owned());
            Some(m)
        } else {
            None
        };

        self.last_hash = hash.clone();

        Some(Capture {
            session_id: session_id.to_owned(),
            pane_id: pane_id.to_owned(),
            target: target.to_owned(),
            ansi: capture_ansi,
            text: capture_text,
            hash,
            cols,
            rows,
            timestamp: Utc::now(),
            metadata,
        })
    }

    pub fn save_hashes(
        &self,
        captures: &[Capture],
    ) -> std::io::Result<()> {
        if let Some(dir) = &self.state_dir {
            fs::create_dir_all(dir)?;
            for c in captures {
                let path = dir.join(format!("{}_{}.hash", c.session_id, c.pane_id));
                fs::write(path, &c.hash)?;
            }
        }
        Ok(())
    }

    fn is_duplicate(
        &self,
        session_id: &str,
        pane_id: &str,
        hash: &str,
    ) -> bool {
        if !self.last_hash.is_empty() && self.last_hash == hash {
            return true;
        }
        if let Some(dir) = &self.state_dir {
            let path = dir.join(format!("{}_{}.hash", session_id, pane_id));
            if let Ok(prev) = fs::read_to_string(path) {
                return prev == hash;
            }
        }
        false
    }

    fn load_prev(
        &self,
        session_id: &str,
        pane_id: &str,
    ) -> Option<String> {
        if let Some(dir) = &self.state_dir {
            let path = dir.join(format!("{}_{}.prev", session_id, pane_id));
            return fs::read_to_string(path).ok();
        }
        None
    }

    fn save_prev(
        &mut self,
        session_id: &str,
        pane_id: &str,
        text: &str,
    ) -> std::io::Result<()> {
        self.last_text = text.to_owned();
        if let Some(dir) = &self.state_dir {
            fs::create_dir_all(dir)?;
            let path = dir.join(format!("{}_{}.prev", session_id, pane_id));
            fs::write(path, text)?;
        }
        Ok(())
    }
}

/// Returns the suffix of `curr` that did not exist as a contiguous block at the
/// end of `prev`. Also returns the indices of the returned lines within `curr`.
fn diff_lines(prev: &[&str], curr: &[&str]) -> (Vec<String>, Vec<usize>) {
    let max = prev.len().min(curr.len());
    for i in (1..=max).rev() {
        if prev[prev.len() - i..] == curr[..i] {
            let diff = curr[i..].iter().map(|s| s.to_string()).collect();
            let indices = (i..curr.len()).collect();
            return (diff, indices);
        }
    }
    (
        curr.iter().map(|s| s.to_string()).collect(),
        (0..curr.len()).collect(),
    )
}

/// Aligns `prev` to the length of `curr` and returns every line in `curr` that
/// differs from the aligned previous line at the same index.
fn diff_lines_index(prev: &[&str], curr: &[&str]) -> (Vec<String>, Vec<usize>) {
    let mut aligned = vec![""; curr.len()];
    match prev.len().cmp(&curr.len()) {
        std::cmp::Ordering::Greater => {
            aligned.copy_from_slice(&prev[prev.len() - curr.len()..]);
        }
        std::cmp::Ordering::Less => {
            let start = curr.len() - prev.len();
            aligned[start..].copy_from_slice(prev);
        }
        std::cmp::Ordering::Equal => {
            aligned.copy_from_slice(prev);
        }
    }

    let mut diff = Vec::new();
    let mut indices = Vec::new();
    for (i, line) in curr.iter().enumerate() {
        if *line != aligned[i] {
            diff.push(line.to_string());
            indices.push(i);
        }
    }
    (diff, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_diff_appends() {
        let prev = vec!["a", "b", "c"];
        let curr = vec!["b", "c", "d"];
        let (diff, idx) = diff_lines(&prev, &curr);
        assert_eq!(diff, vec!["d"]);
        assert_eq!(idx, vec![2]);
    }

    #[test]
    fn index_diff_in_place() {
        let prev = vec!["a", "b", "c"];
        let curr = vec!["a", "X", "c"];
        let (diff, idx) = diff_lines_index(&prev, &curr);
        assert_eq!(diff, vec!["X"]);
        assert_eq!(idx, vec![1]);
    }
}
