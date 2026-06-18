use crate::vt;

pub struct VtCapture {
    vt: vt::Vt,
}

impl VtCapture {
    pub fn new(cols: usize, rows: usize, scrollback: usize) -> Self {
        Self {
            vt: vt::Vt::builder()
                .size(cols, rows)
                .scrollback_limit(scrollback)
                .build(),
        }
    }

    pub fn feed(&mut self, text: &str) {
        self.vt.feed_str(text);
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.vt.resize(cols, rows);
    }

    pub fn capture_text(&self) -> Vec<String> {
        self.vt.text()
    }

    pub fn capture_ansi(&self) -> String {
        self.vt.dump()
    }

    pub fn size(&self) -> (usize, usize) {
        self.vt.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_text_and_ansi() {
        let mut vt = VtCapture::new(80, 24, 1000);
        vt.feed("\x1b[31mhello\x1b[0m world\n");

        let text = vt.capture_text();
        assert_eq!(text[0], "hello world");

        let ansi = vt.capture_ansi();
        assert!(ansi.contains("hello"));
        assert!(ansi.contains("\x1b[31m"));
    }
}
