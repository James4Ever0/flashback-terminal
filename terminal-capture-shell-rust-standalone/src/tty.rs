use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;

use nix::libc;
use nix::pty::Winsize;
use nix::sys::termios::{self, SetArg, Termios};
use tokio::io::unix::AsyncFd;
use tokio::io::{self, Interest};

#[derive(Debug, Clone, Copy)]
pub struct TtySize(pub u16, pub u16);

impl Default for TtySize {
    fn default() -> Self {
        TtySize(80, 24)
    }
}

impl From<Winsize> for TtySize {
    fn from(winsize: Winsize) -> Self {
        TtySize(winsize.ws_col, winsize.ws_row)
    }
}

impl From<TtySize> for Winsize {
    fn from(tty_size: TtySize) -> Self {
        Winsize {
            ws_col: tty_size.0,
            ws_row: tty_size.1,
            ws_xpixel: 0,
            ws_ypixel: 0,
        }
    }
}

pub struct DevTty {
    file: AsyncFd<File>,
    settings: Arc<libc::termios>,
}

impl DevTty {
    pub async fn open() -> anyhow::Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open("/dev/tty")?;

        let file = AsyncFd::new(file)?;
        let settings = make_raw(&file)?;

        Ok(Self {
            file,
            settings: Arc::new(settings),
        })
    }

    /// Duplicate the underlying fd so read and write can happen concurrently.
    pub fn try_clone(&self) -> anyhow::Result<Self> {
        let fd = self.file.as_fd().try_clone_to_owned()?;
        let file = File::from(fd);
        let file = AsyncFd::new(file)?;
        Ok(Self {
            file,
            settings: self.settings.clone(),
        })
    }

    pub fn get_size(&self) -> Winsize {
        let mut winsize = Winsize {
            ws_row: 24,
            ws_col: 80,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { libc::ioctl(self.file.as_raw_fd(), libc::TIOCGWINSZ, &mut winsize) };
        winsize
    }

    pub async fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.file
            .async_io(Interest::READABLE, |mut file| file.read(buf))
            .await
    }

    pub async fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.file
            .async_io(Interest::WRITABLE, |mut file| file.write(buf))
            .await
    }

    pub async fn write_all(&self, mut buf: &[u8]) -> io::Result<()> {
        while !buf.is_empty() {
            let n = self.write(buf).await?;
            buf = &buf[n..];
        }
        Ok(())
    }
}

impl Drop for DevTty {
    fn drop(&mut self) {
        let termios = Termios::from(*self.settings);
        let _ = termios::tcsetattr(self.file.as_fd(), SetArg::TCSANOW, &termios);
    }
}

fn make_raw<F: AsFd>(fd: F) -> anyhow::Result<libc::termios> {
    let termios = termios::tcgetattr(fd.as_fd())?;
    let mut raw_termios = termios.clone();
    termios::cfmakeraw(&mut raw_termios);
    termios::tcsetattr(fd.as_fd(), SetArg::TCSANOW, &raw_termios)?;
    Ok(termios.into())
}

#[allow(dead_code)]
pub struct NullTty {
    size: TtySize,
}

#[allow(dead_code)]
impl NullTty {
    pub fn new(size: TtySize) -> Self {
        Self { size }
    }
}

#[allow(dead_code)]
impl NullTty {
    pub fn get_size(&self) -> Winsize {
        self.size.into()
    }

    pub async fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        std::future::pending::<()>().await;
        unreachable!()
    }

    pub async fn write(&self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }
}
