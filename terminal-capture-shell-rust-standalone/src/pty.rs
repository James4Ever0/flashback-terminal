use std::collections::HashMap;
use std::ffi::{CString, NulError};
use std::os::fd::OwnedFd;
use std::os::unix::io::AsRawFd;

use nix::errno::Errno;
use nix::fcntl::{fcntl, FcntlArg, OFlag};
use nix::pty::{forkpty, ForkptyResult, Winsize};
use nix::sys::signal::{self, SigHandler, Signal};
use nix::sys::wait::{self, WaitPidFlag, WaitStatus};
use nix::unistd::{self, Pid};
use nix::libc;
use tokio::io::unix::AsyncFd;
use tokio::io::{self, Interest};
use tokio::task;

pub struct Pty {
    child: Pid,
    master: AsyncFd<OwnedFd>,
}

impl Pty {
    pub async fn read(&self, buffer: &mut [u8]) -> io::Result<usize> {
        self.master
            .async_io(Interest::READABLE, |fd| match unistd::read(fd, buffer) {
                Ok(n) => Ok(n),
                Err(Errno::EIO) => Ok(0),
                Err(e) => Err(e.into()),
            })
            .await
    }

    pub async fn write(&self, buffer: &[u8]) -> io::Result<usize> {
        self.master
            .async_io(Interest::WRITABLE, |fd| match unistd::write(fd, buffer) {
                Ok(n) => Ok(n),
                Err(Errno::EIO) => Ok(0),
                Err(e) => Err(e.into()),
            })
            .await
    }

    pub fn resize(&self, winsize: Winsize) {
        unsafe { libc::ioctl(self.master.as_raw_fd(), libc::TIOCSWINSZ, &winsize) };
    }

    pub fn kill(&self) {
        let _ = signal::kill(self.child, Signal::SIGTERM);
    }

    #[allow(dead_code)]
    pub async fn wait(&self, options: Option<WaitPidFlag>) -> io::Result<WaitStatus> {
        let pid = self.child;
        task::spawn_blocking(move || Ok(wait::waitpid(pid, options)?)).await?
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        self.kill();
        let _ = wait::waitpid(self.child, None);
    }
}

pub fn spawn<S: AsRef<str>>(
    command: &[S],
    winsize: Winsize,
    extra_env: &HashMap<String, String>,
) -> anyhow::Result<Pty> {
    let result = unsafe { forkpty(Some(&winsize), None) }?;

    match result {
        ForkptyResult::Parent { child, master } => {
            let flags = OFlag::from_bits_truncate(fcntl(&master, FcntlArg::F_GETFL)?);
            fcntl(&master,
                FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK),
            )?;
            let master = AsyncFd::new(master)?;
            Ok(Pty { child, master })
        }
        ForkptyResult::Child => {
            handle_child(command, extra_env)?;
            unreachable!();
        }
    }
}

fn handle_child<S: AsRef<str>>(
    command: &[S],
    extra_env: &HashMap<String, String>,
) -> anyhow::Result<()> {
    let command = command
        .iter()
        .map(|s| CString::new(s.as_ref()))
        .collect::<Result<Vec<CString>, NulError>>()?;

    for (k, v) in extra_env {
        std::env::set_var(k, v);
    }

    unsafe { signal::signal(Signal::SIGPIPE, SigHandler::SigDfl) }?;
    unistd::execvp(&command[0], &command)?;
    unsafe { libc::_exit(1) }
}
