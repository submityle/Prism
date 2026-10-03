//! Child-process spawning, pipes, waiting, and termination
//! (design doc §11 子进程 spawn/wait/pipe).
//!
//! A thin facade over [`std::process`] for launching external tools — asset
//! bakers, shader compilers, packagers — with explicit `argv`, environment,
//! working directory, and standard-stream wiring, then collecting their exit
//! status and captured output.
//!
//! The facade keeps the ergonomic std builder shape but gives Prism its own
//! nameable types ([`Command`], [`Child`], [`Stdio`], [`ExitStatus`],
//! [`Output`]) so the rest of the engine never imports [`std::process`]
//! directly, matching the one-facade rule in design doc §5. The captured pipe
//! handles are re-exported std types ([`ChildStdin`] / [`ChildStdout`] /
//! [`ChildStderr`]) because they already implement [`std::io::Read`] /
//! [`std::io::Write`] and need no wrapping.

use std::ffi::OsStr;
use std::path::Path;

pub use std::process::{ChildStderr, ChildStdin, ChildStdout};

/// How one of a child's standard streams should be configured.
#[derive(Debug)]
pub enum Stdio {
    /// Inherit the corresponding stream from the parent process.
    Inherit,
    /// Attach a pipe so the parent can read from / write to the stream. The
    /// matching handle becomes available on the spawned [`Child`]
    /// (`stdin` / `stdout` / `stderr`).
    Piped,
    /// Connect the stream to the null device (`/dev/null` or `NUL`).
    Null,
}

impl Stdio {
    fn into_std(self) -> std::process::Stdio {
        match self {
            Stdio::Inherit => std::process::Stdio::inherit(),
            Stdio::Piped => std::process::Stdio::piped(),
            Stdio::Null => std::process::Stdio::null(),
        }
    }
}

/// Builder for spawning a child process.
///
/// Mirrors [`std::process::Command`]: construct with [`Command::new`], chain
/// configuration, then [`Command::spawn`], [`Command::output`], or
/// [`Command::status`].
#[derive(Debug)]
pub struct Command {
    inner: std::process::Command,
}

impl Command {
    /// Start building a command that runs `program`.
    ///
    /// `program` is resolved by the OS using the inherited `PATH` unless it is
    /// an absolute or relative path.
    #[must_use]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Self {
            inner: std::process::Command::new(program),
        }
    }

    /// Append a single argument.
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.inner.arg(arg);
        self
    }

    /// Append multiple arguments.
    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.inner.args(args);
        self
    }

    /// Set an environment variable for the child.
    ///
    /// This affects only the spawned child; it is safe and does not mutate the
    /// parent's environment (unlike [`crate::process::env::set_var`]).
    pub fn env<K: AsRef<OsStr>, V: AsRef<OsStr>>(&mut self, key: K, value: V) -> &mut Self {
        self.inner.env(key, value);
        self
    }

    /// Remove an environment variable from the child's environment.
    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        self.inner.env_remove(key);
        self
    }

    /// Clear the entire environment for the child, starting from empty.
    pub fn env_clear(&mut self) -> &mut Self {
        self.inner.env_clear();
        self
    }

    /// Set the child's working directory.
    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.inner.current_dir(dir);
        self
    }

    /// Configure the child's standard input.
    pub fn stdin(&mut self, cfg: Stdio) -> &mut Self {
        self.inner.stdin(cfg.into_std());
        self
    }

    /// Configure the child's standard output.
    pub fn stdout(&mut self, cfg: Stdio) -> &mut Self {
        self.inner.stdout(cfg.into_std());
        self
    }

    /// Configure the child's standard error.
    pub fn stderr(&mut self, cfg: Stdio) -> &mut Self {
        self.inner.stderr(cfg.into_std());
        self
    }

    /// Spawn the child process, returning a handle immediately without waiting.
    ///
    /// Piped streams ([`Stdio::Piped`]) are available on the returned [`Child`].
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the spawn (for example when the
    /// program cannot be found or is not executable).
    pub fn spawn(&mut self) -> std::io::Result<Child> {
        self.inner.spawn().map(Child::from_std)
    }

    /// Spawn the child, wait for it to finish, and capture its full stdout and
    /// stderr.
    ///
    /// Stdout and stderr are captured as pipes regardless of prior
    /// configuration (matching [`std::process::Command::output`]).
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from spawning or waiting.
    pub fn output(&mut self) -> std::io::Result<Output> {
        self.inner.output().map(Output::from_std)
    }

    /// Spawn the child, wait for it to finish, and return only its exit status
    /// (standard streams inherit the parent's unless configured otherwise).
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from spawning or waiting.
    pub fn status(&mut self) -> std::io::Result<ExitStatus> {
        self.inner.status().map(ExitStatus::from_std)
    }

    /// Borrow the underlying [`std::process::Command`] for configuration the
    /// facade does not expose directly.
    #[must_use]
    pub fn as_std(&self) -> &std::process::Command {
        &self.inner
    }

    /// Mutably borrow the underlying [`std::process::Command`].
    #[must_use]
    pub fn as_std_mut(&mut self) -> &mut std::process::Command {
        &mut self.inner
    }
}

/// A handle to a running or finished child process.
///
/// Wraps [`std::process::Child`]. Dropping a [`Child`] does **not** wait for or
/// kill the process (same as std); call [`Child::wait`] or [`Child::kill`]
/// explicitly to avoid leaving zombies/orphans.
#[derive(Debug)]
pub struct Child {
    inner: std::process::Child,
}

impl Child {
    fn from_std(inner: std::process::Child) -> Self {
        Self { inner }
    }

    /// The OS process identifier of the child.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.inner.id()
    }

    /// Take the child's standard-input pipe, if it was configured with
    /// [`Stdio::Piped`]. Returns [`None`] if not piped or already taken.
    #[must_use]
    pub fn take_stdin(&mut self) -> Option<ChildStdin> {
        self.inner.stdin.take()
    }

    /// Take the child's standard-output pipe, if it was configured with
    /// [`Stdio::Piped`]. Returns [`None`] if not piped or already taken.
    #[must_use]
    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.inner.stdout.take()
    }

    /// Take the child's standard-error pipe, if it was configured with
    /// [`Stdio::Piped`]. Returns [`None`] if not piped or already taken.
    #[must_use]
    pub fn take_stderr(&mut self) -> Option<ChildStderr> {
        self.inner.stderr.take()
    }

    /// Block until the child exits, returning its [`ExitStatus`].
    ///
    /// This closes the child's stdin pipe (if still held) first, so a child
    /// blocked on reading stdin can make progress.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from waiting.
    pub fn wait(&mut self) -> std::io::Result<ExitStatus> {
        self.inner.wait().map(ExitStatus::from_std)
    }

    /// Check whether the child has exited without blocking.
    ///
    /// Returns `Ok(Some(status))` if it has exited, `Ok(None)` if still
    /// running.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the status check.
    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        Ok(self.inner.try_wait()?.map(ExitStatus::from_std))
    }

    /// Wait for the child to finish and collect all of its captured stdout /
    /// stderr into an [`Output`].
    ///
    /// Consumes the handle. Requires stdout/stderr to have been configured with
    /// [`Stdio::Piped`] to capture their bytes; otherwise the corresponding
    /// buffer is empty.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from reading the pipes or waiting.
    pub fn wait_with_output(self) -> std::io::Result<Output> {
        self.inner.wait_with_output().map(Output::from_std)
    }

    /// Forcibly terminate the child.
    ///
    /// On Unix this sends `SIGKILL`; the process may need a subsequent
    /// [`Child::wait`] to be reaped. Killing an already-exited process is not
    /// an error.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the kill request.
    pub fn kill(&mut self) -> std::io::Result<()> {
        self.inner.kill()
    }

    /// Borrow the underlying [`std::process::Child`].
    #[must_use]
    pub fn as_std(&self) -> &std::process::Child {
        &self.inner
    }

    /// Mutably borrow the underlying [`std::process::Child`].
    #[must_use]
    pub fn as_std_mut(&mut self) -> &mut std::process::Child {
        &mut self.inner
    }
}

/// The exit status of a finished child process.
///
/// Wraps [`std::process::ExitStatus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitStatus(std::process::ExitStatus);

impl ExitStatus {
    fn from_std(status: std::process::ExitStatus) -> Self {
        Self(status)
    }

    /// Returns `true` if the child exited successfully (exit code `0` on all
    /// platforms, and not terminated by a signal on Unix).
    #[must_use]
    pub fn success(self) -> bool {
        self.0.success()
    }

    /// The exit code, if the child exited normally with one.
    ///
    /// Returns [`None`] when the child was terminated by a signal (Unix) rather
    /// than exiting with a code.
    #[must_use]
    pub fn code(self) -> Option<i32> {
        self.0.code()
    }

    /// Borrow the underlying [`std::process::ExitStatus`].
    #[must_use]
    pub fn as_std(self) -> std::process::ExitStatus {
        self.0
    }
}

impl core::fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.0, f)
    }
}

/// The collected result of running a child to completion.
///
/// Wraps [`std::process::Output`]: the exit status plus fully-captured stdout
/// and stderr byte buffers.
#[derive(Clone, Debug)]
pub struct Output {
    /// The child's exit status.
    pub status: ExitStatus,
    /// Everything the child wrote to standard output.
    pub stdout: Vec<u8>,
    /// Everything the child wrote to standard error.
    pub stderr: Vec<u8>,
}

impl Output {
    fn from_std(output: std::process::Output) -> Self {
        Self {
            status: ExitStatus::from_std(output.status),
            stdout: output.stdout,
            stderr: output.stderr,
        }
    }

    /// Standard output decoded lossily as UTF-8.
    #[must_use]
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Standard error decoded lossily as UTF-8.
    #[must_use]
    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}
