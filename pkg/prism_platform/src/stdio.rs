//! Standard stream handles (stdin / stdout / stderr).
//!
//! This is the standard-streams part of the M5 milestone (design doc §11
//! 标准流, §22 M5). It wires the engine's command-line front end and diagnostic
//! logging to the process's three standard streams through a thin facade over
//! [`std::io`].
//!
//! ## What this layer does and does not do
//! - It exposes the raw and line-buffered handles plus `is_terminal` probes so
//!   `prism_app` (CLI) and `prism_diagnostic` (log sink) can write without each
//!   re-deriving the platform story.
//! - It does **not** implement colored output, Windows console UTF-8 mode
//!   switching, or ANSI handling. Those are display-policy concerns the design
//!   doc assigns to higher layers; this layer only hands over correct byte
//!   streams and tells callers whether a stream is a terminal (so they can
//!   decide whether to emit color).
//! - On the host, [`std::io::Stdout`] / [`std::io::Stderr`] are already
//!   internally line-buffered and locked per write; this facade surfaces that
//!   rather than adding a second buffering layer.
//!
//! Requires the `std` feature.

use std::io::{self, IsTerminal, Stderr, Stdin, Stdout, Write};

/// Which standard stream a handle refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stream {
    /// Standard input (file descriptor 0).
    Stdin,
    /// Standard output (file descriptor 1).
    Stdout,
    /// Standard error (file descriptor 2).
    Stderr,
}

/// A handle to the process's standard output.
///
/// Thin wrapper over [`std::io::Stdout`] (which is a shared, internally locked
/// handle). Prefer [`StandardOutput::write_all`] for diagnostic byte output; it
/// takes the lock once for the whole buffer.
#[must_use]
pub fn stdout() -> StandardOutput {
    StandardOutput(io::stdout())
}

/// A handle to the process's standard error.
#[must_use]
pub fn stderr() -> StandardError {
    StandardError(io::stderr())
}

/// A handle to the process's standard input.
#[must_use]
pub fn stdin() -> StandardInput {
    StandardInput(io::stdin())
}

/// Returns `true` if the given standard stream is connected to a terminal
/// (as opposed to a pipe, file, or null device).
///
/// Upper layers use this to decide whether to emit ANSI color / progress UI.
#[must_use]
pub fn is_terminal(stream: Stream) -> bool {
    match stream {
        Stream::Stdin => io::stdin().is_terminal(),
        Stream::Stdout => io::stdout().is_terminal(),
        Stream::Stderr => io::stderr().is_terminal(),
    }
}

/// Facade handle to standard output.
///
/// Wraps [`std::io::Stdout`]; implements [`std::io::Write`] so it drops straight
/// into existing byte-sink code.
#[derive(Debug)]
pub struct StandardOutput(Stdout);

impl StandardOutput {
    /// Borrow the underlying [`std::io::Stdout`].
    #[must_use]
    pub fn as_std(&self) -> &Stdout {
        &self.0
    }

    /// Write the whole buffer to standard output, taking the lock once.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the underlying write.
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut lock = self.0.lock();
        lock.write_all(buf)?;
        lock.flush()
    }

    /// Returns `true` if standard output is connected to a terminal.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.0.is_terminal()
    }
}

impl Write for StandardOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

/// Facade handle to standard error.
///
/// Wraps [`std::io::Stderr`]; implements [`std::io::Write`].
#[derive(Debug)]
pub struct StandardError(Stderr);

impl StandardError {
    /// Borrow the underlying [`std::io::Stderr`].
    #[must_use]
    pub fn as_std(&self) -> &Stderr {
        &self.0
    }

    /// Write the whole buffer to standard error, taking the lock once.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the underlying write.
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut lock = self.0.lock();
        lock.write_all(buf)?;
        lock.flush()
    }

    /// Returns `true` if standard error is connected to a terminal.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.0.is_terminal()
    }
}

impl Write for StandardError {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

/// Facade handle to standard input.
///
/// Wraps [`std::io::Stdin`]; implements [`std::io::Read`].
#[derive(Debug)]
pub struct StandardInput(Stdin);

impl StandardInput {
    /// Borrow the underlying [`std::io::Stdin`].
    #[must_use]
    pub fn as_std(&self) -> &Stdin {
        &self.0
    }

    /// Read a single line (including the trailing newline, if any) into `buf`,
    /// returning the number of bytes read (`0` at end of input).
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] from the underlying read.
    pub fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
        self.0.read_line(buf)
    }

    /// Returns `true` if standard input is connected to a terminal.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.0.is_terminal()
    }
}

impl io::Read for StandardInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}
