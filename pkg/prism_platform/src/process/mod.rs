//! Process, environment, and command-line facade (design doc §11, §22 M5).
//!
//! The M5 "进程/环境/标准流" layer, minus the standard streams (which live in
//! [`crate::stdio`]). It collects everything about *this* process and its
//! *children* into one place so the engine never reaches for [`std::env`] or
//! [`std::process`] directly:
//!
//! - [`args`] — raw command-line arguments (`argv`) for the CLI/config layer.
//! - [`env`] — environment-variable read / iterate / set / remove.
//! - [`child`] — spawning child processes with `argv`/env/cwd, captured
//!   pipes, waiting, exit status, and kill.
//! - [`signal`] — allocation-free process termination notification. The native
//!   callback only sets an atomic bit; an app runner polls it on its frame
//!   thread and performs ordinary vetoable shutdown.
//!
//! Requires the `std` feature.

pub mod args;
pub mod child;
pub mod env;
pub mod signal;

pub use child::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Output, Stdio};

#[cfg(test)]
mod tests;
