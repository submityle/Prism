//! **M5 — network time synchronization.** Server tick, clock-offset
//! estimation, and the interpolation delay buffer.
//!
//! Networked simulation needs three cooperating pieces, each in its own
//! submodule:
//!
//! - [`ServerTick`]: an authoritative fixed-rate tick clock (tick number +
//!   accumulator), built on the deterministic [`TickClock`](crate::TickClock)
//!   so server and client agree on tick numbering with zero drift.
//! - [`ClockOffsetEstimator`] + [`ClockSync`]: NTP-style round-trip sampling to
//!   estimate the client↔server clock offset, with outlier-resistant filtering
//!   and *smooth, rate-limited* convergence so a corrected offset never snaps
//!   (which would jitter or teleport remote entities).
//! - [`InterpolationBuffer`]: a buffered snapshot timeline rendered at a
//!   configurable interpolation delay, so remote entities play back smoothly in
//!   "the recent past" even under jitter and packet loss.
//!
//! The design-doc risk note is explicit: offset-estimate jitter/jumps cause
//! remote entities to stutter or teleport, so the smoothing strategy and buffer
//! length must be robust. The APIs here are built around that constraint.

mod interp_buffer;
mod offset;
mod server_tick;

pub use interp_buffer::{InterpolationBuffer, Lerp, Sampled};
pub use offset::{ClockOffsetEstimator, ClockSync, OffsetSample};
pub use server_tick::ServerTick;
