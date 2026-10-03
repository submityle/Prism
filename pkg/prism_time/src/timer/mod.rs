//! Timing utilities layered on the clocks: [`Stopwatch`], [`Timer`],
//! [`Cooldown`]/[`Throttle`], and [`SmoothedDelta`].
//!
//! These are the **M3** deliverables of the time kernel. They are deliberately
//! clock-agnostic: each is driven by an explicit `delta` (usually fed from the
//! active [`Time`](crate::Time) clock), which keeps them deterministic and
//! `no_std` friendly.

mod cooldown;
mod countdown;
mod smoothing;
mod stopwatch;

pub use cooldown::{Cooldown, Throttle};
pub use countdown::{Timer, TimerMode};
pub use smoothing::SmoothedDelta;
pub use stopwatch::Stopwatch;
