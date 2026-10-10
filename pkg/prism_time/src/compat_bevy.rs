//! `bevy_time`-shaped compatibility surface.
//!
//! Prism's time types were named to line up with `bevy_time`
//! ([`Time`](crate::Time), [`Real`](crate::Real), [`Virtual`](crate::Virtual),
//! [`Fixed`](crate::Fixed), [`Timer`](crate::Timer),
//! [`TimerMode`](crate::TimerMode), [`Stopwatch`](crate::Stopwatch)), so code
//! written against `bevy_time`'s vocabulary can import this prelude and keep its
//! call sites. This module adds no new behaviour; it only re-exports the
//! existing engine-agnostic types under the familiar `bevy_time` names. It is
//! gated behind the `compat-bevy` feature and pulls in no `bevy_*` crate.

/// Drop-in replacement for `bevy_time::prelude`.
///
/// Re-exports the Prism clocks and timers under the names `bevy_time` uses.
/// The [`Fixed`](crate::Fixed) accumulator here is driven explicitly rather
/// than by a Bevy schedule, but the clock-reading surface matches.
pub mod prelude {
    pub use crate::{Fixed, Real, Stopwatch, Time, Timer, TimerMode, Virtual};
}

pub use crate::{Fixed, Real, Stopwatch, Time, Timer, TimerMode, Virtual};
