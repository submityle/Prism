//! **M4 — determinism.** Exact rational/fixed-point stepping, an integer tick
//! counter, and replay/restore.
//!
//! The variable-nanosecond [`Time<Fixed>`](crate::Time) accumulator is fine for
//! presentation but drifts for steps like `1/60 s`, which has no exact
//! nanosecond representation. This module adds a drift-free alternative:
//!
//! - [`RationalStep`]: an exact `num/den` nanosecond timestep.
//! - [`TickClock`]: a fixed-timestep accumulator that advances an **integer
//!   tick counter** by exact integer arithmetic, so the tick count is
//!   bit-identical across runs — the basis for deterministic physics and
//!   rollback networking.
//! - [`TickSnapshot`]: a serializable snapshot of the tick state
//!   (tick + accumulator + step) that [`TickClock::restore`] can reload so a
//!   replay reproduces identical stepping.
//!
//! The [`TickClock`] API intentionally mirrors [`Time<Fixed>`](crate::Time)
//! (`accumulate`/`expend`/`expend_all`/`overstep_fraction`) so it is a drop-in,
//! drift-free replacement in the classic "fixed timestep with accumulator"
//! loop.

mod rational;
mod tick;

pub use rational::RationalStep;
pub use tick::{TickClock, TickSnapshot};
