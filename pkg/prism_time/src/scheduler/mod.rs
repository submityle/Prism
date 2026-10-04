//! **§24.6 — in-game timer / scheduler.** A deterministic "fire at `T` / after
//! `D` / every `P`" scheduler layered above the raw [`Timer`](crate::Timer).
//!
//! The design doc sketches an ergonomic API:
//!
//! ```text
//! time.schedule_after(Duration::from_secs(3), |w| spawn_wave(w));
//! time.schedule_every(Duration::from_millis(500), |w| tick_regen(w));
//! ```
//!
//! Rather than storing opaque `FnMut` closures (which cannot be made
//! deterministic, [`Clone`]d for double-run replay checks, or run in a `no_std`
//! kernel without an allocator-backed trait object per entry), this layer owns
//! the piece a time kernel can own *deterministically*: a priority queue keyed
//! by exact fire time that emits **due events carrying a user payload**, in a
//! fully reproducible order. The gameplay layer maps each payload back to its
//! action (`spawn_wave`, `tick_regen`, cooldown expiry, buff timeout, ...). See
//! the honest-boundary note in the design doc §24.9.
//!
//! Guarantees:
//! - **Deterministic advancement** — time is accumulated in exact integer
//!   nanoseconds (`u128`); no wall clock, no floating point in the schedule.
//! - **Deterministic ordering** — events due at the same instant fire in
//!   schedule order (a monotonic insertion sequence breaks ties), so two runs
//!   that schedule the same timers in the same order produce byte-identical
//!   [`Fired`] streams — the replay consistency §24.6 asks for.
//! - **Delay / periodic callbacks** — [`Scheduler::schedule_after`],
//!   [`Scheduler::schedule_at`], and [`Scheduler::schedule_every`].
//! - **Cancel / reschedule** — stable [`TimerHandle`]s survive a
//!   [`reschedule`](Scheduler::reschedule); a [`cancel`](Scheduler::cancel)
//!   retires the slot. Stale queue entries are invalidated by a generation +
//!   epoch tag, so cancellation and rescheduling are O(1) with lazy cleanup.
//!
//! The queue is a binary min-heap (`alloc::collections::BinaryHeap`) over exact
//! fire times — the classic timer-queue structure — with lazy invalidation of
//! superseded entries. `no_std + alloc`, no `unsafe`.

mod handle;
mod queue;

pub use handle::{Fired, TimerHandle, TimerKind};
pub use queue::Scheduler;
