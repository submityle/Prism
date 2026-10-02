//! `prism_ui_scheduler` — Loom's interruptible-rendering layer: priority
//! lanes, frame-budget time-slicing and cooperative, preemptible
//! reconciliation.
//!
//! A game UI must hold a frame budget (16.6 ms at 60 Hz, 8.3 ms at 120 Hz)
//! even while rebuilding a large list. This crate lets reconciliation be
//! *interrupted by a budget*: work advances one bounded step at a time and, as
//! a [`Deadline`] approaches, yields the frame and resumes on the next one —
//! so input and animation are never blocked behind a big off-screen rebuild.
//! The design mirrors React Fiber's lanes and time-slicing, but as a
//! deterministic, single-threaded, `no_std` loop with no hidden runtime.
//!
//! The retained, immutable Loom element tree is what makes "pause and resume"
//! safe: a half-finished [`Work`] unit leaves no torn mutable state, so the
//! scheduler can suspend between steps — or let a higher [`Lane`] preempt —
//! without corruption. That is the structural advantage over an imperative
//! immediate-mode UI.
//!
//! # Pieces
//!
//! * [`Lane`] / [`LaneMask`] — the six-class urgency ordering and a compact
//!   pending-lane bitset.
//! * [`Clock`] / [`FrameBudget`] / [`Deadline`] / [`ManualClock`] — the host
//!   clock abstraction and the per-frame slice it opens.
//! * [`Work`] / [`StepOutcome`] / [`FnWork`] / [`OnceWork`] — the resumable
//!   unit of reconciliation.
//! * [`Scheduler`] / [`RunReport`] / [`StopReason`] — the laned, budget-driven
//!   engine that drains work by urgency with preemption.
//! * [`VisibilityWindow`] — maps viewport visibility (from `prism_ui_virtual`)
//!   onto lanes so visible rows build before overscan before idle.
//! * [`ReconcileBatch`] / [`update_budgeted`] — the §9.5 entry point that
//!   builds a windowed list incrementally within a frame budget.
//!
//! # Example
//!
//! ```
//! use prism_ui_scheduler::{
//!     Deadline, FrameBudget, Lane, ManualClock, Scheduler, StepOutcome, Work,
//! };
//!
//! struct Count {
//!     left: u32,
//! }
//! impl Work for Count {
//!     fn step(&mut self) -> StepOutcome {
//!         self.left -= 1;
//!         if self.left == 0 { StepOutcome::Done } else { StepOutcome::More }
//!     }
//! }
//!
//! let clock = ManualClock::new(0);
//! let mut scheduler = Scheduler::new();
//! scheduler.schedule(Lane::Visible, Count { left: 3 });
//! scheduler.schedule(Lane::Input, Count { left: 1 });
//!
//! // The input unit (higher lane) is serviced before the visible one.
//! let report = scheduler.run(&clock, Deadline::NEVER);
//! assert!(report.is_drained());
//! assert_eq!(report.completed, 2);
//! let _ = FrameBudget::DEFAULT;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod budget;
pub mod lane;
pub mod reconcile;
pub mod scheduler;
pub mod visibility;
pub mod work;

pub use budget::{Clock, Deadline, FrameBudget, ManualClock};
pub use lane::{Lane, LaneMask};
pub use reconcile::{update_budgeted, ReconcileBatch};
pub use scheduler::{RunReport, Scheduler, StopReason};
pub use visibility::VisibilityWindow;
pub use work::{BoxWork, FnWork, OnceWork, StepOutcome, Work};
