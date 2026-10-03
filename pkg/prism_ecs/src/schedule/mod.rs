//! Scheduling: composing systems into a runnable [`Schedule`].
//!
//! This is the M1 schedule layer of the design doc (§8.2). It unifies several
//! pieces into one ordering model:
//!
//! * a recursive [`SystemConfigs`] configuration tree with ergonomic conversion
//!   via [`IntoSystemConfigs`] — single systems, tuples, `.chain()`,
//!   `.run_if(..)`, `.in_set(..)`, `.before(..)`/`.after(..)`, `.in_phase(..)`;
//! * the six fixed [`Phase`]s (`First → … → Last`), pre-chained so a system in
//!   an earlier phase always precedes one in a later phase;
//! * named [`SystemSet`]s with set-level ordering edges and shared
//!   run-conditions ([`SetConfig`], applied via
//!   [`Schedule::configure_set`](graph::Schedule::configure_set));
//! * type-erased run conditions ([`BoxedCondition`]/[`Condition`] plus the
//!   [`resource_exists`], [`resource_equals`], [`run_once`], [`not`], [`and`],
//!   and [`or`] helpers);
//! * a [`Schedule`] that resolves all of the above into one directed graph and
//!   topologically sorts it (insertion order as the deterministic tie-break;
//!   a contradictory cycle panics — the deterministic analogue of the §23.4
//!   ambiguity hard-gate);
//! * a [`SingleThreadedExecutor`] that drives the resolved order, honouring set
//!   and system run-conditions.
//!
//! * (with the `multi_thread` feature) a [`MultiThreadedExecutor`] that
//!   partitions the resolved order into conflict-free *waves* and dispatches
//!   each wave onto a [`prism_tasks::TaskPool`] (§8.2 conflict-graph executor,
//!   §24.1 dispatch base), preserving the single-threaded result for
//!   ambiguity-free schedules.
//!
//! # Honestly deferred
//!
//! The fiber job graph for *intra*-system chunk parallelism (§8.3) is a future
//! milestone (M3). It is absent, not stubbed; the per-system
//! [`Access`](crate::query::Access) recorded by the system layer already
//! carries the information both executors need.

pub mod ambiguity;
pub mod condition;
pub mod config;
pub mod executor;
pub mod graph;
pub mod label;
#[cfg(feature = "multi_thread")]
pub mod parallel_executor;
pub mod phase;
pub mod schedules;
pub mod set;
pub mod state;

pub use ambiguity::{Ambiguities, Ambiguity};
pub use condition::{
    and, not, or, resource_equals, resource_exists, run_once, BoxedCondition, Condition,
};
pub use config::{IntoSystemConfigs, SetConfig, SystemConfig, SystemConfigs};
pub use executor::SingleThreadedExecutor;
pub use graph::Schedule;
pub use label::{BoxedScheduleLabel, ScheduleLabel};
#[cfg(feature = "multi_thread")]
pub use parallel_executor::MultiThreadedExecutor;
pub use phase::Phase;
pub use schedules::Schedules;
pub use set::{SystemSet, SystemSetId};
pub use state::{apply_state_transition, in_state, NextState, OnEnter, OnExit, State, States};

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "multi_thread"))]
mod parallel_tests;
