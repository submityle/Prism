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
//! # Honestly deferred
//!
//! The parallel conflict-graph executor and fiber job graph (§8.2–§8.3, which
//! need `prism_tasks`) and the `States` state machine (§8.2/§14) are future
//! milestones. They are absent, not stubbed; the per-system
//! [`Access`](crate::query::Access) recorded by the system layer already
//! carries the information a parallel executor will need.

pub mod condition;
pub mod config;
pub mod executor;
pub mod graph;
pub mod phase;
pub mod set;

pub use condition::{
    and, not, or, resource_equals, resource_exists, run_once, BoxedCondition, Condition,
};
pub use config::{IntoSystemConfigs, SetConfig, SystemConfig, SystemConfigs};
pub use executor::SingleThreadedExecutor;
pub use graph::Schedule;
pub use phase::Phase;
pub use set::{SystemSet, SystemSetId};

#[cfg(test)]
mod tests;
