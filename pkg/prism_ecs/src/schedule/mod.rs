//! Scheduling: composing systems into a runnable [`Schedule`].
//!
//! This is the M1 schedule core of the design doc (§8). It provides a
//! recursive [`SystemConfigs`] configuration tree, ergonomic conversion via
//! [`IntoSystemConfigs`] (single systems, tuples, `.chain()`, `.run_if(..)`),
//! type-erased run conditions ([`BoxedCondition`] plus the [`resource_exists`],
//! [`resource_equals`], and [`run_once`] helpers), and a
//! [`SingleThreadedExecutor`] that drives a [`Schedule`] sequentially.
//!
//! # Honestly deferred
//!
//! The parallel conflict-graph executor + fiber job graph (§8.2–§8.3,
//! needs `prism_tasks`), `SystemSet`s, and run-conditions-as-systems are future
//! milestones. They are absent, not stubbed; the per-system
//! [`Access`](crate::query::Access) recorded by the system layer already
//! carries the information a parallel executor will need.

pub mod condition;
pub mod config;
pub mod executor;

pub use condition::{resource_equals, resource_exists, run_once, BoxedCondition};
pub use config::{IntoSystemConfigs, SystemConfigs};
pub use executor::{Schedule, SingleThreadedExecutor};

#[cfg(test)]
mod tests;
