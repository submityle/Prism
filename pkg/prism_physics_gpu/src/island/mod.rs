//! Partitioning the constraint graph into independent islands and sleeping
//! those that come to rest.
//!
//! An *island* is a connected component of the constraint graph: a maximal set
//! of dynamic particles transitively coupled by constraints, together with the
//! constraints that couple them. Two particles in different islands share no
//! constraint, so their sub-solves are wholly independent — they may be solved
//! in parallel, and an entire island may be put to *sleep* as a unit once all
//! of its particles come to rest.
//!
//! [`build_islands`] performs the partition (union-find over the constraint
//! list, with pinned particles acting as shared anchors that never bridge
//! islands); [`SleepState`] tracks per-particle quiet time and flips islands
//! between awake and asleep as a unit under the thresholds in [`SleepConfig`].
//!
//! Provenance: standard solver island partitioning and velocity-threshold
//! island sleeping (as in `PhysX`, `Box2D`, Chaos). No Unreal Engine source or
//! derived code.

mod build;
mod set;
mod sleep;

pub use build::build_islands;
pub use set::{Island, IslandSet};
pub use sleep::{SleepConfig, SleepState};
