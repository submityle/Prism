//! Broad-phase candidate-pair generation over a [`DynamicBvh`].
//!
//! [`DynamicBvh`]: crate::bvh::DynamicBvh

mod pairs;

pub use pairs::{generate_pairs, BroadPhasePair, PairChanges, PersistentBroadPhase};
