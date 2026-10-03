//! `bevy_utils`-shaped aliases to ease migrating a Bevy codebase onto Prism
//! (the `compat-bevy` feature).
//!
//! # What this is
//! Bevy spreads a handful of container type names (`HashMap`, `HashSet`,
//! `StableHashMap`, `EntityHashMap`, `SmallVec`, …) across `bevy_utils` and
//! friends. Porting code to Prism is then mostly a matter of changing `use`
//! paths. This module re-exports the equivalent Prism types under those
//! familiar names so the diff stays small.
//!
//! # What this is *not*
//! Prism depends on **no** `bevy_*` crate and on **no** `glam`, and this module
//! adds neither. The math types Bevy re-exports from `glam` (`Vec3`, `Quat`,
//! `Mat4`, …) live in the separate `prism_math` crate, not here: this is the
//! *container* compatibility surface, which is what `prism_utils` actually owns.
//! The hashers also differ from Bevy's `aHash` default — Prism uses its own
//! fast [`FxBuildHasher`](crate::hash::FxBuildHasher) and a seed-free
//! [`StableBuildHasher`](crate::hash::StableBuildHasher) — so hashing behaviour
//! is compatible in shape (same API) but not bit-identical to Bevy.

extern crate alloc;

use crate::hash::StableBuildHasher;

/// A fast, non-cryptographic hash map, matching the `bevy_utils::HashMap`
/// alias role (Prism backs it with [`FxBuildHasher`](crate::hash::FxBuildHasher)
/// rather than Bevy's `aHash`).
pub type HashMap<K, V> = crate::hash::HashMap<K, V>;

/// A fast, non-cryptographic hash set, matching `bevy_utils::HashSet`.
pub type HashSet<T> = crate::hash::HashSet<T>;

/// A hash map whose iteration/hashing is stable across runs, matching the
/// `bevy_utils::StableHashMap` role. Backed by the seed-free
/// [`StableBuildHasher`](crate::hash::StableBuildHasher).
pub type StableHashMap<K, V> = std::collections::HashMap<K, V, StableBuildHasher>;

/// A run-stable hash set, matching `bevy_utils::StableHashSet`.
pub type StableHashSet<T> = std::collections::HashSet<T, StableBuildHasher>;

/// An entity-keyed hash map, matching the `bevy::EntityHashMap` role. Prism
/// uses the same fast hasher; a consumer crate narrows the key to its entity
/// id type.
pub type EntityHashMap<K, V> = crate::hash::HashMap<K, V>;

/// A deterministic, insertion-ordered map, matching the role Bevy fills with
/// `indexmap`-backed maps for reproducible iteration.
pub type OrderedMap<K, V> = crate::determinism::OrderedMap<K, V>;

/// A deterministic, insertion-ordered set.
pub type OrderedSet<T> = crate::determinism::OrderedSet<T>;

pub use crate::small_vec::SmallVec;

/// Bevy-flavoured prelude: `use prism_utils::compat_bevy::prelude::*;` to pull
/// the familiar container names into scope during a migration.
pub mod prelude {
    pub use super::{
        EntityHashMap, HashMap, HashSet, OrderedMap, OrderedSet, SmallVec, StableHashMap,
        StableHashSet,
    };
}
