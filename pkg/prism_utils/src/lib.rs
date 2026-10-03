//! # `prism_utils`
//!
//! Prism's foundational container and utility kernel. It sits at the root of
//! the dependency graph alongside `prism_math` and is depended on by the ECS,
//! asset, diagnostic, reflect, and tasks kernels.
//!
//! ## M0 scope
//! - [`ArrayVec`](array_vec::ArrayVec): fixed-capacity, stack-allocated vector.
//! - [`SmallVec`](small_vec::SmallVec): inline storage that spills to the heap.
//! - [`Arena`](arena::Arena): a typed, index-based arena.
//! - [`HashMap`](hash::HashMap)/[`HashSet`](hash::HashSet): a SwissTable-style
//!   facade over a fast [`FxHasher`](hash::FxHasher).
//!
//! ## M1 scope (this build)
//! - [`SlotMap`](slot_map::SlotMap): generational handles with safe
//!   invalidation ([`SlotKey`](slot_map::SlotKey)).
//! - [`SparseSet`](sparse_set::SparseSet): `O(1)` sparse-to-dense storage for
//!   ECS components.
//! - [`BitSet`](bit_set::BitSet): a growable bit set with boolean algebra.
//! - A [`prelude`] re-exporting the above.
//!
//! Later milestones add interning, concurrent containers, determinism, and
//! SIMD hashing.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod array_vec;
pub mod arena;
pub mod bit_set;
pub mod hash;
pub mod prelude;
pub mod slot_map;
pub mod small_vec;
pub mod sparse_set;

pub use array_vec::ArrayVec;
pub use arena::{Arena, ArenaIndex};
pub use bit_set::BitSet;
pub use hash::{FxBuildHasher, FxHasher, HashMap, HashSet};
pub use slot_map::{SlotKey, SlotMap};
pub use small_vec::SmallVec;
pub use sparse_set::SparseSet;

#[cfg(test)]
mod tests;
