//! # prism_utils
//!
//! Prism's foundational container and utility kernel. It sits at the root of
//! the dependency graph alongside `prism_math` and is depended on by the ECS,
//! asset, diagnostic, reflect, and tasks kernels.
//!
//! ## M0 scope (this build)
//! - [`ArrayVec`](array_vec::ArrayVec): fixed-capacity, stack-allocated vector.
//! - [`SmallVec`](small_vec::SmallVec): inline storage that spills to the heap.
//! - [`Arena`](arena::Arena): a typed, index-based arena.
//! - [`HashMap`](hash::HashMap)/[`HashSet`](hash::HashSet): a SwissTable-style
//!   facade over a fast [`FxHasher`](hash::FxHasher).
//! - A [`prelude`] re-exporting the above.
//!
//! Later milestones add `SlotMap`/`SparseSet`/`BitSet`, interning, concurrent
//! containers, determinism, and SIMD hashing.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod array_vec;
pub mod arena;
pub mod hash;
pub mod prelude;
pub mod small_vec;

pub use array_vec::ArrayVec;
pub use arena::{Arena, ArenaIndex};
pub use hash::{FxBuildHasher, FxHasher, HashMap, HashSet};
pub use small_vec::SmallVec;

#[cfg(test)]
mod tests;
