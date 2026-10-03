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
//! ## M1 scope
//! - [`SlotMap`](slot_map::SlotMap): generational handles with safe
//!   invalidation ([`SlotKey`](slot_map::SlotKey)).
//! - [`SparseSet`](sparse_set::SparseSet): `O(1)` sparse-to-dense storage for
//!   ECS components.
//! - [`BitSet`](bit_set::BitSet): a growable bit set with boolean algebra.
//!
//! ## M2 scope (this build): allocators
//! - [`Allocator`](alloc_::Allocator): a stable, Prism-native allocation trait
//!   (does **not** require the nightly `core::alloc::Allocator`).
//! - [`Global`](alloc_::Global): a bridge to the global heap via `alloc::alloc`.
//! - [`Pool`](alloc_::Pool): an `O(1)` fixed-size-block pool allocator with
//!   automatic chunk growth.
//! - [`FrameAllocator`](alloc_::FrameAllocator): a linear bump allocator with
//!   `O(1)` whole-frame [`reset`](alloc_::FrameAllocator::reset).
//! - [`AllocBox`](alloc_::AllocBox): an allocator-aware owning box.
//!
//! ## M3 scope (this build): interning + stable hashing
//! - [`Interner`](intern::Interner)/[`Istr`](intern::Istr): domain-separated
//!   string interning with `O(1)` handle comparison ([`FName`](intern::FName)
//!   is the default tag alias).
//! - [`StableHasher`](hash::StableHasher)/[`StableBuildHasher`](hash::StableBuildHasher):
//!   seed-free FNV-1a hashing that is identical across runs of the same build.
//! - [`ContentHash`](hash::ContentHash): a 128-bit content hash for
//!   content-addressed deduplication.
//!
//! ## M4 scope (this build): determinism tier (done)
//! - [`OrderedMap`](determinism::OrderedMap)/[`OrderedSet`](determinism::OrderedSet):
//!   deterministic, insertion-ordered containers whose iteration order is
//!   bit-identical across runs and platforms (seed-free, address-independent).
//!   They back the four-way determinism contract and reuse the M3
//!   [`StableHasher`](hash::StableHasher) for their lookup index. See the
//!   [`determinism`] module for the full determinism contract, including why
//!   the existing [`FrameAllocator`](alloc_::FrameAllocator)/[`Arena`](arena::Arena)
//!   already provide deterministic allocation.
//!
//! ## M5 scope (this build): concurrent containers (`concurrent` feature)
//! - [`SpscQueue`](concurrent::SpscQueue): bounded, lock-free single-producer
//!   single-consumer ring with cache-line-padded cursors.
//! - [`MpmcQueue`](concurrent::MpmcQueue): bounded, lock-free multi-producer
//!   multi-consumer queue (Vyukov per-slot sequence numbers).
//! - [`ConcurrentHashMap`](concurrent::ConcurrentHashMap): a sharded concurrent
//!   hash map with a per-shard reader-writer-locked read path.
//! - [`Collector`](concurrent::Collector)/[`Guard`](concurrent::Guard):
//!   epoch-based reclamation for safe memory reclamation of the lock-free
//!   structures (defeats use-after-free and `ABA`), demonstrated by the
//!   lock-free [`TreiberStack`](concurrent::TreiberStack).
//!
//! See the [`concurrent`] module for the full correctness posture. The whole
//! tier is gated behind the `concurrent` feature so a single-threaded build
//! pays nothing for it (the conservative default from the design doc §11/§23).
//!
//! ## M6 scope (this build): advanced layout + migration
//! - [`HierarchicalBitSet`](hbitset::HierarchicalBitSet): a layered bit set
//!   with summary levels so "find the next set bit" is close to `O(set bits)`
//!   even over a very large, sparse index domain (big-world entity/archetype
//!   masks).
//! - [`SoaVec`](soa::SoaVec): derive-free structure-of-arrays columnar storage
//!   (one contiguous column per tuple field) for cache-friendly, vectorisable
//!   batch passes.
//! - [`Cow`](cow::Cow): an `Arc`-backed copy-on-write container that is cheap
//!   to clone and only deep-copies on first mutation of a shared value.
//! - [`BuddyAllocator`](alloc_::BuddyAllocator)/[`TlsfAllocator`](alloc_::TlsfAllocator):
//!   offset-based region sub-allocators for large-world / `GPU` heap carving
//!   (buddy coalescing; `O(1)` Two-Level Segregated Fit).
//! - [`compat_bevy`]: `bevy_utils`-shaped container aliases (the `compat-bevy`
//!   feature) so porting a Bevy codebase onto Prism is mostly a `use`-path
//!   change.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.
//!
//! ## `unsafe` policy
//! The container milestones (M0/M1) are entirely safe. M2 introduces raw memory
//! management, which is inherently `unsafe`; the crate therefore does not set
//! `#![forbid(unsafe_code)]`. The workspace `unsafe_code = "deny"` lint still
//! applies and is overridden locally with `#[allow(unsafe_code, reason = …)]`
//! on each individual `unsafe` site, every one of which carries a `// SAFETY:`
//! justification.

pub mod alloc_;
pub mod array_vec;
pub mod arena;
pub mod bit_set;
#[cfg(feature = "compat-bevy")]
pub mod compat_bevy;
#[cfg(feature = "concurrent")]
pub mod concurrent;
pub mod cow;
pub mod determinism;
pub mod hash;
pub mod hbitset;
pub mod intern;
pub mod prelude;
pub mod slot_map;
pub mod small_vec;
pub mod soa;
pub mod sparse_set;

pub use alloc_::{
    AllocBox, AllocError, Allocator, BuddyAllocator, FrameAllocator, Global, Pool, TlsfAllocator,
};
pub use array_vec::ArrayVec;
pub use arena::{Arena, ArenaIndex};
pub use bit_set::BitSet;
pub use cow::Cow;
pub use hbitset::HierarchicalBitSet;
pub use soa::{Soa, SoaVec};
#[cfg(feature = "concurrent")]
pub use concurrent::{
    Collector, ConcurrentHashMap, Guard, LocalHandle, MpmcQueue, SpscConsumer, SpscProducer,
    SpscQueue, TreiberStack,
};
pub use determinism::{OrderedMap, OrderedSet};
pub use hash::{
    stable_hash, stable_hash_bytes, stable_hash_str, ContentHash, FxBuildHasher, FxHasher,
    HashMap, HashSet, StableBuildHasher, StableHasher,
};
pub use intern::{domain, FName, Interner, Istr};
pub use slot_map::{SlotKey, SlotMap};
pub use small_vec::SmallVec;
pub use sparse_set::SparseSet;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_m6;
