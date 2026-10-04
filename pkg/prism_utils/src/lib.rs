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
//! - [`Rcu`](concurrent::Rcu): a read-copy-update cell for read-mostly shared
//!   state (type registry / asset index / config snapshot). Readers never lock
//!   or spin; writers publish a fresh copy and reclaim the old version through
//!   the epoch reclaimer. This is the §24.2 `RCU` form.
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
//! - [`HotCold`](layout::HotCold): a hot/cold field-separated container that
//!   stores a type's hot fields and cold fields in two separate `SoA` halves
//!   so a hot batch pass never loads cold memory, plus the
//!   [`LayoutPlan`](layout::LayoutPlan) `SoA` auto-layout maths (per-column
//!   alignment/stride, hot/cold group description). See the [`layout`] module.
//! - [`Cow`](cow::Cow): an `Arc`-backed copy-on-write container that is cheap
//!   to clone and only deep-copies on first mutation of a shared value.
//! - [`BuddyAllocator`](alloc_::BuddyAllocator)/[`TlsfAllocator`](alloc_::TlsfAllocator):
//!   offset-based region sub-allocators for large-world / `GPU` heap carving
//!   (buddy coalescing; `O(1)` Two-Level Segregated Fit).
//! - [`compat_bevy`]: `bevy_utils`-shaped container aliases (the `compat-bevy`
//!   feature) so porting a Bevy codebase onto Prism is mostly a `use`-path
//!   change.
//!
//! ## M7 scope (this build): hardening + content-addressed dedup
//! - [`GuardedBuffer`](guard::GuardedBuffer)/[`GuardedPool`](guard::GuardedPool):
//!   pure-safe, debug-tier memory-safety hardening (design doc §24.3). The
//!   buffer flanks its payload with canary redzones and poisons it on free to
//!   turn overflows, underflows, use-after-free and double-free into reported
//!   [`GuardError`](guard::GuardError)s; the pool adds generational handles so a
//!   stale [`GuardHandle`](guard::GuardHandle) is rejected rather than silently
//!   aliasing a recycled slot. They are the zero-`unsafe` container complement
//!   to the raw-memory [`GuardedAllocator`](alloc_::GuardedAllocator).
//! - [`InternCache`](intern::InternCache)/[`Interned`](intern::Interned):
//!   content-addressed deduplication (design doc §24.6). Each distinct
//!   `Hash + Eq` value is stored once, keyed by the crate's non-cryptographic
//!   [`stable_hash`](hash::stable_hash) (FNV-1a), and addressed by a cheap,
//!   domain-tagged `Copy` handle, collapsing content equality to an `O(1)`
//!   handle compare for asset / mesh / texture dedup.
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
pub mod arena;
pub mod array_vec;
pub mod bit_set;
#[cfg(feature = "compat-bevy")]
pub mod compat_bevy;
#[cfg(feature = "concurrent")]
pub mod concurrent;
pub mod cow;
pub mod det;
pub mod determinism;
pub mod guard;
pub mod hash;
pub mod hbitset;
pub mod intern;
pub mod layout;
pub mod prelude;
pub mod reloc;
pub mod slot_map;
pub mod small_vec;
pub mod soa;
pub mod sparse_set;

pub use alloc_::{
    AllocBox, AllocError, Allocator, BuddyAllocator, FrameAllocator, Global, Pool, TlsfAllocator,
};
pub use arena::{Arena, ArenaIndex};
pub use array_vec::ArrayVec;
pub use bit_set::BitSet;
#[cfg(feature = "concurrent")]
pub use concurrent::{
    Collector, ConcurrentHashMap, Guard, LocalHandle, MpmcQueue, Rcu, RcuGuard, SpscConsumer,
    SpscProducer, SpscQueue, TreiberStack,
};
pub use cow::Cow;
pub use det::{
    mix64, reproducible_hash_ordered, reproducible_hash_unordered, DeterministicMerge,
    OrderedHashCombiner, UnorderedHashCombiner,
};
#[cfg(feature = "concurrent")]
pub use det::ConcurrentMerge;
pub use determinism::{OrderedMap, OrderedSet};
pub use guard::{GuardConfig, GuardError, GuardHandle, GuardedBuffer, GuardedPool};
pub use hash::{
    stable_hash, stable_hash_bytes, stable_hash_str, ContentHash, FxBuildHasher, FxHasher, HashMap,
    HashSet, StableBuildHasher, StableHasher,
};
pub use hbitset::HierarchicalBitSet;
pub use intern::{domain, FName, InternCache, Interned, Interner, Istr};
pub use layout::{
    align_up, ColumnPlan, ColumnShape, ColumnShapes, GroupLayout, HotCold, LayoutPlan,
    Temperature, CACHE_LINE,
};
pub use reloc::{
    OffsetPtr, OffsetSlice, Reloc, RelocError, RelocMap, RelocMapView, RelocVec, RelocVecView,
};
pub use slot_map::{SlotKey, SlotMap};
pub use small_vec::SmallVec;
pub use soa::{Soa, SoaVec};
pub use sparse_set::SparseSet;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_hotcold;
#[cfg(test)]
mod tests_m6;
