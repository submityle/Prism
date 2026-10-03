//! # Determinism tier (M4)
//!
//! Deterministic, insertion-ordered containers whose iteration order is a pure
//! function of the operation sequence — **bit-identical across runs and across
//! platforms**. This is the container floor of Prism's four-way determinism
//! contract (ECS ordering + task merge + fixed-point time + fixed-point
//! transform): identical inputs must produce an identical iteration order, or a
//! networked simulation desyncs.
//!
//! ## Determinism contract
//!
//! A container in this module guarantees that, for any fixed sequence of
//! operations starting from an empty container, the sequence of elements
//! yielded by iteration is identical on every run and every target. Concretely:
//!
//! 1. **Ordered iteration, seed-independent.** [`OrderedMap`] / [`OrderedSet`]
//!    iterate in *insertion order*, driven by a packed entry vector rather than
//!    by hash-table layout. The order therefore never depends on a random hash
//!    seed, on the hashing algorithm, or on allocation addresses.
//! 2. **Fixed-seed hashing.** The internal lookup index uses the crate's
//!    seed-free [`StableHasher`](crate::hash::StableHasher) via
//!    [`StableBuildHasher`](crate::hash::StableBuildHasher) — never the standard
//!    library's `RandomState`. Even though iteration order does not depend on
//!    the hasher, this keeps the entire structure free of run-to-run entropy
//!    (reproducible probe sequences, reproducible rehash behaviour).
//! 3. **Deterministic removal.** [`OrderedMap::remove`] / [`OrderedSet::remove`]
//!    preserve the relative order of the survivors (an order-stable shift).
//!    The `swap_remove` variants reorder but remain a deterministic function of
//!    the operation sequence.
//! 4. **Reconstruction stability.** Replaying the same insert/remove sequence
//!    on a fresh container always reproduces the same iteration order, so a
//!    container's state can be serialized and rebuilt bit-for-bit on another
//!    machine.
//!
//! ## Deterministic allocation
//!
//! The determinism contract also requires that *allocation order* be
//! address-independent. Prism's existing allocators already satisfy this, so no
//! new allocator ships here — they are reused:
//!
//! - [`FrameAllocator`](crate::alloc_::FrameAllocator) is a linear bump
//!   allocator: it hands out offsets sequentially from a fixed-capacity block
//!   and resets the whole frame in `O(1)`. For a fixed request sequence the
//!   layout (every returned offset) is reproducible and independent of the
//!   global heap's address-space layout.
//! - [`Arena`](crate::arena::Arena) and the handle containers
//!   ([`SlotMap`](crate::slot_map::SlotMap),
//!   [`SparseSet`](crate::sparse_set::SparseSet)) address elements by *index /
//!   generation*, never by pointer, and iterate in dense index order. Handles
//!   and iteration order are therefore reproducible across runs.
//!
//! Deterministic simulation code should allocate transient data from a
//! [`FrameAllocator`](crate::alloc_::FrameAllocator) (or an [`Arena`]) and key
//! long-lived data by handle rather than by address, then use the ordered
//! containers here for any iteration whose order feeds simulation results.
//!
//! ## Stable identity
//!
//! When cross-machine *identity* (not just order) is required, hash content
//! with [`stable_hash_bytes`](crate::hash::stable_hash_bytes) /
//! [`ContentHash`](crate::hash::ContentHash) from the M3 stable-hash tier rather
//! than relying on insertion order, so the same bytes map to the same id on
//! every machine.

pub mod ordered_map;
pub mod ordered_set;

pub use ordered_map::{Entry, OccupiedEntry, OrderedMap, VacantEntry};
pub use ordered_set::OrderedSet;

#[cfg(test)]
mod tests;
