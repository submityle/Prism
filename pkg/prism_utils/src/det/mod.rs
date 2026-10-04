//! # Deterministic merge + reproducible hashing (`det`) — design doc §24.7
//!
//! The "determinism advanced" tier. It layers two platform-independent,
//! bit-reproducible primitives on top of the ordered-container floor in
//! [`determinism`](crate::determinism):
//!
//! - **Reproducible hashing** (module [`hash`]): a non-cryptographic 64-bit
//!   mixing hash in the `SplitMix`/`xxhash` family whose output is a pure function
//!   of its `u64` input stream and therefore identical across runs, builds, and
//!   target architectures. Two combiners sit on top of it — an **ordered**
//!   combiner (output depends on element order) and an **unordered** combiner
//!   (a commutative/associative fold whose output is independent of the order
//!   in which elements are contributed). The unordered combiner is what lets a
//!   parallel reduction fold partial results in any arrival order and still get
//!   a reproducible digest.
//! - **Deterministic merge** (module [`merge`]): [`DeterministicMerge`], an
//!   order-independent accumulator that collects `(key, value)` contributions
//!   and canonicalises them by sorting, so that merging the *same multiset* of
//!   contributions always yields the *same* sequence — no matter the order in
//!   which workers (threads, jobs, frames) produced them. Its
//!   [`reduce`](merge::DeterministicMerge::reduce) form folds all values that
//!   share a key with a user combine function, the deterministic analogue of a
//!   map-reduce "combine by key".
//!
//! The two compose: a parallel job system can produce partial results in a
//! nondeterministic order, feed them through a [`DeterministicMerge`] to obtain
//! a canonical ordering, and hash that ordering with the ordered combiner (or
//! hash the raw multiset directly with the unordered combiner) to get a digest
//! that is bit-identical on every machine. That is the §24.7 contract: a
//! networked or record/replay simulation that reduces concurrently must still
//! agree bit-for-bit.
//!
//! Everything here is **pure safe code** (`#![forbid(unsafe_code)]`) and uses
//! only `core`/`alloc`, except the optional
//! [`ConcurrentMerge`](merge::ConcurrentMerge) which is gated behind the
//! `concurrent` feature because it uses `std` synchronization.
//!
//! ## Honest boundary (design doc §24.8)
//! - The reproducible hash is **not cryptographic**: it is built for stable,
//!   fast content/identity digests, not for security against an adversary who
//!   can choose inputs. Use a real `MAC`/hash for trust boundaries.
//! - [`reduce`](merge::DeterministicMerge::reduce) is order-independent **only
//!   when the user's combine function is commutative and associative** (e.g.
//!   integer add, min/max, set union). A non-associative combine makes the
//!   result depend on grouping and is outside the contract.
//! - The combiners mix a stream of `u64` *lanes*. Hashing a higher-level value
//!   deterministically means first lowering it to a lane sequence with a stable
//!   byte/field order; for raw bytes or strings the M3
//!   [`stable_hash_bytes`](crate::hash::stable_hash_bytes) (FNV-1a) remains the
//!   entry point. This tier adds the *order-aware* and *order-independent*
//!   combination layer on top.

#![forbid(unsafe_code)]


pub mod hash;
pub mod merge;

pub use hash::{
    mix64, reproducible_hash_ordered, reproducible_hash_unordered, OrderedHashCombiner,
    UnorderedHashCombiner,
};
pub use merge::DeterministicMerge;

#[cfg(feature = "concurrent")]
pub use merge::ConcurrentMerge;

#[cfg(test)]
mod tests;
