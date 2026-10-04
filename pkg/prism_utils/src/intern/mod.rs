//! # Interning & content-addressed deduplication (`intern`)
//!
//! Two complementary deduplication facilities that both return a cheap, stable,
//! `Copy` handle so that equality collapses to an `O(1)` integer compare:
//!
//! - [`Interner`] / [`Istr`] (module [`string`]): domain-separated **string**
//!   interning. Each distinct string is stored once; comparing two interned
//!   strings is an integer compare instead of a `strcmp` (the trick behind
//!   engine "names" such as Unreal's `FName`).
//! - [`InternCache`] / [`Interned`] (module [`cache`]): design doc §24.6
//!   **content-addressed** interning for arbitrary `Hash + Eq` content (mesh
//!   blocks, textures, type identities, asset IDs). Each distinct value is
//!   stored once, keyed by a classic non-cryptographic content hash
//!   ([`StableHasher`](crate::hash::StableHasher), FNV-1a — not cryptographic,
//!   not learned), and returns a stable [`Interned`] handle.
//!
//! ## Domain separation
//! Both facilities are generic over an uninhabited [`domain`] marker. Handles
//! from different domains are distinct types, so the compiler rejects mixing a
//! mesh handle with a type handle even though both wrap a `u32`. Each domain is
//! a separate table, which is also how lifetime is bounded: reclaim a whole
//! domain by dropping or [`clear`](InternCache::clear)-ing that one table
//! (design doc §23 risk: no global, unbounded intern table).

pub mod cache;
pub mod domain;
pub mod string;

pub use cache::{InternCache, Interned};
pub use string::{FName, Interner, Istr};

#[cfg(test)]
mod cache_tests;
