//! Resource limits that harden binary deserialization against hostile input
//! (design §24.8).
//!
//! Reflection-driven deserialization reads **untrusted** bytes — a save file,
//! a network packet, a mod — and rebuilds a typed tree guided by the target
//! schema. Two classic amplification attacks must be contained before any
//! allocation or recursion happens:
//!
//! - **Nesting bombs.** A tiny stream of deeply nested composite tags drives
//!   the recursive reader until the native stack overflows (an abort, not a
//!   recoverable error). [`DeserializeLimits::max_depth`] caps the composite
//!   nesting the reader will follow.
//! - **Length bombs ("billion laughs").** A collection node declares an
//!   enormous element count via a 2-byte varint; the naive reader then
//!   pre-allocates for billions of elements and is killed by the allocator
//!   long before the (short) stream is exhausted.
//!   [`DeserializeLimits::max_collection_len`] rejects any single collection
//!   whose declared length exceeds the budget, and the reader additionally
//!   clamps every speculative pre-allocation to the bytes that actually remain
//!   (each element costs at least one byte), so reservations stay proportional
//!   to the input rather than to an attacker-chosen number.
//!
//! The limits are deliberately generous so legitimate AAA payloads (large
//! vertex/index buffers, dense component arrays) pass untouched; they exist to
//! turn a crash or an out-of-memory abort into a clean, recoverable
//! [`DeserializeError`](crate::DeserializeError). Fully trusted, internally
//! produced streams can opt out with [`DeserializeLimits::UNLIMITED`].

/// Bounds applied while decoding a binary stream from a potentially hostile
/// source.
///
/// Construct with [`DeserializeLimits::new`], start from
/// [`DeserializeLimits::DEFAULT`], or disable the guards entirely with
/// [`DeserializeLimits::UNLIMITED`]. [`from_binary`](crate::from_binary) uses
/// [`DeserializeLimits::DEFAULT`]; [`from_binary_with_limits`] takes an explicit
/// policy.
///
/// [`from_binary_with_limits`]: crate::from_binary_with_limits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeserializeLimits {
    /// The deepest composite nesting the reader will follow before rejecting
    /// the stream with [`DeserializeError::DepthLimitExceeded`].
    ///
    /// Each struct, tuple struct, enum, list, array, map, and set that the
    /// reader descends into counts as one level. The root node is depth 1.
    ///
    /// [`DeserializeError::DepthLimitExceeded`]: crate::DeserializeError::DepthLimitExceeded
    pub max_depth: usize,
    /// The largest element count a single collection (list, array, map, set,
    /// or bulk POD blob) may declare before the stream is rejected with
    /// [`DeserializeError::CollectionTooLarge`].
    ///
    /// This is an upper bound on the *declared* length, checked before any
    /// allocation; the reader separately clamps pre-allocation to the bytes
    /// that remain in the stream.
    ///
    /// [`DeserializeError::CollectionTooLarge`]: crate::DeserializeError::CollectionTooLarge
    pub max_collection_len: usize,
}

impl DeserializeLimits {
    /// Generous defaults that pass legitimate payloads while still turning the
    /// classic nesting- and length-bomb attacks into recoverable errors.
    ///
    /// `max_depth` is 128 (far deeper than any realistic hand- or
    /// tool-authored reflect tree) and `max_collection_len` is 64 Mi elements.
    pub const DEFAULT: Self = Self {
        max_depth: 128,
        max_collection_len: 64 * 1024 * 1024,
    };

    /// Limits that impose no bound at all.
    ///
    /// Use only for streams produced by a fully trusted, in-process writer;
    /// never for save files, network input, or user-supplied content.
    pub const UNLIMITED: Self = Self {
        max_depth: usize::MAX,
        max_collection_len: usize::MAX,
    };

    /// Build an explicit policy from a maximum nesting depth and a maximum
    /// per-collection element count.
    #[must_use]
    pub const fn new(max_depth: usize, max_collection_len: usize) -> Self {
        Self {
            max_depth,
            max_collection_len,
        }
    }
}

impl Default for DeserializeLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}
