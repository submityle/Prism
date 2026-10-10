//! Pure, thread-free core for deterministic record/replay of a parallel
//! execution order (design §24.7).
//!
//! The only observable thing that makes an order-sensitive parallel reduction
//! non-deterministic is the *order in which task results are committed* into
//! the shared accumulator. Which worker finishes first, and in what order the
//! commits land, depends on OS scheduling and steal timing. This module gives a
//! serializable [`ExecutionOrder`] — a permutation of task indices — that
//! captures exactly that commit order, plus the serial reference fold
//! ([`fold_in_order`]) a replay must reproduce bit-for-bit.
//!
//! Everything here is a pure function of its inputs: no clock, no threads, no
//! `std`. The façade in the parent module binds it to a real
//! [`TaskPool`](crate::TaskPool).

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// Text-form magic header, pinning the serialization format version.
const TEXT_MAGIC: &str = "prism-order v1";
/// Binary magic bytes (`PROR`).
const BYTES_MAGIC: [u8; 4] = *b"PROR";
/// Binary format version.
const BYTES_VERSION: u8 = 1;
/// Fixed header size of the binary form: magic(4) + version(1) + seed(8) +
/// len(4).
const BYTES_HEADER_LEN: usize = 4 + 1 + 8 + 4;

/// A reproducible `splitmix64` stream, used to synthesize a per-task workload
/// from a seed so that *same seed → same inputs* independent of core count.
///
/// Use [`SeedStream::for_task`] for a stateless, index-addressed derivation
/// (ideal for parallel tasks that must each derive their own value without
/// sharing a mutable stream), or [`SeedStream::new`] + [`SeedStream::next_u64`]
/// for a sequential stream.
#[derive(Clone, Debug)]
pub struct SeedStream {
    /// The running `splitmix64` state.
    state: u64,
}

impl SeedStream {
    /// Start a sequential stream from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the stream and return the next 64-bit value (`splitmix64`).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Derive a reproducible value for `task` under `seed` without any mutable
    /// state. `for_task(seed, i)` is a pure function of `(seed, i)`, so every
    /// worker — and the serial oracle — computes the identical value for a
    /// given task index.
    #[must_use]
    pub fn for_task(seed: u64, task: u64) -> u64 {
        let mut z = seed
            .wrapping_add(task.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Errors from constructing, validating, or parsing an [`ExecutionOrder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayOrderError {
    /// The serialized input ended before a full record could be read.
    Truncated,
    /// The magic header or version did not match this format.
    BadHeader,
    /// A record entry (an index line / word) was malformed.
    BadEntry,
    /// The number of entries did not match the declared length.
    LengthMismatch {
        /// Entry count the header declared.
        expected: usize,
        /// Entry count actually present.
        found: usize,
    },
    /// A task index was `>= len`, so the order cannot index its workload.
    TaskOutOfRange {
        /// The offending task index.
        task: u32,
        /// The exclusive upper bound (the workload length).
        len: usize,
    },
    /// The indices were in range but not a permutation of `0..len` (a value was
    /// repeated or missing).
    NotAPermutation {
        /// The workload length the indices should have permuted.
        len: usize,
    },
}

impl fmt::Display for ReplayOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("replay order: input truncated"),
            Self::BadHeader => f.write_str("replay order: bad magic header or version"),
            Self::BadEntry => f.write_str("replay order: malformed entry"),
            Self::LengthMismatch { expected, found } => {
                write!(
                    f,
                    "replay order: expected {expected} entries, found {found}"
                )
            }
            Self::TaskOutOfRange { task, len } => {
                write!(f, "replay order: task {task} out of range for len {len}")
            }
            Self::NotAPermutation { len } => {
                write!(f, "replay order: indices are not a permutation of 0..{len}")
            }
        }
    }
}

impl core::error::Error for ReplayOrderError {}

/// A serializable, replayable permutation of task indices: the order in which a
/// parallel run committed its tasks' results.
///
/// Construct one by recording a real run (via the façade) or directly with
/// [`ExecutionOrder::new`] / [`ExecutionOrder::canonical`]. Replaying it folds a
/// workload in exactly this order, reproducing the recorded run's result
/// regardless of how many workers the replay happens to use. The record
/// round-trips through [`ExecutionOrder::to_text`]/[`ExecutionOrder::from_text`]
/// and [`ExecutionOrder::to_bytes`]/[`ExecutionOrder::from_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionOrder {
    /// The seed that synthesized the workload (provenance; the caller threads
    /// it into its per-task compute).
    seed: u64,
    /// The commit order: a permutation of `0..order.len()`.
    order: Vec<u32>,
}

impl ExecutionOrder {
    /// Build an order from a seed and an explicit commit sequence, validating
    /// that `order` is a permutation of `0..order.len()`.
    ///
    /// # Errors
    /// Returns [`ReplayOrderError::TaskOutOfRange`] or
    /// [`ReplayOrderError::NotAPermutation`] if the indices are not a clean
    /// permutation.
    pub fn new(seed: u64, order: Vec<u32>) -> Result<Self, ReplayOrderError> {
        validate_permutation(&order)?;
        Ok(Self { seed, order })
    }

    /// The identity order `0, 1, ..., len-1` under `seed`: the canonical,
    /// worker-count-independent commit order.
    ///
    /// # Panics
    /// Panics if `len` exceeds `u32::MAX` (the index width of a stored order).
    #[must_use]
    pub fn canonical(seed: u64, len: usize) -> Self {
        let max = u32::try_from(len).expect("workload length exceeds u32::MAX");
        Self {
            seed,
            order: (0..max).collect(),
        }
    }

    /// Build an order from parts already known to be a valid permutation,
    /// skipping validation. Used by the recorder, which produces the sequence
    /// by appending each task index exactly once.
    pub(crate) fn from_parts(seed: u64, order: Vec<u32>) -> Self {
        debug_assert!(
            validate_permutation(&order).is_ok(),
            "recorder produced a non-permutation order"
        );
        Self { seed, order }
    }

    /// The seed that synthesized the recorded workload.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The commit order: a permutation of `0..len`.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The number of tasks (workload length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the workload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Validate that the stored indices are a permutation of `0..len`.
    ///
    /// # Errors
    /// Returns [`ReplayOrderError::TaskOutOfRange`] or
    /// [`ReplayOrderError::NotAPermutation`] on a malformed order.
    pub fn validate(&self) -> Result<(), ReplayOrderError> {
        validate_permutation(&self.order)
    }

    /// Serialize to a compact, human-readable text form: a
    /// `prism-order v1 seed=<hex> len=<dec>` header followed by one decimal
    /// index per line.
    #[must_use]
    pub fn to_text(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        // Writing to a `String` is infallible.
        let _ = writeln!(
            out,
            "{TEXT_MAGIC} seed={:016x} len={}",
            self.seed,
            self.order.len()
        );
        for id in &self.order {
            let _ = writeln!(out, "{id}");
        }
        out
    }

    /// Parse the text form produced by [`ExecutionOrder::to_text`].
    ///
    /// # Errors
    /// Returns [`ReplayOrderError`] if the header, length, or any index entry
    /// is malformed, or the entries are not a permutation.
    pub fn from_text(text: &str) -> Result<Self, ReplayOrderError> {
        let mut lines = text.lines();
        let header = lines.next().ok_or(ReplayOrderError::Truncated)?;
        let rest = header
            .strip_prefix(TEXT_MAGIC)
            .and_then(|r| r.strip_prefix(' '))
            .ok_or(ReplayOrderError::BadHeader)?;

        let mut toks = rest.split_whitespace();
        let seed_tok = toks.next().ok_or(ReplayOrderError::BadHeader)?;
        let len_tok = toks.next().ok_or(ReplayOrderError::BadHeader)?;
        if toks.next().is_some() {
            return Err(ReplayOrderError::BadHeader);
        }
        let seed_hex = seed_tok
            .strip_prefix("seed=")
            .ok_or(ReplayOrderError::BadHeader)?;
        let seed = u64::from_str_radix(seed_hex, 16).map_err(|_| ReplayOrderError::BadHeader)?;
        let len_str = len_tok
            .strip_prefix("len=")
            .ok_or(ReplayOrderError::BadHeader)?;
        let len: usize = len_str.parse().map_err(|_| ReplayOrderError::BadHeader)?;

        let mut order = Vec::with_capacity(len);
        for line in lines {
            let id: u32 = line
                .trim()
                .parse()
                .map_err(|_| ReplayOrderError::BadEntry)?;
            order.push(id);
        }
        if order.len() != len {
            return Err(ReplayOrderError::LengthMismatch {
                expected: len,
                found: order.len(),
            });
        }
        validate_permutation(&order)?;
        Ok(Self { seed, order })
    }

    /// Serialize to the compact binary form: `PROR` + version + seed (LE u64) +
    /// len (LE u32) + `len` little-endian `u32` indices.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = u32::try_from(self.order.len()).expect("workload length exceeds u32::MAX");
        let mut out = Vec::with_capacity(BYTES_HEADER_LEN + self.order.len() * 4);
        out.extend_from_slice(&BYTES_MAGIC);
        out.push(BYTES_VERSION);
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        for id in &self.order {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out
    }

    /// Parse the binary form produced by [`ExecutionOrder::to_bytes`].
    ///
    /// # Errors
    /// Returns [`ReplayOrderError`] if the magic, version, length, or body is
    /// malformed, or the entries are not a permutation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayOrderError> {
        if bytes.len() < BYTES_HEADER_LEN {
            return Err(ReplayOrderError::Truncated);
        }
        if bytes[0..4] != BYTES_MAGIC || bytes[4] != BYTES_VERSION {
            return Err(ReplayOrderError::BadHeader);
        }
        let seed_arr: [u8; 8] = bytes[5..13].try_into().expect("checked 8-byte seed slice");
        let seed = u64::from_le_bytes(seed_arr);
        let len_arr: [u8; 4] = bytes[13..17].try_into().expect("checked 4-byte len slice");
        let len = u32::from_le_bytes(len_arr) as usize;

        let body = &bytes[BYTES_HEADER_LEN..];
        if body.len() != len * 4 {
            return Err(ReplayOrderError::LengthMismatch {
                expected: len * 4,
                found: body.len(),
            });
        }
        let mut order = Vec::with_capacity(len);
        for chunk in body.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().expect("chunks_exact yields 4-byte slices");
            order.push(u32::from_le_bytes(arr));
        }
        validate_permutation(&order)?;
        Ok(Self { seed, order })
    }
}

/// Validate that `order` is a permutation of `0..order.len()`.
fn validate_permutation(order: &[u32]) -> Result<(), ReplayOrderError> {
    let len = order.len();
    let mut seen = vec![false; len];
    for &id in order {
        let idx = id as usize;
        if idx >= len {
            return Err(ReplayOrderError::TaskOutOfRange { task: id, len });
        }
        if seen[idx] {
            return Err(ReplayOrderError::NotAPermutation { len });
        }
        seen[idx] = true;
    }
    Ok(())
}

/// The serial reference fold a replay must reproduce bit-for-bit.
///
/// Visits each task index in `order`, recomputes its value with `compute`
/// (which **must** be a pure function of the index), and folds it into the
/// accumulator with `combine`. This is the independent oracle: for an
/// order-sensitive `combine`, the result depends entirely on `order`, so a
/// replay that commits in the recorded order yields exactly this value.
pub fn fold_in_order<V, A, F, C>(order: &[u32], init: A, compute: F, combine: C) -> A
where
    F: Fn(usize) -> V,
    C: Fn(A, usize, V) -> A,
{
    let mut acc = init;
    for &id in order {
        let idx = id as usize;
        let value = compute(idx);
        acc = combine(acc, idx, value);
    }
    acc
}
