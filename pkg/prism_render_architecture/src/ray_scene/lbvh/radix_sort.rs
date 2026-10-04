//! Deterministic least-significant-digit radix sort over `(key, payload)` pairs.
//!
//! The linear-`BVH` builder must order primitives by their 30-bit `Morton` key
//! before it can build the radix tree. A counting/radix sort is the natural
//! choice because a `GPU` kernel implements the identical algorithm — per-digit
//! histogram, prefix sum, scatter — so the `CPU` reference here is bit-for-bit
//! reproducible on the device.
//!
//! The sort is **stable**: equal keys keep their input order. Stability matters
//! because the radix-tree builder breaks ties between equal `Morton` keys using
//! the sorted position, so a non-stable permutation would change topology.

use alloc::vec;
use alloc::vec::Vec;

/// Bits consumed per radix pass.
const RADIX_BITS: u32 = 8;
/// Number of buckets per pass (`2^RADIX_BITS`).
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;
/// Mask selecting one digit.
const RADIX_MASK: u32 = (RADIX_BUCKETS as u32) - 1;
/// Passes needed to cover the 30-bit `Morton` key (rounded up to whole digits).
const RADIX_PASSES: u32 = 4;

/// A sortable `(Morton key, primitive index)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MortonEntry {
    /// 30-bit `Morton` key; only the low 30 bits are significant.
    pub key: u32,
    /// Original primitive index this key was computed from.
    pub primitive: u32,
}

/// Stably sorts `entries` ascending by [`MortonEntry::key`].
///
/// Uses four 8-bit least-significant-digit passes with a double buffer, so the
/// result is deterministic and independent of the input permutation beyond the
/// stability guarantee.
pub fn radix_sort(entries: &mut Vec<MortonEntry>) {
    let len = entries.len();
    if len <= 1 {
        return;
    }
    let mut scratch = vec![
        MortonEntry {
            key: 0,
            primitive: 0,
        };
        len
    ];
    let mut src = entries;
    let mut dst = &mut scratch;
    let mut shift = 0;
    let mut pass = 0;
    while pass < RADIX_PASSES {
        let mut histogram = [0_u32; RADIX_BUCKETS];
        for entry in src.iter() {
            let digit = ((entry.key >> shift) & RADIX_MASK) as usize;
            histogram[digit] += 1;
        }
        // Exclusive prefix sum turns counts into starting offsets.
        let mut offset = 0_u32;
        for slot in &mut histogram {
            let count = *slot;
            *slot = offset;
            offset += count;
        }
        for entry in src.iter() {
            let digit = ((entry.key >> shift) & RADIX_MASK) as usize;
            let position = histogram[digit] as usize;
            histogram[digit] += 1;
            dst[position] = *entry;
        }
        core::mem::swap(&mut src, &mut dst);
        shift += RADIX_BITS;
        pass += 1;
    }
    // Four passes is even, so the fully sorted data already lives in `entries`
    // (the original `src`); no final copy is required.
    debug_assert!(src.windows(2).all(|w| w[0].key <= w[1].key));
}
