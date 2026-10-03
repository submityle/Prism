//! A hierarchical (layered) bit set: multi-level summary bits over a dense
//! leaf bit set that make "find the next set bit" close to `O(set bits)` even
//! over a very sparse, very large index domain.
//!
//! # Why layers
//! A flat [`BitSet`](crate::bit_set::BitSet) must scan every intervening word
//! to find the next set bit, which is `O(domain / 64)` in the worst case. For
//! the large, sparse entity / archetype masks of a big world that is far too
//! slow. A hierarchical bit set stacks *summary* levels on top of the leaf
//! bits: bit `w` of a summary word is set iff leaf (or lower-summary) word `w`
//! is non-empty. Finding the next set bit then descends one summary word per
//! level (each a single `trailing_zeros`), skipping whole empty 64-, 4096-,
//! 262144-bit regions at a time.
//!
//! The structure has a fixed capacity chosen at construction; every operation
//! is safe (no `unsafe`), index-based, and allocation-free after construction.
//!
//! # Layout
//! `levels[0]` is the dense leaf layer (one bit per index). `levels[l]` for
//! `l > 0` is a summary layer with one bit per word of `levels[l - 1]`. The
//! top layer always holds exactly one `u64` word, so emptiness is a single
//! word compare.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Number of bits per backing word.
const BITS: usize = u64::BITS as usize;

/// A fixed-capacity hierarchical bit set.
///
/// Construct it with [`HierarchicalBitSet::with_capacity`], then
/// [`insert`](Self::insert) / [`remove`](Self::remove) indices and query the
/// next set bit with [`find_first`](Self::find_first) /
/// [`find_next`](Self::find_next) or walk them all with [`iter`](Self::iter).
#[derive(Clone, Debug)]
pub struct HierarchicalBitSet {
    /// `levels[0]` is the leaf layer; higher layers summarise the layer below.
    levels: Vec<Vec<u64>>,
    /// Addressable capacity in bits (a multiple of 64).
    capacity: usize,
}

impl HierarchicalBitSet {
    /// Create a hierarchical bit set able to address `capacity` indices
    /// (`0..capacity`). The real capacity is rounded up to a multiple of 64.
    ///
    /// The number of summary layers is `ceil(log64(leaf_words))`, so a domain
    /// of a few million bits needs only three or four layers.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let leaf_words = capacity.div_ceil(BITS).max(1);
        let mut levels = Vec::new();
        let mut words = leaf_words;
        loop {
            levels.push(vec![0u64; words]);
            if words <= 1 {
                break;
            }
            words = words.div_ceil(BITS);
        }
        Self {
            levels,
            capacity: leaf_words * BITS,
        }
    }

    /// The addressable capacity in bits (always a multiple of 64).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Decompose a bit index into `(word_index, bit_within_word)`.
    #[inline]
    const fn locate(bit: usize) -> (usize, u32) {
        (bit / BITS, (bit % BITS) as u32)
    }

    /// Insert `index`. Returns `true` if the index was previously clear.
    ///
    /// # Panics
    /// Panics if `index >= capacity()`.
    pub fn insert(&mut self, index: usize) -> bool {
        assert!(
            index < self.capacity,
            "index {index} out of range for hierarchical bit set capacity {}",
            self.capacity
        );
        let (mut word, offset) = Self::locate(index);
        let mask = 1u64 << offset;
        let was_set = self.levels[0][word] & mask != 0;
        self.levels[0][word] |= mask;
        // Any touched word is now non-empty, so set its summary bit all the way
        // to the root. Re-setting an already-set summary bit is idempotent.
        for level in 1..self.levels.len() {
            let (parent, bit) = Self::locate(word);
            self.levels[level][parent] |= 1u64 << bit;
            word = parent;
        }
        !was_set
    }

    /// Remove `index`. Returns `true` if the index was previously set.
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.capacity {
            return false;
        }
        let (word, offset) = Self::locate(index);
        let mask = 1u64 << offset;
        if self.levels[0][word] & mask == 0 {
            return false;
        }
        self.levels[0][word] &= !mask;
        // Clear a summary bit only once the word it summarises becomes empty,
        // and stop as soon as a word is still non-empty.
        let mut word = word;
        let mut level = 0;
        while level + 1 < self.levels.len() {
            if self.levels[level][word] != 0 {
                break;
            }
            let (parent, bit) = Self::locate(word);
            self.levels[level + 1][parent] &= !(1u64 << bit);
            word = parent;
            level += 1;
        }
        true
    }

    /// Returns `true` if `index` is set.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        if index >= self.capacity {
            return false;
        }
        let (word, offset) = Self::locate(index);
        self.levels[0][word] & (1u64 << offset) != 0
    }

    /// Returns `true` if no bit is set.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        // The top layer is a single word summarising the entire set.
        self.levels[self.levels.len() - 1][0] == 0
    }

    /// The number of set bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.levels[0].iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clear every bit.
    pub fn clear(&mut self) {
        for level in &mut self.levels {
            for word in level.iter_mut() {
                *word = 0;
            }
        }
    }

    /// The smallest set index, or `None` if the set is empty.
    #[must_use]
    #[inline]
    pub fn find_first(&self) -> Option<usize> {
        self.find_next(0)
    }

    /// The smallest set index `>= from`, or `None` if there is none.
    ///
    /// This is the hierarchical fast path: it checks `from`'s own leaf word,
    /// then climbs the summary layers to skip empty regions and descends into
    /// the first non-empty sibling, taking one `trailing_zeros` per layer.
    #[must_use]
    pub fn find_next(&self, from: usize) -> Option<usize> {
        if from >= self.capacity {
            return None;
        }
        let (word, offset) = Self::locate(from);
        // Fast path: a later bit in `from`'s own leaf word.
        let masked = self.levels[0][word] & (u64::MAX << offset);
        if masked != 0 {
            return Some(word * BITS + masked.trailing_zeros() as usize);
        }

        // Climb: at each layer look for a sibling word after the current one.
        let mut level = 0;
        let mut cur = word;
        loop {
            level += 1;
            if level >= self.levels.len() {
                return None;
            }
            let (parent, bit) = Self::locate(cur);
            // Siblings strictly after `bit` (the subtree at `bit` is exhausted).
            let higher = if bit == 63 {
                0
            } else {
                self.levels[level][parent] & (u64::MAX << (bit + 1))
            };
            if higher != 0 {
                let next_bit = higher.trailing_zeros() as usize;
                return Some(self.descend(level, parent * BITS + next_bit));
            }
            cur = parent;
        }
    }

    /// Descend from `level` (where `word` is a word index into `level - 1`)
    /// down to the leaf, always taking the lowest set bit, and return the
    /// resulting leaf bit index.
    fn descend(&self, level: usize, mut word: usize) -> usize {
        let mut l = level - 1;
        while l > 0 {
            let bit = self.levels[l][word].trailing_zeros() as usize;
            word = word * BITS + bit;
            l -= 1;
        }
        let bit = self.levels[0][word].trailing_zeros() as usize;
        word * BITS + bit
    }

    /// Iterate the set indices in ascending order.
    #[must_use]
    #[inline]
    pub fn iter(&self) -> HierarchicalOnes<'_> {
        HierarchicalOnes {
            set: self,
            next_from: 0,
        }
    }
}

/// Iterator over the set-bit indices of a [`HierarchicalBitSet`] in ascending
/// order. Produced by [`HierarchicalBitSet::iter`].
pub struct HierarchicalOnes<'a> {
    set: &'a HierarchicalBitSet,
    next_from: usize,
}

impl Iterator for HierarchicalOnes<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let found = self.set.find_next(self.next_from)?;
        self.next_from = found + 1;
        Some(found)
    }
}
