//! A growable bit set (`BitSet`) over `u64` words.
//!
//! Bits are addressed by `usize` index and packed 64 to a word. The set grows
//! on demand to cover the highest bit touched by [`BitSet::set`] /
//! [`BitSet::toggle`], and supports the usual boolean algebra (union,
//! intersection, difference) both in place and producing a new set. Iteration
//! over set bits uses `trailing_zeros` to skip cleared bits quickly, which is
//! the hot path for ECS archetype matching, query filtering, and dirty-flag
//! sets. The implementation is fully safe.

extern crate alloc;

use alloc::vec::Vec;

/// Number of bits per backing word.
const BITS: usize = u64::BITS as usize;

/// A growable set of `usize` bit indices backed by packed `u64` words.
#[derive(Clone, Debug, Default)]
pub struct BitSet {
    words: Vec<u64>,
}

impl PartialEq for BitSet {
    /// Logical set equality: trailing all-zero words are ignored so that two
    /// sets with the same bits compare equal regardless of backing length.
    fn eq(&self, other: &Self) -> bool {
        let max = self.words.len().max(other.words.len());
        (0..max).all(|i| {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            a == b
        })
    }
}

impl Eq for BitSet {}

impl BitSet {
    /// Create an empty bit set.
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Create an empty bit set able to hold at least `bits` bits without
    /// reallocating.
    pub fn with_capacity(bits: usize) -> Self {
        Self {
            words: Vec::with_capacity(bits.div_ceil(BITS)),
        }
    }

    /// Decompose a bit index into `(word_index, bit_within_word)`.
    const fn locate(bit: usize) -> (usize, u32) {
        (bit / BITS, (bit % BITS) as u32)
    }

    /// Grow the backing storage so that `word` is a valid index.
    fn ensure_word(&mut self, word: usize) {
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
    }

    /// Insert `bit`, growing the set if needed. Returns `true` if the bit was
    /// previously clear.
    pub fn set(&mut self, bit: usize) -> bool {
        let (word, offset) = Self::locate(bit);
        self.ensure_word(word);
        let mask = 1u64 << offset;
        let was_set = self.words[word] & mask != 0;
        self.words[word] |= mask;
        !was_set
    }

    /// Remove `bit`. Returns `true` if the bit was previously set.
    pub fn clear(&mut self, bit: usize) -> bool {
        let (word, offset) = Self::locate(bit);
        if word >= self.words.len() {
            return false;
        }
        let mask = 1u64 << offset;
        let was_set = self.words[word] & mask != 0;
        self.words[word] &= !mask;
        was_set
    }

    /// Flip `bit`, growing the set if needed. Returns the new state (`true` if
    /// the bit is now set).
    pub fn toggle(&mut self, bit: usize) -> bool {
        let (word, offset) = Self::locate(bit);
        self.ensure_word(word);
        let mask = 1u64 << offset;
        self.words[word] ^= mask;
        self.words[word] & mask != 0
    }

    /// Whether `bit` is set.
    pub fn contains(&self, bit: usize) -> bool {
        let (word, offset) = Self::locate(bit);
        match self.words.get(word) {
            Some(w) => w & (1u64 << offset) != 0,
            None => false,
        }
    }

    /// Number of set bits (the set's cardinality).
    pub fn len(&self) -> usize {
        self.count_ones()
    }

    /// Whether no bits are set.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// Count of set bits. Equivalent to [`BitSet::len`]; named for the usual
    /// bit-twiddling terminology.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clear every bit while retaining allocated capacity.
    pub fn clear_all(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// In-place union: set every bit that is set in `other`.
    pub fn union_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (dst, src) in self.words.iter_mut().zip(other.words.iter()) {
            *dst |= *src;
        }
    }

    /// In-place intersection: keep only bits also set in `other`.
    pub fn intersect_with(&mut self, other: &BitSet) {
        for (i, dst) in self.words.iter_mut().enumerate() {
            match other.words.get(i) {
                Some(src) => *dst &= *src,
                None => *dst = 0,
            }
        }
    }

    /// In-place difference: clear every bit that is set in `other`.
    pub fn difference_with(&mut self, other: &BitSet) {
        for (dst, src) in self.words.iter_mut().zip(other.words.iter()) {
            *dst &= !*src;
        }
    }

    /// Produce the union of `self` and `other` as a new set.
    pub fn union(&self, other: &BitSet) -> BitSet {
        let mut out = self.clone();
        out.union_with(other);
        out
    }

    /// Produce the intersection of `self` and `other` as a new set.
    pub fn intersection(&self, other: &BitSet) -> BitSet {
        let mut out = self.clone();
        out.intersect_with(other);
        out
    }

    /// Produce the difference `self \ other` as a new set.
    pub fn difference(&self, other: &BitSet) -> BitSet {
        let mut out = self.clone();
        out.difference_with(other);
        out
    }

    /// Iterate over the indices of the set bits in ascending order.
    pub fn iter(&self) -> Ones<'_> {
        Ones {
            words: &self.words,
            word: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }
}

/// Iterator over the set-bit indices of a [`BitSet`], yielded in ascending
/// order. Produced by [`BitSet::iter`].
pub struct Ones<'a> {
    words: &'a [u64],
    word: usize,
    current: u64,
}

impl Iterator for Ones<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current != 0 {
                let offset = self.current.trailing_zeros() as usize;
                // Clear the lowest set bit so the next call advances.
                self.current &= self.current - 1;
                return Some(self.word * BITS + offset);
            }
            self.word += 1;
            if self.word >= self.words.len() {
                return None;
            }
            self.current = self.words[self.word];
        }
    }
}
