//! Platform-independent, reproducible non-cryptographic hashing (§24.7).
//!
//! The core is [`mix64`], the `SplitMix64` finaliser: a bijective avalanche
//! function built from two odd multiplicative constants and three xor-shifts.
//! Because it is pure 64-bit wrapping integer arithmetic, its output is a
//! function of its input alone — identical on every run, every build, and every
//! target architecture (no floating point, no pointer width, no endianness
//! dependence on the `u64` lane values themselves).
//!
//! On top of [`mix64`] sit two stream combiners:
//! - [`OrderedHashCombiner`]: the digest depends on the *order* of the lanes,
//!   so `[1, 2, 3]` and `[3, 2, 1]` hash differently. Use it to fingerprint a
//!   sequence whose order is meaningful (an event log, a canonicalised merge
//!   result).
//! - [`UnorderedHashCombiner`]: a commutative/associative fold, so any
//!   permutation of the same multiset hashes identically. Use it to fingerprint
//!   a set/bag of contributions produced in a nondeterministic (parallel)
//!   order.
//!
//! Both fold a count into the final digest so that, e.g., an all-zero lane
//! stream of different lengths still produces different digests.

/// First `SplitMix64` multiplicative constant.
const C1: u64 = 0xbf58_476d_1ce4_e5b9;
/// Second `SplitMix64` multiplicative constant.
const C2: u64 = 0x94d0_49bb_1331_11eb;
/// The golden-ratio odd seed (same constant `SplitMix64` uses as its increment),
/// reused here as the combiners' initial state and length-mixing salt.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` finaliser: a bijective 64-bit avalanche mix.
///
/// This is the primitive every combiner in this module is built from. It is
/// pure wrapping integer arithmetic, so it is bit-identical across runs,
/// builds, and architectures. Note that `mix64(0) == 0` (the fixed point of the
/// finaliser); the combiners never feed a bare `0` through it alone, folding a
/// non-zero salt in first.
#[inline]
#[must_use]
pub const fn mix64(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(C1);
    z = (z ^ (z >> 27)).wrapping_mul(C2);
    z ^ (z >> 31)
}

/// An **order-sensitive** reproducible hash combiner.
///
/// Feeding the same lanes in the same order always yields the same
/// [`finish`](OrderedHashCombiner::finish) digest; a different order yields a
/// different digest (with overwhelming probability). The running state is
/// rotated before each absorb so that position matters, and the element count
/// is folded into the finaliser so differing lengths never collide trivially.
#[derive(Clone, Copy, Debug)]
pub struct OrderedHashCombiner {
    state: u64,
    count: u64,
}

impl OrderedHashCombiner {
    /// Create an empty combiner seeded with the canonical constant.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SEED,
            count: 0,
        }
    }

    /// Absorb one `u64` lane, updating the running state in an order-dependent
    /// way.
    #[inline]
    pub fn write(&mut self, lane: u64) {
        self.state = mix64(self.state.rotate_left(27) ^ mix64(lane));
        self.count = self.count.wrapping_add(1);
    }

    /// Absorb every lane of an iterator, in order.
    #[inline]
    pub fn write_all<I: IntoIterator<Item = u64>>(&mut self, lanes: I) {
        for lane in lanes {
            self.write(lane);
        }
    }

    /// Finalise and return the 64-bit digest. Does not consume the combiner, so
    /// it can be inspected mid-stream.
    #[inline]
    #[must_use]
    pub const fn finish(&self) -> u64 {
        mix64(self.state ^ self.count.wrapping_mul(SEED))
    }

    /// The number of lanes absorbed so far.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }
}

impl Default for OrderedHashCombiner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// An **order-independent** (commutative/associative) reproducible hash
/// combiner.
///
/// The per-lane contributions are summed with wrapping addition, which is both
/// commutative and associative, so any permutation of the same multiset of
/// lanes produces the same accumulator — and therefore the same
/// [`finish`](UnorderedHashCombiner::finish) digest. This is the combiner a
/// parallel reduction uses: each worker folds its own lanes, and the partial
/// accumulators can themselves be [`merge`](UnorderedHashCombiner::merge)d in
/// any order (the operation is associative) for the same final result.
#[derive(Clone, Copy, Debug)]
pub struct UnorderedHashCombiner {
    acc: u64,
    count: u64,
}

impl UnorderedHashCombiner {
    /// Create an empty combiner.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { acc: 0, count: 0 }
    }

    /// Absorb one `u64` lane. The order of calls does not affect the result.
    #[inline]
    pub fn write(&mut self, lane: u64) {
        self.acc = self.acc.wrapping_add(mix64(lane));
        self.count = self.count.wrapping_add(1);
    }

    /// Absorb every lane of an iterator; order is irrelevant.
    #[inline]
    pub fn write_all<I: IntoIterator<Item = u64>>(&mut self, lanes: I) {
        for lane in lanes {
            self.write(lane);
        }
    }

    /// Fold another combiner's partial state into this one. This is the
    /// associative "combine two partial reductions" operation.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.acc = self.acc.wrapping_add(other.acc);
        self.count = self.count.wrapping_add(other.count);
    }

    /// Finalise and return the 64-bit digest. The count is mixed in so that
    /// multisets differing only by added zero-contributing lanes still differ.
    #[inline]
    #[must_use]
    pub const fn finish(&self) -> u64 {
        mix64(self.acc ^ mix64(self.count ^ SEED))
    }

    /// The number of lanes absorbed so far.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }
}

impl Default for UnorderedHashCombiner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Hash a lane sequence with the **order-sensitive** combiner.
///
/// Convenience wrapper around [`OrderedHashCombiner`]. Equivalent sequences in
/// a different order generally hash differently.
#[inline]
#[must_use]
pub fn reproducible_hash_ordered<I: IntoIterator<Item = u64>>(lanes: I) -> u64 {
    let mut h = OrderedHashCombiner::new();
    h.write_all(lanes);
    h.finish()
}

/// Hash a lane multiset with the **order-independent** combiner.
///
/// Convenience wrapper around [`UnorderedHashCombiner`]. Any permutation of the
/// same lanes hashes identically.
#[inline]
#[must_use]
pub fn reproducible_hash_unordered<I: IntoIterator<Item = u64>>(lanes: I) -> u64 {
    let mut h = UnorderedHashCombiner::new();
    h.write_all(lanes);
    h.finish()
}
