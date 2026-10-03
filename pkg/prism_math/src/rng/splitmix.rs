//! `SplitMix64` — a tiny, fast `u64` mixing generator.
//!
//! Primarily used to expand a single user seed into the larger state the
//! stronger generators ([`crate::rng::Xoshiro256StarStar`]) need, but it is a
//! perfectly good standalone PRNG as well. This is the canonical variant from
//! Vigna's reference code (constant `0x9E3779B97F4A7C15`).

use crate::rng::Rng;

/// The `SplitMix64` generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a 64-bit seed.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produce the next `u64` and advance the state (canonical `SplitMix64`).
    #[inline]
    #[expect(
        clippy::should_implement_trait,
        reason = "a PRNG step is infallible and endless, not an `Iterator::next` yielding `Option`"
    )]
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Rng for SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.next()
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Take the high 32 bits, which pass empirical tests better than the low.
        (self.next() >> 32) as u32
    }
}
