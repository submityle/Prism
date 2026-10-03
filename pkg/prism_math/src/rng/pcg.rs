//! `PCG32` — O'Neill's permuted-congruential generator (XSH-RR, 64→32).
//!
//! A 64-bit LCG state passed through an xorshift-high / random-rotate output
//! permutation to yield 32-bit outputs with excellent statistical quality at a
//! very small state. Streams are selectable via the increment ("sequence").

use crate::rng::Rng;

const MULT: u64 = 6_364_136_223_846_793_005;
/// The `PCG32` (XSH-RR) generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Create a generator from a seed, using the default stream.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, 0)
    }

    /// Create a generator from a seed and a stream selector. Distinct streams
    /// never produce the same sequence, which is handy for independent
    /// per-entity or per-thread generators.
    #[inline]
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        // `inc` must be odd (PCG uses it as `2*stream + 1`).
        let inc = (stream << 1) | 1;
        let mut rng = Self { state: 0, inc };
        // Standard PCG seeding ritual.
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    /// Produce the next `u32` and advance the state.
    #[inline]
    #[expect(
        clippy::should_implement_trait,
        reason = "a PRNG step is infallible and endless, not an `Iterator::next` yielding `Option`"
    )]
    pub fn next(&mut self) -> u32 {
        let old = self.state;
        self.step();
        // XSH-RR output permutation.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

impl Rng for Pcg32 {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // Combine two 32-bit draws (high word first for a stable layout).
        let hi = u64::from(self.next());
        let lo = u64::from(self.next());
        (hi << 32) | lo
    }
}
