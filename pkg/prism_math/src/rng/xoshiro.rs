//! `xoshiro256**` — Blackman & Vigna's all-purpose 64-bit generator.
//!
//! Large period (`2^256 - 1`), excellent statistical quality, and fast. The
//! state is seeded by expanding a single `u64` through [`SplitMix64`] (the
//! author-recommended procedure), and [`Xoshiro256StarStar::jump`] advances the
//! stream by `2^128` steps to carve out non-overlapping sub-sequences.

use crate::rng::Rng;
use crate::rng::splitmix::SplitMix64;

/// The `xoshiro256**` generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

impl Xoshiro256StarStar {
    /// Create a generator from a 64-bit seed (state expanded via `SplitMix64`).
    #[inline]
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Self { s: [sm.next(), sm.next(), sm.next(), sm.next()] }
    }

    /// Create a generator directly from raw state. All-zero state is invalid
    /// for xoshiro, so it is remapped to the `seed = 0` state instead.
    #[inline]
    pub fn from_state(s: [u64; 4]) -> Self {
        if s == [0, 0, 0, 0] {
            Self::new(0)
        } else {
            Self { s }
        }
    }

    /// Produce the next `u64` and advance the state (`**` scrambler).
    #[inline]
    #[expect(
        clippy::should_implement_trait,
        reason = "a PRNG step is infallible and endless, not an `Iterator::next` yielding `Option`"
    )]
    pub fn next(&mut self) -> u64 {
        let result = rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);

        result
    }

    /// Advance the state by `2^128` draws, yielding a non-overlapping stream.
    /// Clone first, then `jump` the clone, to fork independent sub-streams.
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180e_c6d3_3cfd_0aba,
            0xd5a6_1266_f0c9_392c,
            0xa957_2e36_12bf_9a9d,
            0x39ab_dc45_29b1_661c,
        ];
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        let mut s2 = 0u64;
        let mut s3 = 0u64;
        for &jump in &JUMP {
            for bit in 0..64 {
                if (jump & (1u64 << bit)) != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next();
            }
        }
        self.s = [s0, s1, s2, s3];
    }
}

impl Rng for Xoshiro256StarStar {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.next()
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // High bits of `**` output are the strongest.
        (self.next() >> 32) as u32
    }
}
