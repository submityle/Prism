//! Seedable permutation table shared by the gradient-noise generators.

use crate::rng::{Rng, Xoshiro256StarStar};

/// A 512-entry permutation table (the classic Perlin `p[]`, duplicated so hash
/// lookups never need a modulo). Built deterministically from a `u64` seed via
/// a Fisher–Yates shuffle driven by [`Xoshiro256StarStar`].
#[derive(Clone)]
pub struct Permutation {
    table: [u8; 512],
}

impl Permutation {
    /// Build a permutation from a seed.
    pub fn new(seed: u64) -> Self {
        let mut base: [u8; 256] = [0; 256];
        let mut i = 0usize;
        while i < 256 {
            base[i] = i as u8;
            i += 1;
        }

        // Fisher–Yates with a seeded generator.
        let mut rng = Xoshiro256StarStar::new(seed);
        let mut n = 256usize;
        while n > 1 {
            n -= 1;
            let j = rng.range_u64((n as u64) + 1) as usize;
            base.swap(n, j);
        }

        let mut table = [0u8; 512];
        let mut k = 0usize;
        while k < 512 {
            table[k] = base[k & 255];
            k += 1;
        }
        Self { table }
    }

    /// Hash an integer index into `0..=255`.
    #[inline]
    pub fn hash(&self, i: i32) -> u8 {
        self.table[(i & 511) as usize]
    }

    /// Returns a copy of the full 512-entry table so a GPU twin can upload the
    /// identical permutation and keep its integer lookups bit-exact with this
    /// CPU reference (only the fade/gradient floating-point math then needs a
    /// tolerance).
    #[inline]
    pub(crate) fn table_bytes(&self) -> [u8; 512] {
        self.table
    }
}
