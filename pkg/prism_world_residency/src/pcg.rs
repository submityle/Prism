use core::fmt;

/// PCG-style avalanche hash for stable ids, seeds, and dither inputs.
pub const fn pcg_hash_u32(input: u64) -> u32 {
    let mut state = input.wrapping_add(0x853c_49e6_748f_ea9b);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let word = ((state >> 18) ^ state) >> 27;
    let rotation = (state >> 59) as u32;
    let value = word as u32;
    value.rotate_right(rotation)
}

/// Deterministic PCG32 generator with an explicit stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    /// Creates a generator from a seed and odd stream selector.
    pub const fn new(seed: u64, stream: u64) -> Self {
        let increment = (stream << 1) | 1;
        let mut generator = Self {
            state: 0,
            increment,
        };
        let _ = generator.next_u32();
        generator.state = generator.state.wrapping_add(seed);
        let _ = generator.next_u32();
        generator
    }

    /// Returns the next deterministic 32-bit value.
    pub const fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.increment);
        let word = ((old_state >> 18) ^ old_state) >> 27;
        let rotation = (old_state >> 59) as u32;
        (word as u32).rotate_right(rotation)
    }

    /// Returns two consecutive values as one deterministic 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// Returns a value in `[0, upper)` using rejection sampling.
    pub fn next_bounded(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            return 0;
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % upper;
            }
        }
    }

    /// Returns a Q16 fixed-point unit value in `[0, 65535]`.
    pub fn next_unit_q16(&mut self) -> u16 {
        (self.next_u32() >> 16) as u16
    }
}

/// A stable 128-bit key for an invalidatable derived cache.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DerivedCacheKey {
    /// Upper half containing the namespace and content mix.
    pub hi: u64,
    /// Lower half containing the variant and content mix.
    pub lo: u64,
}

impl fmt::Display for DerivedCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}{:016x}", self.hi, self.lo)
    }
}

/// Derives a cache key from namespace, content hash, and representation variant.
pub const fn derive_cache_key(namespace: u64, content_hash: u64, variant: u32) -> DerivedCacheKey {
    let variant_mix = (variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let hi = (namespace ^ content_hash.rotate_left(17) ^ variant_mix)
        .wrapping_mul(0xd6e8_feb8_6659_fd93);
    let lo = (content_hash ^ namespace.rotate_right(23) ^ variant_mix.rotate_left(29))
        .wrapping_mul(0xa5a3_564d_2c5b_3d91);
    DerivedCacheKey { hi, lo }
}
