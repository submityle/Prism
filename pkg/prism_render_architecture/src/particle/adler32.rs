//! `Adler-32` checksum (`RFC 1950`, the running sum used by `zlib`): a
//! pure-integer, modulo-`65521` rolling checksum for cheaply verifying the
//! integrity of `GPU` resource blobs and streamed particle payloads (design §
//! integrity checks).
//!
//! The checksum keeps two running sums over the input bytes. The first sum `a`
//! starts at `1` and accumulates each byte; the second sum `b` starts at `0`
//! and accumulates `a` after every byte. Both sums are kept modulo the largest
//! prime below `2^16`, `65521` (exposed as [`ADLER_MOD`]). The finished
//! checksum packs the two 16-bit sums into one 32-bit word as `(b << 16) | a`.
//! For the empty input the sums stay at their seeds, so the checksum is `1`.
//!
//! The one-shot [`adler32`] and the streaming [`Adler32`] accumulator both use
//! the classic deferred-reduction optimisation: bytes are consumed in blocks of
//! at most `NMAX` (`5552`) before a single modulo reduction is applied to each
//! sum. `NMAX` is the largest block length for which the 32-bit accumulators
//! provably cannot overflow between reductions, so the fast blocked loop yields
//! exactly the same result as the naive byte-at-a-time definition that reduces
//! after every byte. Every operation is an integer add, shift, or remainder;
//! there are no floating-point or transcendental operations.
//!
//! [`combine`] merges the checksums of two consecutive segments when the second
//! segment's length is known, matching `zlib`'s `adler32_combine`. Because
//! `Adler-32` is an affine function of the byte stream, the merge needs only a
//! modular multiply (implemented here with the integer [`mod_mul`]
//! double-and-add loop); a general modular exponentiation helper `mod_pow`
//! (square-and-multiply, no `pow` family calls) is provided for completeness and
//! reuse.
//!
//! Scope: this is an error-detection code, not a hash and not a message
//! authentication code. `Adler-32` is *not* cryptographically secure and
//! collisions are trivial to construct on purpose, so it must never be used to
//! authenticate data or guard against a malicious adversary. It is meant only
//! for catching accidental corruption in transit or storage. For
//! content-addressing or security use a real hash instead.

/// The `Adler-32` modulus: the largest prime below `2^16`.
pub const ADLER_MOD: u32 = 65521;

/// The largest block of bytes that can be summed before a modulo reduction
/// without overflowing the 32-bit accumulators.
///
/// This is the same bound `zlib` uses: it is the largest `n` for which
/// `255 * n * (n + 1) / 2 + (n + 1) * (ADLER_MOD - 1)` still fits in a `u32`.
const NMAX: usize = 5552;

/// Computes the `Adler-32` checksum of `data` in a single call.
///
/// The empty slice yields `1`, and `adler32(b"Wikipedia")` is the well-known
/// reference value `0x11E60398`. The blocked, deferred-reduction loop produces
/// exactly the same value as reducing after every byte.
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(NMAX) {
        for &byte in chunk {
            a = a.wrapping_add(u32::from(byte));
            b = b.wrapping_add(a);
        }
        a %= ADLER_MOD;
        b %= ADLER_MOD;
    }
    (b << 16) | a
}

/// A streaming `Adler-32` accumulator for folding a byte stream in many chunks.
///
/// Construct one with [`Adler32::new`], feed bytes with [`Adler32::update`] as
/// many times as needed, and read the checksum with [`Adler32::finalize`]. By
/// construction, folding a byte stream in one `update` call or in several
/// yields the same checksum as [`adler32`] over the concatenated bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adler32 {
    /// The running first sum, reduced modulo [`ADLER_MOD`] after each update.
    a: u32,
    /// The running second sum, reduced modulo [`ADLER_MOD`] after each update.
    b: u32,
}

impl Adler32 {
    /// Creates a fresh accumulator seeded with `a = 1` and `b = 0`.
    #[must_use]
    pub const fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    /// Folds every byte of `data` into the running sums.
    ///
    /// The sums are reduced modulo [`ADLER_MOD`] at every `NMAX`-byte block
    /// boundary and again at the end of the call, so the accumulator state is
    /// always fully reduced between calls and the blocked loop cannot overflow.
    pub fn update(&mut self, data: &[u8]) {
        let mut a = self.a;
        let mut b = self.b;
        for chunk in data.chunks(NMAX) {
            for &byte in chunk {
                a = a.wrapping_add(u32::from(byte));
                b = b.wrapping_add(a);
            }
            a %= ADLER_MOD;
            b %= ADLER_MOD;
        }
        self.a = a;
        self.b = b;
    }

    /// Returns the finished checksum packed as `(b << 16) | a`.
    #[must_use]
    pub const fn finalize(&self) -> u32 {
        (self.b << 16) | self.a
    }

    /// Resets the accumulator back to its initial `a = 1`, `b = 0` seeds.
    pub fn reset(&mut self) {
        self.a = 1;
        self.b = 0;
    }
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Multiplies `lhs` by `rhs` modulo `modulus` using a double-and-add loop.
///
/// This keeps every intermediate value below `2 * modulus`, so it never
/// overflows even when the mathematical product would exceed `u64`. For a
/// `modulus` of `0` it returns `0`.
fn mod_mul(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if modulus == 0 {
        return 0;
    }
    let mut result: u64 = 0;
    let mut base = lhs % modulus;
    let mut exp = rhs;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = (result + base) % modulus;
        }
        base = (base + base) % modulus;
        exp >>= 1;
    }
    result
}

/// Raises `base` to `exp` modulo `modulus` using square-and-multiply.
///
/// A pure-integer fast-exponentiation helper that relies only on [`mod_mul`]
/// and bit shifts; it never calls any `pow`, `powi`, or `powf` routine. For a
/// `modulus` of `1` (or `0`) it returns `0`, matching the convention that every
/// residue collapses to `0`.
#[cfg(test)]
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }
    let mut result: u64 = 1 % modulus;
    let mut acc = base % modulus;
    let mut remaining = exp;
    while remaining > 0 {
        if (remaining & 1) == 1 {
            result = mod_mul(result, acc, modulus);
        }
        acc = mod_mul(acc, acc, modulus);
        remaining >>= 1;
    }
    result
}

/// Merges the `Adler-32` checksums of two consecutive segments.
///
/// Given `adler1 = adler32(x)`, `adler2 = adler32(y)`, and `len2 = y.len()`,
/// returns `adler32(x ++ y)` without rescanning either segment. This matches
/// `zlib`'s `adler32_combine`. It is not a `const fn` because the modular
/// multiply runs an integer loop.
#[must_use]
pub fn combine(adler1: u32, adler2: u32, len2: u64) -> u32 {
    let modulus = u64::from(ADLER_MOD);
    let rem = len2 % modulus;

    let sum1 = u64::from(adler1 & 0xFFFF);
    let carry1 = u64::from((adler1 >> 16) & 0xFFFF);
    let sum2 = u64::from(adler2 & 0xFFFF);
    let carry2 = u64::from((adler2 >> 16) & 0xFFFF);

    let combined_a = (sum1 + sum2 + modulus - 1) % modulus;
    let product = mod_mul(rem, sum1, modulus);
    let combined_b = (product + carry1 + carry2 + modulus - rem) % modulus;

    ((combined_b << 16) | combined_a) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A tiny deterministic linear congruential generator for test data.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        const fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u8(&mut self) -> u8 {
            // Numerical Recipes constants; wrapping keeps it in-range.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 56) as u8
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.next_u8()).collect()
        }
    }

    /// The naive byte-at-a-time reference definition (reduce after every byte).
    fn naive(data: &[u8]) -> u32 {
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a + u32::from(byte)) % ADLER_MOD;
            b = (b + a) % ADLER_MOD;
        }
        (b << 16) | a
    }

    #[test]
    fn empty_input_is_one() {
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn wikipedia_reference_vector() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn single_byte_lowercase_a() {
        // a = 1 + 97 = 98 = 0x62, b = 98 = 0x62.
        assert_eq!(adler32(b"a"), 0x0062_0062);
    }

    #[test]
    fn short_ascii_abc() {
        assert_eq!(adler32(b"abc"), 0x024D_0127);
    }

    #[test]
    fn single_zero_byte() {
        // a stays 1, b becomes 1.
        assert_eq!(adler32(&[0u8]), 0x0001_0001);
    }

    #[test]
    fn two_zero_bytes() {
        // a stays 1, b becomes 2.
        assert_eq!(adler32(&[0u8, 0u8]), 0x0002_0001);
    }

    #[test]
    fn many_zero_bytes_tracks_length() {
        let data = [0u8; 16];
        // a stays 1, b becomes the length (16 = 0x10).
        assert_eq!(adler32(&data), 0x0010_0001);
    }

    #[test]
    fn all_high_bytes_matches_naive() {
        let data = [0xFFu8; 40];
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn naive_matches_optimized_for_small_lengths() {
        let mut rng = Lcg::new(0x1234_5678);
        for len in 0usize..300 {
            let data = rng.bytes(len);
            assert_eq!(adler32(&data), naive(&data), "mismatch at len {len}");
        }
    }

    #[test]
    fn new_then_finalize_is_one() {
        assert_eq!(Adler32::new().finalize(), 1);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Adler32::default(), Adler32::new());
    }

    #[test]
    fn streaming_single_update_equals_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut acc = Adler32::new();
        acc.update(data);
        assert_eq!(acc.finalize(), adler32(data));
    }

    #[test]
    fn streaming_two_chunks_equals_oneshot() {
        let data = b"Wikipedia";
        let mut acc = Adler32::new();
        acc.update(&data[..4]);
        acc.update(&data[4..]);
        assert_eq!(acc.finalize(), adler32(data));
    }

    #[test]
    fn streaming_byte_by_byte_equals_oneshot() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let data = rng.bytes(512);
        let mut acc = Adler32::new();
        for &byte in &data {
            acc.update(&[byte]);
        }
        assert_eq!(acc.finalize(), adler32(&data));
    }

    #[test]
    fn streaming_arbitrary_splits_equal_oneshot() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        let data = rng.bytes(2000);
        for split in [0usize, 1, 17, 999, 1999, 2000] {
            let mut acc = Adler32::new();
            acc.update(&data[..split]);
            acc.update(&data[split..]);
            assert_eq!(acc.finalize(), adler32(&data), "split {split}");
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut acc = Adler32::new();
        acc.update(b"garbage data");
        acc.reset();
        assert_eq!(acc.finalize(), 1);
    }

    #[test]
    fn reset_then_reuse_matches_fresh() {
        let mut acc = Adler32::new();
        acc.update(b"stale");
        acc.reset();
        acc.update(b"Wikipedia");
        assert_eq!(acc.finalize(), adler32(b"Wikipedia"));
    }

    #[test]
    fn block_boundary_exact_nmax_matches_naive() {
        let mut rng = Lcg::new(0x5552_5552);
        let data = rng.bytes(NMAX);
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn block_boundary_nmax_minus_one_matches_naive() {
        let mut rng = Lcg::new(0x5551_5551);
        let data = rng.bytes(NMAX - 1);
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn block_boundary_nmax_plus_one_matches_naive() {
        let mut rng = Lcg::new(0x5553_5553);
        let data = rng.bytes(NMAX + 1);
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn two_blocks_plus_change_matches_naive() {
        let mut rng = Lcg::new(0xABCD_1234);
        let data = rng.bytes(2 * NMAX + 123);
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn large_all_high_bytes_matches_naive() {
        let data = [0xFFu8; 2 * NMAX + 7];
        assert_eq!(adler32(&data), naive(&data));
    }

    #[test]
    fn combine_two_halves_of_wikipedia() {
        let full = b"Wikipedia";
        let left = &full[..4];
        let right = &full[4..];
        let merged = combine(adler32(left), adler32(right), right.len() as u64);
        assert_eq!(merged, adler32(full));
        assert_eq!(merged, 0x11E6_0398);
    }

    #[test]
    fn combine_matches_concatenation_at_many_splits() {
        let mut rng = Lcg::new(0xFEED_FACE);
        let data = rng.bytes(777);
        for split in [0usize, 1, 2, 50, 388, 776, 777] {
            let left = &data[..split];
            let right = &data[split..];
            let merged = combine(adler32(left), adler32(right), right.len() as u64);
            assert_eq!(merged, adler32(&data), "split {split}");
        }
    }

    #[test]
    fn combine_with_empty_second_segment_is_identity() {
        let adler1 = adler32(b"first segment");
        assert_eq!(combine(adler1, adler32(b""), 0), adler1);
    }

    #[test]
    fn combine_with_empty_first_segment_is_identity() {
        let right = b"second segment";
        let adler2 = adler32(right);
        assert_eq!(combine(adler32(b""), adler2, right.len() as u64), adler2);
    }

    #[test]
    fn combine_three_segments_matches_whole() {
        let mut rng = Lcg::new(0x1111_2222);
        let data = rng.bytes(600);
        let (p1, p2) = (173usize, 420usize);
        let a1 = adler32(&data[..p1]);
        let a2 = adler32(&data[p1..p2]);
        let a3 = adler32(&data[p2..]);
        let left = combine(a1, a2, (p2 - p1) as u64);
        let whole = combine(left, a3, (data.len() - p2) as u64);
        assert_eq!(whole, adler32(&data));
    }

    #[test]
    fn combine_across_block_boundary_matches() {
        let mut rng = Lcg::new(0x9999_0000);
        let data = rng.bytes(2 * NMAX + 64);
        let split = NMAX + 10;
        let merged = combine(
            adler32(&data[..split]),
            adler32(&data[split..]),
            (data.len() - split) as u64,
        );
        assert_eq!(merged, adler32(&data));
    }

    #[test]
    fn mod_pow_exponent_zero_is_one() {
        assert_eq!(mod_pow(5, 0, 65521), 1);
        assert_eq!(mod_pow(0, 0, 65521), 1);
    }

    #[test]
    fn mod_pow_two_to_the_ten_mod_thousand() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
    }

    #[test]
    fn mod_pow_two_to_the_sixteen_mod_adler() {
        // 65536 % 65521 == 15.
        assert_eq!(mod_pow(2, 16, 65521), 15);
    }

    #[test]
    fn mod_pow_square_known_value() {
        assert_eq!(mod_pow(7, 2, 100), 49);
    }

    #[test]
    fn mod_pow_modulus_one_is_zero() {
        assert_eq!(mod_pow(123, 456, 1), 0);
    }

    #[test]
    fn mod_pow_matches_iterative_multiplication() {
        let modulus = 65521u64;
        for base in [0u64, 1, 2, 3, 7, 100, 65520] {
            for exp in 0u64..12 {
                let mut expected = 1u64 % modulus;
                let mut count = 0u64;
                while count < exp {
                    expected = (expected * (base % modulus)) % modulus;
                    count += 1;
                }
                assert_eq!(mod_pow(base, exp, modulus), expected, "{base}^{exp}");
            }
        }
    }

    #[test]
    fn mod_mul_matches_direct_product() {
        let modulus = 65521u64;
        for lhs in [0u64, 1, 2, 65520, 40000] {
            for rhs in [0u64, 1, 7, 65520, 50000] {
                assert_eq!(mod_mul(lhs, rhs, modulus), (lhs * rhs) % modulus);
            }
        }
    }
}
