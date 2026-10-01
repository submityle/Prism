//! `Philox4x32-10` counter-based pseudo-random number generator
//! (Random123 canonical variant), used for reproducible, massively parallel
//! particle randomness (design: stochastic effects).
//!
//! `Philox` is a *counter-based* `PRNG` (`CBRNG`). Unlike a classic stateful
//! engine such as a linear-feedback or `xorshift` register, it never threads a
//! hidden state from one draw to the next. Instead it treats the stream index
//! itself as the input: the i-th block of randomness is a pure, stateless
//! function `philox4x32(ctr, key)` of a `128`-bit counter (four `u32` words)
//! and a `64`-bit key (two `u32` words). This is exactly what a deterministic
//! `GPU`/`CPU` particle system wants, because any worker, tile, or emitter can
//! jump straight to the random block for a given particle index without
//! replaying the whole sequence, and the same frame replays bit-for-bit on
//! every target.
//!
//! The construction is a `10`-round Feistel-like network built on
//! integer multiply-high/multiply-low (`mulhilo`). Each round multiplies two
//! of the four counter words by fixed odd multipliers, splits each `64`-bit
//! product into its high and low halves, and recombines them with the other
//! two counter words and the round key via `XOR`. Between rounds the key is
//! bumped by the two Weyl constants derived from the golden ratio and
//! `sqrt(3)`. Repeated multiply-and-mix is what gives the generator its
//! avalanche: a single input bit influences essentially every output bit after
//! the ten rounds.
//!
//! Every operation here is `32`-bit integer arithmetic, with the only wider
//! value being the transient `u64` product inside `mulhilo`. There is no
//! floating point, no division, and no transcendental function anywhere in
//! generation, so results are identical on every target.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `philox4x32` function plus a thin counter-walking
//! wrapper, and it shares no code with the `squares`, `xoshiro`, or `xorshift`
//! engines elsewhere in this crate. Do not conflate them.
//!
//! This generator is fast and non-cryptographic. A `Philox` stream is
//! predictable once the key is known, so it must never be used for security,
//! key material, or anywhere an adversary could exploit predictability. It
//! exists purely for reproducible, high-throughput simulation randomness.

/// First-lane counter multiplier (an odd `32`-bit constant from Random123).
const M0: u32 = 0xD251_1F53;
/// Second-lane counter multiplier (an odd `32`-bit constant from Random123).
const M1: u32 = 0xCD9E_8D57;
/// First Weyl key-bump constant (fractional bits of the golden ratio).
const W0: u32 = 0x9E37_79B9;
/// Second Weyl key-bump constant (fractional bits of `sqrt(3)`).
const W1: u32 = 0xBB67_AE85;
/// Number of mixing rounds in the canonical `Philox4x32-10` variant.
const ROUNDS: usize = 10;

/// Multiply two `u32` values and return the `(high, low)` halves of the
/// `64`-bit product.
///
/// The product is computed in a transient `u64` and split into its upper and
/// lower `32`-bit words. This is the single multiply primitive the round
/// function is built on; the `u64` is purely an intermediate and never leaks
/// into the public result.
#[inline]
#[must_use]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    (((p >> 32) as u32), (p as u32))
}

/// Apply one `Philox4x32` round to the counter `c` under the round key `k`.
///
/// Two lanes are multiplied by the fixed multipliers and each product is split
/// via [`mulhilo`]; the high and low halves are then cross-combined with the
/// surviving counter words and the key through `XOR`.
#[inline]
#[must_use]
fn round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, c[0]);
    let (hi1, lo1) = mulhilo(M1, c[2]);
    [(hi1 ^ c[1] ^ k[0]), lo1, (hi0 ^ c[3] ^ k[1]), lo0]
}

/// Advance the key by the two Weyl constants (one bump between rounds).
#[inline]
#[must_use]
fn bump_key(k: [u32; 2]) -> [u32; 2] {
    [k[0].wrapping_add(W0), k[1].wrapping_add(W1)]
}

/// Compute the `Philox4x32-10` output block for counter `ctr` under `key`.
///
/// This is the canonical ten-round Random123 generator. It is a pure function
/// of its inputs: the same `(ctr, key)` pair always yields the same four `u32`
/// words, which is the defining property of a counter-based `PRNG`. All
/// arithmetic is `32`-bit wrapping and bitwise work plus the transient `u64`
/// product inside [`mulhilo`], so the result is identical on every target.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::philox_counter::philox4x32;
///
/// let out = philox4x32([0, 0, 0, 0], [0, 0]);
/// assert_eq!(out, [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
/// ```
#[inline]
#[must_use]
pub fn philox4x32(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    for _ in 0..ROUNDS {
        c = round(c, k);
        k = bump_key(k);
    }
    c
}

/// Stateful counter-walking wrapper around [`philox4x32`].
///
/// The stream is defined by a fixed `64`-bit key and a `128`-bit counter that
/// starts at zero and increments by one (little-endian across the four words)
/// after each block is produced. Each [`next_block`](Philox4x32::next_block)
/// call returns the `Philox` output for the current counter and then advances
/// it, so successive calls walk distinct, reproducible blocks of randomness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Philox4x32 {
    ctr: [u32; 4],
    key: [u32; 2],
}

impl Philox4x32 {
    /// Create a new stream for `key`, starting the counter at `[0, 0, 0, 0]`.
    #[inline]
    #[must_use]
    pub fn new(key: [u32; 2]) -> Self {
        Self {
            ctr: [0, 0, 0, 0],
            key,
        }
    }

    /// Return the current counter block, then increment the counter.
    ///
    /// The returned value is `philox4x32(current_ctr, key)`. Afterwards the
    /// `128`-bit counter is incremented by one with little-endian carry: word
    /// `0` wraps first, carrying into word `1`, then `2`, then `3`.
    #[inline]
    pub fn next_block(&mut self) -> [u32; 4] {
        let out = philox4x32(self.ctr, self.key);
        self.increment();
        out
    }

    /// The current counter value (the next block to be produced).
    #[inline]
    #[must_use]
    pub fn counter(&self) -> [u32; 4] {
        self.ctr
    }

    /// The fixed key for this stream.
    #[inline]
    #[must_use]
    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    /// Increment the `128`-bit counter by one with little-endian carry.
    #[inline]
    fn increment(&mut self) {
        for word in &mut self.ctr {
            let (next, carried) = word.overflowing_add(1);
            *word = next;
            if !carried {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    /// Reference counter from the Random123 `digits-of-pi` test vector.
    const PI_CTR: [u32; 4] = [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344];
    /// Reference key from the Random123 `digits-of-pi` test vector.
    const PI_KEY: [u32; 2] = [0xa409_3822, 0x299f_31d0];

    // --- Hard reference vectors (verified against an independent impl) ---

    #[test]
    fn hard_vector_all_zero() {
        assert_eq!(
            philox4x32([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn hard_vector_digits_of_pi() {
        assert_eq!(
            philox4x32(PI_CTR, PI_KEY),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn hard_vector_word0_zero() {
        assert_eq!(philox4x32([0, 0, 0, 0], [0, 0])[0], 0x6627_e8d5);
    }

    #[test]
    fn hard_vector_word1_zero() {
        assert_eq!(philox4x32([0, 0, 0, 0], [0, 0])[1], 0xe169_c58d);
    }

    #[test]
    fn hard_vector_word2_zero() {
        assert_eq!(philox4x32([0, 0, 0, 0], [0, 0])[2], 0xbc57_ac4c);
    }

    #[test]
    fn hard_vector_word3_zero() {
        assert_eq!(philox4x32([0, 0, 0, 0], [0, 0])[3], 0x9b00_dbd8);
    }

    #[test]
    fn hard_vector_pi_word0() {
        assert_eq!(philox4x32(PI_CTR, PI_KEY)[0], 0xd16c_fe09);
    }

    #[test]
    fn hard_vector_pi_word3() {
        assert_eq!(philox4x32(PI_CTR, PI_KEY)[3], 0x2412_6ea1);
    }

    // --- mulhilo primitive ---

    #[test]
    fn mulhilo_zero_operand_is_zero() {
        assert_eq!(mulhilo(0, 0xFFFF_FFFF), (0, 0));
        assert_eq!(mulhilo(0xFFFF_FFFF, 0), (0, 0));
    }

    #[test]
    fn mulhilo_small_product_has_zero_high() {
        assert_eq!(mulhilo(3, 7), (0, 21));
    }

    #[test]
    fn mulhilo_max_square_matches_u64() {
        let a = 0xFFFF_FFFFu32;
        let p = (a as u64) * (a as u64);
        let (hi, lo) = mulhilo(a, a);
        assert_eq!(((hi as u64) << 32) | (lo as u64), p);
    }

    #[test]
    fn mulhilo_recomposes_product() {
        let a = 0x1234_5678u32;
        let b = 0x9ABC_DEF0u32;
        let (hi, lo) = mulhilo(a, b);
        let recomposed = ((hi as u64) << 32) | (lo as u64);
        assert_eq!(recomposed, (a as u64) * (b as u64));
    }

    #[test]
    fn mulhilo_high_only_when_product_overflows_32_bits() {
        let (hi, _lo) = mulhilo(0x1_0000, 0x1_0000);
        assert_eq!(hi, 1);
    }

    // --- Determinism ---

    #[test]
    fn determinism_all_zero() {
        assert_eq!(
            philox4x32([0, 0, 0, 0], [0, 0]),
            philox4x32([0, 0, 0, 0], [0, 0])
        );
    }

    #[test]
    fn determinism_pi_vector() {
        assert_eq!(philox4x32(PI_CTR, PI_KEY), philox4x32(PI_CTR, PI_KEY));
    }

    #[test]
    fn determinism_over_many_counters() {
        for i in 0..256u32 {
            let ctr = [i, i.wrapping_mul(3), 0, 0];
            assert_eq!(philox4x32(ctr, PI_KEY), philox4x32(ctr, PI_KEY));
        }
    }

    // --- Distinct inputs produce distinct outputs ---

    #[test]
    fn distinct_counter_distinct_output() {
        assert!(philox4x32([0, 0, 0, 0], [0, 0]) != philox4x32([1, 0, 0, 0], [0, 0]));
    }

    #[test]
    fn distinct_key_distinct_output() {
        assert!(philox4x32([0, 0, 0, 0], [0, 0]) != philox4x32([0, 0, 0, 0], [1, 0]));
    }

    #[test]
    fn distinct_key_high_word_distinct_output() {
        assert!(philox4x32([5, 0, 0, 0], [0, 0]) != philox4x32([5, 0, 0, 0], [0, 1]));
    }

    #[test]
    fn distinct_counter_high_word_distinct_output() {
        assert!(philox4x32([0, 0, 0, 0], [9, 9]) != philox4x32([0, 0, 0, 1], [9, 9]));
    }

    #[test]
    fn many_counters_low_collision_rate() {
        let mut seen = BTreeSet::new();
        let mut collisions = 0u32;
        for i in 0..4096u32 {
            let block = philox4x32([i, 0, 0, 0], [0, 0]);
            if !seen.insert(block[0]) {
                collisions += 1;
            }
        }
        // Birthday collisions in 4096 draws of a 32-bit space are rare.
        assert!(collisions <= 4);
    }

    #[test]
    fn full_blocks_unique_in_sample() {
        let mut seen = BTreeSet::new();
        for i in 0..2048u32 {
            let block = philox4x32([i, 7, 0, 0], PI_KEY);
            assert!(seen.insert(block));
        }
    }

    // --- Avalanche behaviour ---

    #[test]
    fn counter_bit_flip_avalanches_output() {
        let a = philox4x32([0x0001_0000, 0, 0, 0], [0, 0]);
        let b = philox4x32([0x0001_0040, 0, 0, 0], [0, 0]);
        let diff: u32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        assert!(diff >= 32);
    }

    #[test]
    fn key_bit_flip_avalanches_output() {
        let a = philox4x32([777, 0, 0, 0], [0, 0]);
        let b = philox4x32([777, 0, 0, 0], [1 << 20, 0]);
        let diff: u32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        assert!(diff >= 32);
    }

    #[test]
    fn output_bits_are_well_balanced() {
        let mut ones = 0u32;
        for i in 0..512u32 {
            let block = philox4x32([i, 0, 0, 0], PI_KEY);
            ones += block.iter().map(|w| w.count_ones()).sum::<u32>();
        }
        // 512 blocks * 128 bits = 65536 bits; expect roughly half set.
        assert!((24_000..=42_000).contains(&ones));
    }

    // --- round / bump_key internals ---

    #[test]
    fn single_round_is_deterministic() {
        assert_eq!(round([1, 2, 3, 4], [5, 6]), round([1, 2, 3, 4], [5, 6]));
    }

    #[test]
    fn round_zero_counter_depends_on_key() {
        // With a zero counter, only the key-xor lanes carry the key through.
        let r = round([0, 0, 0, 0], [0xAAAA_AAAA, 0x5555_5555]);
        assert_eq!(r[0], 0xAAAA_AAAA);
        assert_eq!(r[2], 0x5555_5555);
        assert_eq!(r[1], 0);
        assert_eq!(r[3], 0);
    }

    #[test]
    fn bump_key_adds_weyl_constants() {
        assert_eq!(bump_key([0, 0]), [W0, W1]);
    }

    #[test]
    fn bump_key_wraps_on_overflow() {
        assert_eq!(
            bump_key([0xFFFF_FFFF, 0xFFFF_FFFF]),
            [W0.wrapping_sub(1), W1.wrapping_sub(1)]
        );
    }

    // --- Philox4x32 stream wrapper ---

    #[test]
    fn new_starts_counter_at_zero() {
        let stream = Philox4x32::new([1, 2]);
        assert_eq!(stream.counter(), [0, 0, 0, 0]);
        assert_eq!(stream.key(), [1, 2]);
    }

    #[test]
    fn first_block_matches_zero_counter() {
        let mut stream = Philox4x32::new([0, 0]);
        assert_eq!(stream.next_block(), philox4x32([0, 0, 0, 0], [0, 0]));
    }

    #[test]
    fn next_block_advances_counter() {
        let mut stream = Philox4x32::new([3, 4]);
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [1, 0, 0, 0]);
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [2, 0, 0, 0]);
    }

    #[test]
    fn stream_matches_manual_counter_walk() {
        let key = [0xDEAD_BEEF, 0x0BAD_F00D];
        let mut stream = Philox4x32::new(key);
        for i in 0..64u32 {
            let got = stream.next_block();
            let expected = philox4x32([i, 0, 0, 0], key);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn stream_blocks_are_distinct() {
        let mut stream = Philox4x32::new(PI_KEY);
        let mut seen = BTreeSet::new();
        for _ in 0..128 {
            assert!(seen.insert(stream.next_block()));
        }
    }

    #[test]
    fn stream_collects_into_vec() {
        let mut stream = Philox4x32::new([7, 7]);
        let blocks: Vec<[u32; 4]> = (0..8).map(|_| stream.next_block()).collect();
        assert_eq!(blocks.len(), 8);
        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(*block, philox4x32([i as u32, 0, 0, 0], [7, 7]));
        }
    }

    // --- Counter carry edge cases ---

    #[test]
    fn carry_from_word0_into_word1() {
        let mut stream = Philox4x32::new([0, 0]);
        stream.ctr = [0xFFFF_FFFF, 0, 0, 0];
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [0, 1, 0, 0]);
    }

    #[test]
    fn carry_propagates_through_word2() {
        let mut stream = Philox4x32::new([0, 0]);
        stream.ctr = [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0];
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [0, 0, 1, 0]);
    }

    #[test]
    fn carry_propagates_to_top_word() {
        let mut stream = Philox4x32::new([0, 0]);
        stream.ctr = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0];
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [0, 0, 0, 1]);
    }

    #[test]
    fn carry_wraps_fully_to_zero() {
        let mut stream = Philox4x32::new([0, 0]);
        stream.ctr = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF];
        let _ = stream.next_block();
        assert_eq!(stream.counter(), [0, 0, 0, 0]);
    }

    #[test]
    fn block_at_carry_boundary_matches_pure_function() {
        let key = [0x1111_2222, 0x3333_4444];
        let mut stream = Philox4x32::new(key);
        stream.ctr = [0xFFFF_FFFF, 0, 0, 0];
        let got = stream.next_block();
        assert_eq!(got, philox4x32([0xFFFF_FFFF, 0, 0, 0], key));
        assert_eq!(stream.counter(), [0, 1, 0, 0]);
    }

    #[test]
    fn stream_is_copy_and_independent() {
        let mut a = Philox4x32::new([2, 2]);
        let mut b = a;
        let _ = a.next_block();
        assert_eq!(a.counter(), [1, 0, 0, 0]);
        assert_eq!(b.counter(), [0, 0, 0, 0]);
        assert_eq!(b.next_block(), philox4x32([0, 0, 0, 0], [2, 2]));
    }

    #[test]
    fn high_counter_words_affect_output() {
        let a = philox4x32([0, 0, 0, 0], [0, 0]);
        let b = philox4x32([0, 0, 1, 0], [0, 0]);
        let c = philox4x32([0, 1, 0, 0], [0, 0]);
        assert!(a != b);
        assert!(a != c);
        assert!(b != c);
    }
}
