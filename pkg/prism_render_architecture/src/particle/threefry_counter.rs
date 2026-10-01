//! The `Threefry2x32-20` counter-based pseudo-random number generator from the
//! `Random123` family of John K. Salmon, Mark A. Moraes, Ron O. Dror, and
//! David E. Shaw.
//!
//! Unlike a conventional stateful `PRNG` that mutates a hidden register from one
//! draw to the next, a *counter-based* generator is a pure, stateless function
//! of an explicit counter and key. Each `64`-bit counter maps through a reduced
//! `Threefish`-style block mixing function to a `64`-bit block of two `u32`
//! words. Advancing the stream is nothing more than incrementing the counter,
//! so any block can be produced in constant time without replaying the ones
//! before it. That property is exactly what a deterministic particle system
//! needs: the `CPU` reference path and a future `GPU` kernel can each compute
//! the random block for a given particle index and frame independently and
//! still agree bit for bit.
//!
//! The mixing function alternates two operations across twenty rounds. Every
//! round performs an add-rotate-`XOR` step on the two `u32` words, drawing the
//! rotation amount from an eight-entry schedule. After every fourth round a
//! *key injection* folds words of the expanded key schedule back into the
//! state, which is what breaks the otherwise linear structure and gives the
//! generator its statistical quality. The key schedule is three words: the two
//! key words plus a third parity word formed by `XOR`-ing both key words with a
//! fixed constant.
//!
//! Scope and boundaries: this module is deliberately narrow and fully
//! self-contained. It implements only the `2x32` variant with twenty rounds and
//! shares no code with any wider `4x32`, `2x64`, or `4x64` variant, nor with the
//! `Philox` multiply-based members of the same family. Those are separate
//! constructions and must not be conflated with this one.
//!
//! Every transition uses only wrapping integer addition, exclusive-or, and
//! [`u32::rotate_left`]. There is no floating point, no division, and no
//! transcendental function anywhere in generation.
//!
//! This generator is fast and non-cryptographic. Although it descends from a
//! block-cipher-like construction, the reduced round count and public counter
//! make it unsuitable for security, key material, or any setting where an
//! adversary could exploit predictability. It exists purely for reproducible,
//! high-throughput simulation randomness.

/// The parity constant folded into the third key-schedule word.
///
/// Expressed in `hex` as `0x1BD1_1BDA`, this is the `Threefish`/`Threefry`
/// key-schedule parity word `C240` truncated to `32` bits. It ensures the
/// synthetic third key word differs from a plain `XOR` of the two supplied key
/// words, so an all-zero key still drives a non-trivial schedule.
const PARITY: u32 = 0x1BD1_1BDA;

/// The eight rotation amounts cycled through across the twenty rounds.
///
/// Round `r` rotates the second state word left by `ROT[r % 8]`. These
/// constants are the published `Threefry2x32` rotation schedule and are what
/// give each round its diffusion; using any other values would produce a
/// different, non-canonical generator.
const ROT: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// The number of mixing rounds. Twenty rounds is the canonical `Threefry2x32-20`
/// strength endorsed by the `Random123` authors for general use.
const ROUNDS: usize = 20;

/// Computes one `Threefry2x32-20` output block from a counter and key.
///
/// This is the pure, stateless core of the generator. Given a `64`-bit counter
/// as two `u32` words and a `64`-bit key as two `u32` words, it returns the
/// `64`-bit output block as two `u32` words. The same `(ctr, key)` pair always
/// yields the same block, and distinct pairs overwhelmingly yield distinct
/// blocks; this is the property counter-based generation relies on.
///
/// The implementation expands a three-word key schedule, seeds the two state
/// words with the counter plus the first two schedule words, then runs twenty
/// add-rotate-`XOR` rounds, injecting schedule words after every fourth round.
#[must_use]
pub fn threefry2x32(ctr: [u32; 2], key: [u32; 2]) -> [u32; 2] {
    // Expanded three-word key schedule: the two key words plus a parity word.
    let ks = [key[0], key[1], (PARITY ^ key[0]) ^ key[1]];

    // Seed the state with the counter plus the first two schedule words.
    let mut x0 = ctr[0].wrapping_add(ks[0]);
    let mut x1 = ctr[1].wrapping_add(ks[1]);

    // Counts key injections as they happen; takes the values 1..=5. Keeping it
    // a `u32` avoids a lossy `usize`-to-`u32` cast for the final addend.
    let mut inject: u32 = 0;

    for r in 0..ROUNDS {
        // Add-rotate-XOR mixing step.
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(ROT[r % 8]);
        x1 ^= x0;

        // Key injection after every fourth round.
        if (r % 4) == 3 {
            inject += 1; // Injection index in the range 1..=5.
            let s = inject as usize;
            x0 = x0.wrapping_add(ks[s % 3]);
            x1 = x1.wrapping_add(ks[(s + 1) % 3]).wrapping_add(inject);
        }
    }

    [x0, x1]
}

/// A counter-stream wrapper around [`threefry2x32`].
///
/// The value holds a fixed key and a mutable `64`-bit counter. Each call to
/// [`Threefry2x32::next_block`] returns the block for the current counter and
/// then advances the counter by one, treating the two counter words as a
/// little-endian `64`-bit integer. Cloning the value branches an identical
/// stream from the current position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Threefry2x32 {
    /// The `64`-bit counter as two `u32` words, little-endian (`ctr[0]` low).
    ctr: [u32; 2],
    /// The fixed `64`-bit key as two `u32` words.
    key: [u32; 2],
}

impl Threefry2x32 {
    /// Creates a new counter stream for `key`, starting from counter `[0, 0]`.
    #[must_use]
    pub fn new(key: [u32; 2]) -> Self {
        Self { ctr: [0, 0], key }
    }

    /// Creates a counter stream for `key` positioned at an explicit counter.
    ///
    /// This is useful for seeking directly to a particle index or frame without
    /// stepping through intervening blocks.
    #[must_use]
    pub fn with_counter(key: [u32; 2], ctr: [u32; 2]) -> Self {
        Self { ctr, key }
    }

    /// Returns the current counter without advancing the stream.
    #[must_use]
    pub fn counter(&self) -> [u32; 2] {
        self.ctr
    }

    /// Returns the key this stream was constructed with.
    #[must_use]
    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    /// Returns the output block for the current counter, then advances by one.
    ///
    /// The counter advances as a little-endian `64`-bit integer: the low word
    /// increments and, on wrap, carries into the high word.
    pub fn next_block(&mut self) -> [u32; 2] {
        let block = threefry2x32(self.ctr, self.key);
        self.ctr = increment_counter(self.ctr);
        block
    }
}

/// Advances a little-endian `64`-bit counter held as two `u32` words by one.
///
/// The low word increments with wrapping; if it wraps to zero the high word
/// takes the carry, also with wrapping, so the whole `64`-bit space cycles.
fn increment_counter(ctr: [u32; 2]) -> [u32; 2] {
    let low = ctr[0].wrapping_add(1);
    let high = if low == 0 {
        ctr[1].wrapping_add(1)
    } else {
        ctr[1]
    };
    [low, high]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // ---- Hard reference vectors (independently verified) ----

    #[test]
    fn reference_vector_all_zero() {
        assert!(threefry2x32([0, 0], [0, 0]) == [0x6b20_0159, 0x99ba_4efe]);
    }

    #[test]
    fn reference_vector_all_ones() {
        let out = threefry2x32([0xffff_ffff, 0xffff_ffff], [0xffff_ffff, 0xffff_ffff]);
        assert!(out == [0x1cb9_96fc, 0xbb00_2be7]);
    }

    #[test]
    fn reference_vector_pi_digits() {
        let out = threefry2x32([0x243f_6a88, 0x85a3_08d3], [0x1319_8a2e, 0x0370_7344]);
        assert!(out == [0xc492_3a9c, 0x483d_f7a0]);
    }

    // ---- Determinism ----

    #[test]
    fn deterministic_repeat_zero() {
        let a = threefry2x32([0, 0], [0, 0]);
        let b = threefry2x32([0, 0], [0, 0]);
        assert!(a == b);
    }

    #[test]
    fn deterministic_repeat_pi() {
        let ctr = [0x243f_6a88, 0x85a3_08d3];
        let key = [0x1319_8a2e, 0x0370_7344];
        assert!(threefry2x32(ctr, key) == threefry2x32(ctr, key));
    }

    #[test]
    fn deterministic_repeat_arbitrary() {
        let ctr = [0x1234_5678, 0x9abc_def0];
        let key = [0x0f0f_0f0f, 0xf0f0_f0f0];
        let first = threefry2x32(ctr, key);
        for _ in 0..16 {
            assert!(threefry2x32(ctr, key) == first);
        }
    }

    // ---- Sensitivity: different inputs give different outputs ----

    #[test]
    fn different_counter_low_differs() {
        assert!(threefry2x32([0, 0], [0, 0]) != threefry2x32([1, 0], [0, 0]));
    }

    #[test]
    fn different_counter_high_differs() {
        assert!(threefry2x32([0, 0], [0, 0]) != threefry2x32([0, 1], [0, 0]));
    }

    #[test]
    fn different_key_low_differs() {
        assert!(threefry2x32([0, 0], [0, 0]) != threefry2x32([0, 0], [1, 0]));
    }

    #[test]
    fn different_key_high_differs() {
        assert!(threefry2x32([0, 0], [0, 0]) != threefry2x32([0, 0], [0, 1]));
    }

    #[test]
    fn single_bit_counter_flip_differs() {
        let base = threefry2x32([0x8000_0000, 0], [0, 0]);
        let flip = threefry2x32([0x0000_0000, 0], [0, 0]);
        assert!(base != flip);
    }

    #[test]
    fn single_bit_key_flip_differs() {
        let base = threefry2x32([0, 0], [0x0000_0001, 0]);
        let flip = threefry2x32([0, 0], [0x0000_0003, 0]);
        assert!(base != flip);
    }

    #[test]
    fn many_counters_mostly_distinct() {
        let key = [0xdead_beef, 0x1234_5678];
        let mut seen: Vec<[u32; 2]> = Vec::new();
        for i in 0..256u32 {
            let out = threefry2x32([i, 0], key);
            assert!(!seen.contains(&out));
            seen.push(out);
        }
        assert!(seen.len() == 256);
    }

    #[test]
    fn many_keys_mostly_distinct() {
        let ctr = [0x0000_0007, 0x0000_0003];
        let mut seen: Vec<[u32; 2]> = Vec::new();
        for i in 0..256u32 {
            let out = threefry2x32(ctr, [i, 0]);
            assert!(!seen.contains(&out));
            seen.push(out);
        }
        assert!(seen.len() == 256);
    }

    // ---- Output is well-mixed (not trivially the input) ----

    #[test]
    fn output_not_equal_to_counter_zero() {
        assert!(threefry2x32([0, 0], [0, 0]) != [0, 0]);
    }

    #[test]
    fn output_not_equal_to_counter_arbitrary() {
        let ctr = [0x1111_2222, 0x3333_4444];
        assert!(threefry2x32(ctr, [0, 0]) != ctr);
    }

    #[test]
    fn both_words_can_be_nonzero() {
        let out = threefry2x32([0, 0], [0, 0]);
        assert!(out[0] != 0);
        assert!(out[1] != 0);
    }

    // ---- Stream wrapper: new ----

    #[test]
    fn new_starts_at_zero_counter() {
        let g = Threefry2x32::new([5, 9]);
        assert!(g.counter() == [0, 0]);
    }

    #[test]
    fn new_records_key() {
        let g = Threefry2x32::new([0xaaaa_aaaa, 0x5555_5555]);
        assert!(g.key() == [0xaaaa_aaaa, 0x5555_5555]);
    }

    #[test]
    fn first_block_matches_pure_function() {
        let key = [0x1319_8a2e, 0x0370_7344];
        let mut g = Threefry2x32::new(key);
        assert!(g.next_block() == threefry2x32([0, 0], key));
    }

    #[test]
    fn first_block_zero_key_is_reference() {
        let mut g = Threefry2x32::new([0, 0]);
        assert!(g.next_block() == [0x6b20_0159, 0x99ba_4efe]);
    }

    // ---- Stream wrapper: counter advance ----

    #[test]
    fn counter_advances_by_one() {
        let mut g = Threefry2x32::new([0, 0]);
        let _ = g.next_block();
        assert!(g.counter() == [1, 0]);
    }

    #[test]
    fn counter_advances_several() {
        let mut g = Threefry2x32::new([0, 0]);
        for _ in 0..10 {
            let _ = g.next_block();
        }
        assert!(g.counter() == [10, 0]);
    }

    #[test]
    fn stream_matches_manual_counter_sequence() {
        let key = [0x0f0f_0f0f, 0xf0f0_f0f0];
        let mut g = Threefry2x32::new(key);
        for i in 0..64u32 {
            let block = g.next_block();
            assert!(block == threefry2x32([i, 0], key));
        }
    }

    #[test]
    fn stream_blocks_are_distinct() {
        let mut g = Threefry2x32::new([0x1234_5678, 0]);
        let mut seen: Vec<[u32; 2]> = Vec::new();
        for _ in 0..128 {
            let block = g.next_block();
            assert!(!seen.contains(&block));
            seen.push(block);
        }
    }

    #[test]
    fn two_streams_same_key_agree() {
        let key = [0xcafe_babe, 0xfeed_face];
        let mut a = Threefry2x32::new(key);
        let mut b = Threefry2x32::new(key);
        for _ in 0..32 {
            assert!(a.next_block() == b.next_block());
        }
    }

    #[test]
    fn two_streams_different_key_diverge() {
        let mut a = Threefry2x32::new([1, 0]);
        let mut b = Threefry2x32::new([2, 0]);
        assert!(a.next_block() != b.next_block());
    }

    #[test]
    fn clone_branches_identical_stream() {
        let mut a = Threefry2x32::new([0x9e37_79b9, 0x7f4a_7c15]);
        let _ = a.next_block();
        let _ = a.next_block();
        let mut b = a.clone();
        for _ in 0..16 {
            assert!(a.next_block() == b.next_block());
        }
    }

    #[test]
    fn clone_is_equal_before_advance() {
        let a = Threefry2x32::new([7, 11]);
        let b = a.clone();
        assert!(a == b);
    }

    // ---- with_counter seeking ----

    #[test]
    fn with_counter_seeks_directly() {
        let key = [0x0bad_f00d, 0x0000_0001];
        let direct = Threefry2x32::with_counter(key, [100, 0]);
        assert!(direct.counter() == [100, 0]);
        let mut stepped = Threefry2x32::new(key);
        for _ in 0..100 {
            let _ = stepped.next_block();
        }
        assert!(stepped.counter() == direct.counter());
    }

    #[test]
    fn with_counter_block_matches_stepped() {
        let key = [0x0bad_f00d, 0x0000_0001];
        let mut direct = Threefry2x32::with_counter(key, [50, 0]);
        assert!(direct.next_block() == threefry2x32([50, 0], key));
    }

    // ---- Counter carry / wrap boundaries ----

    #[test]
    fn counter_carry_into_high_word() {
        let key = [0, 0];
        let mut g = Threefry2x32::with_counter(key, [0xffff_ffff, 0]);
        let _ = g.next_block();
        assert!(g.counter() == [0, 1]);
    }

    #[test]
    fn counter_no_carry_without_wrap() {
        let mut g = Threefry2x32::with_counter([0, 0], [0xffff_fffe, 7]);
        let _ = g.next_block();
        assert!(g.counter() == [0xffff_ffff, 7]);
    }

    #[test]
    fn counter_full_wraps_to_zero() {
        let mut g = Threefry2x32::with_counter([0, 0], [0xffff_ffff, 0xffff_ffff]);
        let _ = g.next_block();
        assert!(g.counter() == [0, 0]);
    }

    #[test]
    fn increment_counter_basic() {
        assert!(increment_counter([0, 0]) == [1, 0]);
    }

    #[test]
    fn increment_counter_carry() {
        assert!(increment_counter([0xffff_ffff, 0]) == [0, 1]);
    }

    #[test]
    fn increment_counter_high_carry() {
        assert!(increment_counter([0xffff_ffff, 0xffff_ffff]) == [0, 0]);
    }

    #[test]
    fn increment_counter_mid_high() {
        assert!(increment_counter([0xffff_ffff, 5]) == [0, 6]);
    }

    #[test]
    fn block_across_high_word_boundary_matches_pure() {
        let key = [0x5555_5555, 0xaaaa_aaaa];
        let mut g = Threefry2x32::with_counter(key, [0xffff_ffff, 0]);
        let before = g.next_block();
        assert!(before == threefry2x32([0xffff_ffff, 0], key));
        let after = g.next_block();
        assert!(after == threefry2x32([0, 1], key));
    }

    // ---- Key schedule parity behavior ----

    #[test]
    fn parity_makes_zero_key_nontrivial() {
        // With an all-zero key the third schedule word equals PARITY, so the
        // output must not collapse to the bare counter.
        assert!(threefry2x32([3, 4], [0, 0]) != [3, 4]);
    }

    #[test]
    fn swapping_key_words_changes_output() {
        let a = threefry2x32([0, 0], [0x1111_1111, 0x2222_2222]);
        let b = threefry2x32([0, 0], [0x2222_2222, 0x1111_1111]);
        assert!(a != b);
    }

    #[test]
    fn swapping_counter_words_changes_output() {
        let a = threefry2x32([0x1111_1111, 0x2222_2222], [0, 0]);
        let b = threefry2x32([0x2222_2222, 0x1111_1111], [0, 0]);
        assert!(a != b);
    }

    // ---- Avalanche-style sanity (not a statistical test, just mixing) ----

    #[test]
    fn adjacent_counters_differ_in_many_bits() {
        let a = threefry2x32([0, 0], [0, 0]);
        let b = threefry2x32([1, 0], [0, 0]);
        let diff = (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones();
        // A single counter-bit change should flip a healthy fraction of bits.
        assert!(diff >= 16);
    }

    #[test]
    fn adjacent_keys_differ_in_many_bits() {
        let a = threefry2x32([0, 0], [0, 0]);
        let b = threefry2x32([0, 0], [1, 0]);
        let diff = (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones();
        assert!(diff >= 16);
    }

    #[test]
    fn sum_of_blocks_is_order_independent_check() {
        // Collect a handful of blocks and confirm the stream is reproducible by
        // summing with wrapping arithmetic across two independent runs.
        let key = [0x89ab_cdef, 0x0123_4567];
        let run = |n: u32| -> (u32, u32) {
            let mut g = Threefry2x32::new(key);
            let mut acc = (0u32, 0u32);
            for _ in 0..n {
                let b = g.next_block();
                acc.0 = acc.0.wrapping_add(b[0]);
                acc.1 = acc.1.wrapping_add(b[1]);
            }
            acc
        };
        assert!(run(64) == run(64));
    }

    #[test]
    fn blocks_collect_via_iterator() {
        let key = [0x0001_0002, 0x0003_0004];
        let mut g = Threefry2x32::new(key);
        let collected: Vec<[u32; 2]> = (0..8).map(|_| g.next_block()).collect();
        let expected: Vec<[u32; 2]> = (0..8u32).map(|i| threefry2x32([i, 0], key)).collect();
        assert!(collected == expected);
    }
}
