//! Pure-integer bit-reversal primitives for the particle engine's spectral core.
//!
//! Reversing the bit order of an integer is the arithmetic heart of the
//! bit-reversal permutation that an in-place radix-2 `FFT` applies before (or
//! after) its butterfly passes. This module supplies only that permutation
//! algebra: it reverses bits, reverses the low `bits` of a word, walks a
//! bit-reversed counter, tests bit-reversal palindromes, and materializes a
//! permutation table. It deliberately does **not** implement the `FFT` itself,
//! nor any `log2`/`next_pow2`/trailing-zero scan: those live in the sibling
//! `de Bruijn` module, and duplicating them here would be redundant.
//!
//! # The divide-and-conquer reversal
//!
//! Every fixed-width reversal below is the classic mask-and-swap network. We
//! swap adjacent single bits with the `0x5555...` mask, then adjacent bit
//! pairs with `0x3333...`, then nibbles with `0x0f0f...`, and finally swap
//! whole bytes with the standard library's byte-swap (`swap_bytes`, a pure
//! permutation of bytes, not an arithmetic op). The result is bit-for-bit
//! identical on a scalar `CPU` core, a `GPU` integer pipeline, and inside a
//! `const` context. We intentionally avoid the `u32::reverse_bits` intrinsic in
//! the implementation so the exact arithmetic is documented and portable; the
//! tests cross-check every width against that intrinsic.
//!
//! # Bit-reversed traversal
//!
//! A radix-2 `FFT` reads or writes its data in bit-reversed index order. Two
//! helpers support that: `reverse_lowest_bits` reverses just the low `bits`
//! lanes of an index (the full permutation for a `2^bits`-point transform),
//! and `bit_reverse_increment` advances a counter in bit-reversed order by
//! carrying from the most-significant lane downward. Starting at zero and
//! applying the increment `2^bits - 1` times visits every index exactly once in
//! bit-reversed order, which is why the `LSB`/`MSB` carry direction is flipped
//! relative to an ordinary `+1`.

extern crate alloc;
use alloc::vec::Vec;

/// Reverses the bit order of a `u8` using the classic divide-and-conquer
/// mask-and-swap network (swap adjacent bits, then pairs, then nibbles).
///
/// This is a pure permutation of the eight bit lanes: `reverse_bits_u8(0b0000_0001)`
/// is `0b1000_0000`. It performs no arithmetic beyond shifts and masks.
#[must_use]
pub const fn reverse_bits_u8(x: u8) -> u8 {
    let mut v = x;
    v = ((v & 0x55) << 1) | ((v >> 1) & 0x55);
    v = ((v & 0x33) << 2) | ((v >> 2) & 0x33);
    v = ((v & 0x0f) << 4) | ((v >> 4) & 0x0f);
    v
}

/// Reverses the bit order of a `u16` with the mask-and-swap network, finishing
/// with a byte swap so the two halves trade places.
///
/// Pure bit permutation: no transcendental math and no arithmetic add/sub.
#[must_use]
pub const fn reverse_bits_u16(x: u16) -> u16 {
    let mut v = x;
    v = ((v & 0x5555) << 1) | ((v >> 1) & 0x5555);
    v = ((v & 0x3333) << 2) | ((v >> 2) & 0x3333);
    v = ((v & 0x0f0f) << 4) | ((v >> 4) & 0x0f0f);
    v.swap_bytes()
}

/// Reverses the bit order of a `u32` with the mask-and-swap network, finishing
/// with a byte swap.
///
/// This is the workhorse for the `FFT` bit-reversal permutation on 32-bit
/// indices. It is a `const fn` built only from shifts, masks, and a byte
/// permutation, so it evaluates identically at compile time and at run time.
#[must_use]
pub const fn reverse_bits_u32(x: u32) -> u32 {
    let mut v = x;
    v = ((v & 0x5555_5555) << 1) | ((v >> 1) & 0x5555_5555);
    v = ((v & 0x3333_3333) << 2) | ((v >> 2) & 0x3333_3333);
    v = ((v & 0x0f0f_0f0f) << 4) | ((v >> 4) & 0x0f0f_0f0f);
    v.swap_bytes()
}

/// Reverses the bit order of a `u64` with the mask-and-swap network, finishing
/// with a byte swap.
///
/// Pure bit permutation over all 64 lanes, usable in a `const` context.
#[must_use]
pub const fn reverse_bits_u64(x: u64) -> u64 {
    let mut v = x;
    v = ((v & 0x5555_5555_5555_5555) << 1) | ((v >> 1) & 0x5555_5555_5555_5555);
    v = ((v & 0x3333_3333_3333_3333) << 2) | ((v >> 2) & 0x3333_3333_3333_3333);
    v = ((v & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((v >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    v.swap_bytes()
}

/// Reverses only the lowest `bits` lanes of `x`, clearing everything above.
///
/// This is the index map of a `2^bits`-point `FFT` bit-reversal permutation:
/// an index `i` in `0..2^bits` is sent to the value formed by reading its low
/// `bits` lanes back to front. For example with `bits == 3` the index `0b001`
/// maps to `0b100`. `bits` must lie in `0..=32`; `bits == 0` yields `0` and the
/// high bits of the input are always discarded. This helper only describes the
/// permutation and performs no transform of its own.
#[must_use]
pub const fn reverse_lowest_bits(x: u32, bits: u32) -> u32 {
    debug_assert!(bits <= 32);
    if bits == 0 {
        return 0;
    }
    let full = reverse_bits_u32(x);
    full >> (32 - bits)
}

/// Advances `index` by one step of a bit-reversed counter of width `bits`.
///
/// An ordinary `+1` carries from the least-significant lane upward; a
/// bit-reversed counter carries from the most-significant lane downward. We
/// therefore start a probe mask at `1 << (bits - 1)` and walk it toward the
/// `LSB`, flipping lanes until one flips from `0` to `1`. Starting from `0` and
/// applying this `2^bits - 1` times enumerates every index in bit-reversed
/// order, which the `FFT` scheduler uses to stream butterflies without a
/// precomputed table. `bits` must lie in `1..=32`; the returned value stays
/// within the low `bits` lanes.
#[must_use]
pub fn bit_reverse_increment(index: u32, bits: u32) -> u32 {
    debug_assert!((1..=32).contains(&bits));
    let mut idx = index;
    let mut mask = 1u32 << (bits - 1);
    loop {
        idx ^= mask;
        if (idx & mask) != 0 {
            break;
        }
        if mask == 1 {
            break;
        }
        mask >>= 1;
    }
    idx
}

/// Returns whether the low `bits` lanes of `x` are symmetric under bit
/// reversal, i.e. reading those lanes forward equals reading them backward.
///
/// With `bits == 3` the pattern `0b101` is a palindrome while `0b100` is not.
/// `bits` must lie in `0..=32`; `bits == 0` is vacuously symmetric. Only the
/// low `bits` lanes participate; higher lanes are ignored.
#[must_use]
pub const fn is_bit_reversal_palindrome(x: u32, bits: u32) -> bool {
    debug_assert!(bits <= 32);
    if bits == 0 {
        return true;
    }
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let low = x & mask;
    reverse_lowest_bits(low, bits) == low
}

/// Builds the full bit-reversal permutation table for a `2^bits`-point
/// transform: entry `i` holds `reverse_lowest_bits(i, bits)`.
///
/// The table is a self-inverse (involutive) permutation of `0..2^bits`, so
/// applying it twice is the identity, which is why an in-place `FFT` can swap
/// each `i` with its partner once. `bits` is clamped by a `debug_assert` to
/// `0..=20` so the table (up to `2^20` entries) cannot blow up memory; a build
/// with assertions disabled still allocates whatever `bits` requests, so keep
/// the bound in mind.
#[must_use]
pub fn bit_reverse_permutation_table(bits: u32) -> Vec<u32> {
    debug_assert!(bits <= 20);
    let len = 1usize << bits;
    let mut table = Vec::with_capacity(len);
    for i in 0..len {
        table.push(reverse_lowest_bits(i as u32, bits));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_bits_u8_known_examples() {
        assert_eq!(reverse_bits_u8(0b0000_0001), 0b1000_0000);
        assert_eq!(reverse_bits_u8(0b0000_0010), 0b0100_0000);
        assert_eq!(reverse_bits_u8(0b1010_1010), 0b0101_0101);
        assert_eq!(reverse_bits_u8(0), 0);
        assert_eq!(reverse_bits_u8(u8::MAX), u8::MAX);
    }

    #[test]
    fn reverse_bits_u8_cross_check_intrinsic() {
        let mut x = 0u16;
        while x <= u8::MAX as u16 {
            let v = x as u8;
            assert_eq!(reverse_bits_u8(v), v.reverse_bits());
            x += 1;
        }
    }

    #[test]
    fn reverse_bits_u8_single_bit_lanes() {
        let mut k = 0u32;
        while k < 8 {
            let v = 1u8 << k;
            assert_eq!(reverse_bits_u8(v), 1u8 << (7 - k));
            k += 1;
        }
    }

    #[test]
    fn reverse_bits_u8_idempotent_double_reverse() {
        let mut x = 0u16;
        while x <= u8::MAX as u16 {
            let v = x as u8;
            assert_eq!(reverse_bits_u8(reverse_bits_u8(v)), v);
            x += 1;
        }
    }

    #[test]
    fn reverse_bits_u16_known_examples() {
        assert_eq!(reverse_bits_u16(0x0001), 0x8000);
        assert_eq!(reverse_bits_u16(0x8000), 0x0001);
        assert_eq!(reverse_bits_u16(0), 0);
        assert_eq!(reverse_bits_u16(u16::MAX), u16::MAX);
    }

    #[test]
    fn reverse_bits_u16_cross_check_lcg() {
        let mut state = 0x1234u32;
        let mut iter = 0u32;
        while iter < 5_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = state as u16;
            assert_eq!(reverse_bits_u16(v), v.reverse_bits());
            iter += 1;
        }
    }

    #[test]
    fn reverse_bits_u16_single_bit_lanes() {
        let mut k = 0u32;
        while k < 16 {
            let v = 1u16 << k;
            assert_eq!(reverse_bits_u16(v), 1u16 << (15 - k));
            k += 1;
        }
    }

    #[test]
    fn reverse_bits_u16_idempotent_double_reverse() {
        let mut state = 0xBEEFu32;
        let mut iter = 0u32;
        while iter < 5_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = state as u16;
            assert_eq!(reverse_bits_u16(reverse_bits_u16(v)), v);
            iter += 1;
        }
    }

    #[test]
    fn reverse_bits_u32_known_examples() {
        assert_eq!(reverse_bits_u32(0x0000_0001), 0x8000_0000);
        assert_eq!(reverse_bits_u32(0x8000_0000), 0x0000_0001);
        assert_eq!(reverse_bits_u32(0), 0);
        assert_eq!(reverse_bits_u32(u32::MAX), u32::MAX);
        assert_eq!(reverse_bits_u32(0x1234_5678), 0x1234_5678u32.reverse_bits());
    }

    #[test]
    fn reverse_bits_u32_cross_check_intrinsic_sequential() {
        let mut x = 0u32;
        while x <= 200_000 {
            assert_eq!(reverse_bits_u32(x), x.reverse_bits());
            x += 1;
        }
    }

    #[test]
    fn reverse_bits_u32_cross_check_lcg() {
        let mut state = 0x0BAD_F00Du32;
        let mut iter = 0u32;
        while iter < 20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            assert_eq!(reverse_bits_u32(state), state.reverse_bits());
            iter += 1;
        }
    }

    #[test]
    fn reverse_bits_u32_single_bit_lanes() {
        let mut k = 0u32;
        while k < 32 {
            let v = 1u32 << k;
            assert_eq!(reverse_bits_u32(v), 1u32 << (31 - k));
            k += 1;
        }
    }

    #[test]
    fn reverse_bits_u32_idempotent_double_reverse() {
        let mut state = 0xDEAD_BEEFu32;
        let mut iter = 0u32;
        while iter < 20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            assert_eq!(reverse_bits_u32(reverse_bits_u32(state)), state);
            iter += 1;
        }
    }

    #[test]
    fn reverse_bits_u64_known_examples() {
        assert_eq!(
            reverse_bits_u64(0x0000_0000_0000_0001),
            0x8000_0000_0000_0000
        );
        assert_eq!(
            reverse_bits_u64(0x8000_0000_0000_0000),
            0x0000_0000_0000_0001
        );
        assert_eq!(reverse_bits_u64(0), 0);
        assert_eq!(reverse_bits_u64(u64::MAX), u64::MAX);
    }

    #[test]
    fn reverse_bits_u64_cross_check_lcg() {
        let mut state = 0x0123_4567_89AB_CDEFu64;
        let mut iter = 0u32;
        while iter < 20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            assert_eq!(reverse_bits_u64(state), state.reverse_bits());
            iter += 1;
        }
    }

    #[test]
    fn reverse_bits_u64_single_bit_lanes() {
        let mut k = 0u32;
        while k < 64 {
            let v = 1u64 << k;
            assert_eq!(reverse_bits_u64(v), 1u64 << (63 - k));
            k += 1;
        }
    }

    #[test]
    fn reverse_bits_u64_idempotent_double_reverse() {
        let mut state = 0xFEED_FACE_CAFE_BEEFu64;
        let mut iter = 0u32;
        while iter < 20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            assert_eq!(reverse_bits_u64(reverse_bits_u64(state)), state);
            iter += 1;
        }
    }

    #[test]
    fn reverse_lowest_bits_known_small_cases() {
        assert_eq!(reverse_lowest_bits(0b001, 3), 0b100);
        assert_eq!(reverse_lowest_bits(0b100, 3), 0b001);
        assert_eq!(reverse_lowest_bits(0b010, 3), 0b010);
        assert_eq!(reverse_lowest_bits(0b011, 3), 0b110);
        assert_eq!(reverse_lowest_bits(0b110, 3), 0b011);
    }

    #[test]
    fn reverse_lowest_bits_zero_bits_is_zero() {
        assert_eq!(reverse_lowest_bits(0xFFFF_FFFF, 0), 0);
        assert_eq!(reverse_lowest_bits(0, 0), 0);
    }

    #[test]
    fn reverse_lowest_bits_clears_high_bits() {
        // Only the low 4 lanes should survive and reverse.
        assert_eq!(reverse_lowest_bits(0xFFFF_FFF1, 4), 0b1000);
        assert_eq!(reverse_lowest_bits(0xABCD_0001, 4), 0b1000);
    }

    #[test]
    fn reverse_lowest_bits_full_width_matches_reverse_bits() {
        let mut state = 0x0BAD_C0DEu32;
        let mut iter = 0u32;
        while iter < 10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            assert_eq!(reverse_lowest_bits(state, 32), reverse_bits_u32(state));
            iter += 1;
        }
    }

    #[test]
    fn reverse_lowest_bits_involution_within_width() {
        let bits = 5u32;
        let mut i = 0u32;
        while i < (1u32 << bits) {
            assert_eq!(reverse_lowest_bits(reverse_lowest_bits(i, bits), bits), i);
            i += 1;
        }
    }

    #[test]
    fn bit_reverse_increment_full_traversal_bits3() {
        let bits = 3u32;
        let mut idx = 0u32;
        let mut order = Vec::new();
        order.push(idx);
        let mut step = 0u32;
        while step < (1u32 << bits) - 1 {
            idx = bit_reverse_increment(idx, bits);
            order.push(idx);
            step += 1;
        }
        assert_eq!(order, alloc::vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn bit_reverse_increment_full_traversal_bits2() {
        let bits = 2u32;
        let mut idx = 0u32;
        let mut order = Vec::new();
        order.push(idx);
        let mut step = 0u32;
        while step < (1u32 << bits) - 1 {
            idx = bit_reverse_increment(idx, bits);
            order.push(idx);
            step += 1;
        }
        assert_eq!(order, alloc::vec![0, 2, 1, 3]);
    }

    #[test]
    fn bit_reverse_increment_wraps_to_zero() {
        let bits = 3u32;
        // Last element of the bit-reversed sequence is 7; incrementing wraps to 0.
        assert_eq!(bit_reverse_increment(7, bits), 0);
    }

    #[test]
    fn bit_reverse_increment_matches_reverse_of_natural_counter() {
        // Bit-reversed traversal equals reversing the natural counter order.
        let bits = 4u32;
        let mut idx = 0u32;
        let mut step = 0u32;
        while step < (1u32 << bits) {
            assert_eq!(idx, reverse_lowest_bits(step, bits));
            idx = bit_reverse_increment(idx, bits);
            step += 1;
        }
        // After the final increment we are back at zero.
        assert_eq!(idx, 0);
    }

    #[test]
    fn bit_reverse_increment_visits_each_index_once() {
        let bits = 6u32;
        let total = 1u32 << bits;
        let mut seen = alloc::vec![false; total as usize];
        let mut idx = 0u32;
        let mut step = 0u32;
        while step < total {
            assert!(!seen[idx as usize]);
            seen[idx as usize] = true;
            idx = bit_reverse_increment(idx, bits);
            step += 1;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn is_bit_reversal_palindrome_known_small_cases() {
        assert!(is_bit_reversal_palindrome(0b101, 3));
        assert!(is_bit_reversal_palindrome(0b000, 3));
        assert!(is_bit_reversal_palindrome(0b111, 3));
        assert!(is_bit_reversal_palindrome(0b010, 3));
        assert!(!is_bit_reversal_palindrome(0b100, 3));
        assert!(!is_bit_reversal_palindrome(0b001, 3));
    }

    #[test]
    fn is_bit_reversal_palindrome_ignores_high_bits() {
        // High bits above `bits` must not influence the verdict: only the low
        // `bits` lanes count. `0b101` (palindrome) and `0b100` (not) are
        // carried under a run of high ones and must still be judged by their
        // low three lanes alone.
        assert!(is_bit_reversal_palindrome(0xFFFF_F105, 3));
        assert!(!is_bit_reversal_palindrome(0xFFFF_F104, 3));
    }

    #[test]
    fn is_bit_reversal_palindrome_zero_width_vacuous() {
        assert!(is_bit_reversal_palindrome(0xDEAD_BEEF, 0));
    }

    #[test]
    fn is_bit_reversal_palindrome_full_width() {
        assert!(is_bit_reversal_palindrome(0, 32));
        assert!(is_bit_reversal_palindrome(u32::MAX, 32));
        assert!(is_bit_reversal_palindrome(0x8000_0001, 32));
        assert!(!is_bit_reversal_palindrome(0x8000_0000, 32));
    }

    #[test]
    fn permutation_table_known_bits3() {
        let table = bit_reverse_permutation_table(3);
        assert_eq!(table, alloc::vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn permutation_table_lengths() {
        let mut bits = 0u32;
        while bits <= 10 {
            let table = bit_reverse_permutation_table(bits);
            assert_eq!(table.len(), 1usize << bits);
            bits += 1;
        }
    }

    #[test]
    fn permutation_table_is_valid_permutation() {
        let bits = 8u32;
        let table = bit_reverse_permutation_table(bits);
        let len = 1usize << bits;
        let mut seen = alloc::vec![false; len];
        for &v in &table {
            assert!((v as usize) < len);
            assert!(!seen[v as usize]);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn permutation_table_is_involutive() {
        let bits = 8u32;
        let table = bit_reverse_permutation_table(bits);
        let mut i = 0usize;
        while i < table.len() {
            // Applying the permutation twice is the identity.
            assert_eq!(table[table[i] as usize] as usize, i);
            i += 1;
        }
    }

    #[test]
    fn permutation_table_bits0_is_single_identity() {
        let table = bit_reverse_permutation_table(0);
        assert_eq!(table, alloc::vec![0]);
    }

    #[test]
    fn permutation_table_matches_reverse_lowest_bits() {
        let bits = 7u32;
        let table = bit_reverse_permutation_table(bits);
        let mut i = 0u32;
        while i < (1u32 << bits) {
            assert_eq!(table[i as usize], reverse_lowest_bits(i, bits));
            i += 1;
        }
    }

    #[test]
    #[expect(
        clippy::assertions_on_constants,
        reason = "pins a compile-time-evaluated palindrome constant as a regression guard"
    )]
    fn const_evaluation_in_const_context() {
        const RB8: u8 = reverse_bits_u8(0b0000_0001);
        const RB16: u16 = reverse_bits_u16(0x0001);
        const RB32: u32 = reverse_bits_u32(0x0000_0001);
        const RB64: u64 = reverse_bits_u64(0x0000_0000_0000_0001);
        const RLOW: u32 = reverse_lowest_bits(0b001, 3);
        const PAL: bool = is_bit_reversal_palindrome(0b101, 3);
        assert_eq!(RB8, 0b1000_0000);
        assert_eq!(RB16, 0x8000);
        assert_eq!(RB32, 0x8000_0000);
        assert_eq!(RB64, 0x8000_0000_0000_0000);
        assert_eq!(RLOW, 0b100);
        assert!(PAL);
    }
}
