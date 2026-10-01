//! Reflected binary `Gray`-code transcoding: the closed-form integer transform
//! that renumbers a counter so that *consecutive* values differ in exactly one
//! bit (design §27).
//!
//! A `Gray` code (reflected binary code) is a bijection over the whole
//! `u32`/`u64` range with the defining adjacency property: for every `n`, the
//! codes of `n` and `n + 1` differ in a single bit position. That single-bit
//! adjacency is exactly what glitch-free counters, rotary/optical encoders,
//! `Karnaugh`-map ordering, band-reflected structured-light patterns, and
//! error-tolerant frame indices rely on, so it recurs throughout a `GPU`
//! renderer: quantized sort keys that must not thrash under a `+/-1` jitter,
//! progressive sample indices, and hierarchical tile identifiers all benefit
//! from the property that a small numeric step is a small *bit* step.
//!
//! Two closed-form, exact transforms compose here:
//!
//! * **Encode** (`binary -> Gray`) is the single operation `n ^ (n >> 1)`. The
//!   most-significant bit is copied verbatim and every lower bit is xored with
//!   its higher neighbour, which is what folds ("reflects") the counter so the
//!   `LSB` toggles only on alternate steps instead of every step.
//! * **Decode** (`Gray -> binary`) inverts that fold with an in-place xor
//!   prefix scan: `b ^= b >> 1; b ^= b >> 2; b ^= b >> 4; ...`, doubling the
//!   shift each round until it exceeds the word width (five rounds for `u32`,
//!   six for `u64`). The scan reconstructs each binary bit as the parity of all
//!   `Gray` bits at or above it.
//!
//! Sequencing helpers ([`next_gray_u32`], [`next_gray_u64`] and their `prev`
//! counterparts) walk the code in numeric order by decoding to binary, taking a
//! wrapping `+/-1` step, and re-encoding; the wrapping step makes the sequence a
//! closed cycle so stepping past `u32::MAX` returns to `binary_to_gray(0) == 0`.
//! [`gray_diff_bit_index`] reports which bit two codes disagree on when — and
//! only when — they are adjacent (differ in a single bit), using the built-in
//! [`u32::count_ones`] and [`u32::trailing_zeros`] population/scan intrinsics.
//!
//! Every routine here is integer-only — shifts and xor, plus `wrapping_add` /
//! `wrapping_sub` for the sequence step — with no `f32` and no transcendental
//! calls, so this reference is bit-reproducible against a future `GPU` kernel.
//!
//! ## Boundaries
//!
//! This module owns *only* the reversible `Gray`-code renumbering. It is
//! deliberately distinct from its integer-transform neighbours:
//!
//! * [`super::zigzag_delta_encode`] remaps signs and differences neighbours; it
//!   does not reflect a counter into single-bit-adjacent codes.
//! * [`super::bit_pack_u32`] concatenates bit fields; it neither reflects nor
//!   inverts them.
//!
//! Nothing here is lossy and every transform is total over its whole domain,
//! including the `0`, `u32::MAX`, and `u64::MAX` boundaries.

/// Encode a [`u32`] counter into its reflected-binary `Gray` code via
/// `n ^ (n >> 1)`.
///
/// The most-significant set bit is preserved and each lower bit is xored with
/// its higher neighbour. The result is such that `binary_to_gray_u32(n)` and
/// `binary_to_gray_u32(n + 1)` differ in exactly one bit; see
/// [`gray_to_binary_u32`] for the inverse.
#[must_use]
#[inline]
pub const fn binary_to_gray_u32(n: u32) -> u32 {
    n ^ (n >> 1)
}

/// Encode a [`u64`] counter into its reflected-binary `Gray` code; the 64-bit
/// analogue of [`binary_to_gray_u32`], using `n ^ (n >> 1)`.
#[must_use]
#[inline]
pub const fn binary_to_gray_u64(n: u64) -> u64 {
    n ^ (n >> 1)
}

/// Decode a reflected-binary `Gray` code back to the [`u32`] counter it
/// represents: the exact inverse of [`binary_to_gray_u32`].
///
/// Implemented as an xor prefix scan with doubling shifts
/// (`b ^= b >> 1; b ^= b >> 2; b ^= b >> 4; b ^= b >> 8; b ^= b >> 16;`), which
/// rebuilds each binary bit as the parity of all `Gray` bits at or above it.
#[must_use]
#[inline]
pub const fn gray_to_binary_u32(gray: u32) -> u32 {
    let mut b = gray;
    b ^= b >> 1;
    b ^= b >> 2;
    b ^= b >> 4;
    b ^= b >> 8;
    b ^= b >> 16;
    b
}

/// Decode a reflected-binary `Gray` code back to the [`u64`] counter it
/// represents: the exact inverse of [`binary_to_gray_u64`].
///
/// The 64-bit scan runs one extra doubling round (`b ^= b >> 32;`) beyond the
/// `u32` variant so the parity fold covers all 64 bits.
#[must_use]
#[inline]
pub const fn gray_to_binary_u64(gray: u64) -> u64 {
    let mut b = gray;
    b ^= b >> 1;
    b ^= b >> 2;
    b ^= b >> 4;
    b ^= b >> 8;
    b ^= b >> 16;
    b ^= b >> 32;
    b
}

/// Advance one step along the `u32` `Gray`-code cycle: decode `current_gray` to
/// binary, take a wrapping `+ 1` step, and re-encode.
///
/// The wrapping step closes the cycle, so the successor of
/// `binary_to_gray_u32(u32::MAX)` is `binary_to_gray_u32(0) == 0`. The returned
/// code differs from `current_gray` in exactly one bit.
#[must_use]
#[inline]
pub const fn next_gray_u32(current_gray: u32) -> u32 {
    binary_to_gray_u32(gray_to_binary_u32(current_gray).wrapping_add(1))
}

/// Advance one step along the `u64` `Gray`-code cycle; the 64-bit analogue of
/// [`next_gray_u32`].
#[must_use]
#[inline]
pub const fn next_gray_u64(current_gray: u64) -> u64 {
    binary_to_gray_u64(gray_to_binary_u64(current_gray).wrapping_add(1))
}

/// Step one place *backwards* along the `u32` `Gray`-code cycle: the exact
/// inverse of [`next_gray_u32`], using a wrapping `- 1` step.
///
/// The predecessor of `binary_to_gray_u32(0) == 0` wraps to
/// `binary_to_gray_u32(u32::MAX)`.
#[must_use]
#[inline]
pub const fn prev_gray_u32(current_gray: u32) -> u32 {
    binary_to_gray_u32(gray_to_binary_u32(current_gray).wrapping_sub(1))
}

/// Step one place *backwards* along the `u64` `Gray`-code cycle; the 64-bit
/// analogue of [`prev_gray_u32`].
#[must_use]
#[inline]
pub const fn prev_gray_u64(current_gray: u64) -> u64 {
    binary_to_gray_u64(gray_to_binary_u64(current_gray).wrapping_sub(1))
}

/// Report which bit two `u32` `Gray` codes disagree on, but only when they are
/// *adjacent* (differ in a single bit).
///
/// Returns `Some(index)` — the zero-based bit index counted from the `LSB`,
/// obtained via [`u32::trailing_zeros`] — when `a_gray ^ b_gray` has exactly one
/// set bit (checked with [`u32::count_ones`]). Returns `None` when the codes are
/// identical (zero differing bits) or non-adjacent (two or more differing
/// bits). Because sequence neighbours always differ in one bit, this yields the
/// toggled bit for any `next`/`prev` step and cleanly rejects everything else.
#[must_use]
#[inline]
pub const fn gray_diff_bit_index(a_gray: u32, b_gray: u32) -> Option<u32> {
    let diff = a_gray ^ b_gray;
    if diff.count_ones() == 1 {
        Some(diff.trailing_zeros())
    } else {
        None
    }
}

/// The 64-bit analogue of [`gray_diff_bit_index`]: report the single toggled bit
/// index (from the `LSB`) when two `u64` `Gray` codes are adjacent, else `None`.
#[must_use]
#[inline]
pub const fn gray_diff_bit_index_u64(a_gray: u64, b_gray: u64) -> Option<u32> {
    let diff = a_gray ^ b_gray;
    if diff.count_ones() == 1 {
        Some(diff.trailing_zeros())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic linear-congruential generator so the randomized
    /// round-trips are reproducible bit for bit across runs and platforms.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    // --- Known small values -----------------------------------------------

    #[test]
    fn binary_to_gray_u32_known_table() {
        assert_eq!(binary_to_gray_u32(0), 0);
        assert_eq!(binary_to_gray_u32(1), 1);
        assert_eq!(binary_to_gray_u32(2), 3);
        assert_eq!(binary_to_gray_u32(3), 2);
        assert_eq!(binary_to_gray_u32(4), 6);
        assert_eq!(binary_to_gray_u32(5), 7);
        assert_eq!(binary_to_gray_u32(6), 5);
        assert_eq!(binary_to_gray_u32(7), 4);
    }

    #[test]
    fn gray_to_binary_u32_known_table() {
        assert_eq!(gray_to_binary_u32(0), 0);
        assert_eq!(gray_to_binary_u32(1), 1);
        assert_eq!(gray_to_binary_u32(3), 2);
        assert_eq!(gray_to_binary_u32(2), 3);
        assert_eq!(gray_to_binary_u32(6), 4);
        assert_eq!(gray_to_binary_u32(7), 5);
        assert_eq!(gray_to_binary_u32(5), 6);
        assert_eq!(gray_to_binary_u32(4), 7);
    }

    #[test]
    fn binary_to_gray_u64_matches_u32_on_low_range() {
        for n in 0..2000_u32 {
            assert_eq!(
                u64::from(binary_to_gray_u32(n)),
                binary_to_gray_u64(u64::from(n))
            );
        }
    }

    // --- Round-trips -------------------------------------------------------

    #[test]
    fn roundtrip_u32_dense_low_range() {
        for n in 0..=1000_u32 {
            assert_eq!(gray_to_binary_u32(binary_to_gray_u32(n)), n);
        }
    }

    #[test]
    fn roundtrip_u64_dense_low_range() {
        for n in 0..=1000_u64 {
            assert_eq!(gray_to_binary_u64(binary_to_gray_u64(n)), n);
        }
    }

    #[test]
    fn roundtrip_u32_selected_large_values() {
        let values = [
            0x0000_FFFF,
            0x0001_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xAAAA_AAAA,
            0x5555_5555,
            0xDEAD_BEEF,
            u32::MAX,
        ];
        for &n in &values {
            assert_eq!(gray_to_binary_u32(binary_to_gray_u32(n)), n);
        }
    }

    #[test]
    fn roundtrip_u64_selected_large_values() {
        let values = [
            0x0000_0000_FFFF_FFFF,
            0x0000_0001_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0xDEAD_BEEF_CAFE_BABE,
            u64::MAX,
        ];
        for &n in &values {
            assert_eq!(gray_to_binary_u64(binary_to_gray_u64(n)), n);
        }
    }

    #[test]
    fn roundtrip_u32_random_lcg() {
        let mut state = 0x1234_5678_9ABC_DEF0;
        for _ in 0..10_000 {
            let n = (lcg_next(&mut state) >> 32) as u32;
            assert_eq!(gray_to_binary_u32(binary_to_gray_u32(n)), n);
            // And the encode direction inverts the decode too.
            assert_eq!(binary_to_gray_u32(gray_to_binary_u32(n)), n);
        }
    }

    #[test]
    fn roundtrip_u64_random_lcg() {
        let mut state = 0xDEAD_BEEF_CAFE_BABE;
        for _ in 0..10_000 {
            let n = lcg_next(&mut state);
            assert_eq!(gray_to_binary_u64(binary_to_gray_u64(n)), n);
            assert_eq!(binary_to_gray_u64(gray_to_binary_u64(n)), n);
        }
    }

    // --- Single-bit adjacency ---------------------------------------------

    #[test]
    fn consecutive_codes_differ_in_one_bit_u32() {
        for n in 0..5000_u32 {
            let g0 = binary_to_gray_u32(n);
            let g1 = binary_to_gray_u32(n + 1);
            assert_eq!((g0 ^ g1).count_ones(), 1, "n = {n}");
        }
    }

    #[test]
    fn consecutive_codes_differ_in_one_bit_u64() {
        for n in 0..5000_u64 {
            let g0 = binary_to_gray_u64(n);
            let g1 = binary_to_gray_u64(n + 1);
            assert_eq!((g0 ^ g1).count_ones(), 1, "n = {n}");
        }
    }

    #[test]
    fn adjacency_holds_across_power_of_two_boundaries() {
        for k in 0..31_u32 {
            let boundary = 1_u32 << k;
            // n = 2^k - 1 -> 2^k straddles a carry-heavy binary transition.
            let g0 = binary_to_gray_u32(boundary - 1);
            let g1 = binary_to_gray_u32(boundary);
            assert_eq!((g0 ^ g1).count_ones(), 1, "boundary = {boundary}");
        }
    }

    #[test]
    fn adjacency_at_u32_max_wrap_is_one_bit() {
        // The cycle wraps u32::MAX -> 0; those two codes are also neighbours.
        let g_max = binary_to_gray_u32(u32::MAX);
        let g_zero = binary_to_gray_u32(0);
        assert_eq!((g_max ^ g_zero).count_ones(), 1);
    }

    // --- next_gray / prev_gray sequencing ---------------------------------

    #[test]
    fn next_gray_u32_matches_encode_of_successor() {
        for n in 0..5000_u32 {
            assert_eq!(
                next_gray_u32(binary_to_gray_u32(n)),
                binary_to_gray_u32(n + 1)
            );
        }
    }

    #[test]
    fn next_gray_u32_from_zero_is_one() {
        assert_eq!(next_gray_u32(0), 1);
        assert_eq!(next_gray_u32(1), 3);
        assert_eq!(next_gray_u32(3), 2);
        assert_eq!(next_gray_u32(2), 6);
    }

    #[test]
    fn next_gray_u32_wraps_at_max_to_zero() {
        let g_max = binary_to_gray_u32(u32::MAX);
        assert_eq!(next_gray_u32(g_max), 0);
    }

    #[test]
    fn next_gray_u64_matches_encode_of_successor() {
        for n in 0..5000_u64 {
            assert_eq!(
                next_gray_u64(binary_to_gray_u64(n)),
                binary_to_gray_u64(n + 1)
            );
        }
    }

    #[test]
    fn next_gray_u64_wraps_at_max_to_zero() {
        let g_max = binary_to_gray_u64(u64::MAX);
        assert_eq!(next_gray_u64(g_max), 0);
    }

    #[test]
    fn prev_gray_is_inverse_of_next_gray_u32() {
        let mut state = 0x00C0_FFEE_00C0_FFEE;
        for _ in 0..10_000 {
            let g = (lcg_next(&mut state) >> 32) as u32;
            assert_eq!(prev_gray_u32(next_gray_u32(g)), g);
            assert_eq!(next_gray_u32(prev_gray_u32(g)), g);
        }
    }

    #[test]
    fn prev_gray_is_inverse_of_next_gray_u64() {
        let mut state = 0x0F0F_0F0F_1234_5678;
        for _ in 0..10_000 {
            let g = lcg_next(&mut state);
            assert_eq!(prev_gray_u64(next_gray_u64(g)), g);
            assert_eq!(next_gray_u64(prev_gray_u64(g)), g);
        }
    }

    #[test]
    fn prev_gray_u32_wraps_at_zero_to_max_code() {
        assert_eq!(prev_gray_u32(0), binary_to_gray_u32(u32::MAX));
    }

    #[test]
    fn next_gray_walk_is_single_bit_each_step() {
        // Walk a window of the cycle and confirm every hop toggles one bit and
        // stays consistent with gray_diff_bit_index.
        let mut g = binary_to_gray_u32(0);
        for _ in 0..4096 {
            let next = next_gray_u32(g);
            assert_eq!((g ^ next).count_ones(), 1);
            assert!(gray_diff_bit_index(g, next).is_some());
            g = next;
        }
    }

    // --- gray_diff_bit_index ----------------------------------------------

    #[test]
    fn gray_diff_bit_index_adjacent_returns_toggled_bit() {
        for n in 0..2000_u32 {
            let g0 = binary_to_gray_u32(n);
            let g1 = binary_to_gray_u32(n + 1);
            let expected = (g0 ^ g1).trailing_zeros();
            assert_eq!(gray_diff_bit_index(g0, g1), Some(expected));
        }
    }

    #[test]
    fn gray_diff_bit_index_identical_returns_none() {
        assert_eq!(gray_diff_bit_index(0, 0), None);
        assert_eq!(gray_diff_bit_index(0xDEAD_BEEF, 0xDEAD_BEEF), None);
    }

    #[test]
    fn gray_diff_bit_index_non_adjacent_returns_none() {
        // Two set bits differ -> not adjacent.
        assert_eq!(gray_diff_bit_index(0b0000, 0b0011), None);
        // Codes two steps apart in the cycle differ in two bits.
        let g0 = binary_to_gray_u32(10);
        let g2 = binary_to_gray_u32(12);
        assert!((g0 ^ g2).count_ones() >= 2);
        assert_eq!(gray_diff_bit_index(g0, g2), None);
    }

    #[test]
    fn gray_diff_bit_index_specific_positions() {
        // Differ only in bit 0, 5, and 31 respectively.
        assert_eq!(gray_diff_bit_index(0, 1 << 0), Some(0));
        assert_eq!(gray_diff_bit_index(0, 1 << 5), Some(5));
        assert_eq!(gray_diff_bit_index(0, 1 << 31), Some(31));
    }

    #[test]
    fn gray_diff_bit_index_u64_adjacent_and_rejects() {
        let g0 = binary_to_gray_u64(1_000_000);
        let g1 = binary_to_gray_u64(1_000_001);
        let expected = (g0 ^ g1).trailing_zeros();
        assert_eq!(gray_diff_bit_index_u64(g0, g1), Some(expected));
        assert_eq!(gray_diff_bit_index_u64(g0, g0), None);
        assert_eq!(gray_diff_bit_index_u64(0, 0b101), None);
        assert_eq!(gray_diff_bit_index_u64(0, 1 << 63), Some(63));
    }

    // --- Boundary and structural properties -------------------------------

    #[test]
    fn boundary_zero_and_max_u32() {
        assert_eq!(binary_to_gray_u32(0), 0);
        assert_eq!(gray_to_binary_u32(0), 0);
        // u32::MAX = 0xFFFF_FFFF -> gray = 0x8000_0000 (only the top bit set).
        assert_eq!(binary_to_gray_u32(u32::MAX), 0x8000_0000);
        assert_eq!(gray_to_binary_u32(0x8000_0000), u32::MAX);
    }

    #[test]
    fn boundary_zero_and_max_u64() {
        assert_eq!(binary_to_gray_u64(0), 0);
        assert_eq!(gray_to_binary_u64(0), 0);
        assert_eq!(binary_to_gray_u64(u64::MAX), 0x8000_0000_0000_0000);
        assert_eq!(gray_to_binary_u64(0x8000_0000_0000_0000), u64::MAX);
    }

    #[test]
    fn gray_code_preserves_most_significant_bit_u32() {
        // The top bit of the Gray code equals the top bit of the binary value
        // because the fold only xors a bit with a *higher* neighbour.
        let mut state = 0xABCD_1234_5678_9876;
        for _ in 0..5000 {
            let n = (lcg_next(&mut state) >> 32) as u32;
            assert_eq!(binary_to_gray_u32(n) >> 31, n >> 31);
        }
    }

    #[test]
    fn encoding_is_injective_over_small_range() {
        // Distinct inputs give distinct codes; verified by an exhaustive
        // decode over a contiguous block that must reproduce every input.
        for n in 0..20_000_u32 {
            let decoded = gray_to_binary_u32(binary_to_gray_u32(n));
            assert_eq!(decoded, n);
        }
    }

    #[test]
    fn const_evaluation_is_available() {
        // Exercise the const-fn path so a regression to non-const is caught.
        const G5: u32 = binary_to_gray_u32(5);
        const B: u32 = gray_to_binary_u32(G5);
        const NEXT: u32 = next_gray_u32(G5);
        const DIFF: Option<u32> = gray_diff_bit_index(G5, NEXT);
        assert_eq!(G5, 7);
        assert_eq!(B, 5);
        assert_eq!(NEXT, binary_to_gray_u32(6));
        assert!(DIFF.is_some());
    }
}
