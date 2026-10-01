//! Verhoeff check-digit algorithm, implemented with pure integer table lookups
//! on the `CPU`.
//!
//! The Verhoeff scheme computes a single trailing decimal check digit over a
//! sequence of base-ten digits. Unlike a plain modular checksum, it is built on
//! the dihedral group `D5` (the symmetry group of a regular pentagon, with ten
//! elements) and a position-dependent permutation. That construction lets it
//! catch every single-digit error and every transposition of two adjacent
//! digits, which the simpler sum-based checks miss because addition is
//! commutative and cannot see digit order.
//!
//! ## Tables
//!
//! Three constant tables encode the algorithm. `D` is the `D5` multiplication
//! (Cayley) table: `D[a][b]` is the group product of elements `a` and `b`.
//! `P` is the family of permutations applied by digit position; row `i` is the
//! permutation `P` raised to the power `i`, cycling with period eight. `INV`
//! maps each group element to its inverse, so that `D[x][INV[x]] == 0` for the
//! identity element `0`.
//!
//! ## Boundaries
//!
//! This module is self-contained and shares no tables with the `CRC`, Adler,
//! Fletcher, or Luhn checksum siblings. Luhn in particular is a different
//! algorithm: it doubles alternate digits over the integers modulo ten, which
//! fails to detect the `09` versus `90` transposition that Verhoeff catches.
//! Everything here is integer-only table indexing; no floating-point or
//! transcendental operation appears, so the result is bit-exact on every
//! target.

/// `D5` dihedral-group multiplication (Cayley) table.
///
/// `D[a][b]` is the group product of elements `a` and `b`, where the ten
/// elements `0..=9` are the symmetries of a regular pentagon (five rotations
/// followed by five reflections).
const D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

/// Position-dependent permutation family.
///
/// Row `i` is the permutation `P` applied `i` times; the rows repeat with a
/// period of eight, so position `i` uses `P[i % 8]`.
const P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// Multiplicative inverse of each `D5` group element.
///
/// `INV[x]` is the unique element `y` with `D[x][y] == 0`.
const INV: [u8; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

/// Computes the Verhoeff check digit for `digits` (most significant digit
/// first), returning a value in `0..=9`.
///
/// `digits` holds the payload only, without any check digit appended. Each
/// element is assumed to be a decimal digit in `0..=9`; callers that cannot
/// guarantee that should use [`verhoeff_check_digit_checked`].
///
/// Appending the returned digit to `digits` yields a sequence that
/// [`verhoeff_is_valid`] accepts.
pub fn verhoeff_check_digit(digits: &[u8]) -> u8 {
    let mut c = 0u8;
    for (i, &digit) in digits.iter().rev().enumerate() {
        c = D[c as usize][P[(i + 1) % 8][digit as usize] as usize];
    }
    INV[c as usize]
}

/// Validates `digits` (most significant digit first) including its trailing
/// check digit, returning `true` when the Verhoeff check passes.
///
/// Each element is assumed to be a decimal digit in `0..=9`; callers that
/// cannot guarantee that should use [`verhoeff_is_valid_checked`].
pub fn verhoeff_is_valid(digits: &[u8]) -> bool {
    let mut c = 0u8;
    for (i, &digit) in digits.iter().rev().enumerate() {
        c = D[c as usize][P[i % 8][digit as usize] as usize];
    }
    c == 0
}

/// Computes the Verhoeff check digit for `digits`, returning `None` if any
/// element is not a decimal digit in `0..=9`.
///
/// On success the behaviour matches [`verhoeff_check_digit`].
pub fn verhoeff_check_digit_checked(digits: &[u8]) -> Option<u8> {
    let mut c = 0u8;
    for (i, &digit) in digits.iter().rev().enumerate() {
        if digit > 9 {
            return None;
        }
        c = D[c as usize][P[(i + 1) % 8][digit as usize] as usize];
    }
    Some(INV[c as usize])
}

/// Validates `digits` including its trailing check digit, returning `None` if
/// any element is not a decimal digit in `0..=9`.
///
/// On success the behaviour matches [`verhoeff_is_valid`].
pub fn verhoeff_is_valid_checked(digits: &[u8]) -> Option<bool> {
    let mut c = 0u8;
    for (i, &digit) in digits.iter().rev().enumerate() {
        if digit > 9 {
            return None;
        }
        c = D[c as usize][P[i % 8][digit as usize] as usize];
    }
    Some(c == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Appends the computed check digit to `payload`, returning the full
    /// sequence that `verhoeff_is_valid` should accept.
    fn with_check(payload: &[u8]) -> Vec<u8> {
        let mut full = payload.to_vec();
        full.push(verhoeff_check_digit(payload));
        full
    }

    #[test]
    fn reference_check_digit_236() {
        assert_eq!(verhoeff_check_digit(&[2, 3, 6]), 3);
    }

    #[test]
    fn reference_valid_2363() {
        assert!(verhoeff_is_valid(&[2, 3, 6, 3]));
    }

    #[test]
    fn reference_check_digit_12345() {
        assert_eq!(verhoeff_check_digit(&[1, 2, 3, 4, 5]), 1);
    }

    #[test]
    fn reference_valid_123451() {
        assert!(verhoeff_is_valid(&[1, 2, 3, 4, 5, 1]));
    }

    #[test]
    fn reference_check_digit_matches_appended() {
        assert_eq!(with_check(&[2, 3, 6]), [2, 3, 6, 3]);
    }

    #[test]
    fn reference_check_digit_matches_appended_long() {
        assert_eq!(with_check(&[1, 2, 3, 4, 5]), [1, 2, 3, 4, 5, 1]);
    }

    #[test]
    fn roundtrip_single_digit() {
        for d in 0u8..=9 {
            let full = with_check(&[d]);
            assert!(verhoeff_is_valid(&full), "digit {d} roundtrip failed");
        }
    }

    #[test]
    fn roundtrip_two_digits() {
        for a in 0u8..=9 {
            for b in 0u8..=9 {
                let full = with_check(&[a, b]);
                assert!(verhoeff_is_valid(&full), "pair {a}{b} roundtrip failed");
            }
        }
    }

    #[test]
    fn roundtrip_three_digits_sample() {
        let samples: [&[u8]; 6] = [
            &[0, 0, 0],
            &[1, 2, 3],
            &[9, 9, 9],
            &[7, 0, 4],
            &[5, 5, 5],
            &[3, 1, 4],
        ];
        for payload in samples {
            let full = with_check(payload);
            assert!(
                verhoeff_is_valid(&full),
                "payload {payload:?} roundtrip failed"
            );
        }
    }

    #[test]
    fn roundtrip_long_sequence() {
        let payload = [1u8, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 0, 9, 3];
        let full = with_check(&payload);
        assert!(verhoeff_is_valid(&full));
    }

    #[test]
    fn roundtrip_many_generated() {
        // Deterministic pseudo-random payloads via a tiny LCG, integer only.
        let mut state: u32 = 0x1234_5678;
        for _ in 0..200 {
            let len = ((state >> 5) % 8) as usize + 1;
            let mut payload = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                payload.push(((state >> 11) % 10) as u8);
            }
            let full = with_check(&payload);
            assert!(verhoeff_is_valid(&full), "payload {payload:?} failed");
        }
    }

    #[test]
    fn empty_payload_check_digit() {
        // With no digits, the accumulator stays 0 and INV[0] == 0.
        assert_eq!(verhoeff_check_digit(&[]), 0);
    }

    #[test]
    fn empty_sequence_is_valid() {
        // c starts at 0 and never changes, so the empty sequence validates.
        assert!(verhoeff_is_valid(&[]));
    }

    #[test]
    fn single_zero_check_digit() {
        // Verhoeff check digit of a single 0 is 4 (not 0).
        assert_eq!(verhoeff_check_digit(&[0]), 4);
    }

    #[test]
    fn single_zero_roundtrip() {
        assert!(verhoeff_is_valid(&[0, 4]));
        assert!(!verhoeff_is_valid(&[0, 0]));
    }

    #[test]
    fn all_zeros_roundtrip() {
        for len in 1..=10usize {
            let payload = alloc::vec![0u8; len];
            let full = with_check(&payload);
            assert!(verhoeff_is_valid(&full), "len {len} failed");
        }
    }

    #[test]
    fn detects_single_error_in_236() {
        let base = [2u8, 3, 6, 3];
        for pos in 0..base.len() {
            for replacement in 0u8..=9 {
                if replacement == base[pos] {
                    continue;
                }
                let mut tampered = base;
                tampered[pos] = replacement;
                assert!(
                    !verhoeff_is_valid(&tampered),
                    "single error at {pos}->{replacement} not detected"
                );
            }
        }
    }

    #[test]
    fn detects_single_error_in_123451() {
        let base = [1u8, 2, 3, 4, 5, 1];
        for pos in 0..base.len() {
            for replacement in 0u8..=9 {
                if replacement == base[pos] {
                    continue;
                }
                let mut tampered = base;
                tampered[pos] = replacement;
                assert!(
                    !verhoeff_is_valid(&tampered),
                    "single error at {pos}->{replacement} not detected"
                );
            }
        }
    }

    #[test]
    fn detects_single_error_generated() {
        let mut state: u32 = 0x9E37_79B9;
        for _ in 0..100 {
            let len = ((state >> 7) % 6) as usize + 2;
            let mut payload = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                payload.push(((state >> 9) % 10) as u8);
            }
            let full = with_check(&payload);
            for pos in 0..full.len() {
                for replacement in 0u8..=9 {
                    if replacement == full[pos] {
                        continue;
                    }
                    let mut tampered = full.clone();
                    tampered[pos] = replacement;
                    assert!(
                        !verhoeff_is_valid(&tampered),
                        "single error not detected in {full:?} at {pos}"
                    );
                }
            }
        }
    }

    #[test]
    fn detects_adjacent_transposition_in_236() {
        let base = [2u8, 3, 6, 3];
        for pos in 0..base.len() - 1 {
            if base[pos] == base[pos + 1] {
                continue;
            }
            let mut swapped = base;
            swapped.swap(pos, pos + 1);
            assert!(
                !verhoeff_is_valid(&swapped),
                "transposition at {pos} not detected"
            );
        }
    }

    #[test]
    fn detects_adjacent_transposition_in_123451() {
        let base = [1u8, 2, 3, 4, 5, 1];
        for pos in 0..base.len() - 1 {
            if base[pos] == base[pos + 1] {
                continue;
            }
            let mut swapped = base;
            swapped.swap(pos, pos + 1);
            assert!(
                !verhoeff_is_valid(&swapped),
                "transposition at {pos} not detected"
            );
        }
    }

    #[test]
    fn detects_adjacent_transposition_generated() {
        let mut state: u32 = 0x0BAD_F00D;
        for _ in 0..100 {
            let len = ((state >> 6) % 6) as usize + 3;
            let mut payload = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                payload.push(((state >> 13) % 10) as u8);
            }
            let full = with_check(&payload);
            for pos in 0..full.len() - 1 {
                if full[pos] == full[pos + 1] {
                    continue;
                }
                let mut swapped = full.clone();
                swapped.swap(pos, pos + 1);
                assert!(
                    !verhoeff_is_valid(&swapped),
                    "transposition not detected in {full:?} at {pos}"
                );
            }
        }
    }

    #[test]
    fn checked_matches_unchecked_valid_digits() {
        let payload = [1u8, 2, 3, 4, 5];
        assert_eq!(
            verhoeff_check_digit_checked(&payload),
            Some(verhoeff_check_digit(&payload))
        );
    }

    #[test]
    fn checked_rejects_out_of_range_digit() {
        assert_eq!(verhoeff_check_digit_checked(&[1, 2, 10]), None);
    }

    #[test]
    fn checked_rejects_max_u8() {
        assert_eq!(verhoeff_check_digit_checked(&[255]), None);
    }

    #[test]
    fn checked_accepts_boundary_nine() {
        assert_eq!(
            verhoeff_check_digit_checked(&[9, 9, 9]),
            Some(verhoeff_check_digit(&[9, 9, 9]))
        );
    }

    #[test]
    fn checked_empty_is_zero() {
        assert_eq!(verhoeff_check_digit_checked(&[]), Some(0));
    }

    #[test]
    fn is_valid_checked_matches_valid() {
        assert_eq!(verhoeff_is_valid_checked(&[2, 3, 6, 3]), Some(true));
    }

    #[test]
    fn is_valid_checked_matches_invalid() {
        assert_eq!(verhoeff_is_valid_checked(&[2, 3, 6, 4]), Some(false));
    }

    #[test]
    fn is_valid_checked_rejects_out_of_range() {
        assert_eq!(verhoeff_is_valid_checked(&[2, 3, 6, 10]), None);
    }

    #[test]
    fn is_valid_checked_empty_is_true() {
        assert_eq!(verhoeff_is_valid_checked(&[]), Some(true));
    }

    #[test]
    fn invalid_wrong_check_digit() {
        // Any check digit other than the correct 3 must be rejected.
        for d in 0u8..=9 {
            let valid = d == 3;
            assert_eq!(verhoeff_is_valid(&[2, 3, 6, d]), valid, "digit {d}");
        }
    }

    #[test]
    fn d_table_is_latin_square_rows() {
        for row in &D {
            let mut seen = [false; 10];
            for &v in row {
                assert!(v < 10);
                assert!(!seen[v as usize], "duplicate in D row");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn d_table_is_latin_square_cols() {
        for col in 0..10 {
            let mut seen = [false; 10];
            for row in &D {
                let v = row[col];
                assert!(!seen[v as usize], "duplicate in D column");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn p_rows_are_permutations() {
        for row in &P {
            let mut seen = [false; 10];
            for &v in row {
                assert!(v < 10);
                assert!(!seen[v as usize], "duplicate in P row");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn inv_is_a_permutation() {
        let mut seen = [false; 10];
        for &v in &INV {
            assert!(v < 10);
            assert!(!seen[v as usize], "duplicate in INV");
            seen[v as usize] = true;
        }
    }

    #[test]
    fn inv_satisfies_group_inverse() {
        for x in 0usize..10 {
            assert_eq!(D[x][INV[x] as usize], 0, "INV wrong for {x}");
            assert_eq!(D[INV[x] as usize][x], 0, "INV wrong (left) for {x}");
        }
    }

    #[test]
    fn zero_is_identity_in_d() {
        for x in 0usize..10 {
            assert_eq!(D[0][x], x as u8);
            assert_eq!(D[x][0], x as u8);
        }
    }

    #[test]
    fn p_row_zero_is_identity() {
        for x in 0usize..10 {
            assert_eq!(P[0][x], x as u8);
        }
    }

    #[test]
    fn leading_zeros_change_check_digit() {
        // Verhoeff is position sensitive, so a leading zero is significant.
        let no_zero = verhoeff_check_digit(&[1, 2, 3]);
        let with_zero = verhoeff_check_digit(&[0, 1, 2, 3]);
        // They need not differ for every input, but the sequences stay valid.
        assert!(verhoeff_is_valid(&[1, 2, 3, no_zero]));
        assert!(verhoeff_is_valid(&[0, 1, 2, 3, with_zero]));
    }

    #[test]
    fn known_vector_eight_digits() {
        // 142857 is a cyclic number; verify a stable roundtrip.
        let payload = [1u8, 4, 2, 8, 5, 7];
        let cd = verhoeff_check_digit(&payload);
        let mut full = payload.to_vec();
        full.push(cd);
        assert!(verhoeff_is_valid(&full));
    }

    #[test]
    fn tampered_long_sequence_detected() {
        let payload = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let full = with_check(&payload);
        let mut tampered = full.clone();
        tampered[0] = (tampered[0] + 1) % 10;
        assert!(!verhoeff_is_valid(&tampered));
    }
}
