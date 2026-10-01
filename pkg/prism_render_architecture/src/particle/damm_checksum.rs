//! Damm check-digit algorithm, implemented with pure integer table lookups on
//! the `CPU`.
//!
//! The Damm scheme appends a single trailing decimal check digit to a sequence
//! of base-ten digits. It is built on a totally anti-symmetric quasigroup of
//! order ten: a `10x10` operation table whose structure guarantees detection of
//! every single-digit error and every transposition of two adjacent digits.
//! Unlike a plain modular sum, the quasigroup operation is non-commutative, so
//! it can see digit order and catch the `09` versus `90` swap that commutative
//! checks miss.
//!
//! ## Table
//!
//! A single constant [`struct@TABLE`] encodes the quasigroup. `TABLE[i][d]` is
//! the quasigroup product of the running interim value `i` and the next digit
//! `d`. The chosen table is *weak totally anti-symmetric*: its main diagonal is
//! entirely zero (`TABLE[i][i] == 0`), which is exactly the property the Damm
//! construction needs so that the final interim value is zero precisely when
//! the trailing digit is a valid check digit.
//!
//! ## Algorithm
//!
//! Start from an interim value of zero. For each input digit `d`, replace the
//! interim value with `TABLE[interim][d]`. After consuming every digit the
//! interim value is the check digit. To validate, run the same fold over the
//! payload *including* its trailing check digit; the sequence is valid when the
//! final interim value is zero.
//!
//! ## Boundaries
//!
//! This module is self-contained and shares no tables with the Verhoeff,
//! `CRC`, Adler, Fletcher, or Luhn checksum siblings. Verhoeff uses the
//! dihedral group `D5` with position-dependent permutations; Damm needs no
//! permutation family and no inverse table, only this one quasigroup. Luhn is a
//! different scheme again, doubling alternate digits modulo ten. Everything
//! here is integer-only table indexing; no floating-point or transcendental
//! operation appears, so the result is bit-exact on every target.

/// Weak totally anti-symmetric quasigroup operation table of order ten.
///
/// `TABLE[i][d]` is the quasigroup product used by the Damm fold: it maps the
/// current interim value `i` and the next digit `d` to the next interim value.
/// The main diagonal is all zeros (`TABLE[i][i] == 0`), the structural property
/// that makes the construction a valid check-digit scheme.
const TABLE: [[u8; 10]; 10] = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
    [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
    [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
    [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
    [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
    [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
    [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
];

/// Computes the Damm check digit for a sequence of decimal digits.
///
/// Each element of `digits` must be in the range `0..=9`; the payload must not
/// already contain a check digit. The returned value is the digit that, when
/// appended, makes the full sequence valid under [`damm_is_valid`]. An empty
/// slice folds to the initial interim value and yields `0`.
///
/// # Panics
///
/// Panics if any element exceeds `9`, because the value is used directly as a
/// column index into [`struct@TABLE`]. Use [`damm_check_digit_checked`] to
/// reject out-of-range input without panicking.
#[must_use]
pub fn damm_check_digit(digits: &[u8]) -> u8 {
    let mut interim = 0usize;
    for &d in digits {
        assert!(d <= 9, "Damm digit out of range 0..=9");
        interim = TABLE[interim][d as usize] as usize;
    }
    interim as u8
}

/// Validates a sequence whose final element is a Damm check digit.
///
/// `digits` is the payload *with* its trailing check digit. Every element must
/// be in `0..=9`. The sequence is valid when the fold over all digits ends at
/// an interim value of zero. An empty slice is treated as valid, matching the
/// zero initial interim value.
///
/// # Panics
///
/// Panics if any element exceeds `9`. Use [`damm_is_valid_checked`] to reject
/// out-of-range input without panicking.
#[must_use]
pub fn damm_is_valid(digits: &[u8]) -> bool {
    damm_check_digit(digits) == 0
}

/// Computes the Damm check digit, returning [`None`] on out-of-range input.
///
/// Behaves like [`damm_check_digit`] but yields [`None`] as soon as any element
/// exceeds `9`, rather than panicking. An empty slice returns `Some(0)`.
#[must_use]
pub fn damm_check_digit_checked(digits: &[u8]) -> Option<u8> {
    let mut interim = 0usize;
    for &d in digits {
        if d > 9 {
            return None;
        }
        interim = TABLE[interim][d as usize] as usize;
    }
    Some(interim as u8)
}

/// Validates a check-digit-bearing sequence, returning [`None`] on bad input.
///
/// Behaves like [`damm_is_valid`] but yields [`None`] as soon as any element
/// exceeds `9`, rather than panicking. An empty slice returns `Some(true)`.
#[must_use]
pub fn damm_is_valid_checked(digits: &[u8]) -> Option<bool> {
    damm_check_digit_checked(digits).map(|interim| interim == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Appends the computed check digit to a payload, returning the full
    /// sequence ready for validation.
    fn with_check(payload: &[u8]) -> Vec<u8> {
        let mut full = payload.to_vec();
        full.push(damm_check_digit(payload));
        full
    }

    #[test]
    fn reference_vector_572_check_digit() {
        assert_eq!(damm_check_digit(&[5, 7, 2]), 4);
    }

    #[test]
    fn reference_vector_5724_is_valid() {
        assert!(damm_is_valid(&[5, 7, 2, 4]));
    }

    #[test]
    fn empty_payload_check_digit_is_zero() {
        assert_eq!(damm_check_digit(&[]), 0);
    }

    #[test]
    fn empty_sequence_is_valid() {
        assert!(damm_is_valid(&[]));
    }

    #[test]
    fn wrong_check_digit_is_invalid() {
        assert!(!damm_is_valid(&[5, 7, 2, 3]));
    }

    #[test]
    fn every_wrong_trailing_digit_except_four_is_invalid() {
        for cd in 0u8..10 {
            let valid = damm_is_valid(&[5, 7, 2, cd]);
            assert_eq!(valid, cd == 4);
        }
    }

    #[test]
    fn roundtrip_appends_make_valid() {
        let payloads: [&[u8]; 6] = [
            &[0],
            &[9],
            &[1, 2, 3],
            &[5, 7, 2],
            &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            &[4, 4, 4, 4, 4],
        ];
        for payload in payloads {
            let full = with_check(payload);
            assert!(damm_is_valid(&full));
        }
    }

    #[test]
    fn roundtrip_exhaustive_three_digit_payloads() {
        for a in 0u8..10 {
            for b in 0u8..10 {
                for c in 0u8..10 {
                    let full = with_check(&[a, b, c]);
                    assert!(damm_is_valid(&full));
                }
            }
        }
    }

    #[test]
    fn single_digit_errors_detected_three_digit() {
        // Damm detects every single-digit substitution.
        for a in 0u8..10 {
            for b in 0u8..10 {
                for c in 0u8..10 {
                    let full = with_check(&[a, b, c]);
                    for pos in 0..full.len() {
                        for wrong in 0u8..10 {
                            if wrong == full[pos] {
                                continue;
                            }
                            let mut tampered = full.clone();
                            tampered[pos] = wrong;
                            assert!(
                                !damm_is_valid(&tampered),
                                "single error at {pos} undetected: {tampered:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn adjacent_transpositions_detected_three_digit() {
        // Damm detects every transposition of two adjacent digits.
        for a in 0u8..10 {
            for b in 0u8..10 {
                for c in 0u8..10 {
                    let full = with_check(&[a, b, c]);
                    for pos in 0..full.len() - 1 {
                        if full[pos] == full[pos + 1] {
                            continue;
                        }
                        let mut swapped = full.clone();
                        swapped.swap(pos, pos + 1);
                        assert!(
                            !damm_is_valid(&swapped),
                            "transposition at {pos} undetected: {swapped:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn adjacent_transpositions_detected_longer() {
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let full = with_check(&payload);
        for pos in 0..full.len() - 1 {
            if full[pos] == full[pos + 1] {
                continue;
            }
            let mut swapped = full.clone();
            swapped.swap(pos, pos + 1);
            assert!(!damm_is_valid(&swapped));
        }
    }

    #[test]
    fn single_digit_zero_payload() {
        // A lone zero digit folds to TABLE[0][0] == 0.
        assert_eq!(damm_check_digit(&[0]), 0);
        assert!(damm_is_valid(&[0, 0]));
    }

    #[test]
    fn all_zeros_payload_valid_roundtrip() {
        for len in 0usize..12 {
            let payload = alloc::vec![0u8; len];
            let full = with_check(&payload);
            assert!(damm_is_valid(&full));
        }
    }

    #[test]
    fn all_zeros_check_digit_is_zero() {
        for len in 0usize..12 {
            let payload = alloc::vec![0u8; len];
            assert_eq!(damm_check_digit(&payload), 0);
        }
    }

    #[test]
    fn single_digit_payload_each_value() {
        for d in 0u8..10 {
            let full = with_check(&[d]);
            assert!(damm_is_valid(&full));
        }
    }

    #[test]
    fn table_diagonal_is_all_zero() {
        // Weak totally anti-symmetric property: TABLE[i][i] == 0.
        for (i, row) in TABLE.iter().enumerate() {
            assert_eq!(row[i], 0, "diagonal nonzero at {i}");
        }
    }

    #[test]
    fn table_off_diagonal_is_never_zero() {
        for (i, row) in TABLE.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if i != j {
                    assert_ne!(v, 0, "unexpected zero off diagonal at [{i}][{j}]");
                }
            }
        }
    }

    #[test]
    fn table_rows_are_permutations() {
        for (i, row) in TABLE.iter().enumerate() {
            let mut seen = [false; 10];
            for &v in row {
                assert!(v <= 9, "value out of range in row {i}");
                assert!(!seen[v as usize], "row {i} not a permutation");
                seen[v as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "row {i} missing a value");
        }
    }

    #[test]
    fn table_columns_are_permutations() {
        for col in 0usize..10 {
            let mut seen = [false; 10];
            for row in &TABLE {
                let v = row[col];
                assert!(!seen[v as usize], "column {col} not a permutation");
                seen[v as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "column {col} missing a value");
        }
    }

    #[test]
    fn table_is_quasigroup_latin_square() {
        // A quasigroup table is exactly a Latin square: every value appears
        // exactly once in every row and every column. Combined check.
        for v in 0u8..10 {
            for (i, row) in TABLE.iter().enumerate() {
                let count = row.iter().filter(|&&x| x == v).count();
                assert_eq!(count, 1, "value {v} appears {count} times in row {i}");
            }
            for col in 0usize..10 {
                let count = TABLE.iter().filter(|row| row[col] == v).count();
                assert_eq!(count, 1, "value {v} appears {count} times in col {col}");
            }
        }
    }

    #[test]
    fn checked_variant_matches_unchecked_in_range() {
        let payloads: [&[u8]; 5] = [
            &[],
            &[5, 7, 2],
            &[0, 0, 0],
            &[9, 9, 9, 9],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ];
        for payload in payloads {
            assert_eq!(
                damm_check_digit_checked(payload),
                Some(damm_check_digit(payload))
            );
        }
    }

    #[test]
    fn checked_valid_matches_unchecked_in_range() {
        assert_eq!(damm_is_valid_checked(&[5, 7, 2, 4]), Some(true));
        assert_eq!(damm_is_valid_checked(&[5, 7, 2, 3]), Some(false));
        assert_eq!(damm_is_valid_checked(&[]), Some(true));
    }

    #[test]
    fn checked_check_digit_rejects_out_of_range() {
        assert_eq!(damm_check_digit_checked(&[10]), None);
        assert_eq!(damm_check_digit_checked(&[5, 7, 42]), None);
        assert_eq!(damm_check_digit_checked(&[255]), None);
    }

    #[test]
    fn checked_valid_rejects_out_of_range() {
        assert_eq!(damm_is_valid_checked(&[10]), None);
        assert_eq!(damm_is_valid_checked(&[5, 7, 2, 100]), None);
    }

    #[test]
    fn checked_rejects_at_first_bad_element() {
        // The first out-of-range element short-circuits even if later ones are
        // fine.
        assert_eq!(damm_check_digit_checked(&[11, 5, 7, 2]), None);
    }

    #[test]
    fn checked_check_digit_empty_is_some_zero() {
        assert_eq!(damm_check_digit_checked(&[]), Some(0));
    }

    #[test]
    fn checked_boundary_nine_is_accepted() {
        assert_eq!(damm_check_digit_checked(&[9]), Some(damm_check_digit(&[9])));
        assert!(damm_is_valid_checked(&[9, 9]).is_some());
    }

    #[test]
    fn fold_matches_manual_steps_572() {
        // interim: 0 -(5)-> TABLE[0][5]=9 -(7)-> TABLE[9][7]=7 -(2)-> TABLE[7][2]=4
        assert_eq!(TABLE[0][5], 9);
        assert_eq!(TABLE[9][7], 7);
        assert_eq!(TABLE[7][2], 4);
        assert_eq!(damm_check_digit(&[5, 7, 2]), 4);
    }

    #[test]
    fn leading_zeros_are_significant() {
        // Prepending a zero changes the fold, so check digits may differ.
        let a = damm_check_digit(&[1, 2, 3]);
        let b = damm_check_digit(&[0, 1, 2, 3]);
        assert!(damm_is_valid(&[1, 2, 3, a]));
        assert!(damm_is_valid(&[0, 1, 2, 3, b]));
    }

    #[test]
    fn long_sequence_roundtrip() {
        let payload = [3u8, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9];
        let full = with_check(&payload);
        assert!(damm_is_valid(&full));
    }

    #[test]
    fn long_sequence_single_error_detected() {
        let payload = [3u8, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9];
        let full = with_check(&payload);
        let mut tampered = full.clone();
        tampered[4] = (tampered[4] + 1) % 10;
        assert!(!damm_is_valid(&tampered));
    }

    #[test]
    fn check_digit_in_range_for_many_inputs() {
        for a in 0u8..10 {
            for b in 0u8..10 {
                let cd = damm_check_digit(&[a, b]);
                assert!(cd <= 9);
            }
        }
    }

    #[test]
    fn appending_check_digit_twice_is_invalid_unless_zero() {
        // Appending the check digit to an already-valid sequence re-opens the
        // fold; it is valid again only if that extra digit is zero.
        let full = with_check(&[5, 7, 2]);
        let cd2 = damm_check_digit(&full);
        assert_eq!(cd2, 0, "valid sequence must produce a zero check digit");
        assert!(damm_is_valid(&full));
    }

    #[test]
    fn valid_sequence_has_zero_terminal_check_digit() {
        // Any valid sequence folds to zero, so its own check digit is zero.
        let full = with_check(&[1, 2, 3, 4, 5]);
        assert_eq!(damm_check_digit(&full), 0);
    }

    #[test]
    fn distinct_payloads_can_share_check_digit() {
        // Not injective: different payloads may share a check digit, but each
        // still roundtrips.
        let x = with_check(&[1, 2]);
        let y = with_check(&[3, 4]);
        assert!(damm_is_valid(&x));
        assert!(damm_is_valid(&y));
    }

    #[test]
    fn full_table_contents_match_reference() {
        // Guard against accidental edits to the quasigroup table.
        const REFERENCE: [[u8; 10]; 10] = [
            [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
            [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
            [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
            [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
            [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
            [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
            [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
            [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
            [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
            [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
        ];
        assert_eq!(TABLE, REFERENCE);
    }

    #[test]
    fn two_digit_payloads_single_substitution_detected() {
        for a in 0u8..10 {
            for b in 0u8..10 {
                let full = with_check(&[a, b]);
                for pos in 0..full.len() {
                    for wrong in 0u8..10 {
                        if wrong == full[pos] {
                            continue;
                        }
                        let mut t = full.clone();
                        t[pos] = wrong;
                        assert!(!damm_is_valid(&t));
                    }
                }
            }
        }
    }

    #[test]
    fn interim_zero_only_at_valid_terminus() {
        // For payload [5,7,2], only appending 4 produces interim 0.
        let zeros: Vec<u8> = (0u8..10)
            .filter(|&cd| damm_check_digit(&[5, 7, 2, cd]) == 0)
            .collect();
        assert_eq!(zeros, alloc::vec![4]);
    }

    #[test]
    fn checked_and_unchecked_valid_agree_in_range() {
        let full = with_check(&[7, 7, 7]);
        assert_eq!(damm_is_valid_checked(&full), Some(damm_is_valid(&full)));
    }

    #[test]
    fn boundary_nine_payload_roundtrip() {
        let full = with_check(&[9, 9, 9, 9, 9]);
        assert!(damm_is_valid(&full));
    }

    #[test]
    fn check_digit_is_deterministic() {
        let a = damm_check_digit(&[6, 2, 8, 3, 1]);
        let b = damm_check_digit(&[6, 2, 8, 3, 1]);
        assert_eq!(a, b);
    }

    #[test]
    fn checked_long_out_of_range_tail() {
        let mut v = alloc::vec![1u8, 2, 3, 4, 5];
        v.push(99);
        assert_eq!(damm_check_digit_checked(&v), None);
        assert_eq!(damm_is_valid_checked(&v), None);
    }
}
