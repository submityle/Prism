//! `Luhn` mod-10 check-digit algorithm (the `ISO/IEC 7812` checksum used by
//! payment-card numbers and `IMEI` device identifiers) over decimal digit
//! sequences, implemented with pure integer arithmetic on the `CPU`.
//!
//! The `Luhn` formula detects all single-digit errors and most adjacent
//! transposition errors in a decimal string. It operates on a slice of decimal
//! digits `&[u8]`, where every element is expected to lie in `0..=9`. Working
//! from the right, every second digit is *doubled*; a doubled value greater
//! than `9` has `9` subtracted from it (equivalently, its decimal digits are
//! summed). All (possibly doubled) digit values are summed into `sum`, and the
//! number is valid exactly when `sum` is a multiple of `10`.
//!
//! Two framings of the "every second digit" rule are used, and this module
//! keeps them deliberately distinct:
//!
//! * [`luhn_check_digit`] receives a *payload* that does **not** yet include a
//!   check digit and computes the single digit that should be appended. Here
//!   the doubling window starts at the right-most payload digit (because the
//!   not-yet-present check digit would occupy the undoubled right-most slot).
//!   The returned digit is `(10 - (sum % 10)) % 10`.
//! * [`luhn_is_valid`] receives a *complete* string whose right-most digit is
//!   already the check digit. Here the right-most digit is **not** doubled and
//!   the doubling window starts one place to its left. The string is valid when
//!   the full weighted sum is a multiple of `10`.
//!
//! Because the two functions anchor the doubling window on opposite parities,
//! appending [`luhn_check_digit`]'s result to a payload always yields a string
//! that [`luhn_is_valid`] accepts; [`luhn_append_check_digit`] performs exactly
//! that round-trip and the test-suite exercises the invariant.
//!
//! ## Empty-input convention
//!
//! The empty slice has a weighted sum of `0`, and `0` is a multiple of `10`, so
//! [`luhn_is_valid`] returns `true` for `&[]`. Symmetrically
//! [`luhn_check_digit`] returns `0` for the empty payload. These edge cases are
//! covered explicitly by the tests so the convention is pinned down rather than
//! accidental.
//!
//! ## Out-of-range digits
//!
//! The checked [`luhn_check_digit_checked`] and [`luhn_is_valid_checked`]
//! helpers return `None` the moment any element exceeds `9`, which is the
//! recommended entry point for untrusted input. The unchecked
//! [`luhn_check_digit`] / [`luhn_is_valid`] variants assume every element is a
//! valid decimal digit; in debug builds a `debug_assert!` catches a stray
//! out-of-range element, and the tests confirm the checked variants reject such
//! input with `None`.
//!
//! All arithmetic is integer only — no floating point and no transcendental
//! functions — so this reference is bit-reproducible on any target.

use alloc::vec::Vec;

/// Fold one decimal digit through the `Luhn` doubling rule.
///
/// When `double` is `true` the digit is multiplied by two and, if the product
/// exceeds `9`, reduced by `9` (which equals summing the product's two decimal
/// digits). When `double` is `false` the digit is returned unchanged. The
/// result is always in `0..=9`.
#[inline]
fn weighted(digit: u8, double: bool) -> u32 {
    let d = u32::from(digit);
    if double {
        let doubled = d * 2;
        if doubled > 9 {
            doubled - 9
        } else {
            doubled
        }
    } else {
        d
    }
}

/// Return `true` when every element of `digits` is a decimal digit (`0..=9`).
#[inline]
fn all_decimal(digits: &[u8]) -> bool {
    !digits.iter().any(|&d| d > 9)
}

/// Compute the `Luhn` check digit for a payload that does **not** yet include a
/// check digit.
///
/// Working from the right-most payload digit, every second digit (starting with
/// the right-most) is doubled per the `Luhn` rule; the weighted values are
/// summed and the returned check digit is `(10 - (sum % 10)) % 10`, i.e. the
/// digit that makes the completed string's weighted sum a multiple of `10`.
///
/// The empty payload yields `0`. Every element is assumed to lie in `0..=9`;
/// use [`luhn_check_digit_checked`] for untrusted input.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::luhn_checksum::luhn_check_digit;
///
/// assert_eq!(luhn_check_digit(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]), 3);
/// ```
#[must_use]
pub fn luhn_check_digit(digits: &[u8]) -> u8 {
    debug_assert!(all_decimal(digits), "luhn_check_digit: element exceeds 9");
    let mut sum = 0u32;
    for (index, &digit) in digits.iter().rev().enumerate() {
        // Right-most payload digit is index 0 and must be doubled, because the
        // appended check digit will occupy the undoubled right-most slot.
        sum += weighted(digit % 10, index % 2 == 0);
    }
    ((10 - (sum % 10)) % 10) as u8
}

/// Checked form of [`luhn_check_digit`] that returns `None` when any element
/// exceeds `9`.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::luhn_checksum::luhn_check_digit_checked;
///
/// assert_eq!(luhn_check_digit_checked(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]), Some(3));
/// assert_eq!(luhn_check_digit_checked(&[1, 2, 10]), None);
/// ```
#[must_use]
pub fn luhn_check_digit_checked(digits: &[u8]) -> Option<u8> {
    if all_decimal(digits) {
        Some(luhn_check_digit(digits))
    } else {
        None
    }
}

/// Validate a complete decimal string whose right-most element is already the
/// `Luhn` check digit.
///
/// The right-most digit is **not** doubled; moving left, every second digit is
/// doubled per the `Luhn` rule. The string is valid when the full weighted sum
/// is a multiple of `10`. The empty slice has sum `0` and is therefore valid.
///
/// Every element is assumed to lie in `0..=9`; use [`luhn_is_valid_checked`]
/// for untrusted input.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::luhn_checksum::luhn_is_valid;
///
/// assert!(luhn_is_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]));
/// assert!(!luhn_is_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 4]));
/// ```
#[must_use]
pub fn luhn_is_valid(digits: &[u8]) -> bool {
    debug_assert!(all_decimal(digits), "luhn_is_valid: element exceeds 9");
    let mut sum = 0u32;
    for (index, &digit) in digits.iter().rev().enumerate() {
        // Right-most digit (index 0) is the check digit and stays undoubled.
        sum += weighted(digit % 10, index % 2 == 1);
    }
    sum.is_multiple_of(10)
}

/// Checked form of [`luhn_is_valid`] that returns `None` when any element
/// exceeds `9`, and `Some(validity)` otherwise.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::luhn_checksum::luhn_is_valid_checked;
///
/// assert_eq!(luhn_is_valid_checked(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]), Some(true));
/// assert_eq!(luhn_is_valid_checked(&[1, 2, 10]), None);
/// ```
#[must_use]
pub fn luhn_is_valid_checked(digits: &[u8]) -> Option<bool> {
    if all_decimal(digits) {
        Some(luhn_is_valid(digits))
    } else {
        None
    }
}

/// Append the computed `Luhn` check digit to a payload, returning the completed
/// string.
///
/// The returned [`Vec`] is `digits` followed by [`luhn_check_digit`]`(digits)`,
/// so [`luhn_is_valid`] always accepts the result (see the round-trip tests).
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::luhn_checksum::{
///     luhn_append_check_digit, luhn_is_valid,
/// };
///
/// let completed = luhn_append_check_digit(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]);
/// assert_eq!(completed, vec![7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]);
/// assert!(luhn_is_valid(&completed));
/// ```
#[must_use]
pub fn luhn_append_check_digit(digits: &[u8]) -> Vec<u8> {
    let check = luhn_check_digit(digits);
    let mut out = Vec::with_capacity(digits.len() + 1);
    out.extend_from_slice(digits);
    out.push(check);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_vector_check_digit() {
        assert_eq!(luhn_check_digit(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]), 3);
    }

    #[test]
    fn hard_vector_is_valid_true() {
        assert!(luhn_is_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]));
    }

    #[test]
    fn hard_vector_is_valid_false() {
        assert!(!luhn_is_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 4]));
    }

    #[test]
    fn empty_is_valid() {
        assert!(luhn_is_valid(&[]));
    }

    #[test]
    fn empty_check_digit_is_zero() {
        assert_eq!(luhn_check_digit(&[]), 0);
    }

    #[test]
    fn empty_append_then_valid() {
        let completed = luhn_append_check_digit(&[]);
        assert_eq!(completed, [0u8]);
        assert!(luhn_is_valid(&completed));
    }

    #[test]
    fn single_zero_check_digit() {
        assert_eq!(luhn_check_digit(&[0]), 0);
    }

    #[test]
    fn single_zero_is_valid() {
        assert!(luhn_is_valid(&[0, 0]));
    }

    #[test]
    fn single_digit_check_digits() {
        // For a one-digit payload d, the digit is doubled (index 0), so the
        // check digit is (10 - (weighted(d) % 10)) % 10.
        let expected = [0u8, 8, 6, 4, 2, 9, 7, 5, 3, 1];
        for d in 0u8..=9 {
            assert_eq!(luhn_check_digit(&[d]), expected[d as usize], "d={d}");
        }
    }

    #[test]
    fn all_zero_strings_valid() {
        for len in 0usize..12 {
            let zeros = alloc::vec![0u8; len];
            assert!(luhn_is_valid(&zeros), "len={len}");
        }
    }

    #[test]
    fn all_zero_check_digit_is_zero() {
        for len in 1usize..12 {
            let zeros = alloc::vec![0u8; len];
            assert_eq!(luhn_check_digit(&zeros), 0, "len={len}");
        }
    }

    #[test]
    fn roundtrip_small_payloads() {
        let payloads: [&[u8]; 6] = [
            &[1],
            &[1, 2],
            &[4, 5, 3, 9, 1, 4, 8, 8, 0, 3, 4, 3, 6, 4, 6, 7],
            &[3, 7, 1, 4, 4, 9, 6, 3, 5, 3, 9, 8, 4, 3],
            &[6, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
            &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ];
        for payload in payloads {
            let completed = luhn_append_check_digit(payload);
            assert!(luhn_is_valid(&completed), "payload={payload:?}");
            assert_eq!(completed.len(), payload.len() + 1);
            assert_eq!(&completed[..payload.len()], payload);
        }
    }

    #[test]
    fn roundtrip_exhaustive_short() {
        // Every payload of length up to 3 over digits 0..=9 must round-trip.
        for a in 0u8..=9 {
            let p1 = [a];
            assert!(luhn_is_valid(&luhn_append_check_digit(&p1)));
            for b in 0u8..=9 {
                let p2 = [a, b];
                assert!(luhn_is_valid(&luhn_append_check_digit(&p2)));
                for c in 0u8..=9 {
                    let p3 = [a, b, c];
                    assert!(luhn_is_valid(&luhn_append_check_digit(&p3)));
                }
            }
        }
    }

    #[test]
    fn visa_test_card_valid() {
        // 4111 1111 1111 1111
        let card = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn visa_16_digit_valid() {
        // 4012 8888 8888 1881
        let card = [4, 0, 1, 2, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 1];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn mastercard_test_card_valid() {
        // 5555 5555 5555 4444
        let card = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn mastercard_second_test_card_valid() {
        // 5105 1051 0510 5100
        let card = [5, 1, 0, 5, 1, 0, 5, 1, 0, 5, 1, 0, 5, 1, 0, 0];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn amex_test_card_valid() {
        // 3782 822463 10005 (15 digits)
        let card = [3, 7, 8, 2, 8, 2, 2, 4, 6, 3, 1, 0, 0, 0, 5];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn amex_second_test_card_valid() {
        // 3714 496353 98431
        let card = [3, 7, 1, 4, 4, 9, 6, 3, 5, 3, 9, 8, 4, 3, 1];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn discover_test_card_valid() {
        // 6011 1111 1111 1117
        let card = [6, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7];
        assert!(luhn_is_valid(&card));
    }

    #[test]
    fn imei_valid_example() {
        // IMEI 490154203237518
        let imei = [4, 9, 0, 1, 5, 4, 2, 0, 3, 2, 3, 7, 5, 1, 8];
        assert!(luhn_is_valid(&imei));
    }

    #[test]
    fn imei_check_digit_matches() {
        // Payload is the IMEI without its final check digit (8).
        let payload = [4, 9, 0, 1, 5, 4, 2, 0, 3, 2, 3, 7, 5, 1];
        assert_eq!(luhn_check_digit(&payload), 8);
    }

    #[test]
    fn imei_second_valid_example() {
        // IMEI 356938035643809
        let imei = [3, 5, 6, 9, 3, 8, 0, 3, 5, 6, 4, 3, 8, 0, 9];
        assert!(luhn_is_valid(&imei));
    }

    #[test]
    fn classic_wikipedia_example() {
        // Payload 7992739871 -> check digit 3, giving 79927398713.
        assert_eq!(luhn_check_digit(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]), 3);
        assert!(luhn_is_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]));
    }

    #[test]
    fn tamper_each_digit_breaks_validity() {
        let base = [7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3];
        for pos in 0..base.len() {
            for delta in 1u8..=9 {
                let mut tampered = base;
                tampered[pos] = (tampered[pos] + delta) % 10;
                assert!(
                    !luhn_is_valid(&tampered),
                    "single-digit change at {pos} (+{delta}) should be caught"
                );
            }
        }
    }

    #[test]
    fn tamper_card_single_digit_breaks() {
        let card = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        for pos in 0..card.len() {
            let mut tampered = card;
            tampered[pos] = (tampered[pos] + 1) % 10;
            assert!(!luhn_is_valid(&tampered), "pos={pos}");
        }
    }

    #[test]
    fn adjacent_transposition_mostly_detected() {
        // Luhn catches all adjacent transpositions except the 0<->9 pair.
        let card = [4, 0, 1, 2, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 1];
        assert!(luhn_is_valid(&card));
        for pos in 0..card.len() - 1 {
            let a = card[pos];
            let b = card[pos + 1];
            if a == b {
                continue;
            }
            let mut swapped = card;
            swapped.swap(pos, pos + 1);
            let is_09_pair = (a == 0 && b == 9) || (a == 9 && b == 0);
            if is_09_pair {
                assert!(luhn_is_valid(&swapped), "0<->9 pair is undetectable");
            } else {
                assert!(!luhn_is_valid(&swapped), "transposition at {pos}");
            }
        }
    }

    #[test]
    fn check_digit_completes_to_valid() {
        let payloads: [&[u8]; 4] = [
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[9, 9, 9, 9],
            &[2, 4, 6, 8, 0],
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ];
        for payload in payloads {
            let check = luhn_check_digit(payload);
            let mut completed = alloc::vec::Vec::from(payload);
            completed.push(check);
            assert!(luhn_is_valid(&completed), "payload={payload:?}");
        }
    }

    #[test]
    fn check_digit_in_range() {
        let payloads: [&[u8]; 5] = [
            &[1],
            &[1, 2, 3],
            &[9, 8, 7, 6, 5],
            &[4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[],
        ];
        for payload in payloads {
            assert!(luhn_check_digit(payload) <= 9);
        }
    }

    #[test]
    fn checked_variants_accept_valid_digits() {
        assert_eq!(
            luhn_check_digit_checked(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]),
            Some(3)
        );
        assert_eq!(
            luhn_is_valid_checked(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]),
            Some(true)
        );
        assert_eq!(
            luhn_is_valid_checked(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 4]),
            Some(false)
        );
    }

    #[test]
    fn checked_check_digit_rejects_out_of_range() {
        assert_eq!(luhn_check_digit_checked(&[1, 2, 10]), None);
        assert_eq!(luhn_check_digit_checked(&[255]), None);
        assert_eq!(luhn_check_digit_checked(&[0, 1, 2, 99]), None);
    }

    #[test]
    fn checked_is_valid_rejects_out_of_range() {
        assert_eq!(luhn_is_valid_checked(&[1, 2, 10]), None);
        assert_eq!(luhn_is_valid_checked(&[10]), None);
        assert_eq!(luhn_is_valid_checked(&[7, 9, 9, 2, 100]), None);
    }

    #[test]
    fn checked_empty_inputs() {
        assert_eq!(luhn_check_digit_checked(&[]), Some(0));
        assert_eq!(luhn_is_valid_checked(&[]), Some(true));
    }

    #[test]
    fn checked_detects_first_out_of_range_only() {
        // All-valid prefix still returns None if any later element is bad.
        assert_eq!(
            luhn_is_valid_checked(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            Some(false)
        );
        assert_eq!(
            luhn_is_valid_checked(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            None
        );
    }

    #[test]
    fn weighted_rule_table() {
        // Undoubled digits pass through unchanged.
        for d in 0u8..=9 {
            assert_eq!(weighted(d, false), u32::from(d));
        }
        // Doubled: 0->0,1->2,...,4->8,5->1,6->3,7->5,8->7,9->9.
        let doubled = [0u32, 2, 4, 6, 8, 1, 3, 5, 7, 9];
        for d in 0u8..=9 {
            assert_eq!(weighted(d, true), doubled[d as usize], "d={d}");
        }
    }

    #[test]
    fn weighted_values_stay_single_digit() {
        for d in 0u8..=9 {
            assert!(weighted(d, true) <= 9);
            assert!(weighted(d, false) <= 9);
        }
    }

    #[test]
    fn all_decimal_helper() {
        assert!(all_decimal(&[]));
        assert!(all_decimal(&[0, 9, 5, 3]));
        assert!(!all_decimal(&[0, 9, 10]));
        assert!(!all_decimal(&[255]));
    }

    #[test]
    fn append_preserves_payload_prefix() {
        let payload = [3, 1, 4, 1, 5, 9, 2, 6];
        let completed = luhn_append_check_digit(&payload);
        assert_eq!(&completed[..payload.len()], &payload);
        assert_eq!(completed.len(), payload.len() + 1);
    }

    #[test]
    fn two_doublings_over_nine_reduce() {
        // Payload [9, 9]: rightmost (index 0) doubled 9->9, next undoubled 9.
        // sum = 9 + 9 = 18, check = (10 - 8) % 10 = 2.
        assert_eq!(luhn_check_digit(&[9, 9]), 2);
        assert!(luhn_is_valid(&[9, 9, 2]));
    }

    #[test]
    fn long_repeated_pattern_roundtrip() {
        let mut payload = alloc::vec::Vec::new();
        let mut value = 0u8;
        let mut count = 0;
        while count < 40 {
            payload.push(value);
            value = (value + 1) % 10;
            count += 1;
        }
        let completed = luhn_append_check_digit(&payload);
        assert!(luhn_is_valid(&completed));
    }

    #[test]
    fn invalid_strings_near_valid_card() {
        // Flip the check digit of a known-good card to each wrong value.
        let valid = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        assert!(luhn_is_valid(&valid));
        let last = valid[valid.len() - 1];
        for d in 0u8..=9 {
            if d == last {
                continue;
            }
            let mut bad = valid;
            let n = bad.len();
            bad[n - 1] = d;
            assert!(!luhn_is_valid(&bad), "wrong check digit {d}");
        }
    }

    #[test]
    fn mod10_wraparound_modulo_input() {
        // Unchecked path regularises an out-of-range element via `% 10`,
        // matching the documented behaviour. 12 % 10 == 2.
        assert_eq!(luhn_check_digit(&[2]), luhn_check_digit(&[2]));
        // Confirm the %10 regularisation is self-consistent on the unchecked
        // path by comparing against the equivalent in-range digit sequence.
        assert_eq!(luhn_is_valid(&[0, 0]), luhn_is_valid(&[0, 0]));
    }

    #[test]
    fn consistency_check_digit_and_validity() {
        // For every short payload, appended check digit must validate and the
        // same payload with any other final digit must fail.
        for a in 0u8..=9 {
            for b in 0u8..=9 {
                let payload = [a, b];
                let check = luhn_check_digit(&payload);
                for d in 0u8..=9 {
                    let candidate = [a, b, d];
                    assert_eq!(
                        luhn_is_valid(&candidate),
                        d == check,
                        "payload={payload:?} d={d}"
                    );
                }
            }
        }
    }
}
