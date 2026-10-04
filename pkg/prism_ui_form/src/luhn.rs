//! Luhn (mod-10) checksum validation for payment-card and identifier fields.
//!
//! The Luhn algorithm (ISO/IEC 7812-1) is the industry-standard check used by
//! every major card scheme — Visa, Mastercard, American Express, Discover, JCB,
//! Diners Club — as well as IMEI device identifiers and many national ID
//! numbers. It catches all single-digit transcription errors and most adjacent
//! transposition errors, which is exactly why payment forms gate submission on
//! it client-side.
//!
//! This rule is a genuine complement to the generic validators: a checksum is
//! not expressible with [`crate::pattern`] (a regular predicate) or
//! [`crate::int_range`], because validity depends on digit arithmetic rather
//! than shape or magnitude.
//!
//! The implementation uses only integer arithmetic (addition, multiplication,
//! and comparison), so it is allocation-free, deterministic, and
//! `no_std`-friendly, with no banned transcendental math.

use alloc::boxed::Box;
use alloc::string::String;

use crate::error::ValidationError;
use crate::validator::{BoxedValidator, Validator};

/// Returns whether the digits in `input` satisfy the Luhn mod-10 checksum.
///
/// ASCII spaces and hyphens are ignored, so grouped entry such as
/// `4111 1111 1111 1111` or `4111-1111-1111-1111` validates the same as the
/// bare digit string. Any other non-digit character makes the value invalid,
/// as does an input containing no digits at all.
///
/// The check doubles every second digit counting from the right; when a doubled
/// value exceeds nine, nine is subtracted (equivalent to summing its decimal
/// digits). The value passes when the total is a multiple of ten.
pub fn passes_luhn(input: &str) -> bool {
    let mut sum: u32 = 0;
    let mut digit_count: u32 = 0;
    let mut double = false;

    for ch in input.chars().rev() {
        if ch == ' ' || ch == '-' {
            continue;
        }
        let Some(digit) = ch.to_digit(10) else {
            return false;
        };
        let contribution = if double {
            let doubled = digit * 2;
            if doubled > 9 { doubled - 9 } else { doubled }
        } else {
            digit
        };
        sum += contribution;
        digit_count += 1;
        double = !double;
    }

    digit_count > 0 && sum.is_multiple_of(10)
}

/// Rejects values whose digits fail the Luhn checksum.
struct Luhn;

impl Validator<String> for Luhn {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if passes_luhn(value) {
            Ok(())
        } else {
            Err(ValidationError::message_only(
                "Must pass the Luhn checksum.",
            ))
        }
    }
}

/// Require the value's digits to satisfy the Luhn mod-10 checksum, the standard
/// used to sanity-check payment-card numbers and similar identifiers.
///
/// ASCII spaces and hyphens are ignored so grouped input validates naturally.
/// See [`passes_luhn`] for the precise rule.
pub fn luhn() -> BoxedValidator {
    Box::new(Luhn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    /// Well-known Luhn-valid identifiers: the classic textbook example plus the
    /// canonical scheme test numbers (Visa, Mastercard, Amex, Discover, JCB,
    /// Diners).
    const VALID: &[&str] = &[
        "79927398713",       // classic Luhn example
        "4111111111111111",  // Visa test
        "4012888888881881",  // Visa test
        "5555555555554444",  // Mastercard test
        "5105105105105100",  // Mastercard test
        "378282246310005",   // American Express test
        "371449635398431",   // American Express test
        "6011111111111117",  // Discover test
        "6011000990139424",  // Discover test
        "3530111333300000",  // JCB test
        "30569309025904",    // Diners Club test
        "38520000023237",    // Diners Club test
        "0",                 // trivially valid (sum zero)
    ];

    #[test]
    fn golden_valid_numbers_pass() {
        for case in VALID {
            assert!(passes_luhn(case), "expected {case:?} to pass Luhn");
            assert!(luhn().validate(&case.to_string()).is_ok());
        }
    }

    #[test]
    fn separators_are_ignored() {
        assert!(passes_luhn("4111 1111 1111 1111"));
        assert!(passes_luhn("4111-1111-1111-1111"));
        assert!(passes_luhn(" 7992 7398 713 "));
        // Grouping must not change the verdict relative to the bare digits.
        assert_eq!(passes_luhn("4111111111111111"), passes_luhn("4111 1111 1111 1111"));
    }

    #[test]
    fn non_digits_and_empty_are_rejected() {
        assert!(!passes_luhn(""));
        assert!(!passes_luhn("    "));
        assert!(!passes_luhn("notanumber"));
        assert!(!passes_luhn("4111 1111 1111 111a"));
        assert!(!passes_luhn("4111.1111.1111.1111")); // dot is not a separator
        // Fullwidth digits are not ASCII radix-10 digits.
        assert!(!passes_luhn("\u{ff14}111111111111111"));
        let err = luhn().validate(&"notanumber".to_string()).unwrap_err();
        assert_eq!(err.message, "Must pass the Luhn checksum.");
    }

    #[test]
    fn classic_single_valid_check_digit_property() {
        // Of the ten numbers 7992739871X, exactly one (X = 3) is Luhn-valid.
        let mut valid = Vec::new();
        for d in 0..10u32 {
            let candidate = alloc::format!("799273987 1{d}");
            if passes_luhn(&candidate) {
                valid.push(d);
            }
        }
        assert_eq!(valid, [3]);
    }

    // --- Property tests over a deterministic generator ---------------------

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: u32) -> u32 {
            (self.next_u64() % bound as u64) as u32
        }
    }

    /// Build a random base digit string of length `len` (ASCII digits only).
    fn gen_digits(rng: &mut SplitMix64, len: u32) -> String {
        let mut out = Vec::with_capacity(len as usize);
        for _ in 0..len {
            out.push(b'0' + rng.below(10) as u8);
        }
        String::from_utf8(out).expect("digits are ASCII")
    }

    /// Return the unique check digit `d` for which `base + d` is Luhn-valid,
    /// asserting on the way that exactly one such digit exists.
    fn unique_check_digit(base: &str) -> u8 {
        let mut found = Vec::new();
        for d in 0..10u32 {
            let candidate = alloc::format!("{base}{d}");
            if passes_luhn(&candidate) {
                found.push(d as u8);
            }
        }
        assert_eq!(found.len(), 1, "base {base:?} should have one check digit");
        found[0]
    }

    #[test]
    fn check_digit_is_unique_and_completes_a_valid_number() {
        let mut rng = SplitMix64(0x1357_9BDF_0246_8ACE);
        for _ in 0..1_500 {
            let len = 1 + rng.below(18);
            let base = gen_digits(&mut rng, len);
            let check = unique_check_digit(&base);
            let full = alloc::format!("{base}{check}");
            assert!(passes_luhn(&full), "{full:?} should pass with its check digit");
        }
    }

    #[test]
    fn any_single_digit_change_breaks_validity() {
        // The Luhn position maps are bijections on 0..=9, so altering exactly
        // one digit of a valid number to a different digit always invalidates.
        let mut rng = SplitMix64(0x2468_ACE0_1357_9BDF);
        for _ in 0..1_000 {
            let len = 1 + rng.below(16);
            let base = gen_digits(&mut rng, len);
            let check = unique_check_digit(&base);
            let full = alloc::format!("{base}{check}");
            let mut bytes = full.into_bytes();
            let pos = rng.below(bytes.len() as u32) as usize;
            let original = bytes[pos];
            // Shift to a different digit.
            let shift = 1 + rng.below(9) as u8;
            bytes[pos] = b'0' + ((original - b'0' + shift) % 10);
            assert_ne!(bytes[pos], original);
            let mutated = String::from_utf8(bytes).expect("digits are ASCII");
            assert!(
                !passes_luhn(&mutated),
                "{mutated:?} should be invalid after a single-digit change"
            );
        }
    }

    #[test]
    fn inserting_separators_preserves_validity() {
        let mut rng = SplitMix64(0x0F1E_2D3C_4B5A_6978);
        for _ in 0..1_000 {
            let len = 1 + rng.below(16);
            let base = gen_digits(&mut rng, len);
            let check = unique_check_digit(&base);
            let full = alloc::format!("{base}{check}");
            // Interleave spaces/hyphens between every digit.
            let mut spaced = String::new();
            for (index, ch) in full.chars().enumerate() {
                if index > 0 {
                    spaced.push(if index.is_multiple_of(2) { ' ' } else { '-' });
                }
                spaced.push(ch);
            }
            assert_eq!(
                passes_luhn(&spaced),
                passes_luhn(&full),
                "separators changed the verdict for {spaced:?}"
            );
            assert!(passes_luhn(&spaced));
        }
    }
}
