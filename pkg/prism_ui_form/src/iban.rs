//! IBAN (International Bank Account Number) checksum validation for payment
//! fields.
//!
//! An IBAN is defined by ISO 13616 and uses the ISO 7064 `MOD 97-10` check
//! character scheme (the same mod-97 trick used for VAT numbers and shipping
//! container codes). Its layout is:
//!
//! * two uppercase letters — the ISO 3166-1 country code,
//! * two check digits,
//! * up to thirty country-specific BBAN characters (letters and digits).
//!
//! Validation rearranges the string so the country code and check digits move
//! to the end, maps each letter `A..=Z` to the number `10..=35`, interprets the
//! result as one large base-10 integer, and accepts it when that integer is
//! congruent to `1` modulo `97`. Because `MOD 97-10` detects every single-digit
//! transcription error and almost every transposition, this is exactly the
//! gate a payment form wants client-side before it ever hits a bank API.
//!
//! The large integer is never materialised: the remainder is folded one
//! character at a time (`r = (r * 10 + digit) % 97`, or `r = (r * 100 + value)
//! % 97` for a letter), so the check is allocation-light, deterministic, and
//! `no_std`-friendly with no banned transcendental math.
//!
//! # Honest boundary
//!
//! This rule enforces the ISO 13616 *structure and checksum* only. It does not
//! consult the IBAN registry of per-country fixed lengths, so a value with a
//! structurally valid but country-wrong length that still satisfies the
//! checksum would pass here. The registry is a large, slowly changing table and
//! is intentionally out of scope for a dependency-free checksum rule.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::ValidationError;
use crate::validator::{BoxedValidator, Validator};

/// Fold an already-normalized (uppercase `A..=Z` / `0..=9`) character stream
/// into its value modulo `97`, expanding each letter to the two-digit number
/// `10..=35` as ISO 7064 requires.
fn mod97(chars: impl Iterator<Item = char>) -> u32 {
    let mut remainder: u32 = 0;
    for ch in chars {
        if let Some(digit) = ch.to_digit(10) {
            remainder = (remainder * 10 + digit) % 97;
        } else {
            // Guaranteed `A..=Z` by the caller, so this maps to `10..=35`.
            let value = u32::from(ch) - u32::from('A') + 10;
            remainder = (remainder * 100 + value) % 97;
        }
    }
    remainder
}

/// Returns whether `input` is a structurally valid IBAN whose ISO 7064
/// `MOD 97-10` checksum holds.
///
/// ASCII spaces are ignored so the grouped print format `GB82 WEST 1234 5698
/// 7654 32` validates the same as its compact form, and ASCII letters are
/// upper-cased first so lowercase entry is accepted. Any other character, a
/// length outside `5..=34`, a non-letter country code, or non-digit check
/// positions all make the value invalid.
///
/// See the [module docs](self) for the exact algorithm and its honest boundary
/// (per-country registry lengths are not enforced).
pub fn is_valid_iban(input: &str) -> bool {
    // Normalize: drop ASCII spaces and upper-case ASCII letters. Non-ASCII
    // input survives here but is rejected by the alphanumeric check below,
    // which keeps the later slicing on safe `char` boundaries.
    let chars: Vec<char> = input
        .chars()
        .filter(|&c| c != ' ')
        .map(|c| c.to_ascii_uppercase())
        .collect();

    let len = chars.len();
    if !(5..=34).contains(&len) {
        return false;
    }
    // Country code must be two letters; check positions must be two digits.
    if !(chars[0].is_ascii_uppercase() && chars[1].is_ascii_uppercase()) {
        return false;
    }
    if !(chars[2].is_ascii_digit() && chars[3].is_ascii_digit()) {
        return false;
    }
    // Every character must be an uppercase letter or a digit.
    if !chars
        .iter()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return false;
    }

    // Move the first four characters (country code + check digits) to the end,
    // then the whole value must be congruent to 1 modulo 97.
    let rearranged = chars[4..].iter().chain(chars[..4].iter()).copied();
    mod97(rearranged) == 1
}

/// Rejects values that are not a checksum-valid IBAN.
struct Iban;

impl Validator<String> for Iban {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if is_valid_iban(value) {
            Ok(())
        } else {
            Err(ValidationError::message_only("Must be a valid IBAN."))
        }
    }
}

/// Require the value to be a structurally valid IBAN whose ISO 7064
/// `MOD 97-10` checksum holds.
///
/// ASCII spaces are ignored and ASCII letters are upper-cased, so both the
/// grouped print format and lowercase entry validate naturally. See
/// [`is_valid_iban`] for the precise rule.
pub fn iban() -> BoxedValidator {
    Box::new(Iban)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    /// Canonical valid IBANs drawn from the ISO 13616 / national examples,
    /// including Norway's 15-character minimum.
    const VALID: &[&str] = &[
        "GB82WEST12345698765432",
        "DE89370400440532013000",
        "FR1420041010050500013M02606",
        "ES9121000418450200051332",
        "NL91ABNA0417164300",
        "BE68539007547034",
        "CH9300762011623852957",
        "IT60X0542811101000000123456",
        "NO9386011117947",
        "SA0380000000608010167519",
        "GR1601101250000000012300695",
    ];

    #[test]
    fn golden_valid_ibans_pass() {
        for case in VALID {
            assert!(is_valid_iban(case), "expected {case:?} to be valid");
            assert!(iban().validate(&case.to_string()).is_ok());
        }
    }

    #[test]
    fn grouped_and_lowercase_forms_match() {
        assert!(is_valid_iban("GB82 WEST 1234 5698 7654 32"));
        assert!(is_valid_iban("gb82west12345698765432"));
        assert!(is_valid_iban("Gb82 west 1234 5698 7654 32"));
        // Grouping or case must not change the verdict relative to the compact
        // uppercase form.
        assert_eq!(
            is_valid_iban("GB82WEST12345698765432"),
            is_valid_iban("GB82 WEST 1234 5698 7654 32"),
        );
    }

    #[test]
    fn structural_rejections() {
        // Wrong check digits (one off from the valid GB example).
        assert!(!is_valid_iban("GB83WEST12345698765432"));
        // Country code is not two letters.
        assert!(!is_valid_iban("1282WEST12345698765432"));
        // Check positions are not two digits.
        assert!(!is_valid_iban("GBX2WEST12345698765432"));
        // Illegal character (space is stripped, but '-' is not a separator).
        assert!(!is_valid_iban("GB82-WEST-1234-5698-7654-32"));
        // Too short / empty.
        assert!(!is_valid_iban(""));
        assert!(!is_valid_iban("GB82"));
        // Too long (35 characters).
        assert!(!is_valid_iban("GB82WEST123456987654321234567890123"));
        // Non-ASCII must not panic and must be rejected.
        assert!(!is_valid_iban("GB82WEST1234569876543\u{00e9}"));
        let err = iban().validate(&"notaniban".to_string()).unwrap_err();
        assert_eq!(err.message, "Must be a valid IBAN.");
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
            (self.next_u64() % u64::from(bound)) as u32
        }
    }

    /// Build the two ISO 7064 check digits for `country` + `bban`, independent
    /// of [`is_valid_iban`]: rearrange as `bban + country + "00"`, then the
    /// check equals `98 - (value mod 97)`.
    fn compute_check(country: &str, bban: &str) -> String {
        let chars = bban.chars().chain(country.chars()).chain("00".chars());
        let check = 98 - mod97(chars);
        alloc::format!("{check:02}")
    }

    fn gen_country(rng: &mut SplitMix64) -> String {
        let mut out = String::new();
        for _ in 0..2 {
            out.push((b'A' + rng.below(26) as u8) as char);
        }
        out
    }

    /// Random BBAN of `len` uppercase letters and digits.
    fn gen_bban(rng: &mut SplitMix64, len: u32) -> String {
        let mut out = Vec::with_capacity(len as usize);
        for _ in 0..len {
            let v = rng.below(36);
            let byte = if v < 10 {
                b'0' + v as u8
            } else {
                b'A' + (v - 10) as u8
            };
            out.push(byte);
        }
        String::from_utf8(out).expect("alnum is ASCII")
    }

    #[test]
    fn computed_check_digits_yield_valid_iban() {
        let mut rng = SplitMix64(0x1357_9BDF_0246_8ACE);
        for _ in 0..2_000 {
            let country = gen_country(&mut rng);
            // Total length stays within 5..=34: bban length in 1..=30.
            let bban_len = 1 + rng.below(30);
            let bban = gen_bban(&mut rng, bban_len);
            let check = compute_check(&country, &bban);
            let full = alloc::format!("{country}{check}{bban}");
            assert!(
                is_valid_iban(&full),
                "{full:?} built with its own check digits should validate"
            );
        }
    }

    #[test]
    fn grouping_with_spaces_preserves_validity() {
        let mut rng = SplitMix64(0x0F1E_2D3C_4B5A_6978);
        for _ in 0..1_000 {
            let country = gen_country(&mut rng);
            let bban_len = 1 + rng.below(30);
            let bban = gen_bban(&mut rng, bban_len);
            let check = compute_check(&country, &bban);
            let full = alloc::format!("{country}{check}{bban}");
            // Insert a space after every fourth character (print format).
            let mut spaced = String::new();
            for (index, ch) in full.chars().enumerate() {
                if index > 0 && index.is_multiple_of(4) {
                    spaced.push(' ');
                }
                spaced.push(ch);
            }
            assert_eq!(
                is_valid_iban(&spaced),
                is_valid_iban(&full),
                "spacing changed the verdict for {spaced:?}"
            );
            assert!(is_valid_iban(&spaced));
        }
    }

    #[test]
    fn any_single_check_digit_change_breaks_validity() {
        // ISO 7064 MOD 97-10 detects every single-digit substitution, so
        // altering one of the two check digits to a different value always
        // invalidates the IBAN.
        let mut rng = SplitMix64(0x2468_ACE0_1357_9BDF);
        for _ in 0..2_000 {
            let country = gen_country(&mut rng);
            let bban_len = 1 + rng.below(30);
            let bban = gen_bban(&mut rng, bban_len);
            let check = compute_check(&country, &bban);
            let full = alloc::format!("{country}{check}{bban}");

            let mut bytes = full.into_bytes();
            // Positions 2 and 3 are the check digits.
            let pos = 2 + rng.below(2) as usize;
            let original = bytes[pos];
            let shift = 1 + rng.below(9) as u8;
            bytes[pos] = b'0' + ((original - b'0' + shift) % 10);
            assert_ne!(bytes[pos], original);
            let mutated = String::from_utf8(bytes).expect("alnum is ASCII");
            assert!(
                !is_valid_iban(&mutated),
                "{mutated:?} should be invalid after a single check-digit change"
            );
        }
    }
}
