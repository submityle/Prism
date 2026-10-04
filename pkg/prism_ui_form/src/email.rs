//! HTML5 email-address validation (WHATWG HTML Living Standard semantics).
//!
//! This mirrors the grammar the WHATWG HTML standard defines for
//! `<input type="email">`, which is a deliberate ("willful violation")
//! simplification of RFC 5322 that every mainstream browser implements via a
//! single regular expression:
//!
//! ```text
//! /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/
//! ```
//!
//! The implementation below recognises exactly that language without pulling in
//! a regex engine: it uses only ASCII character-class tests and length
//! comparisons, so it stays allocation-free and `no_std`-friendly.
//!
//! Consequences worth noting, all faithful to the standard:
//!
//! * A single-label domain such as `user@example` is **valid** — the standard
//!   does not require a dot or a top-level domain.
//! * The local part permits a rich punctuation set and places no restriction on
//!   dot placement (leading, trailing, and consecutive dots are all accepted).
//! * Whitespace is never permitted anywhere, so the raw value is matched as-is
//!   without trimming — consistent with browser behaviour.

use alloc::boxed::Box;
use alloc::string::String;

use crate::error::ValidationError;
use crate::validator::{BoxedValidator, Validator};

/// Returns whether `value` is a valid email address under the WHATWG HTML
/// Living Standard definition used by `<input type="email">`.
///
/// This recognises the standard's email grammar exactly; see the module
/// documentation for the precise reference regular expression and its
/// consequences.
pub fn is_valid_email(value: &str) -> bool {
    // The grammar is `local "@" domain`; `split_once` isolates the local part
    // at the first `@`. Any subsequent `@` necessarily lands inside `domain`,
    // where it is not a valid label character and is therefore rejected.
    let Some((local, domain)) = value.split_once('@') else {
        return false;
    };
    is_valid_local(local) && is_valid_domain(domain)
}

/// The local part is one or more `atext` characters: the reference-regex
/// character class of ASCII alphanumerics plus the punctuation set listed in
/// [`is_atext`]. It must be non-empty.
fn is_valid_local(local: &str) -> bool {
    !local.is_empty() && local.bytes().all(is_atext)
}

/// Whether `byte` is a permitted local-part ("atext") character.
fn is_atext(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'.' | b'!'
                | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'/'
                | b'='
                | b'?'
                | b'^'
                | b'_'
                | b'`'
                | b'{'
                | b'|'
                | b'}'
                | b'~'
                | b'-'
        )
}

/// The domain is one or more dot-separated labels, each matching the reference
/// regex `[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?`.
fn is_valid_domain(domain: &str) -> bool {
    if domain.is_empty() {
        return false;
    }
    domain.split('.').all(is_valid_label)
}

/// Whether a single domain label is well formed: 1..=63 bytes, starting and
/// ending with an ASCII alphanumeric, with interior bytes drawn from ASCII
/// alphanumerics and hyphens.
fn is_valid_label(label: &str) -> bool {
    let bytes = label.as_bytes();
    let len = bytes.len();
    if len == 0 || len > 63 {
        return false;
    }
    if !bytes[0].is_ascii_alphanumeric() {
        return false;
    }
    if len == 1 {
        return true;
    }
    if !bytes[len - 1].is_ascii_alphanumeric() {
        return false;
    }
    bytes[1..len - 1]
        .iter()
        .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
}

/// Rejects values that are not valid email addresses.
struct Email;

impl Validator<String> for Email {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if is_valid_email(value) {
            Ok(())
        } else {
            Err(ValidationError::message_only(
                "Must be a valid email address.",
            ))
        }
    }
}

/// Require the value to be a valid email address, matching the WHATWG HTML
/// Living Standard grammar for `<input type="email">`.
///
/// The value is matched verbatim, without trimming surrounding whitespace, so
/// this behaves like a browser's native email constraint. See
/// [`is_valid_email`] for the exact grammar.
pub fn email() -> BoxedValidator {
    Box::new(Email)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    /// Golden-positive set: every entry must be accepted. These combine the
    /// WHATWG standard's own examples with the structural edge cases the grammar
    /// explicitly permits.
    const VALID: &[&str] = &[
        "a@b.c",
        "simple@example.com",
        "very.common@example.com",
        "user.name+tag@example.co.uk",
        "disposable.style.email.with+symbol@example.com",
        "o'brien@example.com",
        "x@example",                        // single-label domain is valid
        "user@sub.domain.example.com",      // many labels
        "!#$%&'*+/=?^_`{|}~-@example.com",   // full atext punctuation set
        ".leading.dot@example.com",         // dots anywhere in local part
        "trailing.dot.@example.com",
        "a@a",                              // minimal
        "1234567890@example.com",
        "user@xn--80ak6aa92e.com",          // punycode label
    ];

    /// Golden-negative set: every entry must be rejected.
    const INVALID: &[&str] = &[
        "",                        // empty
        "plainaddress",            // no '@'
        "@example.com",            // empty local part
        "user@",                   // empty domain
        "user@@example.com",       // second '@' lands in the domain
        "user@.com",               // leading empty label
        "user@com.",               // trailing empty label
        "user@exa mple.com",       // space in domain
        "user name@example.com",   // space in local part
        " user@example.com",       // leading whitespace is not trimmed
        "user@example.com ",       // trailing whitespace is not trimmed
        "user@-example.com",       // label may not start with a hyphen
        "user@example-.com",       // label may not end with a hyphen
        "user@example..com",       // empty interior label
        "us\u{00e9}r@example.com",  // non-ASCII in local part
        "user@exämple.com",        // non-ASCII in domain
    ];

    #[test]
    fn golden_positive_set_is_accepted() {
        for case in VALID {
            assert!(
                is_valid_email(case),
                "expected {case:?} to be a valid email"
            );
            assert!(email().validate(&case.to_string()).is_ok());
        }
    }

    #[test]
    fn golden_negative_set_is_rejected() {
        for case in INVALID {
            assert!(
                !is_valid_email(case),
                "expected {case:?} to be an invalid email"
            );
            let err = email().validate(&case.to_string()).unwrap_err();
            assert_eq!(err.message, "Must be a valid email address.");
        }
    }

    #[test]
    fn label_length_boundary_is_exact() {
        // A 63-byte label is the longest the grammar allows; 64 is one too many.
        let label_63 = "a".repeat(63);
        let label_64 = "a".repeat(64);
        assert!(is_valid_email(&alloc::format!("user@{label_63}.com")));
        assert!(!is_valid_email(&alloc::format!("user@{label_64}.com")));
    }

    // --- Property tests over a deterministic generator ---------------------

    /// A tiny `SplitMix64` PRNG so the properties are reproducible without any
    /// external dependency.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    const ATEXT: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!#$%&'*+/=?^_`{|}~-";
    const ALNUM: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const LABEL_INNER: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-";

    fn pick(rng: &mut SplitMix64, set: &[u8]) -> u8 {
        set[rng.below(set.len())]
    }

    /// Build a syntactically valid email from the grammar.
    fn gen_valid(rng: &mut SplitMix64) -> String {
        let mut out = Vec::new();
        let local_len = 1 + rng.below(10);
        for _ in 0..local_len {
            out.push(pick(rng, ATEXT));
        }
        out.push(b'@');
        let labels = 1 + rng.below(3);
        for label_index in 0..labels {
            if label_index > 0 {
                out.push(b'.');
            }
            let label_len = 1 + rng.below(8);
            out.push(pick(rng, ALNUM));
            if label_len >= 3 {
                for _ in 0..label_len - 2 {
                    out.push(pick(rng, LABEL_INNER));
                }
            }
            if label_len >= 2 {
                out.push(pick(rng, ALNUM));
            }
        }
        String::from_utf8(out).expect("generated email is ASCII")
    }

    #[test]
    fn generated_valid_emails_are_accepted() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..2_000 {
            let email_text = gen_valid(&mut rng);
            assert!(
                is_valid_email(&email_text),
                "generator produced {email_text:?} but validator rejected it"
            );
        }
    }

    #[test]
    fn every_accepted_email_has_exactly_one_at() {
        let mut rng = SplitMix64(0x0FED_CBA9_8765_4321);
        for _ in 0..2_000 {
            let email_text = gen_valid(&mut rng);
            assert_eq!(email_text.matches('@').count(), 1);
        }
    }

    #[test]
    fn appending_a_space_always_invalidates() {
        // A space is outside every character class in the grammar, so no valid
        // email survives having one appended.
        let mut rng = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..2_000 {
            let mut email_text = gen_valid(&mut rng);
            email_text.push(' ');
            assert!(
                !is_valid_email(&email_text),
                "{email_text:?} should be invalid after appending a space"
            );
        }
    }

    #[test]
    fn removing_the_at_always_invalidates() {
        // With the sole '@' removed there is no separator, so the grammar can
        // never match.
        let mut rng = SplitMix64(0x00C0_FFEE_0BAD_F00D);
        for _ in 0..2_000 {
            let email_text = gen_valid(&mut rng);
            let without_at: String = email_text.chars().filter(|&c| c != '@').collect();
            assert!(
                !is_valid_email(&without_at),
                "{without_at:?} should be invalid with no '@'"
            );
        }
    }
}
