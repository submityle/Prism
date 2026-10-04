//! URL percent-encoding and decoding (RFC 3986) with form-query semantics.
//!
//! Route locations arrive as already-encoded URL strings: a space is `%20`,
//! an encoded slash inside a segment is `%2F`, and non-ASCII text is a run of
//! UTF-8 bytes each written as `%XX`. Query strings additionally follow the
//! `application/x-www-form-urlencoded` convention where `+` denotes a space.
//! This module decodes those forms back into human-readable text and encodes
//! text back into URL-safe form, so [`Location`](crate::Location) can expose
//! decoded segments while callers can build links symmetrically.
//!
//! Decoding is lossy only for byte sequences that are not valid UTF-8, which
//! are replaced with the Unicode replacement character; every valid round
//! trip is exact.

use alloc::string::String;
use alloc::vec::Vec;

/// Uppercase hex digits used when percent-encoding a byte.
const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Returns whether `byte` is an RFC 3986 *unreserved* character that never
/// needs encoding: `A`-`Z`, `a`-`z`, `0`-`9`, `-`, `.`, `_`, `~`.
fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

/// Parses a single ASCII hex digit into its value, or `None` if not hex.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Pushes the `%XX` encoding of `byte` onto `out`.
fn push_percent(out: &mut String, byte: u8) {
    out.push('%');
    out.push(HEX[(byte >> 4) as usize] as char);
    out.push(HEX[(byte & 0x0f) as usize] as char);
}

/// Decodes the raw bytes of a percent-encoded string, treating `+` as a space
/// when `plus_as_space` is set (the form-query convention).
fn decode_bytes(input: &str, plus_as_space: bool) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                    (Some(hi), Some(lo)) => {
                        out.push((hi << 4) | lo);
                        i += 3;
                    }
                    // A stray `%` not followed by two hex digits is kept
                    // verbatim, matching lenient browser behaviour.
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' if plus_as_space => {
                out.push(b' ');
                i += 1;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    out
}

/// Percent-decodes `input` per RFC 3986, assembling the decoded bytes as UTF-8.
///
/// `%XX` escapes become their byte value; all other characters are kept as-is.
/// Use this for path segments and fragments, which do not use `+` for spaces.
#[must_use]
pub fn percent_decode(input: &str) -> String {
    String::from_utf8_lossy(&decode_bytes(input, false)).into_owned()
}

/// Decodes `input` as an `application/x-www-form-urlencoded` component.
///
/// Behaves like [`percent_decode`] but additionally maps `+` to a space. Use
/// this for query-string keys and values.
#[must_use]
pub fn form_decode(input: &str) -> String {
    String::from_utf8_lossy(&decode_bytes(input, true)).into_owned()
}

/// Percent-encodes `input` so every non-unreserved byte becomes `%XX`.
///
/// Suitable for a single path segment: reserved characters such as `/` and `?`
/// are escaped so they cannot be mistaken for structural delimiters.
#[must_use]
pub fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        if is_unreserved(byte) {
            out.push(byte as char);
        } else {
            push_percent(&mut out, byte);
        }
    }
    out
}

/// Encodes `input` as an `application/x-www-form-urlencoded` component.
///
/// A space becomes `+`; every other non-unreserved byte becomes `%XX`
/// (so a literal `+` is encoded as `%2B`). The inverse of [`form_decode`].
#[must_use]
pub fn form_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        if byte == b' ' {
            out.push('+');
        } else if is_unreserved(byte) {
            out.push(byte as char);
        } else {
            push_percent(&mut out, byte);
        }
    }
    out
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::{form_decode, form_encode, percent_decode, percent_encode};
    use alloc::string::String;

    #[test]
    fn decodes_known_vectors() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("%2Fetc%2Fpasswd"), "/etc/passwd");
        assert_eq!(percent_decode("caf%C3%A9"), "café");
        assert_eq!(percent_decode("%E4%B8%AD%E6%96%87"), "中文");
    }

    #[test]
    fn form_decode_maps_plus_to_space_but_percent_decode_keeps_it() {
        assert_eq!(form_decode("a+b"), "a b");
        assert_eq!(form_decode("a%2Bb"), "a+b");
        assert_eq!(percent_decode("a+b"), "a+b");
    }

    #[test]
    fn lenient_with_malformed_escapes() {
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_decode("%2"), "%2");
    }

    #[test]
    fn encodes_reserved_but_keeps_unreserved() {
        assert_eq!(percent_encode("a b/c"), "a%20b%2Fc");
        assert_eq!(percent_encode("A-Z_a.z~0"), "A-Z_a.z~0");
        assert_eq!(form_encode("a b+c"), "a+b%2Bc");
    }

    struct Rng(u32);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    fn random_string(rng: &mut Rng) -> String {
        let len = (rng.next_u32() % 12) as usize;
        let mut s = String::new();
        for _ in 0..len {
            // Draw from a mix of ASCII specials, letters, and multi-byte chars.
            let choice = rng.next_u32() % 5;
            let ch = match choice {
                0 => char::from(b'a' + (rng.next_u32() % 26) as u8),
                1 => char::from(b'0' + (rng.next_u32() % 10) as u8),
                2 => [' ', '/', '?', '&', '=', '#', '+', '%'][(rng.next_u32() % 8) as usize],
                3 => ['é', 'ü', 'ñ', 'ø'][(rng.next_u32() % 4) as usize],
                _ => ['中', '文', '日', '本', '🚀'][(rng.next_u32() % 5) as usize],
            };
            s.push(ch);
        }
        s
    }

    #[test]
    fn percent_round_trips_arbitrary_text() {
        let mut rng = Rng(0x2024_1004);
        for _ in 0..5000 {
            let s = random_string(&mut rng);
            assert_eq!(percent_decode(&percent_encode(&s)), s);
            assert_eq!(form_decode(&form_encode(&s)), s);
        }
    }
}
