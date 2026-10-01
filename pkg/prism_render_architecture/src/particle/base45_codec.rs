//! `Base45` binary-to-text codec (RFC 9285), implemented with pure integer
//! `radix`-45 arithmetic on the `CPU`.
//!
//! `Base45` is the `QR`-code-oriented transfer encoding standardised in RFC
//! 9285. It walks the input in fixed 2-byte groups. Each full pair is read as
//! one `big-endian` 16-bit number `n` (`n = b0 * 256 + b1`, so `0..=65535`)
//! and rewritten as exactly **three** `base45` digits by repeated division by
//! 45; the digits are emitted least-significant first, so the three output
//! characters `[c, d, e]` satisfy `n = c + d * 45 + e * 2025`. A single
//! trailing byte (`0..=255`) is written as exactly **two** digits `[c, d]`
//! with `n = c + d * 45`. Two bytes therefore become three characters and a
//! lone tail byte becomes two.
//!
//! The alphabet has 45 members (`"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"`,
//! value `0..=44`): the ten digits, the twenty-six upper-case `ASCII` letters,
//! and nine symbols including `SPACE`. These characters are exactly the ones
//! encodable in a `QR` alphanumeric segment, which is why `Base45` beats
//! `base64` for `QR` payload density even though its 3-for-2 expansion is
//! numerically larger per byte.
//!
//! ## Boundaries
//!
//! This is a deliberately separate implementation from the sibling
//! binary-to-text codecs in this crate and shares none of their tables or
//! helpers:
//!
//! * `base64` ([`super::base64`]) and `base32_rfc4648`
//!   ([`super::base32_rfc4648`]) are power-of-two **bit** re-groupings (6-bit
//!   and 5-bit windows) with no `radix` arithmetic: output length is a fixed
//!   closed-form function of input length and every character maps a slice of
//!   bits via `shift`/`mask`.
//! * `base58_codec` ([`super::base58_codec`]) is an *arbitrary-precision*
//!   big-number base conversion over the whole message (long division of the
//!   entire integer), not an independent per-group transform.
//! * `ascii85_codec` ([`super::ascii85_codec`]) is a fixed 4-byte-group
//!   `radix`-85 conversion (four bytes to five characters) with a `'z'`
//!   zero-compression sentinel and optional `<~`/`~>` delimiters.
//!
//! Only `Base45` uses the RFC 9285 grouping of *2 bytes to 3 characters* (and
//! *1 byte to 2 characters* for the tail) over `radix` 45 with the
//! least-significant-digit-first ordering above. That group structure, the
//! 45-member `QR` alphabet, and the digit order are the defining differences
//! from every neighbour.
//!
//! All arithmetic here is integer `+`, `*`, `/`, and `%` (plus `big-endian`
//! byte shuffling) — no floating point and no transcendental functions — so
//! the reference is bit-reproducible on any target.

use alloc::string::String;
use alloc::vec::Vec;

/// The RFC 9285 `Base45` alphabet: digit value `i` (`0..=44`) maps to
/// `ALPHABET[i]`.
const ALPHABET: &[u8; 45] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

/// Sentinel stored for every `ASCII` byte that is not a member of the
/// alphabet. Valid `base45` digit values are `0..=44`, so `-1` can never
/// collide with a legitimate entry.
const INVALID: i8 = -1;

/// Largest 16-bit group value a 3-character group may represent.
const MAX_PAIR: u32 = 65535;

/// Largest single-byte value a 2-character tail group may represent.
const MAX_SINGLE: u32 = 255;

/// Builds the 128-entry reverse-lookup table (`ASCII byte -> base45 digit`).
/// Every `ASCII` byte not present in [`ALPHABET`] maps to [`INVALID`].
const fn build_decode_lut() -> [i8; 128] {
    let mut lut = [INVALID; 128];
    let mut i = 0usize;
    while i < 45 {
        lut[ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    lut
}

/// Reverse lookup from `ASCII` byte to `base45` digit value.
const DECODE_LUT: [i8; 128] = build_decode_lut();

/// Error returned by [`decode`] when the input is not a valid `Base45` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base45Error {
    /// The input contained a character that is not a member of the `Base45`
    /// alphabet. The offending character is carried for diagnostics.
    InvalidChar(char),
    /// The input length is congruent to 1 modulo 3, which can never be a
    /// valid sequence of 3-character and tail 2-character groups.
    InvalidLength,
    /// A 3-character group decoded to a value greater than `65535`, or a
    /// 2-character tail group decoded to a value greater than `255`.
    Overflow,
}

/// Encodes an arbitrary byte slice into a `Base45` string per RFC 9285.
///
/// Full 2-byte groups become three characters (`n = c + d * 45 + e * 2025`,
/// least-significant digit first) and a lone trailing byte becomes two
/// characters (`n = c + d * 45`). The empty slice encodes to the empty string.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::new();
    for chunk in data.chunks(2) {
        if chunk.len() == 2 {
            let n = u32::from(chunk[0]) * 256 + u32::from(chunk[1]);
            let c = n % 45;
            let rest = n / 45;
            let d = rest % 45;
            let e = rest / 45;
            out.push(ALPHABET[c as usize] as char);
            out.push(ALPHABET[d as usize] as char);
            out.push(ALPHABET[e as usize] as char);
        } else {
            let n = u32::from(chunk[0]);
            let c = n % 45;
            let d = n / 45;
            out.push(ALPHABET[c as usize] as char);
            out.push(ALPHABET[d as usize] as char);
        }
    }
    out
}

/// Decodes a `Base45` string (RFC 9285) back into the original byte vector.
///
/// Characters are mapped to `base45` digit values, grouped into 3-character
/// (and a possible trailing 2-character) blocks, and converted with
/// `n = c + d * 45 + e * 2025` (or `n = c + d * 45` for the tail). The empty
/// string decodes to the empty vector.
///
/// # Errors
///
/// Returns [`Base45Error::InvalidChar`] for any non-alphabet character,
/// [`Base45Error::InvalidLength`] when the length is `1` modulo `3`, and
/// [`Base45Error::Overflow`] when a group exceeds its representable range
/// (`65535` for a full group, `255` for a tail group).
pub fn decode(text: &str) -> Result<Vec<u8>, Base45Error> {
    let mut digits: Vec<u8> = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let code = ch as u32;
        if code >= 128 {
            return Err(Base45Error::InvalidChar(ch));
        }
        let value = DECODE_LUT[code as usize];
        if value == INVALID {
            return Err(Base45Error::InvalidChar(ch));
        }
        digits.push(value as u8);
    }

    if digits.len() % 3 == 1 {
        return Err(Base45Error::InvalidLength);
    }

    let mut out: Vec<u8> = Vec::with_capacity(digits.len() / 3 * 2 + 1);
    for group in digits.chunks(3) {
        if group.len() == 3 {
            let n = u32::from(group[0]) + u32::from(group[1]) * 45 + u32::from(group[2]) * 2025;
            if n > MAX_PAIR {
                return Err(Base45Error::Overflow);
            }
            out.push((n >> 8) as u8);
            out.push((n & 0xFF) as u8);
        } else {
            // `group.len() == 2`; a length of 1 was rejected as `InvalidLength`.
            let n = u32::from(group[0]) + u32::from(group[1]) * 45;
            if n > MAX_SINGLE {
                return Err(Base45Error::Overflow);
            }
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Small deterministic linear-congruential generator for property tests
    /// (keeps the test suite self-contained, no external crates).
    fn next_rand(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    // --- Authoritative RFC 9285 reference vectors (encode) -------------------

    #[test]
    fn encode_ab_reference_vector() {
        assert_eq!(encode(b"AB"), "BB8");
    }

    #[test]
    fn encode_hello_reference_vector() {
        assert_eq!(encode(b"Hello!!"), "%69 VD92EX0");
    }

    #[test]
    fn encode_base45_reference_vector() {
        assert_eq!(encode(b"base-45"), "UJCLQE7W581");
    }

    #[test]
    fn encode_ietf_reference_vector() {
        assert_eq!(encode(b"ietf!"), "QED8WEX0");
    }

    // --- Authoritative RFC 9285 reference vectors (decode) -------------------

    #[test]
    fn decode_ab_reference_vector() {
        assert_eq!(decode("BB8").unwrap(), b"AB".to_vec());
    }

    #[test]
    fn decode_hello_reference_vector() {
        assert_eq!(decode("%69 VD92EX0").unwrap(), b"Hello!!".to_vec());
    }

    #[test]
    fn decode_base45_reference_vector() {
        assert_eq!(decode("UJCLQE7W581").unwrap(), b"base-45".to_vec());
    }

    #[test]
    fn decode_ietf_reference_vector() {
        assert_eq!(decode("QED8WEX0").unwrap(), b"ietf!".to_vec());
    }

    // --- Empty input ---------------------------------------------------------

    #[test]
    fn encode_empty_is_empty() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn decode_empty_is_empty() {
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    // --- Single byte / tail behaviour ----------------------------------------

    #[test]
    fn encode_single_zero_byte() {
        // n = 0 -> c = 0 ('0'), d = 0 ('0').
        assert_eq!(encode(&[0]), "00");
    }

    #[test]
    fn encode_single_max_byte() {
        // n = 255 -> c = 30 ('U'), d = 5 ('5').
        assert_eq!(encode(&[255]), "U5");
    }

    #[test]
    fn decode_single_byte_pair_zero() {
        assert_eq!(decode("00").unwrap(), vec![0u8]);
    }

    #[test]
    fn decode_single_byte_max() {
        // n = 30 + 5 * 45 = 255.
        assert_eq!(decode("U5").unwrap(), vec![255u8]);
    }

    #[test]
    fn decode_max_full_group() {
        // "FGW" = 15 + 16 * 45 + 32 * 2025 = 65535 -> [0xFF, 0xFF].
        assert_eq!(decode("FGW").unwrap(), vec![0xFFu8, 0xFF]);
    }

    #[test]
    fn encode_max_full_group() {
        assert_eq!(encode(&[0xFF, 0xFF]), "FGW");
    }

    #[test]
    fn encode_two_zero_bytes() {
        // n = 0 -> "000".
        assert_eq!(encode(&[0, 0]), "000");
    }

    // --- Length / shape properties -------------------------------------------

    #[test]
    fn encode_even_length_is_triple_pairs() {
        let data = [1u8, 2, 3, 4, 5, 6];
        assert_eq!(encode(&data).len(), data.len() / 2 * 3);
    }

    #[test]
    fn encode_odd_length_has_two_char_tail() {
        let data = [1u8, 2, 3, 4, 5];
        // Two full pairs (6 chars) + one tail byte (2 chars) = 8.
        assert_eq!(encode(&data).len(), 8);
    }

    #[test]
    fn encode_emits_only_alphabet_chars() {
        let data: Vec<u8> = (0..=255u8).collect();
        let enc = encode(&data);
        for ch in enc.bytes() {
            assert!(ALPHABET.contains(&ch));
        }
    }

    // --- Invalid characters --------------------------------------------------

    #[test]
    fn decode_rejects_lowercase_letter() {
        // Lower-case letters are not part of the `Base45` alphabet.
        assert_eq!(decode("ab").unwrap_err(), Base45Error::InvalidChar('a'));
    }

    #[test]
    fn decode_rejects_exclamation() {
        assert_eq!(decode("!!").unwrap_err(), Base45Error::InvalidChar('!'));
    }

    #[test]
    fn decode_rejects_tab_char() {
        assert_eq!(decode("\t\t").unwrap_err(), Base45Error::InvalidChar('\t'));
    }

    #[test]
    fn decode_rejects_non_ascii_char() {
        assert_eq!(decode("é0").unwrap_err(), Base45Error::InvalidChar('é'));
    }

    #[test]
    fn decode_rejects_invalid_char_in_second_position() {
        assert_eq!(decode("0a0").unwrap_err(), Base45Error::InvalidChar('a'));
    }

    #[test]
    fn decode_accepts_space_in_alphabet() {
        // The `Hello!!` vector embeds a SPACE (alphabet index 36); make sure a
        // crafted group containing it decodes cleanly.
        assert_eq!(decode(" 00").unwrap().len(), 2);
    }

    // --- Invalid length (len % 3 == 1) ---------------------------------------

    #[test]
    fn decode_rejects_length_one() {
        assert_eq!(decode("Z").unwrap_err(), Base45Error::InvalidLength);
    }

    #[test]
    fn decode_rejects_length_four() {
        assert_eq!(decode("0000").unwrap_err(), Base45Error::InvalidLength);
    }

    #[test]
    fn decode_rejects_length_seven() {
        assert_eq!(decode("0000000").unwrap_err(), Base45Error::InvalidLength);
    }

    #[test]
    fn decode_accepts_length_two_and_three() {
        assert!(decode("00").is_ok());
        assert!(decode("000").is_ok());
    }

    // --- Overflow ------------------------------------------------------------

    #[test]
    fn decode_overflow_zzz() {
        // "ZZZ" = 35 + 35 * 45 + 35 * 2025 = 72485 > 65535.
        assert_eq!(decode("ZZZ").unwrap_err(), Base45Error::Overflow);
    }

    #[test]
    fn decode_overflow_just_above_max_full_group() {
        // "GGW" = 16 + 16 * 45 + 32 * 2025 = 65536, one above the max pair.
        assert_eq!(decode("GGW").unwrap_err(), Base45Error::Overflow);
    }

    #[test]
    fn decode_no_overflow_at_exact_max_full_group() {
        assert!(decode("FGW").is_ok());
    }

    #[test]
    fn decode_overflow_two_char_tail() {
        // "V5" = 31 + 5 * 45 = 256 > 255.
        assert_eq!(decode("V5").unwrap_err(), Base45Error::Overflow);
    }

    #[test]
    fn decode_overflow_two_char_tail_large() {
        // "0:" = 0 + 44 * 45 = 1980 > 255.
        assert_eq!(decode("0:").unwrap_err(), Base45Error::Overflow);
    }

    #[test]
    fn decode_no_overflow_at_exact_max_tail() {
        // "U5" = 255, the largest valid tail value.
        assert!(decode("U5").is_ok());
    }

    // --- Alphabet integrity / index mapping ----------------------------------

    #[test]
    fn alphabet_has_45_members() {
        assert_eq!(ALPHABET.len(), 45);
    }

    #[test]
    fn alphabet_is_unique() {
        for (i, &a) in ALPHABET.iter().enumerate() {
            for &b in &ALPHABET[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn alphabet_index_mapping_is_correct() {
        assert_eq!(ALPHABET[0], b'0');
        assert_eq!(ALPHABET[9], b'9');
        assert_eq!(ALPHABET[10], b'A');
        assert_eq!(ALPHABET[35], b'Z');
        assert_eq!(ALPHABET[36], b' ');
        assert_eq!(ALPHABET[37], b'$');
        assert_eq!(ALPHABET[38], b'%');
        assert_eq!(ALPHABET[39], b'*');
        assert_eq!(ALPHABET[40], b'+');
        assert_eq!(ALPHABET[41], b'-');
        assert_eq!(ALPHABET[42], b'.');
        assert_eq!(ALPHABET[43], b'/');
        assert_eq!(ALPHABET[44], b':');
    }

    #[test]
    fn decode_lut_round_trips_every_alphabet_entry() {
        for (i, &ch) in ALPHABET.iter().enumerate() {
            assert_eq!(DECODE_LUT[ch as usize], i as i8);
        }
    }

    #[test]
    fn decode_lut_marks_non_members_invalid() {
        // A few bytes that are definitely not in the alphabet.
        assert_eq!(DECODE_LUT[b'a' as usize], INVALID);
        assert_eq!(DECODE_LUT[b'!' as usize], INVALID);
        assert_eq!(DECODE_LUT[b'~' as usize], INVALID);
        assert_eq!(DECODE_LUT[0], INVALID);
    }

    // --- Round-trip coverage -------------------------------------------------

    #[test]
    fn roundtrip_every_single_byte() {
        for b in 0u8..=255 {
            let data = [b];
            assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
        }
    }

    #[test]
    fn roundtrip_every_two_byte_pair_sample() {
        // Sweep the full 16-bit space in strides to keep the test quick while
        // still exercising every residue class.
        let mut n = 0u32;
        while n <= MAX_PAIR {
            let data = [(n >> 8) as u8, (n & 0xFF) as u8];
            assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
            n += 97;
        }
    }

    #[test]
    fn roundtrip_all_lengths_0_to_256() {
        for len in 0..=256usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let enc = encode(&data);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn roundtrip_random_blocks_mixed_lengths() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..64 {
            let len = (next_rand(&mut state) as usize) % 300;
            let data: Vec<u8> = (0..len).map(|_| next_rand(&mut state)).collect();
            let enc = encode(&data);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn roundtrip_descending_byte_pattern() {
        let data: Vec<u8> = (0..=255u8).rev().collect();
        assert_eq!(decode(&encode(&data)).unwrap(), data);
    }

    #[test]
    fn roundtrip_all_zero_block() {
        let data = vec![0u8; 64];
        let enc = encode(&data);
        assert_eq!(enc, "0".repeat(96));
        assert_eq!(decode(&enc).unwrap(), data);
    }

    #[test]
    fn roundtrip_text_bytes() {
        let data = b"The quick brown fox jumps over the lazy dog.";
        assert_eq!(decode(&encode(data)).unwrap(), data.to_vec());
    }

    #[test]
    fn encode_decode_odd_length_tail() {
        // 5 bytes -> two pairs + one tail byte; verify shape and round-trip.
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0x42];
        let enc = encode(&data);
        assert_eq!(enc.len(), 8);
        assert_eq!(decode(&enc).unwrap(), data.to_vec());
    }
}
