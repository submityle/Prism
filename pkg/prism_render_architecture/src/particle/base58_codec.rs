//! Base58 binary-to-text encoding using the `Bitcoin` alphabet
//! (`123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz`), implemented
//! with pure integer arbitrary-precision arithmetic on the `CPU`.
//!
//! `Base58` reinterprets an arbitrary `big-endian` byte stream as a single
//! nonnegative integer and rewrites that integer in `radix` 58, emitting one
//! printable `ASCII` character per `base58` digit. The alphabet is the
//! `Bitcoin` set: the ten digits and the fifty-two `ASCII` letters with the
//! four visually ambiguous glyphs `0` (zero), `O` (capital o), `I` (capital
//! i) and `l` (lower-case L) removed, so hand-copied or dictated identifiers
//! (addresses, content hashes, asset keys) resist transcription errors.
//!
//! ## Boundaries
//!
//! This is a deliberately separate implementation from the sibling bit-group
//! codecs in this crate — `base16` (`hex_codec`), `base32_rfc4648`, and
//! `base64` — and shares none of their tables or helpers. Those codecs slice
//! the input into fixed-width bit groups (4, 5, or 6 bits) and map each group
//! independently to one output character; encoding is a local, position-wise
//! `shift`/`mask` operation and the output length is a fixed, exact function
//! of the input length.
//!
//! `Base58` is fundamentally different: it is an arbitrary-precision `radix`
//! conversion. There is no fixed bit grouping and no padding. The whole input
//! is one `Base256` integer that is repeatedly divided by 58, collecting
//! remainders as `base58` digits (classic `big-integer` long division, done
//! here with a little-endian `Vec<u8>` as the accumulator and nothing but
//! integer `+`, `*`, `/`, and `%`). Because division mixes every byte into
//! every digit, output length is not a closed-form function of input length,
//! and a single changed input byte can change the entire output.
//!
//! Leading zero bytes need special handling precisely because they vanish
//! under integer division (a leading `0x00` contributes nothing to the
//! numeric value). `Base58` therefore follows the standard convention: each
//! leading zero byte is encoded as one explicit leading `1` character (`1` is
//! the digit for value zero), and on decode each leading `1` is restored to a
//! leading zero byte. The empty input maps to the empty string.
//!
//! All arithmetic is integer only — no floating point and no transcendental
//! functions — so this reference is bit-reproducible on any target.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// The `Bitcoin` `Base58` alphabet: digit value `i` (`0..=57`) maps to
/// `ALPHABET[i]`. The ambiguous glyphs `0`, `O`, `I`, and `l` are absent.
const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Sentinel stored for every `ASCII` byte that is not a member of the
/// alphabet. Valid `base58` digit values are `0..=57`, so `-1` can never
/// collide with a legitimate entry.
const INVALID: i8 = -1;

/// Builds the 128-entry reverse-lookup table (`ASCII byte -> base58 digit`).
/// Every `ASCII` byte not present in [`ALPHABET`] maps to [`INVALID`].
const fn build_decode_lut() -> [i8; 128] {
    let mut lut = [INVALID; 128];
    let mut i = 0usize;
    while i < 58 {
        lut[ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    lut
}

/// Reverse lookup from `ASCII` byte to `base58` digit value.
const DECODE_LUT: [i8; 128] = build_decode_lut();

/// Error returned by [`decode`] when the input is not a valid `Base58`
/// string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Base58Error {
    /// The input contained a character that is not a member of the `Bitcoin`
    /// `Base58` alphabet. The offending character is carried for diagnostics.
    InvalidChar(char),
}

/// Encodes an arbitrary `big-endian` byte slice as a `Base58` string using
/// the `Bitcoin` alphabet.
///
/// Leading zero bytes are emitted as leading `1` characters; the remaining
/// value is converted by repeated division by 58. The empty slice encodes to
/// the empty string.
pub fn encode(data: &[u8]) -> String {
    let zeros = data.iter().take_while(|&&b| b == 0).count();

    // Little-endian accumulator of base58 digits. Each input byte is folded
    // into the running Base256 number, which is simultaneously rewritten in
    // radix 58 via long division.
    let mut digits: Vec<u8> = Vec::new();
    for &byte in data {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) * 256;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut out = String::with_capacity(zeros + digits.len());
    for _ in 0..zeros {
        out.push('1');
    }
    for &d in digits.iter().rev() {
        out.push(ALPHABET[d as usize] as char);
    }
    out
}

/// Decodes a `Base58` string (`Bitcoin` alphabet) back into the original
/// `big-endian` byte slice.
///
/// Leading `1` characters are restored to leading zero bytes; the remaining
/// characters are interpreted as a `radix`-58 number and rewritten in
/// `Base256`. Any character outside the alphabet is rejected with
/// [`Base58Error::InvalidChar`]. The empty string decodes to the empty slice.
pub fn decode(s: &str) -> Result<Vec<u8>, Base58Error> {
    let zeros = s.chars().take_while(|&c| c == '1').count();

    // Little-endian accumulator of Base256 bytes, built by rewriting the
    // radix-58 number in radix 256 via long division.
    let mut result: Vec<u8> = Vec::new();
    for c in s.chars() {
        let value = decode_digit(c)?;
        let mut carry = value as u32;
        for b in result.iter_mut() {
            carry += (*b as u32) * 58;
            *b = (carry % 256) as u8;
            carry /= 256;
        }
        while carry > 0 {
            result.push((carry % 256) as u8);
            carry /= 256;
        }
    }

    let mut out: Vec<u8> = alloc::vec![0u8; zeros];
    out.reserve(result.len());
    out.extend(result.iter().rev());
    Ok(out)
}

/// Maps a single character to its `base58` digit value, rejecting any
/// character that is not a member of the `Bitcoin` alphabet.
fn decode_digit(c: char) -> Result<u8, Base58Error> {
    let code = c as u32;
    if code < 128 {
        let value = DECODE_LUT[code as usize];
        if value != INVALID {
            return Ok(value as u8);
        }
    }
    Err(Base58Error::InvalidChar(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector covering interior and trailing bytes with a two-byte
    /// leading-zero prefix (a classic published `Base58` test vector).
    const PREFIX_ZERO: [u8; 6] = [0x00, 0x00, 0x28, 0x7f, 0xb4, 0xcd];

    #[test]
    fn encode_empty_is_empty_string() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn encode_single_zero_byte_is_one() {
        assert_eq!(encode(&[0x00]), "1");
    }

    #[test]
    fn encode_two_zero_bytes_is_two_ones() {
        assert_eq!(encode(&[0x00, 0x00]), "11");
    }

    #[test]
    fn encode_three_zero_bytes_is_three_ones() {
        assert_eq!(encode(&[0x00, 0x00, 0x00]), "111");
    }

    #[test]
    fn encode_five_zero_bytes() {
        assert_eq!(encode(&[0, 0, 0, 0, 0]), "11111");
    }

    #[test]
    fn encode_reference_prefix_zero_vector() {
        assert_eq!(encode(&PREFIX_ZERO), "11233QC4");
    }

    #[test]
    fn encode_hello_world_vector() {
        assert_eq!(encode(b"Hello World!"), "2NEpo7TZRRrLZSi2U");
    }

    #[test]
    fn encode_hello_vector() {
        assert_eq!(encode(b"hello"), "Cn8eVZg");
    }

    #[test]
    fn encode_abc_vector() {
        assert_eq!(encode(b"abc"), "ZiCa");
    }

    #[test]
    fn encode_single_letter_a() {
        assert_eq!(encode(&[0x61]), "2g");
    }

    #[test]
    fn encode_zero_then_one() {
        assert_eq!(encode(&[0x00, 0x01]), "12");
    }

    #[test]
    fn encode_single_ff() {
        assert_eq!(encode(&[0xff]), "5Q");
    }

    #[test]
    fn encode_two_ff() {
        assert_eq!(encode(&[0xff, 0xff]), "LUv");
    }

    #[test]
    fn encode_deadbeef_vector() {
        assert_eq!(encode(&[0xde, 0xad, 0xbe, 0xef]), "6h8cQN");
    }

    #[test]
    fn encode_one_two_three() {
        assert_eq!(encode(&[1, 2, 3]), "Ldp");
    }

    #[test]
    fn encode_pangram_vector() {
        assert_eq!(
            encode(b"The quick brown fox jumps over the lazy dog."),
            "USm3fpXnKG5EUBx2ndxBDMPVciP5hGey2Jh4NDv6gmeo1LkMeiKrLJUUBk6Z"
        );
    }

    #[test]
    fn encode_five_byte_block_vector() {
        assert_eq!(encode(&[0x51, 0x6b, 0x6f, 0xcd, 0x0f]), "ABnLTmg");
    }

    #[test]
    fn encode_four_byte_block_vector_a() {
        assert_eq!(encode(&[0x57, 0x2e, 0x47, 0x94]), "3EFU7m");
    }

    #[test]
    fn encode_four_byte_block_vector_b() {
        assert_eq!(encode(&[0x10, 0xc8, 0x51, 0x1e]), "Rt5zm");
    }

    #[test]
    fn decode_empty_is_empty_vec() {
        assert_eq!(decode(""), Ok(Vec::new()));
    }

    #[test]
    fn decode_one_is_single_zero_byte() {
        assert_eq!(decode("1"), Ok(alloc::vec![0x00]));
    }

    #[test]
    fn decode_two_ones_is_two_zero_bytes() {
        assert_eq!(decode("11"), Ok(alloc::vec![0x00, 0x00]));
    }

    #[test]
    fn decode_reference_prefix_zero_vector() {
        assert_eq!(decode("11233QC4"), Ok(PREFIX_ZERO.to_vec()));
    }

    #[test]
    fn decode_hello_world_vector() {
        assert_eq!(decode("2NEpo7TZRRrLZSi2U"), Ok(b"Hello World!".to_vec()));
    }

    #[test]
    fn decode_abc_vector() {
        assert_eq!(decode("ZiCa"), Ok(b"abc".to_vec()));
    }

    #[test]
    fn decode_single_ff() {
        assert_eq!(decode("5Q"), Ok(alloc::vec![0xff]));
    }

    #[test]
    fn decode_deadbeef_vector() {
        assert_eq!(decode("6h8cQN"), Ok(alloc::vec![0xde, 0xad, 0xbe, 0xef]));
    }

    #[test]
    fn decode_pangram_vector_round_trips() {
        let text = b"The quick brown fox jumps over the lazy dog.";
        assert_eq!(decode(&encode(text)), Ok(text.to_vec()));
    }

    #[test]
    fn decode_rejects_digit_zero() {
        assert_eq!(decode("1110"), Err(Base58Error::InvalidChar('0')));
    }

    #[test]
    fn decode_rejects_capital_o() {
        assert_eq!(decode("1O1"), Err(Base58Error::InvalidChar('O')));
    }

    #[test]
    fn decode_rejects_capital_i() {
        assert_eq!(decode("abcI"), Err(Base58Error::InvalidChar('I')));
    }

    #[test]
    fn decode_rejects_lower_l() {
        assert_eq!(decode("xyzl"), Err(Base58Error::InvalidChar('l')));
    }

    #[test]
    fn decode_rejects_space() {
        assert_eq!(decode("ab cd"), Err(Base58Error::InvalidChar(' ')));
    }

    #[test]
    fn decode_rejects_punctuation() {
        assert_eq!(decode("Zi!a"), Err(Base58Error::InvalidChar('!')));
    }

    #[test]
    fn decode_rejects_non_ascii() {
        assert_eq!(
            decode("Zi\u{00e9}a"),
            Err(Base58Error::InvalidChar('\u{00e9}'))
        );
    }

    #[test]
    fn round_trip_every_single_byte() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            assert_eq!(decode(&encode(&byte)), Ok(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_of_ramp() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            assert_eq!(decode(&encode(slice)), Ok(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_leading_zero_runs() {
        for leading in 0..=8usize {
            let mut data: Vec<u8> = alloc::vec![0u8; leading];
            data.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
            let encoded = encode(&data);
            let ones = encoded.chars().take_while(|&c| c == '1').count();
            assert_eq!(ones, leading);
            assert_eq!(decode(&encoded), Ok(data));
        }
    }

    #[test]
    fn round_trip_lcg_random_blocks() {
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let count = (state >> 58) as usize;
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let mut inner = state;
            for _ in 0..count {
                inner = inner
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                data.push((inner >> 56) as u8);
            }
            assert_eq!(decode(&encode(&data)), Ok(data));
        }
    }

    #[test]
    fn large_ramp_block_round_trips() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        let encoded = encode(&full);
        assert_eq!(decode(&encoded), Ok(full));
    }

    #[test]
    fn large_ff_block_round_trips() {
        let data: Vec<u8> = alloc::vec![0xffu8; 128];
        assert_eq!(decode(&encode(&data)), Ok(data));
    }

    #[test]
    fn encoded_output_never_contains_ambiguous_glyphs() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        let encoded = encode(&full);
        for forbidden in ['0', 'O', 'I', 'l'] {
            assert!(!encoded.contains(forbidden));
        }
    }

    #[test]
    fn alphabet_has_fifty_eight_unique_bytes() {
        let mut seen = [false; 128];
        let mut unique = 0usize;
        for &b in ALPHABET.iter() {
            if !seen[b as usize] {
                seen[b as usize] = true;
                unique += 1;
            }
        }
        assert_eq!(unique, 58);
    }

    #[test]
    fn decode_digit_round_trips_whole_alphabet() {
        for (index, &b) in ALPHABET.iter().enumerate() {
            assert_eq!(decode_digit(b as char), Ok(index as u8));
        }
    }

    #[test]
    fn error_is_clone_and_eq() {
        let err = Base58Error::InvalidChar('0');
        assert_eq!(err.clone(), Base58Error::InvalidChar('0'));
        assert_ne!(Base58Error::InvalidChar('0'), Base58Error::InvalidChar('O'));
    }
}
