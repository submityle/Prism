//! `basE91` binary-to-text encoding (Joachim Henke's `basE91`), implemented
//! with pure integer bit arithmetic on the `CPU`.
//!
//! `basE91` packs an arbitrary byte stream into a stream of printable `ASCII`
//! characters drawn from a 91-symbol alphabet. Unlike fixed bit-group codecs
//! such as `base64` (which always maps 6 input bits to one output character),
//! `basE91` works on a sliding bit accumulator and emits a pair of characters
//! for roughly every 13 or 14 accumulated bits, choosing the wider 14-bit
//! group whenever the 13-bit value would overflow the usable range. This
//! variable grouping is what lets `basE91` reach an expansion ratio close to
//! the theoretical `log2(256) / log2(91)` optimum, beating `base64`'s fixed
//! `4:3` blow-up.
//!
//! ## Algorithm
//!
//! Encoding keeps a little-endian bit accumulator `b` (a [`u32`]) and a count
//! `n` of valid bits inside it. Each input byte is shifted into the top of the
//! accumulator. Whenever more than 13 bits are available, the low 13 bits are
//! examined: if that value exceeds 88 it is emitted as a 13-bit group,
//! otherwise a 14-bit group is taken instead. The chosen value `v` is written
//! as two characters, `ALPHABET[v % 91]` followed by `ALPHABET[v / 91]`. After
//! the input is exhausted any residual bits are flushed as one or two trailing
//! characters.
//!
//! Decoding inverts the process. Characters are consumed in pairs: the first
//! character of a pair seeds the group value `v`, the second adds
//! `index * 91`, and the combined value is shifted into a decode accumulator
//! from which whole bytes are drained. A lone trailing character contributes
//! the final partial byte.
//!
//! ## Boundaries
//!
//! This module is a standalone reference and shares no tables or helpers with
//! the sibling bit-group codecs (`base16`, `base32_rfc4648`, `base64`) or the
//! radix codecs (`base58`). All arithmetic is integer only — no floating point
//! and no transcendental functions — so the output is bit-reproducible on any
//! target.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// The canonical 91-symbol `basE91` alphabet: group-digit value `i`
/// (`0..=90`) maps to `ALPHABET[i]`. The order is the uppercase `ASCII`
/// letters, the lowercase `ASCII` letters, the ten digits, and finally a run
/// of punctuation ending in the double-quote character.
const ALPHABET: &[u8; 91] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~\"";

/// Sentinel stored for every byte that is not a member of [`ALPHABET`]. Valid
/// group-digit values are `0..=90`, so `-1` can never collide with a real
/// entry.
const INVALID: i8 = -1;

/// Builds the 256-entry reverse-lookup table (`byte -> basE91 digit`). Every
/// byte not present in [`ALPHABET`] maps to [`INVALID`].
const fn build_decode_lut() -> [i8; 256] {
    let mut lut = [INVALID; 256];
    let mut i = 0usize;
    while i < 91 {
        lut[ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    lut
}

/// The reverse-lookup table shared by [`base91_decode`].
const DECODE_LUT: [i8; 256] = build_decode_lut();

/// Looks up the group-digit value for one input byte, returning [`None`] for
/// any byte outside [`ALPHABET`].
fn decode_digit(byte: u8) -> Option<i32> {
    let value = DECODE_LUT[byte as usize];
    if value < 0 {
        None
    } else {
        Some(value as i32)
    }
}

/// Encodes an arbitrary byte slice into its `basE91` text representation.
///
/// The empty input maps to the empty string. The returned [`String`] contains
/// only characters from [`ALPHABET`], all of which are printable `ASCII`.
pub fn base91_encode(data: &[u8]) -> String {
    let mut out = String::new();
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in data {
        acc |= (byte as u32) << bits;
        bits += 8;
        if bits > 13 {
            let mut value = acc & 8191;
            if value > 88 {
                acc >>= 13;
                bits -= 13;
            } else {
                value = acc & 16383;
                acc >>= 14;
                bits -= 14;
            }
            out.push(char::from(ALPHABET[(value % 91) as usize]));
            out.push(char::from(ALPHABET[(value / 91) as usize]));
        }
    }

    if bits > 0 {
        out.push(char::from(ALPHABET[(acc % 91) as usize]));
        if bits > 7 || acc > 90 {
            out.push(char::from(ALPHABET[(acc / 91) as usize]));
        }
    }

    out
}

/// Decodes a `basE91` text slice back into the original bytes.
///
/// Returns [`None`] if the input contains any byte that is not a member of
/// [`ALPHABET`]. The empty input decodes to an empty vector. For any input
/// `data`, `base91_decode(base91_encode(data).as_bytes()) == Some(data)`.
pub fn base91_decode(s: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    let mut value: i32 = -1;

    for &byte in s {
        let digit = decode_digit(byte)?;
        if value < 0 {
            value = digit;
        } else {
            value += digit * 91;
            acc |= (value as u32) << bits;
            bits += if (value & 8191) > 88 { 13 } else { 14 };
            while bits >= 8 {
                out.push((acc & 0xFF) as u8);
                acc >>= 8;
                bits -= 8;
            }
            value = -1;
        }
    }

    if value != -1 {
        out.push(((acc | ((value as u32) << bits)) & 0xFF) as u8);
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic `LCG` used only to generate reproducible random
    /// round-trip inputs; it never needs cryptographic quality.
    #[cfg(test)]
    fn next_rand(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    #[test]
    fn alphabet_has_91_unique_entries() {
        assert_eq!(ALPHABET.len(), 91);
        let mut seen = [false; 256];
        for &byte in ALPHABET.iter() {
            assert!(!seen[byte as usize], "duplicate alphabet entry");
            seen[byte as usize] = true;
        }
    }

    #[test]
    fn alphabet_prefix_is_alnum() {
        assert_eq!(&ALPHABET[0..26], b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        assert_eq!(&ALPHABET[26..52], b"abcdefghijklmnopqrstuvwxyz");
        assert_eq!(&ALPHABET[52..62], b"0123456789");
    }

    #[test]
    fn alphabet_last_entry_is_double_quote() {
        assert_eq!(ALPHABET[90], b'"');
    }

    #[test]
    fn decode_lut_round_trips_every_alphabet_index() {
        for (i, &byte) in ALPHABET.iter().enumerate() {
            assert_eq!(decode_digit(byte), Some(i as i32));
        }
    }

    #[test]
    fn empty_encode_is_empty_string() {
        assert_eq!(base91_encode(b""), "");
    }

    #[test]
    fn empty_decode_is_empty_vec() {
        assert_eq!(base91_decode(b""), Some(Vec::new()));
    }

    #[test]
    fn encode_test_vector_is_fixed() {
        // Determined by this implementation with the canonical alphabet and
        // confirmed by the round-trip below.
        assert_eq!(base91_encode(b"test"), "fPNKd");
    }

    #[test]
    fn decode_test_vector_round_trips() {
        assert_eq!(base91_decode(b"fPNKd"), Some(b"test".to_vec()));
    }

    #[test]
    fn single_byte_zero_round_trips() {
        let data = [0x00u8];
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn single_byte_ff_round_trips() {
        let data = [0xFFu8];
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn single_byte_mid_round_trips() {
        let data = [0x7Fu8];
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn two_bytes_round_trip() {
        let data = [0x4D, 0x61];
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn three_bytes_round_trip() {
        let data = [0x4D, 0x61, 0x6E];
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn ascii_phrase_round_trips() {
        let data = b"The quick brown fox jumps over the lazy dog.";
        let enc = base91_encode(data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
    }

    #[test]
    fn all_zeros_round_trips() {
        for len in 0..64usize {
            let data = alloc::vec![0u8; len];
            let enc = base91_encode(&data);
            assert_eq!(base91_decode(enc.as_bytes()), Some(data));
        }
    }

    #[test]
    fn all_ff_round_trips() {
        for len in 0..64usize {
            let data = alloc::vec![0xFFu8; len];
            let enc = base91_encode(&data);
            assert_eq!(base91_decode(enc.as_bytes()), Some(data));
        }
    }

    #[test]
    fn ascending_bytes_round_trip() {
        let data: Vec<u8> = (0..=255u16).map(|v| v as u8).collect();
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data));
    }

    #[test]
    fn descending_bytes_round_trip() {
        let data: Vec<u8> = (0..=255u16).rev().map(|v| v as u8).collect();
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data));
    }

    #[test]
    fn single_byte_exhaustive_round_trip() {
        for v in 0..=255u16 {
            let data = [v as u8];
            let enc = base91_encode(&data);
            assert_eq!(
                base91_decode(enc.as_bytes()),
                Some(data.to_vec()),
                "round-trip failed for byte {v}"
            );
        }
    }

    #[test]
    fn two_byte_exhaustive_round_trip() {
        for hi in 0..=255u16 {
            for lo in (0..=255u16).step_by(17) {
                let data = [hi as u8, lo as u8];
                let enc = base91_encode(&data);
                assert_eq!(base91_decode(enc.as_bytes()), Some(data.to_vec()));
            }
        }
    }

    #[test]
    fn encoded_output_is_all_alphabet() {
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let data: Vec<u8> = (0..200).map(|_| next_rand(&mut state)).collect();
        let enc = base91_encode(&data);
        for byte in enc.bytes() {
            assert!(
                decode_digit(byte).is_some(),
                "encoder emitted non-alphabet byte"
            );
        }
    }

    #[test]
    fn decode_rejects_space() {
        assert_eq!(base91_decode(b"fP NKd"), None);
    }

    #[test]
    fn decode_rejects_high_byte() {
        assert_eq!(base91_decode(&[0x80]), None);
    }

    #[test]
    fn decode_rejects_newline() {
        assert_eq!(base91_decode(b"fPNK\nd"), None);
    }

    #[test]
    fn decode_rejects_backslash() {
        // Backslash is not part of the alphabet (the quote char is).
        assert_eq!(base91_decode(b"\\"), None);
    }

    #[test]
    fn decode_accepts_double_quote_digit() {
        // The double-quote character is alphabet index 90, so it is valid.
        assert!(base91_decode(b"\"\"").is_some());
    }

    #[test]
    fn encode_is_deterministic() {
        let data = b"determinism matters for reproducible builds";
        let first = base91_encode(data);
        let second = base91_encode(data);
        assert_eq!(first, second);
    }

    #[test]
    fn decode_is_deterministic() {
        let data = b"determinism matters for reproducible builds";
        let enc = base91_encode(data);
        let first = base91_decode(enc.as_bytes());
        let second = base91_decode(enc.as_bytes());
        assert_eq!(first, second);
    }

    #[test]
    fn random_round_trip_len_1() {
        random_round_trip_for_len(1, 0x0000_0000_0000_0001);
    }

    #[test]
    fn random_round_trip_len_2() {
        random_round_trip_for_len(2, 0x0000_0000_0000_0002);
    }

    #[test]
    fn random_round_trip_len_3() {
        random_round_trip_for_len(3, 0x0000_0000_0000_0003);
    }

    #[test]
    fn random_round_trip_len_4() {
        random_round_trip_for_len(4, 0x0000_0000_0000_0004);
    }

    #[test]
    fn random_round_trip_len_5() {
        random_round_trip_for_len(5, 0x0000_0000_0000_0005);
    }

    #[test]
    fn random_round_trip_len_7() {
        random_round_trip_for_len(7, 0x0000_0000_0000_0007);
    }

    #[test]
    fn random_round_trip_len_8() {
        random_round_trip_for_len(8, 0x0000_0000_0000_0008);
    }

    #[test]
    fn random_round_trip_len_11() {
        random_round_trip_for_len(11, 0x0000_0000_0000_000B);
    }

    #[test]
    fn random_round_trip_len_13() {
        random_round_trip_for_len(13, 0x0000_0000_0000_000D);
    }

    #[test]
    fn random_round_trip_len_16() {
        random_round_trip_for_len(16, 0x0000_0000_0000_0010);
    }

    #[test]
    fn random_round_trip_len_17() {
        random_round_trip_for_len(17, 0x0000_0000_0000_0011);
    }

    #[test]
    fn random_round_trip_len_23() {
        random_round_trip_for_len(23, 0x0000_0000_0000_0017);
    }

    #[test]
    fn random_round_trip_len_29() {
        random_round_trip_for_len(29, 0x0000_0000_0000_001D);
    }

    #[test]
    fn random_round_trip_len_31() {
        random_round_trip_for_len(31, 0x0000_0000_0000_001F);
    }

    #[test]
    fn random_round_trip_len_32() {
        random_round_trip_for_len(32, 0x0000_0000_0000_0020);
    }

    #[test]
    fn random_round_trip_len_37() {
        random_round_trip_for_len(37, 0x0000_0000_0000_0025);
    }

    #[test]
    fn random_round_trip_len_41() {
        random_round_trip_for_len(41, 0x0000_0000_0000_0029);
    }

    #[test]
    fn random_round_trip_len_48() {
        random_round_trip_for_len(48, 0x0000_0000_0000_0030);
    }

    #[test]
    fn random_round_trip_len_53() {
        random_round_trip_for_len(53, 0x0000_0000_0000_0035);
    }

    #[test]
    fn random_round_trip_len_63() {
        random_round_trip_for_len(63, 0x0000_0000_0000_003F);
    }

    #[test]
    fn random_round_trip_all_lengths_sweep() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for len in 0..64usize {
            let data: Vec<u8> = (0..len).map(|_| next_rand(&mut state)).collect();
            let enc = base91_encode(&data);
            assert_eq!(
                base91_decode(enc.as_bytes()),
                Some(data),
                "random round-trip failed at len {len}"
            );
        }
    }

    /// Helper: build `len` pseudo-random bytes from `seed` and assert the
    /// encode/decode round-trip is lossless.
    fn random_round_trip_for_len(len: usize, seed: u64) {
        let mut state = seed;
        let data: Vec<u8> = (0..len).map(|_| next_rand(&mut state)).collect();
        let enc = base91_encode(&data);
        assert_eq!(base91_decode(enc.as_bytes()), Some(data));
    }
}
