//! `Hamming` `SECDED` error-correcting codec: `Hamming(7,4)` single-error
//! correction plus the extended `Hamming(8,4)` `SECDED` (single-error-correct,
//! double-error-detect) code, built from pure-integer bit operations.
//!
//! # Scope
//!
//! This module is an **error-correcting code** (`ECC`). On encode it adds
//! redundant parity bits to a 4-bit data `nibble`; on decode it computes a
//! `syndrome` (the exclusive-or of the parity checks) to *locate and repair* a
//! single corrupted bit, and the extended form adds one overall `parity` bit so
//! that a double-bit error is *detected* rather than silently miscorrected.
//! Encoding adds redundancy, decoding removes it and heals single-bit damage.
//!
//! This is deliberately distinct from the sibling `popcount_hamming` module,
//! which is a pure *metric*: it measures the `Hamming` distance / `popcount` /
//! `parity` *between* two bit strings and never changes any data. We reuse the
//! *concept* of `Hamming` distance here (a single bit flip is a distance-one
//! corruption that this codec reverses) but we do **not** reimplement any
//! distance, weight, or `popcount` metric function. If you want to measure how
//! many bits two values differ by, use `popcount_hamming`; if you want to store
//! a `nibble` so that one flipped bit can be recovered, use this module.
//!
//! # Bit layout
//!
//! The `Hamming(7,4)` codeword uses the textbook `P1 P2 D1 P4 D2 D3 D4`
//! position order, positions `1..=7` from the least-significant end. Codeword
//! position `i` is stored in bit `i-1` of the returned `u8` (so position `1`
//! is the `LSB`); bit `7` is unused by `Hamming(7,4)` and carries the overall
//! `parity` bit of `Hamming(8,4)`.
//!
//! | bit | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` |
//! |-----|-----|-----|-----|-----|-----|-----|-----|-----|
//! | pos | `1` | `2` | `3` | `4` | `5` | `6` | `7` | ext |
//! | use | `P1`| `P2`| `D1`| `P4`| `D2`| `D3`| `D4`| `P0`|
//!
//! Parity bits cover the positions whose index has the matching power-of-two
//! bit set, so `P1` checks positions `1,3,5,7`, `P2` checks `2,3,6,7`, and `P4`
//! checks `4,5,6,7`. On decode each parity bit is recomputed and compared; the
//! three check bits read as a 3-bit number give the `syndrome`, which is `0`
//! for a clean word or the `1..=7` position of the single flipped bit.
//!
//! # Worked reference vectors
//!
//! The input `nibble`'s bit `0` is `D1`, bit `1` is `D2`, bit `2` is `D3`, and
//! bit `3` is `D4`. For `nibble = 0b1011` (`D1=1, D2=1, D3=0, D4=1`):
//! `P1 = D1^D2^D4 = 1`, `P2 = D1^D3^D4 = 0`, `P4 = D2^D3^D4 = 0`, giving the
//! codeword bits `1 0 1 0 1 0 1` (`LSB` first) `= 0b1010101 = 85`. For
//! `nibble = 0b0001` (`D1=1` only): `P1=1, P2=1, P4=0`, codeword `= 0b0000111
//! = 7`, and the overall-`parity` extension sets bit `7` (three set bits is
//! odd) to make `Hamming(8,4)` `= 0b10000111 = 135`.
//!
//! No floating-point, transcendental, or `unsafe` operations are used; every
//! step is an integer shift, mask, or exclusive-or.

/// Outcome of decoding a `Hamming(7,4)` codeword via [`hamming74_decode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hamming74Status {
    /// The `syndrome` was zero: the codeword was accepted unchanged.
    Ok,
    /// A single-bit error was located and repaired at the given codeword
    /// position (`1..=7`, where position `1` is the `LSB`).
    CorrectedBit(u8),
}

/// Outcome of decoding an extended `Hamming(8,4)` `SECDED` codeword via
/// [`hamming84_decode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecdedStatus {
    /// No error was detected.
    NoError,
    /// A single-bit error was detected and corrected.
    Corrected,
    /// A double-bit error was detected; it is reported but **not** corrected,
    /// because `SECDED` cannot locate two simultaneous flips.
    DoubleErrorDetected,
}

/// Encodes a 4-bit data `nibble` into a 7-bit `Hamming(7,4)` codeword.
///
/// Only the low four bits of `nibble` are used; any higher bits are ignored.
/// The returned `u8` holds the codeword in the `P1 P2 D1 P4 D2 D3 D4` layout
/// described in the module documentation, with bit `7` always zero.
pub fn hamming74_encode(nibble: u8) -> u8 {
    let d1 = nibble & 1;
    let d2 = (nibble >> 1) & 1;
    let d3 = (nibble >> 2) & 1;
    let d4 = (nibble >> 3) & 1;
    let p1 = d1 ^ d2 ^ d4;
    let p2 = d1 ^ d3 ^ d4;
    let p4 = d2 ^ d3 ^ d4;
    p1 | (p2 << 1) | (d1 << 2) | (p4 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6)
}

/// Decodes a 7-bit `Hamming(7,4)` codeword, correcting any single-bit error.
///
/// Returns the recovered 4-bit `nibble` together with a [`Hamming74Status`]
/// describing whether the word was clean or a bit was repaired. The syndrome
/// is computed from the three parity checks; a non-zero syndrome names the
/// `1..=7` position of the flipped bit, which is then toggled back.
pub fn hamming74_decode(code: u8) -> (u8, Hamming74Status) {
    let syndrome = hamming74_syndrome(code);
    if syndrome == 0 {
        (extract_nibble(code), Hamming74Status::Ok)
    } else {
        let corrected = code ^ error_mask(syndrome);
        (
            extract_nibble(corrected),
            Hamming74Status::CorrectedBit(syndrome),
        )
    }
}

/// Encodes a 4-bit data `nibble` into an 8-bit extended `Hamming(8,4)`
/// `SECDED` codeword.
///
/// This is the `Hamming(7,4)` codeword with one extra overall `parity` bit in
/// bit `7`, chosen so that the total number of set bits in the whole 8-bit
/// word is even. That extra bit is what lets [`hamming84_decode`] tell a
/// single error (odd parity) apart from a double error (even parity).
pub fn hamming84_encode(nibble: u8) -> u8 {
    let base = hamming74_encode(nibble);
    let parity = overall_parity(base);
    base | (parity << 7)
}

/// Decodes an 8-bit extended `Hamming(8,4)` `SECDED` codeword.
///
/// Returns the recovered 4-bit `nibble` and a [`SecdedStatus`]. The decision
/// combines the `Hamming` `syndrome` with the overall `parity` check of all
/// eight bits:
///
/// * even overall parity and zero syndrome: [`SecdedStatus::NoError`];
/// * odd overall parity: a single error, repaired, [`SecdedStatus::Corrected`]
///   (a zero syndrome means the flipped bit was the overall-parity bit itself,
///   so the data is already intact);
/// * even overall parity but non-zero syndrome: two flips,
///   [`SecdedStatus::DoubleErrorDetected`], returned uncorrected.
pub fn hamming84_decode(code: u8) -> (u8, SecdedStatus) {
    let syndrome = hamming74_syndrome(code);
    let overall = overall_parity(code);
    if overall == 1 {
        let corrected = if syndrome == 0 {
            code
        } else {
            code ^ error_mask(syndrome)
        };
        (extract_nibble(corrected), SecdedStatus::Corrected)
    } else if syndrome == 0 {
        (extract_nibble(code), SecdedStatus::NoError)
    } else {
        (extract_nibble(code), SecdedStatus::DoubleErrorDetected)
    }
}

/// Computes the 3-bit `Hamming(7,4)` `syndrome` of a codeword.
///
/// Each parity bit is recomputed as the exclusive-or of the positions it
/// covers and compared with the stored bit; the three results form a number
/// in `0..=7`. Bit `7` of `code` is ignored, so this also serves the
/// `Hamming(8,4)` decoder. A zero result means every parity check passed.
fn hamming74_syndrome(code: u8) -> u8 {
    let p1 = code & 1;
    let p2 = (code >> 1) & 1;
    let d1 = (code >> 2) & 1;
    let p4 = (code >> 3) & 1;
    let d2 = (code >> 4) & 1;
    let d3 = (code >> 5) & 1;
    let d4 = (code >> 6) & 1;
    let s1 = p1 ^ d1 ^ d2 ^ d4;
    let s2 = p2 ^ d1 ^ d3 ^ d4;
    let s4 = p4 ^ d2 ^ d3 ^ d4;
    s1 | (s2 << 1) | (s4 << 2)
}

/// Maps a non-zero `syndrome` (`1..=7`, a codeword position) to the `u8` mask
/// that toggles the corresponding bit.
///
/// Position `i` lives in bit `i-1`, so the mask is `1 << (syndrome - 1)`,
/// expressed here as `(1 << syndrome) >> 1` to stay inside pure shifts.
fn error_mask(syndrome: u8) -> u8 {
    (1u8 << syndrome) >> 1
}

/// Extracts the 4-bit data `nibble` from a (clean or already-corrected)
/// codeword by gathering the `D1 D2 D3 D4` positions.
fn extract_nibble(code: u8) -> u8 {
    let d1 = (code >> 2) & 1;
    let d2 = (code >> 4) & 1;
    let d3 = (code >> 5) & 1;
    let d4 = (code >> 6) & 1;
    d1 | (d2 << 1) | (d3 << 2) | (d4 << 3)
}

/// Returns the overall even-`parity` bit (`0` or `1`) of all eight bits of
/// `code`, folded with shifts and exclusive-or (`XOR`) rather than a count.
fn overall_parity(code: u8) -> u8 {
    let mut v = code;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    v & 1
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Hand-computed encode reference vectors ----

    #[test]
    fn encode_nibble_0000_is_zero() {
        assert_eq!(hamming74_encode(0b0000), 0);
    }

    #[test]
    fn encode_nibble_0001_reference() {
        // D1 only: P1=1, P2=1, P4=0 -> 0b0000111 = 7.
        assert_eq!(hamming74_encode(0b0001), 7);
    }

    #[test]
    fn encode_nibble_0010_reference() {
        // D2 only: P1=1, P2=0, P4=1 -> bits 0,3,4 -> 0b0011001 = 25.
        assert_eq!(hamming74_encode(0b0010), 25);
    }

    #[test]
    fn encode_nibble_0100_reference() {
        // D3 only: P1=0, P2=1, P4=1 -> bits 1,3,5 -> 0b0101010 = 42.
        assert_eq!(hamming74_encode(0b0100), 42);
    }

    #[test]
    fn encode_nibble_1000_reference() {
        // D4 only: P1=1, P2=1, P4=1 -> bits 0,1,3,6 -> 0b1001011 = 75.
        assert_eq!(hamming74_encode(0b1000), 75);
    }

    #[test]
    fn encode_nibble_1011_reference() {
        // Worked example from the module docs: 0b1010101 = 85.
        assert_eq!(hamming74_encode(0b1011), 85);
    }

    #[test]
    fn encode_nibble_1111_is_all_set() {
        // Every data bit set makes every parity bit set: 0b1111111 = 127.
        assert_eq!(hamming74_encode(0b1111), 127);
    }

    #[test]
    fn encode_ignores_high_bits_of_input() {
        for nibble in 0u8..16 {
            let high = nibble | 0b1111_0000;
            assert_eq!(hamming74_encode(high), hamming74_encode(nibble));
        }
    }

    #[test]
    fn encode_leaves_bit_seven_clear() {
        for nibble in 0u8..16 {
            assert_eq!(hamming74_encode(nibble) & 0b1000_0000, 0);
        }
    }

    // ---- Hamming(7,4) clean round-trips ----

    #[test]
    fn decode_roundtrip_all_nibbles() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            let (data, status) = hamming74_decode(code);
            assert_eq!(data, nibble);
            assert_eq!(status, Hamming74Status::Ok);
        }
    }

    #[test]
    fn decode_clean_word_reports_ok() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            let (_, status) = hamming74_decode(code);
            assert_eq!(status, Hamming74Status::Ok);
        }
    }

    #[test]
    fn valid_codewords_have_zero_syndrome() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            assert_eq!(hamming74_syndrome(code), 0);
        }
    }

    #[test]
    fn extract_nibble_of_clean_word_matches_input() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            assert_eq!(extract_nibble(code), nibble);
        }
    }

    // ---- Hamming(7,4) exhaustive single-bit correction ----

    #[test]
    fn single_flip_recovers_data_for_all_words() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            for pos in 1u8..=7 {
                let corrupted = code ^ ((1u8 << pos) >> 1);
                let (data, _) = hamming74_decode(corrupted);
                assert_eq!(data, nibble);
            }
        }
    }

    #[test]
    fn single_flip_reports_correct_position_for_all_words() {
        for nibble in 0u8..16 {
            let code = hamming74_encode(nibble);
            for pos in 1u8..=7 {
                let corrupted = code ^ ((1u8 << pos) >> 1);
                let (_, status) = hamming74_decode(corrupted);
                assert_eq!(status, Hamming74Status::CorrectedBit(pos));
            }
        }
    }

    #[test]
    fn single_flip_at_position_one() {
        let code = hamming74_encode(0b1011);
        let (data, status) = hamming74_decode(code ^ 0b0000_0001);
        assert_eq!(data, 0b1011);
        assert_eq!(status, Hamming74Status::CorrectedBit(1));
    }

    #[test]
    fn single_flip_at_position_seven() {
        let code = hamming74_encode(0b1011);
        let (data, status) = hamming74_decode(code ^ 0b0100_0000);
        assert_eq!(data, 0b1011);
        assert_eq!(status, Hamming74Status::CorrectedBit(7));
    }

    #[test]
    fn single_flip_on_zero_word() {
        for pos in 1u8..=7 {
            let corrupted = (1u8 << pos) >> 1;
            let (data, status) = hamming74_decode(corrupted);
            assert_eq!(data, 0);
            assert_eq!(status, Hamming74Status::CorrectedBit(pos));
        }
    }

    #[test]
    fn single_flip_on_all_ones_word() {
        let code = hamming74_encode(0b1111);
        for pos in 1u8..=7 {
            let corrupted = code ^ ((1u8 << pos) >> 1);
            let (data, status) = hamming74_decode(corrupted);
            assert_eq!(data, 0b1111);
            assert_eq!(status, Hamming74Status::CorrectedBit(pos));
        }
    }

    #[test]
    fn syndrome_equals_position_for_each_flip() {
        let code = hamming74_encode(0b0110);
        for pos in 1u8..=7 {
            let corrupted = code ^ ((1u8 << pos) >> 1);
            assert_eq!(hamming74_syndrome(corrupted), pos);
        }
    }

    #[test]
    fn error_mask_targets_expected_bit() {
        assert_eq!(error_mask(1), 0b0000_0001);
        assert_eq!(error_mask(4), 0b0000_1000);
        assert_eq!(error_mask(7), 0b0100_0000);
    }

    // ---- Hamming(8,4) encode / references ----

    #[test]
    fn encode84_nibble_0000_is_zero() {
        assert_eq!(hamming84_encode(0b0000), 0);
    }

    #[test]
    fn encode84_nibble_0001_reference() {
        // Base 0b0000111 has three set bits (odd) -> parity bit set: 135.
        assert_eq!(hamming84_encode(0b0001), 135);
    }

    #[test]
    fn encode84_nibble_1011_reference() {
        // Base 85 has four set bits (even) -> parity bit clear: still 85.
        assert_eq!(hamming84_encode(0b1011), 85);
    }

    #[test]
    fn encode84_nibble_1111_is_all_set() {
        // Base 127 has seven set bits (odd) -> parity bit set: 255.
        assert_eq!(hamming84_encode(0b1111), 255);
    }

    #[test]
    fn encode84_preserves_low_seven_bits() {
        for nibble in 0u8..16 {
            let code84 = hamming84_encode(nibble);
            assert_eq!(code84 & 0b0111_1111, hamming74_encode(nibble));
        }
    }

    #[test]
    fn encode84_total_parity_is_even() {
        for nibble in 0u8..16 {
            assert_eq!(overall_parity(hamming84_encode(nibble)), 0);
        }
    }

    // ---- Hamming(8,4) clean round-trips ----

    #[test]
    fn decode84_roundtrip_all_nibbles() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            let (data, status) = hamming84_decode(code);
            assert_eq!(data, nibble);
            assert_eq!(status, SecdedStatus::NoError);
        }
    }

    #[test]
    fn decode84_clean_word_reports_no_error() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            let (_, status) = hamming84_decode(code);
            assert_eq!(status, SecdedStatus::NoError);
        }
    }

    // ---- Hamming(8,4) exhaustive single-bit correction ----

    #[test]
    fn decode84_single_flip_recovers_data_everywhere() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            for bit in 0u8..8 {
                let corrupted = code ^ (1u8 << bit);
                let (data, status) = hamming84_decode(corrupted);
                assert_eq!(data, nibble);
                assert_eq!(status, SecdedStatus::Corrected);
            }
        }
    }

    #[test]
    fn decode84_single_flip_status_is_corrected() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            for bit in 0u8..8 {
                let corrupted = code ^ (1u8 << bit);
                let (_, status) = hamming84_decode(corrupted);
                assert_eq!(status, SecdedStatus::Corrected);
            }
        }
    }

    #[test]
    fn decode84_parity_bit_flip_leaves_data_intact() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            let corrupted = code ^ 0b1000_0000;
            let (data, status) = hamming84_decode(corrupted);
            assert_eq!(data, nibble);
            assert_eq!(status, SecdedStatus::Corrected);
        }
    }

    #[test]
    fn decode84_single_data_flip_specific_values() {
        let code = hamming84_encode(0b1101);
        for bit in 0u8..7 {
            let corrupted = code ^ (1u8 << bit);
            let (data, status) = hamming84_decode(corrupted);
            assert_eq!(data, 0b1101);
            assert_eq!(status, SecdedStatus::Corrected);
        }
    }

    // ---- Hamming(8,4) exhaustive double-error detection ----

    #[test]
    fn decode84_double_flip_is_always_detected() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            for a in 0u8..8 {
                for b in 0u8..8 {
                    if a < b {
                        let corrupted = code ^ (1u8 << a) ^ (1u8 << b);
                        let (_, status) = hamming84_decode(corrupted);
                        assert_eq!(status, SecdedStatus::DoubleErrorDetected);
                    }
                }
            }
        }
    }

    #[test]
    fn decode84_double_flip_is_never_miscorrected() {
        for nibble in 0u8..16 {
            let code = hamming84_encode(nibble);
            for a in 0u8..8 {
                for b in 0u8..8 {
                    if a < b {
                        let corrupted = code ^ (1u8 << a) ^ (1u8 << b);
                        let (_, status) = hamming84_decode(corrupted);
                        assert!(status != SecdedStatus::Corrected);
                        assert!(status != SecdedStatus::NoError);
                    }
                }
            }
        }
    }

    #[test]
    fn decode84_double_flip_specific_pair() {
        let code = hamming84_encode(0b1011);
        let corrupted = code ^ 0b0000_0001 ^ 0b0001_0000;
        let (_, status) = hamming84_decode(corrupted);
        assert_eq!(status, SecdedStatus::DoubleErrorDetected);
    }

    #[test]
    fn decode84_double_flip_including_parity_bit() {
        let code = hamming84_encode(0b0110);
        let corrupted = code ^ 0b0000_0100 ^ 0b1000_0000;
        let (_, status) = hamming84_decode(corrupted);
        assert_eq!(status, SecdedStatus::DoubleErrorDetected);
    }

    // ---- Overall-parity helper ----

    #[test]
    fn overall_parity_counts_odd_and_even() {
        assert_eq!(overall_parity(0b0000_0000), 0);
        assert_eq!(overall_parity(0b0000_0001), 1);
        assert_eq!(overall_parity(0b0000_0011), 0);
        assert_eq!(overall_parity(0b0000_0111), 1);
        assert_eq!(overall_parity(0b1111_1111), 0);
    }

    #[test]
    fn overall_parity_matches_single_bit_words() {
        for bit in 0u8..8 {
            assert_eq!(overall_parity(1u8 << bit), 1);
        }
    }

    // ---- Enum derives ----

    #[test]
    fn hamming74_status_equality() {
        assert_eq!(Hamming74Status::Ok, Hamming74Status::Ok);
        assert_eq!(
            Hamming74Status::CorrectedBit(3),
            Hamming74Status::CorrectedBit(3)
        );
        assert!(Hamming74Status::CorrectedBit(2) != Hamming74Status::CorrectedBit(5));
        assert!(Hamming74Status::Ok != Hamming74Status::CorrectedBit(1));
    }

    #[test]
    fn secded_status_equality() {
        assert_eq!(SecdedStatus::NoError, SecdedStatus::NoError);
        assert_eq!(SecdedStatus::Corrected, SecdedStatus::Corrected);
        assert_eq!(
            SecdedStatus::DoubleErrorDetected,
            SecdedStatus::DoubleErrorDetected
        );
        assert!(SecdedStatus::NoError != SecdedStatus::Corrected);
    }

    #[test]
    fn status_values_are_copy() {
        let s = Hamming74Status::CorrectedBit(4);
        let a = s;
        let b = s;
        assert_eq!(a, b);
    }

    #[test]
    fn hamming74_and_hamming84_agree_on_data_bits() {
        for nibble in 0u8..16 {
            let (d7, _) = hamming74_decode(hamming74_encode(nibble));
            let (d8, _) = hamming84_decode(hamming84_encode(nibble));
            assert_eq!(d7, d8);
            assert_eq!(d7, nibble);
        }
    }
}
