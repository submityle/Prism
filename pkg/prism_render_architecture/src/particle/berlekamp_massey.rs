//! `Berlekamp`–`Massey` over `GF(2)`: recover the shortest `LFSR` (its linear
//! complexity and connection polynomial) that reproduces a given binary
//! sequence, used for analysing and compactly describing the pseudo-random bit
//! streams that drive particle seeding and dithering (design § deterministic
//! noise).
//!
//! The input is a sequence of bits `s[0..n]`, each stored as a `u8` whose low
//! bit is the value (`0` or `1`); any higher bits are ignored. The algorithm
//! returns the *linear complexity* `L` — the number of stages of the shortest
//! binary `LFSR` that generates the sequence — together with the connection
//! polynomial `C(x) = C[0] + C[1] x + ... + C[L] x^L` over `GF(2)`, stored as a
//! coefficient vector of length `L + 1` with the constant term `C[0] == 1`.
//! The recurrence convention is: for every `n >= L`,
//! `s[n] = XOR_{i=1..=L} ( C[i] AND s[n-i] )`.
//!
//! The core loop is the classic two-polynomial `Berlekamp`–`Massey` iteration
//! specialised to `GF(2)`. Over the two-element field the only nonzero scalar is
//! `1`, so every field multiply collapses to a logical `AND` and every field
//! add/subtract collapses to `XOR`; there is no need to track or invert a
//! discrepancy scalar. At step `i` the current polynomial `C` predicts `s[i]`
//! and the *discrepancy* `d` is the `XOR` of the prediction with the true bit.
//! When `d == 0` the current `LFSR` already explains the new bit and only the
//! shift counter `m` advances. When `d != 0` the saved earlier polynomial `b`
//! is `XOR`-ed into `C` shifted by `m` positions; if `2*L <= i` the register
//! must also grow, so `L`, the backup polynomial, and the counter are updated.
//!
//! Everything here is integer and boolean bit manipulation — `AND`, `XOR`,
//! shifts, and index arithmetic — with no floating-point, `f32`, or
//! transcendental operations, and the module depends on no other crate module.
//! `Berlekamp`–`Massey` is a structural tool: it is not a cryptographic test on
//! its own, though the linear complexity it reports is a standard randomness
//! statistic (for example in the `m-sequence` round-trip checks below).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Result of running [`berlekamp_massey`] on a binary sequence.
///
/// `linear_complexity` is the number of stages `L` of the shortest `LFSR` that
/// reproduces the input. `connection_poly` holds the connection polynomial
/// `C(x)` over `GF(2)` as coefficients `C[0..=L]` (length `L + 1`); the constant
/// term `C[0]` is always `1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BmResult {
    /// Linear complexity `L`: the stage count of the minimal `LFSR`.
    pub linear_complexity: usize,
    /// Connection polynomial coefficients `C[0..=L]` over `GF(2)`, `C[0] == 1`.
    pub connection_poly: Vec<u8>,
}

/// Run the `Berlekamp`–`Massey` algorithm over `GF(2)` on the bit sequence
/// `seq`.
///
/// Each element of `seq` contributes only its low bit (`value & 1`), so inputs
/// may be stored as plain `0`/`1` bytes. The returned [`BmResult`] gives the
/// linear complexity `L` and the connection polynomial `C[0..=L]` with
/// `C[0] == 1`, such that `s[n] = XOR_{i=1..=L} ( C[i] AND s[n-i] )` for all
/// `n >= L`. The empty sequence has complexity `0` and connection polynomial
/// `[1]`.
pub fn berlekamp_massey(seq: &[u8]) -> BmResult {
    let n = seq.len();
    if n == 0 {
        return BmResult {
            linear_complexity: 0,
            connection_poly: vec![1],
        };
    }

    // `c` is the current connection polynomial, `b` the last one that caused a
    // length change. Both start as the constant polynomial `1`.
    let mut c = vec![0u8; n];
    c[0] = 1;
    let mut b = vec![0u8; n];
    b[0] = 1;
    let mut l = 0usize;
    let mut m = 1usize;

    for i in 0..n {
        // Discrepancy between the predicted and actual bit `s[i]`.
        let mut d = seq[i] & 1;
        for j in 1..=l {
            d ^= c[j] & (seq[i - j] & 1);
        }

        if d == 0 {
            m += 1;
        } else if 2 * l <= i {
            let t = c.clone();
            for j in 0..(n - m) {
                c[j + m] ^= b[j];
            }
            l = i + 1 - l;
            b = t;
            m = 1;
        } else {
            for j in 0..(n - m) {
                c[j + m] ^= b[j];
            }
            m += 1;
        }
    }

    // Materialise `C[0..=L]`. `l` can reach `n` (e.g. the one-bit input `[1]`),
    // in which case the trailing coefficients are implicitly zero and are not
    // stored in `c`; pad them here so the polynomial always has length `L + 1`.
    let mut connection_poly = vec![0u8; l + 1];
    for (j, slot) in connection_poly.iter_mut().enumerate() {
        if j < c.len() {
            *slot = c[j];
        }
    }

    BmResult {
        linear_complexity: l,
        connection_poly,
    }
}

/// Regenerate a bit sequence from a connection polynomial and seed bits using
/// the linear recurrence `s[n] = XOR_{i=1..=L} ( C[i] AND s[n-i] )`.
///
/// `connection_poly` is `C[0..=L]` (length `L + 1`, `C[0] == 1`); `L` is taken
/// to be `connection_poly.len() - 1`. The first `L` output bits are copied from
/// `seed` (its low bits; missing seed bits are treated as `0`), and every later
/// bit is produced by the recurrence. The result has exactly `len` bits. This
/// is the inverse companion to [`berlekamp_massey`]: feeding the recovered
/// polynomial together with the first `L` bits of the original sequence must
/// reproduce that sequence.
pub fn lfsr_regenerate(connection_poly: &[u8], seed: &[u8], len: usize) -> Vec<u8> {
    let l = connection_poly.len().saturating_sub(1);
    let mut out = vec![0u8; len];
    for i in 0..len {
        if i < l {
            out[i] = if i < seed.len() { seed[i] & 1 } else { 0 };
        } else {
            let mut v = 0u8;
            for j in 1..=l {
                v ^= connection_poly[j] & (out[i - j] & 1);
            }
            out[i] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A degree-`d` primitive polynomial over `GF(2)` as connection
    /// coefficients `C[0..=d]` (`C[0] == 1`, `C[d] == 1`). Driving
    /// [`lfsr_regenerate`] with one of these and a nonzero seed yields a
    /// maximal-length `m-sequence` whose linear complexity is exactly `d`.
    fn primitive_poly(d: usize) -> Vec<u8> {
        let mut c = vec![0u8; d + 1];
        c[0] = 1;
        match d {
            1 => {
                c[1] = 1;
            }
            2 => {
                c[1] = 1;
                c[2] = 1;
            }
            3 => {
                c[1] = 1;
                c[3] = 1;
            }
            4 => {
                c[1] = 1;
                c[4] = 1;
            }
            5 => {
                c[2] = 1;
                c[5] = 1;
            }
            6 => {
                c[1] = 1;
                c[6] = 1;
            }
            7 => {
                c[1] = 1;
                c[7] = 1;
            }
            8 => {
                c[2] = 1;
                c[3] = 1;
                c[4] = 1;
                c[8] = 1;
            }
            _ => {
                c[d] = 1;
            }
        }
        c
    }

    /// Generate a maximal-length sequence of degree `d` long enough (more than
    /// one full period, well over `2*d` bits) for `Berlekamp`–`Massey` to
    /// recover its full linear complexity.
    fn gen_mseq(d: usize) -> Vec<u8> {
        let c = primitive_poly(d);
        let seed = vec![1u8; d];
        let len = (1usize << d) + d;
        lfsr_regenerate(&c, &seed, len)
    }

    /// Run `Berlekamp`–`Massey`, then regenerate from the recovered polynomial
    /// and the first `L` bits of the input; the result must equal the input.
    fn roundtrip_ok(seq: &[u8]) -> bool {
        let r = berlekamp_massey(seq);
        let l = r.linear_complexity;
        let seed = &seq[..l];
        let regen = lfsr_regenerate(&r.connection_poly, seed, seq.len());
        regen == seq
    }

    // ---- hard reference anchors -------------------------------------------

    #[test]
    fn empty_has_complexity_zero() {
        assert_eq!(berlekamp_massey(&[]).linear_complexity, 0);
    }

    #[test]
    fn empty_connection_poly_is_one() {
        assert_eq!(berlekamp_massey(&[]).connection_poly, vec![1u8]);
    }

    #[test]
    fn impulse_has_complexity_one() {
        assert_eq!(berlekamp_massey(&[1, 0, 0, 0, 0]).linear_complexity, 1);
    }

    #[test]
    fn impulse_connection_poly() {
        // s[n] = 0 for n >= 1, i.e. C = [1, 0].
        assert_eq!(
            berlekamp_massey(&[1, 0, 0, 0, 0]).connection_poly,
            vec![1u8, 0]
        );
    }

    #[test]
    fn all_ones_has_complexity_one() {
        assert_eq!(berlekamp_massey(&[1, 1, 1, 1, 1, 1]).linear_complexity, 1);
    }

    #[test]
    fn all_ones_connection_poly_is_shift() {
        // s[n] = s[n-1], i.e. C = [1, 1].
        assert_eq!(
            berlekamp_massey(&[1, 1, 1, 1, 1, 1]).connection_poly,
            vec![1u8, 1]
        );
    }

    #[test]
    fn alternating_has_complexity_two() {
        assert_eq!(
            berlekamp_massey(&[1, 0, 1, 0, 1, 0, 1, 0]).linear_complexity,
            2
        );
    }

    #[test]
    fn alternating_roundtrips() {
        assert!(roundtrip_ok(&[1, 0, 1, 0, 1, 0, 1, 0]));
    }

    #[test]
    fn all_zeros_has_complexity_zero() {
        assert_eq!(berlekamp_massey(&[0, 0, 0, 0]).linear_complexity, 0);
    }

    #[test]
    fn all_zeros_connection_poly_is_one() {
        assert_eq!(berlekamp_massey(&[0, 0, 0, 0]).connection_poly, vec![1u8]);
    }

    // ---- boundary / tiny inputs -------------------------------------------

    #[test]
    fn single_zero_has_complexity_zero() {
        assert_eq!(berlekamp_massey(&[0]).linear_complexity, 0);
    }

    #[test]
    fn single_zero_roundtrips() {
        assert!(roundtrip_ok(&[0]));
    }

    #[test]
    fn single_one_has_complexity_one() {
        assert_eq!(berlekamp_massey(&[1]).linear_complexity, 1);
    }

    #[test]
    fn single_one_roundtrips() {
        assert!(roundtrip_ok(&[1]));
    }

    #[test]
    fn single_one_connection_poly_len() {
        assert_eq!(berlekamp_massey(&[1]).connection_poly.len(), 2);
    }

    #[test]
    fn two_zeros_has_complexity_zero() {
        assert_eq!(berlekamp_massey(&[0, 0]).linear_complexity, 0);
    }

    #[test]
    fn two_ones_roundtrips() {
        assert!(roundtrip_ok(&[1, 1]));
    }

    // ---- invariants on the returned polynomial ----------------------------

    #[test]
    fn connection_poly_constant_term_is_one() {
        let cases: &[&[u8]] = &[
            &[],
            &[0],
            &[1],
            &[1, 0, 0, 0, 0],
            &[1, 1, 1, 1, 1, 1],
            &[1, 0, 1, 0, 1, 0, 1, 0],
            &[0, 0, 0, 0],
            &[1, 1, 0, 1, 0, 0, 1],
        ];
        for seq in cases {
            assert_eq!(berlekamp_massey(seq).connection_poly[0], 1);
        }
    }

    #[test]
    fn connection_poly_len_is_complexity_plus_one() {
        let cases: &[&[u8]] = &[
            &[],
            &[1],
            &[1, 0, 0, 0, 0],
            &[1, 0, 1, 0, 1, 0, 1, 0],
            &[1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        ];
        for seq in cases {
            let r = berlekamp_massey(seq);
            assert_eq!(r.connection_poly.len(), r.linear_complexity + 1);
        }
    }

    #[test]
    fn complexity_never_exceeds_length() {
        for len in 0..=12usize {
            for mask in 0..(1u32 << len) {
                let seq: Vec<u8> = (0..len).map(|i| ((mask >> i) & 1) as u8).collect();
                assert!(berlekamp_massey(&seq).linear_complexity <= len);
            }
        }
    }

    // ---- lfsr_regenerate direct behaviour ---------------------------------

    #[test]
    fn regenerate_degree_zero_is_all_zero() {
        assert_eq!(lfsr_regenerate(&[1], &[], 5), vec![0u8, 0, 0, 0, 0]);
    }

    #[test]
    fn regenerate_shift_repeats_seed_bit() {
        // C = [1, 1] => s[n] = s[n-1].
        assert_eq!(lfsr_regenerate(&[1, 1], &[1], 4), vec![1u8, 1, 1, 1]);
    }

    #[test]
    fn regenerate_period_two() {
        // C = [1, 0, 1] => s[n] = s[n-2].
        assert_eq!(
            lfsr_regenerate(&[1, 0, 1], &[1, 0], 6),
            vec![1u8, 0, 1, 0, 1, 0]
        );
    }

    #[test]
    fn regenerate_matches_fibonacci_lfsr() {
        // C = [1, 1, 1] => s[n] = s[n-1] ^ s[n-2], seed [1, 1].
        assert_eq!(
            lfsr_regenerate(&[1, 1, 1], &[1, 1], 6),
            vec![1u8, 1, 0, 1, 1, 0]
        );
    }

    #[test]
    fn regenerate_respects_output_length() {
        assert_eq!(lfsr_regenerate(&[1, 1], &[1], 0), Vec::<u8>::new());
        assert_eq!(lfsr_regenerate(&[1, 1], &[1], 1), vec![1u8]);
        assert_eq!(lfsr_regenerate(&[1, 1], &[1], 3), vec![1u8, 1, 1]);
    }

    // ---- m-sequence golden round-trips, degrees 1..=8 ---------------------

    #[test]
    fn mseq_degree_1_recovers_complexity() {
        let seq = gen_mseq(1);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 1);
        assert_eq!(r.connection_poly.len(), 2);
        assert_eq!(r.connection_poly[0], 1);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_2_recovers_complexity() {
        let seq = gen_mseq(2);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 2);
        assert_eq!(r.connection_poly.len(), 3);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_3_recovers_complexity() {
        let seq = gen_mseq(3);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 3);
        assert_eq!(r.connection_poly.len(), 4);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_4_recovers_complexity() {
        let seq = gen_mseq(4);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 4);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_5_recovers_complexity() {
        let seq = gen_mseq(5);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 5);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_6_recovers_complexity() {
        let seq = gen_mseq(6);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 6);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_7_recovers_complexity() {
        let seq = gen_mseq(7);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 7);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_degree_8_recovers_complexity() {
        let seq = gen_mseq(8);
        let r = berlekamp_massey(&seq);
        assert_eq!(r.linear_complexity, 8);
        assert!(roundtrip_ok(&seq));
    }

    #[test]
    fn mseq_recovered_poly_has_expected_degree_bit() {
        // The recovered connection polynomial must be a genuine degree-`d`
        // polynomial, so its top coefficient is set.
        for d in 1..=8usize {
            let seq = gen_mseq(d);
            let r = berlekamp_massey(&seq);
            assert_eq!(r.connection_poly.len(), d + 1);
            assert_eq!(r.connection_poly[d], 1);
            assert_eq!(r.connection_poly[0], 1);
        }
    }

    #[test]
    fn mseq_all_degrees_roundtrip() {
        for d in 1..=8usize {
            assert!(
                roundtrip_ok(&gen_mseq(d)),
                "round-trip failed at degree {d}"
            );
        }
    }

    // ---- arbitrary-sequence round-trips -----------------------------------

    #[test]
    fn arbitrary_roundtrip_a() {
        assert!(roundtrip_ok(&[1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]));
    }

    #[test]
    fn arbitrary_roundtrip_b() {
        assert!(roundtrip_ok(&[0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]));
    }

    #[test]
    fn arbitrary_roundtrip_c() {
        assert!(roundtrip_ok(&[1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1]));
    }

    #[test]
    fn arbitrary_roundtrip_d() {
        assert!(roundtrip_ok(&[
            0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0
        ]));
    }

    #[test]
    fn high_bits_ignored_in_input() {
        // Values are reduced mod 2, so 2/3 behave like 0/1.
        let masked = berlekamp_massey(&[3, 2, 3, 2, 3, 2, 3, 2]);
        let plain = berlekamp_massey(&[1, 0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(masked.linear_complexity, plain.linear_complexity);
        assert_eq!(masked.connection_poly, plain.connection_poly);
    }

    #[test]
    fn exhaustive_roundtrip_up_to_len_8() {
        for len in 0..=8usize {
            for mask in 0..(1u32 << len) {
                let seq: Vec<u8> = (0..len).map(|i| ((mask >> i) & 1) as u8).collect();
                assert!(roundtrip_ok(&seq), "round-trip failed for {seq:?}");
            }
        }
    }

    #[test]
    fn exhaustive_poly_constant_term_up_to_len_8() {
        for len in 1..=8usize {
            for mask in 0..(1u32 << len) {
                let seq: Vec<u8> = (0..len).map(|i| ((mask >> i) & 1) as u8).collect();
                assert_eq!(berlekamp_massey(&seq).connection_poly[0], 1);
            }
        }
    }

    #[test]
    fn all_zeros_various_lengths() {
        for len in 0..=10usize {
            let seq = vec![0u8; len];
            let r = berlekamp_massey(&seq);
            assert_eq!(r.linear_complexity, 0);
            assert_eq!(r.connection_poly, vec![1u8]);
            assert!(roundtrip_ok(&seq));
        }
    }

    #[test]
    fn all_ones_various_lengths() {
        for len in 1..=10usize {
            let seq = vec![1u8; len];
            let r = berlekamp_massey(&seq);
            assert_eq!(r.linear_complexity, 1);
            assert!(roundtrip_ok(&seq));
        }
    }

    #[test]
    fn alternating_various_lengths() {
        for len in 1..=12usize {
            let seq: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
            assert!(roundtrip_ok(&seq));
        }
    }

    #[test]
    fn bmresult_is_clonable_and_eq() {
        let r = berlekamp_massey(&[1, 0, 1, 0, 1, 0]);
        let c = r.clone();
        assert_eq!(r, c);
    }
}
