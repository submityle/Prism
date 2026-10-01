//! Reed-Solomon error-correcting codes over `GF(2^8)`: systematic encoding,
//! syndrome-based detection, and full Berlekamp-Massey / Chien / Forney
//! correction of up to `t = nsym / 2` symbol errors.
//!
//! A Reed-Solomon code treats a message as the high-order coefficients of a
//! polynomial over the finite field `GF(2^8)`. Encoding appends `nsym` parity
//! symbols so that the whole codeword is divisible by a fixed generator
//! polynomial `g(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^(nsym-1))`, where
//! `alpha` is the field generator `0x02`. Because the code is defined by
//! evaluation at consecutive powers of `alpha`, a received word is error-free
//! exactly when all `nsym` syndromes `S_i = C(alpha^i)` vanish, and up to
//! `t = nsym / 2` corrupted symbols can be located and repaired.
//!
//! # Relationship to [`super::galois_field_256`]
//!
//! The sibling module [`super::galois_field_256`] is the authoritative
//! `GF(2^8)` arithmetic layer, using the same primitive polynomial `0x11D`
//! (`x^8 + x^4 + x^3 + x^2 + 1`) and the same generator `0x02`. Because module
//! initialization order within the crate is not fixed, this module does **not**
//! link against that one; it instead re-implements the handful of field
//! operations it needs (`rs_gf_add` / `rs_gf_mul` / `rs_gf_pow` /
//! `rs_gf_inverse` / `rs_gf_div`) as a small self-contained copy. The numeric
//! results are identical to [`super::galois_field_256`] by construction (same
//! `0x11D`, same generator `0x02`); the `rs_`-prefixed names exist purely to
//! avoid ambiguity, not to describe a different algebra.
//!
//! # Boundaries
//!
//! * [`super::galois_field_256`] owns the low-level field algebra; this module
//!   is the polynomial coding scheme built on top of that algebra.
//! * The `crc*` modules (`crc8_variants`, `crc16_ccitt`, `crc32`) and
//!   [`super::hamming_secded`] are *different* error-handling families: `CRC`
//!   is a detection-only checksum with no correction capability, and Hamming
//!   `SECDED` corrects a single *bit* and detects two. Reed-Solomon here is a
//!   polynomial evaluation code over `GF(2^8)` that corrects whole *symbol*
//!   (byte) errors, `t = nsym / 2` of them.
//!
//! All arithmetic is integer only: no `f32`, no transcendental functions, and
//! field exponentiation is a plain integer square-and-multiply loop. The module
//! is `no_std` and uses `alloc` only for the variable-length `Vec` buffers that
//! encoding and decoding inherently require.

use alloc::vec::Vec;

/// Primitive polynomial `x^8 + x^4 + x^3 + x^2 + 1`, i.e. `0x11D` in the 9-bit
/// representation that keeps the `x^8` term.
const PRIMITIVE_POLY: u16 = 0x11D;

/// Low 8 bits of [`PRIMITIVE_POLY`] (`0x1D`), the reduction mask applied after a
/// multiply-by-`x` overflows past the `x^8` term.
const REDUCTION_BYTE: u8 = (PRIMITIVE_POLY & 0xFF) as u8;

/// Field generator (primitive element) `0x02`; its multiplicative order under
/// `0x11D` is 255, so its powers enumerate every nonzero field element.
const GENERATOR: u8 = 0x02;

/// Error returned by [`rs_correct`] when the received word cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RsError {
    /// The number of symbol errors exceeds the correction capacity
    /// `t = nsym / 2`, or the located error pattern failed re-verification.
    TooManyErrors,
}

/// Field addition in `GF(2^8)`: bitwise `XOR`. Subtraction is identical, since
/// every element is its own additive inverse (`a - b == a + b == a ^ b`).
///
/// Named as a free function (`rs_gf_add`) rather than an inherent `add` so no
/// operator-overload clippy lint applies and the field layer stays explicit.
#[must_use]
pub fn rs_gf_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Field multiplication in `GF(2^8)` via the carry-less "Russian peasant"
/// method: shift-and-conditional-`XOR` with reduction against `0x11D`.
///
/// Known values: `rs_gf_mul(2, 2) == 4`, `rs_gf_mul(0x80, 2) == 0x1D`, and
/// `rs_gf_mul(0x53, 0xCA) == 0x8F`. Pure integer arithmetic.
#[must_use]
pub fn rs_gf_mul(a: u8, b: u8) -> u8 {
    let mut result: u8 = 0;
    let mut aa: u8 = a;
    let mut bb: u8 = b;
    while bb != 0 {
        if (bb & 1) != 0 {
            result ^= aa;
        }
        let high = aa & 0x80;
        aa <<= 1;
        if high != 0 {
            aa ^= REDUCTION_BYTE;
        }
        bb >>= 1;
    }
    result
}

/// Field exponentiation `base^exp` in `GF(2^8)` via integer square-and-multiply.
///
/// `rs_gf_pow(a, 0) == 1` for every `a` (including `a == 0`), and
/// `rs_gf_pow(0, e) == 0` for `e > 0`. No floating point or `log`/`exp` is used.
#[must_use]
pub fn rs_gf_pow(base: u8, exp: u32) -> u8 {
    let mut result: u8 = 1;
    let mut b: u8 = base;
    let mut e: u32 = exp;
    while e > 0 {
        if (e & 1) == 1 {
            result = rs_gf_mul(result, b);
        }
        b = rs_gf_mul(b, b);
        e >>= 1;
    }
    result
}

/// Multiplicative inverse in `GF(2^8)` via Fermat's little theorem:
/// `a^(-1) == a^254` because the nonzero group has order 255.
///
/// By convention `rs_gf_inverse(0) == 0` (zero has no true inverse).
#[must_use]
pub fn rs_gf_inverse(a: u8) -> u8 {
    if a == 0 {
        0
    } else {
        rs_gf_pow(a, 254)
    }
}

/// Field division `a / b == a * b^(-1)` in `GF(2^8)`.
///
/// With the `rs_gf_inverse(0) == 0` convention, dividing by `0` yields `0`.
#[must_use]
pub fn rs_gf_div(a: u8, b: u8) -> u8 {
    rs_gf_mul(a, rs_gf_inverse(b))
}

/// Builds the Reed-Solomon generator polynomial for `nsym` parity symbols:
/// `g(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^(nsym-1))`, `alpha == 0x02`.
///
/// The returned coefficients are in descending-degree order with a leading
/// `1`, so the length is `nsym + 1`. For `nsym == 0` the product is empty and
/// the function returns `[1]`. For example `rs_generator_poly(2) == [1, 3, 2]`.
#[must_use]
pub fn rs_generator_poly(nsym: usize) -> Vec<u8> {
    let mut g: Vec<u8> = alloc::vec![1u8];
    for i in 0..nsym {
        let root = rs_gf_pow(GENERATOR, i as u32);
        // Multiply the running product g(x) by the monomial (x + alpha^i).
        let mut next = alloc::vec![0u8; g.len() + 1];
        for (j, &gj) in g.iter().enumerate() {
            next[j] ^= gj;
            next[j + 1] ^= rs_gf_mul(gj, root);
        }
        g = next;
    }
    g
}

/// Systematic Reed-Solomon encoding: returns `msg` followed by `nsym` parity
/// symbols.
///
/// The parity is the `GF(2^8)` polynomial remainder of `msg(x) * x^nsym`
/// divided by the generator polynomial, so the resulting codeword is divisible
/// by `g(x)` and all syndromes vanish. With `nsym == 0` the message is returned
/// unchanged.
#[must_use]
pub fn rs_encode(msg: &[u8], nsym: usize) -> Vec<u8> {
    let gen_poly = rs_generator_poly(nsym);
    // Working buffer holds msg(x) * x^nsym; synthetic division happens in place.
    let mut work = alloc::vec![0u8; msg.len() + nsym];
    work[..msg.len()].copy_from_slice(msg);
    for i in 0..msg.len() {
        let coef = work[i];
        if coef != 0 {
            for (j, &gj) in gen_poly.iter().enumerate() {
                work[i + j] ^= rs_gf_mul(gj, coef);
            }
        }
    }
    // The low `nsym` cells now hold the remainder (parity); prepend the message.
    let mut out = Vec::with_capacity(msg.len() + nsym);
    out.extend_from_slice(msg);
    out.extend_from_slice(&work[msg.len()..]);
    out
}

/// Computes the `nsym` Reed-Solomon syndromes of a codeword:
/// `S_i == C(alpha^i)` for `i` in `0..nsym`, evaluated by Horner's method.
///
/// A correctly formed codeword yields all-zero syndromes; any nonzero syndrome
/// signals at least one symbol error.
#[must_use]
pub fn rs_calc_syndromes(codeword: &[u8], nsym: usize) -> Vec<u8> {
    let mut syndromes = Vec::with_capacity(nsym);
    for i in 0..nsym {
        let x = rs_gf_pow(GENERATOR, i as u32);
        let mut acc: u8 = 0;
        for &c in codeword {
            acc = rs_gf_mul(acc, x) ^ c;
        }
        syndromes.push(acc);
    }
    syndromes
}

/// Returns `true` when every syndrome of `codeword` is zero, i.e. the word is a
/// valid codeword with no detectable errors.
///
/// With `nsym == 0` there are no parity checks and this is vacuously `true`.
#[must_use]
pub fn rs_check(codeword: &[u8], nsym: usize) -> bool {
    rs_calc_syndromes(codeword, nsym).iter().all(|&s| s == 0)
}

/// Berlekamp-Massey synthesis: returns the error-locator polynomial
/// `Lambda(x)` (ascending-degree coefficients, `Lambda[0] == 1`) together with
/// its degree `L`, the number of errors it claims to locate.
fn berlekamp_massey(syndromes: &[u8]) -> (Vec<u8>, usize) {
    let n = syndromes.len();
    let mut cur: Vec<u8> = alloc::vec![1u8];
    let mut prev: Vec<u8> = alloc::vec![1u8];
    let mut l: usize = 0;
    let mut m: usize = 1;
    let mut b: u8 = 1;
    for round in 0..n {
        // Discrepancy at this step: syndrome minus the predicted value.
        let mut delta = syndromes[round];
        for i in 1..=l {
            delta ^= rs_gf_mul(cur[i], syndromes[round - i]);
        }
        if delta == 0 {
            m += 1;
        } else if 2 * l <= round {
            let saved = cur.clone();
            let coef = rs_gf_div(delta, b);
            if cur.len() < prev.len() + m {
                cur.resize(prev.len() + m, 0);
            }
            for (i, &pi) in prev.iter().enumerate() {
                cur[i + m] ^= rs_gf_mul(coef, pi);
            }
            l = round + 1 - l;
            prev = saved;
            b = delta;
            m = 1;
        } else {
            let coef = rs_gf_div(delta, b);
            if cur.len() < prev.len() + m {
                cur.resize(prev.len() + m, 0);
            }
            for (i, &pi) in prev.iter().enumerate() {
                cur[i + m] ^= rs_gf_mul(coef, pi);
            }
            m += 1;
        }
    }
    (cur, l)
}

/// Evaluates a polynomial (ascending-degree coefficients) at `x` in `GF(2^8)`.
fn eval_poly(poly: &[u8], x: u8) -> u8 {
    let mut acc: u8 = 0;
    let mut power: u8 = 1;
    for &c in poly {
        acc ^= rs_gf_mul(c, power);
        power = rs_gf_mul(power, x);
    }
    acc
}

/// Chien search: returns the error positions (as offsets from the end of the
/// codeword) at which the locator polynomial evaluates to zero.
///
/// Position `i` corresponds to the root `alpha^(-i)`; the codeword index is
/// `n - 1 - i`.
fn chien_search(locator: &[u8], n: usize) -> Vec<usize> {
    let mut positions = Vec::new();
    for i in 0..n {
        // alpha^(-i) == alpha^(255 - i) within one full period.
        let exp = ((255 - (i % 255)) % 255) as u32;
        let x_inv = rs_gf_pow(GENERATOR, exp);
        if eval_poly(locator, x_inv) == 0 {
            positions.push(i);
        }
    }
    positions
}

/// Full Reed-Solomon decoding: locates and corrects up to `t = nsym / 2`
/// symbol errors, returning the repaired codeword.
///
/// The pipeline is syndrome computation, Berlekamp-Massey error-locator
/// synthesis, Chien search for error positions, and the Forney algorithm for
/// error magnitudes. A clean word (all-zero syndromes) is returned unchanged.
/// Returns [`RsError::TooManyErrors`] when the error count exceeds `t` or the
/// corrected word fails re-verification.
///
/// # Errors
///
/// Returns [`RsError::TooManyErrors`] if the received word carries more than
/// `t = nsym / 2` symbol errors (or an otherwise undecodable error pattern).
pub fn rs_correct(codeword: &[u8], nsym: usize) -> Result<Vec<u8>, RsError> {
    let syndromes = rs_calc_syndromes(codeword, nsym);
    if syndromes.iter().all(|&s| s == 0) {
        return Ok(codeword.to_vec());
    }
    let (locator, l) = berlekamp_massey(&syndromes);
    if l == 0 || l > nsym / 2 {
        return Err(RsError::TooManyErrors);
    }
    let n = codeword.len();
    let positions = chien_search(&locator, n);
    if positions.len() != l {
        return Err(RsError::TooManyErrors);
    }
    // Error evaluator Omega(x) = (S(x) * Lambda(x)) mod x^nsym.
    let mut omega = alloc::vec![0u8; nsym];
    for (i, &si) in syndromes.iter().enumerate() {
        for (j, &lj) in locator.iter().enumerate() {
            if i + j < nsym {
                omega[i + j] ^= rs_gf_mul(si, lj);
            }
        }
    }
    let mut out = codeword.to_vec();
    for &pos in &positions {
        let x_inv = rs_gf_pow(GENERATOR, ((255 - (pos % 255)) % 255) as u32);
        let x_pos = rs_gf_pow(GENERATOR, (pos % 255) as u32);
        let numerator = rs_gf_mul(x_pos, eval_poly(&omega, x_inv));
        // Formal derivative Lambda'(x) keeps only odd-index terms over GF(2).
        let mut denom: u8 = 0;
        let mut idx: usize = 1;
        while idx < locator.len() {
            denom ^= rs_gf_mul(locator[idx], rs_gf_pow(x_inv, (idx - 1) as u32));
            idx += 2;
        }
        if denom == 0 {
            return Err(RsError::TooManyErrors);
        }
        let magnitude = rs_gf_div(numerator, denom);
        out[n - 1 - pos] ^= magnitude;
    }
    // Re-verify: a genuine correction must drive every syndrome back to zero.
    if rs_check(&out, nsym) {
        Ok(out)
    } else {
        Err(RsError::TooManyErrors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A classic short reference message used across encode/decode tests.
    const REF_MSG: [u8; 8] = [0x40, 0xD2, 0x75, 0x47, 0x76, 0x17, 0x32, 0x06];
    /// Parity symbols produced by `rs_encode(REF_MSG, 4)`.
    const REF_PARITY_4: [u8; 4] = [0xDB, 0x6C, 0x64, 0x26];

    // ---- Field arithmetic: addition ----

    #[test]
    fn gf_add_is_xor() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(rs_gf_add(a, b), a ^ b);
            }
        }
    }

    #[test]
    fn gf_add_self_is_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_add(a, a), 0);
        }
    }

    #[test]
    fn gf_add_identity_is_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_add(a, 0), a);
        }
    }

    // ---- Field arithmetic: multiplication ----

    #[test]
    fn gf_mul_reference_two_squared() {
        assert_eq!(rs_gf_mul(2, 2), 4);
    }

    #[test]
    fn gf_mul_reference_high_bit_reduces() {
        assert_eq!(rs_gf_mul(0x80, 2), 0x1D);
    }

    #[test]
    fn gf_mul_reference_cross_vector() {
        assert_eq!(rs_gf_mul(0x53, 0xCA), 0x8F);
    }

    #[test]
    fn gf_mul_identity() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_mul(a, 1), a);
            assert_eq!(rs_gf_mul(1, a), a);
        }
    }

    #[test]
    fn gf_mul_by_zero() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_mul(a, 0), 0);
            assert_eq!(rs_gf_mul(0, a), 0);
        }
    }

    #[test]
    fn gf_mul_is_commutative() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                assert_eq!(rs_gf_mul(a, b), rs_gf_mul(b, a));
            }
        }
    }

    #[test]
    fn gf_mul_is_associative_sampled() {
        let samples = [0u8, 1, 2, 3, 0x1D, 0x53, 0x80, 0xCA, 0xFF];
        for &a in &samples {
            for &b in &samples {
                for &c in &samples {
                    let left = rs_gf_mul(rs_gf_mul(a, b), c);
                    let right = rs_gf_mul(a, rs_gf_mul(b, c));
                    assert_eq!(left, right);
                }
            }
        }
    }

    #[test]
    fn gf_mul_distributes_over_add_sampled() {
        let samples = [0u8, 1, 2, 7, 0x1D, 0x53, 0x80, 0xCA, 0xFF];
        for &a in &samples {
            for &b in &samples {
                for &c in &samples {
                    let left = rs_gf_mul(a, rs_gf_add(b, c));
                    let right = rs_gf_add(rs_gf_mul(a, b), rs_gf_mul(a, c));
                    assert_eq!(left, right);
                }
            }
        }
    }

    // ---- Field arithmetic: inverse and division ----

    #[test]
    fn gf_inverse_roundtrip_exhaustive() {
        for a in 1..=u8::MAX {
            assert_eq!(rs_gf_mul(a, rs_gf_inverse(a)), 1);
        }
    }

    #[test]
    fn gf_inverse_of_zero_is_zero() {
        assert_eq!(rs_gf_inverse(0), 0);
    }

    #[test]
    fn gf_inverse_of_one_is_one() {
        assert_eq!(rs_gf_inverse(1), 1);
    }

    #[test]
    fn gf_inverse_is_involution() {
        for a in 1..=u8::MAX {
            assert_eq!(rs_gf_inverse(rs_gf_inverse(a)), a);
        }
    }

    #[test]
    fn gf_div_equals_mul_by_inverse() {
        let samples = [0u8, 1, 2, 7, 0x1D, 0x53, 0x80, 0xCA, 0xFF];
        for &a in &samples {
            for b in 1..=u8::MAX {
                assert_eq!(rs_gf_div(a, b), rs_gf_mul(a, rs_gf_inverse(b)));
            }
        }
    }

    #[test]
    fn gf_div_self_is_one() {
        for a in 1..=u8::MAX {
            assert_eq!(rs_gf_div(a, a), 1);
        }
    }

    #[test]
    fn gf_div_by_one_is_identity() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_div(a, 1), a);
        }
    }

    // ---- Field arithmetic: exponentiation ----

    #[test]
    fn gf_pow_zero_exponent_is_one() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_pow(a, 0), 1);
        }
    }

    #[test]
    fn gf_pow_one_exponent_is_base() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_pow(a, 1), a);
        }
    }

    #[test]
    fn gf_pow_two_matches_square() {
        for a in 0..=u8::MAX {
            assert_eq!(rs_gf_pow(a, 2), rs_gf_mul(a, a));
        }
    }

    #[test]
    fn gf_pow_reference_generator_eighth() {
        assert_eq!(rs_gf_pow(2, 8), 0x1D);
    }

    #[test]
    fn gf_pow_matches_repeated_mul() {
        for a in 0..=u8::MAX {
            for exp in 0u32..=12 {
                let mut expected: u8 = 1;
                let mut k = 0u32;
                while k < exp {
                    expected = rs_gf_mul(expected, a);
                    k += 1;
                }
                assert_eq!(rs_gf_pow(a, exp), expected);
            }
        }
    }

    #[test]
    fn gf_pow_full_order_is_one() {
        for a in 1..=u8::MAX {
            assert_eq!(rs_gf_pow(a, 255), 1);
        }
    }

    #[test]
    fn gf_pow_zero_base_positive_exponent_is_zero() {
        for exp in 1u32..=16 {
            assert_eq!(rs_gf_pow(0, exp), 0);
        }
    }

    #[test]
    fn generator_enumerates_all_nonzero() {
        let mut seen = [false; 256];
        for i in 0u32..255 {
            let v = rs_gf_pow(GENERATOR, i);
            assert_ne!(v, 0);
            assert!(!seen[v as usize], "duplicate element {v}");
            seen[v as usize] = true;
        }
        for (v, &hit) in seen.iter().enumerate().skip(1) {
            assert!(hit, "missing nonzero element {v}");
        }
    }

    // ---- Generator polynomial ----

    #[test]
    fn generator_poly_nsym_zero_is_one() {
        assert_eq!(rs_generator_poly(0), alloc::vec![1u8]);
    }

    #[test]
    fn generator_poly_nsym_one() {
        // g(x) = x - alpha^0 = x + 1 -> [1, 1].
        assert_eq!(rs_generator_poly(1), alloc::vec![1u8, 1]);
    }

    #[test]
    fn generator_poly_nsym_two_reference() {
        // g(x) = (x + 1)(x + 2) = x^2 + 3x + 2.
        assert_eq!(rs_generator_poly(2), alloc::vec![1u8, 3, 2]);
    }

    #[test]
    fn generator_poly_nsym_three_reference() {
        assert_eq!(rs_generator_poly(3), alloc::vec![1u8, 7, 14, 8]);
    }

    #[test]
    fn generator_poly_nsym_four_reference() {
        assert_eq!(rs_generator_poly(4), alloc::vec![1u8, 15, 54, 120, 64]);
    }

    #[test]
    fn generator_poly_degree_matches_nsym() {
        for nsym in 0..=12 {
            assert_eq!(rs_generator_poly(nsym).len(), nsym + 1);
        }
    }

    #[test]
    fn generator_poly_is_monic() {
        for nsym in 0..=12 {
            assert_eq!(rs_generator_poly(nsym)[0], 1);
        }
    }

    #[test]
    fn generator_poly_roots_are_consecutive_powers() {
        // Every alpha^i for i in 0..nsym must be a root of g(x).
        for nsym in 1..=8 {
            let g = rs_generator_poly(nsym);
            // Evaluate g (descending coefficients) via Horner at each root.
            for i in 0..nsym {
                let root = rs_gf_pow(GENERATOR, i as u32);
                let mut acc: u8 = 0;
                for &coef in &g {
                    acc = rs_gf_mul(acc, root) ^ coef;
                }
                assert_eq!(acc, 0, "alpha^{i} not a root for nsym={nsym}");
            }
        }
    }

    // ---- Encoding ----

    #[test]
    fn encode_length_is_message_plus_parity() {
        for nsym in 0..=10 {
            let cw = rs_encode(&REF_MSG, nsym);
            assert_eq!(cw.len(), REF_MSG.len() + nsym);
        }
    }

    #[test]
    fn encode_is_systematic_prefix() {
        let cw = rs_encode(&REF_MSG, 4);
        assert_eq!(&cw[..REF_MSG.len()], &REF_MSG);
    }

    #[test]
    fn encode_reference_parity_vector() {
        let cw = rs_encode(&REF_MSG, 4);
        assert_eq!(&cw[REF_MSG.len()..], &REF_PARITY_4);
    }

    #[test]
    fn encode_nsym_zero_is_identity() {
        let cw = rs_encode(&REF_MSG, 0);
        assert_eq!(cw, REF_MSG.to_vec());
    }

    #[test]
    fn encode_empty_message_is_zero_parity() {
        let cw = rs_encode(&[], 4);
        assert_eq!(cw, alloc::vec![0u8, 0, 0, 0]);
    }

    // ---- Syndromes and detection ----

    #[test]
    fn syndromes_length_matches_nsym() {
        let cw = rs_encode(&REF_MSG, 6);
        assert_eq!(rs_calc_syndromes(&cw, 6).len(), 6);
    }

    #[test]
    fn clean_codeword_has_zero_syndromes() {
        for nsym in [2usize, 4, 6, 8, 10] {
            let cw = rs_encode(&REF_MSG, nsym);
            assert!(rs_calc_syndromes(&cw, nsym).iter().all(|&s| s == 0));
        }
    }

    #[test]
    fn rs_check_true_on_valid_codeword() {
        for nsym in [2usize, 4, 6, 8] {
            let cw = rs_encode(&REF_MSG, nsym);
            assert!(rs_check(&cw, nsym));
        }
    }

    #[test]
    fn single_symbol_error_yields_nonzero_syndrome() {
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[2] ^= 0x55;
        assert!(rs_calc_syndromes(&bad, 4).iter().any(|&s| s != 0));
    }

    #[test]
    fn single_bit_error_yields_nonzero_syndrome() {
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[0] ^= 0x01;
        assert!(rs_calc_syndromes(&bad, 4).iter().any(|&s| s != 0));
    }

    #[test]
    fn rs_check_false_on_corrupted_codeword() {
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[5] ^= 0xA3;
        assert!(!rs_check(&bad, 4));
    }

    #[test]
    fn rs_check_nsym_zero_is_vacuously_true() {
        assert!(rs_check(&REF_MSG, 0));
    }

    // ---- Correction: Berlekamp-Massey / Chien / Forney ----

    #[test]
    fn correct_clean_codeword_is_unchanged() {
        let cw = rs_encode(&REF_MSG, 4);
        assert_eq!(rs_correct(&cw, 4), Ok(cw.clone()));
    }

    #[test]
    fn correct_single_symbol_error() {
        let cw = rs_encode(&REF_MSG, 4);
        for pos in 0..cw.len() {
            let mut bad = cw.clone();
            bad[pos] ^= 0x5A;
            assert_eq!(rs_correct(&bad, 4), Ok(cw.clone()), "pos={pos}");
        }
    }

    #[test]
    fn correct_two_symbol_errors() {
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[1] ^= 0xAB;
        bad[7] ^= 0x3C;
        assert_eq!(rs_correct(&bad, 4), Ok(cw.clone()));
    }

    #[test]
    fn correct_two_errors_various_pairs() {
        let cw = rs_encode(&REF_MSG, 4);
        for i in 0..cw.len() {
            for j in (i + 1)..cw.len() {
                let mut bad = cw.clone();
                bad[i] ^= 0x11;
                bad[j] ^= 0x80;
                assert_eq!(rs_correct(&bad, 4), Ok(cw.clone()), "i={i} j={j}");
            }
        }
    }

    #[test]
    fn correct_three_symbol_errors_with_t_three() {
        let cw = rs_encode(&REF_MSG, 6);
        let mut bad = cw.clone();
        bad[0] ^= 0x7F;
        bad[4] ^= 0x12;
        bad[9] ^= 0xCC;
        assert_eq!(rs_correct(&bad, 6), Ok(cw.clone()));
    }

    #[test]
    fn correct_recovers_original_message_prefix() {
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[3] ^= 0x9E;
        bad[6] ^= 0x44;
        let fixed = rs_correct(&bad, 4).expect("decodable");
        assert_eq!(&fixed[..REF_MSG.len()], &REF_MSG);
    }

    #[test]
    fn correct_rejects_too_many_errors() {
        // t = 2 for nsym = 4, so three symbol errors is uncorrectable.
        let cw = rs_encode(&REF_MSG, 4);
        let mut bad = cw.clone();
        bad[0] ^= 0x3F;
        bad[3] ^= 0x11;
        bad[6] ^= 0x80;
        assert_eq!(rs_correct(&bad, 4), Err(RsError::TooManyErrors));
    }
}
