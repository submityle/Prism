//! Montgomery modular multiplication (`REDC`) for odd 64-bit moduli.
//!
//! This module is the particle engine's pure-integer tool for computing
//! `(a * b) mod n` without ever taking a 128-bit remainder in the hot loop.
//! Many deterministic subsystems — stateless hash RNGs, polynomial evaluation
//! over prime fields, cyclic-code arithmetic — repeatedly multiply residues
//! modulo a fixed odd modulus `n`. The schoolbook approach computes the full
//! 128-bit product and then an expensive `u128` division. Montgomery
//! multiplication replaces that division with two multiplies and a shift by
//! working in the *Montgomery domain*, where every residue `a` is represented
//! as `a * R mod n` for a fixed radix `R = 2^64`.
//!
//! # The `REDC` idea
//!
//! Division by `n` is slow, but division by `R = 2^64` is a free right shift.
//! `REDC` (Montgomery reduction) exploits this: given `t < n * R`, it returns
//! `t * R^-1 mod n` using only multiplies, an add, and a `>> 64`. The trick is
//! to add a carefully chosen multiple `m * n` of the modulus so that the low
//! 64 bits of `t + m * n` are zero; the shift then yields an exact quotient.
//!
//! Concretely, with `n_prime = -n^-1 mod 2^64` (so `n * n_prime ≡ -1 mod R`):
//!
//! ```text
//! m = (t mod R) * n_prime  mod R          // forces low word to cancel
//! u = (t + m * n) / R                      // exact; this is a >> 64
//! if u >= n { u - n } else { u }           // single conditional subtract
//! ```
//!
//! Because `t + m * n ≡ t + (t * n_prime * n) ≡ t * (1 + n_prime * n) ≡ 0
//! (mod R)`, the division by `R` is exact. The result lies in `[0, 2n)`, so one
//! conditional subtraction brings it into `[0, n)`.
//!
//! # Domain conversion
//!
//! * `to_mont(a) = REDC(a * r2)` where `r2 = R^2 mod n`, mapping `a` to
//!   `a * R mod n`.
//! * `from_mont(x) = REDC(x)`, mapping `x = a * R mod n` back to `a`.
//! * `mul_mod(a, b)` converts both inputs in, multiplies the Montgomery
//!   representatives, reduces once, and converts out, yielding `(a * b) mod n`.
//!
//! # Preconditions
//!
//! The modulus `n` must be **odd** and **greater than 1**. Oddness guarantees
//! that `n` is invertible modulo the power-of-two radix `R = 2^64`, which is
//! what makes `n_prime` exist. Even moduli have no such inverse and are
//! rejected by [`Montgomery::new`]. Inputs to [`Montgomery::mul_mod`] are
//! expected to be reduced (`a < n`, `b < n`).
//!
//! # Properties
//!
//! * No `unsafe`, no floating point: everything is `u64` / `u128` `wrapping`
//!   and checked integer arithmetic, so the result is bit-for-bit portable
//!   between a scalar `CPU` and a `GPU` integer pipeline.
//! * `mul_mod(a, b) == ((a as u128 * b as u128) % n as u128) as u64` for all
//!   `a, b < n`, verified against the schoolbook reference in the tests.

/// Number of Newton–Raphson iterations needed to invert an odd `u64` modulo
/// `2^64`. Each step doubles the number of correct low bits (Hensel lifting);
/// starting from one correct bit, six doublings reach `2^64` correct bits
/// (`1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64`).
const NEWTON_INVERSE_ITERS: usize = 6;

/// Montgomery multiplier context for a fixed odd modulus `n` with radix
/// `R = 2^64`.
///
/// Holds the modulus together with the two precomputed constants that make
/// `REDC` cheap: `n_prime = -n^-1 mod 2^64` and `r2 = R^2 mod n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery {
    /// The odd modulus `n` (`n > 1`).
    n: u64,
    /// `n_prime = -n^-1 mod 2^64`, satisfying `n * n_prime ≡ -1 mod 2^64`.
    n_prime: u64,
    /// `r2 = R^2 mod n = (2^64 mod n)^2 mod n`, used to enter the Montgomery
    /// domain.
    r2: u64,
}

impl Montgomery {
    /// Builds a Montgomery context for the odd modulus `n`.
    ///
    /// # Preconditions
    ///
    /// `n` must be odd and `n > 1`. These are enforced with `assert!`, since an
    /// even modulus has no inverse modulo `R = 2^64` and the algorithm is
    /// undefined for `n <= 1`.
    #[must_use]
    pub fn new(n: u64) -> Self {
        assert!(n > 1, "Montgomery modulus must be greater than 1");
        assert!((n & 1) == 1, "Montgomery modulus must be odd");

        // n_prime = -n^-1 mod 2^64 via Newton–Raphson (Hensel lifting).
        let mut inv: u64 = 1;
        for _ in 0..NEWTON_INVERSE_ITERS {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(inv)));
        }
        let n_prime = inv.wrapping_neg();

        // r2 = R^2 mod n = (2^64 mod n)^2 mod n, computed in u128.
        let r_mod_n = (1u128 << 64) % (n as u128);
        let r2 = ((r_mod_n * r_mod_n) % (n as u128)) as u64;

        Self { n, n_prime, r2 }
    }

    /// The modulus `n` this context was built for.
    #[must_use]
    pub fn modulus(&self) -> u64 {
        self.n
    }

    /// Montgomery reduction: returns `t * R^-1 mod n` for `t < n * R`.
    ///
    /// Implements the `REDC` algorithm described in the module docs. The low
    /// word `m = (t mod R) * n_prime mod R` is chosen so that `t + m * n` is
    /// divisible by `R = 2^64`; the division is then a `>> 64`.
    ///
    /// For moduli close to `2^64` the intermediate `t + m * n` can be up to
    /// `2 * n * R`, which does not fit in a `u128`. We therefore add with
    /// `overflowing_add` and fold the 129th-bit carry back in after the shift:
    /// if the true sum is `carry * 2^128 + sum`, then dividing by `R` gives
    /// `(carry << 64) + (sum >> 64)`. The parentheses around each shift and the
    /// `m * n` product are intentional to keep operator precedence explicit.
    #[must_use]
    fn redc(&self, t: u128) -> u64 {
        let n_u128 = self.n as u128;
        let m = (t as u64).wrapping_mul(self.n_prime);
        let m_times_n = (m as u128) * n_u128;
        let (sum, carry) = t.overflowing_add(m_times_n);
        let u = (sum >> 64) + ((carry as u128) << 64);
        if u >= n_u128 {
            (u - n_u128) as u64
        } else {
            u as u64
        }
    }

    /// Converts `a` into the Montgomery domain, returning `a * R mod n`.
    ///
    /// Computed as `REDC(a * r2)`, which maps `a` (expected `a < n`) to its
    /// Montgomery representative.
    #[must_use]
    pub fn to_mont(&self, a: u64) -> u64 {
        self.redc((a as u128) * (self.r2 as u128))
    }

    /// Converts a Montgomery representative back to an ordinary residue.
    ///
    /// Given `a_mont = a * R mod n`, returns `a mod n` via `REDC(a_mont)`.
    #[must_use]
    pub fn from_mont(&self, a_mont: u64) -> u64 {
        self.redc(a_mont as u128)
    }

    /// Returns `(a * b) mod n` for reduced inputs `a < n`, `b < n`.
    ///
    /// Enters the Montgomery domain for both operands, multiplies the
    /// representatives, reduces once, and exits the domain. Equivalent to the
    /// schoolbook `((a as u128 * b as u128) % n as u128) as u64`.
    #[must_use]
    pub fn mul_mod(&self, a: u64, b: u64) -> u64 {
        let a_mont = self.to_mont(a);
        let b_mont = self.to_mont(b);
        let prod_mont = self.redc((a_mont as u128) * (b_mont as u128));
        self.from_mont(prod_mont)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Schoolbook reference: `(a * b) mod n` via a full 128-bit remainder.
    #[cfg(test)]
    fn naive_mul_mod(a: u64, b: u64, n: u64) -> u64 {
        (((a as u128) * (b as u128)) % (n as u128)) as u64
    }

    /// Small xorshift64 generator for deterministic pseudo-random test inputs.
    #[cfg(test)]
    fn next_rand(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn new_rejects_requires_odd_bit() {
        let m = Montgomery::new(3);
        assert_eq!(m.modulus(), 3);
    }

    #[test]
    #[should_panic(expected = "odd")]
    fn new_panics_on_even_modulus() {
        let _ = Montgomery::new(4);
    }

    #[test]
    #[should_panic(expected = "greater than 1")]
    fn new_panics_on_one() {
        let _ = Montgomery::new(1);
    }

    #[test]
    #[should_panic(expected = "greater than 1")]
    fn new_panics_on_zero() {
        let _ = Montgomery::new(0);
    }

    #[test]
    fn n_prime_satisfies_minus_inverse_small_prime() {
        let m = Montgomery::new(1_000_000_007);
        // n * n_prime ≡ -1 mod 2^64, i.e. n*n_prime + 1 == 0 (mod 2^64).
        assert_eq!(m.n.wrapping_mul(m.n_prime).wrapping_add(1), 0);
    }

    #[test]
    fn n_prime_satisfies_minus_inverse_many() {
        let mut n: u64 = 3;
        let mut count = 0;
        while n < 5_000 {
            let m = Montgomery::new(n);
            assert_eq!(m.n.wrapping_mul(m.n_prime).wrapping_add(1), 0);
            n += 2;
            count += 1;
        }
        assert!(count > 100);
    }

    #[test]
    fn n_prime_minus_inverse_large_odds() {
        let moduli = [
            0xFFFF_FFFF_FFFF_FFFFu64,
            0xFFFF_FFFF_FFFF_FFFDu64,
            0x8000_0000_0000_0001u64,
            0x1234_5678_9ABC_DEF1u64,
            12_345_678_901_234_567u64,
        ];
        for &n in &moduli {
            let m = Montgomery::new(n);
            assert_eq!(m.n.wrapping_mul(m.n_prime).wrapping_add(1), 0);
        }
    }

    #[test]
    fn golden_specific_product() {
        let m = Montgomery::new(1_000_000_007);
        assert_eq!(m.mul_mod(123_456_789, 987_654_321), 259_106_859);
    }

    #[test]
    fn golden_matches_naive_for_specific() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        assert_eq!(
            m.mul_mod(123_456_789, 987_654_321),
            naive_mul_mod(123_456_789, 987_654_321, n)
        );
    }

    #[test]
    fn roundtrip_identity_small() {
        let n = 97u64;
        let m = Montgomery::new(n);
        for a in 0..n {
            assert_eq!(m.from_mont(m.to_mont(a)), a);
        }
    }

    #[test]
    fn roundtrip_identity_prime() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let samples = [0u64, 1, 2, 42, 1000, 999_999, n - 1, n / 2];
        for &a in &samples {
            assert_eq!(m.from_mont(m.to_mont(a)), a);
        }
    }

    #[test]
    fn roundtrip_identity_random() {
        let n = 0xFFFF_FFFF_FFFF_FFC5u64; // large odd modulus
        let m = Montgomery::new(n);
        let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
        for _ in 0..1_000 {
            let a = next_rand(&mut state) % n;
            assert_eq!(m.from_mont(m.to_mont(a)), a);
        }
    }

    #[test]
    fn mul_mod_zero_left() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        assert_eq!(m.mul_mod(0, 123_456), 0);
    }

    #[test]
    fn mul_mod_zero_right() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        assert_eq!(m.mul_mod(123_456, 0), 0);
    }

    #[test]
    fn mul_mod_both_zero() {
        let n = 97u64;
        let m = Montgomery::new(n);
        assert_eq!(m.mul_mod(0, 0), 0);
    }

    #[test]
    fn mul_mod_identity_one() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        for &a in &[0u64, 1, 2, 500, n - 1] {
            assert_eq!(m.mul_mod(a, 1), a);
            assert_eq!(m.mul_mod(1, a), a);
        }
    }

    #[test]
    fn mul_mod_max_minus_one_square() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let a = n - 1;
        assert_eq!(m.mul_mod(a, a), naive_mul_mod(a, a, n));
    }

    #[test]
    fn mul_mod_a_is_n_minus_one() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let a = n - 1;
        for &b in &[0u64, 1, 2, 999, 1_000_000, n - 1] {
            assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
        }
    }

    #[test]
    fn mul_mod_b_is_n_minus_one() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let b = n - 1;
        for &a in &[0u64, 1, 2, 999, 1_000_000, n - 1] {
            assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
        }
    }

    #[test]
    fn mul_mod_commutes() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0x0123_4567_89AB_CDEFu64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            assert_eq!(m.mul_mod(a, b), m.mul_mod(b, a));
        }
    }

    #[test]
    fn mul_mod_is_deterministic() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let first = m.mul_mod(123_456_789, 987_654_321);
        for _ in 0..50 {
            assert_eq!(m.mul_mod(123_456_789, 987_654_321), first);
        }
    }

    #[test]
    fn cross_check_prime_1e9_7() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0x1111_2222_3333_4444u64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
        }
    }

    #[test]
    fn cross_check_small_odd_prime() {
        let n = 97u64;
        let m = Montgomery::new(n);
        for a in 0..n {
            for b in 0..n {
                assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
            }
        }
    }

    #[test]
    fn cross_check_exhaustive_small_odd_composite() {
        let n = 99u64; // 9 * 11, odd composite
        let m = Montgomery::new(n);
        for a in 0..n {
            for b in 0..n {
                assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
            }
        }
    }

    #[test]
    fn cross_check_odd_composite_random() {
        let n = 1_000_000_009u64 * 3 + 2; // guaranteed odd composite region
        let n = if (n & 1) == 1 { n } else { n + 1 };
        let m = Montgomery::new(n);
        let mut state = 0xAAAA_5555_AAAA_5555u64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
        }
    }

    #[test]
    fn cross_check_large_prime_like_modulus() {
        let n = 0xFFFF_FFFF_FFFF_FFC5u64; // largest 64-bit prime
        let m = Montgomery::new(n);
        let mut state = 0x7F4A_7C15_9E37_79B9u64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
        }
    }

    #[test]
    fn cross_check_many_moduli() {
        let mut mod_state = 0x1357_9BDF_2468_ACE0u64;
        let mut total = 0;
        for _ in 0..30 {
            let mut n = next_rand(&mut mod_state) | 1; // force odd
            if n <= 1 {
                n = 3;
            }
            let m = Montgomery::new(n);
            let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ n;
            for _ in 0..40 {
                let a = next_rand(&mut state) % n;
                let b = next_rand(&mut state) % n;
                assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
                total += 1;
            }
        }
        assert!(total > 100);
    }

    #[test]
    fn to_mont_matches_definition() {
        let n = 97u64;
        let m = Montgomery::new(n);
        let r_mod_n = ((1u128 << 64) % (n as u128)) as u64;
        for a in 0..n {
            // Montgomery representative is a * R mod n.
            let expected = (((a as u128) * (r_mod_n as u128)) % (n as u128)) as u64;
            assert_eq!(m.to_mont(a), expected);
        }
    }

    #[test]
    fn from_mont_inverts_to_mont_composite() {
        let n = 15u64;
        let m = Montgomery::new(n);
        for a in 0..n {
            assert_eq!(m.from_mont(m.to_mont(a)), a);
        }
    }

    #[test]
    fn mul_mod_associative_sample() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0x2468_1357_9BDF_ACE0u64;
        for _ in 0..200 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            let c = next_rand(&mut state) % n;
            let left = m.mul_mod(m.mul_mod(a, b), c);
            let right = m.mul_mod(a, m.mul_mod(b, c));
            assert_eq!(left, right);
        }
    }

    #[test]
    fn mul_mod_distributes_over_naive_add() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0xFEDC_BA98_7654_3210u64;
        for _ in 0..200 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            let c = next_rand(&mut state) % n;
            let bc = ((b as u128 + c as u128) % n as u128) as u64;
            let left = m.mul_mod(a, bc);
            let right = ((m.mul_mod(a, b) as u128 + m.mul_mod(a, c) as u128) % n as u128) as u64;
            assert_eq!(left, right);
        }
    }

    #[test]
    fn result_always_reduced() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0x0F1E_2D3C_4B5A_6978u64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            let b = next_rand(&mut state) % n;
            assert!(m.mul_mod(a, b) < n);
        }
    }

    #[test]
    fn to_mont_result_reduced() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0xABCD_1234_5678_9F0Eu64;
        for _ in 0..500 {
            let a = next_rand(&mut state) % n;
            assert!(m.to_mont(a) < n);
        }
    }

    #[test]
    fn from_mont_result_reduced() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let mut state = 0x5A5A_A5A5_5A5A_A5A5u64;
        for _ in 0..500 {
            let x = next_rand(&mut state) % n;
            assert!(m.from_mont(x) < n);
        }
    }

    #[test]
    fn mul_mod_matches_naive_power_sequence() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        // Compute a^k by repeated mul_mod and compare to naive each step.
        let base = 7u64;
        let mut acc = 1u64;
        let mut naive_acc = 1u64;
        for _ in 0..64 {
            acc = m.mul_mod(acc, base);
            naive_acc = naive_mul_mod(naive_acc, base, n);
            assert_eq!(acc, naive_acc);
        }
    }

    #[test]
    fn small_moduli_three_five_seven() {
        for &n in &[3u64, 5, 7, 9, 11, 13] {
            let m = Montgomery::new(n);
            for a in 0..n {
                for b in 0..n {
                    assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
                }
            }
        }
    }

    #[test]
    fn modulus_accessor_matches_input() {
        for &n in &[3u64, 97, 1_000_000_007, 0xFFFF_FFFF_FFFF_FFC5] {
            assert_eq!(Montgomery::new(n).modulus(), n);
        }
    }

    #[test]
    fn r2_matches_definition() {
        let moduli = [3u64, 97, 1_000_000_007, 0xFFFF_FFFF_FFFF_FFC5];
        for &n in &moduli {
            let m = Montgomery::new(n);
            let r_mod_n = (1u128 << 64) % (n as u128);
            let expected = ((r_mod_n * r_mod_n) % (n as u128)) as u64;
            assert_eq!(m.r2, expected);
        }
    }

    #[test]
    fn to_from_mont_roundtrip_large_composite() {
        let n = 0xFFFF_FFFF_FFFF_FFF9u64; // odd composite-ish large value
        let n = if (n & 1) == 1 { n } else { n + 1 };
        let m = Montgomery::new(n);
        let mut state = 0x3141_5926_5358_9793u64;
        for _ in 0..1_000 {
            let a = next_rand(&mut state) % n;
            assert_eq!(m.from_mont(m.to_mont(a)), a);
        }
    }

    #[test]
    fn clone_copy_equivalence() {
        let m = Montgomery::new(1_000_000_007);
        let c = m;
        assert_eq!(m, c);
        assert_eq!(c.mul_mod(123_456_789, 987_654_321), 259_106_859);
    }

    #[test]
    fn boundary_products_near_modulus() {
        let n = 1_000_000_007u64;
        let m = Montgomery::new(n);
        let vals = [n - 1, n - 2, n - 3, 1, 2, 3];
        for &a in &vals {
            for &b in &vals {
                assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
            }
        }
    }

    #[test]
    fn stress_wide_modulus_sweep() {
        let mut mod_state = 0xC0FF_EE00_D15E_A5E5u64;
        let mut checks = 0;
        for _ in 0..20 {
            let n = next_rand(&mut mod_state) | 1;
            let n = if n > 1 { n } else { 3 };
            let m = Montgomery::new(n);
            let mut state = 0xBADC_0FFE_E0DD_F00Du64 ^ n;
            for _ in 0..25 {
                let a = next_rand(&mut state) % n;
                let b = next_rand(&mut state) % n;
                assert_eq!(m.mul_mod(a, b), naive_mul_mod(a, b, n));
                checks += 1;
            }
        }
        assert!(checks >= 100);
    }
}
