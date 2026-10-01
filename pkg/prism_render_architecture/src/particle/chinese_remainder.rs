//! Chinese Remainder Theorem (`CRT`) solver using pairwise merging that
//! supports non-coprime moduli.
//!
//! The classical `CRT` reconstructs a single residue modulo the product of
//! pairwise-coprime moduli. This module implements the more general pairwise
//! merge: two congruences `x ≡ r1 (mod m1)` and `x ≡ r2 (mod m2)` are combined
//! into one congruence modulo `lcm(m1, m2)` whenever a solution exists, and a
//! list of congruences is reduced by folding this merge left to right. A
//! solution exists for a pair exactly when `r2 - r1` is divisible by
//! `gcd(m1, m2)`; otherwise the system is inconsistent and no `x` satisfies
//! both congruences.
//!
//! The merge is driven by the extended Euclidean algorithm, which yields the
//! `Bézout` coefficients `p, q` with `m1 * p + m2 * q = gcd(m1, m2)`. Those
//! coefficients place the combined solution onto the shared lattice spaced by
//! `lcm(m1, m2)`.
//!
//! ## Numerics
//!
//! Inputs (residues and moduli) are supplied as `u64`. Every intermediate
//! quantity is widened to `i128` so that the signed `Bézout` coefficients, the
//! difference `r2 - r1`, and the products that appear during the merge do not
//! overflow or lose sign, and the final result is regularized into the
//! half-open range `[0, lcm)` before being narrowed back to `u64`. No
//! floating-point arithmetic is used anywhere; the reference is integer-exact.
//!
//! ## Boundaries
//!
//! This solver is distinct from sibling number-theoretic helpers: it does not
//! perform modular exponentiation or primality testing, and it does not assume
//! coprime moduli the way a product-form `CRT` would. The empty-input case for
//! [`crt`] deliberately returns `None`, because there is no modulus to reduce
//! against and therefore no well-defined representative residue.

/// Extended Euclidean algorithm.
///
/// Returns `(g, x, y)` such that `a * x + b * y = g`, where `g = gcd(a, b)`.
/// The computation is a straight integer recurrence over `i128`, mirroring the
/// iterative form used throughout this crate.
fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i128, 0i128);
    let (mut old_t, mut t) = (0i128, 1i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
    }
    (old_r, old_s, old_t)
}

/// Merges the two congruences `x ≡ r1 (mod m1)` and `x ≡ r2 (mod m2)` into a
/// single congruence `x ≡ solution (mod lcm)`.
///
/// Returns `Some((solution, lcm))` where `solution` lies in `[0, lcm)` and
/// `lcm = lcm(m1, m2)`, or `None` when the two congruences are inconsistent
/// (that is, when `r2 - r1` is not divisible by `gcd(m1, m2)`) or when either
/// modulus is zero.
///
/// Residues are reduced modulo their own modulus on entry, so callers may pass
/// any `u64` residue. All intermediate arithmetic is performed in `i128`.
pub fn crt_pair(r1: u64, m1: u64, r2: u64, m2: u64) -> Option<(u64, u64)> {
    if m1 == 0 || m2 == 0 {
        return None;
    }

    let m1i = m1 as i128;
    let m2i = m2 as i128;
    let r1i = (r1 % m1) as i128;
    let r2i = (r2 % m2) as i128;

    let (g, p, _q) = ext_gcd(m1i, m2i);

    let delta = r2i - r1i;
    if delta % g != 0 {
        return None;
    }

    let lcm = m1i / g * m2i;
    let step = m2i / g;

    // diff is the reduced difference; combine it with the `Bézout` coefficient
    // `p` modulo `step` to find the multiple of m1 that reaches the solution.
    let diff = delta / g;
    let tmp = ((diff % step) * (p % step)) % step;

    let mut x = (r1i + m1i * tmp) % lcm;
    if x < 0 {
        x += lcm;
    }

    Some((x as u64, lcm as u64))
}

/// Solves a system of congruences `x ≡ residues[i] (mod moduli[i])` by folding
/// [`crt_pair`] across the list.
///
/// Returns `Some(x)` with `x` in `[0, lcm(moduli))` when a simultaneous
/// solution exists. Returns `None` when the two slices have different lengths,
/// when any modulus is zero, when the input is empty, or when the system is
/// inconsistent (which only arises for non-coprime moduli).
///
/// The empty-input convention is `None`: with no modulus there is no range to
/// reduce into and hence no canonical representative.
pub fn crt(residues: &[u64], moduli: &[u64]) -> Option<u64> {
    if residues.len() != moduli.len() || residues.is_empty() {
        return None;
    }
    if moduli.iter().any(|&m| m == 0) {
        return None;
    }

    let mut cur_r = residues[0] % moduli[0];
    let mut cur_m = moduli[0];

    for (&r, &m) in residues.iter().zip(moduli.iter()).skip(1) {
        let (nr, nm) = crt_pair(cur_r, cur_m, r, m)?;
        cur_r = nr;
        cur_m = nm;
    }

    Some(cur_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Integer `gcd` for `u128`, used only by the naive reference.
    fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }

    /// Integer `lcm` for `u128`, used only by the naive reference.
    fn lcm_u128(a: u128, b: u128) -> u128 {
        if a == 0 || b == 0 {
            return 0;
        }
        a / gcd_u128(a, b) * b
    }

    /// Brute-force `CRT` reference: scans `[0, lcm)` for the smallest `x` that
    /// satisfies every congruence. Only valid for small moduli.
    fn naive_crt(residues: &[u64], moduli: &[u64]) -> Option<u64> {
        if residues.len() != moduli.len() || residues.is_empty() {
            return None;
        }
        if moduli.iter().any(|&m| m == 0) {
            return None;
        }
        let mut lcm: u128 = 1;
        for &m in moduli {
            lcm = lcm_u128(lcm, m as u128);
        }
        let mut x: u128 = 0;
        while x < lcm {
            let ok = moduli
                .iter()
                .zip(residues.iter())
                .all(|(&m, &r)| x % (m as u128) == (r as u128) % (m as u128));
            if ok {
                return Some(x as u64);
            }
            x += 1;
        }
        None
    }

    /// Minimal `splitmix64` generator for deterministic random test vectors.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Returns a value in `[lo, hi]` (inclusive) with `lo <= hi`.
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            let span = hi - lo + 1;
            lo + (self.next_u64() % span)
        }
    }

    // ---- ext_gcd ----------------------------------------------------------

    #[test]
    fn ext_gcd_bezout_identity_small() {
        for a in 1i128..=40 {
            for b in 1i128..=40 {
                let (g, x, y) = ext_gcd(a, b);
                assert_eq!(a * x + b * y, g);
            }
        }
    }

    #[test]
    fn ext_gcd_returns_gcd() {
        let (g, _, _) = ext_gcd(12, 18);
        assert_eq!(g, 6);
        let (g2, _, _) = ext_gcd(35, 64);
        assert_eq!(g2, 1);
    }

    #[test]
    fn ext_gcd_zero_operand() {
        let (a, b) = (7i128, 0i128);
        let (g, x, y) = ext_gcd(a, b);
        assert_eq!(g, 7);
        assert_eq!(a * x + b * y, 7);
    }

    // ---- crt_pair ---------------------------------------------------------

    #[test]
    fn crt_pair_coprime_basic() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) -> 8 (mod 15)
        assert_eq!(crt_pair(2, 3, 3, 5), Some((8, 15)));
    }

    #[test]
    fn crt_pair_returns_lcm_coprime() {
        let (_, lcm) = crt_pair(1, 4, 2, 9).unwrap();
        assert_eq!(lcm, 36);
    }

    #[test]
    fn crt_pair_non_coprime_solvable() {
        // 0 (mod 4) and 0 (mod 6) -> 0 (mod 12)
        assert_eq!(crt_pair(0, 4, 0, 6), Some((0, 12)));
    }

    #[test]
    fn crt_pair_non_coprime_solvable_nonzero() {
        // 2 (mod 4) and 8 (mod 6): both reduce to even residues sharing 2 mod 2.
        let got = crt_pair(2, 4, 8, 6).unwrap();
        assert_eq!(got.1, 12);
        assert_eq!(got.0 % 4, 2);
        assert_eq!(got.0 % 6, 8 % 6);
    }

    #[test]
    fn crt_pair_non_coprime_unsolvable() {
        // 1 (mod 4) and 2 (mod 6) are incompatible mod gcd=2.
        assert_eq!(crt_pair(1, 4, 2, 6), None);
    }

    #[test]
    fn crt_pair_zero_modulus_none() {
        assert_eq!(crt_pair(0, 0, 1, 5), None);
        assert_eq!(crt_pair(1, 5, 0, 0), None);
    }

    #[test]
    fn crt_pair_reduces_residues() {
        // Residues larger than the modulus must be reduced first.
        assert_eq!(crt_pair(5, 3, 8, 5), crt_pair(2, 3, 3, 5));
    }

    #[test]
    fn crt_pair_order_independent() {
        let a = crt_pair(2, 3, 3, 5).unwrap();
        let b = crt_pair(3, 5, 2, 3).unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn crt_pair_lcm_coprime_is_product() {
        let (_, lcm) = crt_pair(0, 7, 0, 11).unwrap();
        assert_eq!(lcm, 77);
    }

    #[test]
    fn crt_pair_lcm_non_coprime() {
        let (_, lcm) = crt_pair(0, 8, 0, 12).unwrap();
        assert_eq!(lcm, 24);
    }

    #[test]
    fn crt_pair_solution_in_range() {
        let (x, lcm) = crt_pair(13, 17, 4, 19).unwrap();
        assert!(x < lcm);
    }

    #[test]
    fn crt_pair_cross_validation() {
        let (x, _) = crt_pair(2, 3, 3, 5).unwrap();
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
    }

    // ---- crt: fixed reference vectors -------------------------------------

    #[test]
    fn crt_classic_three_modulus() {
        assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some(23));
    }

    #[test]
    fn crt_second_reference_vector() {
        assert_eq!(crt(&[1, 4, 6], &[3, 5, 7]), Some(34));
    }

    #[test]
    fn crt_two_modulus() {
        assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
    }

    #[test]
    fn crt_single_modulus_reduces() {
        assert_eq!(crt(&[9], &[4]), Some(1));
    }

    #[test]
    fn crt_single_modulus_identity() {
        for r in 0u64..20 {
            for m in 1u64..13 {
                assert_eq!(crt(&[r], &[m]), Some(r % m));
            }
        }
    }

    #[test]
    fn crt_single_modulus_one() {
        assert_eq!(crt(&[5], &[1]), Some(0));
    }

    #[test]
    fn crt_non_coprime_solvable_zero() {
        assert_eq!(crt(&[0, 0], &[4, 6]), Some(0));
    }

    #[test]
    fn crt_non_coprime_unsolvable() {
        assert_eq!(crt(&[1, 2], &[4, 6]), None);
    }

    #[test]
    fn crt_non_coprime_solvable_nonzero() {
        // 2 (mod 4), 2 (mod 6) -> 2 (mod 12)
        assert_eq!(crt(&[2, 2], &[4, 6]), Some(2));
    }

    #[test]
    fn crt_four_modulus() {
        let x = crt(&[1, 2, 3, 4], &[2, 3, 5, 7]).unwrap();
        assert_eq!(x % 2, 1);
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
        assert_eq!(x % 7, 4);
    }

    #[test]
    fn crt_large_coprime() {
        let x = crt(&[10, 20, 30], &[11, 13, 17]).unwrap();
        assert_eq!(x % 11, 10);
        assert_eq!(x % 13, 20 % 13);
        assert_eq!(x % 17, 30 % 17);
    }

    #[test]
    fn crt_all_same_residue() {
        // x ≡ 1 everywhere -> 1
        assert_eq!(crt(&[1, 1, 1], &[3, 4, 5]), Some(1));
    }

    #[test]
    fn crt_zero_residues_all() {
        // x ≡ 0 everywhere -> 0
        assert_eq!(crt(&[0, 0, 0], &[3, 4, 5]), Some(0));
    }

    // ---- crt: boundaries --------------------------------------------------

    #[test]
    fn crt_length_mismatch_none() {
        assert_eq!(crt(&[1, 2], &[3]), None);
        assert_eq!(crt(&[1], &[3, 5]), None);
    }

    #[test]
    fn crt_zero_modulus_none() {
        assert_eq!(crt(&[1, 2], &[0, 5]), None);
        assert_eq!(crt(&[1, 2], &[3, 0]), None);
    }

    #[test]
    fn crt_empty_none() {
        let empty: [u64; 0] = [];
        assert_eq!(crt(&empty, &empty), None);
    }

    // ---- crt: structural properties ---------------------------------------

    #[test]
    fn crt_solution_in_range() {
        let x = crt(&[2, 3, 2], &[3, 5, 7]).unwrap();
        assert!(x < 3 * 5 * 7);
    }

    #[test]
    fn crt_cross_validation_three() {
        let residues = [4u64, 1, 3];
        let moduli = [5u64, 7, 8];
        let x = crt(&residues, &moduli).unwrap();
        for (&r, &m) in residues.iter().zip(moduli.iter()) {
            assert_eq!(x % m, r % m);
        }
    }

    #[test]
    fn crt_cross_validation_two() {
        let x = crt(&[3, 4], &[7, 9]).unwrap();
        assert_eq!(x % 7, 3);
        assert_eq!(x % 9, 4);
    }

    #[test]
    fn crt_solution_unique_mod_lcm() {
        // The returned solution plus lcm also satisfies the system, confirming
        // the result is the canonical representative in [0, lcm).
        let residues = [2u64, 3, 2];
        let moduli = [3u64, 5, 7];
        let x = crt(&residues, &moduli).unwrap();
        let lcm = 3 * 5 * 7;
        let shifted = x + lcm;
        for (&r, &m) in residues.iter().zip(moduli.iter()) {
            assert_eq!(shifted % m, r % m);
        }
    }

    #[test]
    fn crt_pair_chain_equiv_crt() {
        // Folding crt_pair by hand matches crt().
        let (r01, m01) = crt_pair(2, 3, 3, 5).unwrap();
        let (r012, _m012) = crt_pair(r01, m01, 2, 7).unwrap();
        assert_eq!(Some(r012), crt(&[2, 3, 2], &[3, 5, 7]));
    }

    #[test]
    fn crt_negative_normalization() {
        // A case whose unnormalized intermediate is negative must still yield a
        // non-negative representative.
        let x = crt(&[2, 3, 2], &[3, 5, 7]).unwrap();
        assert_eq!(x, 23);
    }

    // ---- crt: naive cross-checks ------------------------------------------

    #[test]
    fn crt_matches_naive_fixed_cases() {
        let cases: [(&[u64], &[u64]); 5] = [
            (&[2, 3, 2], &[3, 5, 7]),
            (&[1, 4, 6], &[3, 5, 7]),
            (&[0, 0], &[4, 6]),
            (&[1, 2], &[4, 6]),
            (&[2, 2], &[4, 6]),
        ];
        for (residues, moduli) in cases {
            assert_eq!(crt(residues, moduli), naive_crt(residues, moduli));
        }
    }

    #[test]
    fn crt_random_coprime_cross_check() {
        let mut rng = SplitMix64::new(0x1234_5678);
        // Fixed pairwise-coprime pool keeps lcm small for the naive scan.
        let pool = [3u64, 5, 7, 11, 13];
        for _ in 0..200 {
            let count = rng.range(1, pool.len() as u64) as usize;
            let moduli: Vec<u64> = pool[..count].to_vec();
            let residues: Vec<u64> = moduli.iter().map(|&m| rng.range(0, m - 1)).collect();
            assert_eq!(crt(&residues, &moduli), naive_crt(&residues, &moduli));
        }
    }

    #[test]
    fn crt_random_mixed_cross_check() {
        // Includes non-coprime moduli so both solvable and unsolvable systems
        // are exercised against the brute-force reference.
        let mut rng = SplitMix64::new(0x9ABC_DEF0);
        for _ in 0..300 {
            let m1 = rng.range(2, 12);
            let m2 = rng.range(2, 12);
            let r1 = rng.range(0, 50);
            let r2 = rng.range(0, 50);
            let residues = [r1, r2];
            let moduli = [m1, m2];
            assert_eq!(crt(&residues, &moduli), naive_crt(&residues, &moduli));
        }
    }

    #[test]
    fn crt_random_triple_cross_check() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF);
        for _ in 0..200 {
            let moduli = [rng.range(2, 8), rng.range(2, 8), rng.range(2, 8)];
            let residues = [rng.range(0, 30), rng.range(0, 30), rng.range(0, 30)];
            assert_eq!(crt(&residues, &moduli), naive_crt(&residues, &moduli));
        }
    }

    #[test]
    fn crt_pair_random_cross_check() {
        let mut rng = SplitMix64::new(0x0F0F_0F0F);
        for _ in 0..300 {
            let m1 = rng.range(2, 15);
            let m2 = rng.range(2, 15);
            let r1 = rng.range(0, 40);
            let r2 = rng.range(0, 40);
            let viapair = crt_pair(r1, m1, r2, m2).map(|(x, _)| x);
            let vianaive = naive_crt(&[r1, r2], &[m1, m2]);
            assert_eq!(viapair, vianaive);
        }
    }

    #[test]
    fn crt_random_bigger_two_mod_cross_check() {
        let mut rng = SplitMix64::new(0xABCD_1234);
        let pool = [9u64, 16, 25, 7, 11];
        for _ in 0..200 {
            let i = rng.range(0, (pool.len() - 1) as u64) as usize;
            let mut j = rng.range(0, (pool.len() - 1) as u64) as usize;
            if j == i {
                j = (j + 1) % pool.len();
            }
            let m1 = pool[i];
            let m2 = pool[j];
            let r1 = rng.range(0, m1 - 1);
            let r2 = rng.range(0, m2 - 1);
            assert_eq!(crt(&[r1, r2], &[m1, m2]), naive_crt(&[r1, r2], &[m1, m2]));
        }
    }
}
