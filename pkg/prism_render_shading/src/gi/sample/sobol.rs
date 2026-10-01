//! Owen-scrambled Sobol' low-discrepancy sampler — CPU golden.
//!
//! The GI ray budget is tiny (1–2 samples per probe per frame), so sample
//! placement quality dominates the converged image.  We draw every random
//! decision from the first two dimensions of the Sobol' (0, 2)-sequence and
//! apply hash-based *Owen scrambling* (nested uniform scrambling) to decorrelate
//! pixels, frames, and dimensions without destroying the stratification that
//! gives Sobol' its low discrepancy.
//!
//! # References
//! * Sobol' / Antonov–Saleev XOR recurrence for the classic 2-D (0, 2)-sequence.
//! * Burley, *Practical Hash-based Owen Scrambling* (JCGT 2020) — the
//!   Laine–Karras permutation used by [\`nested_uniform_scramble\`].
//!
//! All functions are deterministic and allocation-free; the GPU twin reproduces
//! them bit-for-bit using the same \`u32\` arithmetic.

/// First Sobol' dimension: the base-2 radical inverse (van der Corput), i.e.
/// the bit-reversal of \`index\` in fixed point.
#[inline]
pub fn sobol_dim0(index: u32) -> u32 {
    index.reverse_bits()
}

/// Second Sobol' dimension via the Antonov–Saleev XOR recurrence with the
/// classic generator (direction numbers \`v_k = 2^31 >> (shifted triangular)\`),
/// yielding a (0, 2)-sequence together with [\`sobol_dim0\`].
#[inline]
pub fn sobol_dim1(mut index: u32) -> u32 {
    let mut v: u32 = 0x8000_0000;
    let mut result: u32 = 0;
    while index != 0 {
        if index & 1 == 1 {
            result ^= v;
        }
        v ^= v >> 1;
        index >>= 1;
    }
    result
}

/// The Laine–Karras permutation used as the core of hash-based Owen scrambling.
#[inline]
fn laine_karras_permutation(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Hash-based nested uniform (Owen) scramble of a 32-bit fixed-point value.
///
/// Owen scrambling randomly flips the sub-tree at every bit depth; done with a
/// seeded hash it preserves the sequence's \`(0, 2)\` stratification while making
/// independently-seeded draws statistically decorrelated.
#[inline]
pub fn nested_uniform_scramble(mut x: u32, seed: u32) -> u32 {
    x = x.reverse_bits();
    x = laine_karras_permutation(x, seed);
    x.reverse_bits()
}

/// Mixes an arbitrary set of integer coordinates into a well-distributed 32-bit
/// seed (a finalised multiply-xor hash), used to derive per-pixel/-frame scramble
/// seeds.
#[inline]
pub fn hash_seed(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Combines two integers into a single hashed seed (order sensitive).
#[inline]
pub fn hash_combine(a: u32, b: u32) -> u32 {
    hash_seed(a ^ hash_seed(b).wrapping_mul(0x9e37_79b9))
}

/// Converts a 32-bit fixed-point sample to an \`f32\` in \`[0, 1)\` using the top
/// 24 bits so the result is exactly representable (no rounding to 1.0).
#[inline]
pub fn to_unit_f32(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

/// Draws one Owen-scrambled Sobol' 2-D point for sample \`index\`, decorrelated by
/// \`seed\`.  Returns \`(u, v)\` in \`[0, 1)^2\`.
#[inline]
pub fn sample_2d(index: u32, seed: u32) -> (f32, f32) {
    // Scramble the sample index itself (shuffles which point we draw) and each
    // output dimension with distinct derived seeds.
    let shuffled = nested_uniform_scramble(index, hash_combine(seed, 0x5851_f42d));
    let x = nested_uniform_scramble(sobol_dim0(shuffled), hash_combine(seed, 0xa136_aaad));
    let y = nested_uniform_scramble(sobol_dim1(shuffled), hash_combine(seed, 0x136a_aad0));
    (to_unit_f32(x), to_unit_f32(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_values_are_in_range() {
        for i in 0..4096u32 {
            let (u, v) = sample_2d(i, 0xdead_beef);
            assert!((0.0..1.0).contains(&u), "u out of range: {u}");
            assert!((0.0..1.0).contains(&v), "v out of range: {v}");
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(sample_2d(37, 7), sample_2d(37, 7));
        assert_eq!(sobol_dim1(12345), sobol_dim1(12345));
    }

    #[test]
    fn distinct_seeds_decorrelate() {
        let a = sample_2d(10, 1);
        let b = sample_2d(10, 2);
        assert_ne!(a, b, "different seeds must produce different points");
    }

    #[test]
    fn owen_scramble_is_a_bijection_over_a_block() {
        // A nested uniform scramble with a fixed seed must be a permutation:
        // feeding every value in a 12-bit block yields every value exactly once.
        let seed = 0x1234_5678;
        let mut seen = vec![false; 1 << 12];
        for x in 0u32..(1 << 12) {
            // Scramble in the top bits so the low 20 bits stay zero and the
            // output block is comparable.
            let s = nested_uniform_scramble(x << 20, seed) >> 20;
            assert!(!seen[s as usize], "collision at {x} -> {s}");
            seen[s as usize] = true;
        }
        assert!(seen.into_iter().all(|b| b));
    }

    #[test]
    fn first_sobol_point_is_the_origin() {
        // Index 0 is the sequence origin before scrambling.
        assert_eq!(sobol_dim0(0), 0);
        assert_eq!(sobol_dim1(0), 0);
    }

    #[test]
    fn sobol_2d_is_well_stratified_in_elementary_intervals() {
        // A (0, 2)-sequence places exactly one of its first 2^k points in each
        // elementary interval of area 2^-k.  Check the 4x4 (k = 4) grid for the
        // first 16 *unscrambled* points.
        let mut counts = [[0u32; 4]; 4];
        for i in 0..16u32 {
            let x = to_unit_f32(sobol_dim0(i));
            let y = to_unit_f32(sobol_dim1(i));
            let cx = (x * 4.0) as usize;
            let cy = (y * 4.0) as usize;
            counts[cy.min(3)][cx.min(3)] += 1;
        }
        for row in counts {
            for c in row {
                assert_eq!(c, 1, "every 4x4 cell holds exactly one of 16 points");
            }
        }
    }

    #[test]
    fn mean_approaches_one_half() {
        // Owen-scrambled Sobol' integrates the identity to 1/2 with low error.
        let n = 4096u32;
        let mut sum = (0.0f64, 0.0f64);
        for i in 0..n {
            let (u, v) = sample_2d(i, 0xcafe_f00d);
            sum.0 += u as f64;
            sum.1 += v as f64;
        }
        let mean = (sum.0 / n as f64, sum.1 / n as f64);
        assert!((mean.0 - 0.5).abs() < 0.01, "u mean = {}", mean.0);
        assert!((mean.1 - 0.5).abs() < 0.01, "v mean = {}", mean.1);
    }
}
