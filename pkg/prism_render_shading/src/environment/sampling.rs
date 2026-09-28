//! Low-discrepancy sampling primitives shared by the split-sum IBL passes.
//!
//! Both the environment-BRDF table ([`super::brdf_lut`]) and the prefiltered
//! radiance map ([`super::prefilter`]) importance-sample the GGX lobe with the
//! Hammersley sequence, so the sequence, its Van der Corput base-two radical
//! inverse, and the GGX half-vector draw live here as one reference the GPU
//! shaders mirror bit-for-bit.  All transcendentals route through
//! `bevy_math::ops` for cross-platform determinism.

use bevy_math::ops;

/// Low-discrepancy Van der Corput radical inverse in base 2.
pub(super) fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_left(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    // 1 / 2^32 maps the shuffled integer into [0, 1).
    (bits as f32) * 2.328_306_4e-10
}

/// The `i`-th of `n` points of the Hammersley sequence on the unit square.
pub(super) fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [(i as f32) / (n as f32), radical_inverse_vdc(i)]
}

/// Draws a GGX-distributed half vector in the tangent frame whose normal is
/// `+Z`, given a uniform sample `xi` and perceptual `roughness`.
pub(super) fn importance_sample_ggx(xi: [f32; 2], roughness: f32) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = core::f32::consts::TAU * xi[0];
    // Inverse-CDF of the GGX NDF projected onto cos(theta).
    let cos_theta = ops::sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1]));
    let sin_theta = ops::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    [
        sin_theta * ops::cos(phi),
        sin_theta * ops::sin(phi),
        cos_theta,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_matches_known_values() {
        // Van der Corput base-2 of 1,2,3 are 0.5, 0.25, 0.75.
        assert!((radical_inverse_vdc(1) - 0.5).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(2) - 0.25).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(3) - 0.75).abs() < 1.0e-6);
        assert!(radical_inverse_vdc(0).abs() < 1.0e-6);
    }

    #[test]
    fn hammersley_spans_unit_square() {
        let n = 16;
        for i in 0..n {
            let [x, y] = hammersley(i, n);
            assert!((0.0..1.0).contains(&x));
            assert!((0.0..1.0).contains(&y));
        }
        // First coordinate is the uniform sweep.
        assert!((hammersley(0, n)[0]).abs() < 1.0e-6);
        assert!((hammersley(n - 1, n)[0] - (n - 1) as f32 / n as f32).abs() < 1.0e-6);
    }

    #[test]
    fn ggx_half_vector_is_unit_and_upper_hemisphere() {
        for roughness in [0.0, 0.25, 0.5, 1.0] {
            for i in 0..32 {
                let h = importance_sample_ggx(hammersley(i, 32), roughness);
                let len = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
                assert!((len - 1.0).abs() < 1.0e-4);
                assert!(h[2] >= -1.0e-4, "half vector should stay in +Z hemisphere");
            }
        }
    }

    #[test]
    fn mirror_roughness_collapses_to_normal() {
        // At roughness 0 the GGX lobe is a delta at the normal (+Z).
        for i in 0..8 {
            let h = importance_sample_ggx(hammersley(i, 8), 0.0);
            assert!((h[2] - 1.0).abs() < 1.0e-4);
        }
    }
}
