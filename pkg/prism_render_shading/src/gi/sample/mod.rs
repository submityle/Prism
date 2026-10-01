//! Low-discrepancy sampling for the GI ray budget — CPU golden.
//!
//! The GI integrator can afford only a handful of rays per probe per frame, so
//! *where* those samples land dominates convergence.  This module provides the
//! deterministic, allocation-free sampling primitives that every GI pass shares,
//! split into three cooperating pieces:
//!
//! * [`sobol`] — Owen-scrambled Sobol' (0, 2)-sequence: high-quality
//!   stratification across the intra-frame sample index, decorrelated per
//!   pixel/frame/dimension by hash-based nested-uniform (Owen) scrambling.
//! * [`blue_noise`] — a texture-free, R2 low-discrepancy sampler with per-pixel
//!   Cranley–Patterson rotation and a per-frame golden-ratio stride; its residual
//!   error is "blue-noise-like" over space and time so TAA/denoising clean it up
//!   far better than white noise.
//! * [`mapping`] — warps canonical `[0, 1)^2` points onto the disk, the
//!   cosine-weighted hemisphere (Lambertian importance sampling) and the uniform
//!   hemisphere, plus the tangent-frame rotation onto a world-space normal.
//!
//! The high-level entry point is [`low_discrepancy_sample_2d`], which fuses the
//! two samplers: it draws an Owen-scrambled Sobol' point (good stratification
//! across `sample`) while deriving the scramble seed from the pixel, frame and a
//! caller-supplied dimension offset (good spatio-temporal decorrelation).  This
//! is the per-pixel-scrambled QMC construction used by modern real-time path
//! guiding; the GPU/WESL twin reproduces it bit-for-bit.
//!
//! All items are deterministic pure functions with unit tests, forming the
//! numerical reference the GPU passes must match under real-device parity.

pub mod blue_noise;
pub mod mapping;
pub mod sobol;

pub use blue_noise::{animated_sample_2d, r2_sample_2d};
pub use mapping::{
    concentric_disk, cosine_hemisphere, cosine_hemisphere_pdf, cosine_hemisphere_world,
    orthonormal_basis, uniform_hemisphere, uniform_hemisphere_pdf, world_from_local,
};
pub use sobol::{
    hash_combine, hash_seed, nested_uniform_scramble, sample_2d, sobol_dim0, sobol_dim1,
    to_unit_f32,
};

/// Draws the primary low-discrepancy 2-D sample for a GI ray.
///
/// * `pixel` — integer framebuffer coordinate (spatial decorrelation).
/// * `frame` — temporal frame index (temporal decorrelation + progressive
///   refinement across frames).
/// * `sample` — intra-frame sample index within this pixel's ray budget; this is
///   the dimension that stays Sobol'-stratified.
/// * `dim_offset` — a per-decision salt so independent 2-D draws in the same
///   pixel/frame (e.g. bounce 0 direction vs. light pick) do not correlate.
///
/// Returns `(u, v) ∈ [0, 1)^2`, ready to feed [`mapping`].
///
/// The scramble seed mixes `pixel`, `frame` and `dim_offset`; the Sobol' sample
/// index is `sample`.  Holding the seed fixed and sweeping `sample` yields a
/// well-stratified sequence; sweeping `pixel`/`frame`/`dim_offset` reseeds the
/// Owen scramble so neighbouring pixels, successive frames and distinct
/// decisions are statistically independent.
#[inline]
pub fn low_discrepancy_sample_2d(
    pixel: (u32, u32),
    frame: u32,
    sample: u32,
    dim_offset: u32,
) -> (f32, f32) {
    let pixel_seed = hash_combine(pixel.0.wrapping_mul(0x9e37_79b9) ^ pixel.1, 0x85eb_ca6b);
    let seed = hash_combine(hash_combine(pixel_seed, frame), dim_offset ^ 0xc2b2_ae35);
    sample_2d(sample, seed)
}

/// Convenience: a world-space cosine-weighted hemisphere direction for a GI ray,
/// combining [`low_discrepancy_sample_2d`] with [`mapping::cosine_hemisphere_world`].
#[inline]
pub fn cosine_direction(
    pixel: (u32, u32),
    frame: u32,
    sample: u32,
    dim_offset: u32,
    normal: [f32; 3],
) -> [f32; 3] {
    let (u, v) = low_discrepancy_sample_2d(pixel, frame, sample, dim_offset);
    cosine_hemisphere_world(u, v, normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_in_range() {
        for frame in 0..16u32 {
            for s in 0..8u32 {
                let (u, v) = low_discrepancy_sample_2d((17, 42), frame, s, 0);
                assert!((0.0..1.0).contains(&u), "u = {u}");
                assert!((0.0..1.0).contains(&v), "v = {v}");
            }
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            low_discrepancy_sample_2d((3, 9), 5, 2, 7),
            low_discrepancy_sample_2d((3, 9), 5, 2, 7)
        );
    }

    #[test]
    fn pixels_frames_and_dims_decorrelate() {
        let base = low_discrepancy_sample_2d((10, 10), 0, 0, 0);
        assert_ne!(base, low_discrepancy_sample_2d((11, 10), 0, 0, 0), "pixel");
        assert_ne!(base, low_discrepancy_sample_2d((10, 10), 1, 0, 0), "frame");
        assert_ne!(base, low_discrepancy_sample_2d((10, 10), 0, 0, 1), "dim");
    }

    #[test]
    fn fixed_seed_sequence_is_well_stratified() {
        // Holding pixel/frame/dim fixed and sweeping `sample` must stay spread:
        // the two 1-D marginals each hit every quarter of [0,1) within 4 samples.
        let mut quads = [[false; 4]; 2];
        for s in 0..4u32 {
            let (u, v) = low_discrepancy_sample_2d((1, 1), 0, s, 0);
            quads[0][(u * 4.0) as usize] = true;
            quads[1][(v * 4.0) as usize] = true;
        }
        assert!(quads[0].iter().all(|&b| b), "u not stratified: {quads:?}");
        assert!(quads[1].iter().all(|&b| b), "v not stratified: {quads:?}");
    }

    #[test]
    fn cosine_direction_is_unit_in_normal_hemisphere() {
        let n = [0.0, 0.0, 1.0];
        for s in 0..32u32 {
            let d = cosine_direction((5, 5), 0, s, 0, n);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "not unit: {d:?}");
            assert!(d[2] >= -1e-4, "below hemisphere: {d:?}");
        }
    }
}
