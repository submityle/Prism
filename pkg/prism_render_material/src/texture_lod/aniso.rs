//! Anisotropic sample-tap generation: the discrete UV positions a shader must
//! fetch (and their weights) to reproduce hardware anisotropic filtering inside
//! a visibility-buffer or ray-traced shading path, where the fixed-function
//! anisotropic sampler is unavailable.
//!
//! Hardware anisotropy walks the *major* axis of the screen-space UV footprint,
//! taking several trilinear taps whose LOD is driven by the *minor* axis (see
//! [`super::mip::AnisotropicMip`]). This module reproduces that walk in closed
//! form: given the footprint's major-axis UV vector and the chosen anisotropy
//! ratio, it lays `N = ceil(anisotropy)` taps symmetrically about the shaded
//! UV, each carrying an equal `1/N` weight and the shared minor-axis LOD. No
//! learning path -- a CPU golden reproduces a GPU twin exactly.
//!
//! # Conventions
//! * `major_axis_uv` is the longer screen-axis footprint vector expressed in UV
//!   units (not texels); [`super::RayDifferential::major_axis_uv`] produces it.
//! * Taps are centred on the shaded UV and span the full major-axis footprint:
//!   tap `i` sits at `center + ((i + 0.5)/N - 0.5) * major_axis_uv`.
//! * Tap count is `ceil(anisotropy)` clamped to `[1, MAX_ANISO_TAPS]`.
//! * Weights are uniform (`1/N`) and sum to 1, matching the standard
//!   equal-weight anisotropic average.
//!
//! # References
//! Barkans, "High-Quality Rendering Using the Talisman Architecture" (1997,
//! anisotropic footprint walk); OpenGL/Vulkan anisotropic filtering model;
//! Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.2.2.

use super::mip::AnisotropicMip;

/// Upper bound on anisotropic taps; matches the common `maxAnisotropy = 16`
/// hardware cap and bounds the on-stack tap buffer.
pub const MAX_ANISO_TAPS: usize = 16;

/// A fixed-capacity set of anisotropic sample taps centred on a shaded UV.
#[derive(Clone, Copy, Debug)]
pub struct AnisoTaps {
    uvs: [[f32; 2]; MAX_ANISO_TAPS],
    len: usize,
    weight: f32,
    lod: f32,
}

impl AnisoTaps {
    /// Number of active taps (`>= 1`).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: there is always at least one (centre) tap.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Uniform per-tap weight (`1 / len`); the weights sum to 1.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Shared trilinear LOD (from the minor footprint axis) for every tap.
    #[inline]
    #[must_use]
    pub fn lod(&self) -> f32 {
        self.lod
    }

    /// The active tap UV positions.
    #[inline]
    #[must_use]
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs[..self.len]
    }
}

/// Lay out anisotropic taps for a shaded texel.
///
/// * `center_uv` — the shaded surface UV.
/// * `major_axis_uv` — the major footprint axis in UV units (step direction and
///   extent); a zero vector collapses to a single centre tap.
/// * `aniso` — the minor-axis LOD + anisotropy ratio from
///   [`super::RayDifferential::anisotropic_mip`].
#[must_use]
pub fn anisotropic_taps(
    center_uv: [f32; 2],
    major_axis_uv: [f32; 2],
    aniso: &AnisotropicMip,
) -> AnisoTaps {
    // Tap count tracks the anisotropy ratio, floored at 1 and capped.
    let ratio = if aniso.anisotropy.is_finite() {
        aniso.anisotropy.max(1.0)
    } else {
        1.0
    };
    let n = (ratio.ceil() as usize).clamp(1, MAX_ANISO_TAPS);

    let mut uvs = [[0.0_f32; 2]; MAX_ANISO_TAPS];
    let inv_n = 1.0 / n as f32;
    for (i, slot) in uvs.iter_mut().take(n).enumerate() {
        // Symmetric parametric position in [-0.5, 0.5) across the footprint.
        let t = (i as f32 + 0.5) * inv_n - 0.5;
        slot[0] = center_uv[0] + t * major_axis_uv[0];
        slot[1] = center_uv[1] + t * major_axis_uv[1];
    }

    AnisoTaps {
        uvs,
        len: n,
        weight: inv_n,
        lod: aniso.lod,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aniso(ratio: f32, lod: f32) -> AnisotropicMip {
        AnisotropicMip {
            lod,
            anisotropy: ratio,
        }
    }

    #[test]
    fn ratio_one_is_single_center_tap() {
        let taps = anisotropic_taps([0.5, 0.5], [0.25, 0.0], &aniso(1.0, 2.0));
        assert_eq!(taps.len(), 1);
        assert!((taps.weight() - 1.0).abs() < 1.0e-6);
        assert_eq!(taps.uvs()[0], [0.5, 0.5]);
    }

    #[test]
    fn tap_count_is_ceil_of_ratio() {
        assert_eq!(anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(3.2, 0.0)).len(), 4);
        assert_eq!(anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(4.0, 0.0)).len(), 4);
    }

    #[test]
    fn tap_count_is_capped() {
        let taps = anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(999.0, 0.0));
        assert_eq!(taps.len(), MAX_ANISO_TAPS);
    }

    #[test]
    fn weights_sum_to_one() {
        let taps = anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(7.0, 0.0));
        let sum: f32 = taps.uvs().iter().map(|_| taps.weight()).sum();
        assert!((sum - 1.0).abs() < 1.0e-5, "sum={sum}");
    }

    #[test]
    fn taps_are_symmetric_about_center() {
        let taps = anisotropic_taps([0.5, 0.5], [0.4, 0.0], &aniso(4.0, 0.0));
        let uvs = taps.uvs();
        // Mean of the tap positions equals the centre (symmetric layout).
        let mean_u: f32 = uvs.iter().map(|p| p[0]).sum::<f32>() / uvs.len() as f32;
        let mean_v: f32 = uvs.iter().map(|p| p[1]).sum::<f32>() / uvs.len() as f32;
        assert!((mean_u - 0.5).abs() < 1.0e-6, "mean_u={mean_u}");
        assert!((mean_v - 0.5).abs() < 1.0e-6, "mean_v={mean_v}");
    }

    #[test]
    fn taps_span_the_major_axis() {
        let taps = anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(4.0, 0.0));
        let uvs = taps.uvs();
        let min_u = uvs.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_u = uvs.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        // First/last tap centres sit at +/- (0.5 - 0.5/N) * |axis|.
        let expected = 0.5 - 0.5 / 4.0;
        assert!((max_u - expected).abs() < 1.0e-6, "max_u={max_u}");
        assert!((min_u + expected).abs() < 1.0e-6, "min_u={min_u}");
    }

    #[test]
    fn zero_axis_collapses_to_center() {
        let taps = anisotropic_taps([0.3, 0.7], [0.0, 0.0], &aniso(4.0, 1.0));
        for uv in taps.uvs() {
            assert_eq!(*uv, [0.3, 0.7]);
        }
    }

    #[test]
    fn non_finite_ratio_is_single_tap() {
        let taps = anisotropic_taps([0.1, 0.2], [1.0, 1.0], &aniso(f32::NAN, 0.0));
        assert_eq!(taps.len(), 1);
    }

    #[test]
    fn lod_is_propagated() {
        let taps = anisotropic_taps([0.0, 0.0], [1.0, 0.0], &aniso(2.0, 3.5));
        assert_eq!(taps.lod(), 3.5);
    }
}
