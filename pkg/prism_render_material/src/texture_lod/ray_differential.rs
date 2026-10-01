//! Igehy-style ray differentials: screen-space partial derivatives of the UV
//! coordinate, used to reproduce hardware `textureGrad` LOD (isotropic and
//! anisotropic) for a primary visibility-buffer hit.
//!
//! Given `(du/dx, dv/dx)` and `(du/dy, dv/dy)` — the change in UV per screen
//! pixel in x and y — scaled into texel space, the two screen-axis footprint
//! vectors have lengths `p_x` and `p_y`. The isotropic LOD is `log2(max(p_x,
//! p_y))`; the anisotropic LOD samples along the minor axis with an anisotropy
//! ratio `p_max / p_min`.
//!
//! # Conventions
//! * Derivatives are in UV units per pixel; this type scales them by the
//!   texture dimensions internally.
//! * `max_anisotropy` is clamped to `>= 1`; the returned ratio never exceeds it.
//! * All results are finite for zero/degenerate derivatives (footprint floored
//!   at one subtexel), biasing toward mip 0.
//!
//! # References
//! Igehy, "Tracing Ray Differentials", SIGGRAPH 1999; Ewins et al.,
//! "MIP-Map Level Selection for Texture Mapping", 1998; OpenGL/Vulkan
//! `textureGrad` LOD definition.

use super::mip::{mip_from_isotropic_footprint, AnisotropicMip};

/// UV partial derivatives with respect to screen x/y, in UV units per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayDifferential {
    /// `(du/dx, dv/dx)`.
    pub d_dx: [f32; 2],
    /// `(du/dy, dv/dy)`.
    pub d_dy: [f32; 2],
}

impl RayDifferential {
    /// Build from explicit screen-space UV derivatives.
    #[inline]
    #[must_use]
    pub fn new(d_dx: [f32; 2], d_dy: [f32; 2]) -> Self {
        Self { d_dx, d_dy }
    }

    /// Lengths of the two screen-axis footprint vectors in texels, as
    /// `(len_x, len_y)`.
    #[must_use]
    fn axis_lengths_texels(self, tex_width: u32, tex_height: u32) -> (f32, f32) {
        let w = tex_width as f32;
        let h = tex_height as f32;
        let lx = ((self.d_dx[0] * w).powi(2) + (self.d_dx[1] * h).powi(2)).sqrt();
        let ly = ((self.d_dy[0] * w).powi(2) + (self.d_dy[1] * h).powi(2)).sqrt();
        (lx, ly)
    }

    /// Isotropic (trilinear) LOD: `log2(max(len_x, len_y))`, clamped to
    /// `[0, max_mip]`.
    #[must_use]
    pub fn isotropic_mip(self, tex_width: u32, tex_height: u32, max_mip: f32) -> f32 {
        let (lx, ly) = self.axis_lengths_texels(tex_width, tex_height);
        mip_from_isotropic_footprint(lx.max(ly), max_mip)
    }

    /// Anisotropic LOD: base LOD from the minor axis plus the clamped
    /// major/minor anisotropy ratio.
    #[must_use]
    pub fn anisotropic_mip(
        self,
        tex_width: u32,
        tex_height: u32,
        max_mip: f32,
        max_anisotropy: f32,
    ) -> AnisotropicMip {
        let (lx, ly) = self.axis_lengths_texels(tex_width, tex_height);
        let p_max = lx.max(ly).max(f32::MIN_POSITIVE);
        let p_min = lx.min(ly).max(f32::MIN_POSITIVE);
        let max_aniso = max_anisotropy.max(1.0);
        let anisotropy = (p_max / p_min).clamp(1.0, max_aniso);
        // Sample along the minor axis: the effective footprint is the major
        // length divided by the number of taps we will take along it.
        let lod = mip_from_isotropic_footprint(p_max / anisotropy, max_mip);
        AnisotropicMip { lod, anisotropy }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aligned_derivative_matches_log2() {
        // 1/256 UV per pixel on a 256-texel axis -> 1 texel/pixel footprint.
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let lod = rd.isotropic_mip(256, 256, 8.0);
        assert!(lod.abs() < 1.0e-5, "lod={lod}");
    }

    #[test]
    fn doubling_derivative_adds_one_mip() {
        let rd = RayDifferential::new([2.0 / 256.0, 0.0], [0.0, 2.0 / 256.0]);
        let lod = rd.isotropic_mip(256, 256, 8.0);
        assert!((lod - 1.0).abs() < 1.0e-5, "lod={lod}");
    }

    #[test]
    fn isotropic_takes_the_larger_axis() {
        // x axis has 4 texel footprint, y axis 1 texel -> max picks x -> log2(4)=2.
        let rd = RayDifferential::new([4.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let lod = rd.isotropic_mip(256, 256, 8.0);
        assert!((lod - 2.0).abs() < 1.0e-5, "lod={lod}");
    }

    #[test]
    fn anisotropy_ratio_and_lowered_lod() {
        // major=8 texels, minor=1 texel -> anisotropy 8, lod from 8/8=1 -> 0.
        let rd = RayDifferential::new([8.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let a = rd.anisotropic_mip(256, 256, 8.0, 16.0);
        assert!((a.anisotropy - 8.0).abs() < 1.0e-4, "aniso={}", a.anisotropy);
        assert!(a.lod.abs() < 1.0e-5, "lod={}", a.lod);
    }

    #[test]
    fn anisotropy_is_capped() {
        let rd = RayDifferential::new([64.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let a = rd.anisotropic_mip(256, 256, 8.0, 4.0);
        assert!((a.anisotropy - 4.0).abs() < 1.0e-5, "aniso={}", a.anisotropy);
    }

    #[test]
    fn zero_derivative_is_finite_mip_zero() {
        let rd = RayDifferential::new([0.0, 0.0], [0.0, 0.0]);
        let lod = rd.isotropic_mip(1024, 1024, 10.0);
        assert!(lod.is_finite());
        assert_eq!(lod, 0.0);
    }
}
