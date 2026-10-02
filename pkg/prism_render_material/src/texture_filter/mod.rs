//! The filtered-fetch stage of the manual texture sampler: turn a resolved
//! sampling plan into a final RGBA colour.
//!
//! [`super::texture_sample`] resolves a surface UV + footprint into a
//! [`SampleResolved`] (continuous LOD, anisotropic taps, resident pages). This
//! module consumes that plan and the raw texels behind a [`TexelSource`] to
//! produce the filtered colour a forward raster pass would get from the
//! fixed-function sampler, completing the software sampling pipeline
//! **resolve -> fetch -> filter** for visibility-buffer and ray-traced shading.
//!
//! Filtering is the standard three-level hierarchy, all in closed form:
//!
//! 1. [`bilinear`] — four-neighbour blend within one mip, with per-corner wrap.
//! 2. [`trilinear`] — blend the two mips bracketing the continuous LOD.
//! 3. [`filter_resolved`] — average the anisotropic taps (uniform weights),
//!    each evaluated trilinearly, reproducing hardware anisotropic filtering.
//!
//! Pure `f32` arithmetic with defensive clamps and no AI/ML path, so a CPU
//! golden reproduces a GPU twin exactly.
//!
//! # Conventions
//! * Texel-centre and mip-sizing conventions match [`bilinear`] and
//!   [`super::texture_lod`]; see those modules.
//! * Tap UVs may lie outside `[0, 1]` for a wide footprint; they are passed to
//!   [`bilinear`] verbatim and folded at texel granularity by the wrap modes,
//!   matching how hardware re-folds each anisotropic tap.
//! * When [`SampleResolved::border`] is set (a [`WrapMode::ClampToBorder`] axis
//!   was out of range at the sample centre) [`filter_resolved`] returns the
//!   border colour directly, consistent with the clamp-to-edge addressing the
//!   resolve stage applied.
//!
//! # References
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.2.
//! * Williams, "Pyramidal Parametrics" (SIGGRAPH 1983) — trilinear mipmapping.

mod bicubic;
mod bilinear;
mod bspline;
mod texel_wrap;

pub use bicubic::{bicubic_catmull_rom, catmull_rom_weights};
pub use bspline::{bspline_cubic, bspline_cubic_fast, bspline_cubic_weights};
pub use bilinear::{bilinear, TexelSource};
pub use texel_wrap::{wrap_texel, TexelAddr};

use super::texture_addressing::WrapMode;
use super::texture_lod::trilinear_mip;
use super::texture_sample::SampleResolved;

/// Coarsest valid mip for a base dimension pair (single-texel largest axis).
#[inline]
#[must_use]
fn max_mip_of(width: u32, height: u32) -> u32 {
    width.max(height).max(1).ilog2()
}

#[inline]
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Trilinearly sample `src` at continuous `lod`: bilinear at the two bracketing
/// mips, blended by the fractional LOD.
#[must_use]
pub fn trilinear<S: TexelSource>(
    src: &S,
    uv: [f32; 2],
    lod: f32,
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    let (w, h) = src.dimensions();
    let tri = trilinear_mip(lod, max_mip_of(w, h));
    let fine = bilinear(src, tri.fine, uv, wrap_u, wrap_v, border_color);
    if tri.coarse == tri.fine || tri.frac == 0.0 {
        return fine;
    }
    let coarse = bilinear(src, tri.coarse, uv, wrap_u, wrap_v, border_color);
    lerp4(fine, coarse, tri.frac)
}

/// Evaluate a fully resolved sample: anisotropic average of trilinear taps.
///
/// Each tap in [`SampleResolved::taps`] is sampled trilinearly at the shared
/// tap LOD and accumulated with its uniform weight. The isotropic (single-tap)
/// case degenerates to a plain trilinear fetch. Returns `border_color`
/// unchanged when [`SampleResolved::border`] is set.
#[must_use]
pub fn filter_resolved<S: TexelSource>(
    src: &S,
    resolved: &SampleResolved,
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    if resolved.border {
        return border_color;
    }
    let lod = resolved.taps.lod();
    let weight = resolved.taps.weight();
    let mut acc = [0.0_f32; 4];
    for &uv in resolved.taps.uvs() {
        let c = trilinear(src, uv, lod, wrap_u, wrap_v, border_color);
        acc[0] += c[0] * weight;
        acc[1] += c[1] * weight;
        acc[2] += c[2] * weight;
        acc[3] += c[3] * weight;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_sample::{resolve_cone, resolve_differential, SampleRequest};
    use crate::texture_lod::{RayDifferential, TriangleLodConstant, VirtualTexture};

    /// A ramp whose per-mip constant lets us check mip selection: mip m returns
    /// red = m, so a sampled colour reveals which level(s) were read.
    struct MipRamp {
        w: u32,
        h: u32,
    }
    impl TexelSource for MipRamp {
        fn dimensions(&self) -> (u32, u32) {
            (self.w, self.h)
        }
        fn texel(&self, mip: u32, _x: u32, _y: u32) -> [f32; 4] {
            [mip as f32, 0.0, 0.0, 1.0]
        }
    }

    fn ramp() -> MipRamp {
        MipRamp { w: 256, h: 256 }
    }

    #[test]
    fn trilinear_integer_lod_reads_single_mip() {
        let c = trilinear(&ramp(), [0.5, 0.5], 3.0, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((c[0] - 3.0).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn trilinear_fraction_blends_two_mips() {
        // lod 2.25 -> 0.75*mip2 + 0.25*mip3 = 2.25.
        let c = trilinear(&ramp(), [0.5, 0.5], 2.25, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((c[0] - 2.25).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn trilinear_clamps_above_max_mip() {
        // 256x256 -> max mip 8; lod 99 clamps to mip 8.
        let c = trilinear(&ramp(), [0.5, 0.5], 99.0, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((c[0] - 8.0).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn filter_resolved_isotropic_matches_trilinear() {
        let vt = VirtualTexture::new(256, 256, 128);
        let tri = TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            256,
            256,
        );
        let req = SampleRequest::isotropic([0.5, 0.5], WrapMode::Repeat);
        let r = resolve_cone(vt, &req, tri, 0.02, 1.0);
        let via_resolved = filter_resolved(&ramp(), &r, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let direct = trilinear(&ramp(), [0.5, 0.5], r.lod, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((via_resolved[0] - direct[0]).abs() < 1.0e-6, "{via_resolved:?} vs {direct:?}");
    }

    #[test]
    fn filter_resolved_anisotropic_averages_taps() {
        // Constant-per-mip ramp: anisotropic average still equals the tap LOD
        // value (all taps share lod), proving the weighted sum normalises to 1.
        let vt = VirtualTexture::new(256, 256, 128);
        let rd = RayDifferential::new([8.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([0.5, 0.5], WrapMode::Repeat, 16.0);
        let r = resolve_differential(vt, &req, rd);
        assert!(r.taps.len() >= 8);
        let c = filter_resolved(&ramp(), &r, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let expected = trilinear(&ramp(), [0.5, 0.5], r.taps.lod(), WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((c[0] - expected[0]).abs() < 1.0e-5, "{c:?} vs {expected:?}");
    }

    #[test]
    fn filter_resolved_border_returns_border_color() {
        let vt = VirtualTexture::new(256, 256, 128);
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([1.5, 0.5], WrapMode::ClampToBorder, 1.0);
        let r = resolve_differential(vt, &req, rd);
        assert!(r.border);
        let border = [7.0, 7.0, 7.0, 7.0];
        let c = filter_resolved(&ramp(), &r, WrapMode::ClampToBorder, WrapMode::ClampToBorder, border);
        assert_eq!(c, border);
    }
}
