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
//! A parallel **Catmull-Rom cubic** path mirrors the last two levels for the
//! "high quality" sampler mode: [`trilinear_bicubic`] blends two bicubic mips by
//! the fractional LOD, and [`filter_resolved_bicubic`] averages anisotropic
//! [`trilinear_bicubic`] taps. These trade 16 taps per mip for C1 continuity
//! (no bilinear diamond seams under magnification) while keeping the same
//! resolve -> fetch -> filter contract and the exact-linear-ramp guarantee.
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
mod mitchell;
mod texel_wrap;

pub use bicubic::{bicubic_catmull_rom, catmull_rom_weights};
pub use bilinear::{bilinear, TexelSource};
pub use bspline::{bspline_cubic, bspline_cubic_fast, bspline_cubic_weights};
pub use mitchell::{cubic_mitchell, mitchell_netravali_weights, MITCHELL_B, MITCHELL_C};
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

/// Bicubic (Catmull-Rom) trilinear sample of `src` at continuous `lod`.
///
/// The cubic analogue of [`trilinear`]: evaluates the C1 Catmull-Rom 4x4 fetch
/// ([`bicubic_catmull_rom`]) at the two mips bracketing the continuous LOD and
/// blends them by the fractional level. This is the "high quality" sampler mode
/// an AAA pipeline selects for strong magnification (lightmaps, UI atlases,
/// up-sample prefilters) where bilinear's C0 diamond seams would show, while
/// still prefiltering shimmer through the LOD blend. It reproduces linear ramps
/// exactly within and across levels and interpolates the original texels at
/// texel centres.
///
/// Cubic kernels overshoot by design (ringing at steep edges); the result is
/// intentionally **not** clamped to the input range so it matches a GPU cubic
/// twin bit-closely.
#[must_use]
pub fn trilinear_bicubic<S: TexelSource>(
    src: &S,
    uv: [f32; 2],
    lod: f32,
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    let (w, h) = src.dimensions();
    let tri = trilinear_mip(lod, max_mip_of(w, h));
    let fine = bicubic_catmull_rom(src, tri.fine, uv, wrap_u, wrap_v, border_color);
    if tri.coarse == tri.fine || tri.frac == 0.0 {
        return fine;
    }
    let coarse = bicubic_catmull_rom(src, tri.coarse, uv, wrap_u, wrap_v, border_color);
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

/// Bicubic analogue of [`filter_resolved`]: anisotropic average of
/// [`trilinear_bicubic`] taps.
///
/// Each anisotropic tap is evaluated with the Catmull-Rom cubic trilinear fetch
/// instead of the bilinear one, giving the high-quality cubic sampler mode the
/// full resolve -> fetch -> filter pipeline (anisotropy plus cubic magnification
/// quality in one pass). The isotropic (single-tap) case degenerates to a plain
/// [`trilinear_bicubic`] fetch; `border_color` is returned unchanged when
/// [`SampleResolved::border`] is set.
#[must_use]
pub fn filter_resolved_bicubic<S: TexelSource>(
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
        let c = trilinear_bicubic(src, uv, lod, wrap_u, wrap_v, border_color);
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
    use crate::texture_lod::{RayDifferential, TriangleLodConstant, VirtualTexture};
    use crate::texture_sample::{resolve_cone, resolve_differential, SampleRequest};

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
        let c = trilinear(
            &ramp(),
            [0.5, 0.5],
            3.0,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!((c[0] - 3.0).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn trilinear_fraction_blends_two_mips() {
        // lod 2.25 -> 0.75*mip2 + 0.25*mip3 = 2.25.
        let c = trilinear(
            &ramp(),
            [0.5, 0.5],
            2.25,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!((c[0] - 2.25).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn trilinear_clamps_above_max_mip() {
        // 256x256 -> max mip 8; lod 99 clamps to mip 8.
        let c = trilinear(
            &ramp(),
            [0.5, 0.5],
            99.0,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
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
        let via_resolved =
            filter_resolved(&ramp(), &r, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let direct = trilinear(
            &ramp(),
            [0.5, 0.5],
            r.lod,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!(
            (via_resolved[0] - direct[0]).abs() < 1.0e-6,
            "{via_resolved:?} vs {direct:?}"
        );
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
        let expected = trilinear(
            &ramp(),
            [0.5, 0.5],
            r.taps.lod(),
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
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
        let c = filter_resolved(
            &ramp(),
            &r,
            WrapMode::ClampToBorder,
            WrapMode::ClampToBorder,
            border,
        );
        assert_eq!(c, border);
    }

    /// A 256-sized source whose red channel is a planar ramp in texel space at
    /// every mip, so a linear-reproducing filter returns the plane exactly.
    struct PlaneAll {
        bx: f32,
        by: f32,
    }
    impl TexelSource for PlaneAll {
        fn dimensions(&self) -> (u32, u32) {
            (256, 256)
        }
        fn texel(&self, _mip: u32, x: u32, y: u32) -> [f32; 4] {
            [self.bx * x as f32 + self.by * y as f32, 0.0, 0.0, 1.0]
        }
    }

    #[test]
    fn trilinear_bicubic_integer_lod_matches_single_mip() {
        // frac == 0 path must equal the bare bicubic fetch at that mip.
        let src = PlaneAll { bx: 0.3, by: -0.2 };
        let uv = [0.3718, 0.6421];
        let got = trilinear_bicubic(&src, uv, 2.0, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let want = bicubic_catmull_rom(&src, 2, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!((got[0] - want[0]).abs() < 1.0e-6, "{got:?} vs {want:?}");
    }

    #[test]
    fn trilinear_bicubic_constant_per_mip_matches_trilinear() {
        // On a per-mip constant source bicubic == bilinear == the constant, so
        // the cubic trilinear blend must agree with the bilinear one.
        for &lod in &[0.0_f32, 2.25, 3.75, 7.0] {
            let b = trilinear_bicubic(
                &ramp(),
                [0.5, 0.5],
                lod,
                WrapMode::Repeat,
                WrapMode::Repeat,
                [0.0; 4],
            );
            let t = trilinear(
                &ramp(),
                [0.5, 0.5],
                lod,
                WrapMode::Repeat,
                WrapMode::Repeat,
                [0.0; 4],
            );
            assert!((b[0] - t[0]).abs() < 1.0e-6, "lod {lod}: {b:?} vs {t:?}");
        }
    }

    #[test]
    fn trilinear_bicubic_reproduces_plane_across_lod() {
        // A plane in texel space is linear; both bracketing mips reproduce it
        // and the LOD blend of equal planes is still the plane. At mip m the
        // texel pitch doubles, so the plane value at the sample centre scales.
        let src = PlaneAll { bx: 0.25, by: 0.0 };
        let uv = [0.4, 0.4];
        let lod = 1.5_f32;
        let got = trilinear_bicubic(&src, uv, lod, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        // Independent reference: bicubic reproduces the plane at each mip, so the
        // result equals the LOD-blended bicubic evaluations.
        let fine = bicubic_catmull_rom(&src, 1, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let coarse = bicubic_catmull_rom(&src, 2, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let want = fine[0] + (coarse[0] - fine[0]) * 0.5;
        assert!((got[0] - want).abs() < 1.0e-5, "{got:?} vs {want}");
    }

    #[test]
    fn trilinear_bicubic_is_deterministic() {
        let src = PlaneAll { bx: 0.1, by: 0.3 };
        let a = trilinear_bicubic(
            &src,
            [0.37, 0.59],
            2.3,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        let b = trilinear_bicubic(
            &src,
            [0.37, 0.59],
            2.3,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn filter_resolved_bicubic_isotropic_matches_trilinear_bicubic() {
        let vt = VirtualTexture::new(256, 256, 128);
        let tri = TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            256,
            256,
        );
        let req = SampleRequest::isotropic([0.5, 0.5], WrapMode::Repeat);
        let r = resolve_cone(vt, &req, tri, 0.02, 1.0);
        let via =
            filter_resolved_bicubic(&ramp(), &r, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let direct = trilinear_bicubic(
            &ramp(),
            [0.5, 0.5],
            r.lod,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!((via[0] - direct[0]).abs() < 1.0e-6, "{via:?} vs {direct:?}");
    }

    #[test]
    fn filter_resolved_bicubic_border_returns_border_color() {
        let vt = VirtualTexture::new(256, 256, 128);
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([1.5, 0.5], WrapMode::ClampToBorder, 1.0);
        let r = resolve_differential(vt, &req, rd);
        assert!(r.border);
        let border = [5.0, 6.0, 7.0, 8.0];
        let c = filter_resolved_bicubic(
            &ramp(),
            &r,
            WrapMode::ClampToBorder,
            WrapMode::ClampToBorder,
            border,
        );
        assert_eq!(c, border);
    }
}
