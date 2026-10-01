//! The manual texture sampler: the single entry point a visibility-buffer or
//! ray-traced shading path calls to turn a surface UV + footprint into a
//! concrete fetch plan.
//!
//! A forward raster pass gets texture filtering for free from the
//! fixed-function sampler: the hardware derives screen-space derivatives, picks
//! a mip, walks the anisotropic footprint, folds wrap modes, and streams the
//! right virtual-texture pages. A visibility-buffer resolve or a ray-traced hit
//! has none of that -- it must reproduce the whole pipeline in software. The
//! sibling modules supply each stage in closed form:
//!
//! * [`texture_addressing`](super::texture_addressing) — wrap/address the UV.
//! * [`texture_lod`](super::texture_lod) — ray-cone / ray-differential LOD,
//!   anisotropy ratio, and virtual-texture page residency.
//!
//! This module stitches them into two resolve functions so callers never have
//! to re-derive the ordering (address -> LOD -> anisotropy -> residency):
//!
//! * [`resolve_differential`] — primary visibility hits with screen-space UV
//!   derivatives, giving `textureGrad`-equivalent anisotropic filtering.
//! * [`resolve_cone`] — secondary (reflection/refraction/GI) rays carrying a
//!   scalar ray cone, giving robust isotropic trilinear LOD.
//!
//! Everything is pure analytic `f32` arithmetic with defensive clamps and no
//! AI/ML path, so a CPU golden reproduces a GPU twin bit-for-bit.
//!
//! # Conventions
//! * Resolve order is fixed: **address the UV first**, then compute LOD and
//!   anisotropy from the *footprint* (which is independent of wrapping), then
//!   lay taps around the addressed centre, then request residency at the
//!   addressed centre. This matches hardware, where wrap addressing and LOD are
//!   orthogonal.
//! * [`SampleResolved::taps`] are in UV space around the addressed centre and
//!   may stray outside `[0, 1]` for a wide footprint. **Per-tap re-wrapping is
//!   the caller's responsibility** for a tiling (`Repeat`/`Mirror`) texture,
//!   exactly as hardware re-folds each anisotropic tap; the resolver only
//!   addresses the centre so residency targets the right page.
//! * A `ClampToBorder` axis out of range sets [`SampleResolved::border`]; the
//!   caller should emit the sampler border colour and skip the fetch.
//!
//! # References
//! * Akenine-Moller et al., "Texture Level of Detail Strategies for Real-Time
//!   Ray Tracing", Ray Tracing Gems, 2019.
//! * Igehy, "Tracing Ray Differentials", SIGGRAPH 1999.
//! * Vulkan `textureGrad` / `VkSamplerCreateInfo` filtering model.

mod descriptor;

pub use descriptor::{SampleRequest, SampleResolved};

use super::texture_addressing::address_uv;
use super::texture_lod::{
    anisotropic_taps, cone_mip_level, AnisotropicMip, RayDifferential, TriangleLodConstant,
    VirtualTexture,
};

/// Resolve a sample for a primary visibility hit using screen-space UV
/// derivatives (anisotropic, `textureGrad`-equivalent).
///
/// The footprint drives both the trilinear LOD (minor axis) and the number of
/// anisotropic taps (major/minor ratio), independently of how the centre UV is
/// wrapped. See the [module docs](self) for the fixed resolve order and the
/// per-tap wrapping contract.
#[must_use]
pub fn resolve_differential(
    vt: VirtualTexture,
    req: &SampleRequest,
    rd: RayDifferential,
) -> SampleResolved {
    let addressed = address_uv(req.uv, req.wrap_u, req.wrap_v);
    let w = vt.width();
    let h = vt.height();
    let aniso = rd.anisotropic_mip(w, h, vt.max_mip_f32(), req.max_anisotropy);
    let major = rd.major_axis_uv(w, h);
    let taps = anisotropic_taps(addressed.uv, major, &aniso);
    let pages = vt.residency(addressed.uv, aniso.lod);
    SampleResolved {
        border: addressed.border,
        lod: aniso.lod,
        pages,
        taps,
    }
}

/// Resolve a sample for a secondary ray carrying a scalar ray cone (isotropic
/// trilinear). A single centre tap is produced; the LOD comes from the cone
/// width projected onto the surface via the per-triangle texel/world ratio.
///
/// * `tri` — the shaded triangle's texel/world LOD constant (Delta).
/// * `cone_width` — the ray-cone width at the hit (world units).
/// * `n_dot_d` — dot of the surface normal and ray direction (grazing-safe).
#[must_use]
pub fn resolve_cone(
    vt: VirtualTexture,
    req: &SampleRequest,
    tri: TriangleLodConstant,
    cone_width: f32,
    n_dot_d: f32,
) -> SampleResolved {
    let addressed = address_uv(req.uv, req.wrap_u, req.wrap_v);
    let lod = cone_mip_level(tri, cone_width, n_dot_d, vt.max_mip_f32());
    // A ray cone is isotropic: a single centre tap at the resolved LOD.
    let iso = AnisotropicMip {
        lod,
        anisotropy: 1.0,
    };
    let taps = anisotropic_taps(addressed.uv, [0.0, 0.0], &iso);
    let pages = vt.residency(addressed.uv, lod);
    SampleResolved {
        border: addressed.border,
        lod,
        pages,
        taps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_addressing::WrapMode;

    fn vt() -> VirtualTexture {
        VirtualTexture::new(256, 256, 128)
    }

    #[test]
    fn differential_isotropic_footprint_single_tap() {
        // 1 texel/pixel on both axes -> LOD 0, anisotropy 1 -> one centre tap.
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([0.5, 0.5], WrapMode::Repeat, 16.0);
        let r = resolve_differential(vt(), &req, rd);
        assert!(r.lod.abs() < 1.0e-5, "lod={}", r.lod);
        assert_eq!(r.taps.len(), 1);
        assert_eq!(r.taps.uvs()[0], [0.5, 0.5]);
        assert!(!r.border);
    }

    #[test]
    fn differential_anisotropic_footprint_multiple_taps() {
        // 8:1 footprint -> anisotropy ~8 -> several taps along the major axis.
        let rd = RayDifferential::new([8.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([0.5, 0.5], WrapMode::Repeat, 16.0);
        let r = resolve_differential(vt(), &req, rd);
        assert!(r.taps.len() >= 8, "taps={}", r.taps.len());
        // Taps spread along U (the major axis here).
        let min_u = r.taps.uvs().iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_u = r.taps.uvs().iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        assert!(max_u - min_u > 0.0);
    }

    #[test]
    fn differential_respects_max_anisotropy_cap() {
        let rd = RayDifferential::new([16.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        // Cap anisotropy at 2 -> at most ceil(2) == 2 taps despite 16:1 footprint.
        let req = SampleRequest::new([0.5, 0.5], WrapMode::Repeat, 2.0);
        let r = resolve_differential(vt(), &req, rd);
        assert!(r.taps.len() <= 2, "taps={}", r.taps.len());
    }

    #[test]
    fn differential_wraps_center_before_residency() {
        // u = 1.25 under Repeat folds to 0.25; residency must target that page.
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([1.25, 0.5], WrapMode::Repeat, 1.0);
        let r = resolve_differential(vt(), &req, rd);
        // 0.25 lands in the first (left) 128-texel page column.
        assert_eq!(r.pages[0].page_x, 0);
        assert!((r.taps.uvs()[0][0] - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn border_flag_propagates() {
        let rd = RayDifferential::new([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0]);
        let req = SampleRequest::new([1.5, 0.5], WrapMode::ClampToBorder, 1.0);
        let r = resolve_differential(vt(), &req, rd);
        assert!(r.border);
    }

    #[test]
    fn cone_is_isotropic_single_tap() {
        let tri = TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            256,
            256,
        );
        let req = SampleRequest::isotropic([0.5, 0.5], WrapMode::ClampToEdge);
        let r = resolve_cone(vt(), &req, tri, 0.01, 1.0);
        assert_eq!(r.taps.len(), 1);
        assert_eq!(r.taps.uvs()[0], [0.5, 0.5]);
        assert!(r.lod.is_finite());
        assert_eq!(r.pages[0].mip, r.taps.lod().floor() as u32);
    }

    #[test]
    fn cone_wider_cone_selects_coarser_lod() {
        let tri = TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            256,
            256,
        );
        let req = SampleRequest::isotropic([0.5, 0.5], WrapMode::ClampToEdge);
        let narrow = resolve_cone(vt(), &req, tri, 0.001, 1.0);
        let wide = resolve_cone(vt(), &req, tri, 0.1, 1.0);
        assert!(wide.lod >= narrow.lod, "wide={} narrow={}", wide.lod, narrow.lod);
    }

    #[test]
    fn cone_grazing_angle_stays_finite() {
        let tri = TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            256,
            256,
        );
        let req = SampleRequest::isotropic([0.5, 0.5], WrapMode::ClampToEdge);
        // n_dot_d -> 0 (grazing) must clamp, not blow up to inf/NaN.
        let r = resolve_cone(vt(), &req, tri, 0.01, 0.0);
        assert!(r.lod.is_finite());
    }
}
