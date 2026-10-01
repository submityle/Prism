//! Stylized (NPR) `MatCap` (material-capture / lit-sphere) shading for hair.
//!
//! A `MatCap` bakes an entire lighting environment *and* a material response
//! into a single pre-lit sphere texture that is indexed purely by the
//! view-space surface normal. It is a workhorse of the anime / stylized
//! pipeline (Blender's "matcap" viewport, Genshin-style character sheens,
//! Sketchfab's lit-sphere shading) because an artist can hand-paint the exact
//! rim, sheen and terminator shape they want and get a consistent, camera-
//! stable read with no runtime lights at all. For hair it complements the
//! analytic [`crate::stylized_hair`] / [`crate::hair_angel_ring`] bands: the
//! `MatCap` supplies the soft base body shade and ambient wrap while the
//! angel-ring stack adds the crisp additive highlight rings on top.
//!
//! This module owns the *sampling math* as a deterministic CPU golden: the
//! view-basis reconstruction that turns a world-space [`ShadingFrame`] into a
//! `MatCap` texture coordinate, and the clamp-to-edge bilinear fetch from a
//! caller-owned [`MatCapTexture`]. The real GPU path samples a bound
//! `texture_2d` through `hair_matcap.wesl`; the container has no GPU, so the UV
//! transform and the bilinear filter live here builtin-for-builtin so a green
//! golden here plus a green WESL compile pins both sides of the contract. Actual
//! texel parity is confirmed on real hardware.
//!
//! **View-basis reconstruction.** A `MatCap` is classically indexed by the
//! normal expressed in *view* space (`uv = n_view.xy * 0.5 + 0.5`). The shading
//! frame only carries world-space vectors, so rather than demand a camera
//! matrix this module rebuilds a screen basis from the view direction `V` (the
//! surface-to-camera vector) and a world-space `up_hint`: screen-right is
//! `normalize(cross(up_hint, V))`, screen-up is `cross(V, right)`, and the
//! `MatCap` coordinates are the normal projected onto that basis. This is the
//! standard "cheap `MatCap`" reconstruction and is fully view-stable: rotating
//! the head rotates the sampled highlight exactly as a real lit sphere would.
//! A degenerate `up_hint` (parallel to `V` or zero) falls back to the frame
//! tangent so the basis never collapses.
//!
//! Nothing samples a real random source and no input panics: an empty texture
//! returns black, a zero-length view or up hint degrades gracefully, and UVs are
//! clamped to `[0, 1]` before the fetch. Emissive is added exactly once so the
//! resolve integrator can composite the `MatCap` body shade without double
//! counting.

use alloc::vec::Vec;

use crate::vecmath::{cross, dot, mul, mul_scalar, normalize_or};
use crate::{ShadingFrame, SurfaceSample};

/// Squared-length floor below which a reconstructed basis vector is treated as
/// degenerate. Mirrors the `normalize_or` epsilon used across the shading
/// reference and the `1e-12` guard in the WESL twin.
const BASIS_EPSILON: f32 = 1e-12;

/// A pre-lit `MatCap` sphere texture stored as row-major linear-RGB texels.
///
/// Texels are addressed `texels[y * width + x]` with `x` increasing rightward
/// and `y` increasing downward (row `0` is the top edge, matching the
/// `MatCap` art convention where the top of the sphere is the top of the
/// image). [`Self::sample_bilinear`] does clamp-to-edge bilinear filtering, so
/// UVs at or beyond the border replicate the edge texel rather than wrapping.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatCapTexture {
    width: usize,
    height: usize,
    texels: Vec<[f32; 3]>,
}

impl MatCapTexture {
    /// Builds a `MatCap` texture from row-major linear-RGB texels.
    ///
    /// Returns an empty texture (sampling black) when `width`/`height` is `0`
    /// or when `texels.len()` does not cover `width * height`; this keeps the
    /// constructor total and panic-free for untrusted asset dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize, texels: Vec<[f32; 3]>) -> Self {
        if width == 0 || height == 0 || texels.len() < width.saturating_mul(height) {
            return Self::default();
        }
        Self {
            width,
            height,
            texels,
        }
    }

    /// Texture width in texels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Texture height in texels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// `true` when the texture carries no samplable texels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Fetches the nearest texel, clamping the integer coordinates to the valid
    /// range. Returns black for an empty texture.
    #[must_use]
    fn texel(&self, x: usize, y: usize) -> [f32; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        let cx = x.min(self.width - 1);
        let cy = y.min(self.height - 1);
        self.texels[cy * self.width + cx]
    }

    /// Clamp-to-edge bilinear sample at `uv` in `[0, 1]`.
    ///
    /// `uv` outside `[0, 1]` is clamped first, so the border texel is
    /// replicated (never wrapped). An empty texture samples black. The filter
    /// uses the pixel-center convention `p = uv * (dim - 1)` so `uv = 0` hits
    /// the first texel center and `uv = 1` hits the last.
    #[must_use]
    pub fn sample_bilinear(&self, uv: [f32; 2]) -> [f32; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        let u = uv[0].clamp(0.0, 1.0);
        let v = uv[1].clamp(0.0, 1.0);
        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        // `fx`/`fy` are already in `[0, dim-1]`, so the floors are non-negative
        // and finite; the `min` in `texel` guards the `+1` neighbour.
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x0 + 1, y0);
        let c01 = self.texel(x0, y0 + 1);
        let c11 = self.texel(x0 + 1, y0 + 1);
        let top = lerp3(c00, c10, tx);
        let bottom = lerp3(c01, c11, tx);
        lerp3(top, bottom, ty)
    }
}

/// Component-wise linear interpolation, matching the GPU `mix` intrinsic.
#[inline]
#[must_use]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Reconstructs the `MatCap` texture coordinate for one shaded fragment.
///
/// Builds a screen basis from the view direction and `up_hint`, projects the
/// (normalized) surface normal onto it, and maps the `[-1, 1]` projection to
/// `[0, 1]`. The result is always clamped to `[0, 1]`. A degenerate view or
/// `up_hint` falls back to the frame tangent (then `+X`) so the basis never
/// collapses and the function never returns `NaN`.
#[must_use]
pub fn hair_matcap_uv(frame: &ShadingFrame, up_hint: [f32; 3]) -> [f32; 2] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    // Screen-right is perpendicular to both the view and the up hint. When the
    // up hint is parallel to the view (or zero) `cross` collapses, so fall back
    // to the frame tangent and finally a fixed axis.
    let mut right = cross(up_hint, v);
    if dot(right, right) < BASIS_EPSILON {
        right = cross(frame.tangent, v);
    }
    let right = normalize_or(right, [1.0, 0.0, 0.0]);
    // Screen-up completes the right-handed basis; `v` and `right` are unit and
    // (near) orthogonal, so this is already unit length up to rounding.
    let up = normalize_or(cross(v, right), [0.0, 1.0, 0.0]);
    let x = dot(n, right);
    let y = dot(n, up);
    [
        (x * 0.5 + 0.5).clamp(0.0, 1.0),
        (y * 0.5 + 0.5).clamp(0.0, 1.0),
    ]
}

/// Authoring parameters for the hair `MatCap` body shade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairMatCapParams {
    /// Extra linear tint multiplied onto the sampled `MatCap` radiance (on top
    /// of the surface base color), letting per-strand color drive the shade.
    pub tint: [f32; 3],
    /// Scales the sampled `MatCap` radiance; `0` disables the body shade,
    /// leaving only emissive.
    pub strength: f32,
    /// World-space up hint used to rebuild the screen basis for the `MatCap`
    /// lookup (typically the camera or world up axis).
    pub up_hint: [f32; 3],
}

impl Default for HairMatCapParams {
    fn default() -> Self {
        Self {
            tint: [1.0; 3],
            strength: 1.0,
            up_hint: [0.0, 1.0, 0.0],
        }
    }
}

/// Evaluates the `MatCap` body shade for one hair fragment.
///
/// Samples `matcap` at the reconstructed view-space UV, modulates it by the
/// surface base color, the authored `tint` and `strength`, and adds the surface
/// emissive exactly once. This is a view-only lobe (no analytic light term): it
/// supplies the ambient/body shade that the additive angel-ring highlights and
/// cel ramp compose on top of. Negative `strength` is clamped to `0`.
#[must_use]
pub fn evaluate_hair_matcap(
    surface: &SurfaceSample,
    frame: &ShadingFrame,
    params: &HairMatCapParams,
    matcap: &MatCapTexture,
) -> [f32; 3] {
    let strength = params.strength.max(0.0);
    let uv = hair_matcap_uv(frame, params.up_hint);
    let sampled = matcap.sample_bilinear(uv);
    let tinted = mul(mul(sampled, surface.base_color), params.tint);
    let body = mul_scalar(tinted, strength);
    [
        body[0] + surface.emissive[0],
        body[1] + surface.emissive[1],
        body[2] + surface.emissive[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn frame(normal: [f32; 3], view: [f32; 3]) -> ShadingFrame {
        ShadingFrame {
            normal,
            view,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        }
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < EPS && (a[1] - b[1]).abs() < EPS && (a[2] - b[2]).abs() < EPS
    }

    /// A 2x2 texture with a distinct color per corner exercises bilinear blends.
    fn quad() -> MatCapTexture {
        MatCapTexture::new(
            2,
            2,
            alloc::vec![
                [1.0, 0.0, 0.0], // (0,0) top-left
                [0.0, 1.0, 0.0], // (1,0) top-right
                [0.0, 0.0, 1.0], // (0,1) bottom-left
                [1.0, 1.0, 0.0], // (1,1) bottom-right
            ],
        )
    }

    #[test]
    fn empty_texture_samples_black_and_reports_empty() {
        let tex = MatCapTexture::new(0, 4, alloc::vec![]);
        assert!(tex.is_empty());
        assert!(close(tex.sample_bilinear([0.5, 0.5]), [0.0, 0.0, 0.0]));
        // Dimensions that under-cover the texel buffer also fall back to empty.
        let short = MatCapTexture::new(4, 4, alloc::vec![[1.0; 3]; 3]);
        assert!(short.is_empty());
    }

    #[test]
    fn bilinear_hits_exact_texel_centers_at_the_corners() {
        let tex = quad();
        assert!(close(tex.sample_bilinear([0.0, 0.0]), [1.0, 0.0, 0.0]));
        assert!(close(tex.sample_bilinear([1.0, 0.0]), [0.0, 1.0, 0.0]));
        assert!(close(tex.sample_bilinear([0.0, 1.0]), [0.0, 0.0, 1.0]));
        assert!(close(tex.sample_bilinear([1.0, 1.0]), [1.0, 1.0, 0.0]));
    }

    #[test]
    fn bilinear_blends_at_the_center() {
        let tex = quad();
        let mid = tex.sample_bilinear([0.5, 0.5]);
        // Average of the four corners.
        assert!(close(mid, [0.5, 0.5, 0.25]));
    }

    #[test]
    fn sampling_clamps_to_edge_never_wraps() {
        let tex = quad();
        // Beyond the borders replicates the nearest edge texel.
        assert!(close(tex.sample_bilinear([-1.0, -1.0]), [1.0, 0.0, 0.0]));
        assert!(close(tex.sample_bilinear([2.0, 2.0]), [1.0, 1.0, 0.0]));
    }

    #[test]
    fn uv_is_centered_when_normal_faces_camera() {
        // Normal along +Z, view along +Z, up along +Y => normal projects to the
        // basis origin, uv = (0.5, 0.5).
        let f = frame([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let uv = hair_matcap_uv(&f, [0.0, 1.0, 0.0]);
        assert!((uv[0] - 0.5).abs() < EPS);
        assert!((uv[1] - 0.5).abs() < EPS);
    }

    #[test]
    fn uv_moves_with_the_normal_on_the_screen_basis() {
        // View +Z, up +Y => right = cross(up, view) = +X. A normal tilted toward
        // +X should push u above 0.5; tilted toward +Y should push v above 0.5.
        let right_tilt = frame([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let uv_r = hair_matcap_uv(&right_tilt, [0.0, 1.0, 0.0]);
        assert!(uv_r[0] > 0.5 + EPS);
        assert!((uv_r[1] - 0.5).abs() < EPS);

        let up_tilt = frame([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let uv_u = hair_matcap_uv(&up_tilt, [0.0, 1.0, 0.0]);
        assert!((uv_u[0] - 0.5).abs() < EPS);
        assert!(uv_u[1] > 0.5 + EPS);
    }

    #[test]
    fn degenerate_up_hint_falls_back_without_nan() {
        // Up hint parallel to the view collapses the primary cross product; the
        // tangent fallback keeps the basis finite and clamped.
        let f = frame([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let uv = hair_matcap_uv(&f, [0.0, 0.0, 1.0]);
        assert!(uv[0].is_finite() && uv[1].is_finite());
        assert!(uv[0] >= 0.0 && uv[0] <= 1.0);
        assert!(uv[1] >= 0.0 && uv[1] <= 1.0);
    }

    #[test]
    fn zero_view_and_up_do_not_panic() {
        let f = frame([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let uv = hair_matcap_uv(&f, [0.0, 0.0, 0.0]);
        assert!(uv[0].is_finite() && uv[1].is_finite());
    }

    #[test]
    fn evaluate_modulates_base_color_and_tint() {
        let surface = SurfaceSample {
            base_color: [0.5, 0.5, 0.5],
            emissive: [0.0; 3],
            ..Default::default()
        };
        let f = frame([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        // Uniform white texture so the sample is [1,1,1] everywhere.
        let tex = MatCapTexture::new(2, 2, alloc::vec![[1.0; 3]; 4]);
        let params = HairMatCapParams {
            tint: [1.0, 0.5, 0.25],
            strength: 1.0,
            up_hint: [0.0, 1.0, 0.0],
        };
        let out = evaluate_hair_matcap(&surface, &f, &params, &tex);
        // 1 (sample) * 0.5 (base) * tint.
        assert!(close(out, [0.5, 0.25, 0.125]));
    }

    #[test]
    fn evaluate_adds_emissive_exactly_once() {
        let surface = SurfaceSample {
            base_color: [1.0; 3],
            emissive: [0.1, 0.2, 0.3],
            ..Default::default()
        };
        let f = frame([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let tex = MatCapTexture::new(2, 2, alloc::vec![[1.0; 3]; 4]);
        // Zero strength removes the body shade, leaving only emissive.
        let params = HairMatCapParams {
            tint: [1.0; 3],
            strength: 0.0,
            up_hint: [0.0, 1.0, 0.0],
        };
        let out = evaluate_hair_matcap(&surface, &f, &params, &tex);
        assert!(close(out, [0.1, 0.2, 0.3]));
    }

    #[test]
    fn negative_strength_is_clamped_to_zero() {
        let surface = SurfaceSample::default();
        let f = frame([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let tex = MatCapTexture::new(2, 2, alloc::vec![[1.0; 3]; 4]);
        let params = HairMatCapParams {
            tint: [1.0; 3],
            strength: -5.0,
            up_hint: [0.0, 1.0, 0.0],
        };
        let out = evaluate_hair_matcap(&surface, &f, &params, &tex);
        assert!(close(out, surface.emissive));
    }

    #[test]
    fn evaluate_is_idempotent_for_fixed_inputs() {
        let surface = SurfaceSample::default();
        let f = frame([0.2, 0.3, 0.9], [0.1, 0.0, 1.0]);
        let tex = quad();
        let params = HairMatCapParams::default();
        let a = evaluate_hair_matcap(&surface, &f, &params, &tex);
        let b = evaluate_hair_matcap(&surface, &f, &params, &tex);
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }
}
