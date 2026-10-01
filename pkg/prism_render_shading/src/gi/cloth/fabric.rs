//! Combined cloth BRDF: a sheen lobe layered over a subsurface-tinted diffuse.
//!
//! Golden CPU reference for a complete fabric surface response. It composes the
//! fabric sheen lobe from [`super::charlie`] or [`super::velvet`] with a cloth
//! diffuse that is softened by *wrap* lighting and tinted toward a *subsurface*
//! colour, approximating the shallow light transport through yarn.
//!
//! The response is split into two complementary entry points:
//!
//! * [`cloth_brdf`] — the reciprocal BRDF `f_r(wi, wo)` with no cosine
//!   foreshortening and no directional wrap. It is a true BRDF (symmetric in
//!   `wi`/`wo`) made of a tinted Lambert diffuse plus the sheen lobe, suitable
//!   wherever a BRDF value is required (importance sampling, MIS, LUT bakes).
//! * [`cloth_shade`] — the shading-time outgoing weight that additionally
//!   applies the directional [`cloth_diffuse_wrap`] foreshortening to the
//!   diffuse term and the `n·l` cosine to the sheen term. This is what a direct
//!   lighting loop accumulates per light.
//!
//! Keeping wrap in the *foreshortening* (and only an energy renormalisation in
//! the BRDF) means [`cloth_brdf`] stays strictly reciprocal while the visible
//! soft, waxy diffuse of cloth is still reproduced by [`cloth_shade`].
//!
//! # Conventions
//! * `wi` (toward the light), `wo` (toward the eye), and the shading normal `n`
//!   are unit vectors in a shared world/tangent frame. Cosines are dot products
//!   with `n`; back-facing configurations return zero.
//! * Colours (`diffuse_color`, `subsurface_color`, `sheen_color`) are linear
//!   reflectances clamped componentwise to `[0, 1]` so the diffuse albedo can
//!   never exceed unity (energy clamp).
//! * `wrap ∈ [0, 1]` widens the diffuse terminator; `sheen_roughness ∈ [0, 1]`.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. Every routine is a deterministic pure function (no RNG,
//!   I/O, GPU, globals, or `unsafe`) and clamps defensively so it never returns
//!   `NaN` or infinity.
//!
//! # References
//! * Estevez & Kulla 2017, *Production Friendly Microfacet Sheen BRDF*.
//! * Khronos `KHR_materials_sheen`.
//! * Gotanda / Burley wrapped-diffuse and subsurface-tint approximations.

use bevy_math::Vec3;
use core::f32::consts::FRAC_1_PI;

use super::{charlie, velvet};

/// Which microfacet sheen lobe backs the cloth response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheenModel {
    /// Estevez-Kulla "Charlie" lobe with the Neubelt visibility.
    Charlie,
    /// Estevez-Kulla "Charlie" lobe with the soft-shadowing visibility.
    CharlieSoft,
    /// Ashikhmin inverted-Gaussian velvet lobe.
    Velvet,
}

impl Default for SheenModel {
    #[inline]
    fn default() -> Self {
        SheenModel::Charlie
    }
}

/// Parameters describing a fabric surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothParams {
    /// Base diffuse reflectance (linear, clamped to `[0, 1]` on use).
    pub diffuse_color: Vec3,
    /// Deep subsurface tint the diffuse blends toward (linear, clamped).
    pub subsurface_color: Vec3,
    /// Blend `[0, 1]` from `diffuse_color` toward `subsurface_color`.
    pub subsurface_blend: f32,
    /// Sheen rim reflectance (linear, clamped to `[0, 1]` on use).
    pub sheen_color: Vec3,
    /// Sheen roughness `[0, 1]`.
    pub sheen_roughness: f32,
    /// Diffuse wrap `[0, 1]`; `0` is Lambert, larger values soften the shadow
    /// terminator and wrap light around the surface.
    pub wrap: f32,
    /// Which sheen lobe to evaluate.
    pub model: SheenModel,
}

impl Default for ClothParams {
    #[inline]
    fn default() -> Self {
        ClothParams {
            diffuse_color: Vec3::splat(0.5),
            subsurface_color: Vec3::splat(0.5),
            subsurface_blend: 0.0,
            sheen_color: Vec3::splat(0.2),
            sheen_roughness: 0.3,
            wrap: 0.0,
            model: SheenModel::Charlie,
        }
    }
}

impl ClothParams {
    /// The effective, energy-clamped diffuse tint:
    /// `lerp(diffuse_color, subsurface_color, blend)` with every component in
    /// `[0, 1]`.
    #[inline]
    pub fn tinted_albedo(&self) -> Vec3 {
        let t = self.subsurface_blend.clamp(0.0, 1.0);
        let base = clamp01(self.diffuse_color);
        let sss = clamp01(self.subsurface_color);
        clamp01(base + (sss - base) * t)
    }
}

/// Directional wrapped-diffuse foreshortening `saturate((n·l + w)/(1 + w))`.
///
/// Replaces the raw `max(n·l, 0)` cosine for cloth so light bleeds slightly past
/// the terminator (`w > 0`). `w` is clamped to `[0, 1]`; the result lies in
/// `[0, 1]` and reduces to `max(n·l, 0)` when `w = 0`.
#[inline]
pub fn cloth_diffuse_wrap(n_dot_l: f32, wrap: f32) -> f32 {
    let w = wrap.clamp(0.0, 1.0);
    let ndl = n_dot_l.clamp(-1.0, 1.0);
    let v = (ndl + w) / (1.0 + w);
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Scalar energy renormalisation for the wrapped diffuse.
///
/// Spreading the diffuse response over a wider terminator would otherwise add
/// energy, so the Lambert lobe is scaled by `1/(1 + w)` (the factor that keeps
/// the wrapped half-cosine integral unit-bounded). Direction-independent, so it
/// does not affect reciprocity of [`cloth_brdf`]. `w` is clamped to `[0, 1]`.
#[inline]
pub fn wrap_energy_norm(wrap: f32) -> f32 {
    1.0 / (1.0 + wrap.clamp(0.0, 1.0))
}

/// Evaluates the selected colourless sheen lobe `D·V` from cosines.
#[inline]
fn sheen_lobe(model: SheenModel, mu_v: f32, mu_l: f32, mu_h: f32, roughness: f32) -> f32 {
    match model {
        SheenModel::Charlie => charlie::sheen_lobe_neubelt(mu_v, mu_l, mu_h, roughness),
        SheenModel::CharlieSoft => charlie::sheen_lobe_charlie(mu_v, mu_l, mu_h, roughness),
        SheenModel::Velvet => velvet::velvet_lobe(mu_v, mu_l, mu_h, roughness),
    }
}

/// The reciprocal cloth BRDF `f_r(wi, wo)` (no cosine foreshortening).
///
/// `f_r = tinted_albedo·(1/π)·wrap_norm + sheen_color·D·V`. The diffuse term is
/// a tinted Lambert scaled by [`wrap_energy_norm`] (a direction-independent
/// scalar, so reciprocity is preserved), and the sheen term is the symmetric
/// lobe `D·V` from [`super::charlie`] / [`super::velvet`]. Returns [`Vec3::ZERO`]
/// for a below-horizon configuration (`n·l ≤ 0` or `n·v ≤ 0`). Every component
/// is finite and non-negative.
#[inline]
pub fn cloth_brdf(wi: Vec3, wo: Vec3, n: Vec3, params: &ClothParams) -> Vec3 {
    let nn = n.normalize_or_zero();
    if nn.length_squared() <= 0.0 {
        return Vec3::ZERO;
    }
    let mu_l = nn.dot(wi);
    let mu_v = nn.dot(wo);
    if mu_l <= 0.0 || mu_v <= 0.0 {
        return Vec3::ZERO;
    }
    // Tinted Lambert diffuse, wrap-renormalised (reciprocal: no direction term).
    let diffuse = params.tinted_albedo() * (FRAC_1_PI * wrap_energy_norm(params.wrap));

    // Symmetric sheen lobe D·V, tinted by the sheen colour.
    let h = (wi + wo).normalize_or_zero();
    let sheen = if h.length_squared() > 0.0 {
        let mu_h = nn.dot(h);
        let lobe = sheen_lobe(params.model, mu_v, mu_l, mu_h, params.sheen_roughness);
        clamp01(params.sheen_color) * lobe
    } else {
        Vec3::ZERO
    };

    sanitize(diffuse + sheen)
}

/// Shading-time outgoing weight for one light: the cloth response folded with
/// its foreshortening.
///
/// `out = tinted_albedo·(1/π)·wrap_norm·wrap_cosine + sheen_color·D·V·(n·l)`.
/// The diffuse uses the directional [`cloth_diffuse_wrap`] foreshortening (soft
/// terminator) while the sheen uses the ordinary `n·l`. Multiply the result by
/// incident radiance to get reflected radiance. Returns [`Vec3::ZERO`] below the
/// horizon; every component is finite and non-negative.
#[inline]
pub fn cloth_shade(wi: Vec3, wo: Vec3, n: Vec3, params: &ClothParams) -> Vec3 {
    let nn = n.normalize_or_zero();
    if nn.length_squared() <= 0.0 {
        return Vec3::ZERO;
    }
    let mu_l = nn.dot(wi);
    let mu_v = nn.dot(wo);
    if mu_v <= 0.0 {
        return Vec3::ZERO;
    }
    // Diffuse: wrap foreshortening lets a little light past the terminator even
    // when n·l is slightly negative, so it is evaluated before the mu_l gate.
    let wrap_cos = cloth_diffuse_wrap(mu_l, params.wrap);
    let diffuse =
        params.tinted_albedo() * (FRAC_1_PI * wrap_energy_norm(params.wrap) * wrap_cos);

    // Sheen only contributes for an above-horizon light.
    let sheen = if mu_l > 0.0 {
        let h = (wi + wo).normalize_or_zero();
        if h.length_squared() > 0.0 {
            let mu_h = nn.dot(h);
            let lobe = sheen_lobe(params.model, mu_v, mu_l, mu_h, params.sheen_roughness);
            clamp01(params.sheen_color) * (lobe * mu_l)
        } else {
            Vec3::ZERO
        }
    } else {
        Vec3::ZERO
    };

    sanitize(diffuse + sheen)
}

/// Componentwise clamp of a colour/vector to `[0, 1]`.
#[inline]
fn clamp01(v: Vec3) -> Vec3 {
    Vec3::new(
        v.x.clamp(0.0, 1.0),
        v.y.clamp(0.0, 1.0),
        v.z.clamp(0.0, 1.0),
    )
}

/// Replaces any non-finite component with `0` and floors negatives at `0`.
#[inline]
fn sanitize(v: Vec3) -> Vec3 {
    let f = |x: f32| if x.is_finite() { x.max(0.0) } else { 0.0 };
    Vec3::new(f(v.x), f(v.y), f(v.z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    fn basic_params() -> ClothParams {
        ClothParams {
            diffuse_color: Vec3::new(0.6, 0.4, 0.3),
            subsurface_color: Vec3::new(0.8, 0.2, 0.1),
            subsurface_blend: 0.4,
            sheen_color: Vec3::new(0.3, 0.3, 0.35),
            sheen_roughness: 0.4,
            wrap: 0.0,
            model: SheenModel::Charlie,
        }
    }

    /// With `wrap = 0` the BRDF is a genuine BRDF and must be reciprocal:
    /// swapping light and view leaves `f_r` unchanged for every sheen model.
    #[test]
    fn cloth_brdf_reciprocal() {
        let n = Vec3::Z;
        let wi = Vec3::new(0.2, 0.1, 0.974).normalize();
        let wo = Vec3::new(-0.3, 0.25, 0.92).normalize();
        for model in [SheenModel::Charlie, SheenModel::CharlieSoft, SheenModel::Velvet] {
            let mut p = basic_params();
            p.model = model;
            let a = cloth_brdf(wi, wo, n, &p);
            let b = cloth_brdf(wo, wi, n, &p);
            assert!((a - b).length() < 1.0e-6, "{model:?}: {a:?} vs {b:?}");
        }
    }

    /// The pure diffuse albedo is energy-clamped: `∫ f_d·cosθ dω = albedo ≤ 1`
    /// per channel (sheen set to zero, wrap zero).
    #[test]
    fn cloth_diffuse_albedo_bounded() {
        let n = Vec3::Z;
        let wo = Vec3::new(0.3, 0.0, 0.954).normalize();
        let mut p = basic_params();
        p.sheen_color = Vec3::ZERO;
        p.subsurface_blend = 0.0;
        p.wrap = 0.0;
        // Cosine-weighted hemisphere integral of the BRDF (uniform lattice).
        let samples = 40_000usize;
        let golden = core::f32::consts::PI * (3.0 - (5.0f32).sqrt());
        let mut acc = Vec3::ZERO;
        for i in 0..samples {
            let cos_t = (i as f32 + 0.5) / samples as f32;
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let phi = golden * i as f32;
            let (sp, cp) = ops::sin_cos(phi);
            let wi = Vec3::new(sin_t * cp, sin_t * sp, cos_t);
            acc += cloth_brdf(wi, wo, n, &p) * cos_t;
        }
        // Uniform-hemisphere weight 2π/N.
        let albedo = acc * (core::f32::consts::TAU / samples as f32);
        let expect = p.tinted_albedo();
        assert!(
            (albedo.x - expect.x).abs() < 0.02
                && albedo.x <= 1.0 + 1.0e-4
                && albedo.y <= 1.0 + 1.0e-4
                && albedo.z <= 1.0 + 1.0e-4,
            "albedo={albedo:?} expect={expect:?}"
        );
    }

    #[test]
    fn tinted_albedo_blends_and_clamps() {
        let p = ClothParams {
            diffuse_color: Vec3::new(1.0, 0.0, 0.0),
            subsurface_color: Vec3::new(0.0, 1.0, 0.0),
            subsurface_blend: 0.5,
            ..basic_params()
        };
        let t = p.tinted_albedo();
        assert!((t - Vec3::new(0.5, 0.5, 0.0)).length() < 1.0e-6, "t={t:?}");
        // Out-of-range colours are clamped before blending.
        let p2 = ClothParams {
            diffuse_color: Vec3::splat(5.0),
            subsurface_color: Vec3::splat(-2.0),
            subsurface_blend: 0.0,
            ..basic_params()
        };
        assert_eq!(p2.tinted_albedo(), Vec3::ONE);
    }

    #[test]
    fn wrap_softens_terminator() {
        // At the terminator (n·l = 0) wrap lets light through; Lambert gives 0.
        assert_eq!(cloth_diffuse_wrap(0.0, 0.0), 0.0);
        assert!(cloth_diffuse_wrap(0.0, 0.5) > 0.0);
        // Just past the terminator wrap is still positive.
        assert!(cloth_diffuse_wrap(-0.2, 0.5) > 0.0);
        // Full wrap keeps the head-on value within unit range.
        assert!((0.0..=1.0).contains(&cloth_diffuse_wrap(1.0, 1.0)));
    }

    #[test]
    fn wrap_energy_norm_reduces_with_wrap() {
        assert!((wrap_energy_norm(0.0) - 1.0).abs() < 1.0e-7);
        assert!(wrap_energy_norm(1.0) < wrap_energy_norm(0.0));
        assert!(wrap_energy_norm(1.0) > 0.0);
    }

    #[test]
    fn cloth_shade_non_negative_and_finite() {
        let n = Vec3::Z;
        let wo = Vec3::new(0.2, 0.1, 0.974).normalize();
        let mut p = basic_params();
        p.wrap = 0.3;
        for model in [SheenModel::Charlie, SheenModel::CharlieSoft, SheenModel::Velvet] {
            p.model = model;
            for i in 0..32 {
                let a = i as f32 / 32.0 * core::f32::consts::PI;
                let (s, c) = ops::sin_cos(a);
                let wi = Vec3::new(s, 0.0, c);
                let out = cloth_shade(wi, wo, n, &p);
                assert!(
                    out.is_finite() && out.min_element() >= 0.0,
                    "{model:?} out={out:?} i={i}"
                );
            }
        }
    }

    #[test]
    fn below_horizon_is_zero() {
        let n = Vec3::Z;
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let wi = Vec3::new(0.0, 0.0, 1.0);
        let p = basic_params();
        // View below horizon → zero for both entry points.
        assert_eq!(cloth_brdf(wi, Vec3::NEG_Z, n, &p), Vec3::ZERO);
        assert_eq!(cloth_shade(wi, Vec3::NEG_Z, n, &p), Vec3::ZERO);
        // Light below horizon → BRDF zero (shade may still have wrap diffuse at 0 wrap = 0).
        assert_eq!(cloth_brdf(Vec3::NEG_Z, wo, n, &p), Vec3::ZERO);
    }

    #[test]
    fn sheen_increases_total_response() {
        let n = Vec3::Z;
        // Asymmetric geometry so the half vector is off-normal and the
        // grazing-peaked sheen NDF is non-zero.
        let wi = Vec3::new(0.95, 0.0, 0.312).normalize();
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let mut no_sheen = basic_params();
        no_sheen.sheen_color = Vec3::ZERO;
        let with_sheen = basic_params();
        let a = cloth_brdf(wi, wo, n, &no_sheen);
        let b = cloth_brdf(wi, wo, n, &with_sheen);
        assert!(b.length() > a.length(), "a={a:?} b={b:?}");
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        let p = basic_params();
        let n = Vec3::Z;
        assert_eq!(cloth_brdf(Vec3::ZERO, Vec3::Z, n, &p), Vec3::ZERO);
        assert_eq!(cloth_brdf(Vec3::Z, Vec3::Z, Vec3::ZERO, &p), Vec3::ZERO);
        let out = cloth_shade(Vec3::Z, Vec3::Z, n, &p);
        assert!(out.is_finite());
        assert!(cloth_diffuse_wrap(f32::NAN, 0.5).is_finite());
    }
}
