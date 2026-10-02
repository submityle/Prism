//! Rough diffuse `BRDF` reference for particle shading — trig-free Oren-Nayar
//! and Burley / Disney diffuse (design §17 `PBR` diffuse, §18 stylized base).
//!
//! Lambert is the cheap default, but real rough dielectrics (dust, chalk, dry
//! smoke, clay-like debris) do not darken uniformly toward grazing angles: they
//! *retro-reflect*, staying flat and even brightening at the silhouette. This
//! module owns the `CPU` reference for two production rough-diffuse closures so
//! a future `GPU` draw kernel (and the stylized `NPR` base of §18, which reads
//! the same shared lighting data) can match it bit for bit. It is the diffuse
//! dual of the specular pieces owned by [`super::specular_aa`] and
//! [`super::fresnel_rim`]; it deliberately does not evaluate any specular lobe.
//!
//! Two closures compose here, both transcendental-free:
//!
//! 1. **Oren-Nayar (trig-free form)** — the classic Oren-Nayar qualitative
//!    model (Oren & Nayar, *Generalization of Lambert's Reflectance Model*,
//!    `SIGGRAPH` 1994) is normally written with `acos`/`sin`/`tan` of the
//!    incident and exitant angles. This module uses the algebraically identical
//!    direction-only rewrite popularized by `Fujii` and by Gotanda
//!    (`CEDEC`/`SIGGRAPH` course notes), which expresses the same `A + B` lobe
//!    purely from dot products: with `NoL`, `NoV` the clamped cosines and
//!    `LoV = dot(L, V)`, let `s = LoV - NoL * NoV` and
//!    `t = if s > 0 { max(NoL, NoV) } else { 1 }`. The roughness `sigma` (the
//!    slope-distribution standard deviation, in radians) feeds the standard
//!    coefficients `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)` and
//!    `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`, giving reflected radiance
//!    `albedo / pi * NoL * (A + B * s / t)`. Only `dot`, `max`, and division are
//!    used — no trigonometry — so the result is deterministic and portable (see
//!    [`oren_nayar_coeffs`] and [`oren_nayar`]).
//! 2. **Burley / Disney diffuse** — the roughness-aware retro-reflection term
//!    from Burley's *Physically-Based Shading at Disney* (`SIGGRAPH` 2012):
//!    with the half vector `H = normalize(L + V)` and `LoH = dot(L, H)`,
//!    `FD90 = 0.5 + 2 * roughness * LoH^2` drives two `Schlick`-style grazing
//!    lobes,
//!    `f = albedo / pi * (1 + (FD90 - 1) * (1 - NoL)^5) * (1 + (FD90 - 1) * (1 - NoV)^5)`.
//!    The fifth power reuses the integer
//!    multiply loop [`super::fresnel_rim::power_u32`] rather than `powf`, so it
//!    stays transcendental-free (see [`burley_diffuse`]).
//!
//! Both closures take unit-length [`Vec3`] directions and a linear-`RGB`
//! `albedo`, return a non-negative linear-`RGB` [`Vec3`], and gate to zero when
//! the surface is back-lit or back-facing (`NoL <= 0` or `NoV <= 0`). Only
//! `sqrt` (through [`Vec3::normalize_or_zero`]) and rational arithmetic are
//! used, matching the workspace determinism lint.

use super::Vec3;

/// Reciprocal of pi, `1 / pi`, used to normalize the diffuse lobes so a white
/// Lambert surface integrates to unit albedo. Spelled as a constant because the
/// crate is transcendental-free and cannot call a pi-returning intrinsic.
const INV_PI: f32 = core::f32::consts::FRAC_1_PI;

/// Classic Oren-Nayar denominator offset for the `A` (base) coefficient; from
/// the `0.33` term in `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)`.
const A_OFFSET: f32 = 0.33;

/// Classic Oren-Nayar denominator offset for the `B` (inter-reflection) term;
/// from the `0.09` term in `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`.
const B_OFFSET: f32 = 0.09;

/// Leading factor of the Oren-Nayar `A` coefficient (`0.5`).
const A_SCALE: f32 = 0.5;

/// Leading factor of the Oren-Nayar `B` coefficient (`0.45`).
const B_SCALE: f32 = 0.45;

/// Fifth power used by both grazing lobes of the Burley diffuse term.
const BURLEY_POWER: u32 = 5;

/// Minimum denominator for the Oren-Nayar `s / t` ratio, guarding the
/// degenerate case where `s > 0` yet `max(NoL, NoV)` is (numerically) zero —
/// which would otherwise yield `inf * 0 = NaN`. The result is multiplied by
/// `NoL` afterward, so clamping `t` here only removes the `NaN`, never changes
/// a visible value.
const MIN_T: f32 = 1e-6;

/// The Oren-Nayar `A` (base lobe) and `B` (inter-reflection lobe) coefficients
/// for a slope standard deviation `sigma` (in radians).
///
/// `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)` and
/// `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`. At `sigma = 0` this is `(1, 0)`, so
/// the model collapses to Lambert; as `sigma` grows, `A` decreases and `B`
/// increases, strengthening the grazing retro-reflection. `sigma` is clamped to
/// be non-negative so a stray negative input cannot produce a negative lobe.
#[must_use]
pub fn oren_nayar_coeffs(sigma: f32) -> (f32, f32) {
    let s = sigma.max(0.0);
    let sigma_sq = s * s;
    let a = 1.0 - A_SCALE * sigma_sq / (sigma_sq + A_OFFSET);
    let b = B_SCALE * sigma_sq / (sigma_sq + B_OFFSET);
    (a, b)
}

/// Trig-free Oren-Nayar reflected radiance for a rough diffuse surface.
///
/// `normal`, `light`, and `view` are unit directions (the light and view point
/// *away* from the surface); `albedo` is linear-`RGB`; `sigma` is the roughness
/// (slope standard deviation, radians). Returns
/// `albedo / pi * NoL * (A + B * s / t)` with `s = dot(L, V) - NoL * NoV` and
/// `t = if s > 0 { max(NoL, NoV) } else { 1 }`. Back-lit or back-facing
/// geometry (`NoL <= 0` or `NoV <= 0`) returns [`Vec3::ZERO`]. At `sigma = 0`
/// this reduces exactly to Lambert, `albedo / pi * NoL`.
#[must_use]
pub fn oren_nayar(normal: Vec3, light: Vec3, view: Vec3, albedo: Vec3, sigma: f32) -> Vec3 {
    let n_dot_l = normal.dot(light).max(0.0);
    let n_dot_v = normal.dot(view).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Vec3::ZERO;
    }

    let (a, b) = oren_nayar_coeffs(sigma);
    let s = light.dot(view) - n_dot_l * n_dot_v;
    // Fujii/Gotanda direction-only denominator: the larger cosine when the
    // azimuthal term is positive, otherwise 1 (no attenuation).
    let t = if s > 0.0 {
        n_dot_l.max(n_dot_v).max(MIN_T)
    } else {
        1.0
    };

    let lobe = a + b * s / t;
    // The qualitative model keeps the lobe non-negative; clamp defends against
    // extreme sigma and off-unit inputs rather than ever lowering a valid value.
    let scale = INV_PI * n_dot_l * lobe.max(0.0);
    albedo.scale(scale)
}

/// Burley / Disney roughness-aware diffuse `BRDF`.
///
/// `normal`, `light`, and `view` are unit directions; `albedo` is linear-`RGB`;
/// `roughness` is the perceptual roughness in `0..=1`. Returns
/// `albedo / pi * (1 + (FD90 - 1) * (1 - NoL)^5) * (1 + (FD90 - 1) * (1 - NoV)^5)`
/// with
/// `FD90 = 0.5 + 2 * roughness * LoH^2`, the fifth power evaluated by
/// [`super::fresnel_rim::power_u32`]. This returns the reflectance (`BRDF`)
/// value, not reflected radiance, so the caller applies the `NoL` cosine of the
/// rendering equation. Back-lit or back-facing geometry returns [`Vec3::ZERO`].
/// At `roughness = 0` and head-on geometry it reduces to Lambert `albedo / pi`.
#[must_use]
pub fn burley_diffuse(normal: Vec3, light: Vec3, view: Vec3, albedo: Vec3, roughness: f32) -> Vec3 {
    let n_dot_l = normal.dot(light).max(0.0);
    let n_dot_v = normal.dot(view).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Vec3::ZERO;
    }

    let half = light.add(view).normalize_or_zero();
    let l_dot_h = light.dot(half).max(0.0);
    let r = roughness.max(0.0);
    // Disney retro-reflection target reflectance at grazing (`FD90`); the `2.0`
    // and `0.5` are the published Burley constants.
    let fd90 = 0.5 + 2.0 * r * (l_dot_h * l_dot_h);

    let fl = 1.0 + (fd90 - 1.0) * super::fresnel_rim::power_u32(1.0 - n_dot_l, BURLEY_POWER);
    let fv = 1.0 + (fd90 - 1.0) * super::fresnel_rim::power_u32(1.0 - n_dot_v, BURLEY_POWER);

    albedo.scale(INV_PI * fl * fv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    fn vec_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    #[test]
    fn coeffs_at_zero_sigma_are_lambert() {
        let (a, b) = oren_nayar_coeffs(0.0);
        assert!(approx_eq(a, 1.0));
        assert!(approx_eq(b, 0.0));
    }

    #[test]
    fn coeffs_are_monotone_in_sigma() {
        // A decreases, B increases as roughness grows.
        let (a0, b0) = oren_nayar_coeffs(0.1);
        let (a1, b1) = oren_nayar_coeffs(0.5);
        let (a2, b2) = oren_nayar_coeffs(1.0);
        assert!(a1 < a0);
        assert!(a2 < a1);
        assert!(b1 > b0);
        assert!(b2 > b1);
        // Both lobes stay bounded and non-negative.
        for &a in &[a0, a1, a2] {
            assert!((0.0..=1.0).contains(&a));
        }
        for &b in &[b0, b1, b2] {
            assert!(b >= 0.0);
        }
    }

    #[test]
    fn negative_sigma_is_clamped_to_lambert() {
        assert_eq!(oren_nayar_coeffs(-3.0), oren_nayar_coeffs(0.0));
    }

    #[test]
    fn oren_nayar_sigma_zero_is_lambert() {
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::new(0.8, 0.5, 0.2);
        // Several non-degenerate geometries must match Lambert exactly at sigma = 0.
        let cases = [
            (unit(0.0, 0.0, 1.0), unit(0.0, 0.0, 1.0)),
            (unit(0.3, 0.1, 0.9), unit(0.0, 0.2, 1.0)),
            (unit(-0.4, 0.2, 0.8), unit(0.5, -0.1, 0.7)),
        ];
        for (l, v) in cases {
            let got = oren_nayar(n, l, v, albedo, 0.0);
            let n_dot_l = n.dot(l).max(0.0);
            let lambert = albedo.scale(INV_PI * n_dot_l);
            assert!(vec_approx_eq(got, lambert));
        }
    }

    #[test]
    fn oren_nayar_grazing_is_brighter_with_roughness() {
        // Grazing retro-reflection: light and view nearly aligned near the
        // silhouette. A rough surface must be at least as bright as Lambert and
        // strictly brighter here, where the B lobe is positive.
        let n = unit(0.0, 0.0, 1.0);
        let dir = unit(0.98, 0.0, 0.2);
        let albedo = Vec3::splat(0.9);
        let smooth = oren_nayar(n, dir, dir, albedo, 0.0);
        let rough = oren_nayar(n, dir, dir, albedo, 0.6);
        assert!(rough.x > smooth.x);
        assert!(rough.y > smooth.y);
        assert!(rough.z > smooth.z);
    }

    #[test]
    fn oren_nayar_is_non_negative() {
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::splat(1.0);
        let dirs = [
            unit(0.0, 0.0, 1.0),
            unit(0.7, 0.0, 0.7),
            unit(-0.6, 0.5, 0.6),
            unit(0.9, 0.1, 0.3),
        ];
        for &l in &dirs {
            for &v in &dirs {
                for &sigma in &[0.0_f32, 0.25, 0.75, 1.5] {
                    let r = oren_nayar(n, l, v, albedo, sigma);
                    assert!(r.x >= 0.0 && r.y >= 0.0 && r.z >= 0.0);
                }
            }
        }
    }

    #[test]
    fn oren_nayar_back_facing_is_zero() {
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::splat(1.0);
        // Light below the horizon.
        let below = oren_nayar(n, unit(0.0, 0.0, -1.0), unit(0.0, 0.0, 1.0), albedo, 0.5);
        assert!(vec_approx_eq(below, Vec3::ZERO));
        // View below the horizon.
        let behind = oren_nayar(n, unit(0.0, 0.0, 1.0), unit(0.0, 0.0, -1.0), albedo, 0.5);
        assert!(vec_approx_eq(behind, Vec3::ZERO));
    }

    #[test]
    fn oren_nayar_is_energy_bounded() {
        // The lobe (A + B * s / t) is bounded, so the reflected value never
        // exceeds albedo / pi * NoL by more than the bounded B contribution.
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::splat(1.0);
        let dir = unit(0.95, 0.0, 0.31);
        let r = oren_nayar(n, dir, dir, albedo, 2.0);
        // A + B is below ~1.3 for any sigma here; the result stays well under 1.
        assert!(r.x < 1.0 && r.y < 1.0 && r.z < 1.0);
        assert!(r.x.is_finite() && r.y.is_finite() && r.z.is_finite());
    }

    #[test]
    fn burley_roughness_zero_is_lambert_head_on() {
        // Head-on: NoL = NoV = 1 so both grazing lobes are 1, giving albedo / pi
        // regardless of FD90.
        let n = unit(0.0, 0.0, 1.0);
        let l = unit(0.0, 0.0, 1.0);
        let v = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::new(0.6, 0.7, 0.8);
        let got = burley_diffuse(n, l, v, albedo, 0.0);
        assert!(vec_approx_eq(got, albedo.scale(INV_PI)));
    }

    #[test]
    fn burley_roughness_zero_is_near_lambert_off_axis() {
        // At roughness 0, FD90 = 0.5, so the lobes darken grazing angles below
        // Lambert but stay close for moderate angles.
        let n = unit(0.0, 0.0, 1.0);
        let l = unit(0.3, 0.0, 0.95);
        let v = unit(0.0, 0.2, 0.98);
        let albedo = Vec3::splat(1.0);
        let got = burley_diffuse(n, l, v, albedo, 0.0);
        let lambert = albedo.scale(INV_PI);
        // Within ~15% of Lambert for these near-normal directions.
        assert!((got.x - lambert.x).abs() < 0.15 * lambert.x);
    }

    #[test]
    fn burley_retro_reflection_brightens_at_grazing() {
        // Grazing retro (L = V near the silhouette, LoH = 1): roughness lifts
        // FD90 above 1, so a rough surface is brighter than a smooth one.
        let n = unit(0.0, 0.0, 1.0);
        let dir = unit(0.97, 0.0, 0.24);
        let albedo = Vec3::splat(0.9);
        let smooth = burley_diffuse(n, dir, dir, albedo, 0.0);
        let rough = burley_diffuse(n, dir, dir, albedo, 1.0);
        assert!(rough.x > smooth.x);
        assert!(rough.y > smooth.y);
        assert!(rough.z > smooth.z);
    }

    #[test]
    fn burley_is_non_negative_and_back_facing_is_zero() {
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::splat(1.0);
        let dirs = [
            unit(0.0, 0.0, 1.0),
            unit(0.8, 0.0, 0.6),
            unit(-0.5, 0.4, 0.76),
        ];
        for &l in &dirs {
            for &v in &dirs {
                for &rough in &[0.0_f32, 0.5, 1.0] {
                    let r = burley_diffuse(n, l, v, albedo, rough);
                    assert!(r.x >= 0.0 && r.y >= 0.0 && r.z >= 0.0);
                }
            }
        }
        let below = burley_diffuse(n, unit(0.0, 0.0, -1.0), unit(0.0, 0.0, 1.0), albedo, 0.5);
        assert!(vec_approx_eq(below, Vec3::ZERO));
    }

    #[test]
    fn burley_is_energy_bounded() {
        // FD90 is bounded for roughness in 0..=1 and LoH in 0..=1, so the
        // product of the two lobes stays finite and modest.
        let n = unit(0.0, 0.0, 1.0);
        let albedo = Vec3::splat(1.0);
        let dir = unit(0.99, 0.0, 0.14);
        let r = burley_diffuse(n, dir, dir, albedo, 1.0);
        assert!(r.x.is_finite());
        // Both grazing lobes peak at FD90 = 2.5 as NoL, NoV -> 0, so the lobe
        // product is bounded by 2.5 * 2.5 = 6.25 and the result by 6.25 / pi.
        assert!(r.x <= 6.25 * INV_PI + CMP_EPS);
    }

    #[test]
    fn both_closures_are_deterministic() {
        let n = unit(0.1, 0.2, 0.97);
        let l = unit(0.5, 0.1, 0.86);
        let v = unit(-0.2, 0.3, 0.93);
        let albedo = Vec3::new(0.3, 0.6, 0.9);
        assert_eq!(
            oren_nayar(n, l, v, albedo, 0.4),
            oren_nayar(n, l, v, albedo, 0.4)
        );
        assert_eq!(
            burley_diffuse(n, l, v, albedo, 0.4),
            burley_diffuse(n, l, v, albedo, 0.4)
        );
    }
}
