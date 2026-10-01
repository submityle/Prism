//! Analytic light-source sampling for next-event estimation — CPU golden.
//!
//! Next-event estimation (NEE) connects a shading point directly to the lights
//! each bounce, which only converges quickly when the lights are *importance
//! sampled* in the solid-angle measure the rendering equation integrates over.
//! This module is the backend-neutral numerical reference for that sampling:
//! given a shading-point origin and a light description plus a canonical
//! `(u, v) ∈ [0, 1)^2` draw, it returns the sampled incident direction, the
//! distance to the sampled light point and the probability density **with
//! respect to solid angle** at the origin.
//!
//! Supported emitters:
//! * [`sample_sphere`] — a spherical light sampled by Shirley's uniform-cone
//!   construction: the sphere subtends a cone of half-angle `θ_max` with
//!   `sin²θ_max = r² / d²`, directions are drawn uniformly inside that cone and
//!   the solid-angle pdf is the constant `1 / (2π (1 − cos θ_max))`.
//! * [`sample_rectangle`] — a planar rectangular area light sampled uniformly
//!   over its surface, with the area pdf converted to solid angle through the
//!   geometric Jacobian `dist² / (cos θ_l · A)`.
//! * [`sample_disk`] — a planar disk light sampled with the Shirley–Chiu
//!   concentric map, using the same area-to-solid-angle conversion.
//! * [`sample_spot`] — a spotlight: a positional *delta* emitter whose single
//!   connection direction has a discrete (delta) pdf, paired with
//!   [`spot_attenuation`] for the smooth cone falloff.
//! * [`sample_directional`] — a directional (sun) light: a *delta* emitter at
//!   infinity with a single incident direction.
//!
//! # Conventions
//! * `wi` is the **unit** direction from the shading point *towards* the light;
//!   `distance` is the length of that connection segment (`f32::INFINITY` for a
//!   directional light).
//! * `pdf` is a **solid-angle** density at the origin.  For the two *delta*
//!   emitters (spot, directional) there is no density: [`LightSample::pdf`] is
//!   set to `1.0` and [`LightSample::is_delta`] is `true`, the convention the
//!   MIS layer uses to force a unit weight.
//! * All math is `f32` to mirror the GPU twin; transcendental calls go through
//!   [`bevy_math::ops`] and `sqrt` through the inherent method.
//! * Every routine is defensive: non-finite inputs, zero radius / area, a
//!   coincident origin, an origin inside a sphere, and back-facing area samples
//!   are all handled by a deterministic, non-negative fallback.  No routine
//!   ever returns a `NaN` direction, distance or pdf.
//! * All functions are deterministic pure functions: no RNG, no I/O, no GPU,
//!   no global state and no `unsafe`.

use bevy_math::{ops, Vec3};
use core::f32::consts::{PI, TAU};

use crate::gi::sample::mapping::{concentric_disk, orthonormal_basis};

/// The result of sampling a light source from a shading point.
///
/// All geometry is expressed at the shading-point origin: [`wi`](Self::wi) is a
/// unit direction towards the light and [`pdf`](Self::pdf) is a solid-angle
/// density there.  A degenerate sample carries [`LightSample::NONE`]
/// (zero pdf), which upstream NEE must treat as "no contribution".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSample {
    /// Unit direction from the shading point towards the sampled light point.
    pub wi: Vec3,
    /// Distance from the shading point to the sampled light point
    /// (`f32::INFINITY` for a directional light).
    pub distance: f32,
    /// Probability density of this sample with respect to solid angle at the
    /// shading point.  `1.0` for a delta emitter (see [`is_delta`](Self::is_delta)).
    pub pdf: f32,
    /// Whether the emitter is a Dirac-delta light (directional / spot / point)
    /// for which the solid-angle density is not a finite number.
    pub is_delta: bool,
}

impl Default for LightSample {
    #[inline]
    fn default() -> Self {
        Self::NONE
    }
}

impl LightSample {
    /// A degenerate "no light" sample: zero direction, zero distance, zero pdf.
    ///
    /// Its zero pdf marks it as carrying no energy; NEE must skip it.
    pub const NONE: Self = Self {
        wi: Vec3::ZERO,
        distance: 0.0,
        pdf: 0.0,
        is_delta: false,
    };

    /// Whether this sample is valid, i.e. carries a usable direction and either
    /// a strictly-positive solid-angle pdf or a delta flag.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.wi != Vec3::ZERO
            && self.wi.is_finite()
            && self.distance >= 0.0
            && (self.is_delta || (self.pdf.is_finite() && self.pdf > 0.0))
    }
}

/// A tiny positive epsilon used to reject coincident / degenerate geometry.
const EPS: f32 = 1.0e-12;

/// Solid-angle pdf of Shirley's uniform-cone sampling for a cone of half-angle
/// `θ_max` given `cos θ_max`.
///
/// A cone subtends the solid angle `2π (1 − cos θ_max)`, so a uniform draw has
/// the constant density `1 / (2π (1 − cos θ_max))`.  Returns `0` for a
/// degenerate cone (`cos θ_max ≥ 1`) or a non-finite input.
#[inline]
pub fn uniform_cone_pdf(cos_theta_max: f32) -> f32 {
    if !cos_theta_max.is_finite() {
        return 0.0;
    }
    let one_minus = (1.0 - cos_theta_max).max(0.0);
    if one_minus <= EPS {
        return 0.0;
    }
    let pdf = 1.0 / (TAU * one_minus);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Builds a unit direction at polar angle `θ` (given by `cos_theta`) and
/// azimuth `phi` around the unit `axis`.
///
/// Falls back to `axis` when the constructed vector is degenerate so the result
/// is always a finite unit vector.
#[inline]
fn direction_in_cone(axis: Vec3, cos_theta: f32, phi: f32) -> Vec3 {
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let (t, b) = orthonormal_basis(axis.to_array());
    let tangent = Vec3::from_array(t);
    let bitangent = Vec3::from_array(b);
    let dir = tangent * (sin_theta * ops::cos(phi))
        + bitangent * (sin_theta * ops::sin(phi))
        + axis * cos_theta;
    let normalized = dir.normalize_or_zero();
    if normalized == Vec3::ZERO {
        axis
    } else {
        normalized
    }
}

/// Samples a spherical light of `radius` centred at `center` from `origin`.
///
/// Uses Shirley's uniform-cone construction: the sphere subtends a cone of
/// half-angle `θ_max` with `sin²θ_max = r² / d²` (`d` the centre distance), a
/// direction is drawn uniformly inside that cone, and the exact intersection
/// distance along `wi` is returned.  The solid-angle pdf is the constant
/// `1 / (2π (1 − cos θ_max))`.
///
/// When `origin` lies inside the sphere (`d ≤ r`) the cone degenerates to the
/// full set of directions, so the routine falls back to a uniform sphere of
/// directions with pdf `1 / 4π`.  Non-finite inputs or a non-positive radius
/// yield [`LightSample::NONE`].
#[inline]
pub fn sample_sphere(origin: Vec3, center: Vec3, radius: f32, u: f32, v: f32) -> LightSample {
    if !origin.is_finite() || !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return LightSample::NONE;
    }
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let to_center = center - origin;
    let dist_sq = to_center.length_squared();
    if !dist_sq.is_finite() {
        return LightSample::NONE;
    }

    // Origin inside the sphere: every direction can hit the light, so draw a
    // uniform direction over the full sphere (pdf = 1 / 4π).
    if dist_sq <= radius * radius {
        let cos_theta = 1.0 - 2.0 * u;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = TAU * v;
        let wi = Vec3::new(sin_theta * ops::cos(phi), sin_theta * ops::sin(phi), cos_theta)
            .normalize_or_zero();
        if wi == Vec3::ZERO {
            return LightSample::NONE;
        }
        let pdf = 1.0 / (4.0 * PI);
        return LightSample {
            wi,
            distance: radius,
            pdf,
            is_delta: false,
        };
    }

    let dist = dist_sq.sqrt();
    if dist <= EPS {
        return LightSample::NONE;
    }
    let axis = to_center / dist;
    // sin²θ_max = r² / d²  ⇒  cos θ_max = sqrt(1 − r²/d²).
    let sin2_max = (radius * radius / dist_sq).clamp(0.0, 1.0);
    let cos_theta_max = (1.0 - sin2_max).max(0.0).sqrt();

    // Uniform cone draw: cos θ linear in u between 1 and cos θ_max.
    let cos_theta = 1.0 - u * (1.0 - cos_theta_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * v;
    let wi = direction_in_cone(axis, cos_theta, phi);

    // Exact distance to the near sphere intersection along the sampled ray:
    //   t = d cos θ − sqrt(r² − d² sin²θ).
    let under = radius * radius - dist_sq * sin_theta * sin_theta;
    let distance = (dist * cos_theta - under.max(0.0).sqrt()).max(0.0);

    let pdf = uniform_cone_pdf(cos_theta_max);
    if pdf <= 0.0 {
        return LightSample::NONE;
    }
    LightSample {
        wi,
        distance,
        pdf,
        is_delta: false,
    }
}

/// Converts an **area** pdf (density per unit light area) to a **solid-angle**
/// pdf at the shading point.
///
/// The change of variables from surface area to projected solid angle carries
/// the Jacobian `dist² / cos θ_l`, where `θ_l` is the angle between the light
/// normal and the connection direction, so `p_ω = p_A · dist² / cos θ_l`.
/// Returns `0` for a degenerate (zero cosine / zero distance) configuration.
#[inline]
pub fn area_to_solid_angle_pdf(area_pdf: f32, distance: f32, cos_light: f32) -> f32 {
    let cos_light = cos_light.max(0.0);
    if area_pdf <= 0.0
        || !area_pdf.is_finite()
        || cos_light <= EPS
        || distance <= EPS
        || !distance.is_finite()
    {
        return 0.0;
    }
    let pdf = area_pdf * distance * distance / cos_light;
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Converts a **solid-angle** pdf at the shading point back to an **area** pdf
/// on the light surface: the inverse of [`area_to_solid_angle_pdf`].
///
/// `p_A = p_ω · cos θ_l / dist²`.  Returns `0` for a degenerate configuration.
#[inline]
pub fn solid_angle_to_area_pdf(solid_angle_pdf: f32, distance: f32, cos_light: f32) -> f32 {
    let cos_light = cos_light.max(0.0);
    if solid_angle_pdf <= 0.0
        || !solid_angle_pdf.is_finite()
        || distance <= EPS
        || !distance.is_finite()
    {
        return 0.0;
    }
    let pdf = solid_angle_pdf * cos_light / (distance * distance);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Samples a planar rectangular area light defined by a `corner` and two edge
/// vectors `edge_u`, `edge_v`, from `origin`.
///
/// The point `corner + u·edge_u + v·edge_v` is drawn uniformly over the
/// rectangle (area pdf `1 / A`, `A = |edge_u × edge_v|`), then converted to a
/// solid-angle pdf with the `dist² / (cos θ_l · A)` Jacobian.  The emitter is
/// treated as two-sided, so the light-normal cosine uses its absolute value.
///
/// A zero-area rectangle, a coincident sample point, or an edge-on connection
/// (`cos θ_l = 0`) yields [`LightSample::NONE`].
#[inline]
pub fn sample_rectangle(
    origin: Vec3,
    corner: Vec3,
    edge_u: Vec3,
    edge_v: Vec3,
    u: f32,
    v: f32,
) -> LightSample {
    if !origin.is_finite() || !corner.is_finite() || !edge_u.is_finite() || !edge_v.is_finite() {
        return LightSample::NONE;
    }
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let point = corner + edge_u * u + edge_v * v;
    let normal_vec = edge_u.cross(edge_v);
    let area = normal_vec.length();
    if area <= EPS || !area.is_finite() {
        return LightSample::NONE;
    }
    let light_normal = normal_vec / area;

    let to_light = point - origin;
    let dist_sq = to_light.length_squared();
    if dist_sq <= EPS || !dist_sq.is_finite() {
        return LightSample::NONE;
    }
    let distance = dist_sq.sqrt();
    let wi = to_light / distance;

    // Two-sided emitter: cosine of the light normal against the back-direction.
    let cos_light = light_normal.dot(-wi).abs();
    if cos_light <= EPS {
        return LightSample::NONE;
    }
    let area_pdf = 1.0 / area;
    let pdf = area_to_solid_angle_pdf(area_pdf, distance, cos_light);
    if pdf <= 0.0 {
        return LightSample::NONE;
    }
    LightSample {
        wi,
        distance,
        pdf,
        is_delta: false,
    }
}

/// Samples a planar disk light of `radius` centred at `center` with unit
/// `normal`, from `origin`.
///
/// The disk is sampled with the Shirley–Chiu concentric map (uniform in area,
/// `A = π r²`) and the area pdf converted to solid angle exactly as for
/// [`sample_rectangle`].  The emitter is two-sided.  A non-positive radius, a
/// degenerate normal, a coincident sample point or an edge-on connection yields
/// [`LightSample::NONE`].
#[inline]
pub fn sample_disk(
    origin: Vec3,
    center: Vec3,
    normal: Vec3,
    radius: f32,
    u: f32,
    v: f32,
) -> LightSample {
    if !origin.is_finite()
        || !center.is_finite()
        || !normal.is_finite()
        || !radius.is_finite()
        || radius <= 0.0
    {
        return LightSample::NONE;
    }
    let light_normal = normal.normalize_or_zero();
    if light_normal == Vec3::ZERO {
        return LightSample::NONE;
    }
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let (dx, dy) = concentric_disk(u, v);
    let (t, b) = orthonormal_basis(light_normal.to_array());
    let tangent = Vec3::from_array(t);
    let bitangent = Vec3::from_array(b);
    let point = center + tangent * (dx * radius) + bitangent * (dy * radius);

    let to_light = point - origin;
    let dist_sq = to_light.length_squared();
    if dist_sq <= EPS || !dist_sq.is_finite() {
        return LightSample::NONE;
    }
    let distance = dist_sq.sqrt();
    let wi = to_light / distance;

    let cos_light = light_normal.dot(-wi).abs();
    if cos_light <= EPS {
        return LightSample::NONE;
    }
    let area = PI * radius * radius;
    if area <= EPS {
        return LightSample::NONE;
    }
    let area_pdf = 1.0 / area;
    let pdf = area_to_solid_angle_pdf(area_pdf, distance, cos_light);
    if pdf <= 0.0 {
        return LightSample::NONE;
    }
    LightSample {
        wi,
        distance,
        pdf,
        is_delta: false,
    }
}

/// Samples a spotlight located at `light_pos` from `origin`.
///
/// A spotlight is a *positional delta* emitter: there is exactly one connection
/// direction (towards `light_pos`), so the returned sample is flagged as delta
/// with a sentinel `pdf = 1.0`.  The smooth angular cutoff is supplied
/// separately by [`spot_attenuation`].  A coincident origin yields
/// [`LightSample::NONE`].
#[inline]
pub fn sample_spot(origin: Vec3, light_pos: Vec3) -> LightSample {
    if !origin.is_finite() || !light_pos.is_finite() {
        return LightSample::NONE;
    }
    let to_light = light_pos - origin;
    let dist_sq = to_light.length_squared();
    if dist_sq <= EPS || !dist_sq.is_finite() {
        return LightSample::NONE;
    }
    let distance = dist_sq.sqrt();
    let wi = to_light / distance;
    LightSample {
        wi,
        distance,
        pdf: 1.0,
        is_delta: true,
    }
}

/// Smooth angular attenuation of a spotlight along the incident direction `wi`
/// (shading-point → light), for a cone with cosines `cos_inner ≥ cos_outer`.
///
/// `spot_axis` is the unit direction the spotlight *points* (light → scene); the
/// angle between the light-to-shading-point direction `−wi` and `spot_axis`
/// drives a smoothstep falloff: `1` inside the inner cone, `0` outside the outer
/// cone, smoothly interpolated between.  Returns a value in `[0, 1]`; a
/// degenerate cone (`cos_inner ≤ cos_outer`) collapses to a hard cutoff.
#[inline]
pub fn spot_attenuation(wi: Vec3, spot_axis: Vec3, cos_inner: f32, cos_outer: f32) -> f32 {
    let axis = spot_axis.normalize_or_zero();
    let dir = wi.normalize_or_zero();
    if axis == Vec3::ZERO || dir == Vec3::ZERO {
        return 0.0;
    }
    // Cosine between the light's pointing axis and the direction light→point.
    let cos_angle = axis.dot(-dir).clamp(-1.0, 1.0);
    if !cos_inner.is_finite() || !cos_outer.is_finite() {
        return 0.0;
    }
    let span = cos_inner - cos_outer;
    if span <= EPS {
        // Degenerate cone: hard cutoff at the inner cosine.
        return if cos_angle >= cos_inner { 1.0 } else { 0.0 };
    }
    let t = ((cos_angle - cos_outer) / span).clamp(0.0, 1.0);
    // Smoothstep for a visually soft edge.
    (t * t * (3.0 - 2.0 * t)).clamp(0.0, 1.0)
}

/// Samples a directional (sun) light whose emission travels along `light_dir`.
///
/// A directional light is a *delta* emitter at infinity: the single incident
/// direction is `−light_dir` (shading-point → light), the distance is
/// `f32::INFINITY`, and the sample is flagged delta with sentinel `pdf = 1.0`.
/// A degenerate (zero / non-finite) `light_dir` yields [`LightSample::NONE`].
#[inline]
pub fn sample_directional(light_dir: Vec3) -> LightSample {
    if !light_dir.is_finite() {
        return LightSample::NONE;
    }
    let wi = (-light_dir).normalize_or_zero();
    if wi == Vec3::ZERO {
        return LightSample::NONE;
    }
    LightSample {
        wi,
        distance: f32::INFINITY,
        pdf: 1.0,
        is_delta: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stratified midpoint sample for cell `(i, j)` of an `n × n`
    /// grid of the unit square — avoids any RNG while converging like QMC.
    fn strat(i: u32, j: u32, n: u32) -> (f32, f32) {
        (
            (i as f32 + 0.5) / n as f32,
            (j as f32 + 0.5) / n as f32,
        )
    }

    #[test]
    fn sphere_directions_lie_inside_the_cone() {
        let origin = Vec3::ZERO;
        let center = Vec3::new(0.0, 0.0, 5.0);
        let radius = 1.0;
        let d = (center - origin).length();
        let cos_theta_max = (1.0 - (radius * radius) / (d * d)).max(0.0).sqrt();
        let axis = (center - origin).normalize();
        for i in 0..8 {
            for j in 0..8 {
                let (u, v) = strat(i, j, 8);
                let s = sample_sphere(origin, center, radius, u, v);
                assert!(s.is_valid());
                // Direction must stay within the subtended cone.
                let c = axis.dot(s.wi);
                assert!(c >= cos_theta_max - 1.0e-4, "cos={c} max={cos_theta_max}");
                assert!(s.wi.is_finite());
                assert!(s.distance.is_finite() && s.distance > 0.0);
            }
        }
    }

    #[test]
    fn sphere_pdf_is_the_uniform_cone_density() {
        let origin = Vec3::ZERO;
        let center = Vec3::new(2.0, 0.0, 0.0);
        let radius = 0.5;
        let d = (center - origin).length();
        let cos_theta_max = (1.0 - (radius * radius) / (d * d)).max(0.0).sqrt();
        let expected = 1.0 / (TAU * (1.0 - cos_theta_max));
        let s = sample_sphere(origin, center, radius, 0.3, 0.7);
        assert!((s.pdf - expected).abs() / expected < 1.0e-4, "pdf={}", s.pdf);
    }

    #[test]
    fn sphere_pdf_integrates_to_one_over_the_cone() {
        // The uniform-cone pdf is constant and the subtended solid angle is
        // 2π(1−cosθmax); their product is exactly 1.
        let cos_theta_max = 0.8_f32;
        let pdf = uniform_cone_pdf(cos_theta_max);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        assert!((pdf * solid_angle - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn sphere_origin_inside_falls_back_to_uniform_sphere() {
        let s = sample_sphere(Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0), 1.0, 0.4, 0.6);
        assert!(s.is_valid());
        assert!((s.pdf - 1.0 / (4.0 * PI)).abs() < 1.0e-6);
        assert!((s.wi.length() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn sphere_degenerate_inputs_return_none() {
        assert_eq!(
            sample_sphere(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0), 0.0, 0.5, 0.5),
            LightSample::NONE
        );
        assert_eq!(
            sample_sphere(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0), -1.0, 0.5, 0.5),
            LightSample::NONE
        );
    }

    #[test]
    fn rectangle_sample_lies_on_the_light_and_is_valid() {
        let origin = Vec3::ZERO;
        let corner = Vec3::new(-1.0, -1.0, 4.0);
        let edge_u = Vec3::new(2.0, 0.0, 0.0);
        let edge_v = Vec3::new(0.0, 2.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let (u, v) = strat(i, j, 4);
                let s = sample_rectangle(origin, corner, edge_u, edge_v, u, v);
                assert!(s.is_valid());
                // The hit point reconstructed from wi*distance must have z=4.
                let hit = origin + s.wi * s.distance;
                assert!((hit.z - 4.0).abs() < 1.0e-4, "hit={hit:?}");
            }
        }
    }

    #[test]
    fn rectangle_solid_angle_matches_analytic_axis_formula() {
        // A square of half-width a centred on the +Z axis at distance d has the
        // exact solid angle Ω = 4·atan( a² / (d·sqrt(2a² + d²)) ).  Estimate it
        // by uniform-area sampling: Ω ≈ (1/N) Σ cosθ_l / dist² · A.
        let origin = Vec3::ZERO;
        let a = 1.0_f32;
        let d = 3.0_f32;
        let corner = Vec3::new(-a, -a, d);
        let edge_u = Vec3::new(2.0 * a, 0.0, 0.0);
        let edge_v = Vec3::new(0.0, 2.0 * a, 0.0);
        let area = (edge_u.cross(edge_v)).length();

        let n = 64u32;
        let mut acc = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let (u, v) = strat(i, j, n);
                let point = corner + edge_u * u + edge_v * v;
                let to_light = point - origin;
                let dist_sq = to_light.length_squared();
                let dist = dist_sq.sqrt();
                let wi = to_light / dist;
                let cos_light = Vec3::NEG_Z.dot(-wi).abs();
                acc += (cos_light / dist_sq) as f64;
            }
        }
        let mc = acc / (n as f64 * n as f64) * area as f64;
        let analytic =
            4.0 * ops::atan(a * a / (d * (2.0 * a * a + d * d).sqrt())) as f32;
        let rel = ((mc - analytic as f64) / analytic as f64).abs();
        assert!(rel < 0.03, "mc={mc} analytic={analytic} rel={rel}");
    }

    #[test]
    fn area_solid_angle_round_trip() {
        let area_pdf = 0.25_f32;
        let distance = 3.0_f32;
        let cos_light = 0.6_f32;
        let sa = area_to_solid_angle_pdf(area_pdf, distance, cos_light);
        let back = solid_angle_to_area_pdf(sa, distance, cos_light);
        assert!((back - area_pdf).abs() / area_pdf < 1.0e-5, "back={back}");
    }

    #[test]
    fn area_conversion_degenerate_is_zero() {
        assert_eq!(area_to_solid_angle_pdf(1.0, 1.0, 0.0), 0.0);
        assert_eq!(area_to_solid_angle_pdf(1.0, 0.0, 1.0), 0.0);
        assert_eq!(solid_angle_to_area_pdf(1.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn disk_sample_is_within_radius_and_valid() {
        let origin = Vec3::ZERO;
        let center = Vec3::new(0.0, 0.0, 5.0);
        let normal = Vec3::NEG_Z;
        let radius = 1.5_f32;
        for i in 0..6 {
            for j in 0..6 {
                let (u, v) = strat(i, j, 6);
                let s = sample_disk(origin, center, normal, radius, u, v);
                assert!(s.is_valid());
                let hit = origin + s.wi * s.distance;
                // Hit lies in the disk plane (z=5) within the radius.
                assert!((hit.z - 5.0).abs() < 1.0e-3, "hit={hit:?}");
                let planar = Vec3::new(hit.x, hit.y, 0.0).length();
                assert!(planar <= radius + 1.0e-3, "planar={planar}");
            }
        }
    }

    #[test]
    fn spot_is_delta_and_points_at_the_light() {
        let s = sample_spot(Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0));
        assert!(s.is_delta);
        assert_eq!(s.pdf, 1.0);
        assert!((s.distance - 4.0).abs() < 1.0e-5);
        assert!((s.wi - Vec3::Z).length() < 1.0e-5);
    }

    #[test]
    fn spot_attenuation_falls_off_across_the_cone() {
        let axis = Vec3::Z; // light points along +Z
        let cos_inner = ops::cos(0.2);
        let cos_outer = ops::cos(0.4);
        // On-axis towards the light: wi = +Z, −wi = −Z... build geometry so the
        // shading point is behind the light's beam. Light at origin pointing +Z,
        // shading point at +Z means wi (point→light) = −Z, −wi = +Z = axis.
        let on_axis = spot_attenuation(Vec3::NEG_Z, axis, cos_inner, cos_outer);
        assert!((on_axis - 1.0).abs() < 1.0e-5, "on_axis={on_axis}");
        // Far outside the cone → zero.
        let outside = spot_attenuation(
            Vec3::new(1.0, 0.0, 0.0),
            axis,
            cos_inner,
            cos_outer,
        );
        assert_eq!(outside, 0.0);
        // Everything stays in [0, 1].
        let mid = spot_attenuation(
            Vec3::new(0.0, -0.3, -1.0),
            axis,
            cos_inner,
            cos_outer,
        );
        assert!((0.0..=1.0).contains(&mid));
    }

    #[test]
    fn directional_is_delta_at_infinity() {
        let s = sample_directional(Vec3::new(0.0, -1.0, 0.0));
        assert!(s.is_delta);
        assert_eq!(s.pdf, 1.0);
        assert!(s.distance.is_infinite());
        assert!((s.wi - Vec3::Y).length() < 1.0e-5);
        assert_eq!(sample_directional(Vec3::ZERO), LightSample::NONE);
    }

    #[test]
    fn no_sampler_ever_emits_nan() {
        let samples = [
            sample_sphere(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0), 1.0, 0.0, 0.0),
            sample_sphere(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0), 1.0, 1.0, 1.0),
            sample_rectangle(
                Vec3::ZERO,
                Vec3::new(-1.0, -1.0, 4.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                1.0,
                1.0,
            ),
            sample_disk(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, 1.0, 0.0, 0.0),
        ];
        for s in samples {
            assert!(s.wi.is_finite());
            assert!(s.pdf.is_finite());
            assert!(s.distance.is_finite() || s.distance.is_infinite());
            assert!(!s.pdf.is_nan());
        }
    }
}
