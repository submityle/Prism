//! Parallax-corrected iris sampling — CPU golden reference.
//!
//! Behind the cornea sits the aqueous-filled *anterior chamber*; at its back
//! wall lies the iris, the pigmented diaphragm whose central aperture is the
//! pupil.  Because the iris is set back from the refracting surface, the point
//! on the iris that a viewing ray reaches depends on the ray's angle — the
//! classic eye *parallax* that makes the iris appear to shift and the eye read
//! as a transparent dome rather than a painted disc.  This module turns a
//! refracted ray (see [`super::cornea`]) into an iris texture coordinate and
//! layers two further effects:
//!
//! * **Pupil dilation.** The iris musculature opens and closes the pupil.
//!   Rather than re-authoring the iris texture per aperture, the sample radius
//!   is radially re-mapped so the texture's rest-pose pupil edge is dragged to
//!   the live pupil radius while the limbal (outer) edge stays pinned — the
//!   iris tissue stretches or compresses between the two.
//! * **Limbal darkening.** The outer rim of the iris (the limbal ring) is a
//!   darker band; a smooth radial falloff reproduces it.
//!
//! # Conventions
//! * Work happens in the eye's *local frame* with the optical axis `+Z`
//!   pointing out toward the camera, so a refracted ray travelling into the eye
//!   has `z < 0`.  The iris lies in a plane parallel to `XY` at depth
//!   `anterior_chamber_depth` behind the entry point.
//! * Texture coordinates are the usual `[0, 1]^2`; the iris centre is `0.5` and
//!   the iris outer edge sits at radius `iris_radius` in UV units (commonly
//!   `0.5`).  Lengths handed to [`iris_plane_offset`] and the matching
//!   `iris_radius` share whatever world unit the caller chooses (millimetres in
//!   the physical defaults).
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals
//!   or `unsafe`).  Grazing rays (`|z| → 0`), zero radii and collapsed
//!   remap intervals are all floored, and every output is finite.
//! * `f32` arithmetic mirrors the WESL/GPU twin; `sqrt` is inherent.
//!
//! # References
//! * Jimenez et al., *Next Generation Character Rendering* (GDC 2013) — the
//!   refraction-offset iris parallax and pupil-scaling scheme adopted here.

use bevy_math::{Vec2, Vec3};

/// Smallest magnitude allowed for a cosine/denominator before flooring.
const MIN_DENOM: f32 = 1.0e-5;

/// Physical defaults for a human eye, in millimetres.
pub const ANTERIOR_CHAMBER_DEPTH_MM: f32 = 3.0;
/// Typical iris radius in millimetres (≈6 mm diameter).
pub const IRIS_RADIUS_MM: f32 = 6.0;

/// Planar displacement, on the iris plane, from the point where the refracted
/// ray enters to where it strikes the iris.
///
/// `refracted_dir` is the (local) propagation direction into the eye (`z < 0`);
/// `depth` is the perpendicular distance from the entry plane to the iris
/// plane.  The ray is traced `t = depth / (-z)` along its direction and the
/// resulting `xy` travel is returned, in the same length unit as `depth`.
///
/// Rays with a vanishing `z` component (grazing the surface) would travel an
/// unbounded distance; the inward cosine is floored at [`MIN_DENOM`] so the
/// offset stays finite.
#[inline]
pub fn iris_plane_offset(refracted_dir: Vec3, depth: f32) -> Vec2 {
    let depth = depth.max(0.0);
    // Inward component: ray goes into the eye, so z is negative.
    let inward = (-refracted_dir.z).max(MIN_DENOM);
    let t = depth / inward;
    Vec2::new(refracted_dir.x, refracted_dir.y) * t
}

/// Parallax-corrected iris UV for a refracted ray.
///
/// Starts from `center_uv` (the un-parallaxed sample, usually the iris centre
/// `0.5`), traces the ray to the iris plane via [`iris_plane_offset`] and maps
/// the world-space offset into UV units using `iris_radius` (world units that
/// correspond to a UV radius of `0.5`).
#[inline]
pub fn iris_uv_parallax(
    center_uv: Vec2,
    refracted_dir: Vec3,
    depth: f32,
    iris_radius: f32,
) -> Vec2 {
    let offset = iris_plane_offset(refracted_dir, depth);
    // world-radius `iris_radius` <=> UV radius 0.5.
    let uv_per_world = 0.5 / iris_radius.max(MIN_DENOM);
    center_uv + offset * uv_per_world
}

/// Radially re-maps a UV sample to account for pupil dilation.
///
/// The iris texture is authored with the pupil edge at `pupil_rest` (a radius
/// in UV units) and the iris outer edge at `iris_edge`.  For a live pupil
/// radius `pupil_now`, samples inside the live pupil are scaled from the rest
/// pupil, and samples in the surrounding annulus are linearly re-mapped so the
/// iris edge stays pinned:
///
/// ```text
/// r' = r * pupil_rest / pupil_now                              (r <= pupil_now)
/// r' = pupil_rest + (r - pupil_now) * (iris_edge - pupil_rest)
///                                   / (iris_edge - pupil_now)   (otherwise)
/// ```
///
/// The angle around `center` is preserved; the sample at the exact centre is
/// returned unchanged.  All radii are clamped to a sane order
/// (`0 <= pupil < iris_edge`) and the output radius is clamped to the iris
/// disc so the lookup never leaves the authored region.
#[inline]
pub fn pupil_radial_remap(
    uv: Vec2,
    center: Vec2,
    pupil_rest: f32,
    pupil_now: f32,
    iris_edge: f32,
) -> Vec2 {
    let iris_edge = iris_edge.max(MIN_DENOM);
    // Keep radii ordered and strictly inside the iris disc.
    let pupil_rest = pupil_rest.clamp(0.0, iris_edge - MIN_DENOM);
    let pupil_now = pupil_now.clamp(MIN_DENOM, iris_edge - MIN_DENOM);

    let delta = uv - center;
    let r = delta.length();
    if r < MIN_DENOM {
        return uv; // Centre: direction undefined, nothing to re-map.
    }
    let dir = delta / r;

    let r_mapped = if r <= pupil_now {
        r * (pupil_rest / pupil_now)
    } else {
        let outer = (iris_edge - pupil_rest) / (iris_edge - pupil_now).max(MIN_DENOM);
        pupil_rest + (r - pupil_now) * outer
    };
    let r_mapped = r_mapped.clamp(0.0, iris_edge);
    center + dir * r_mapped
}

/// Smooth Hermite interpolation `smoothstep(edge0, edge1, x)` in `[0, 1]`.
///
/// Degenerate intervals (`edge1 <= edge0`) collapse to a hard step so the
/// result is always finite.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 - edge0 <= MIN_DENOM {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Limbal-ring darkening multiplier for a normalised radius.
///
/// `r_norm` is the distance from the iris centre in `[0, 1]` (`1` = iris edge).
/// Darkening ramps in from `ring_start` toward the edge via [`smoothstep`],
/// reaching a minimum brightness of `1 - strength` at the rim.  Returns a
/// multiplier in `[0, 1]`.
#[inline]
pub fn limbal_darkening(r_norm: f32, ring_start: f32, strength: f32) -> f32 {
    let strength = strength.clamp(0.0, 1.0);
    let ring_start = ring_start.clamp(0.0, 1.0);
    let ramp = smoothstep(ring_start, 1.0, r_norm.max(0.0));
    (1.0 - strength * ramp).clamp(0.0, 1.0)
}

/// Geometry of the anterior chamber and iris disc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrisGeometry {
    /// Depth from the refracting surface to the iris plane (world units).
    pub anterior_chamber_depth: f32,
    /// Iris outer radius in the same world units; maps to a UV radius of `0.5`.
    pub iris_radius: f32,
}

impl IrisGeometry {
    /// Builds geometry, flooring both lengths at zero / [`MIN_DENOM`].
    #[inline]
    pub fn new(anterior_chamber_depth: f32, iris_radius: f32) -> Self {
        Self {
            anterior_chamber_depth: anterior_chamber_depth.max(0.0),
            iris_radius: iris_radius.max(MIN_DENOM),
        }
    }
}

impl Default for IrisGeometry {
    #[inline]
    fn default() -> Self {
        Self::new(ANTERIOR_CHAMBER_DEPTH_MM, IRIS_RADIUS_MM)
    }
}

/// Pupil and limbal styling in UV units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrisStyle {
    /// Rest-pose pupil radius the texture was authored with (UV units).
    pub pupil_rest: f32,
    /// Iris outer edge radius in UV units (commonly `0.5`).
    pub iris_edge: f32,
    /// Normalised radius (`0..1`) at which limbal darkening starts.
    pub limbal_start: f32,
    /// Peak limbal darkening strength in `[0, 1]`.
    pub limbal_strength: f32,
}

impl Default for IrisStyle {
    #[inline]
    fn default() -> Self {
        Self {
            pupil_rest: 0.14,
            iris_edge: 0.5,
            limbal_start: 0.82,
            limbal_strength: 0.65,
        }
    }
}

/// Result of sampling the iris for one refracted ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrisSample {
    /// Final iris texture coordinate (parallax- and pupil-corrected).
    pub uv: Vec2,
    /// Limbal darkening multiplier in `[0, 1]`.
    pub limbal: f32,
    /// Normalised radius from the iris centre in `[0, 1]`.
    pub radius_norm: f32,
}

/// Full iris sample: parallax, pupil re-map and limbal darkening in one call.
///
/// `refracted_dir` is the local into-eye direction from [`super::cornea`],
/// `pupil_now` the live pupil radius in UV units, `center_uv` the iris centre
/// (usually `Vec2::splat(0.5)`).  The returned [`IrisSample`] carries the UV to
/// fetch the iris texture with and the darkening multiplier to apply.
#[inline]
pub fn sample_iris(
    center_uv: Vec2,
    refracted_dir: Vec3,
    pupil_now: f32,
    geometry: IrisGeometry,
    style: IrisStyle,
) -> IrisSample {
    let parallax_uv = iris_uv_parallax(
        center_uv,
        refracted_dir,
        geometry.anterior_chamber_depth,
        geometry.iris_radius,
    );
    let uv = pupil_radial_remap(
        parallax_uv,
        center_uv,
        style.pupil_rest,
        pupil_now,
        style.iris_edge,
    );
    let radius_norm = ((uv - center_uv).length() / style.iris_edge.max(MIN_DENOM)).clamp(0.0, 1.0);
    let limbal = limbal_darkening(radius_norm, style.limbal_start, style.limbal_strength);
    IrisSample {
        uv,
        limbal,
        radius_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_ray_hits_the_center() {
        // A ray straight down the axis reaches the iris with zero offset.
        let offset = iris_plane_offset(Vec3::NEG_Z, 3.0);
        assert!(offset.length() < 1e-6, "offset={offset:?}");
    }

    #[test]
    fn offset_grows_with_depth_and_angle() {
        let dir = Vec3::new(0.3, 0.0, -0.9).normalize();
        let shallow = iris_plane_offset(dir, 1.0).length();
        let deep = iris_plane_offset(dir, 3.0).length();
        assert!(deep > shallow, "deep={deep} shallow={shallow}");
        // Analytic: |xy|/|z| * depth.
        let expected = (dir.x / -dir.z) * 3.0;
        assert!((iris_plane_offset(dir, 3.0).x - expected).abs() < 1e-5);
    }

    #[test]
    fn grazing_ray_offset_stays_finite() {
        let dir = Vec3::new(1.0, 0.0, 0.0); // z == 0, grazing
        let offset = iris_plane_offset(dir, 3.0);
        assert!(offset.is_finite(), "offset={offset:?}");
    }

    #[test]
    fn parallax_uv_shifts_from_center() {
        let center = Vec2::splat(0.5);
        let dir = Vec3::new(0.2, 0.0, -0.98).normalize();
        let uv = iris_uv_parallax(center, dir, 3.0, 6.0);
        assert!(uv.x > 0.5, "expected +u shift, uv={uv:?}");
        assert!((uv.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pupil_remap_identity_when_rest_equals_now() {
        let center = Vec2::splat(0.5);
        let uv = Vec2::new(0.7, 0.55);
        let out = pupil_radial_remap(uv, center, 0.2, 0.2, 0.5);
        assert!((out - uv).length() < 1e-5, "out={out:?}");
    }

    #[test]
    fn pupil_dilation_pushes_samples_outward_inside_pupil() {
        // Dilating the pupil (now > rest) shrinks the sampled texture radius,
        // dragging interior texels toward the center.
        let center = Vec2::splat(0.5);
        let uv = Vec2::new(0.5 + 0.1, 0.5); // r = 0.1
        let out = pupil_radial_remap(uv, center, 0.1, 0.2, 0.5);
        let r_out = (out - center).length();
        assert!(r_out < 0.1, "r_out={r_out}");
        // Expected: r * rest/now = 0.1 * 0.1/0.2 = 0.05.
        assert!((r_out - 0.05).abs() < 1e-5, "r_out={r_out}");
    }

    #[test]
    fn pupil_remap_pins_the_iris_edge() {
        let center = Vec2::splat(0.5);
        let uv = Vec2::new(0.5 + 0.5, 0.5); // r = iris edge
        let out = pupil_radial_remap(uv, center, 0.1, 0.25, 0.5);
        let r_out = (out - center).length();
        assert!((r_out - 0.5).abs() < 1e-5, "edge not pinned: {r_out}");
    }

    #[test]
    fn pupil_remap_center_is_stable() {
        let center = Vec2::splat(0.5);
        let out = pupil_radial_remap(center, center, 0.1, 0.3, 0.5);
        assert!((out - center).length() < 1e-6);
    }

    #[test]
    fn pupil_remap_stays_within_iris_disc() {
        let center = Vec2::splat(0.5);
        for i in 0..=10 {
            let r = i as f32 / 10.0 * 0.5;
            let uv = center + Vec2::new(r, 0.0);
            let out = pupil_radial_remap(uv, center, 0.15, 0.4, 0.5);
            let r_out = (out - center).length();
            assert!(r_out <= 0.5 + 1e-5, "escaped disc: {r_out}");
        }
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_degenerate_interval_is_a_step() {
        assert_eq!(smoothstep(0.5, 0.5, 0.4), 0.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.6), 1.0);
    }

    #[test]
    fn limbal_center_is_bright_edge_is_dark() {
        let center = limbal_darkening(0.0, 0.8, 0.6);
        let edge = limbal_darkening(1.0, 0.8, 0.6);
        assert!((center - 1.0).abs() < 1e-6, "center={center}");
        assert!((edge - 0.4).abs() < 1e-6, "edge={edge}");
        assert!(center > edge);
    }

    #[test]
    fn limbal_is_bounded_everywhere() {
        for i in 0..=20 {
            let r = i as f32 / 20.0;
            let d = limbal_darkening(r, 0.75, 0.9);
            assert!((0.0..=1.0).contains(&d), "d={d}");
        }
    }

    #[test]
    fn sample_iris_produces_finite_bounded_outputs() {
        let geo = IrisGeometry::default();
        let style = IrisStyle::default();
        let dir = Vec3::new(0.15, -0.1, -0.98).normalize();
        let s = sample_iris(Vec2::splat(0.5), dir, 0.2, geo, style);
        assert!(s.uv.is_finite());
        assert!((0.0..=1.0).contains(&s.limbal));
        assert!((0.0..=1.0).contains(&s.radius_norm));
    }

    #[test]
    fn sample_iris_axial_lands_at_center() {
        let s = sample_iris(
            Vec2::splat(0.5),
            Vec3::NEG_Z,
            0.2,
            IrisGeometry::default(),
            IrisStyle::default(),
        );
        assert!((s.uv - Vec2::splat(0.5)).length() < 1e-5, "uv={:?}", s.uv);
        assert!((s.limbal - 1.0).abs() < 1e-6);
    }
}
