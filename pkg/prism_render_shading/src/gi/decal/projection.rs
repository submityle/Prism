//! Box / clip-space deferred-decal projection (CPU golden reference).
//!
//! A deferred decal is a unit cube `[-0.5, 0.5]^3` placed in the world by an
//! affine transform.  During the deferred-decal pass the renderer walks the
//! G-buffer, reconstructs each pixel's world position and geometric normal,
//! and asks "does this decal cover this pixel, and how strongly?".  This module
//! is the backend-neutral reference for that query, mirroring Unreal Engine's
//! deferred-decal projection and the clustered-decal projection described by
//! Bartosz Wroński.
//!
//! The decal is defined by a `world -> decal` matrix [`DecalProjector::world_to_decal`].
//! A world position is mapped into decal space; if it lands inside the unit
//! cube the decal covers it.  The in-cube coordinate yields the decal atlas UV
//! (local `xy`, remapped to `[0, 1]`) and a projection depth (local `z`,
//! remapped to `[0, 1]`).  Coverage is attenuated by two fades:
//!
//! * **Angle fade.**  A decal projects along its local depth axis.  A surface
//!   whose normal faces back toward the projector receives the decal fully; a
//!   surface turned edge-on (or facing away) receives it weakly or not at all.
//!   The fade is a cosine `smoothstep` between [`DecalProjector::angle_fade_end`]
//!   and [`DecalProjector::angle_fade_start`].
//! * **Edge fade.**  Pixels near the cube's lateral walls fade out smoothly over
//!   [`DecalProjector::edge_fade`] (measured in local units) so decals do not
//!   terminate in a hard rectangle.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::exp()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * The projection axis is the decal's local `+Z` mapped to world space; depth
//!   increases along it (near wall at local `z = -0.5`, far wall at `+0.5`).
//! * A surface's *facing* term is `dot(normal, -projection_dir)`: `+1` when the
//!   surface stares straight back at the projector, `0` edge-on, negative when
//!   back-facing.  Only non-negative facing contributes.
//! * Defensive clamping everywhere: non-finite inputs, singular matrices, and
//!   degenerate fade ranges all fall back gracefully so no `NaN`/`inf` escapes.

use bevy_math::{Mat4, Vec2, Vec3};

/// Smallest absolute matrix determinant treated as invertible.
///
/// A `world -> decal` matrix flatter than this cannot be inverted reliably, so
/// the projection direction falls back to the identity local `+Z` axis.
const MIN_DETERMINANT: f32 = 1.0e-12;

/// Half-extent of the decal unit cube along every local axis.
const HALF_EXTENT: f32 = 0.5;

/// Result of a successful decal coverage query.
///
/// Returned by [`DecalProjector::project`] only when the queried world position
/// lands inside the decal volume.  All fields are finite and in range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalHit {
    /// Decal atlas coordinate in `[0, 1]^2` (local `xy` remapped from
    /// `[-0.5, 0.5]`).  `(0.5, 0.5)` is the cube centre.
    pub uv: Vec2,
    /// Projection depth in `[0, 1]` (local `z` remapped from `[-0.5, 0.5]`).
    /// `0` is the near wall, `1` the far wall.
    pub depth: f32,
    /// Combined coverage weight in `[0, 1]`: the product of the angle fade and
    /// the edge fade.  `0` means the decal is present but fully attenuated.
    pub fade: f32,
}

/// A box decal: a unit cube positioned by a `world -> decal` transform.
///
/// Construct with [`DecalProjector::new`], which precomputes the world-space
/// projection direction from the inverse transform so [`DecalProjector::project`]
/// never has to invert a matrix per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalProjector {
    /// Affine transform mapping world positions into the decal's local unit
    /// cube `[-0.5, 0.5]^3`.
    pub world_to_decal: Mat4,
    /// Unit world-space direction along which the decal projects (local `+Z`
    /// mapped to world).  Precomputed and renormalised on construction.
    pub projection_dir: Vec3,
    /// Cosine of the angle at which coverage first reaches full strength.
    ///
    /// Clamped to `[0, 1]`; coverage is `1` once `facing >= angle_fade_start`.
    pub angle_fade_start: f32,
    /// Cosine of the angle at which coverage reaches zero.
    ///
    /// Clamped to `[0, 1]`; coverage is `0` once `facing <= angle_fade_end`.
    /// Must be strictly below `angle_fade_start` for a soft ramp; otherwise the
    /// fade degenerates to a hard cut at `angle_fade_end`.
    pub angle_fade_end: f32,
    /// Lateral soft-edge width in local units, clamped to `[0, 0.5]`.
    ///
    /// Pixels within this distance of a cube wall fade toward zero.  `0`
    /// disables edge fade (hard rectangular cutoff).
    pub edge_fade: f32,
}

impl DecalProjector {
    /// Builds a projector from a `world -> decal` matrix and fade parameters.
    ///
    /// The world-space projection direction is derived from the inverse
    /// transform's local `+Z` column.  A singular or non-finite transform falls
    /// back to the world `+Z` axis so a mis-authored decal still behaves.
    #[inline]
    pub fn new(
        world_to_decal: Mat4,
        angle_fade_start: f32,
        angle_fade_end: f32,
        edge_fade: f32,
    ) -> Self {
        let projection_dir = Self::derive_projection_dir(world_to_decal);
        Self {
            world_to_decal,
            projection_dir,
            angle_fade_start: sanitize_cosine(angle_fade_start, 0.5),
            angle_fade_end: sanitize_cosine(angle_fade_end, 0.0),
            edge_fade: sanitize_edge(edge_fade),
        }
    }

    /// Returns the world-space local `+Z` axis of a `world -> decal` matrix.
    ///
    /// Falls back to [`Vec3::Z`] when the matrix is non-finite or too close to
    /// singular to invert.
    #[inline]
    fn derive_projection_dir(world_to_decal: Mat4) -> Vec3 {
        let det = world_to_decal.determinant();
        if !det.is_finite() || det.abs() < MIN_DETERMINANT {
            return Vec3::Z;
        }
        let decal_to_world = world_to_decal.inverse();
        if !matrix_is_finite(&decal_to_world) {
            return Vec3::Z;
        }
        let axis = decal_to_world.transform_vector3(Vec3::Z);
        let dir = axis.normalize_or_zero();
        if dir.length_squared() > 0.0 {
            dir
        } else {
            Vec3::Z
        }
    }

    /// Queries decal coverage at a world position with a world-space normal.
    ///
    /// Returns `Some(DecalHit)` when the position lies inside the decal cube,
    /// `None` otherwise.  The normal is renormalised defensively; a degenerate
    /// (zero-length) normal is treated as perfectly front-facing so geometry
    /// without a usable normal still receives the decal body (angle fade `1`).
    ///
    /// Non-finite inputs always miss.
    #[inline]
    pub fn project(&self, world_pos: Vec3, world_normal: Vec3) -> Option<DecalHit> {
        if !vec3_is_finite(world_pos) {
            return None;
        }

        let local = self.world_to_decal.transform_point3(world_pos);
        if !vec3_is_finite(local) {
            return None;
        }

        // Bounds test against the unit cube.
        if local.x.abs() > HALF_EXTENT
            || local.y.abs() > HALF_EXTENT
            || local.z.abs() > HALF_EXTENT
        {
            return None;
        }

        let uv = Vec2::new(local.x + HALF_EXTENT, local.y + HALF_EXTENT);
        let depth = (local.z + HALF_EXTENT).clamp(0.0, 1.0);

        let angle = self.angle_fade(world_normal);
        let edge = self.edge_fade(local);
        let fade = (angle * edge).clamp(0.0, 1.0);

        Some(DecalHit { uv, depth, fade })
    }

    /// Angle-based coverage in `[0, 1]` for a world-space surface normal.
    ///
    /// `facing = dot(normalize(n), -projection_dir)`; the result is
    /// `smoothstep(angle_fade_end, angle_fade_start, facing)`.  Back-facing
    /// surfaces (`facing <= 0`) always yield `0`.
    #[inline]
    pub fn angle_fade(&self, world_normal: Vec3) -> f32 {
        let n = world_normal.normalize_or_zero();
        // A missing normal cannot be back-facing; treat it as fully aligned so
        // the decal body is not silently erased.
        if n.length_squared() == 0.0 {
            return 1.0;
        }
        let facing = n.dot(-self.projection_dir);
        if facing <= 0.0 {
            return 0.0;
        }
        smoothstep(self.angle_fade_end, self.angle_fade_start, facing)
    }

    /// Lateral edge coverage in `[0, 1]` for an in-cube local position.
    ///
    /// Uses the distance from the position to the nearest `xy` wall; pixels
    /// within `edge_fade` of a wall ramp from `1` (interior) to `0` (wall).
    /// With `edge_fade == 0` the result is `1` everywhere inside the cube.
    #[inline]
    pub fn edge_fade(&self, local: Vec3) -> f32 {
        if self.edge_fade <= 0.0 {
            return 1.0;
        }
        // Distance to the nearest lateral wall, in `[0, 0.5]`.
        let dist = HALF_EXTENT - local.x.abs().max(local.y.abs());
        let dist = dist.max(0.0);
        smoothstep(0.0, self.edge_fade, dist)
    }
}

/// Clamps a cosine fade threshold to `[0, 1]`, substituting a default when the
/// supplied value is non-finite.
#[inline]
fn sanitize_cosine(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

/// Clamps the edge-fade width to `[0, 0.5]`, substituting `0` when non-finite.
#[inline]
fn sanitize_edge(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, HALF_EXTENT)
    } else {
        0.0
    }
}

/// Hermite `smoothstep` in `[0, 1]`.
///
/// Returns `0` at or below `edge0`, `1` at or above `edge1`, and a smooth cubic
/// ramp between.  A non-ascending `[edge0, edge1]` degenerates to a hard step at
/// `edge0` so a mis-ordered fade range never divides by zero.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if !(span > 0.0) {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Returns `true` when every component of `v` is finite.
#[inline]
fn vec3_is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Returns `true` when every column of `m` is finite.
#[inline]
fn matrix_is_finite(m: &Mat4) -> bool {
    [m.x_axis, m.y_axis, m.z_axis, m.w_axis]
        .iter()
        .all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite() && c.w.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4;

    /// Returns `true` when `a` and `b` agree to within `eps`.
    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Identity projector: world space *is* decal space, projecting along `+Z`.
    fn identity_projector() -> DecalProjector {
        DecalProjector::new(Mat4::IDENTITY, 0.5, 0.0, 0.1)
    }

    #[test]
    fn center_hit_has_half_uv() {
        let p = identity_projector();
        // Normal faces straight back at the projector (-projection_dir).
        let hit = p
            .project(Vec3::ZERO, -p.projection_dir)
            .expect("centre is inside the cube");
        assert!(approx(hit.uv.x, 0.5, 1.0e-6), "uv.x = {}", hit.uv.x);
        assert!(approx(hit.uv.y, 0.5, 1.0e-6), "uv.y = {}", hit.uv.y);
        assert!(approx(hit.depth, 0.5, 1.0e-6), "depth = {}", hit.depth);
        assert!(hit.fade > 0.0, "centre should be covered: {}", hit.fade);
    }

    #[test]
    fn out_of_bounds_misses() {
        let p = identity_projector();
        assert!(p.project(Vec3::new(2.0, 0.0, 0.0), Vec3::NEG_Z).is_none());
        assert!(p.project(Vec3::new(0.0, -0.9, 0.0), Vec3::NEG_Z).is_none());
        assert!(p.project(Vec3::new(0.0, 0.0, 0.51), Vec3::NEG_Z).is_none());
    }

    #[test]
    fn back_facing_normal_fades_to_zero() {
        let p = identity_projector();
        // projection_dir is +Z; a normal pointing along +Z is back-facing.
        let hit = p
            .project(Vec3::ZERO, p.projection_dir)
            .expect("still geometrically inside");
        assert!(approx(hit.fade, 0.0, 1.0e-6), "back face fade = {}", hit.fade);
        assert!(approx(p.angle_fade(Vec3::Z), 0.0, 1.0e-6));
    }

    #[test]
    fn edge_fade_is_monotonic_toward_wall() {
        let p = identity_projector();
        // Walk from the centre out toward the +x wall; edge fade must not rise.
        let mut prev = f32::INFINITY;
        let steps = 24;
        for i in 0..=steps {
            let x = (i as f32 / steps as f32) * 0.5; // 0 .. 0.5
            let local = Vec3::new(x, 0.0, 0.0);
            let f = p.edge_fade(local);
            assert!((0.0..=1.0).contains(&f), "edge fade out of range: {f}");
            assert!(f <= prev + 1.0e-6, "edge fade rose at x={x}: {f} > {prev}");
            prev = f;
        }
        // Interior is fully covered, the wall is fully faded.
        assert!(approx(p.edge_fade(Vec3::ZERO), 1.0, 1.0e-6));
        assert!(approx(p.edge_fade(Vec3::new(0.5, 0.0, 0.0)), 0.0, 1.0e-6));
    }

    #[test]
    fn angle_fade_is_monotonic_in_facing() {
        let p = identity_projector();
        let mut prev = -1.0;
        for i in 0..=20 {
            // Tilt the normal from edge-on toward straight-on.
            let t = i as f32 / 20.0;
            let n = Vec3::new(1.0 - t, 0.0, -t).normalize_or_zero();
            let f = p.angle_fade(n);
            assert!((0.0..=1.0).contains(&f));
            assert!(f >= prev - 1.0e-6, "angle fade dropped: {f} < {prev}");
            prev = f;
        }
    }

    #[test]
    fn translated_decal_projects_relative_to_its_box() {
        // Decal cube centred at world (10, 0, 0): world_to_decal translates back.
        let world_to_decal = Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0));
        let p = DecalProjector::new(world_to_decal, 0.5, 0.0, 0.0);
        let hit = p
            .project(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Z)
            .expect("decal centre");
        assert!(approx(hit.uv.x, 0.5, 1.0e-6));
        assert!(p.project(Vec3::ZERO, Vec3::NEG_Z).is_none(), "origin now outside");
    }

    #[test]
    fn singular_matrix_falls_back_to_z_axis() {
        // A zero scale on z collapses the cube; determinant is zero.
        let singular = Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0));
        let p = DecalProjector::new(singular, 0.5, 0.0, 0.0);
        assert_eq!(p.projection_dir, Vec3::Z);
    }

    #[test]
    fn non_finite_inputs_miss_without_panic() {
        let p = identity_projector();
        assert!(p.project(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::NEG_Z).is_none());
        // A non-finite normal still produces a finite fade (missing-normal path).
        let hit = p.project(Vec3::ZERO, Vec3::new(f32::INFINITY, 0.0, 0.0));
        assert!(hit.is_some());
        assert!(hit.unwrap().fade.is_finite());
    }

    #[test]
    fn missing_normal_is_treated_as_front_facing() {
        let p = identity_projector();
        assert!(approx(p.angle_fade(Vec3::ZERO), 1.0, 1.0e-6));
    }

    #[test]
    fn fade_range_degenerate_is_hard_cut() {
        // start <= end collapses the ramp to a hard threshold at `end`.
        let p = DecalProjector::new(Mat4::IDENTITY, 0.2, 0.2, 0.0);
        assert!(approx(p.angle_fade(-p.projection_dir), 1.0, 1.0e-6));
    }

    #[test]
    fn matrix_finiteness_helper_detects_nan() {
        let mut m = Mat4::IDENTITY;
        m.x_axis = Vec4::new(f32::NAN, 0.0, 0.0, 0.0);
        assert!(!matrix_is_finite(&m));
        assert!(matrix_is_finite(&Mat4::IDENTITY));
    }
}
