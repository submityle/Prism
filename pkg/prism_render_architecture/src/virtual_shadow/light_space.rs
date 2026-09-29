//! Projection of world-space bounds onto a directional light's shadow plane.
//!
//! The clipmap decision layer ([`super::clipmap`], [`super::coverage`]) works
//! entirely in 2D light space: it never sees world positions, only the `(x, y)`
//! a receiver occupies on the plane perpendicular to the light. This module is
//! the bridge that produces those coordinates. It builds an orthonormal basis
//! whose two in-plane axes span that plane, then projects a world-space
//! [`SceneBounds`] box onto the axes to yield the tight 2D axis-aligned footprint
//! the coverage pass consumes.
//!
//! Projecting an axis-aligned box onto an arbitrary axis is exact and cheap: the
//! projected centre is the dot of the box centre with the axis, and the
//! projected half-extent is the dot of the box half-extents with the axis'
//! component magnitudes. No corner enumeration is needed. The layer keeps all of
//! this GPU-independent and trigonometry-free (only a normalize, i.e. a square
//! root, is used) so it stays deterministic and unit-testable.

use crate::gpu_scene::SceneBounds;

/// Orthonormal basis of a directional light's shadow space.
///
/// `right` and `up` are unit, mutually perpendicular in-plane axes spanning the
/// plane perpendicular to `direction`; `direction` is the unit light travel
/// direction (the depth axis, discarded by the 2D clipmap). A receiver's light
/// space `x` is its projection onto `right` and `y` its projection onto `up`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLightBasis {
    /// In-plane horizontal axis (unit).
    pub right: [f32; 3],
    /// In-plane vertical axis (unit).
    pub up: [f32; 3],
    /// Light travel direction / depth axis (unit).
    pub direction: [f32; 3],
}

impl DirectionalLightBasis {
    /// Builds a stable orthonormal basis from a light `direction`.
    ///
    /// Returns `None` when `direction` is (near) zero-length and no meaningful
    /// plane exists. The in-plane axes are derived from a world up-hint, falling
    /// back to a side hint when the light points nearly straight up or down so
    /// the cross product never degenerates.
    #[must_use]
    pub fn from_direction(direction: [f32; 3]) -> Option<Self> {
        let dir = normalize(direction)?;
        // Pick a reference axis least parallel to the light to avoid a
        // near-zero cross product; world up unless the light is near-vertical.
        let hint = if abs(dir[1]) > 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let right = normalize(cross(hint, dir))?;
        // `dir` and `right` are unit and perpendicular, so their cross is unit.
        let up = cross(dir, right);
        Some(Self {
            right,
            up,
            direction: dir,
        })
    }

    /// Projects a world-space point onto the shadow plane's `(x, y)`.
    #[must_use]
    pub fn project_point(&self, point: [f32; 3]) -> [f32; 2] {
        [dot(point, self.right), dot(point, self.up)]
    }

    /// Projects an axis-aligned world box onto its tight 2D light-space AABB.
    ///
    /// Returns `(min, max)` where each axis extent is the box centre's
    /// projection plus or minus the projected half-extent. Because a box's
    /// projection onto an axis is symmetric about its centre, this is exact
    /// without enumerating the eight corners.
    #[must_use]
    pub fn project_bounds(&self, bounds: &SceneBounds) -> ([f32; 2], [f32; 2]) {
        let center = self.project_point(bounds.center);
        let ex = projected_extent(bounds.half_extents, self.right);
        let ey = projected_extent(bounds.half_extents, self.up);
        (
            [center[0] - ex, center[1] - ey],
            [center[0] + ex, center[1] + ey],
        )
    }
}

/// Projected half-extent of an axis-aligned box onto `axis`.
///
/// For a box with half-extents `h`, the support along any axis `a` is
/// `|a.x|·h.x + |a.y|·h.y + |a.z|·h.z`.
#[must_use]
fn projected_extent(half_extents: [f32; 3], axis: [f32; 3]) -> f32 {
    abs(axis[0]) * half_extents[0] + abs(axis[1]) * half_extents[1] + abs(axis[2]) * half_extents[2]
}

/// Dot product of two 3-vectors.
#[must_use]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
#[must_use]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes a 3-vector, or `None` when it is (near) zero-length.
#[must_use]
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = dot(v, v);
    if len_sq <= f32::EPSILON {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// Absolute value that stays clippy-clean (no disallowed float methods).
#[must_use]
fn abs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        abs(a - b) <= 1.0e-5
    }

    #[test]
    fn zero_direction_has_no_basis() {
        assert!(DirectionalLightBasis::from_direction([0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn basis_is_orthonormal() {
        let b = DirectionalLightBasis::from_direction([0.3, -1.0, 0.5]).expect("basis");
        // Unit length.
        assert!(approx(dot(b.right, b.right), 1.0));
        assert!(approx(dot(b.up, b.up), 1.0));
        assert!(approx(dot(b.direction, b.direction), 1.0));
        // Mutually perpendicular.
        assert!(approx(dot(b.right, b.up), 0.0));
        assert!(approx(dot(b.right, b.direction), 0.0));
        assert!(approx(dot(b.up, b.direction), 0.0));
    }

    #[test]
    fn near_vertical_light_still_orthonormal() {
        // Straight down would degenerate a world-up hint; the side hint saves it.
        let b = DirectionalLightBasis::from_direction([0.0, -1.0, 0.0]).expect("basis");
        assert!(approx(dot(b.right, b.up), 0.0));
        assert!(approx(dot(b.right, b.right), 1.0));
        assert!(approx(dot(b.up, b.up), 1.0));
    }

    #[test]
    fn top_down_light_projects_ground_plane_directly() {
        // Light pointing down -Y: the shadow plane is the world XZ plane, so a
        // box's XZ half-extents become its light-space footprint size.
        let b = DirectionalLightBasis::from_direction([0.0, -1.0, 0.0]).expect("basis");
        let bounds = SceneBounds {
            center: [10.0, 5.0, -4.0],
            radius: 3.0,
            half_extents: [2.0, 7.0, 3.0],
            _padding: 0.0,
        };
        let (min, max) = b.project_bounds(&bounds);
        // Footprint size along each axis is twice the in-plane half-extent; the
        // vertical (Y) half-extent does not inflate the ground footprint.
        let w = max[0] - min[0];
        let h = max[1] - min[1];
        assert!(approx(w, 4.0) || approx(w, 6.0));
        assert!(approx(h, 4.0) || approx(h, 6.0));
        // The two in-plane extents are exactly the X (2) and Z (3) half-extents.
        assert!(approx(w + h, 4.0 + 6.0));
    }

    #[test]
    fn point_projection_matches_bounds_center() {
        let b = DirectionalLightBasis::from_direction([1.0, -2.0, 0.5]).expect("basis");
        let bounds = SceneBounds {
            center: [3.0, -1.0, 2.0],
            radius: 0.0,
            half_extents: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };
        let (min, max) = b.project_bounds(&bounds);
        let p = b.project_point([3.0, -1.0, 2.0]);
        // A zero-size box collapses to its centre's projection.
        assert!(approx(min[0], p[0]));
        assert!(approx(max[0], p[0]));
        assert!(approx(min[1], p[1]));
        assert!(approx(max[1], p[1]));
    }
}
