//! Six-plane view-frustum culling of axis-aligned boxes and spheres for the
//! particle culling pass (design §12, §13).
//!
//! The culling stage already holds the six oriented planes that bound the
//! visible volume; this module answers the yes/no/maybe question for each
//! bounds primitive: is an emitter's `AABB` or bounding sphere fully inside the
//! frustum, fully outside it, or straddling its boundary? That verdict gates
//! whether a whole emitter's particles are simulated, sorted, and rasterized
//! this frame, so the test must be cheap, branch-light, and bit-for-bit stable
//! between the reference `CPU` path and the future `GPU` culling kernel.
//!
//! # Symmetric-projection-radius test (no per-corner loop)
//! Rather than transform all eight box corners against every plane, this module
//! uses the classic p-vertex / n-vertex shortcut expressed as a *signed
//! center distance* plus a *projected radius*. For a plane with unit inward
//! normal `n` and offset `d` (inside half-space `n·p + d >= 0`):
//!
//! * the box center's signed distance is `s = n·center + d`;
//! * the box's extent projected onto `n` is
//!   `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z`, the half-width of the box's
//!   shadow on the plane normal.
//!
//! The most-inside corner (the p-vertex) then sits at `s + r` and the
//! most-outside corner (the n-vertex) at `s - r`. Hence for each plane:
//!
//! * `s + r < 0` — even the p-vertex is on the outside half-space, so the box
//!   is wholly outside this plane and therefore outside the frustum;
//! * `s - r < 0 <= s + r` — the box straddles this plane (n-vertex out,
//!   p-vertex in), contributing an `Intersecting` vote;
//! * `s - r >= 0` — the box is wholly inside this plane.
//!
//! A box is [`Visibility::Outside`] as soon as *any* plane rejects it,
//! [`Visibility::Intersecting`] when every plane admits the p-vertex but some
//! plane cuts the box, and [`Visibility::Inside`] when every plane fully
//! contains it. The sphere test is the same arithmetic with the projected
//! radius replaced by the scalar radius, so it never forms a square root.
//!
//! # Numerics
//! The only primitives used are `f32` add/sub/mul and [`f32::abs`]; there is no
//! `sqrt`, no transcendental function, and therefore no chance of a `NaN` from
//! this module's own math. `f32` magnitudes are never compared with `==`/`!=`:
//! sign decisions go through the tolerant [`CULL_EPS`] band so a primitive that
//! merely *touches* a plane is admitted rather than flickering between verdicts.
//! Callers must pass planes whose normals point **into** the frustum; the plane
//! normals are assumed already unit length so `s` reads as a true signed
//! Euclidean distance (see the module boundary below).
//!
//! # Module boundary
//! This module *consumes* a finished `[Plane; 6]`; it never derives one.
//!
//! * [`crate::particle::frustum_plane_extract`] is the producer: it runs the
//!   Gribb-Hartmann extraction on a combined view-projection (`VP`) matrix to
//!   build and normalize the six planes. This module takes that array as input.
//! * A hypothetical single-plane `plane_aabb_classify` would classify a box
//!   against *one* plane; here we fold that per-plane decision across all six
//!   planes into a single frustum verdict.
//! * [`crate::particle::occlusion`] and [`crate::particle::sort_cull`] implement
//!   *other* culling strategies (hierarchical-depth occlusion, distance/sort
//!   quantization). This module is strictly the frustum-vs-bounds geometric
//!   predicate they build on.

/// Tolerant epsilon for `f32` sign decisions; direct `==`/`!=` on `f32` is
/// forbidden by the crate's math rules, so every "inside/outside" test compares
/// against this band instead of against exact zero.
pub const CULL_EPS: f32 = 1e-6;

/// Number of planes bounding a frustum: left, right, bottom, top, near, far.
pub const PLANE_COUNT: usize = 6;

/// Dot (inner) product of two 3-vectors stored as `[f32; 3]`. Provided as a
/// free function so the plain array math type needs no operator-trait surface.
#[must_use]
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A frustum bounding plane in the form `n·p + d = 0`.
///
/// `normal` is `n = (nx, ny, nz)` and is assumed **unit length** and pointing
/// **into** the frustum, so the inside half-space is `n·p + d >= 0` and
/// `n·p + d` is a signed Euclidean distance from the plane. Producing and
/// normalizing planes is the job of
/// [`crate::particle::frustum_plane_extract`]; this module only reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit inward-facing plane normal `(nx, ny, nz)`.
    pub normal: [f32; 3],
    /// Plane offset `d` in the equation `n·p + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from its inward normal and offset.
    #[must_use]
    pub const fn new(normal: [f32; 3], d: f32) -> Self {
        Self { normal, d }
    }

    /// Signed distance from `point` to this plane. Positive means the point is
    /// on the inside (frustum) half-space, negative means outside.
    #[must_use]
    pub fn signed_distance(&self, point: [f32; 3]) -> f32 {
        v_dot(self.normal, point) + self.d
    }
}

/// An axis-aligned bounding box described by its `center` and non-negative
/// `half`-extents (so the box spans `center ± half` on each axis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Box center in the same space as the planes.
    pub center: [f32; 3],
    /// Non-negative half-extents along `x`, `y`, `z`.
    pub half: [f32; 3],
}

impl Aabb {
    /// Builds a box from its center and half-extents.
    #[must_use]
    pub const fn new(center: [f32; 3], half: [f32; 3]) -> Self {
        Self { center, half }
    }
}

/// A bounding sphere described by its `center` and non-negative `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Sphere center in the same space as the planes.
    pub center: [f32; 3],
    /// Non-negative sphere radius.
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere from its center and radius.
    #[must_use]
    pub const fn new(center: [f32; 3], radius: f32) -> Self {
        Self { center, radius }
    }
}

/// The three-way visibility verdict of a bounds primitive against the frustum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// Wholly contained by every plane's inside half-space.
    Inside,
    /// Rejected by at least one plane; wholly outside the frustum.
    Outside,
    /// Admitted by every plane's p-vertex but cut by at least one plane.
    Intersecting,
}

impl Visibility {
    /// Whether this verdict means the primitive contributes anything visible,
    /// i.e. it is [`Visibility::Inside`] or [`Visibility::Intersecting`].
    #[must_use]
    pub fn is_visible(self) -> bool {
        !matches!(self, Visibility::Outside)
    }
}

/// Half-width of the box's projection onto the axis `normal`, i.e.
/// `|n_x|·h_x + |n_y|·h_y + |n_z|·h_z`. This is the p-vertex/n-vertex offset
/// `r` such that the extreme corners sit at `center_distance ± r`.
#[must_use]
pub fn projected_radius(normal: [f32; 3], half: [f32; 3]) -> f32 {
    normal[0].abs() * half[0] + normal[1].abs() * half[1] + normal[2].abs() * half[2]
}

/// Classifies an `AABB` against the six frustum planes and returns whether it is
/// [`Visibility::Inside`], [`Visibility::Outside`], or
/// [`Visibility::Intersecting`].
///
/// Uses the symmetric projected-radius form: for each plane the box center's
/// signed distance `s` and projected radius `r` place the p-vertex at `s + r`
/// and the n-vertex at `s - r`. A single plane whose p-vertex is outside
/// (`s + r < 0`) short-circuits to [`Visibility::Outside`]; a plane that cuts
/// the box (`s - r < 0 <= s + r`) records an intersection; otherwise the box is
/// fully inside that plane. All sign tests use the [`CULL_EPS`] band.
#[must_use]
pub fn cull_aabb(planes: &[Plane; 6], aabb: &Aabb) -> Visibility {
    let mut intersecting = false;
    for plane in planes.iter() {
        let s = plane.signed_distance(aabb.center);
        let r = projected_radius(plane.normal, aabb.half);
        if s + r < -CULL_EPS {
            return Visibility::Outside;
        }
        if s - r < -CULL_EPS {
            intersecting = true;
        }
    }
    if intersecting {
        Visibility::Intersecting
    } else {
        Visibility::Inside
    }
}

/// Classifies a bounding sphere against the six frustum planes.
///
/// This is [`cull_aabb`]'s arithmetic with the projected radius replaced by the
/// scalar `radius`, so it forms no square root: for each plane the center's
/// signed distance `s` places the near and far sphere surfaces at `s ± radius`.
/// A plane with `s + radius < 0` rejects the sphere ([`Visibility::Outside`]);
/// a plane with `s - radius < 0 <= s + radius` cuts it
/// ([`Visibility::Intersecting`]); otherwise the sphere is inside that plane.
#[must_use]
pub fn cull_sphere(planes: &[Plane; 6], sphere: &Sphere) -> Visibility {
    let mut intersecting = false;
    for plane in planes.iter() {
        let s = plane.signed_distance(sphere.center);
        if s + sphere.radius < -CULL_EPS {
            return Visibility::Outside;
        }
        if s - sphere.radius < -CULL_EPS {
            intersecting = true;
        }
    }
    if intersecting {
        Visibility::Intersecting
    } else {
        Visibility::Inside
    }
}

/// Fast boolean visibility for an `AABB`: `true` when the box is inside or
/// intersecting, `false` when it is fully outside. Short-circuits on the first
/// rejecting plane, so it is cheaper than [`cull_aabb`] when the exact verdict
/// is not needed. Its result always agrees with `cull_aabb(..).is_visible()`.
#[must_use]
pub fn is_visible_aabb(planes: &[Plane; 6], aabb: &Aabb) -> bool {
    for plane in planes.iter() {
        let s = plane.signed_distance(aabb.center);
        let r = projected_radius(plane.normal, aabb.half);
        if s + r < -CULL_EPS {
            return false;
        }
    }
    true
}

/// Fast boolean visibility for a sphere: `true` when the sphere is inside or
/// intersecting, `false` when it is fully outside. Agrees with
/// `cull_sphere(..).is_visible()`.
#[must_use]
pub fn is_visible_sphere(planes: &[Plane; 6], sphere: &Sphere) -> bool {
    for plane in planes.iter() {
        let s = plane.signed_distance(sphere.center);
        if s + sphere.radius < -CULL_EPS {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `1 / sqrt(2)` as a literal so the diagonal frustum planes stay unit
    /// length without a runtime `sqrt` in a `const` context.
    const INV_SQRT2: f32 = core::f32::consts::FRAC_1_SQRT_2;

    /// A standard symmetric perspective frustum: apex at the origin looking down
    /// `+z` with a 45° half-angle, near plane at `z = 1`, far plane at
    /// `z = 100`. All normals point inward and are unit length.
    const SYMMETRIC: [Plane; 6] = [
        // left:  x >= -z  ->  x + z >= 0
        Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 0.0),
        // right: x <=  z  ->  z - x >= 0
        Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 0.0),
        // bottom: y >= -z ->  y + z >= 0
        Plane::new([0.0, INV_SQRT2, INV_SQRT2], 0.0),
        // top:    y <=  z ->  z - y >= 0
        Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 0.0),
        // near:  z >= 1
        Plane::new([0.0, 0.0, 1.0], -1.0),
        // far:   z <= 100
        Plane::new([0.0, 0.0, -1.0], 100.0),
    ];

    /// An oblique frustum whose planes are tilted off the world axes, to make
    /// sure the projected-radius math is not accidentally axis-aligned. It is a
    /// slab-like volume clipped by four skew planes plus near/far in `z`.
    const OBLIQUE: [Plane; 6] = [
        // +x-ish and +z tilt
        Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 2.0),
        // -x-ish and +z tilt
        Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 2.0),
        // +y-ish and +z tilt
        Plane::new([0.0, INV_SQRT2, INV_SQRT2], 2.0),
        // -y-ish and +z tilt
        Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 2.0),
        // near / far
        Plane::new([0.0, 0.0, 1.0], -1.0),
        Plane::new([0.0, 0.0, -1.0], 100.0),
    ];

    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn v_dot_matches_manual_sum() {
        assert!(approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn projected_radius_sums_absolute_contributions() {
        // Negative normal lanes must contribute their magnitude.
        assert!(approx(
            projected_radius([-1.0, 2.0, -3.0], [2.0, 3.0, 4.0]),
            1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0,
        ));
    }

    #[test]
    fn signed_distance_is_positive_inside() {
        // Near plane z >= 1: a point at z = 50 is 49 units inside.
        assert!(approx(SYMMETRIC[4].signed_distance([0.0, 0.0, 50.0]), 49.0));
    }

    #[test]
    fn small_box_at_center_is_inside() {
        let b = Aabb::new([0.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Inside);
    }

    #[test]
    fn box_far_left_is_outside() {
        let b = Aabb::new([-500.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_far_right_is_outside() {
        let b = Aabb::new([500.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_far_below_is_outside() {
        let b = Aabb::new([0.0, -500.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_far_above_is_outside() {
        let b = Aabb::new([0.0, 500.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_behind_near_plane_is_outside() {
        // z well below the near plane (behind the apex).
        let b = Aabb::new([0.0, 0.0, -20.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_beyond_far_plane_is_outside() {
        let b = Aabb::new([0.0, 0.0, 500.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn box_straddling_left_plane_intersects() {
        // Center sits right on the x = -z boundary at z = 50, so the box cuts
        // the left plane but stays within all others.
        let b = Aabb::new([-50.0, 0.0, 50.0], [5.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
    }

    #[test]
    fn box_straddling_near_plane_intersects() {
        // Center at z = 1 (on the near plane) with a half-extent in z.
        let b = Aabb::new([0.0, 0.0, 1.0], [0.2, 0.2, 2.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
    }

    #[test]
    fn box_straddling_far_plane_intersects() {
        let b = Aabb::new([0.0, 0.0, 100.0], [1.0, 1.0, 5.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
    }

    #[test]
    fn huge_box_containing_frustum_intersects() {
        // A box big enough to swallow the whole frustum straddles every plane.
        let b = Aabb::new([0.0, 0.0, 50.0], [1000.0, 1000.0, 1000.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
    }

    #[test]
    fn thin_flat_box_inside_is_inside() {
        // Degenerate zero-thickness box (half.y = 0) fully inside the frustum.
        let b = Aabb::new([0.0, 0.0, 50.0], [2.0, 0.0, 2.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Inside);
    }

    #[test]
    fn thin_flat_box_across_plane_intersects() {
        // Zero-thickness plane-like box lying across the near plane.
        let b = Aabb::new([0.0, 0.0, 1.0], [10.0, 10.0, 0.0]);
        // half.z = 0, so it touches only if center is exactly on the plane; give
        // it a tiny thickness to make the straddle unambiguous.
        let b = Aabb::new([b.center[0], b.center[1], b.center[2]], [10.0, 10.0, 0.5]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
    }

    #[test]
    fn point_box_deep_inside_is_inside() {
        // Zero-extent box (a point) well inside.
        let b = Aabb::new([0.0, 0.0, 50.0], [0.0, 0.0, 0.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Inside);
    }

    #[test]
    fn point_box_outside_is_outside() {
        let b = Aabb::new([0.0, 0.0, 500.0], [0.0, 0.0, 0.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Outside);
    }

    #[test]
    fn sphere_at_center_is_inside() {
        let s = Sphere::new([0.0, 0.0, 50.0], 2.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Inside);
    }

    #[test]
    fn sphere_far_left_is_outside() {
        let s = Sphere::new([-500.0, 0.0, 50.0], 2.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Outside);
    }

    #[test]
    fn sphere_beyond_far_plane_is_outside() {
        let s = Sphere::new([0.0, 0.0, 500.0], 10.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Outside);
    }

    #[test]
    fn sphere_behind_near_plane_is_outside() {
        let s = Sphere::new([0.0, 0.0, -50.0], 10.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Outside);
    }

    #[test]
    fn sphere_crossing_far_plane_intersects() {
        // Center exactly on the far plane (z = 100), radius pokes through both
        // sides -> the sphere is cut by that plane.
        let s = Sphere::new([0.0, 0.0, 100.0], 5.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Intersecting);
    }

    #[test]
    fn sphere_internally_tangent_to_far_plane_is_inside() {
        // Center at z = 95 with radius 5: the far surface just touches z = 100.
        // A touching (tangent) sphere is admitted as Inside, not Intersecting.
        let s = Sphere::new([0.0, 0.0, 95.0], 5.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Inside);
    }

    #[test]
    fn sphere_externally_tangent_to_far_plane_intersects() {
        // Center at z = 105 with radius 5: the near surface just touches z = 100
        // from outside. Tangent-from-outside counts as a touch, i.e. cut, not a
        // full rejection.
        let s = Sphere::new([0.0, 0.0, 105.0], 5.0);
        assert_eq!(cull_sphere(&SYMMETRIC, &s), Visibility::Intersecting);
    }

    #[test]
    fn oblique_frustum_center_box_is_inside() {
        // The oblique volume admits points near the z axis for mid-range z.
        let b = Aabb::new([0.0, 0.0, 50.0], [0.5, 0.5, 0.5]);
        assert_eq!(cull_aabb(&OBLIQUE, &b), Visibility::Inside);
    }

    #[test]
    fn oblique_frustum_offset_box_is_outside() {
        // Push far along +x so the tilted -x plane (normal has +x lane) rejects.
        let b = Aabb::new([400.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&OBLIQUE, &b), Visibility::Outside);
    }

    #[test]
    fn oblique_frustum_straddle_intersects() {
        // A wide box centered on the axis reaches through the tilted side plane.
        let b = Aabb::new([0.0, 0.0, 50.0], [60.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&OBLIQUE, &b), Visibility::Intersecting);
    }

    #[test]
    fn is_visible_aabb_agrees_with_inside() {
        let b = Aabb::new([0.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert_eq!(
            is_visible_aabb(&SYMMETRIC, &b),
            cull_aabb(&SYMMETRIC, &b).is_visible(),
        );
        assert!(is_visible_aabb(&SYMMETRIC, &b));
    }

    #[test]
    fn is_visible_aabb_agrees_with_outside() {
        let b = Aabb::new([0.0, 0.0, 500.0], [1.0, 1.0, 1.0]);
        assert_eq!(
            is_visible_aabb(&SYMMETRIC, &b),
            cull_aabb(&SYMMETRIC, &b).is_visible(),
        );
        assert!(!is_visible_aabb(&SYMMETRIC, &b));
    }

    #[test]
    fn is_visible_aabb_agrees_with_intersecting() {
        let b = Aabb::new([-50.0, 0.0, 50.0], [5.0, 1.0, 1.0]);
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Intersecting);
        assert_eq!(
            is_visible_aabb(&SYMMETRIC, &b),
            cull_aabb(&SYMMETRIC, &b).is_visible(),
        );
        assert!(is_visible_aabb(&SYMMETRIC, &b));
    }

    #[test]
    fn is_visible_sphere_agrees_inside() {
        let s = Sphere::new([0.0, 0.0, 50.0], 2.0);
        assert_eq!(
            is_visible_sphere(&SYMMETRIC, &s),
            cull_sphere(&SYMMETRIC, &s).is_visible(),
        );
        assert!(is_visible_sphere(&SYMMETRIC, &s));
    }

    #[test]
    fn is_visible_sphere_agrees_outside() {
        let s = Sphere::new([0.0, 0.0, 500.0], 5.0);
        assert_eq!(
            is_visible_sphere(&SYMMETRIC, &s),
            cull_sphere(&SYMMETRIC, &s).is_visible(),
        );
        assert!(!is_visible_sphere(&SYMMETRIC, &s));
    }

    #[test]
    fn visibility_is_visible_flags() {
        assert!(Visibility::Inside.is_visible());
        assert!(Visibility::Intersecting.is_visible());
        assert!(!Visibility::Outside.is_visible());
    }

    #[test]
    fn box_on_left_boundary_touch_is_inside() {
        // p-vertex exactly on the left plane (touch): the tolerant band admits
        // it as Inside rather than flickering to Intersecting.
        let b = Aabb::new([-49.0, 0.0, 50.0], [1.0, 1.0, 0.0]);
        // signed distance of the +x-most corner to left plane: ((-48)+50)/sqrt2
        // > 0, and n-vertex ((-50)+50)/sqrt2 = 0 -> touch, treated as Inside.
        assert_eq!(cull_aabb(&SYMMETRIC, &b), Visibility::Inside);
    }

    #[test]
    fn plane_count_constant_is_six() {
        assert_eq!(PLANE_COUNT, SYMMETRIC.len());
    }
}
