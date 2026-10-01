//! Analytic signed distance primitives for the `CPU` golden path.
//!
//! The mesh pipeline bakes a signed distance field from triangles, but implicit
//! modelling also needs *analytic* primitives whose exact distance is known in
//! closed form. These are the atoms the domain and CSG operators
//! ([`super::sdf_domain`], [`super::sdf_csg`]) compose into complex shapes, and
//! they are what `AAA` tools evaluate when ray-marching procedural geometry:
//! fast, allocation-free, and exact everywhere rather than sampled on a grid.
//!
//! Each function returns the signed Euclidean distance from a query point to
//! the surface — negative inside, positive outside, zero on the boundary —
//! following Inigo Quilez's canonical formulations. [`sphere`], [`box_sdf`],
//! and [`round_box`] bound convex solids; [`plane`] is a half-space;
//! [`torus`] and [`capsule`] cover the common swept shapes. The box uses the
//! split interior/exterior form so the distance stays exact (not merely a
//! bound) both inside and out.
//!
//! Every primitive is built from `abs`, `min`, `max`, `clamp`, dot products,
//! and the `sqrt` inside a vector length — all permitted — so the module is
//! transcendental-free and reproducible.

/// Euclidean length of a 3-vector.
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Signed distance from `point` to a sphere of `radius` centred at the origin.
///
/// Negative inside, positive outside; simply the point's distance from the
/// centre minus the radius.
pub fn sphere(point: [f32; 3], radius: f32) -> f32 {
    length(point) - radius
}

/// Signed distance from `point` to an axis-aligned box of the given
/// `half_extent`, centred at the origin.
///
/// Uses the exact interior/exterior split: the exterior term is the length of
/// the positive overshoot past each face, and the interior term is the largest
/// (least negative) face distance clamped at zero, so the result is a true
/// distance on both sides of the surface rather than a conservative bound.
pub fn box_sdf(point: [f32; 3], half_extent: [f32; 3]) -> f32 {
    let q = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let outside = length([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]);
    let inside = q[0].max(q[1].max(q[2])).min(0.0);
    outside + inside
}

/// Signed distance to an axis-aligned box with rounded edges and corners.
///
/// Inflates [`box_sdf`] outward by `radius`, filleting every edge to that
/// radius while keeping the exact distance field.
pub fn round_box(point: [f32; 3], half_extent: [f32; 3], radius: f32) -> f32 {
    box_sdf(point, half_extent) - radius
}

/// Signed distance from `point` to a plane with unit `normal` whose signed
/// offset from the origin is `offset`.
///
/// `normal` must already be unit length; the result is `dot(point, normal) +
/// offset`, positive on the side the normal points toward.
pub fn plane(point: [f32; 3], normal: [f32; 3], offset: f32) -> f32 {
    dot(point, normal) + offset
}

/// Signed distance from `point` to a torus in the `xz` plane with the given
/// `major_radius` (ring centre to tube centre) and `minor_radius` (tube).
///
/// The point is reduced to its distance from the ring circle in the `xz`
/// plane paired with its `y` offset, then compared against the tube radius.
pub fn torus(point: [f32; 3], major_radius: f32, minor_radius: f32) -> f32 {
    let ring = (point[0] * point[0] + point[2] * point[2]).sqrt() - major_radius;
    (ring * ring + point[1] * point[1]).sqrt() - minor_radius
}

/// Signed distance from `point` to a capsule: the segment from `a` to `b`
/// swept by a sphere of `radius`.
///
/// Projects the point onto the segment (clamping the parameter to the
/// endpoints), then returns the distance to that closest segment point minus
/// the radius. A degenerate segment (`a == b`) reduces to a sphere at `a`.
pub fn capsule(point: [f32; 3], a: [f32; 3], b: [f32; 3], radius: f32) -> f32 {
    let pa = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ba_len_sq = dot(ba, ba);
    // Clamp the projection so the closest point stays on the finite segment;
    // a zero-length segment pins the parameter at the start point.
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
    length(closest) - radius
}

#[cfg(test)]
mod tests {
    use super::{box_sdf, capsule, plane, round_box, sphere, torus};

    #[test]
    fn sphere_is_centre_distance_minus_radius() {
        assert!((sphere([2.0, 0.0, 0.0], 1.0) - 1.0).abs() < 1e-6);
        assert!((sphere([0.0, 0.0, 0.0], 1.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn box_exact_inside_and_outside() {
        let half = [1.0, 1.0, 1.0];
        // Outside one face: distance is the overshoot.
        assert!((box_sdf([2.0, 0.0, 0.0], half) - 1.0).abs() < 1e-6);
        // Dead centre: negative distance to the nearest face.
        assert!((box_sdf([0.0, 0.0, 0.0], half) - (-1.0)).abs() < 1e-6);
        // Diagonal corner overshoot: length of the positive remainder.
        let corner = box_sdf([2.0, 2.0, 2.0], half);
        assert!((corner - (3.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn round_box_insets_by_radius() {
        let half = [1.0, 1.0, 1.0];
        let sharp = box_sdf([2.0, 0.0, 0.0], half);
        let round = round_box([2.0, 0.0, 0.0], half, 0.25);
        assert!((sharp - round - 0.25).abs() < 1e-6);
    }

    #[test]
    fn plane_is_signed_offset_along_normal() {
        assert!((plane([0.0, 2.0, 0.0], [0.0, 1.0, 0.0], 0.0) - 2.0).abs() < 1e-6);
        assert!((plane([0.0, -2.0, 0.0], [0.0, 1.0, 0.0], 0.0) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn torus_measures_distance_to_the_tube() {
        // On the ring centre circle: inside the tube by the minor radius.
        assert!((torus([2.0, 0.0, 0.0], 2.0, 0.5) - (-0.5)).abs() < 1e-6);
        // One unit outward in the plane: half a unit past the tube.
        assert!((torus([3.0, 0.0, 0.0], 2.0, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn capsule_projects_onto_the_segment() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        // Beside the segment midpoint: on the surface at radius 0.5.
        assert!((capsule([0.5, 0.5, 0.0], a, b, 0.5)).abs() < 1e-6);
        // Past the segment, projection clamps to the endpoint cap.
        assert!((capsule([1.0, 0.5, 0.0], a, b, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn capsule_degenerate_segment_is_a_sphere() {
        let a = [1.0, 1.0, 1.0];
        // Zero-length segment: distance reduces to a sphere about `a`.
        let d = capsule([3.0, 1.0, 1.0], a, a, 0.5);
        assert!((d - 1.5).abs() < 1e-6);
    }
}
