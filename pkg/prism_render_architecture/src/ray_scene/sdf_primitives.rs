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
//! [`torus`] and [`capsule`] cover the common swept shapes;
//! [`capped_cylinder`], [`capped_cone`], and [`hex_prism`] are the extruded
//! and revolved solids that round out the catalogue. The box, cylinder, cone,
//! and hex prism all use the split interior/exterior form so the distance
//! stays exact (not merely a bound) both inside and out.
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

/// Squared Euclidean length of a 2-vector (`dot(v, v)`), used by the
/// nearest-feature comparisons in the cone solver.
fn dot2_2(v: [f32; 2]) -> f32 {
    v[0] * v[0] + v[1] * v[1]
}

/// Euclidean length of a 2-vector.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Signed distance from `point` to a capped cylinder aligned with the `y`
/// axis, of the given `radius` and `half_height` (so it spans `y` in
/// `[-half_height, half_height]`), centred at the origin.
///
/// The point is reduced to its radial distance in the `xz` plane paired with
/// its `y` offset, then measured against the rectangular cross-section with
/// the exact interior/exterior split, giving a true distance on both sides of
/// the side wall and the end caps.
pub fn capped_cylinder(point: [f32; 3], half_height: f32, radius: f32) -> f32 {
    let radial = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let d = [radial - radius, point[1].abs() - half_height];
    let inside = d[0].max(d[1]).min(0.0);
    let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
    inside + outside
}

/// Signed distance from `point` to a capped cone aligned with the `y` axis,
/// spanning `y` in `[-half_height, half_height]`, with `bottom_radius` at the
/// lower cap and `top_radius` at the upper cap.
///
/// Follows Inigo Quilez's exact capped-cone solver: it compares the squared
/// distance to the nearer cap rim against the squared distance to the slanted
/// side segment (projecting onto it with a clamped parameter), takes the
/// smaller, and signs it negative when the point lies inside both the lateral
/// and the cap slabs. Setting `top_radius == bottom_radius` recovers a
/// cylinder; a zero cap radius recovers a true cone tip.
pub fn capped_cone(
    point: [f32; 3],
    half_height: f32,
    bottom_radius: f32,
    top_radius: f32,
) -> f32 {
    let q = [(point[0] * point[0] + point[2] * point[2]).sqrt(), point[1]];
    let k1 = [top_radius, half_height];
    let k2 = [top_radius - bottom_radius, 2.0 * half_height];
    // Snap the radius to whichever cap the point faces for the rim distance.
    let cap_radius = if q[1] < 0.0 { bottom_radius } else { top_radius };
    let ca = [q[0] - q[0].min(cap_radius), q[1].abs() - half_height];
    // Project onto the slanted side segment, parameter clamped to the caps.
    let k1_minus_q = [k1[0] - q[0], k1[1] - q[1]];
    let proj =
        ((k1_minus_q[0] * k2[0] + k1_minus_q[1] * k2[1]) / dot2_2(k2)).clamp(0.0, 1.0);
    let cb = [q[0] - k1[0] + k2[0] * proj, q[1] - k1[1] + k2[1] * proj];
    let sign = if cb[0] < 0.0 && ca[1] < 0.0 { -1.0 } else { 1.0 };
    sign * dot2_2(ca).min(dot2_2(cb)).sqrt()
}

/// Signed distance from `point` to a regular hexagonal prism extruded along
/// the `z` axis, with the given hexagon `apothem` (centre-to-flat-face
/// distance) and `half_depth` along `z`, centred at the origin.
///
/// Folds the point into one hexagon sextant using the precomputed constant
/// normal `k = (-cos 30 degrees, sin 30 degrees, 1 / sqrt 3)`, then applies
/// the exact interior/exterior split against the folded face and the depth
/// caps. Every trigonometric term is baked into a constant, so evaluation is
/// transcendental-free. A flat face sits at distance `apothem` along `y`.
pub fn hex_prism(point: [f32; 3], apothem: f32, half_depth: f32) -> f32 {
    // k = (-cos(30 degrees), sin(30 degrees), 1 / sqrt(3)); baked constants.
    const K: [f32; 3] = [-0.866_025_4, 0.5, 0.577_35];
    let mut p = [point[0].abs(), point[1].abs(), point[2].abs()];
    // Reflect across the sextant boundary so one slice covers the hexagon.
    let fold = 2.0 * (K[0] * p[0] + K[1] * p[1]).min(0.0);
    p[0] -= fold * K[0];
    p[1] -= fold * K[1];
    let clamped_x = p[0].clamp(-K[2] * apothem, K[2] * apothem);
    let face = [p[0] - clamped_x, p[1] - apothem];
    let sign = if p[1] - apothem < 0.0 { -1.0 } else { 1.0 };
    let d = [length2(face) * sign, p[2] - half_depth];
    let inside = d[0].max(d[1]).min(0.0);
    let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
    inside + outside
}

#[cfg(test)]
mod tests {
    use super::{
        box_sdf, capped_cone, capped_cylinder, capsule, hex_prism, plane, round_box, sphere,
        torus,
    };

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

    #[test]
    fn capped_cylinder_side_cap_and_corner() {
        let (h, r) = (1.0, 1.0);
        // One unit past the side wall.
        assert!((capped_cylinder([2.0, 0.0, 0.0], h, r) - 1.0).abs() < 1e-6);
        // Dead centre: equidistant from side and caps, one unit inside.
        assert!((capped_cylinder([0.0, 0.0, 0.0], h, r) - (-1.0)).abs() < 1e-6);
        // One unit above the top cap on the axis.
        assert!((capped_cylinder([0.0, 2.0, 0.0], h, r) - 1.0).abs() < 1e-6);
        // Diagonal past the top rim: length of the (1, 1) overshoot.
        let corner = capped_cylinder([2.0, 2.0, 0.0], h, r);
        assert!((corner - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn capped_cone_reduces_to_cylinder_when_radii_match() {
        // Equal radii: the slanted side is vertical, so it matches a cylinder.
        let cone = capped_cone([2.0, 0.0, 0.0], 1.0, 1.0, 1.0);
        let cyl = capped_cylinder([2.0, 0.0, 0.0], 1.0, 1.0);
        assert!((cone - cyl).abs() < 1e-6);
        // Interior point is signed negative.
        assert!((capped_cone([0.0, 0.0, 0.0], 1.0, 1.0, 1.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn capped_cone_measures_distance_above_the_tip() {
        // Tip cone: bottom radius 1, top radius 0, apex at y = +half_height.
        // A point one unit above the apex is one unit outside.
        let d = capped_cone([0.0, 2.0, 0.0], 1.0, 1.0, 0.0);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hex_prism_apothem_face_and_depth() {
        let (apothem, half_depth) = (1.0, 2.0);
        // Centre: one apothem inside the nearest flat face.
        assert!((hex_prism([0.0, 0.0, 0.0], apothem, half_depth) - (-1.0)).abs() < 1e-6);
        // The +y flat face sits at y = apothem; one unit beyond it.
        assert!((hex_prism([0.0, 2.0, 0.0], apothem, half_depth) - 1.0).abs() < 1e-6);
        // One unit past the +z depth cap.
        assert!((hex_prism([0.0, 0.0, 3.0], apothem, half_depth) - 1.0).abs() < 1e-6);
    }
}
