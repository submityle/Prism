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
//! and revolved solids; [`box_frame`] is the hollow wireframe of a box and
//! [`octahedron`] is the exact dual of the cube. The box, cylinder, cone, hex
//! prism, box frame, and octahedron all yield a true distance (not merely a
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

/// Signed distance from `point` to the hollow wireframe of an axis-aligned
/// box: the twelve square-section bars running along the edges of a box of
/// the given `half_extent`, each bar `thickness` wide, centred at the origin.
///
/// Follows Inigo Quilez's exact `sdBoxFrame`. The point is folded into the
/// positive octant and offset by the frame thickness, then the three
/// axis-aligned bar families are measured with the interior/exterior split and
/// combined with a minimum, so the hollow interior and the bar solids both
/// carry a true distance.
pub fn box_frame(point: [f32; 3], half_extent: [f32; 3], thickness: f32) -> f32 {
    let p = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let q = [
        (p[0] + thickness).abs() - thickness,
        (p[1] + thickness).abs() - thickness,
        (p[2] + thickness).abs() - thickness,
    ];
    // One bar family per axis: keep that axis sharp while rounding the others.
    let bar_x = length([p[0].max(0.0), q[1].max(0.0), q[2].max(0.0)])
        + p[0].max(q[1].max(q[2])).min(0.0);
    let bar_y = length([q[0].max(0.0), p[1].max(0.0), q[2].max(0.0)])
        + q[0].max(p[1].max(q[2])).min(0.0);
    let bar_z = length([q[0].max(0.0), q[1].max(0.0), p[2].max(0.0)])
        + q[0].max(q[1].max(p[2])).min(0.0);
    bar_x.min(bar_y).min(bar_z)
}

/// Signed distance from `point` to a regular octahedron (the dual of the cube)
/// with the given `radius` from the centre to each of its six vertices along
/// the axes, centred at the origin.
///
/// Follows Inigo Quilez's exact `sdOctahedron`. The point is folded into the
/// positive octant; when it sits inside the central slab the distance is the
/// scaled `L1` excess over the face plane, otherwise the dominant axis is
/// rotated into place and the distance is measured to the slanted face via a
/// clamped projection. The inscribed radius is `radius / sqrt 3`.
pub fn octahedron(point: [f32; 3], radius: f32) -> f32 {
    let p = [point[0].abs(), point[1].abs(), point[2].abs()];
    let m = p[0] + p[1] + p[2] - radius;
    // Rotate the dominant axis into `q.x` so one slanted face solves all eight.
    let q = if 3.0 * p[0] < m {
        [p[0], p[1], p[2]]
    } else if 3.0 * p[1] < m {
        [p[1], p[2], p[0]]
    } else if 3.0 * p[2] < m {
        [p[2], p[0], p[1]]
    } else {
        // Inside the central slab: L1 excess scaled onto the face normal.
        return m * 0.577_350_26;
    };
    let k = (0.5 * (q[2] - q[1] + radius)).clamp(0.0, radius);
    length([q[0], q[1] - radius + k, q[2] - k])
}

/// Approximate signed distance from `point` to an axis-aligned ellipsoid with
/// per-axis `radii`, centred at the origin.
///
/// Unlike the other primitives this is **not** an exact Euclidean distance:
/// an ellipsoid has no closed-form distance, so this uses Inigo Quilez's
/// gradient-corrected bound `k0 * (k0 - 1) / k1`, where `k0 = length(p / r)`
/// and `k1 = length(p / r / r)`. The sign is correct everywhere (negative
/// inside, positive outside) and the zero level set is the true surface, but
/// off-surface magnitudes are a close approximation rather than the metric
/// distance; it degrades for very eccentric radii. Any `radii` component must
/// be non-zero. Suitable for sphere tracing and `CSG`, where a slight
/// underestimate of distance only costs extra marching steps.
pub fn ellipsoid_sdf(point: [f32; 3], radii: [f32; 3]) -> f32 {
    let scaled = [point[0] / radii[0], point[1] / radii[1], point[2] / radii[2]];
    let k0 = length(scaled);
    let k1 = length([
        scaled[0] / radii[0],
        scaled[1] / radii[1],
        scaled[2] / radii[2],
    ]);
    // At the exact centre `k1` is zero; the surface is `k0 = 0` away, so the
    // distance is simply the smallest radius (nearest surface point).
    if k1 <= f32::MIN_POSITIVE {
        return -radii[0].min(radii[1]).min(radii[2]);
    }
    k0 * (k0 - 1.0) / k1
}

/// Signed distance from `point` to a square-based pyramid of the given
/// `height`, resting on the `y = 0` plane with a unit base (side 1, corners at
/// `(+/-0.5, 0, +/-0.5)`) and apex at `(0, height, 0)`.
///
/// This is Inigo Quilez's exact pyramid distance. The query is folded into one
/// octant by taking `abs` of the base-plane coordinates and sorting them, so a
/// single slanted face solves all four; the distance is then the minimum of
/// the squared distances to that face and to the base-edge region, combined
/// with the folded coordinate and signed by whether the point sits above the
/// base. Exact (not a bound) on both sides. Built from `abs`, `min`, `max`,
/// `clamp`, `signum`, and a single `sqrt`, so it stays transcendental-free.
pub fn pyramid(point: [f32; 3], height: f32) -> f32 {
    let m2 = height * height + 0.25;
    // Fold into one octant of the base plane, then sort so x >= z.
    let mut px = point[0].abs();
    let mut pz = point[2].abs();
    if pz > px {
        core::mem::swap(&mut px, &mut pz);
    }
    px -= 0.5;
    pz -= 0.5;
    let py = point[1];

    let qx = pz;
    let qy = height * py - 0.5 * px;
    let qz = height * px + 0.5 * py;

    let s = (-qx).max(0.0);
    let t = ((qy - 0.5 * pz) / (m2 + 0.25)).clamp(0.0, 1.0);

    let a = m2 * (qx + s) * (qx + s) + qy * qy;
    let b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
    // Inside the wedge above both the slanted face and the base edge the point
    // projects straight down the face, so the planar squared distance is zero.
    let d2 = if qy.min(-qx * m2 - qy * 0.5) > 0.0 {
        0.0
    } else {
        a.min(b)
    };

    ((d2 + qz * qz) / m2).sqrt() * qz.max(-py).signum()
}

/// Signed distance from `point` to a chain link: a torus of ring radius `r1`
/// and tube radius `r2` stretched by `half_length` along the `y` axis (so the
/// straight sides are `2 * half_length` long and the ring lies in the `x`-`y`
/// plane about the `z` axis).
///
/// This is Inigo Quilez's exact `sdLink`: the `y` coordinate is collapsed onto
/// the stadium's straight run (`max(|y| - half_length, 0)`), reducing the query
/// to the planar torus distance. With `half_length = 0` it is exactly a torus.
/// Built from `abs`, `max`, and two vector lengths, so it stays
/// transcendental-free.
pub fn link(point: [f32; 3], half_length: f32, r1: f32, r2: f32) -> f32 {
    let qy = (point[1].abs() - half_length).max(0.0);
    let planar = length2([point[0], qy]) - r1;
    length2([planar, point[2]]) - r2
}

/// Signed distance from `point` to a cut sphere: a sphere of radius `radius`
/// sliced by the horizontal plane `y = cut_height`, keeping the portion with
/// `y >= cut_height` and capping the removed bottom with a flat disc of radius
/// `w = sqrt(radius^2 - cut_height^2)` at that height.
///
/// This is Inigo Quilez's exact `sdCutSphere`. The query is reduced to the
/// `(radial, y)` half-plane, where `radial = length(point.xz)`. A single
/// comparison `s` selects the governing feature: the spherical cap when the
/// point sits in the sphere's angular sector, the flat disc face when it lies
/// directly under the cap, or the circular rim where cap meets disc otherwise.
/// Built from `abs`, `min`, `max`, and two vector lengths, so it stays
/// transcendental-free.
///
/// The slice height is only geometrically meaningful for
/// `cut_height` in `[-radius, radius]`; the cap-radius `sqrt` is guarded with
/// `.max(0.0)` so out-of-range inputs degrade to a plain hemisphere boundary
/// instead of producing `NaN`.
pub fn cut_sphere(point: [f32; 3], radius: f32, cut_height: f32) -> f32 {
    let r = radius;
    let h = cut_height;
    let w = (r * r - h * h).max(0.0).sqrt();
    let qx = length2([point[0], point[2]]);
    let qy = point[1];
    let s = ((h - r) * qx * qx + w * w * (h + r - 2.0 * qy)).max(h * qx - w * qy);
    if s < 0.0 {
        length2([qx, qy]) - r
    } else if qx < w {
        h - qy
    } else {
        length2([qx - w, qy - h])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        box_frame, box_sdf, capped_cone, capped_cylinder, capsule, cut_sphere, ellipsoid_sdf,
        hex_prism,
        link, octahedron, plane, pyramid, round_box, sphere, torus,
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

    #[test]
    fn box_frame_outside_bar_and_hollow_centre() {
        let half = [1.0, 1.0, 1.0];
        let e = 0.1;
        // One unit out along +x from the (y=1, z=1) edge bar: on the bar axis.
        assert!((box_frame([2.0, 1.0, 1.0], half, e) - 1.0).abs() < 1e-6);
        // Dead centre lies in the hollow, far from every bar (positive).
        let centre = box_frame([0.0, 0.0, 0.0], half, e);
        assert!((centre - (1.28f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn octahedron_vertex_centre_and_face() {
        let r = 1.0;
        // Vertex sits at (1, 0, 0); one unit beyond it along the axis.
        assert!((octahedron([2.0, 0.0, 0.0], r) - 1.0).abs() < 1e-6);
        // Centre: negative inscribed radius r / sqrt(3).
        assert!((octahedron([0.0, 0.0, 0.0], r) - (-0.577_350_26)).abs() < 1e-6);
        // A point on the +++ face plane is on the surface.
        assert!(octahedron([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], r).abs() < 1e-6);
    }

    #[test]
    fn ellipsoid_sphere_case_matches_exact_distance() {
        // Equal radii degenerate to a sphere, where the IQ bound is exact.
        let r = [1.0, 1.0, 1.0];
        // On axis at distance 2 from a unit sphere: distance 1.
        assert!((ellipsoid_sdf([2.0, 0.0, 0.0], r) - 1.0).abs() < 1e-5);
        // On the surface: zero.
        assert!(ellipsoid_sdf([1.0, 0.0, 0.0], r).abs() < 1e-5);
    }

    #[test]
    fn ellipsoid_sign_is_correct_inside_and_out() {
        let r = [2.0, 1.0, 0.5];
        // Centre is inside (negative).
        assert!(ellipsoid_sdf([0.0, 0.0, 0.0], r) < 0.0);
        // A point just inside the +x tip (x < 2) is negative.
        assert!(ellipsoid_sdf([1.9, 0.0, 0.0], r) < 0.0);
        // A point outside the +x tip (x > 2) is positive.
        assert!(ellipsoid_sdf([2.5, 0.0, 0.0], r) > 0.0);
    }

    #[test]
    fn ellipsoid_zero_level_set_is_the_surface() {
        let r = [2.0, 1.0, 0.5];
        // Each axis tip lies on the surface (distance ~ 0).
        assert!(ellipsoid_sdf([2.0, 0.0, 0.0], r).abs() < 1e-5);
        assert!(ellipsoid_sdf([0.0, 1.0, 0.0], r).abs() < 1e-5);
        assert!(ellipsoid_sdf([0.0, 0.0, 0.5], r).abs() < 1e-5);
    }

    #[test]
    fn ellipsoid_centre_is_negative_smallest_radius() {
        let r = [2.0, 1.0, 0.5];
        assert!((ellipsoid_sdf([0.0, 0.0, 0.0], r) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn pyramid_apex_lies_on_the_surface() {
        // The apex sits at (0, h, 0); the exact distance there is zero.
        assert!(pyramid([0.0, 1.0, 0.0], 1.0).abs() < 1e-6);
        assert!(pyramid([0.0, 3.0, 0.0], 3.0).abs() < 1e-6);
    }

    #[test]
    fn pyramid_point_above_apex_is_the_vertical_gap() {
        // A point one unit above the apex is exactly one unit outside, for any
        // height (the apex is the nearest surface point straight down).
        assert!((pyramid([0.0, 2.0, 0.0], 1.0) - 1.0).abs() < 1e-5);
        assert!((pyramid([0.0, 4.0, 0.0], 3.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pyramid_base_corner_lies_on_the_surface() {
        // The unit base has corners at (+/-0.5, 0, +/-0.5); each is on-surface.
        assert!(pyramid([0.5, 0.0, 0.5], 1.0).abs() < 1e-5);
        assert!(pyramid([-0.5, 0.0, -0.5], 2.0).abs() < 1e-5);
    }

    #[test]
    fn pyramid_interior_point_is_negative() {
        // A point on the axis a quarter of the way up is strictly inside.
        assert!(pyramid([0.0, 0.25, 0.0], 1.0) < 0.0);
    }

    #[test]
    fn pyramid_side_point_matches_base_edge_distance() {
        // Far out along +x at base level, the nearest surface point is the base
        // edge midpoint (0.5, 0, 0), so the distance is the planar overshoot.
        assert!((pyramid([3.0, 0.0, 0.0], 1.0) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn link_reduces_to_a_torus_with_zero_length() {
        // half_length = 0 is a plain torus: the ring centreline is -r2 inside
        // the tube and the outer equator sits on the surface.
        assert!((link([1.0, 0.0, 0.0], 0.0, 1.0, 0.3) - (-0.3)).abs() < 1e-6);
        assert!(link([1.3, 0.0, 0.0], 0.0, 1.0, 0.3).abs() < 1e-6);
    }

    #[test]
    fn link_straight_section_tracks_the_tube() {
        // Along the stretched y run the cross-section is still the tube: the
        // centreline is -r2 and the +x surface point is on the boundary.
        assert!((link([1.0, 0.5, 0.0], 0.5, 1.0, 0.3) - (-0.3)).abs() < 1e-6);
        assert!(link([1.3, 0.5, 0.0], 0.5, 1.0, 0.3).abs() < 1e-6);
    }

    #[test]
    fn link_hole_centre_is_outside_the_solid() {
        // The centre of the ring hole is r1 - r2 outside the tube.
        assert!((link([0.0, 0.0, 0.0], 0.5, 1.0, 0.3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn cut_sphere_spherical_cap_matches_the_sphere() {
        // With radius 1 cut at the equator (h = 0) the retained cap is the
        // upper hemisphere, so the spherical part is still exactly the sphere:
        // the top pole is on the surface and the point above it is at distance 1.
        assert!(cut_sphere([0.0, 1.0, 0.0], 1.0, 0.0).abs() < 1e-6);
        assert!((cut_sphere([0.0, 2.0, 0.0], 1.0, 0.0) - 1.0).abs() < 1e-6);
        // The equatorial rim where the cut plane meets the sphere is on the
        // surface, reached through the rim branch.
        assert!(cut_sphere([1.0, 0.0, 0.0], 1.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn cut_sphere_flat_disc_face_is_planar() {
        // Directly under the cap the governing feature is the flat disc at
        // y = h: the disc centre is on the surface and points below it measure
        // their vertical drop to the plane.
        assert!(cut_sphere([0.0, 0.0, 0.0], 1.0, 0.0).abs() < 1e-6);
        assert!((cut_sphere([0.0, -0.5, 0.0], 1.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((cut_sphere([0.0, -1.0, 0.0], 1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cut_sphere_interior_is_negative() {
        // A point inside the retained volume reports a negative distance equal
        // to its depth below the spherical cap.
        assert!((cut_sphere([0.0, 0.5, 0.0], 1.0, 0.0) - (-0.5)).abs() < 1e-6);
    }
}
