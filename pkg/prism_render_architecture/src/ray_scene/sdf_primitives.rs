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

/// Signed distance from `point` to a rhombic prism: a rhombus in the `xz`
/// plane with half-diagonals `half_diag_x` (along `x`) and `half_diag_z`
/// (along `z`), extruded to `half_height` along `y`, with its edges rounded by
/// `rounding`.
///
/// This is Inigo Quilez's exact `sdRhombus`. The point is folded into the
/// first octant, projected onto the nearest rhombus edge via the clamped
/// parameter `f`, then the planar edge distance (signed by which side of the
/// edge the point lies on) is paired with the vertical cap distance and
/// resolved with the standard rounded-box interior/exterior combine. Built
/// from `abs`, `sign`, `clamp`, `min`, `max`, and vector lengths, so it stays
/// transcendental-free.
pub fn rhombus(
    point: [f32; 3],
    half_diag_x: f32,
    half_diag_z: f32,
    half_height: f32,
    rounding: f32,
) -> f32 {
    let px = point[0].abs();
    let py = point[1].abs();
    let pz = point[2].abs();
    let bx = half_diag_x;
    let bz = half_diag_z;
    // `ndot(b, b - 2*p.xz) = bx*(bx - 2*px) - bz*(bz - 2*pz)`; the clamped
    // ratio is the fractional position of the foot of the perpendicular along
    // the rhombus edge.
    let ndot = bx * (bx - 2.0 * px) - bz * (bz - 2.0 * pz);
    let denom = bx * bx + bz * bz;
    let f = (ndot / denom).clamp(-1.0, 1.0);
    let foot_x = 0.5 * bx * (1.0 - f);
    let foot_z = 0.5 * bz * (1.0 + f);
    let edge = length2([px - foot_x, pz - foot_z]);
    let side = (px * bz + pz * bx - bx * bz).signum();
    let qx = edge * side - rounding;
    let qy = py - half_height;
    qx.max(qy).min(0.0) + length2([qx.max(0.0), qy.max(0.0)])
}

/// Signed distance from `point` to a vesica lens: the intersection of two
/// spheres of radius `radius` whose centres sit at `(+-half_separation, 0, 0)`
/// in the equatorial plane, revolved into the convex lens that is symmetric
/// about the `y` axis. The lens has cusps on the `y` axis at
/// `+-sqrt(radius^2 - half_separation^2)` and an equatorial circle of radius
/// `radius - half_separation` in the `xz` plane.
///
/// This is Inigo Quilez's exact 2D `sdVesica` applied in the meridian
/// half-plane `(length(point.xz), |point.y|)`; because the profile is convex
/// the nearest surface point always lies in the query's meridian plane, so the
/// revolved distance is exact everywhere (including the on-axis cusps). The
/// branch test picks the spherical arc or the cusp tip as the governing
/// feature. Built from `abs`, `sign`, `min`, `max`, and vector lengths, so it
/// stays transcendental-free.
///
/// `half_separation` is only meaningful for `0 <= half_separation <= radius`;
/// the cusp-height `sqrt` is guarded with `.max(0.0)` so out-of-range inputs
/// degrade gracefully instead of producing `NaN`.
pub fn vesica(point: [f32; 3], radius: f32, half_separation: f32) -> f32 {
    let r = radius;
    let d = half_separation;
    let qx = length2([point[0], point[2]]);
    let qy = point[1].abs();
    let b = (r * r - d * d).max(0.0).sqrt();
    if (qy - b) * d > qx * b {
        length2([qx, qy - b]) * d.signum()
    } else {
        length2([qx + d, qy]) - r
    }
}

/// Signed distance from `point` to a capped torus: a torus arc whose tube of
/// radius `tube_radius` is swept around a major circle of radius
/// `major_radius` in the `x`-`y` plane, but only across an angular aperture of
/// `+-half_angle` about the `+y` axis (the rest of the ring is removed and
/// closed off with flat disc caps).
///
/// `sin_cos_aperture` is the baked `(sin(half_angle), cos(half_angle))` of that
/// half-angle, so the caller folds the only trigonometry into two constants
/// and this routine stays transcendental-free. With `half_angle = PI` it is a
/// full torus; with `half_angle = PI/2` it is a half torus.
///
/// This is Inigo Quilez's exact `sdCappedTorus`: the query is folded across
/// the `yz` plane, a single comparison selects whether the nearest feature is
/// the swept tube (`k = length(point.xy)`) or one of the flat end caps
/// (`k = dot(point.xy, sin_cos_aperture)`), and the tube radius is subtracted
/// from the resulting ring distance. Built from `abs`, `dot`, `min`/`max`
/// comparisons, and two square roots.
pub fn capped_torus(
    point: [f32; 3],
    sin_cos_aperture: [f32; 2],
    major_radius: f32,
    tube_radius: f32,
) -> f32 {
    let px = point[0].abs();
    let py = point[1];
    let pz = point[2];
    let [sin_a, cos_a] = sin_cos_aperture;
    // Past the aperture the nearest feature is the flat cap, reached by
    // projecting onto the cap direction; inside it the full ring applies.
    let k = if cos_a * px > sin_a * py {
        px * sin_a + py * cos_a
    } else {
        length2([px, py])
    };
    let p_dot_p = px * px + py * py + pz * pz;
    (p_dot_p + major_radius * major_radius - 2.0 * major_radius * k).max(0.0).sqrt() - tube_radius
}

/// Signed distance from `point` to a solid equilateral triangular prism
/// extruded along the `z` axis, centred at the origin.
///
/// The cross-section is an equilateral triangle lying in the `xy` plane with
/// side length `2 * size`: its apex sits at `(0, 2 * size / sqrt 3)` and its
/// base edge runs between `(-size, -size / sqrt 3)` and `(size, -size / sqrt
/// 3)`, giving an inradius of `size / sqrt 3`. The prism spans `-half_depth` to
/// `half_depth` along `z`.
///
/// Evaluates Inigo Quilez's exact equilateral-triangle 2D distance (an `x`
/// fold, a single slanted-edge fold, then a base-edge clamp, so corners carry
/// the true vertex distance) and extrudes it exactly: the planar distance and
/// the depth-cap excess are combined with the standard interior/exterior split.
/// Every trigonometric term is baked into the `sqrt 3` constant, so evaluation
/// is transcendental-free, and because the triangle is convex the extrusion
/// stays an exact signed distance.
pub fn triangular_prism(point: [f32; 3], size: f32, half_depth: f32) -> f32 {
    /// Baked `sqrt 3`, the equilateral-triangle fold constant.
    const SQRT3: f32 = 1.732_050_8;
    // Exact 2D equilateral-triangle distance evaluated in the xy plane.
    let mut x = point[0].abs() - size;
    let mut y = point[1] + size / SQRT3;
    if x + SQRT3 * y > 0.0 {
        let folded_x = (x - SQRT3 * y) * 0.5;
        let folded_y = (-SQRT3 * x - y) * 0.5;
        x = folded_x;
        y = folded_y;
    }
    x -= x.clamp(-2.0 * size, 0.0);
    let planar = -length2([x, y]) * if y < 0.0 { -1.0 } else { 1.0 };
    // Exact extrusion of the planar distance against the depth caps.
    let cap = [planar, point[2].abs() - half_depth];
    let inside = cap[0].max(cap[1]).min(0.0);
    let outside = length2([cap[0].max(0.0), cap[1].max(0.0)]);
    inside + outside
}

/// Signed distance from `point` to a solid angular sector: the intersection
/// of a ball of the given `radius` (centred at the origin) with an infinite
/// cone whose apex is at the origin and whose axis points along `+y`.
///
/// The cone half-angle is supplied pre-baked as `sin_cos = (sin angle, cos
/// angle)`, so the caller bakes the only trigonometry and evaluation here is
/// transcendental-free. The shape is the classic "ice-cream cone" region used
/// for spotlight and sector volumes.
///
/// Follows Inigo Quilez's exact `sdSolidAngle`. The point is reduced to its
/// meridian coordinates `(radial distance from the axis, height)`; the ball is
/// measured directly while the cone flank is measured against the clamped
/// projection onto the flank ray, and the two are combined so the rounded cap,
/// the straight flank, and the apex each carry a true distance.
pub fn solid_angle(point: [f32; 3], sin_cos: [f32; 2], radius: f32) -> f32 {
    // Meridian coordinates: radial distance from the +y axis, then height.
    let q = [length2([point[0], point[2]]), point[1]];
    // Distance to the bounding sphere.
    let ball = length2(q) - radius;
    // Distance to the cone flank: project onto the flank ray, clamped to the
    // sphere radius so the flank terminates at the cap, then measure the gap.
    let proj = (q[0] * sin_cos[0] + q[1] * sin_cos[1]).clamp(0.0, radius);
    let flank = length2([q[0] - sin_cos[0] * proj, q[1] - sin_cos[1] * proj]);
    // Sign the flank distance by which side of the flank ray the point lies on.
    let flank_sign = if sin_cos[1] * q[0] - sin_cos[0] * q[1] < 0.0 {
        -1.0
    } else {
        1.0
    };
    ball.max(flank * flank_sign)
}

/// Signed distance from `point` to a round cone: the convex hull of a sphere
/// of radius `r1` centred at the origin and a sphere of radius `r2` centred at
/// `(0, h, 0)`, i.e. a tapered capsule running along the `+y` axis.
///
/// Follows Inigo Quilez's exact `sdRoundCone`. The point is reduced to meridian
/// coordinates `(radial distance from the y axis, height)`; the slope constant
/// `b = (r1 - r2) / h` and its complement `a = sqrt(1 - b * b)` define the
/// flank normal. Below the lower cap the bottom sphere governs, above the flank
/// band the top sphere governs, and in between the exact flank-plane distance
/// applies, so every region carries a true signed distance. The lone `sqrt` is
/// guarded against the degenerate `|b| > 1` case and no transcendental function
/// is used otherwise.
pub fn round_cone_sdf(point: [f32; 3], r1: f32, r2: f32, h: f32) -> f32 {
    let q = [length2([point[0], point[2]]), point[1]];
    let b = (r1 - r2) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    // Project onto the flank normal to pick the governing region.
    let k = -b * q[0] + a * q[1];
    if k < 0.0 {
        // Below the lower cap: the bottom sphere governs.
        length2(q) - r1
    } else if k > a * h {
        // Above the flank band: the top sphere governs.
        length2([q[0], q[1] - h]) - r2
    } else {
        // On the flank band: exact distance to the tangent cone plane.
        a * q[0] + b * q[1] - r1
    }
}

/// Signed distance from `point` to a cut hollow sphere: the thin spherical
/// cap of the sphere of the given `radius` (centred at the origin) that lies
/// below the plane `y = cut_height`, inflated to a shell of half-`thickness`.
///
/// Follows Inigo Quilez's exact `sdCutHollowSphere`. The rim radius where the
/// cut plane meets the sphere is `rim = sqrt(radius^2 - cut_height^2)` (guarded
/// against a cut above the pole). In meridian coordinates `(radial distance
/// from the y axis, height)`, points past the rim's radial cone take the exact
/// distance to the rim circle while the rest take the distance to the sphere
/// surface; subtracting `thickness` turns the zero-thickness cap into a shell.
/// Only `sqrt` is used, so evaluation is transcendental-free.
pub fn cut_hollow_sphere(point: [f32; 3], radius: f32, cut_height: f32, thickness: f32) -> f32 {
    // Radius of the circular rim carved by the plane y = cut_height.
    let rim = (radius * radius - cut_height * cut_height).max(0.0).sqrt();
    let q = [length2([point[0], point[2]]), point[1]];
    // Past the rim's radial cone the rim circle governs; otherwise the sphere.
    let surface = if cut_height * q[0] < rim * q[1] {
        length2([q[0] - rim, q[1] - cut_height])
    } else {
        (length2(q) - radius).abs()
    };
    surface - thickness
}

#[cfg(test)]
mod tests {
    use super::{
        box_frame, box_sdf, capped_cone, capped_cylinder, capped_torus, capsule, cut_hollow_sphere, cut_sphere,
        ellipsoid_sdf, hex_prism, length2, link, octahedron, plane, pyramid, rhombus, round_box,
        round_cone_sdf, solid_angle, sphere, torus, triangular_prism, vesica,
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

    #[test]
    fn rhombus_edge_and_vertex_lie_on_the_surface() {
        // Unit rhombus (half-diagonals 1, half-height 1, no rounding): the edge
        // midpoint x+z=1 and the x-axis vertex both sit on the boundary.
        assert!(rhombus([0.5, 0.0, 0.5], 1.0, 1.0, 1.0, 0.0).abs() < 1e-6);
        assert!(rhombus([1.0, 0.0, 0.0], 1.0, 1.0, 1.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn rhombus_interior_measures_the_nearest_edge() {
        // The centre is a half-diagonal's perpendicular 1/sqrt(2) inside the
        // nearest slanted edge, which is closer than the vertical cap.
        let expected = -(0.5f32).sqrt();
        assert!((rhombus([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 0.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn rhombus_above_the_cap_is_the_vertical_gap() {
        // Directly above the rhombus the governing feature is the top cap, so
        // the distance is the overshoot past half-height.
        assert!((rhombus([0.0, 2.0, 0.0], 1.0, 1.0, 1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vesica_equator_and_cusp_lie_on_the_surface() {
        // r = 1, d = 0.6 -> cusp height b = 0.8, equator radius r - d = 0.4.
        // The equatorial rim and the top cusp both sit on the boundary.
        assert!(vesica([0.4, 0.0, 0.0], 1.0, 0.6).abs() < 1e-6);
        assert!(vesica([0.0, 0.8, 0.0], 1.0, 0.6).abs() < 1e-6);
    }

    #[test]
    fn vesica_above_the_cusp_measures_the_tip() {
        // Straight above the cusp the governing feature is the tip, so the
        // distance is the gap past the cusp height.
        assert!((vesica([0.0, 1.8, 0.0], 1.0, 0.6) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vesica_centre_is_the_inset_to_the_arc() {
        // The centre is radius - half_separation inside the nearest spherical
        // arc, reported as a negative distance.
        assert!((vesica([0.0, 0.0, 0.0], 1.0, 0.6) - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn vesica_degenerates_to_a_sphere_without_separation() {
        // With half_separation = 0 both generating spheres coincide, so the
        // lens is exactly a sphere of the given radius.
        assert!(vesica([1.0, 0.0, 0.0], 1.0, 0.0).abs() < 1e-6);
        assert!((vesica([0.0, 0.0, 0.0], 1.0, 0.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn capped_torus_tube_cross_section_is_exact() {
        // Half torus (half_angle = 90deg -> sin_cos = (1, 0)), major 1, tube
        // 0.2. On the +y ring centreline the distance is -tube_radius; the ring
        // surface is reached at the tube radius both radially and in z.
        let sc = [1.0, 0.0];
        assert!((capped_torus([0.0, 1.0, 0.0], sc, 1.0, 0.2) - (-0.2)).abs() < 1e-6);
        assert!(capped_torus([0.0, 1.2, 0.0], sc, 1.0, 0.2).abs() < 1e-6);
        assert!(capped_torus([0.0, 1.0, 0.2], sc, 1.0, 0.2).abs() < 1e-6);
    }

    #[test]
    fn capped_torus_outside_the_tube_tracks_the_ring() {
        // A point beyond the tube on the ring plane is its radial gap to the
        // tube surface.
        assert!((capped_torus([0.0, 1.5, 0.0], [1.0, 0.0], 1.0, 0.2) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn capped_torus_past_the_aperture_measures_the_end_cap() {
        // half_angle = 30deg: the ring ends at angle 30deg from +y, centreline
        // endpoint (sin30, cos30). A query past the aperture (at 45deg) is
        // governed by the flat end cap, i.e. its distance to that endpoint minus
        // the tube radius.
        let sin_a = 0.5f32;
        let cos_a = (0.75f32).sqrt(); // cos 30deg
        let sc = [sin_a, cos_a];
        let major = 1.0f32;
        let tube = 0.2f32;
        let q = [(0.5f32).sqrt(), (0.5f32).sqrt(), 0.0]; // 45deg on the ring radius
        let cap_centre = [sin_a * major, cos_a * major];
        let expected = length2([q[0] - cap_centre[0], q[1] - cap_centre[1]]) - tube;
        assert!((capped_torus(q, sc, major, tube) - expected).abs() < 1e-6);
    }

    /// Baked `sqrt 3` for the triangular-prism geometry assertions.
    const SQRT3_T: f32 = 1.732_050_8;

    /// Minimum distance from a 2D `point` to the equilateral triangle used by
    /// [`triangular_prism`] with the given `size`, measured by brute force over
    /// its three edges. Serves as an independent reference for the analytic
    /// planar distance.
    fn tri_edge_distance(point: [f32; 2], size: f32) -> f32 {
        let verts = [
            [0.0, 2.0 * size / SQRT3_T],
            [-size, -size / SQRT3_T],
            [size, -size / SQRT3_T],
        ];
        let mut best = f32::MAX;
        for i in 0..3 {
            let a = verts[i];
            let b = verts[(i + 1) % 3];
            let ab = [b[0] - a[0], b[1] - a[1]];
            let ap = [point[0] - a[0], point[1] - a[1]];
            let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]))
                .clamp(0.0, 1.0);
            let c = [a[0] + ab[0] * t, a[1] + ab[1] * t];
            let d = length2([point[0] - c[0], point[1] - c[1]]);
            best = best.min(d);
        }
        best
    }

    #[test]
    fn triangular_prism_centre_is_negative_inradius() {
        // Deep prism: the planar inradius dominates the depth cap.
        let inradius = 1.0 / SQRT3_T;
        assert!((triangular_prism([0.0, 0.0, 0.0], 1.0, 10.0) - (-inradius)).abs() < 1e-6);
        // Shallow prism: the depth half-thickness dominates instead.
        assert!((triangular_prism([0.0, 0.0, 0.0], 1.0, 0.1) - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn triangular_prism_apex_and_base_distances() {
        // Straight above the apex: nearest feature is the apex vertex.
        let apex_y = 2.0 / SQRT3_T;
        assert!((triangular_prism([0.0, apex_y + 1.0, 0.0], 1.0, 10.0) - 1.0).abs() < 1e-6);
        // Below the horizontal base edge: nearest feature is that edge.
        let base_y = 1.0 / SQRT3_T;
        assert!((triangular_prism([0.0, -base_y - 0.5, 0.0], 1.0, 10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn triangular_prism_extrudes_exactly_along_z() {
        // Over the interior footprint, past the depth cap: pure z overshoot.
        assert!((triangular_prism([0.0, 0.0, 10.0], 1.0, 2.0) - 8.0).abs() < 1e-6);
        // Corner of the end cap: diagonal of planar excess and depth excess.
        let apex_y = 2.0 / SQRT3_T;
        let expected = length2([0.5, 0.5]);
        let got = triangular_prism([0.0, apex_y + 0.5, 2.5], 1.0, 2.0);
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn triangular_prism_planar_matches_brute_force() {
        // Deep prism so the result reduces to the exact planar triangle SDF for
        // z = 0 queries; compare exterior points against the brute-force edge
        // distance.
        for &p in &[[2.0, 2.0], [-1.5, 0.3], [0.7, -2.0], [3.0, 0.0]] {
            let got = triangular_prism([p[0], p[1], 0.0], 1.0, 10.0);
            let reference = tri_edge_distance(p, 1.0);
            assert!(
                (got - reference).abs() < 1e-5,
                "point {p:?}: got {got}, reference {reference}"
            );
        }
    }

    #[test]
    fn solid_angle_axis_and_cap_distances() {
        // sin/cos of a 30-degree half-angle, baked without trigonometry.
        let sc = [0.5f32, (0.75f32).sqrt()];
        let ra = 2.0f32;
        // Interior axis point: nearest feature is the cone flank, at the
        // perpendicular distance height * sin(angle), signed negative inside.
        assert!((solid_angle([0.0, 1.0, 0.0], sc, ra) - (-0.5)).abs() < 1e-6);
        // On the spherical cap along the axis.
        assert!(solid_angle([0.0, 2.0, 0.0], sc, ra).abs() < 1e-6);
        // Beyond the cap along the axis: pure radial overshoot.
        assert!((solid_angle([0.0, 3.0, 0.0], sc, ra) - 1.0).abs() < 1e-6);
        // Below the apex (outside the cone): nearest feature is the apex.
        assert!((solid_angle([0.0, -1.0, 0.0], sc, ra) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn solid_angle_flank_distances() {
        let sc = [0.5f32, (0.75f32).sqrt()];
        let ra = 2.0f32;
        // Equatorial point outside the cone: perpendicular distance to the
        // flank ray equals radial * cos(angle) while the projection stays in
        // range.
        let expected = 1.5 * (0.75f32).sqrt();
        assert!((solid_angle([1.5, 0.0, 0.0], sc, ra) - expected).abs() < 1e-6);
        // Interior off-axis point: inside both the ball and the cone, governed
        // by the flank with a hand-derived reference.
        assert!((solid_angle([0.5, 1.5, 0.0], sc, ra) - (-0.316_987_3)).abs() < 1e-5);
    }

    #[test]
    fn round_cone_sphere_caps_and_interior() {
        let (r1, r2, h) = (1.0f32, 0.4f32, 2.0f32);
        // Bottom sphere centre: negative its radius.
        assert!((round_cone_sdf([0.0, 0.0, 0.0], r1, r2, h) - (-1.0)).abs() < 1e-6);
        // Top sphere centre: negative its radius.
        assert!((round_cone_sdf([0.0, 2.0, 0.0], r1, r2, h) - (-0.4)).abs() < 1e-6);
        // Bottom tip sits on the lower sphere surface.
        assert!(round_cone_sdf([0.0, -1.0, 0.0], r1, r2, h).abs() < 1e-6);
    }

    #[test]
    fn round_cone_exterior_regions() {
        let (r1, r2, h) = (1.0f32, 0.4f32, 2.0f32);
        // Above the top cap: distance to the top sphere.
        assert!((round_cone_sdf([0.0, 3.0, 0.0], r1, r2, h) - 0.6).abs() < 1e-6);
        // Radially out at the base: governed by the bottom sphere.
        assert!((round_cone_sdf([2.0, 0.0, 0.0], r1, r2, h) - 1.0).abs() < 1e-6);
        // On the tapered flank band: exact tangent-plane distance.
        let expected = (0.91f32).sqrt() + 0.3 - 1.0;
        assert!((round_cone_sdf([1.0, 1.0, 0.0], r1, r2, h) - expected).abs() < 1e-6);
    }

    #[test]
    fn round_cone_equal_radii_is_a_capsule() {
        // Equal radii collapse the flank to a cylinder, i.e. a capsule.
        let r = 0.5f32;
        // Mid-height, exactly one radius off the axis: on the surface.
        assert!(round_cone_sdf([0.5, 1.0, 0.0], r, r, 2.0).abs() < 1e-6);
        // Mid-height interior point.
        assert!((round_cone_sdf([0.2, 1.0, 0.0], r, r, 2.0) - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn cut_hollow_sphere_shell_centre_on_the_cap() {
        let (r, h, t) = (1.0f32, 0.5f32, 0.1f32);
        let rim = (0.75f32).sqrt();
        // Bottom pole lies on the cap arc: the shell centre is at -thickness.
        assert!((cut_hollow_sphere([0.0, -1.0, 0.0], r, h, t) - (-0.1)).abs() < 1e-6);
        // The rim point itself is also on the arc.
        assert!((cut_hollow_sphere([rim, 0.5, 0.0], r, h, t) - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn cut_hollow_sphere_rim_and_sphere_branches() {
        let (r, h, t) = (1.0f32, 0.5f32, 0.1f32);
        // Above the cut plane on the axis: nearest feature is the rim circle.
        let expected_rim = (3.0f32).sqrt() - t;
        assert!((cut_hollow_sphere([0.0, 2.0, 0.0], r, h, t) - expected_rim).abs() < 1e-6);
        // Radially outside at rim height: the sphere surface governs.
        let expected_sphere = (4.25f32).sqrt() - 1.0 - t;
        assert!((cut_hollow_sphere([2.0, 0.5, 0.0], r, h, t) - expected_sphere).abs() < 1e-6);
        // Inside the sphere, below the cap on the axis: sphere-surface distance.
        assert!((cut_hollow_sphere([0.0, -0.6, 0.0], r, h, t) - 0.3).abs() < 1e-6);
    }
}
