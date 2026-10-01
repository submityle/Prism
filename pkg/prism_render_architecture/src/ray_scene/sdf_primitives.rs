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

/// Squared Euclidean length of a 3-vector (`dot(v, v)`), used by the
/// nearest-feature comparisons in the exact triangle solver.
fn dot2_3(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Component-wise difference `a - b` of two 3-vectors.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a x b` of two 3-vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

/// Exact surface normal (unit gradient) of [`plane`]: the plane's own
/// `normal`, which is constant everywhere. `normal` must already be unit
/// length (the same precondition as [`plane`]); the gradient of
/// `dot(point, normal) + offset` is exactly `normal`.
pub fn plane_gradient(normal: [f32; 3]) -> [f32; 3] {
    normal
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

/// Exact surface normal (unit gradient of the signed distance) of [`sphere`]
/// at `point`: the outward radial direction `point / |point|`. Returns the
/// zero vector at the degenerate centre where the normal is undefined.
///
/// A sphere's SDF is a true distance field, so its gradient is already unit
/// length everywhere off the centre; this is the analytic normal with zero
/// finite-difference error, suitable as the shading normal on the analytic
/// golden path and as the ground truth for the sampled-field normal estimators.
pub fn sphere_gradient(point: [f32; 3]) -> [f32; 3] {
    let l = length(point);
    if l == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [point[0] / l, point[1] / l, point[2] / l]
}

/// Exact surface normal (unit gradient) of [`box_sdf`] at `point` for a box of
/// `half_extent`.
///
/// Outside the box the gradient points along the positive overshoot
/// `max(|p| - b, 0)`, normalised and re-signed per axis; inside, the nearest
/// face is the single least-negative axis, so the normal is the unit axis
/// vector along that face (sign of the corresponding coordinate). The result is
/// unit length away from edges/corners (the measure-zero creases where the
/// normal is genuinely undefined). This is the analytic normal of
/// [`box_sdf`] with no finite-difference error.
pub fn box_gradient(point: [f32; 3], half_extent: [f32; 3]) -> [f32; 3] {
    let q = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let m = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let len = length(m);
    if len > 0.0 {
        // Exterior: normalise the overshoot and restore each axis' sign.
        return [
            point[0].signum() * m[0] / len,
            point[1].signum() * m[1] / len,
            point[2].signum() * m[2] / len,
        ];
    }
    // Interior: nearest face is the least-negative axis (largest q_i).
    if q[0] >= q[1] && q[0] >= q[2] {
        [point[0].signum(), 0.0, 0.0]
    } else if q[1] >= q[2] {
        [0.0, point[1].signum(), 0.0]
    } else {
        [0.0, 0.0, point[2].signum()]
    }
}

/// Exact surface normal (unit gradient) of [`torus`] at `point` for the ring of
/// `major_radius` and tube `minor_radius`.
///
/// With `rho = |p.xz|` and the reduced coordinates `q = (rho - major, p.y)`,
/// the gradient is `((q.x / |q|) * p.xz / rho, q.y / |q|)` — the planar
/// component points radially in the `xz` plane while the `y` component follows
/// the tube cross-section; it is unit length by construction. On the central
/// `y` axis (`rho = 0`) the planar direction is undefined, so the pure `+y`
/// axis is returned as a stable fallback.
pub fn torus_gradient(point: [f32; 3], major_radius: f32, minor_radius: f32) -> [f32; 3] {
    let _ = minor_radius; // the normal is independent of the tube radius
    let rho = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let qx = rho - major_radius;
    let qy = point[1];
    let l = (qx * qx + qy * qy).sqrt();
    if l == 0.0 {
        return [0.0, 1.0, 0.0];
    }
    if rho == 0.0 {
        // On the symmetry axis the planar direction is undefined.
        return [0.0, qy.signum(), 0.0];
    }
    let radial = (qx / l) / rho;
    [radial * point[0], qy / l, radial * point[2]]
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

/// Exact surface normal (unit gradient) of [`capsule`] at `point` for the
/// segment `a`-`b`: the unit vector from the nearest point on the capsule's
/// skeleton segment toward `point`.
///
/// The capsule SDF is `length(point - closest) - radius` where `closest` is
/// the clamped projection onto the segment, so its gradient is simply
/// `(point - closest) / |point - closest|` — pointing radially away from the
/// axis both inside and outside the tube. On the skeleton itself (and for a
/// degenerate zero-length segment with `point == a`) the direction is
/// undefined, so the zero vector is returned.
pub fn capsule_gradient(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let pa = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ba_len_sq = dot(ba, ba);
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
    let l = length(closest);
    if l == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [closest[0] / l, closest[1] / l, closest[2] / l]
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

/// Signed distance from `point` to a "Death Star": a large sphere of radius
/// `large_radius` centred at the origin with a smaller spherical bite of
/// radius `small_radius` carved out, the carving sphere centred at
/// `(bite_distance, 0, 0)` along the `+x` axis. The subtraction leaves a
/// crescent-shaped crater whose circular lip (the rim where the two spheres
/// intersect) is the sharpest feature.
///
/// This is Inigo Quilez's exact `sdDeathStar`. Working in the meridian
/// half-plane `(axial, radial) = (point.x, length(point.yz))`, the rim of the
/// crater sits at `(a, b)` where `a = (ra^2 - rb^2 + d^2) / (2 d)` is the
/// axial coordinate of the intersection circle and `b = sqrt(ra^2 - a^2)` its
/// radial coordinate. A single half-plane test selects whether the nearest
/// feature is that rim circle (points facing into the crater lip) or the body
/// of the solid, which is the large sphere intersected with the complement of
/// the biting sphere. Built from `sqrt`, `min`, `max`, and vector lengths, so
/// it stays transcendental-free.
///
/// The rim `sqrt` is guarded with `.max(0.0)`, so degenerate configurations
/// where the biting sphere no longer clips the body collapse to a plain sphere
/// boundary instead of producing `NaN`.
pub fn death_star(point: [f32; 3], large_radius: f32, small_radius: f32, bite_distance: f32) -> f32 {
    let ra = large_radius;
    let rb = small_radius;
    let d = bite_distance;
    // Axial/radial coordinates of the circular rim where the two spheres meet.
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = (ra * ra - a * a).max(0.0).sqrt();
    // Meridian query: axial distance along x, radial distance off the x axis.
    let p = [point[0], length2([point[1], point[2]])];
    if p[0] * b - p[1] * a > d * (b - p[1]).max(0.0) {
        // Facing the crater lip: nearest feature is the rim circle at (a, b).
        length2([p[0] - a, p[1] - b])
    } else {
        // Large sphere intersected with the complement of the biting sphere.
        (length2(p) - ra).max(-(length2([p[0] - d, p[1]]) - rb))
    }
}

/// Signed distance from `point` to a solid right circular cone with its apex
/// at the origin, opening downward along `-y`, with base `base_radius` at
/// height `-height` (so the cap disc lies in the plane `y = -height`). The
/// solid is bounded by the slanted lateral surface and the flat circular base.
///
/// This is Inigo Quilez's exact `sdCone`, reformulated to take the base radius
/// and height directly (`q = (base_radius, -height)` is the apex-to-rim edge in
/// the meridian plane) so no trigonometry is needed. In the meridian plane
/// `w = (length(point.xz), point.y)` the distance is the smaller of the squared
/// distances to the lateral edge segment (`a`) and to the base cap segment
/// (`b`), with the sign recovered from the two half-plane tests. Built from
/// `clamp`, `sign`, `min`, `max`, dot products, and a single `sqrt`, so it
/// stays transcendental-free.
pub fn cone_sdf(point: [f32; 3], base_radius: f32, height: f32) -> f32 {
    // Apex-to-base-rim edge in the meridian (radial, axial) plane.
    let q = [base_radius, -height];
    let w = [length2([point[0], point[2]]), point[1]];
    // Nearest point on the lateral edge segment (clamped projection).
    let t = ((w[0] * q[0] + w[1] * q[1]) / dot2_2(q)).clamp(0.0, 1.0);
    let a = [w[0] - q[0] * t, w[1] - q[1] * t];
    // Nearest point on the base cap segment (radial clamp, fixed axial).
    let u = (w[0] / q[0]).clamp(0.0, 1.0);
    let b = [w[0] - q[0] * u, w[1] - q[1]];
    let k = q[1].signum();
    let d = dot2_2(a).min(dot2_2(b));
    let sign = (k * (w[0] * q[1] - w[1] * q[0])).max(k * (w[1] - q[1]));
    d.max(0.0).sqrt() * sign.signum()
}

/// Unsigned distance from `point` to the infinite line through the origin with
/// direction `direction`. The direction need not be normalised; it must be
/// non-zero.
///
/// Computed as the length of the component of `point` perpendicular to
/// `direction`, i.e. `point` minus its projection onto the line. An infinite
/// line has no interior, so the result is always non-negative. Built from dot
/// products and a single `sqrt`, so it stays transcendental-free.
pub fn line_sdf(point: [f32; 3], direction: [f32; 3]) -> f32 {
    let dd = dot(direction, direction);
    let t = dot(point, direction) / dd;
    length([
        point[0] - direction[0] * t,
        point[1] - direction[1] * t,
        point[2] - direction[2] * t,
    ])
}

/// Signed distance from `point` to a capped cylinder aligned with the `y`
/// axis whose vertical edges are rounded. The lateral surface sits at radius
/// `outer_radius` from the axis, the flat top and bottom caps lie at
/// `y = +-half_height`, and the circular edge joining cap to side is filleted
/// with radius `rounding`. For a meaningful solid, `rounding` must not exceed
/// either `outer_radius` or `half_height`.
///
/// This is Inigo Quilez's exact `sdRoundedCylinder`, reparameterised so the
/// arguments describe the outer silhouette directly (the raw formula insets
/// the rectangular profile by `rounding` on both axes before the rounded-box
/// combine). In the meridian plane the problem reduces to a rounded rectangle
/// of half-extents `(outer_radius - rounding, half_height - rounding)` offset
/// outward by `rounding`. Built from `abs`, `min`, `max`, and vector lengths,
/// so it stays transcendental-free.
pub fn rounded_cylinder(point: [f32; 3], outer_radius: f32, rounding: f32, half_height: f32) -> f32 {
    // Meridian rounded-rectangle offsets: radial reach and vertical reach of
    // the inset profile, measured against the fillet radius.
    let dx = length2([point[0], point[2]]) - (outer_radius - rounding);
    let dy = point[1].abs() - (half_height - rounding);
    dx.max(dy).min(0.0) + length2([dx.max(0.0), dy.max(0.0)]) - rounding
}

/// Signed distance from `point` to an infinite circular cylinder of the given
/// `radius` whose axis is parallel to the `y` axis and passes through
/// `(axis_xz[0], axis_xz[1])` in the `xz` plane.
///
/// This is Inigo Quilez's exact `sdInfiniteCylinder`: the distance is purely a
/// function of the radial offset in the `xz` plane, independent of `y`.
/// Negative inside the cylinder, positive outside. Built from a single vector
/// length, so it stays transcendental-free.
pub fn infinite_cylinder(point: [f32; 3], axis_xz: [f32; 2], radius: f32) -> f32 {
    length2([point[0] - axis_xz[0], point[2] - axis_xz[1]]) - radius
}

/// Unsigned distance from `point` to the triangle with vertices `a`, `b`, `c`
/// in arbitrary 3D orientation.
///
/// A triangle is a 2-manifold with no interior, so there is no "inside": the
/// distance is non-negative everywhere and zero exactly on the (closed)
/// triangular patch. This is the exact Euclidean distance to the nearest point
/// of the triangle, the atom from which arbitrary triangle *soups* and thin
/// planar features are assembled before the mesh baker takes over.
///
/// This is Inigo Quilez's exact `udTriangle`: the three edge normals (each the
/// cross of an edge with the face normal `nor = cross(b-a, a-c)`) partition
/// space into the region whose nearest feature is the triangle *interior* (the
/// perpendicular foot lands inside all three edges, so `sign` of the three edge
/// tests sums to the in-face value) versus the region governed by an *edge or
/// vertex*. In the face region the distance is the perpendicular projection
/// onto the plane; otherwise it is the minimum over the three edges of the
/// distance to the clamped foot of the perpendicular, which collapses to a
/// vertex when the clamp saturates. Built from `dot`, `cross`, `clamp`,
/// `sign`, `min`, and a single final `sqrt`, so it stays transcendental-free.
///
/// Degenerate (collinear / zero-area) triangles make `nor` vanish; the face
/// branch divides by `dot2(nor)`, so callers must pass a non-degenerate
/// triangle for a meaningful face-region distance (the edge branch stays
/// well-defined regardless).
pub fn triangle_sdf(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let cb = sub3(c, b);
    let pb = sub3(point, b);
    let ac = sub3(a, c);
    let pc = sub3(point, c);
    let nor = cross(ba, ac);

    // Edge-region test: each term is +-1 depending on which side of the edge's
    // in-plane normal the query falls. A sum below 2 means the perpendicular
    // foot escapes the triangle through at least one edge, so an edge/vertex
    // governs; otherwise the face interior governs.
    let edge_region = dot(cross(ba, nor), pa).signum()
        + dot(cross(cb, nor), pb).signum()
        + dot(cross(ac, nor), pc).signum()
        < 2.0;

    let squared = if edge_region {
        let e0 = {
            let t = (dot(ba, pa) / dot2_3(ba)).clamp(0.0, 1.0);
            dot2_3(sub3([ba[0] * t, ba[1] * t, ba[2] * t], pa))
        };
        let e1 = {
            let t = (dot(cb, pb) / dot2_3(cb)).clamp(0.0, 1.0);
            dot2_3(sub3([cb[0] * t, cb[1] * t, cb[2] * t], pb))
        };
        let e2 = {
            let t = (dot(ac, pc) / dot2_3(ac)).clamp(0.0, 1.0);
            dot2_3(sub3([ac[0] * t, ac[1] * t, ac[2] * t], pc))
        };
        e0.min(e1).min(e2)
    } else {
        let np = dot(nor, pa);
        np * np / dot2_3(nor)
    };
    squared.sqrt()
}

/// Unsigned distance from `point` to the planar convex quadrilateral with
/// vertices `a`, `b`, `c`, `d` given in winding order.
///
/// The four-sided analogue of [`triangle_sdf`] and the natural atom for walls,
/// panels, blade cards, and any flat quad patch before the mesh baker takes
/// over. Like a triangle a quad is a 2-manifold with no interior, so the
/// distance is non-negative everywhere and zero exactly on the (closed) patch.
///
/// This is Inigo Quilez's exact `udQuad`: the four edge normals (each the cross
/// of an edge with the face normal `nor = cross(b-a, a-d)`) classify the query
/// into the face-interior region versus an edge/vertex region. When the four
/// edge-sign tests sum below `3` the perpendicular foot escapes through an
/// edge, so the result is the minimum over the four edges of the distance to
/// the clamped foot (collapsing to a vertex when the clamp saturates);
/// otherwise it is the perpendicular projection onto the plane. Built from
/// `dot`, `cross`, `clamp`, `sign`, `min`, and a single final `sqrt`, so it
/// stays transcendental-free.
///
/// For an exact distance the four vertices must be **coplanar and wound convex**
/// (consistent order, no self-intersection); the face branch divides by
/// `dot2(nor)`, so a degenerate / collinear quad has no meaningful face-region
/// distance (the edge branch stays well-defined regardless).
pub fn quad_sdf(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let cb = sub3(c, b);
    let pb = sub3(point, b);
    let dc = sub3(d, c);
    let pc = sub3(point, c);
    let ad = sub3(a, d);
    let pd = sub3(point, d);
    let nor = cross(ba, ad);

    let edge_region = dot(cross(ba, nor), pa).signum()
        + dot(cross(cb, nor), pb).signum()
        + dot(cross(dc, nor), pc).signum()
        + dot(cross(ad, nor), pd).signum()
        < 3.0;

    let squared = if edge_region {
        let e0 = {
            let t = (dot(ba, pa) / dot2_3(ba)).clamp(0.0, 1.0);
            dot2_3(sub3([ba[0] * t, ba[1] * t, ba[2] * t], pa))
        };
        let e1 = {
            let t = (dot(cb, pb) / dot2_3(cb)).clamp(0.0, 1.0);
            dot2_3(sub3([cb[0] * t, cb[1] * t, cb[2] * t], pb))
        };
        let e2 = {
            let t = (dot(dc, pc) / dot2_3(dc)).clamp(0.0, 1.0);
            dot2_3(sub3([dc[0] * t, dc[1] * t, dc[2] * t], pc))
        };
        let e3 = {
            let t = (dot(ad, pd) / dot2_3(ad)).clamp(0.0, 1.0);
            dot2_3(sub3([ad[0] * t, ad[1] * t, ad[2] * t], pd))
        };
        e0.min(e1).min(e2).min(e3)
    } else {
        let np = dot(nor, pa);
        np * np / dot2_3(nor)
    };
    squared.sqrt()
}

/// Signed distance from `point` to a capped cone (conical frustum) with
/// arbitrary endpoints: a circular cap of radius `ra` centred at `a` and a
/// circular cap of radius `rb` centred at `b`, with a straight lateral surface
/// between them.
///
/// This is the general-orientation, two-radius generalisation of
/// [`capped_cylinder`] and [`capped_cone`] (which are axis-aligned and share a
/// base at the origin): it places both caps anywhere in space, so it directly
/// models tapered limbs, bolts, nozzles, and trunk segments without a wrapping
/// transform. With `ra == rb` it degenerates to a capped cylinder along `a->b`.
///
/// This is Inigo Quilez's exact `sdCappedCone(p, a, b, ra, rb)`: the query is
/// reduced to the axial coordinate `paba` (fractional position along `a->b`)
/// and the radial distance `x` from the axis, then the nearest feature is the
/// smaller of the distance to a cap rim (`cax`/`cay`) and the distance to the
/// slanted lateral line (`cbx`/`cby`), with an interior sign test. Built from
/// `dot`, `clamp`, `abs`, `min`/`max`, and two square roots, so it stays
/// transcendental-free and yields a true distance (not a bound) inside and out.
///
/// Degenerate inputs (`a == b`) make `baba` vanish and the axial projection
/// divide by zero; callers must pass distinct endpoints.
pub fn capped_cone_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], ra: f32, rb: f32) -> f32 {
    let rba = rb - ra;
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let baba = dot(ba, ba);
    let papa = dot(pa, pa);
    // Fractional axial coordinate of the query's projection onto `a->b`.
    let paba = dot(pa, ba) / baba;
    // Perpendicular (radial) distance from the axis, guarded against tiny
    // negative round-off before the square root.
    let x = (papa - paba * paba * baba).max(0.0).sqrt();

    // Nearest cap-rim feature: radial overshoot past whichever cap governs,
    // paired with the axial overshoot past the `[0, 1]` segment.
    let cax = (x - if paba < 0.5 { ra } else { rb }).max(0.0);
    let cay = (paba - 0.5).abs() - 0.5;

    // Nearest lateral-line feature: foot of the perpendicular onto the slanted
    // side, clamped to the frustum's extent.
    let k = rba * rba + baba;
    let f = ((rba * (x - ra) + paba * baba) / k).clamp(0.0, 1.0);
    let cbx = x - ra - f * rba;
    let cby = paba - f;

    // Interior when both the lateral and axial residuals are negative.
    let sign = if cbx < 0.0 && cay < 0.0 { -1.0 } else { 1.0 };
    sign * (cax * cax + cay * cay * baba)
        .min(cbx * cbx + cby * cby * baba)
        .sqrt()
}

/// Signed distance from `point` to a round cone with arbitrary endpoints: the
/// convex hull of a sphere of radius `r1` centred at `a` and a sphere of radius
/// `r2` centred at `b` — a tapered capsule between two arbitrary points.
///
/// This is the general-orientation generalisation of [`round_cone_sdf`] (which
/// is pinned to the `+y` axis with its lower sphere at the origin): both end
/// spheres may sit anywhere, so it models rounded tapered limbs, tentacles, and
/// claws directly. With `r1 == r2` it is a [`capsule`]; with the two spheres
/// touching it is a sphere.
///
/// This is Inigo Quilez's exact arbitrary-endpoint `sdRoundCone(p, a, b, r1,
/// r2)`: the query is split into axial (`y`), beyond-far-cap (`z = y - l2`),
/// and squared-radial (`x2`) components in units scaled by `l2 = dot(b-a,b-a)`,
/// and a single comparison against the slope term `k` selects whether the near
/// sphere, the far sphere, or the exact tangent flank governs. Built from
/// `dot`, `sign`, `min`/`max`, and square roots, so it stays transcendental-free
/// and yields a true signed distance (not a bound) inside and out.
///
/// Degenerate inputs (`a == b`) make `l2` vanish and the `1/l2` scaling divide
/// by zero; callers must pass distinct endpoints (use [`sphere`] for a point).
pub fn round_cone_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], r1: f32, r2: f32) -> f32 {
    let ba = sub3(b, a);
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;

    let pa = sub3(point, a);
    let y = dot(pa, ba);
    let z = y - l2;
    // Squared perpendicular component: `dot2(pa*l2 - ba*y)`.
    let perp = [
        pa[0] * l2 - ba[0] * y,
        pa[1] * l2 - ba[1] * y,
        pa[2] * l2 - ba[2] * y,
    ];
    let x2 = dot2_3(perp);
    let y2 = y * y * l2;
    let z2 = z * z * l2;

    // Slope threshold selecting the governing feature.
    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        // Beyond the far cap: the `b` sphere governs.
        (x2 + z2).sqrt() * il2 - r2
    } else if y.signum() * a2 * y2 < k {
        // Before the near cap: the `a` sphere governs.
        (x2 + y2).sqrt() * il2 - r1
    } else {
        // Along the tangent flank between the caps.
        (x2 * a2 * il2).sqrt() * il2 + y * rr * il2 - r1
    }
}

/// Signed distance from `point` to a capped cylinder with arbitrary endpoints:
/// a solid cylinder of the given `radius` whose axis runs from `a` to `b`,
/// closed by a flat cap at each end.
///
/// This is the general-orientation generalisation of [`capped_cylinder`]
/// (which is pinned to the `y` axis and centred at the origin): the axis may
/// point anywhere, so it models pipes, bars, and bones placed directly in world
/// space without a separate transform. Passing `a = [0, -h, 0]`, `b = [0, h, 0]`
/// reproduces [`capped_cylinder`] exactly.
///
/// This is Inigo Quilez's exact `sdCylinder(p, a, b, r)`: the query is split
/// into a radial residual `x` (distance from the axis minus the radius) and an
/// axial residual `y` (overshoot past the end caps), both expressed in units
/// scaled by `baba = dot(b - a, b - a)` to avoid a divide until the final
/// `sqrt`. The interior case takes the negative of the nearer squared residual;
/// the exterior case sums the positive residuals (corner-correct where the rim
/// meets a cap). Built from `dot`, `abs`, `min`/`max`, `sign`, and one square
/// root, so it stays transcendental-free and yields a true distance (not a
/// bound) inside and out.
///
/// Degenerate inputs (`a == b`) make `baba` vanish and the final `1/baba`
/// scaling divide by zero; callers must pass distinct endpoints.
pub fn cylinder_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], radius: f32) -> f32 {
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);

    // Radial residual: perpendicular distance from the (infinite) axis, scaled
    // by `baba`, minus the radius (also scaled). `pa*baba - ba*paba` is the
    // component of `pa` orthogonal to `ba`, times `baba`.
    let perp = [
        pa[0] * baba - ba[0] * paba,
        pa[1] * baba - ba[1] * paba,
        pa[2] * baba - ba[2] * paba,
    ];
    let x = length(perp) - radius * baba;
    // Axial residual: overshoot past either flat cap, scaled by `baba`.
    let y = (paba - baba * 0.5).abs() - baba * 0.5;

    let x2 = x * x;
    let y2 = y * y * baba;

    // Interior when both residuals are negative: distance is the negated nearer
    // squared residual. Exterior: sum the squared positive residuals so the
    // rim/cap corner stays Euclidean-correct.
    let d = if x.max(y) < 0.0 {
        -(x2.min(y2))
    } else {
        (if x > 0.0 { x2 } else { 0.0 }) + (if y > 0.0 { y2 } else { 0.0 })
    };

    d.signum() * d.abs().sqrt() / baba
}

/// Signed distance from `point` to a regular octagonal prism: a regular octagon
/// of inradius (apothem) `radius` in the `xy` plane, extruded along the `z`
/// axis to span `[-half_depth, half_depth]`, centred at the origin.
///
/// This is the eight-sided companion of [`hex_prism`] and shares its exact
/// construction: the cross-section is folded through its mirror planes so a
/// single wedge represents the whole polygon, the nearest flat is measured with
/// the interior/exterior split, and the result is combined with the `z` slab so
/// the side faces, end caps, and their shared edges all carry a true distance
/// (not a bound). It models the octagonal stock AAA content uses for bolt
/// heads, nuts, columns, and chamfered posts.
///
/// Follows Inigo Quilez's exact `sdOctogonPrism`. The baked constants are
/// `(-cos(pi/8), sin(pi/8), tan(pi/8))`, so the function stays
/// transcendental-free.
pub fn octagon_prism(point: [f32; 3], radius: f32, half_depth: f32) -> f32 {
    // (-cos(22.5 degrees), sin(22.5 degrees), tan(22.5 degrees) = sqrt(2) - 1).
    const K: [f32; 3] = [-0.923_879_5, 0.382_683_4, 0.414_213_56];
    let mut p = [point[0].abs(), point[1].abs(), point[2].abs()];

    // Two reflections fold the absolute-value quadrant down to a single octant
    // wedge: one against the ( K.x,  K.y) plane, one against the (-K.x, K.y)
    // plane. After both, the governing feature is always the top flat edge.
    let fold0 = 2.0 * (K[0] * p[0] + K[1] * p[1]).min(0.0);
    p[0] -= fold0 * K[0];
    p[1] -= fold0 * K[1];
    let fold1 = 2.0 * (-K[0] * p[0] + K[1] * p[1]).min(0.0);
    p[0] -= fold1 * -K[0];
    p[1] -= fold1 * K[1];

    // Slide onto the top flat (half-length tan(pi/8) * radius) and drop to it.
    let clamped_x = p[0].clamp(-K[2] * radius, K[2] * radius);
    let face = [p[0] - clamped_x, p[1] - radius];
    let sign = if p[1] - radius < 0.0 { -1.0 } else { 1.0 };
    let d = [length2(face) * sign, p[2] - half_depth];
    let inside = d[0].max(d[1]).min(0.0);
    let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
    inside + outside
}

/// Signed distance from `point` to an *infinite* circular cone: the unbounded
/// solid whose apex sits at the origin and whose surface opens along the `+y`
/// axis with half-angle given by `sin_cos = [sin(angle), cos(angle)]`.
///
/// Where [`cone_sdf`] is a finite cone with a base cap and [`solid_angle`] is a
/// cone sector clipped by a bounding sphere, this cone extends forever: it is
/// the exact primitive for carving conical bevels, spotlight volumes, and
/// chamfers via the CSG operators. Pre-baking the aperture as `[sin, cos]`
/// (as [`solid_angle`] does) keeps the function transcendental-free.
///
/// This is Inigo Quilez's exact infinite `sdCone`. In the meridian plane
/// `q = (length(point.xz), point.y)` the flank is the ray from the origin along
/// `c = sin_cos`; the distance is the length of `q` minus its (non-negative)
/// projection onto `c`, which collapses to the apex distance behind the cone,
/// and the sign flips inside the solid. Built from `clamp`/`max`, dot products,
/// and a single `sqrt`, so it stays transcendental-free.
pub fn infinite_cone(point: [f32; 3], sin_cos: [f32; 2]) -> f32 {
    let q = [length2([point[0], point[2]]), point[1]];
    // Project onto the flank ray, clamped at the apex (no negative extent).
    let proj = (q[0] * sin_cos[0] + q[1] * sin_cos[1]).max(0.0);
    let w = [q[0] - sin_cos[0] * proj, q[1] - sin_cos[1] * proj];
    let d = length2(w);
    // Inside the solid cone when the point sits on the axis side of the flank.
    if sin_cos[1] * q[0] - sin_cos[0] * q[1] < 0.0 {
        -d
    } else {
        d
    }
}

/// Signed distance to a two-dimensional circular sector ("pie slice") of
/// radius `radius`, centred on the `+y` axis and opening symmetrically with
/// half-aperture given by `sin_cos = [sin(angle), cos(angle)]`.
///
/// This is Inigo Quilez's exact `sdPie`. Folding `x` to its magnitude exploits
/// the mirror symmetry so only one straight edge needs solving: `l` is the
/// distance to the bounding circle, `m` the distance to that edge (projecting
/// onto the edge ray with the parameter clamped to `[0, radius]`), and the two
/// combine with `max`, the edge term signed by the half-plane test
/// `sin*|x| - cos*y`. Pair it with the `extrude` domain operator to build a
/// wedge prism, or with a radial offset for a lathed ring segment. Uses only
/// `abs`, `clamp`, `min`/`max`, `sign` and `sqrt`, so it stays
/// transcendental-free.
pub fn pie(point: [f32; 2], sin_cos: [f32; 2], radius: f32) -> f32 {
    let px = point[0].abs();
    let p = [px, point[1]];
    let l = length2(p) - radius;
    // Distance to the straight edge ray along `sin_cos`, clamped to the radius.
    let t = (p[0] * sin_cos[0] + p[1] * sin_cos[1]).clamp(0.0, radius);
    let m = length2([p[0] - sin_cos[0] * t, p[1] - sin_cos[1] * t]);
    let edge_sign = (sin_cos[1] * p[0] - sin_cos[0] * p[1]).signum();
    l.max(m * edge_sign)
}

/// Signed distance to a two-dimensional crescent ("moon"): the disk of radius
/// `ra` centred at the origin with a disk of radius `rb` subtracted, the hole
/// centred `d` units along `+x`.
///
/// This is Inigo Quilez's exact `sdMoon`. Folding `y` to its magnitude exploits
/// the crescent's mirror symmetry. The intersection of the two circles gives
/// the cusp `(a, b)`; when the query point projects beyond that cusp the exact
/// distance is to the cusp tip itself, otherwise it is the constructive-solid
/// difference `max(|p| - ra, -(|p - (d,0)| - rb))`. Handling the cusp
/// explicitly is what keeps the field exact where a naive CSG `max` would
/// overshoot. Uses only `abs`, `max`, `sqrt` and dot products, so it stays
/// transcendental-free. Requires an overlapping configuration
/// (`|ra - rb| < d < ra + rb`).
pub fn moon(point: [f32; 2], d: f32, ra: f32, rb: f32) -> f32 {
    let p = [point[0], point[1].abs()];
    // Intersection point (a, b) of the two circle boundaries (b >= 0).
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = (ra * ra - a * a).max(0.0).sqrt();
    // Beyond the cusp the nearest feature is the cusp tip itself.
    if d * (p[0] * b - p[1] * a) > d * d * (b - p[1]).max(0.0) {
        return length2([p[0] - a, p[1] - b]);
    }
    // Otherwise the difference of the two disks is exact.
    let outer = length2(p) - ra;
    let inner = length2([p[0] - d, p[1]]) - rb;
    outer.max(-inner)
}

/// Signed distance to a two-dimensional rounded "X": two diagonal bars of
/// half-length `w / 2` crossing at the origin, each stroke rounded by radius
/// `r`.
///
/// This is Inigo Quilez's exact `sdRoundedX`. Folding to the first quadrant via
/// `abs` collapses the four-fold symmetry; the point is then projected onto the
/// diagonal skeleton segment `y = x` (parameter `min(x + y, w) / 2`, clamped to
/// the arm length) and offset by the stroke radius. The result is an exact,
/// everywhere-smooth field built only from `abs`, `min` and a single `sqrt`, so
/// it stays transcendental-free. Pair it with the `extrude` domain operator to
/// cut a rounded cross through a slab.
pub fn rounded_x(point: [f32; 2], w: f32, r: f32) -> f32 {
    let p = [point[0].abs(), point[1].abs()];
    let m = (p[0] + p[1]).min(w) * 0.5;
    length2([p[0] - m, p[1] - m]) - r
}

/// Signed distance to a two-dimensional rounded cross (plus sign): two bars of
/// half-length `arm` and half-`thickness` crossing at the origin, every corner
/// rounded by radius `r`. Requires `arm >= thickness`.
///
/// This is Inigo Quilez's exact `sdCross`. Folding the point into the octant
/// `x >= y >= 0` collapses the plus to a single box there, so the exterior is a
/// plain box distance. The interior is the subtle part: the nearest boundary of
/// a point in the central square is the *reentrant* corner, which a naive union
/// of two box fields misses; the `w = (thickness - x, -k)` branch measures that
/// corner distance exactly. Rounding then insets the whole field by `r`. Built
/// from `abs`, `min`/`max`, `sign` and a single `sqrt`, so it stays
/// transcendental-free.
pub fn cross_2d(point: [f32; 2], arm: f32, thickness: f32, r: f32) -> f32 {
    // Fold into the octant x >= y >= 0; the plus reduces to one box there.
    let mut p = [point[0].abs(), point[1].abs()];
    if p[1] > p[0] {
        p = [p[1], p[0]];
    }
    let q = [p[0] - arm, p[1] - thickness];
    let k = q[0].max(q[1]);
    // Outside the box uses the ordinary box corner distance; inside, measure to
    // the reentrant corner via (thickness - x, -k).
    let w = if k > 0.0 { q } else { [thickness - p[0], -k] };
    k.signum() * length2([w[0].max(0.0), w[1].max(0.0)]) - r
}

/// Signed distance to a two-dimensional *rounded cross* of vertical reach
/// `h`: a four-armed cross whose horizontal arms reach `x = +-1` and whose
/// vertical arms reach `y = +-h`, with the four re-entrant junctions between
/// the arms joined by circular fillets of radius `k = (h + 1/h) / 2`. For a
/// well-formed shape `h` must be positive.
///
/// This is Inigo Quilez's exact `sdRoundedCross`. The point is folded into the
/// first quadrant by `abs`; inside the wedge below the line through the tip
/// `(0, h)` and the fillet centre `(1, k)` the nearest boundary is the fillet
/// arc, so the distance is `k - length(p - (1, k))` (negative inside the
/// solid, where the arc bulges away from its centre). Elsewhere the nearest
/// feature is one of the two convex tips `(1, 0)` or `(0, h)`, giving
/// `min(length(p - (0, h)), length(p - (1, 0)))`. The fillet centre lies at
/// distance `k` from both tips, so the two branches meet continuously. Built
/// from `abs`, `min`, a division and a single `sqrt`, so it stays
/// transcendental-free. Pair it with the `extrude` domain operator to turn the
/// profile into a 3D cross prism.
pub fn rounded_cross_2d(point: [f32; 2], h: f32) -> f32 {
    let k = 0.5 * (h + 1.0 / h);
    let p = [point[0].abs(), point[1].abs()];
    if p[0] < 1.0 && p[1] < p[0] * (k - h) + h {
        k - length2([p[0] - 1.0, p[1] - k])
    } else {
        length2([p[0], p[1] - h]).min(length2([p[0] - 1.0, p[1]]))
    }
}

/// Signed distance to a two-dimensional *horseshoe* (an open omega/U ring): a
/// circular band of centreline `radius` and half-`thickness`, cut open by a
/// wedge of half-angle `theta` (passed pre-baked as `sin_cos = [sin(theta),
/// cos(theta)]`) whose two free ends are extended by straight `arm`-long
/// prongs capped flat. The opening faces `+y`; the bend sits at `-y`.
///
/// This is Inigo Quilez's exact `sdHorseshoe`. The query is folded across the
/// `y` axis (`abs(x)`) to exploit the mirror symmetry, then rotated by the
/// opening half-angle; the radial magnitude `l = |p|` is carried through so a
/// piecewise `select` can stitch the circular bend onto the two straight
/// prongs (matching the GLSL `vec2` constructor, both lanes read the rotated
/// pair simultaneously). What remains is the distance to a half-infinite
/// rounded strip in the rotated frame: offset by the prong length and ring
/// thickness, `b = (q.x - arm, |q.y - radius| - thickness)`, closed by the
/// standard box field `length(max(b, 0)) + min(0, max(b.x, b.y))`. Built from
/// `abs`, `min`/`max`, `sign`, dot products and a single `sqrt`, so it stays
/// transcendental-free. Pair it with the `extrude` domain operator to raise a
/// horseshoe prism, or with `revolution` for a toroidal clip.
pub fn horseshoe_2d(
    point: [f32; 2],
    sin_cos: [f32; 2],
    radius: f32,
    arm: f32,
    thickness: f32,
) -> f32 {
    // This crate bakes angles as `[sin, cos]`; IQ's `c` is `(cos, sin)`.
    let (c_cos, c_sin) = (sin_cos[1], sin_cos[0]);
    let px = point[0].abs();
    let l = length2([px, point[1]]);
    // Rotate (|x|, y) by mat2(-cos, sin; sin, cos).
    let rx = -c_cos * px + c_sin * point[1];
    let ry = c_sin * px + c_cos * point[1];
    // Piecewise select: outside the arc keep the rotated pair; inside the bend
    // fall back to the radial magnitude so the prongs join the ring smoothly.
    let qx = if ry > 0.0 || rx > 0.0 { rx } else { l * (-c_cos).signum() };
    let qy = if rx > 0.0 { ry } else { l };
    // Half-infinite rounded strip offset by prong length and ring thickness.
    let bx = qx - arm;
    let by = (qy - radius).abs() - thickness;
    length2([bx.max(0.0), by.max(0.0)]) + bx.max(by).min(0.0)
}

/// Unsigned distance from `point` to the line segment `a`-`b` in the plane.
///
/// The point is projected onto the segment with the parameter clamped to
/// `[0, 1]`, so endpoints are handled exactly: beyond an end the distance is to
/// that endpoint, otherwise it is the perpendicular drop. This is the planar
/// companion to the 3D `capsule` skeleton. Built from a dot product, a clamp
/// and one `sqrt`, so it stays transcendental-free.
pub fn segment_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let pa = [point[0] - a[0], point[1] - a[1]];
    let ba = [b[0] - a[0], b[1] - a[1]];
    let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1])).clamp(0.0, 1.0);
    length2([pa[0] - ba[0] * h, pa[1] - ba[1] * h])
}

/// Signed distance to a two-dimensional oriented box: the rectangle whose two
/// short ends are centred at `a` and `b` with full width `thickness`.
///
/// This is Inigo Quilez's exact `sdOrientedBox`. The query is translated to the
/// box centre and rotated into the box's local frame by projecting onto the
/// unit axis `d = (b - a)/|b - a|` and its perpendicular, after which it is the
/// ordinary axis-aligned box distance with half-extents `(|b - a|/2,
/// thickness/2)`. Built from `abs`, `min`/`max` and `sqrt`, so it stays
/// transcendental-free. Pair it with `extrude` for a slanted bar prism.
pub fn oriented_box_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2], thickness: f32) -> f32 {
    let l = length2([b[0] - a[0], b[1] - a[1]]);
    let d = [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
    let c = [point[0] - 0.5 * (a[0] + b[0]), point[1] - 0.5 * (a[1] + b[1])];
    // Rotate into the box frame: local x along `d`, local y along the normal.
    let q0 = (d[0] * c[0] + d[1] * c[1]).abs() - 0.5 * l;
    let q1 = (-d[1] * c[0] + d[0] * c[1]).abs() - 0.5 * thickness;
    length2([q0.max(0.0), q1.max(0.0)]) + q0.max(q1).min(0.0)
}

/// Signed distance to a two-dimensional parallelogram centred at the origin
/// with half-width `half_width`, half-height `half_height` and horizontal
/// `skew` (the top edge is shifted `+skew` relative to the bottom edge).
///
/// This is Inigo Quilez's exact `sdParallelogram`. The lower half is folded onto
/// the upper half, then the field is the smaller of the distances to the
/// horizontal edge (parameter clamped to the half-width) and to the slanted
/// edge (projection onto the skew vector clamped to the side). The interior sign
/// is recovered from the signed areas accumulated in the `y` channel of `d`.
/// Built from `abs`, `clamp`, `min` and `sqrt`, so it stays transcendental-free.
pub fn parallelogram(point: [f32; 2], half_width: f32, half_height: f32, skew: f32) -> f32 {
    let e = [skew, half_height];
    let mut p = if point[1] < 0.0 { [-point[0], -point[1]] } else { point };
    // Distance to the horizontal (top) edge.
    let mut w = [p[0] - e[0], p[1] - e[1]];
    w[0] -= w[0].clamp(-half_width, half_width);
    let mut d = [w[0] * w[0] + w[1] * w[1], -w[1]];
    // Signed area selects the near slanted edge; fold again across it.
    let s = p[0] * e[1] - p[1] * e[0];
    if s < 0.0 {
        p = [-p[0], -p[1]];
    }
    let v = [p[0] - half_width, p[1]];
    let g = ((v[0] * e[0] + v[1] * e[1]) / (e[0] * e[0] + e[1] * e[1])).clamp(-1.0, 1.0);
    let v = [v[0] - e[0] * g, v[1] - e[1] * g];
    d = [
        d[0].min(v[0] * v[0] + v[1] * v[1]),
        d[1].min(half_width * half_height - s.abs()),
    ];
    d[0].sqrt() * (-d[1]).signum()
}

/// Signed distance to a two-dimensional rhombus centred at the origin with
/// half-diagonals `half_diag = [bx, by]` (vertices at `(+/-bx, 0)` and
/// `(0, +/-by)`).
///
/// This is Inigo Quilez's exact `sdRhombus`. Folding to the first quadrant via
/// `abs` collapses the four-fold symmetry; the nearest point on the slanted edge
/// is found with the parameter `h` built from `ndot` (`a.x*b.x - a.y*b.y`), and
/// the interior sign comes from the edge half-plane test. Built from `abs`,
/// `clamp`, `sign` and `sqrt`, so it stays transcendental-free.
pub fn rhombus_2d(point: [f32; 2], half_diag: [f32; 2]) -> f32 {
    let b = half_diag;
    let p = [point[0].abs(), point[1].abs()];
    // ndot(b - 2p, b) / dot(b, b), clamped to the edge span.
    let nd = (b[0] - 2.0 * p[0]) * b[0] - (b[1] - 2.0 * p[1]) * b[1];
    let h = (nd / (b[0] * b[0] + b[1] * b[1])).clamp(-1.0, 1.0);
    let q = [p[0] - 0.5 * b[0] * (1.0 - h), p[1] - 0.5 * b[1] * (1.0 + h)];
    let d = length2(q);
    d * (p[0] * b[1] + p[1] * b[0] - b[0] * b[1]).signum()
}

/// Signed distance to a two-dimensional isosceles trapezoid centred at the
/// origin: bottom half-width `bottom_half` (at `y = -half_height`), top
/// half-width `top_half` (at `y = +half_height`).
///
/// This is Inigo Quilez's exact `sdTrapezoid`. After folding `x` to its
/// magnitude the field is the smaller of two candidate distances: `ca` to the
/// capped horizontal edges and `cb` to the slanted side (projection onto the
/// side direction `k2` clamped to the segment). The interior sign is set when
/// the point lies left of the slanted edge and below the top. Built from `abs`,
/// `clamp`, `min` and `sqrt`, so it stays transcendental-free.
pub fn trapezoid_isosceles(point: [f32; 2], bottom_half: f32, top_half: f32, half_height: f32) -> f32 {
    let (r1, r2, he) = (bottom_half, top_half, half_height);
    let k1 = [r2, he];
    let k2 = [r2 - r1, 2.0 * he];
    let p = [point[0].abs(), point[1]];
    let edge = if p[1] < 0.0 { r1 } else { r2 };
    let ca = [p[0] - p[0].min(edge), p[1].abs() - he];
    let t = (((k1[0] - p[0]) * k2[0] + (k1[1] - p[1]) * k2[1]) / (k2[0] * k2[0] + k2[1] * k2[1]))
        .clamp(0.0, 1.0);
    let cb = [p[0] - k1[0] + k2[0] * t, p[1] - k1[1] + k2[1] * t];
    let s = if cb[0] < 0.0 && ca[1] < 0.0 { -1.0 } else { 1.0 };
    s * (ca[0] * ca[0] + ca[1] * ca[1]).min(cb[0] * cb[0] + cb[1] * cb[1]).sqrt()
}

/// Signed distance to a two-dimensional circular arc band: a stroke of
/// thickness `thickness` wrapped around the circle of `radius`, centred on the
/// `+y` axis and spanning the half-aperture given by
/// `sin_cos = [sin(angle), cos(angle)]`.
///
/// This is Inigo Quilez's exact `sdArc`. Folding `x` to its magnitude exploits
/// the arc's mirror symmetry. Inside the aperture the distance is to the circle
/// (`||p| - radius|`); outside it the nearest feature is the arc endpoint
/// `sin_cos * radius`. Subtracting `thickness` turns the skeleton into a solid
/// band. Built from `abs`, dot products and `sqrt`, so it stays
/// transcendental-free. Pair it with `extrude` to build a curved wall section.
pub fn arc(point: [f32; 2], sin_cos: [f32; 2], radius: f32, thickness: f32) -> f32 {
    let p = [point[0].abs(), point[1]];
    let skeleton = if sin_cos[1] * p[0] > sin_cos[0] * p[1] {
        // Beyond the aperture: distance to the arc endpoint.
        length2([p[0] - sin_cos[0] * radius, p[1] - sin_cos[1] * radius])
    } else {
        // Within the aperture: distance to the circle itself.
        (length2(p) - radius).abs()
    };
    skeleton - thickness
}

/// Signed distance to a two-dimensional isosceles triangle with its apex at the
/// origin and a horizontal base of half-width `half_base` at `y = height`
/// (vertices `(0, 0)`, `(+/-half_base, height)`).
///
/// This is Inigo Quilez's exact `sdTriangleIsosceles`. After folding `x` to its
/// magnitude the field is the component-wise minimum of the distance to the
/// slanted edge (projection onto the apex-to-corner vector) and to the capped
/// base edge; the interior sign is recovered from the two edge half-plane
/// tests. Built from `abs`, `clamp`, `min`, `sign` and `sqrt`, so it stays
/// transcendental-free.
pub fn isosceles_triangle_2d(point: [f32; 2], half_base: f32, height: f32) -> f32 {
    let q = [half_base, height];
    let p = [point[0].abs(), point[1]];
    let t = ((p[0] * q[0] + p[1] * q[1]) / (q[0] * q[0] + q[1] * q[1])).clamp(0.0, 1.0);
    let a = [p[0] - q[0] * t, p[1] - q[1] * t];
    let tb = (p[0] / q[0]).clamp(0.0, 1.0);
    let b = [p[0] - q[0] * tb, p[1] - q[1]];
    let s = -q[1].signum();
    let dx = (a[0] * a[0] + a[1] * a[1]).min(b[0] * b[0] + b[1] * b[1]);
    let dy = (s * (p[0] * q[1] - p[1] * q[0])).min(s * (p[1] - q[1]));
    -dx.sqrt() * dy.signum()
}

/// Signed distance to a two-dimensional cut disk: the disk of `radius` with the
/// cap above the horizontal line `y = cut_height` sliced off, keeping the lower
/// body with a flat top edge (requires `-radius < cut_height < radius`).
///
/// This is Inigo Quilez's exact `sdCutDisk`, evaluated on the `y`-reflected
/// configuration so that the retained half-plane is `y <= cut_height`. The
/// precomputed chord half-width `w = sqrt(radius^2 - cut_height^2)` and the
/// discriminant `s` classify the query into three Voronoi regions: the circular
/// arc, the straight chord, and the two sharp chord corners. Returning the
/// matching closed form keeps the field exact where a naive disk/half-plane
/// intersection would round the corners. Built from `abs`, `min`/`max` and
/// `sqrt`, so it stays transcendental-free.
pub fn cut_disk_2d(point: [f32; 2], radius: f32, cut_height: f32) -> f32 {
    // Reflect in y so IQ's cap-above core yields the flat-top (keep y <= cut)
    // orientation: SDF_{y<=cut}(x, y) = core_{y>=-cut}(x, -y).
    let r = radius;
    let h = -cut_height;
    let w = (r * r - h * h).max(0.0).sqrt();
    let p = [point[0].abs(), -point[1]];
    let s = ((h - r) * p[0] * p[0] + w * w * (h + r - 2.0 * p[1])).max(h * p[0] - w * p[1]);
    if s < 0.0 {
        length2(p) - r
    } else if p[0] < w {
        h - p[1]
    } else {
        length2([p[0] - w, p[1] - h])
    }
}

/// Signed distance to a two-dimensional uneven capsule: the convex hull of a
/// disk of radius `r_bottom` centred at the origin and a disk of radius
/// `r_top` centred at `(0, h)` (a round cone / tapered stadium).
///
/// This is Inigo Quilez's exact `sdUnevenCapsule`. After folding `x` the slope
/// `b = (r_bottom - r_top)/h` and `a = sqrt(1 - b^2)` define the external
/// tangent; the parameter `k` selects the bottom cap, the top cap, or the
/// tangent flank, each with its own exact distance. Built from `abs` and
/// `sqrt`, so it stays transcendental-free. Requires `|r_bottom - r_top| <= h`.
pub fn uneven_capsule_2d(point: [f32; 2], r_bottom: f32, r_top: f32, h: f32) -> f32 {
    let p = [point[0].abs(), point[1]];
    let b = (r_bottom - r_top) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    let k = -b * p[0] + a * p[1];
    if k < 0.0 {
        length2(p) - r_bottom
    } else if k > a * h {
        length2([p[0], p[1] - h]) - r_top
    } else {
        a * p[0] + b * p[1] - r_bottom
    }
}

/// Signed distance to a two-dimensional regular hexagon centred at the origin
/// with inradius (apothem) `apothem`, oriented flat-side up (horizontal top and
/// bottom edges at `y = +/-apothem`).
///
/// This is Inigo Quilez's exact `sdHexagon`. The baked constant
/// `k = (-cos 30deg, sin 30deg, tan 30deg)` lets a single reflection fold the
/// query into one sextant, after which the shape reduces to a capped edge whose
/// signed distance is `length(p) * sign(p.y)`. The transcendental values live
/// only in the compile-time constants, so the runtime path uses just `abs`,
/// `clamp`, `min`, `sign` and `sqrt`.
pub fn regular_hexagon_2d(point: [f32; 2], apothem: f32) -> f32 {
    // k = (-cos(pi/6), sin(pi/6), tan(pi/6)) as compile-time constants.
    const KX: f32 = -0.866_025_4;
    const KY: f32 = 0.5;
    const KZ: f32 = 0.577_350_26;
    let mut p = [point[0].abs(), point[1].abs()];
    let fold = 2.0 * (KX * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - fold * KX, p[1] - fold * KY];
    p = [p[0] - p[0].clamp(-KZ * apothem, KZ * apothem), p[1] - apothem];
    length2(p) * p[1].signum()
}

/// Exact signed distance to a filled, apex-up equilateral triangle centred on
/// the origin (Inigo Quilez `sdEquilateralTriangle`).
///
/// `half_width` is the half-length of the horizontal base: the base vertices
/// sit at `(+/-half_width, -half_width/sqrt 3)` and the apex at
/// `(0, 2*half_width/sqrt 3)`, so the centroid lands on the origin and the
/// centre distance is `-half_width/sqrt 3`.
///
/// A reflection about `x = 0` plus one fold across the `k = sqrt 3` edge
/// collapses the query into one 60-degree wedge, after which the shape reduces
/// to a single clamped edge whose signed distance is `-length(p) * sign(p.y)`.
/// `sqrt 3` is the only transcendental and it lives in a compile-time
/// constant, so the runtime path uses only `abs`, `clamp`, `min`, `sign`
/// and `sqrt`.
pub fn equilateral_triangle_2d(point: [f32; 2], half_width: f32) -> f32 {
    // k = sqrt(3) as a compile-time constant.
    const K: f32 = 1.732_050_8;
    let r = half_width;
    let mut p = [point[0].abs() - r, point[1] + r / K];
    if p[0] + K * p[1] > 0.0 {
        p = [(p[0] - K * p[1]) * 0.5, (-K * p[0] - p[1]) * 0.5];
    }
    p[0] -= p[0].clamp(-2.0 * r, 0.0);
    -length2(p) * p[1].signum()
}

/// Exact signed distance to a filled, flat-top regular pentagon centred on the
/// origin (Inigo Quilez `sdPentagon`).
///
/// `apothem` is the perpendicular distance from the centre to each edge, so the
/// top edge lies on `y = apothem` and the centre distance is `-apothem`. The
/// circumradius is `apothem / cos(pi/5)`.
///
/// `p.x` is mirrored into the right half-plane and two reflections across the
/// upper-left and upper-right edges fold the query into the top sector, which
/// reduces to a single clamped edge with signed distance
/// `length(p) * sign(p.y)`. The pentagon's fixed interior trig values
/// (`cos 36deg`, `sin 36deg`, `tan 36deg`) are compile-time constants,
/// leaving the runtime path on `abs`, `clamp`, `min`, `sign` and `sqrt`.
pub fn regular_pentagon_2d(point: [f32; 2], apothem: f32) -> f32 {
    // k = (cos(pi/5), sin(pi/5), tan(pi/5)) as compile-time constants.
    const KX: f32 = 0.809_017;
    const KY: f32 = 0.587_785_25;
    const KZ: f32 = 0.726_542_5;
    let r = apothem;
    let mut p = [point[0].abs(), point[1]];
    let f1 = 2.0 * ((-KX) * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - f1 * (-KX), p[1] - f1 * KY];
    let f2 = 2.0 * (KX * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - f2 * KX, p[1] - f2 * KY];
    p = [p[0] - p[0].clamp(-r * KZ, r * KZ), p[1] - r];
    length2(p) * p[1].signum()
}

/// Exact signed distance to a filled, flat-top regular octagon centred on the
/// origin (Inigo Quilez `sdOctagon`).
///
/// `apothem` is the perpendicular distance from the centre to each edge, so the
/// top edge lies on `y = apothem`, the flats are axis-aligned and the centre
/// distance is `-apothem`. The circumradius is `apothem / cos(pi/8)`.
///
/// `p` is folded into the first quadrant and two reflections across the two
/// diagonal edges collapse the query into the top sector, leaving a single
/// clamped edge with signed distance `length(p) * sign(p.y)`. The fixed
/// interior trig values (`cos 22.5deg`, `sin 22.5deg`, `tan 22.5deg`) are
/// compile-time constants, so the runtime path uses only `abs`, `clamp`,
/// `min`, `sign` and `sqrt`.
pub fn regular_octagon_2d(point: [f32; 2], apothem: f32) -> f32 {
    // k = (-cos(pi/8), sin(pi/8), tan(pi/8)) as compile-time constants.
    const KX: f32 = -0.923_879_5;
    const KY: f32 = 0.382_683_43;
    const KZ: f32 = 0.414_213_56;
    let r = apothem;
    let mut p = [point[0].abs(), point[1].abs()];
    let f1 = 2.0 * (KX * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - f1 * KX, p[1] - f1 * KY];
    let f2 = 2.0 * ((-KX) * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - f2 * (-KX), p[1] - f2 * KY];
    p = [p[0] - p[0].clamp(-KZ * r, KZ * r), p[1] - r];
    length2(p) * p[1].signum()
}

/// Exact signed distance to a filled six-pointed hexagram (Star of David)
/// centred on the origin (Inigo Quilez `sdHexagram`).
///
/// `r` is the mid-scale parameter: the six outer tips sit at radius `2*r`
/// (angles 30deg + 60deg*k) and the six inner vertices at radius `2*r/sqrt 3`
/// (angles 60deg*k), so the centre distance is `-2*r/sqrt 3` (the inner
/// vertex is the nearest boundary feature to the centre).
///
/// `p` is folded into the first quadrant and two reflections across the
/// `k.xy` and `k.yx` edges collapse the query into one of the twelve
/// congruent wedges, after which the shape reduces to a single clamped edge
/// with signed distance `length(p) * sign(p.y)`. The fixed hexagonal trig
/// values (`cos 30deg`, `tan 30deg`, `sqrt 3`) are compile-time constants,
/// so the runtime path uses only `abs`, `clamp`, `min`, `sign` and `sqrt`.
pub fn hexagram_2d(point: [f32; 2], r: f32) -> f32 {
    // k = (-0.5, cos(pi/6), tan(pi/6), sqrt(3)) as compile-time constants.
    const KX: f32 = -0.5;
    const KY: f32 = 0.866_025_4;
    const KZ: f32 = 0.577_350_26;
    const KW: f32 = 1.732_050_8;
    let mut p = [point[0].abs(), point[1].abs()];
    let f1 = 2.0 * (KX * p[0] + KY * p[1]).min(0.0);
    p = [p[0] - f1 * KX, p[1] - f1 * KY];
    // Second reflection uses k.yx = (KY, KX).
    let f2 = 2.0 * (KY * p[0] + KX * p[1]).min(0.0);
    p = [p[0] - f2 * KY, p[1] - f2 * KX];
    p = [p[0] - p[0].clamp(KZ * r, KW * r), p[1] - r];
    length2(p) * p[1].signum()
}

/// Exact signed distance to the Inigo Quilez unit heart (`sdHeart`), with its
/// bottom tip at the origin and its top cusp at `(0, 1)`.
///
/// The upper lobes are circular arcs of radius `sqrt 2 / 4` centred at
/// `(+/-0.25, 0.75)`; the lower sides are straight flanks running from the tip
/// up to `(+/-0.5, 0.5)`. Points above the `|x| + y = 1` diagonal measure
/// against the lobe circle; the rest take the nearer of the top cusp and the
/// diagonal flank, signed by `sign(|x| - y)`. The only transcendental is the
/// compile-time constant `sqrt 2 / 4`, so the runtime path uses only `abs`,
/// `max`, `min`, `sign` and `sqrt`.
pub fn heart_2d(point: [f32; 2]) -> f32 {
    // sqrt(2)/4 as a compile-time constant (lobe circle radius).
    const R: f32 = 0.353_553_38;
    let x = point[0].abs();
    let y = point[1];
    if y + x > 1.0 {
        // Upper lobe: distance to the right lobe circle (left is the mirror).
        let dx = x - 0.25;
        let dy = y - 0.75;
        (dx * dx + dy * dy).sqrt() - R
    } else {
        // Lower region: nearer of the top cusp (0,1) and the flank ray y = x.
        let cx = x;
        let cy = y - 1.0;
        let d_cusp = cx * cx + cy * cy;
        let s = 0.5 * (x + y).max(0.0);
        let fx = x - s;
        let fy = y - s;
        let d_flank = fx * fx + fy * fy;
        d_cusp.min(d_flank).sqrt() * (x - y).signum()
    }
}

/// Exact signed distance to the Inigo Quilez three-arc "egg" (`sdEgg`) with its
/// axis of symmetry on `x = 0`.
///
/// `ra` is the radius of the circular bottom (centred on the origin) and
/// `rb` is the radius of the rounded top cap (centred at `(0, sqrt 3 * (ra -
/// rb))`); the two cheeks are arcs of radius `2*(ra - rb) + rb` centred at
/// `(-/+(ra - rb), 0)`. The query is folded to `x >= 0` and routed to the
/// bottom circle, the top cap, or a cheek by the sign of `y` and the
/// `sqrt 3 (x + r) < y` test, so the whole shape stays `C1` continuous. The
/// only transcendental is the compile-time `sqrt 3`, leaving the runtime path
/// on `abs`, `min` (branch select) and `sqrt`.
pub fn egg_2d(point: [f32; 2], ra: f32, rb: f32) -> f32 {
    // k = sqrt(3) as a compile-time constant.
    const K: f32 = 1.732_050_8;
    let px = point[0].abs();
    let py = point[1];
    let r = ra - rb;
    let d = if py < 0.0 {
        // Bottom: circle of radius ra centred at the origin.
        length2([px, py]) - r
    } else if K * (px + r) < py {
        // Top cap: circle of radius rb centred at (0, sqrt(3) * r).
        length2([px, py - K * r])
    } else {
        // Cheek: arc of radius 2r + rb centred at (-r, 0).
        length2([px + r, py]) - 2.0 * r
    };
    d - rb
}

/// Exact signed distance to a filled simple polygon in 2D (Inigo Quilez
/// `sdPolygon`). `verts` lists the vertices in order; the sign is
/// winding-independent (negative inside, positive outside) and the magnitude is
/// the true Euclidean distance to the nearest edge.
///
/// Every edge contributes its clamped point-to-segment distance and an even-odd
/// crossing test flips the running sign, so the routine resolves convex and
/// concave outlines alike using only `dot`, `clamp`, `min`, `sqrt` and
/// comparisons. Fewer than three vertices bound no interior, so the distance to
/// the first vertex is returned (or `f32::INFINITY` for an empty slice).
pub fn polygon_2d(point: [f32; 2], verts: &[[f32; 2]]) -> f32 {
    let n = verts.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let mut d = {
        let w = [point[0] - verts[0][0], point[1] - verts[0][1]];
        w[0] * w[0] + w[1] * w[1]
    };
    let mut s = 1.0f32;
    for i in 0..n {
        let j = (i + n - 1) % n;
        let e = [verts[j][0] - verts[i][0], verts[j][1] - verts[i][1]];
        let w = [point[0] - verts[i][0], point[1] - verts[i][1]];
        let dot_ee = e[0] * e[0] + e[1] * e[1];
        let t = if dot_ee > 0.0 {
            ((e[0] * w[0] + e[1] * w[1]) / dot_ee).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = [w[0] - e[0] * t, w[1] - e[1] * t];
        d = d.min(b[0] * b[0] + b[1] * b[1]);
        let c0 = point[1] >= verts[i][1];
        let c1 = point[1] < verts[j][1];
        let c2 = e[0] * w[1] > e[1] * w[0];
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }
    }
    s * d.sqrt()
}

/// Exact signed distance to an axis-aligned rounded rectangle in 2D with
/// independent per-corner radii (Inigo Quilez `sdRoundedBox`).
///
/// `half_extent` is the box half-size and `radii` lists the corner radii as
/// `[top_right, bottom_right, top_left, bottom_left]`; the active corner radius
/// is selected by the query's quadrant. Each radius must not exceed the smaller
/// half-extent. The straight edges fall out of the inset-box `min(max(..),0)`
/// term and the corners from `length(max(q,0))`, so the result is exact and
/// uses only `abs`, `min`, `max` and `sqrt`.
pub fn rounded_box_2d(point: [f32; 2], half_extent: [f32; 2], radii: [f32; 4]) -> f32 {
    // radii = [top_right, bottom_right, top_left, bottom_left]; pick by quadrant.
    let rx = if point[0] > 0.0 { radii[0] } else { radii[2] };
    let ry = if point[0] > 0.0 { radii[1] } else { radii[3] };
    let r = if point[1] > 0.0 { rx } else { ry };
    let qx = point[0].abs() - half_extent[0] + r;
    let qy = point[1].abs() - half_extent[1] + r;
    qx.max(qy).min(0.0) + length2([qx.max(0.0), qy.max(0.0)]) - r
}

/// Exact signed distance to an oriented 3D vesica (a lens / rugby-ball shape)
/// whose axis is the segment `a`-`b` and which bulges to radial half-width
/// `width` at its midpoint (Inigo Quilez `sdVesicaSegment`).
///
/// The surface is the revolution of a circular arc about the `a`-`b` axis: it
/// tapers to points at both endpoints and reaches its maximum radial half-width
/// `width` at the centre. The query is reduced to axial/radial cylindrical
/// coordinates about the axis and the nearest surface point is either the shared
/// endpoint tip or a point on the generating arc, selected by a single linear
/// test. The result is therefore exact and uses only `dot`, `sqrt` (through
/// `length`), division, `min`/`max` and comparisons.
pub fn vesica_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], width: f32) -> f32 {
    let c = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let ba = sub3(b, a);
    let l = length(ba);
    let v = [ba[0] / l, ba[1] / l, ba[2] / l];
    let pc = sub3(point, c);
    let y = dot(pc, v);
    let perp = [pc[0] - y * v[0], pc[1] - y * v[1], pc[2] - y * v[2]];
    let qx = length(perp);
    let qy = y.abs();
    let r = 0.5 * l;
    let d = 0.5 * (r * r - width * width) / width;
    // Nearest feature: the shared tip at radial 0, axial r (when the query sits
    // past the taper) or the generating arc centred at axial 0, radial -d with
    // radius d + width (everywhere else).
    let (hx, hy, hz) = if r * qx < d * (qy - r) {
        (0.0f32, r, 0.0f32)
    } else {
        (-d, 0.0f32, d + width)
    };
    length2([qx - hx, qy - hy]) - hz
}

/// Exact signed distance to a circle of radius `radius` centred at the origin
/// in 2D (Inigo Quilez `sdCircle`).
///
/// Negative inside, zero on the rim and positive outside. The field is simply
/// the radial distance minus the radius, so it is exact everywhere and uses
/// only `sqrt` (through `length2`) and a subtraction.
pub fn circle_2d(point: [f32; 2], radius: f32) -> f32 {
    length2(point) - radius
}

/// Exact 2D surface normal (unit gradient) of [`circle_2d`] at `point`: the
/// outward radial unit vector `point / |point|`.
///
/// The circle field is `|point| - radius`, whose gradient is the normalised
/// position independent of `radius`. At the centre the direction is undefined,
/// so the zero vector is returned.
pub fn circle_2d_gradient(point: [f32; 2]) -> [f32; 2] {
    let l = length2(point);
    if l == 0.0 {
        return [0.0, 0.0];
    }
    [point[0] / l, point[1] / l]
}

/// Exact signed distance to an upward-pointing regular five-pointed star in 2D
/// (Inigo Quilez `sdStar5`).
///
/// `radius` is the outer-tip radius and `inner_ratio` in `(0, 1)` scales the
/// inner-vertex radius (`inner = radius * inner_ratio`). The query is folded
/// into one 36 degree wedge by two mirror reflections against fixed
/// `cos 36 deg`/`sin 36 deg` constants and reduced to a single edge distance, so
/// the field is exact and uses only `abs`, `max`, `clamp`, `sqrt`, `sign` and
/// dot products (no runtime trigonometry).
pub fn star5_2d(point: [f32; 2], radius: f32, inner_ratio: f32) -> f32 {
    // k1 = (cos 36 deg, -sin 36 deg); the second reflection mirrors it across
    // the y axis. Both are compile-time constants, keeping the fold trig-free.
    const K1X: f32 = 0.809_017;
    const K1Y: f32 = -0.587_785_25;
    let mut px = point[0].abs();
    let mut py = point[1];
    let d1 = (K1X * px + K1Y * py).max(0.0);
    px -= 2.0 * d1 * K1X;
    py -= 2.0 * d1 * K1Y;
    let d2 = (-K1X * px + K1Y * py).max(0.0);
    px -= 2.0 * d2 * (-K1X);
    py -= 2.0 * d2 * K1Y;
    px = px.abs();
    py -= radius;
    // Edge running from the outer tip towards the neighbouring inner vertex.
    let bax = inner_ratio * (-K1Y);
    let bay = inner_ratio * K1X - 1.0;
    let bb = bax * bax + bay * bay;
    let h = ((px * bax + py * bay) / bb).clamp(0.0, radius);
    let dx = px - bax * h;
    let dy = py - bay * h;
    (dx * dx + dy * dy).sqrt() * (py * bax - px * bay).signum()
}

/// Exact signed distance to an upward-pointing regular pentagram (the {5/2}
/// star) of outer radius `radius` in 2D.
///
/// A pentagram is the five-pointed star whose inner concave vertices sit at the
/// golden-ratio radius `radius * (3 - sqrt 5) / 2 = radius / phi^2`, so this
/// delegates to `star5_2d` with that fixed ratio and inherits its exact,
/// trigonometry-free evaluation.
pub fn pentagram_2d(point: [f32; 2], radius: f32) -> f32 {
    // (3 - sqrt 5) / 2 = 1 / phi^2 is the pentagram inner/outer radius ratio.
    const INNER_RATIO: f32 = 0.381_966_02;
    star5_2d(point, radius, INNER_RATIO)
}

/// Exact signed distance to a 2D vesica (a symmetric lens) in 2D
/// (Inigo Quilez `sdVesica`).
///
/// The lens is the intersection of two circles of radius `radius` whose centres
/// sit at `(-offset, 0)` and `(+offset, 0)` (requires `radius > offset`). It
/// spans `radius - offset` along x and `sqrt(radius^2 - offset^2)` along y, with
/// the two cusps at the circle intersections. The query is folded into the first
/// quadrant and the nearest feature is either a cusp or one of the two arcs,
/// selected by a single linear test, so the field is exact and uses only `abs`,
/// `sqrt` (through `length2`), `sign` and comparisons.
pub fn vesica_2d(point: [f32; 2], radius: f32, offset: f32) -> f32 {
    let px = point[0].abs();
    let py = point[1].abs();
    let b = (radius * radius - offset * offset).sqrt();
    if (py - b) * offset > px * b {
        length2([px, py - b]) * offset.signum()
    } else {
        length2([px + offset, py]) - radius
    }
}

/// Exact signed distance to an arbitrary triangle in 2D with vertices `a`,
/// `b` and `c` (Inigo Quilez `sdTriangle`).
///
/// Each of the three edges contributes a point-to-segment distance (the
/// projection parameter clamped to the edge) and the unsigned distance is the
/// smallest of the three. The interior sign is recovered winding-independently
/// by taking the component-wise minimum of the signed edge areas scaled by the
/// triangle's own orientation `s`: a point that lies on the inner side of every
/// edge keeps a positive minimum area and is reported negative. Built from
/// `clamp`, `min`, `signum` and `sqrt`, so it is exact, transcendental-free and
/// valid for either vertex winding.
pub fn triangle_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    let e0 = [b[0] - a[0], b[1] - a[1]];
    let e1 = [c[0] - b[0], c[1] - b[1]];
    let e2 = [a[0] - c[0], a[1] - c[1]];
    let v0 = [point[0] - a[0], point[1] - a[1]];
    let v1 = [point[0] - b[0], point[1] - b[1]];
    let v2 = [point[0] - c[0], point[1] - c[1]];
    let pq0 = {
        let t = ((v0[0] * e0[0] + v0[1] * e0[1]) / dot2_2(e0)).clamp(0.0, 1.0);
        [v0[0] - e0[0] * t, v0[1] - e0[1] * t]
    };
    let pq1 = {
        let t = ((v1[0] * e1[0] + v1[1] * e1[1]) / dot2_2(e1)).clamp(0.0, 1.0);
        [v1[0] - e1[0] * t, v1[1] - e1[1] * t]
    };
    let pq2 = {
        let t = ((v2[0] * e2[0] + v2[1] * e2[1]) / dot2_2(e2)).clamp(0.0, 1.0);
        [v2[0] - e2[0] * t, v2[1] - e2[1] * t]
    };
    let s = (e0[0] * e2[1] - e0[1] * e2[0]).signum();
    let dx = dot2_2(pq0).min(dot2_2(pq1)).min(dot2_2(pq2));
    let dy = (s * (v0[0] * e0[1] - v0[1] * e0[0]))
        .min(s * (v1[0] * e1[1] - v1[1] * e1[0]))
        .min(s * (v2[0] * e2[1] - v2[1] * e2[0]));
    -dx.sqrt() * dy.signum()
}

/// Exact signed distance to an axis-aligned rectangle in 2D centred at the
/// origin with half-extents `half_extent` (Inigo Quilez `sdBox`).
///
/// Folding the query into the first quadrant with `abs` reduces the problem to
/// the corner offset `q = |point| - half_extent`. Outside the box the distance
/// is `length(max(q, 0))` (the straight edges give a zero component, the corner
/// region both), and inside it is the negative `max(q.x, q.y)`. Built from
/// `abs`, `min`, `max` and `sqrt`, so it is exact and transcendental-free. This
/// is the sharp-cornered specialisation of `rounded_box_2d` with zero radii.
pub fn box_2d(point: [f32; 2], half_extent: [f32; 2]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    length2([qx.max(0.0), qy.max(0.0)]) + qx.max(qy).min(0.0)
}

/// Exact 2D surface normal (unit gradient) of [`box_2d`] at `point` for the
/// axis-aligned rectangle of half-extents `half_extent`.
///
/// Mirrors the 3D [`box_gradient`] in the plane. With
/// `q = |point| - half_extent` and `m = max(q, 0)`: outside the rectangle the
/// gradient is the normalised overshoot `m / |m|` with each axis' original
/// sign restored; inside, the nearest edge is the least-negative axis (largest
/// `q_i`) and the gradient is the unit vector along that axis. Points exactly on
/// a face/corner crease (where the direction is undefined) are a measure-zero
/// set and resolve to one of the adjacent faces.
pub fn box_2d_gradient(point: [f32; 2], half_extent: [f32; 2]) -> [f32; 2] {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    let mx = qx.max(0.0);
    let my = qy.max(0.0);
    let len = length2([mx, my]);
    if len > 0.0 {
        return [point[0].signum() * mx / len, point[1].signum() * my / len];
    }
    if qx >= qy {
        [point[0].signum(), 0.0]
    } else {
        [0.0, point[1].signum()]
    }
}

/// Exact signed distance to a vertical capsule in 3D: the segment from the
/// origin to `(0, height, 0)` inflated by radius `radius` (Inigo Quilez
/// `sdVerticalCapsule`).
///
/// Clamping the query's height into `[0, height]` snaps it onto the nearest
/// point of the axis segment; the field is then the distance to that point
/// minus the radius. This is the axis-aligned specialisation of the general
/// `capsule`, kept as a dedicated entry because it avoids the segment
/// projection and is the common upright-pill case. Built from `clamp`, `min`,
/// `max` and `sqrt`, so it is exact and transcendental-free.
pub fn vertical_capsule(point: [f32; 3], height: f32, radius: f32) -> f32 {
    let qy = point[1] - point[1].clamp(0.0, height);
    length([point[0], qy, point[2]]) - radius
}

/// Exact surface normal (unit gradient) of [`vertical_capsule`] at `point` for
/// a capsule of `height` along `+y`.
///
/// Mirrors [`capsule_gradient`] specialised to the upright axis: with
/// `qy = p.y - clamp(p.y, 0, height)` the offset from the nearest skeleton
/// point is `(p.x, qy, p.z)`, and the normal is that vector normalised. On the
/// skeleton segment (where the offset vanishes) the direction is undefined and
/// the zero vector is returned.
pub fn vertical_capsule_gradient(point: [f32; 3], height: f32) -> [f32; 3] {
    let qy = point[1] - point[1].clamp(0.0, height);
    let v = [point[0], qy, point[2]];
    let l = length(v);
    if l == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Exact signed distance to a filled annulus (ring / washer) in 2D centred at
/// the origin, with mid-line radius `radius` and half-thickness `half_width`.
///
/// The field of the circle of radius `radius` is `length(point) - radius`; its
/// absolute value is the distance to that circle, and subtracting the band
/// half-width yields the signed distance to the ring between radii
/// `radius - half_width` and `radius + half_width`. Built from `abs`, `min`,
/// `max` and `sqrt`, so it is exact and transcendental-free. For a hollow ring
/// outline this is the `onion` of `circle_2d`; kept as a dedicated primitive
/// because annular bands are a ubiquitous building block.
pub fn annulus_2d(point: [f32; 2], radius: f32, half_width: f32) -> f32 {
    (length2(point) - radius).abs() - half_width
}

/// Exact signed distance to a 2D capsule (stadium): the segment from `a` to
/// `b` inflated by radius `radius`.
///
/// The query is projected onto the segment with the projection parameter
/// clamped to `[0, 1]`, giving the exact distance to the nearest point of the
/// segment; subtracting `radius` yields the signed distance to the rounded
/// stadium (two end discs joined by a slab). Built from `clamp`, `min`, `max`
/// and `sqrt`, so it is exact and transcendental-free. This is the general
/// arbitrary-endpoint companion to the upright `uneven_capsule_2d`.
pub fn capsule_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2], radius: f32) -> f32 {
    segment_2d(point, a, b) - radius
}

/// Exact signed distance to an oriented 2D vesica (lens) whose pointed tips sit
/// at `a` and `b` with maximum half-width `w` measured perpendicular to the
/// `a`-`b` axis at the midpoint.
///
/// This is the 2D companion to `vesica_segment`. The query is mapped into the
/// lens-local frame (`axial` along the unit tip direction, `perp` along the
/// perpendicular) and then delegated to the already-exact origin-centred
/// `vesica_2d(_, radius, offset)`. For that canonical lens the tips lie at
/// `(0, +/-sqrt(radius^2 - offset^2))` and the apex half-width is
/// `radius - offset`; solving `sqrt(radius^2 - offset^2) = |a-b|/2` and
/// `radius - offset = w` yields `radius = (half^2 + w^2) / (2w)` and
/// `offset = (half^2 - w^2) / (2w)` with `half = |a-b|/2`. Built from `sqrt`,
/// `abs`, `min` and `signum` (via `vesica_2d`), so it is exact and
/// transcendental-free. Requires `0 < w < half` for a proper lens.
pub fn oriented_vesica_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2], w: f32) -> f32 {
    let cx = (a[0] + b[0]) * 0.5;
    let cy = (a[1] + b[1]) * 0.5;
    let bax = b[0] - a[0];
    let bay = b[1] - a[1];
    let l = length2([bax, bay]);
    let vx = bax / l;
    let vy = bay / l;
    let pcx = point[0] - cx;
    let pcy = point[1] - cy;
    let axial = pcx * vx + pcy * vy;
    let perp = -pcx * vy + pcy * vx;
    let half = l * 0.5;
    let radius = (half * half + w * w) / (2.0 * w);
    let offset = (half * half - w * w) / (2.0 * w);
    vesica_2d([perp, axial], radius, offset)
}

/// Exact signed distance to a rectangular frame (hollow box outline) in 2D:
/// the `thickness`-neighbourhood of the axis-aligned rectangle outline with
/// half-extents `half_extent` centred at the origin.
///
/// `box_2d` is the exact signed distance to the solid rectangle, so its
/// absolute value is the exact unsigned distance to the rectangle *boundary
/// curve*; subtracting `thickness` offsets that curve into a band straddling
/// the outline (outer wall at `half_extent + thickness`, inner hole at
/// `half_extent - thickness`, corners rounded with radius `thickness`). The
/// result is negative inside the wall and positive in both the hole and the
/// exterior. Built from `abs`, `min`, `max` and `sqrt` (via `box_2d`), so it is
/// exact and transcendental-free.
pub fn box_frame_2d(point: [f32; 2], half_extent: [f32; 2], thickness: f32) -> f32 {
    box_2d(point, half_extent).abs() - thickness
}

/// Exact signed distance to a 2D tunnel / archway (Inigo Quilez `sdTunnel`):
/// a flat-bottomed rectangle of half-width `half_width` spanning `y` in
/// `[-height, 0]` capped by a semicircle of radius `half_width` over `y >= 0`.
///
/// Folding with `abs` on `x` and flipping `y` reduces the query to one quadrant.
/// Two candidate distances are combined: `d1` measures the rectangular walls
/// (clamped corner offset) while `d2` switches, for the capped region, to the
/// radial distance `length(p) - half_width` so the semicircular arch is exact.
/// The nearer squared distance is taken and the interior sign recovered from
/// `max(q.x, q.y) < 0`. Built from `abs`, `min`, `max` and `sqrt`, so it is exact
/// and transcendental-free.
pub fn tunnel_2d(point: [f32; 2], half_width: f32, height: f32) -> f32 {
    let px = point[0].abs();
    let py = -point[1];
    let mut qx = px - half_width;
    let qy = py - height;
    let d1 = dot2_2([qx.max(0.0), qy]);
    qx = if py > 0.0 { qx } else { length2([px, py]) - half_width };
    let d2 = dot2_2([qx, qy.max(0.0)]);
    let d = d1.min(d2).sqrt();
    if qx.max(qy) < 0.0 { -d } else { d }
}

/// Exact unsigned distance from `point` to the finite 3D line segment
/// `a`-`b` (Inigo Quilez `udSegment`).
///
/// Projects the point onto the segment, clamping the parameter to `[0, 1]`
/// so the closest point stays between the endpoints, then returns the
/// distance to that closest point. A degenerate segment (`a == b`) reduces
/// to the distance to `a`. This is the zero-radius core of `capsule`,
/// built only from `dot`, `clamp`, and `sqrt` (transcendental-free).
pub fn segment_3d(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let pa = sub3(point, a);
    let ba = sub3(b, a);
    let ba_len_sq = dot(ba, ba);
    // Clamp the projection so the closest point stays on the finite segment;
    // a zero-length segment pins the parameter at the start point.
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    length([pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h])
}

#[cfg(test)]
mod tests {
    use super::{
        annulus_2d, arc, box_2d, box_2d_gradient, box_frame, box_frame_2d, box_gradient, box_sdf, capped_cone, capped_cone_segment, capped_cylinder, capped_torus, capsule, capsule_2d, capsule_gradient, circle_2d, circle_2d_gradient, cone_sdf, cross_2d, cut_disk_2d, cut_hollow_sphere,
        cut_sphere, cylinder_segment, death_star, egg_2d, ellipsoid_sdf, equilateral_triangle_2d, heart_2d, hex_prism, hexagram_2d, horseshoe_2d, infinite_cone, infinite_cylinder, isosceles_triangle_2d, length2, line_sdf, link, moon,
        octagon_prism, octahedron, oriented_box_2d, oriented_vesica_2d, parallelogram, pentagram_2d, pie, plane, plane_gradient, polygon_2d, pyramid, quad_sdf, regular_hexagon_2d, regular_octagon_2d, regular_pentagon_2d, rhombus, rhombus_2d, round_box, round_cone_sdf,
        round_cone_segment, rounded_box_2d, rounded_cross_2d, rounded_cylinder, rounded_x,
        segment_2d, segment_3d, solid_angle, sphere, sphere_gradient, star5_2d, torus, torus_gradient, trapezoid_isosceles, triangle_2d, triangle_sdf, triangular_prism, tunnel_2d,
        uneven_capsule_2d, vertical_capsule, vertical_capsule_gradient, vesica, vesica_2d, vesica_segment,
    };

    // Independent brute-force reference: densely sample the segment and take
    // the minimum Euclidean distance to the query point.
    fn segment_3d_bruteforce(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
        let mut best = f32::INFINITY;
        let steps = 4000;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let q = [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ];
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            best = best.min(d);
        }
        best
    }

    #[test]
    fn segment_3d_analytic_cases() {
        let a = [-1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        // Endpoints and midpoint lie on the segment.
        assert!(segment_3d(a, a, b).abs() < 1e-6);
        assert!(segment_3d(b, a, b).abs() < 1e-6);
        assert!(segment_3d([0.0, 0.0, 0.0], a, b).abs() < 1e-6);
        // Perpendicular offset from the middle equals the offset distance.
        assert!((segment_3d([0.0, 3.0, 0.0], a, b) - 3.0).abs() < 1e-6);
        assert!((segment_3d([0.0, 0.0, 4.0], a, b) - 4.0).abs() < 1e-6);
        // Beyond an endpoint: distance to that endpoint (3-4-5 triangle).
        assert!((segment_3d([4.0, 3.0, 0.0], a, b) - 3.0f32.hypot(3.0)).abs() < 1e-6);
        // Degenerate (zero-length) segment reduces to distance to the point.
        let d = [2.0, 2.0, 1.0];
        let expect = (4.0f32 + 4.0 + 1.0).sqrt();
        assert!((segment_3d([0.0, 0.0, 0.0], d, d) - expect).abs() < 1e-6);
    }

    #[test]
    fn segment_3d_matches_bruteforce_reference() {
        // Deterministic pseudo-random sweep over segments and query points.
        let mut state: u32 = 0x9e37_79b9;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 8.0 - 4.0
        };
        let mut maxerr = 0.0f32;
        for _ in 0..3000 {
            let a = [next(), next(), next()];
            let b = [next(), next(), next()];
            let p = [next(), next(), next()];
            let got = segment_3d(p, a, b);
            let want = segment_3d_bruteforce(p, a, b);
            maxerr = maxerr.max((got - want).abs());
        }
        // Error is bounded by the brute-force sampling resolution, not the
        // (exact) formula.
        assert!(maxerr < 1e-2, "segment_3d maxerr = {maxerr}");
    }

    #[test]
    fn cylinder_segment_matches_axis_aligned_capped_cylinder() {
        // With endpoints on the y axis symmetric about the origin, the general
        // solver must agree with the specialised `capped_cylinder` everywhere.
        let a = [0.0, -1.5, 0.0];
        let b = [0.0, 1.5, 0.0];
        let r = 0.75;
        let samples = [
            [0.0, 0.0, 0.0],   // interior centre
            [0.5, 0.0, 0.0],   // interior, off axis
            [2.0, 0.0, 0.0],   // outside the side wall
            [0.0, 3.0, 0.0],   // outside past the top cap
            [0.75, 1.5, 0.0],  // on the top rim
            [1.2, 2.1, 0.4],   // outside the rim corner
        ];
        for p in samples {
            let general = cylinder_segment(p, a, b, r);
            let aligned = capped_cylinder(p, 1.5, r);
            assert!(
                (general - aligned).abs() < 1e-5,
                "mismatch at {p:?}: general={general} aligned={aligned}"
            );
        }
    }

    #[test]
    fn cylinder_segment_side_and_cap_distances_are_exact() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        let r = 1.0;
        // Straight out from the side wall at mid-height: distance is radius gap.
        assert!((cylinder_segment([3.0, 1.0, 0.0], a, b, r) - 2.0).abs() < 1e-5);
        // Directly above the top cap on the axis: distance is the axial gap.
        assert!((cylinder_segment([0.0, 3.5, 0.0], a, b, r) - 1.5).abs() < 1e-5);
        // Interior point: negative distance to the nearest wall (side at 0.4).
        assert!((cylinder_segment([0.6, 1.0, 0.0], a, b, r) - (-0.4)).abs() < 1e-5);
    }

    #[test]
    fn cylinder_segment_is_rigid_motion_invariant() {
        // The signed distance is a geometric quantity: translating the query and
        // both endpoints by the same offset must leave it unchanged.
        let a = [0.2, -1.0, 0.3];
        let b = [0.6, 1.0, -0.4];
        let r = 0.5;
        let p = [1.3, 0.2, 0.9];
        let offset = [4.0, -2.0, 7.0];
        let shift = |v: [f32; 3]| [v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]];
        let base = cylinder_segment(p, a, b, r);
        let moved = cylinder_segment(shift(p), shift(a), shift(b), r);
        assert!((base - moved).abs() < 1e-5, "base={base} moved={moved}");
    }

    // Independent exact 2D signed distance to a convex polygon (IQ sdPolygon),
    // used to cross-check the octagon prism's cross-section. Transcendental use
    // is fine here: test code is not bound by the module's purity rule.
    fn polygon_sdf2(px: f32, py: f32, verts: &[[f32; 2]]) -> f32 {
        let n = verts.len();
        let mut d = {
            let w = [px - verts[0][0], py - verts[0][1]];
            w[0] * w[0] + w[1] * w[1]
        };
        let mut s = 1.0_f32;
        for i in 0..n {
            let j = (i + n - 1) % n;
            let e = [verts[j][0] - verts[i][0], verts[j][1] - verts[i][1]];
            let w = [px - verts[i][0], py - verts[i][1]];
            let t = (e[0] * w[0] + e[1] * w[1]) / (e[0] * e[0] + e[1] * e[1]);
            let t = t.clamp(0.0, 1.0);
            let b = [w[0] - e[0] * t, w[1] - e[1] * t];
            d = d.min(b[0] * b[0] + b[1] * b[1]);
            let c = [py >= verts[i][1], py < verts[j][1], e[0] * w[1] > e[1] * w[0]];
            if (c[0] && c[1] && c[2]) || (!c[0] && !c[1] && !c[2]) {
                s = -s;
            }
        }
        s * d.sqrt()
    }

    // Reference octagon prism: exact 2D octagon distance (apothem `r`, flats on
    // the axes) extruded along z by `h`, combined with the standard slab rule.
    fn octagon_prism_reference(point: [f32; 3], r: f32, h: f32) -> f32 {
        let big_r = r / (std::f32::consts::PI / 8.0).cos();
        let mut verts = [[0.0_f32; 2]; 8];
        for (k, v) in verts.iter_mut().enumerate() {
            let ang = std::f32::consts::PI / 8.0 + std::f32::consts::FRAC_PI_4 * k as f32;
            *v = [big_r * ang.cos(), big_r * ang.sin()];
        }
        let d2 = polygon_sdf2(point[0], point[1], &verts);
        let dz = point[2].abs() - h;
        let outside = (d2.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt();
        let inside = d2.max(dz).min(0.0);
        inside + outside
    }

    #[test]
    fn oriented_vesica_2d_closed_forms() {
        // Horizontal lens, tips at (+/-1, 0), half-width 0.5 => radius 1.25, offset 0.75.
        let a = [-1.0_f32, 0.0];
        let b = [1.0_f32, 0.0];
        let w = 0.5_f32;
        assert!((oriented_vesica_2d([0.0, 0.0], a, b, w) - (-0.5)).abs() < 1e-6);
        assert!(oriented_vesica_2d([1.0, 0.0], a, b, w).abs() < 1e-6);
        assert!(oriented_vesica_2d([-1.0, 0.0], a, b, w).abs() < 1e-6);
        assert!(oriented_vesica_2d([0.0, 0.5], a, b, w).abs() < 1e-6);
        assert!(oriented_vesica_2d([0.0, -0.5], a, b, w).abs() < 1e-6);
        assert!((oriented_vesica_2d([0.0, 2.0], a, b, w) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn oriented_vesica_2d_matches_ordered_arc_reference() {
        // Independent ordered-arc world-space boundary reference (test-only trig).
        fn reference(p: [f32; 2], a: [f32; 2], b: [f32; 2], w: f32) -> f32 {
            let cx = (a[0] + b[0]) * 0.5;
            let cy = (a[1] + b[1]) * 0.5;
            let bax = b[0] - a[0];
            let bay = b[1] - a[1];
            let l = (bax * bax + bay * bay).sqrt();
            let vx = bax / l;
            let vy = bay / l;
            let nx = -vy;
            let ny = vx;
            let half = l * 0.5;
            let radius = (half * half + w * w) / (2.0 * w);
            let offset = (half * half - w * w) / (2.0 * w);
            let phi0 = (offset / radius).acos();
            let to_world =
                |lx: f32, ly: f32| [cx + nx * lx + vx * ly, cy + ny * lx + vy * ly];
            let m = 1200usize;
            let mut pts: Vec<[f32; 2]> = Vec::with_capacity(2 * (m + 1));
            for i in 0..=m {
                let phi = -phi0 + (2.0 * phi0) * (i as f32) / (m as f32);
                pts.push(to_world(-offset + radius * phi.cos(), radius * phi.sin()));
            }
            for i in 0..=m {
                let phi = phi0 - (2.0 * phi0) * (i as f32) / (m as f32);
                pts.push(to_world(offset - radius * phi.cos(), radius * phi.sin()));
            }
            let mut d = f32::INFINITY;
            for i in 0..pts.len() - 1 {
                let [x1, y1] = pts[i];
                let [x2, y2] = pts[i + 1];
                let ex = x2 - x1;
                let ey = y2 - y1;
                let t = (((p[0] - x1) * ex + (p[1] - y1) * ey) / (ex * ex + ey * ey))
                    .clamp(0.0, 1.0);
                let qx = p[0] - (x1 + ex * t);
                let qy = p[1] - (y1 + ey * t);
                d = d.min((qx * qx + qy * qy).sqrt());
            }
            let c1 = [cx + nx * (-offset), cy + ny * (-offset)];
            let c2 = [cx + nx * offset, cy + ny * offset];
            let inside = ((p[0] - c1[0]).powi(2) + (p[1] - c1[1]).powi(2)).sqrt() <= radius
                && ((p[0] - c2[0]).powi(2) + (p[1] - c2[1]).powi(2)).sqrt() <= radius;
            if inside { -d } else { d }
        }

        let a = [-0.7_f32, 0.4];
        let b = [1.3_f32, -0.6];
        let w = 0.45_f32;
        let mut maxerr = 0.0_f32;
        for gy in -8..=8 {
            for gx in -8..=8 {
                let p = [gx as f32 * 0.35, gy as f32 * 0.35];
                let got = oriented_vesica_2d(p, a, b, w);
                let want = reference(p, a, b, w);
                maxerr = maxerr.max((got - want).abs());
            }
        }
        assert!(maxerr < 5e-3, "maxerr = {maxerr}");
    }

    #[test]
    fn box_frame_2d_closed_forms() {
        let b = [1.3_f32, 0.8];
        let t = 0.25_f32;
        // Outer and inner wall faces along +x are the zero set.
        assert!(box_frame_2d([b[0] + t, 0.0], b, t).abs() < 1e-6);
        assert!(box_frame_2d([b[0] - t, 0.0], b, t).abs() < 1e-6);
        // Centre of the wall (on the outline itself) is the most interior: -t.
        assert!((box_frame_2d([b[0], 0.0], b, t) - (-t)).abs() < 1e-6);
        // Rectangle centre sits inside the hole; nearest wall is the top edge.
        assert!((box_frame_2d([0.0, 0.0], b, t) - (b[1] - t)).abs() < 1e-6);
        // Far exterior point measures to the outer +x face.
        assert!((box_frame_2d([3.0, 0.0], b, t) - (3.0 - b[0] - t)).abs() < 1e-6);
    }

    #[test]
    fn box_frame_2d_matches_outline_distance_reference() {
        // Independent reference: brute-force unsigned distance to the four
        // rectangle edges, minus the frame thickness (no reuse of box_2d).
        fn dist_seg(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
            let ex = b[0] - a[0];
            let ey = b[1] - a[1];
            let t = (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (ex * ex + ey * ey))
                .clamp(0.0, 1.0);
            let qx = p[0] - (a[0] + ex * t);
            let qy = p[1] - (a[1] + ey * t);
            (qx * qx + qy * qy).sqrt()
        }
        fn reference(p: [f32; 2], b: [f32; 2], t: f32) -> f32 {
            let c = [
                [b[0], b[1]],
                [-b[0], b[1]],
                [-b[0], -b[1]],
                [b[0], -b[1]],
            ];
            let d = dist_seg(p, c[0], c[1])
                .min(dist_seg(p, c[1], c[2]))
                .min(dist_seg(p, c[2], c[3]))
                .min(dist_seg(p, c[3], c[0]));
            d - t
        }
        let b = [1.3_f32, 0.8];
        let t = 0.25_f32;
        let mut maxerr = 0.0_f32;
        for gy in -20..=20 {
            for gx in -25..=25 {
                let p = [gx as f32 * 0.1, gy as f32 * 0.1];
                let got = box_frame_2d(p, b, t);
                let want = reference(p, b, t);
                maxerr = maxerr.max((got - want).abs());
            }
        }
        assert!(maxerr < 1e-5, "maxerr = {maxerr}");
    }

    #[test]
    fn tunnel_2d_closed_forms() {
        let w = 1.2_f32;
        let h = 0.9_f32;
        assert!(tunnel_2d([0.0, w], w, h).abs() < 1e-6); // arch apex
        assert!(tunnel_2d([0.0, -h], w, h).abs() < 1e-6); // bottom centre
        assert!(tunnel_2d([-w, -0.5], w, h).abs() < 1e-6); // on left wall
        // Interior centre: nearest wall is whichever of (w, h) is smaller.
        assert!((tunnel_2d([0.0, 0.0], w, h) - (-h.min(w))).abs() < 1e-6);
        assert!((tunnel_2d([0.0, -h - 0.3], w, h) - 0.3).abs() < 1e-6); // below floor
        assert!((tunnel_2d([0.0, w + 0.4], w, h) - 0.4).abs() < 1e-6); // above apex
    }

    #[test]
    fn tunnel_2d_matches_region_boundary_reference() {
        // Independent reference: region = rect(|x|<=w, -h<=y<=0) U
        // half-disk(x^2+y^2<=w^2, y>=0); unsigned distance to the sampled
        // boundary (floor, two walls, semicircular arch), signed by membership.
        fn dist_seg(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
            let ex = b[0] - a[0];
            let ey = b[1] - a[1];
            let t = (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (ex * ex + ey * ey))
                .clamp(0.0, 1.0);
            let qx = p[0] - (a[0] + ex * t);
            let qy = p[1] - (a[1] + ey * t);
            (qx * qx + qy * qy).sqrt()
        }
        let w = 1.2_f32;
        let h = 0.9_f32;
        let m = 1000usize;
        let mut arc: Vec<[f32; 2]> = Vec::with_capacity(m + 1);
        for i in 0..=m {
            let a = std::f32::consts::PI * (i as f32) / (m as f32);
            arc.push([w * a.cos(), w * a.sin()]);
        }
        let reference = |p: [f32; 2]| -> f32 {
            let mut d = dist_seg(p, [-w, -h], [w, -h])
                .min(dist_seg(p, [-w, -h], [-w, 0.0]))
                .min(dist_seg(p, [w, -h], [w, 0.0]));
            for i in 0..m {
                d = d.min(dist_seg(p, arc[i], arc[i + 1]));
            }
            let inside = (p[0].abs() <= w && p[1] >= -h && p[1] <= 0.0)
                || (p[0] * p[0] + p[1] * p[1] <= w * w && p[1] >= 0.0);
            if inside { -d } else { d }
        };
        let mut maxerr = 0.0_f32;
        for gy in -24..=24 {
            for gx in -24..=24 {
                let p = [gx as f32 * 0.1, gy as f32 * 0.1];
                maxerr = maxerr.max((tunnel_2d(p, w, h) - reference(p)).abs());
            }
        }
        assert!(maxerr < 5e-3, "maxerr = {maxerr}");
    }

    #[test]
    fn octagon_prism_extrudes_exactly_along_z() {
        // Well inside the cross-section, only the z slab governs.
        let r = 1.0;
        let h = 0.5;
        assert!((octagon_prism([0.0, 0.0, h + 2.0], r, h) - 2.0).abs() < 1e-5);
        assert!((octagon_prism([0.0, 0.0, 0.0], r, h) - (-h)).abs() < 1e-5);
        // Straight out from the top flat edge (normal +y) at mid-depth: the gap
        // to the apothem is exact.
        assert!((octagon_prism([0.0, r + 1.5, 0.0], r, h) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn octagon_prism_matches_independent_polygon_reference() {
        let r = 1.3;
        let h = 0.7;
        // Deterministic pseudo-random sweep across interior, faces, corners,
        // caps, and the exterior corner regions.
        let mut seed = 0x1234_5678_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..400 {
            let p = [
                (next() - 0.5) * 6.0,
                (next() - 0.5) * 6.0,
                (next() - 0.5) * 4.0,
            ];
            let got = octagon_prism(p, r, h);
            let want = octagon_prism_reference(p, r, h);
            assert!(
                (got - want).abs() < 1e-4,
                "mismatch at {p:?}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn infinite_cone_axis_distances_are_exact() {
        // Aperture of 30 degrees, pre-baked as [sin, cos].
        let alpha = 30.0_f32.to_radians();
        let sc = [alpha.sin(), alpha.cos()];

        // On the +y axis at height V (inside the solid): the nearest flank point
        // sits perpendicular at distance V*sin(alpha), with the interior sign.
        let v = 2.5_f32;
        let got = infinite_cone([0.0, v, 0.0], sc);
        let want = -v * alpha.sin();
        assert!((got - want).abs() < 1e-5, "axis interior: got={got} want={want}");

        // Below the apex the whole cone retreats behind, so the closest feature
        // is the apex itself at exactly the depth, with the exterior sign.
        let got_below = infinite_cone([0.0, -v, 0.0], sc);
        assert!((got_below - v).abs() < 1e-5, "apex distance: got={got_below} want={v}");

        // The apex is exactly on the surface.
        assert!(infinite_cone([0.0, 0.0, 0.0], sc).abs() < 1e-6);
    }

    #[test]
    fn infinite_cone_matches_independent_meridian_reference() {
        // Cross-check IQ's closed form against an independent geometric solve in
        // the meridian half-plane: project onto the flank ray, fall back to the
        // apex behind it, and sign by the axis-side half-plane.
        let alpha = 42.0_f32.to_radians();
        let (s, c) = (alpha.sin(), alpha.cos());
        let sc = [s, c];

        // Deterministic LCG samples spanning interior/exterior and both y signs.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 // in [0, 1)
        };
        for _ in 0..256 {
            let x = (next() - 0.5) * 8.0;
            let y = (next() - 0.5) * 8.0;
            let z = (next() - 0.5) * 8.0;

            let u = (x * x + z * z).sqrt();
            let v = y;
            let proj = u * s + v * c;
            let perp = u * c - v * s;
            let dist = if proj >= 0.0 {
                perp.abs()
            } else {
                (u * u + v * v).sqrt()
            };
            let want = if perp < 0.0 { -dist } else { dist };

            let got = infinite_cone([x, y, z], sc);
            assert!(
                (got - want).abs() < 1e-4,
                "mismatch at ({x},{y},{z}): got={got} want={want}"
            );
        }
    }

    #[test]
    fn pie_axis_arc_and_edge_distances_are_exact() {
        let alpha = 50.0_f32.to_radians();
        let (s, c) = (alpha.sin(), alpha.cos());
        let r = 2.0_f32;

        // Interior point on the +y axis: the nearest boundary is the closer of
        // the bounding arc (r - h) and either straight edge (h*sin(alpha)).
        let h = 1.2_f32;
        let got = pie([0.0, h], [s, c], r);
        let want = -(r - h).min(h * s);
        assert!((got - want).abs() < 1e-5, "axis interior: got={got} want={want}");

        // The apex (origin) is a vertex of the sector, hence on the surface.
        assert!(pie([0.0, 0.0], [s, c], r).abs() < 1e-6);

        // A point just outside the arc straight along +y is (dist = depth - r).
        let far = 3.0_f32;
        let got_arc = pie([0.0, far], [s, c], r);
        assert!((got_arc - (far - r)).abs() < 1e-5, "beyond arc: got={got_arc}");
    }

    #[test]
    fn pie_matches_brute_force_boundary() {
        // Independent reference: the sector is the disk intersected with the
        // angular wedge, so its boundary is the two straight edges (t in [0,r])
        // plus the arc (theta in [-alpha, alpha]). Compare the signed distance
        // to a dense brute-force scan of that boundary.
        let alpha = 50.0_f32.to_radians();
        let (s, c) = (alpha.sin(), alpha.cos());
        let r = 2.0_f32;

        // Pre-sample the boundary polyline.
        let mut boundary: Vec<[f32; 2]> = Vec::new();
        let n = 2000;
        for k in 0..=n {
            let t = r * k as f32 / n as f32;
            boundary.push([s * t, c * t]);   // +edge
            boundary.push([-s * t, c * t]);  // -edge (mirror)
        }
        for k in 0..=n {
            let theta = -alpha + 2.0 * alpha * k as f32 / n as f32;
            boundary.push([r * theta.sin(), r * theta.cos()]);
        }

        let mut state: u32 = 0x9e37_79b9;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..200 {
            let x = (next() - 0.5) * 6.0;
            let y = (next() - 0.5) * 6.0;

            // Unsigned distance to the boundary samples.
            let mut best = f32::INFINITY;
            for b in &boundary {
                let dx = x - b[0];
                let dy = y - b[1];
                best = best.min((dx * dx + dy * dy).sqrt());
            }
            // Inside iff within the wedge AND within the disk.
            let px = x.abs();
            let in_wedge = c * px - s * y <= 0.0;
            let in_disk = (x * x + y * y).sqrt() <= r;
            let want = if in_wedge && in_disk { -best } else { best };

            let got = pie([x, y], [s, c], r);
            assert!(
                (got - want).abs() < 3e-3,
                "mismatch at ({x},{y}): got={got} want={want}"
            );
        }
    }

    #[test]
    fn moon_matches_brute_force_boundary() {
        // Independent reference: the crescent is {inside circle A} minus
        // {inside circle B}. Its boundary is the arc of A that lies outside B
        // plus the arc of B that lies inside A. Compare the signed distance to
        // a dense brute-force scan of that boundary.
        let d = 0.6_f32;
        let ra = 1.0_f32;
        let rb = 0.8_f32;

        let mut boundary: Vec<[f32; 2]> = Vec::new();
        let n = 3000;
        for k in 0..n {
            let t = std::f32::consts::TAU * k as f32 / n as f32;
            // Arc of A kept where it is outside B.
            let pa = [ra * t.cos(), ra * t.sin()];
            if ((pa[0] - d) * (pa[0] - d) + pa[1] * pa[1]).sqrt() >= rb {
                boundary.push(pa);
            }
            // Arc of B kept where it is inside A.
            let pb = [d + rb * t.cos(), rb * t.sin()];
            if (pb[0] * pb[0] + pb[1] * pb[1]).sqrt() <= ra {
                boundary.push(pb);
            }
        }
        assert!(!boundary.is_empty());

        let mut state: u32 = 0xdead_beef;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..200 {
            let x = (next() - 0.5) * 4.0;
            let y = (next() - 0.5) * 4.0;

            let mut best = f32::INFINITY;
            for q in &boundary {
                let dx = x - q[0];
                let dy = y - q[1];
                best = best.min((dx * dx + dy * dy).sqrt());
            }
            let in_a = (x * x + y * y).sqrt() <= ra;
            let out_b = ((x - d) * (x - d) + y * y).sqrt() >= rb;
            let want = if in_a && out_b { -best } else { best };

            let got = moon([x, y], d, ra, rb);
            assert!(
                (got - want).abs() < 4e-3,
                "mismatch at ({x},{y}): got={got} want={want}"
            );
        }
    }

    #[test]
    fn moon_is_symmetric_about_the_x_axis() {
        let (d, ra, rb) = (0.6_f32, 1.0_f32, 0.8_f32);
        for &p in &[[0.3_f32, 0.4_f32], [-0.5, 0.9], [1.2, 0.2]] {
            let up = moon(p, d, ra, rb);
            let down = moon([p[0], -p[1]], d, ra, rb);
            assert!((up - down).abs() < 1e-6, "asymmetry at {p:?}");
        }
    }

    #[test]
    fn rounded_x_matches_segment_skeleton_reference() {
        // Independent reference: the rounded X is the two crossing diagonal
        // segments offset by r. Compare against the exact distance to that
        // two-segment skeleton minus r.
        let w = 1.6_f32;
        let r = 0.25_f32;
        let half = w * 0.5;
        let seg_dist = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
            let pa = [p[0] - a[0], p[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let denom = ba[0] * ba[0] + ba[1] * ba[1];
            let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / denom).clamp(0.0, 1.0);
            let dx = pa[0] - ba[0] * h;
            let dy = pa[1] - ba[1] * h;
            (dx * dx + dy * dy).sqrt()
        };
        let s1a = [-half, -half];
        let s1b = [half, half];
        let s2a = [-half, half];
        let s2b = [half, -half];

        let mut state: u32 = 0x5151_a5a5;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..256 {
            let x = (next() - 0.5) * 4.0;
            let y = (next() - 0.5) * 4.0;
            let p = [x, y];
            let want = seg_dist(p, s1a, s1b).min(seg_dist(p, s2a, s2b)) - r;
            let got = rounded_x(p, w, r);
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at ({x},{y}): got={got} want={want}"
            );
        }
    }

    #[test]
    fn rounded_x_centre_and_arm_tips() {
        let w = 2.0_f32;
        let r = 0.3_f32;
        // The centre lies on both bars, so distance is -r.
        assert!((rounded_x([0.0, 0.0], w, r) + r).abs() < 1e-6);
        // An arm tip sits on the skeleton end, still fully inside the stroke.
        assert!((rounded_x([w * 0.5, w * 0.5], w, r) + r).abs() < 1e-6);
        // Just outside an arm tip along the diagonal: distance grows past -r.
        let got = rounded_x([w * 0.5 + 0.5, w * 0.5 + 0.5], w, r);
        let want = (0.5_f32 * 0.5 + 0.5 * 0.5).sqrt() - r;
        assert!((got - want).abs() < 1e-5, "beyond tip: got={got} want={want}");
    }

    #[test]
    fn cross_2d_matches_polyline_boundary() {
        // Independent reference: brute-force signed distance to the sharp plus
        // outline (12 edges), with the sign from an explicit inside test.
        let arm = 1.5_f32;
        let th = 0.5_f32;
        // Outline vertices walked counter-clockwise.
        let verts: [[f32; 2]; 12] = [
            [arm, th], [th, th], [th, arm], [-th, arm],
            [-th, th], [-arm, th], [-arm, -th], [-th, -th],
            [-th, -arm], [th, -arm], [th, -th], [arm, -th],
        ];
        let seg_dist = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
            let pa = [p[0] - a[0], p[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let denom = ba[0] * ba[0] + ba[1] * ba[1];
            let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / denom).clamp(0.0, 1.0);
            let dx = pa[0] - ba[0] * h;
            let dy = pa[1] - ba[1] * h;
            (dx * dx + dy * dy).sqrt()
        };

        let mut state: u32 = 0x0bad_f00d;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..256 {
            let x = (next() - 0.5) * 5.0;
            let y = (next() - 0.5) * 5.0;
            let p = [x, y];
            let mut best = f32::INFINITY;
            for k in 0..12 {
                best = best.min(seg_dist(p, verts[k], verts[(k + 1) % 12]));
            }
            let in_h = x.abs() <= arm && y.abs() <= th;
            let in_v = x.abs() <= th && y.abs() <= arm;
            let want = if in_h || in_v { -best } else { best };
            let got = cross_2d(p, arm, th, 0.0);
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at ({x},{y}): got={got} want={want}"
            );
        }
    }

    #[test]
    fn cross_2d_rounding_insets_the_sharp_field() {
        // Rounding by r grows the shape uniformly, i.e. subtracts r everywhere.
        let (arm, th, r) = (1.2_f32, 0.4_f32, 0.2_f32);
        for &p in &[[0.0_f32, 0.0_f32], [1.0, 0.3], [0.3, 1.0], [2.0, 2.0], [0.5, 0.5]] {
            let sharp = cross_2d(p, arm, th, 0.0);
            let rounded = cross_2d(p, arm, th, r);
            assert!((rounded - (sharp - r)).abs() < 1e-6, "offset wrong at {p:?}");
        }
        // The nearest boundary to the centre is the reentrant corner at
        // (thickness, thickness), a distance thickness*sqrt(2) away, then inset
        // by r. A naive 'nearest wall' guess of thickness would be wrong here.
        let centre = cross_2d([0.0, 0.0], arm, th, r);
        let want_centre = -th * std::f32::consts::SQRT_2 - r;
        assert!((centre - want_centre).abs() < 1e-6, "centre: got={centre} want={want_centre}");
    }

    #[test]
    fn rounded_cross_2d_tips_surface_origin_and_symmetry() {
        // The four convex tips (+-1, 0) and (0, +-h) lie exactly on the
        // surface; the centre is interior with the exact closed-form depth
        // k - sqrt(1 + k^2); and the field is symmetric under axis reflection.
        for &h in &[1.0_f32, 1.4, 2.0, 0.7] {
            let k = 0.5 * (h + 1.0 / h);
            for &tip in &[[1.0_f32, 0.0], [-1.0, 0.0], [0.0, h], [0.0, -h]] {
                assert!(
                    rounded_cross_2d(tip, h).abs() < 1e-6,
                    "tip {tip:?} not on surface for h={h}"
                );
            }
            let origin = rounded_cross_2d([0.0, 0.0], h);
            let want = k - (1.0 + k * k).sqrt();
            assert!((origin - want).abs() < 1e-6, "origin h={h}: {origin} vs {want}");
            // Reflection symmetry across both axes.
            for &p in &[[0.37_f32, 0.52], [0.8, 0.1], [1.3, 0.9]] {
                let base = rounded_cross_2d(p, h);
                for &q in &[[-p[0], p[1]], [p[0], -p[1]], [-p[0], -p[1]]] {
                    assert!((rounded_cross_2d(q, h) - base).abs() < 1e-6, "symmetry h={h} p={p:?}");
                }
            }
        }
    }

    #[test]
    fn rounded_cross_2d_fillet_arc_lies_on_the_surface() {
        // The innermost point of the first-quadrant fillet arc (the arc of
        // radius k centred at (1, k)) is the circle point nearest the origin,
        // P = (1, k) * (1 - k / |(1, k)|). It sits on the boundary, so f = 0.
        // Derived purely from the fillet geometry, independent of the branch
        // formula's exterior half.
        for &h in &[1.0_f32, 1.4, 2.0, 0.7] {
            let k = 0.5 * (h + 1.0 / h);
            let cmag = (1.0 + k * k).sqrt();
            let s = 1.0 - k / cmag;
            let pt = [1.0 * s, k * s];
            // Confirm the sample really falls in the fillet wedge.
            assert!(pt[0] < 1.0 && pt[1] < pt[0] * (k - h) + h, "fillet wedge h={h}");
            assert!(
                rounded_cross_2d(pt, h).abs() < 1e-5,
                "fillet-arc point off surface h={h}: {}",
                rounded_cross_2d(pt, h)
            );
        }
    }

    #[test]
    fn rounded_cross_2d_exterior_is_distance_to_the_nearest_tip() {
        // Beyond an arm tip the nearest feature is the convex tip corner, so
        // the distance is the plain Euclidean distance to that point -- an
        // independent closed form that does not use the fillet branch.
        for &h in &[1.0_f32, 1.4, 2.0, 0.7] {
            // Straight out the +x axis past the tip (1, 0).
            let dx = 0.75_f32;
            assert!((rounded_cross_2d([1.0 + dx, 0.0], h) - dx).abs() < 1e-6, "x tip h={h}");
            // Straight up past the +y tip (0, h).
            let dy = 0.6_f32;
            assert!((rounded_cross_2d([0.0, h + dy], h) - dy).abs() < 1e-6, "y tip h={h}");
            // Diagonally beyond the +x tip: exact corner distance.
            let off = [1.0 + 0.4, 0.3_f32];
            let want = (0.4_f32 * 0.4 + 0.3 * 0.3).sqrt();
            assert!((rounded_cross_2d(off, h) - want).abs() < 1e-6, "x corner h={h}");
        }
    }

    #[test]
    fn rounded_cross_2d_is_an_exact_distance_field() {
        // Verify the Eikonal property |grad f| = 1 off the symmetry axes (the
        // axes are medial-axis creases where a central difference cancels one
        // component). A true signed-distance field satisfies this everywhere it
        // is differentiable, confirming the formula returns exact Euclidean
        // distance rather than a mere bound.
        let eps = 1e-3_f32;
        for &h in &[1.0_f32, 1.4, 2.0, 0.7] {
            for &p in &[
                [0.55_f32, 0.33],
                [0.2, 0.9],
                [1.4, 0.5],
                [0.4, 1.7],
                [0.9, 0.9],
                [1.1, 0.2],
            ] {
                let fx = (rounded_cross_2d([p[0] + eps, p[1]], h)
                    - rounded_cross_2d([p[0] - eps, p[1]], h))
                    / (2.0 * eps);
                let fy = (rounded_cross_2d([p[0], p[1] + eps], h)
                    - rounded_cross_2d([p[0], p[1] - eps], h))
                    / (2.0 * eps);
                let grad = (fx * fx + fy * fy).sqrt();
                assert!((grad - 1.0).abs() < 5e-3, "eikonal h={h} p={p:?}: |grad|={grad}");
            }
        }
    }

    // Horseshoe helper: evaluate from a raw angle (tests may use trig).
    fn horseshoe(point: [f32; 2], theta: f32, r: f32, arm: f32, th: f32) -> f32 {
        horseshoe_2d(point, [theta.sin(), theta.cos()], r, arm, th)
    }

    #[test]
    fn horseshoe_2d_ring_walls_and_symmetry() {
        // Across opening angles and ring geometries, the deep bend at -y sits
        // one half-thickness inside (nearest feature is the radial wall), both
        // ring walls at the bend lie on the surface, a radial step beyond the
        // outer wall reads back exactly, and the field is mirror-symmetric in x.
        for &theta in &[1.0_f32, 0.6, 1.3] {
            for &(r, arm, th) in &[(1.0_f32, 0.4_f32, 0.15_f32), (1.3, 0.5, 0.2), (0.8, 0.3, 0.1)] {
                let bend = horseshoe([0.0, -r], theta, r, arm, th);
                assert!((bend + th).abs() < 1e-6, "bend depth theta={theta} r={r}: {bend}");
                let outer = horseshoe([0.0, -(r + th)], theta, r, arm, th);
                assert!(outer.abs() < 1e-6, "outer wall theta={theta} r={r}: {outer}");
                let inner = horseshoe([0.0, -(r - th)], theta, r, arm, th);
                assert!(inner.abs() < 1e-6, "inner wall theta={theta} r={r}: {inner}");
                let step = 0.3_f32;
                let out = horseshoe([0.0, -(r + th + step)], theta, r, arm, th);
                assert!((out - step).abs() < 1e-6, "radial step theta={theta} r={r}: {out}");
                for &p in &[[0.5_f32, 0.7], [0.9, -0.2], [1.4, 1.1]] {
                    let a = horseshoe(p, theta, r, arm, th);
                    let b = horseshoe([-p[0], p[1]], theta, r, arm, th);
                    assert!((a - b).abs() < 1e-6, "mirror theta={theta} p={p:?}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn horseshoe_2d_is_an_exact_distance_field() {
        // The Eikonal property |grad f| = 1 (off the x symmetry axis, which is a
        // medial-axis crease where a central difference cancels one component)
        // proves the formula returns exact Euclidean distance, not a bound.
        let eps = 1e-3_f32;
        for &theta in &[1.0_f32, 0.6, 1.3] {
            let (r, arm, th) = (1.0_f32, 0.4_f32, 0.15_f32);
            for &p in &[
                [0.55_f32, 0.33],
                [-0.8, 0.9],
                [1.4, -0.5],
                [0.4, -1.7],
                [1.1, 0.2],
                [-1.3, -0.6],
            ] {
                let fx = (horseshoe([p[0] + eps, p[1]], theta, r, arm, th)
                    - horseshoe([p[0] - eps, p[1]], theta, r, arm, th))
                    / (2.0 * eps);
                let fy = (horseshoe([p[0], p[1] + eps], theta, r, arm, th)
                    - horseshoe([p[0], p[1] - eps], theta, r, arm, th))
                    / (2.0 * eps);
                let grad = (fx * fx + fy * fy).sqrt();
                assert!((grad - 1.0).abs() < 5e-3, "eikonal theta={theta} p={p:?}: |grad|={grad}");
            }
        }
    }

    // Central-difference gradient of a 3D scalar field, the independent
    // reference the analytic primitive gradients are checked against.
    fn central_grad3(f: &dyn Fn([f32; 3]) -> f32, p: [f32; 3]) -> [f32; 3] {
        let h = 1e-4_f32;
        let mut g = [0.0_f32; 3];
        for i in 0..3 {
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            g[i] = (f(a) - f(b)) / (2.0 * h);
        }
        g
    }

    fn unit_len3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn sphere_gradient_is_the_exact_unit_radial_normal() {
        // Matches a central difference of the exact SDF, is unit length, and
        // degrades to zero at the undefined centre.
        for &p in &[[1.0_f32, 0.0, 0.0], [0.3, -0.7, 1.2], [-2.0, 0.5, -0.4]] {
            let g = sphere_gradient(p);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad3(&|q| sphere(q, 1.3), p);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "sphere grad p={p:?} axis {k}");
            }
        }
        assert_eq!(sphere_gradient([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn box_gradient_matches_central_difference_off_creases() {
        let b = [0.7_f32, 1.0, 0.5];
        // Points chosen away from faces/edges (the measure-zero creases).
        for &p in &[
            [1.4_f32, 0.2, -0.1],   // exterior, +x face region
            [0.1, 1.6, 0.2],        // exterior, +y face region
            [-1.3, -1.4, -1.1],     // exterior corner octant
            [0.2, -0.3, 0.1],       // deep interior, dominant -nearest face
            [0.55, 0.1, 0.1],       // interior near +x face (x least-negative)
        ] {
            let g = box_gradient(p, b);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad3(&|q| box_sdf(q, b), p);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 1e-3, "box grad p={p:?} axis {k}: {} vs {}", g[k], fd[k]);
            }
        }
        // Interior point closest to the +x face points along +x.
        assert_eq!(box_gradient([0.6, 0.05, 0.05], b), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn torus_gradient_matches_central_difference_and_axis_fallback() {
        let (major, minor) = (1.0_f32, 0.3);
        for &p in &[
            [1.4_f32, 0.1, 0.0],
            [0.9, 0.25, 0.6],
            [-1.2, -0.2, 0.3],
            [0.0, 0.4, 1.3],
        ] {
            let g = torus_gradient(p, major, minor);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad3(&|q| torus(q, major, minor), p);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "torus grad p={p:?} axis {k}: {} vs {}", g[k], fd[k]);
            }
        }
        // On the central y axis the planar direction is undefined -> +y axis.
        assert_eq!(torus_gradient([0.0, 0.5, 0.0], major, minor), [0.0, 1.0, 0.0]);
        assert_eq!(torus_gradient([0.0, -0.5, 0.0], major, minor), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn plane_gradient_is_the_constant_plane_normal() {
        // The gradient of `dot(point, normal) + offset` is exactly `normal`,
        // independent of the sample point, and already unit length.
        for &n in &[
            [0.0_f32, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.267_261_24, 0.534_522_5, 0.801_783_7], // normalized (1,2,3)
        ] {
            let g = plane_gradient(n);
            assert_eq!(g, n);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit n={n:?}");
            // Matches a central difference of the exact plane field anywhere.
            let fd = central_grad3(&|q| plane(q, n, -0.4), [0.3, -0.2, 0.7]);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "plane grad n={n:?} axis {k}");
            }
        }
    }

    #[test]
    fn capsule_gradient_matches_central_difference_off_skeleton() {
        let (a, b, r) = ([-0.5_f32, 0.2, 0.0], [0.6, 0.1, 0.4], 0.3);
        // Points away from the skeleton segment (the measure-zero crease).
        for &p in &[
            [1.2_f32, 0.3, 0.1],
            [-1.0, 0.4, -0.3],
            [0.05, 0.9, 0.2],
            [0.1, -0.6, 0.5],
        ] {
            let g = capsule_gradient(p, a, b);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad3(&|q| capsule(q, a, b, r), p);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "capsule grad p={p:?} axis {k}: {} vs {}", g[k], fd[k]);
            }
        }
        // On the skeleton (point == a) the radial direction is undefined.
        assert_eq!(capsule_gradient(a, a, b), [0.0, 0.0, 0.0]);
        // Degenerate zero-length segment with point at the shared endpoint.
        assert_eq!(capsule_gradient(a, a, a), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn vertical_capsule_gradient_matches_central_difference_off_skeleton() {
        let (height, r) = (1.0_f32, 0.25);
        for &p in &[
            [0.5_f32, 0.3, 0.1],
            [-0.4, 1.3, 0.2],
            [0.2, -0.5, 0.3],
            [0.6, 0.8, -0.3],
        ] {
            let g = vertical_capsule_gradient(p, height);
            assert!((unit_len3(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad3(&|q| vertical_capsule(q, height, r), p);
            for k in 0..3 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "vcap grad p={p:?} axis {k}: {} vs {}", g[k], fd[k]);
            }
        }
        // On the axis segment the radial direction is undefined -> zero.
        assert_eq!(vertical_capsule_gradient([0.0, 0.5, 0.0], height), [0.0, 0.0, 0.0]);
    }

    // Central-difference gradient of a 2D scalar field, the independent
    // reference the analytic 2D primitive gradients are checked against.
    fn central_grad2(f: &dyn Fn([f32; 2]) -> f32, p: [f32; 2]) -> [f32; 2] {
        let h = 1e-4_f32;
        let mut g = [0.0_f32; 2];
        for i in 0..2 {
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            g[i] = (f(a) - f(b)) / (2.0 * h);
        }
        g
    }

    fn unit_len2(v: [f32; 2]) -> f32 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    #[test]
    fn circle_2d_gradient_is_the_exact_unit_radial_normal() {
        for &p in &[[1.0_f32, 0.0], [0.3, -0.7], [-2.0, 0.5]] {
            let g = circle_2d_gradient(p);
            assert!((unit_len2(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad2(&|q| circle_2d(q, 1.3), p);
            for k in 0..2 {
                assert!((g[k] - fd[k]).abs() < 2e-3, "circle grad p={p:?} axis {k}");
            }
        }
        assert_eq!(circle_2d_gradient([0.0, 0.0]), [0.0, 0.0]);
    }

    #[test]
    fn box_2d_gradient_matches_central_difference_off_creases() {
        let b = [0.7_f32, 0.4];
        // Points away from faces/corners (the measure-zero creases).
        for &p in &[
            [1.2_f32, 0.1],   // exterior, +x edge region
            [0.1, 0.9],       // exterior, +y edge region
            [-1.3, -1.0],     // exterior corner quadrant
            [0.3, -0.1],      // interior, nearest -y edge
            [0.55, 0.1],      // interior, nearest +x edge
        ] {
            let g = box_2d_gradient(p, b);
            assert!((unit_len2(g) - 1.0).abs() < 1e-6, "unit p={p:?}");
            let fd = central_grad2(&|q| box_2d(q, b), p);
            for k in 0..2 {
                assert!((g[k] - fd[k]).abs() < 1e-3, "box2d grad p={p:?} axis {k}: {} vs {}", g[k], fd[k]);
            }
        }
        // Interior point closest to the +x edge points along +x.
        assert_eq!(box_2d_gradient([0.55, 0.05], b), [1.0, 0.0]);
    }

    // Exact unsigned distance to a 2D segment, used to cross-check `segment_2d`
    // independently (and as a boundary skeleton for the oriented box corners).
    fn segment_ref(px: f32, py: f32, a: [f32; 2], b: [f32; 2]) -> f32 {
        let (ax, ay, bx, by) = (a[0], a[1], b[0], b[1]);
        let (ex, ey) = (bx - ax, by - ay);
        let (wx, wy) = (px - ax, py - ay);
        let t = ((wx * ex + wy * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
        let (dx, dy) = (wx - ex * t, wy - ey * t);
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn segment_2d_endpoints_perpendicular_and_caps() {
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        // On the segment: zero.
        assert!(segment_2d([0.0, 0.0], a, b).abs() < 1e-6);
        // Perpendicular drop above the midpoint.
        assert!((segment_2d([0.0, 2.0], a, b) - 2.0).abs() < 1e-6);
        // Beyond an endpoint: distance to that endpoint.
        assert!((segment_2d([3.0, 0.0], a, b) - 2.0).abs() < 1e-6);
        assert!((segment_2d([-2.0, 1.0], a, b) - (1.0f32 + 1.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn segment_2d_matches_reference_under_rotation() {
        let a = [0.3, -0.7];
        let b = [1.4, 0.9];
        let samples: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 2.0],
            [2.0, 2.0],
            [0.8, 0.1],
            [1.4, 0.9],
        ];
        for p in samples {
            let got = segment_2d(p, a, b);
            let want = segment_ref(p[0], p[1], a, b);
            assert!((got - want).abs() < 1e-6, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn oriented_box_2d_axis_aligned_closed_forms() {
        // Endpoints on the x axis span length 2 (half-length 1); full width 0.5.
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        let th = 0.5;
        // Straight out from the long side: gap is width/2 subtracted.
        assert!((oriented_box_2d([0.0, 2.0], a, b, th) - 1.75).abs() < 1e-6);
        // Straight past the short end on the axis.
        assert!((oriented_box_2d([3.0, 0.0], a, b, th) - 2.0).abs() < 1e-6);
        // Dead centre: nearest wall is the long side at width/2 = 0.25.
        assert!((oriented_box_2d([0.0, 0.0], a, b, th) - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn oriented_box_2d_matches_polygon_reference() {
        // A slanted bar; cross-check against its four corners as a polygon.
        let a: [f32; 2] = [-0.6, -0.8];
        let b: [f32; 2] = [1.0, 0.4];
        let th = 0.5;
        let l = {
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            (dx * dx + dy * dy).sqrt()
        };
        let d = [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
        let n = [-d[1], d[0]];
        let c = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let hl = 0.5 * l;
        let ht = 0.5 * th;
        // Corners wound CCW in the box's local frame.
        let corner = |sl: f32, st: f32| {
            [c[0] + d[0] * (sl * hl) + n[0] * (st * ht), c[1] + d[1] * (sl * hl) + n[1] * (st * ht)]
        };
        let verts = [corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0), corner(-1.0, -1.0)];
        let samples: [[f32; 2]; 7] = [
            c,
            [c[0] + n[0] * 1.5, c[1] + n[1] * 1.5],
            [c[0] + d[0] * 2.0, c[1] + d[1] * 2.0],
            [0.0, 0.0],
            [1.2, 0.8],
            [-0.5, -1.2],
            [0.2, -0.2],
        ];
        for p in samples {
            let got = oriented_box_2d(p, a, b, th);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn parallelogram_reduces_to_a_box_without_skew() {
        let (wi, he) = (1.0f32, 0.5);
        assert!((parallelogram([0.0, 0.0], wi, he, 0.0) - (-0.5)).abs() < 1e-6);
        assert!((parallelogram([2.0, 0.0], wi, he, 0.0) - 1.0).abs() < 1e-6);
        assert!((parallelogram([0.0, 1.0], wi, he, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn parallelogram_matches_polygon_reference() {
        let (wi, he, sk) = (1.0f32, 0.6, 0.4);
        // Vertices wound CCW: bottom-right, top-right, top-left, bottom-left.
        let verts = [
            [wi - sk, -he],
            [wi + sk, he],
            [-wi + sk, he],
            [-wi - sk, -he],
        ];
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.4, 0.3],
            [1.6, 0.6],
            [-1.6, -0.6],
            [0.0, 1.2],
            [0.0, -1.2],
            [1.2, -0.3],
            [-0.9, 0.2],
        ];
        for p in samples {
            let got = parallelogram(p, wi, he, sk);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn rhombus_2d_vertices_centre_and_edges() {
        let b = [1.0f32, 1.0];
        // Vertex sits on the boundary.
        assert!(rhombus_2d([1.0, 0.0], b).abs() < 1e-6);
        // Centre: inradius is bx*by/sqrt(bx^2+by^2) = 1/sqrt(2).
        assert!((rhombus_2d([0.0, 0.0], b) - (-0.5f32.sqrt())).abs() < 1e-6);
        // Along the x axis beyond the vertex: plain radial gap.
        assert!((rhombus_2d([2.0, 0.0], b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rhombus_2d_matches_polygon_reference() {
        let b = [1.3f32, 0.8];
        let verts = [[b[0], 0.0], [0.0, b[1]], [-b[0], 0.0], [0.0, -b[1]]];
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.5, 0.2],
            [1.5, 0.0],
            [0.0, 1.1],
            [0.9, 0.9],
            [-0.6, -0.3],
            [0.3, -0.7],
            [-1.0, 0.4],
        ];
        for p in samples {
            let got = rhombus_2d(p, b);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn trapezoid_reduces_to_a_box_with_equal_widths() {
        // r1 == r2 is a rectangle of half-width 1, half-height 1.
        assert!((trapezoid_isosceles([0.0, 0.0], 1.0, 1.0, 1.0) - (-1.0)).abs() < 1e-6);
        assert!((trapezoid_isosceles([2.0, 0.0], 1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((trapezoid_isosceles([0.0, 2.0], 1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoid_matches_polygon_reference() {
        let (r1, r2, he) = (1.0f32, 0.5, 0.8);
        // CCW corners: bottom-right, top-right, top-left, bottom-left.
        let verts = [[r1, -he], [r2, he], [-r2, he], [-r1, -he]];
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.6, 0.0],
            [1.4, -0.8],
            [0.0, 1.2],
            [0.0, -1.2],
            [0.9, 0.6],
            [-0.7, -0.4],
            [1.1, 0.2],
        ];
        for p in samples {
            let got = trapezoid_isosceles(p, r1, r2, he);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn arc_is_symmetric_and_banded_on_axis() {
        let ta = std::f32::consts::FRAC_PI_4;
        let sc = [ta.sin(), ta.cos()];
        let (ra, rb) = (1.0f32, 0.1);
        // On the circle centreline (top of the arc): interior by the thickness.
        assert!((arc([0.0, ra], sc, ra, rb) - (-rb)).abs() < 1e-6);
        // Radially outside the band on the +y axis.
        assert!((arc([0.0, ra + 0.3], sc, ra, rb) - (0.3 - rb)).abs() < 1e-6);
        // Mirror symmetry about the y axis.
        for p in [[0.7f32, 0.5], [1.2, -0.3], [0.2, 1.1]] {
            let l = arc([p[0], p[1]], sc, ra, rb);
            let r = arc([-p[0], p[1]], sc, ra, rb);
            assert!((l - r).abs() < 1e-6, "asymmetry at {p:?}: {l} vs {r}");
        }
    }

    #[test]
    fn arc_matches_brute_force_skeleton() {
        // Independent reference: minimum distance to a dense sampling of the arc
        // skeleton (centre line of the band), then inset by the thickness.
        let ta = std::f32::consts::FRAC_PI_3; // 60 degree half-aperture
        let sc = [ta.sin(), ta.cos()];
        let (ra, rb) = (1.2f32, 0.15);
        let brute = |px: f32, py: f32| -> f32 {
            let n = 4000;
            let mut best = f32::INFINITY;
            for i in 0..=n {
                let a = -ta + (2.0 * ta) * (i as f32) / (n as f32);
                // Arc centred on +y: x = sin(a)*ra, y = cos(a)*ra.
                let sx = a.sin() * ra;
                let sy = a.cos() * ra;
                let d = ((px - sx) * (px - sx) + (py - sy) * (py - sy)).sqrt();
                best = best.min(d);
            }
            best - rb
        };
        let samples: [[f32; 2]; 8] = [
            [0.0, 1.2],
            [0.0, 1.6],
            [1.1, 0.6],
            [1.3, 0.1],
            [-0.9, 0.9],
            [0.5, -0.4],
            [1.5, 1.5],
            [0.0, -1.0],
        ];
        for p in samples {
            let got = arc(p, sc, ra, rb);
            let want = brute(p[0], p[1]);
            assert!((got - want).abs() < 5e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn isosceles_triangle_2d_vertices_and_matches_polygon() {
        let (half_base, height) = (0.6f32, 1.2f32);
        // The three vertices lie on the surface.
        for v in [[0.0f32, 0.0], [half_base, height], [-half_base, height]] {
            assert!(isosceles_triangle_2d(v, half_base, height).abs() < 1e-6, "vertex {v:?}");
        }
        let verts = [[0.0, 0.0], [half_base, height], [-half_base, height]];
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.6],
            [0.0, -0.3],
            [0.0, 1.5],
            [0.3, 1.0],
            [0.8, 1.1],
            [-0.5, 0.9],
            [0.2, 0.2],
            [0.0, 1.19],
        ];
        for p in samples {
            let got = isosceles_triangle_2d(p, half_base, height);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn cut_disk_2d_arc_chord_and_corner_regions() {
        let (r, h) = (1.0f32, 0.5f32);
        let w = (r * r - h * h).sqrt();
        // Bottom of the disk sits on the retained arc.
        assert!(cut_disk_2d([0.0, -1.0], r, h).abs() < 1e-6);
        // Straight above the flat top: vertical gap to y = h.
        assert!((cut_disk_2d([0.0, 1.5], r, h) - 1.0).abs() < 1e-6);
        // A corner of the chord lies on the surface.
        assert!(cut_disk_2d([w, h], r, h).abs() < 1e-6);
    }

    #[test]
    fn cut_disk_2d_matches_brute_force_boundary() {
        let (r, h) = (1.0f32, 0.35f32);
        let w = (r * r - h * h).sqrt();
        // Independent reference: nearest distance to the retained boundary (the
        // circular arc with y <= h plus the chord segment at y = h), signed by
        // the disk/half-plane intersection test (keep y <= h).
        let brute = |px: f32, py: f32| -> f32 {
            let mut best = f32::INFINITY;
            let n = 4000;
            for i in 0..=n {
                let a = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
                let (cx, cy) = (r * a.cos(), r * a.sin());
                if cy <= h + 1e-6 {
                    best = best.min(((px - cx).powi(2) + (py - cy).powi(2)).sqrt());
                }
                let t = -w + 2.0 * w * (i as f32) / (n as f32);
                best = best.min(((px - t).powi(2) + (py - h).powi(2)).sqrt());
            }
            let inside = (px * px + py * py).sqrt() < r && py < h;
            if inside { -best } else { best }
        };
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.0, -0.8],
            [0.6, 0.2],
            [0.0, 0.8],
            [1.2, 0.4],
            [-0.9, 0.0],
            [0.5, 0.34],
            [-0.3, -0.5],
        ];
        for p in samples {
            let got = cut_disk_2d(p, r, h);
            let want = brute(p[0], p[1]);
            assert!((got - want).abs() < 5e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn uneven_capsule_2d_caps_and_flank() {
        let (r1, r2, h) = (0.6f32, 0.3f32, 1.0f32);
        // Deepest point of the bottom cap.
        assert!((uneven_capsule_2d([0.0, 0.0], r1, r2, h) - (-r1)).abs() < 1e-6);
        // Deepest point of the top cap.
        assert!((uneven_capsule_2d([0.0, h], r1, r2, h) - (-r2)).abs() < 1e-6);
        // Straight out from the bottom cap along x.
        assert!((uneven_capsule_2d([r1 + 0.5, 0.0], r1, r2, h) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn uneven_capsule_2d_matches_envelope_reference() {
        let (r1, r2, h) = (0.6f32, 0.25f32, 1.1f32);
        // Independent reference: lower envelope of the swept disk family
        // c(t) = (0, t*h), radius(t) = lerp(r1, r2, t), t in [0, 1].
        let brute = |px: f32, py: f32| -> f32 {
            let mut best = f32::INFINITY;
            let n = 6000;
            for i in 0..=n {
                let t = (i as f32) / (n as f32);
                let cy = t * h;
                let rad = r1 + t * (r2 - r1);
                let d = ((px).powi(2) + (py - cy).powi(2)).sqrt() - rad;
                best = best.min(d);
            }
            best
        };
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.5],
            [0.7, 0.0],
            [0.4, 1.1],
            [0.5, 0.5],
            [-0.6, 0.3],
            [0.9, 0.9],
            [0.0, -0.4],
            [0.3, 1.4],
        ];
        for p in samples {
            let got = uneven_capsule_2d(p, r1, r2, h);
            let want = brute(p[0], p[1]);
            assert!((got - want).abs() < 2e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn regular_hexagon_2d_apothem_and_matches_polygon() {
        let apothem = 1.0f32;
        // The flats sit at y = +/-apothem on the surface.
        assert!(regular_hexagon_2d([0.0, apothem], apothem).abs() < 1e-6);
        assert!((regular_hexagon_2d([0.0, 0.0], apothem) - (-apothem)).abs() < 1e-6);
        // Circumradius vertices on the x axis.
        let big_r = apothem / (std::f32::consts::PI / 6.0).cos();
        let mut verts = [[0.0f32; 2]; 6];
        for (k, v) in verts.iter_mut().enumerate() {
            let ang = std::f32::consts::FRAC_PI_3 * k as f32;
            *v = [big_r * ang.cos(), big_r * ang.sin()];
        }
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.5, 0.3],
            [1.3, 0.0],
            [0.0, 1.3],
            [0.9, 0.9],
            [-1.1, 0.4],
            [0.6, -0.8],
            [1.1547, 0.0],
        ];
        for p in samples {
            let got = regular_hexagon_2d(p, apothem);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn equilateral_triangle_2d_matches_polygon() {
        let r = 1.0f32;
        const K: f32 = 1.732_050_8f32;
        // Apex-up vertices; centroid on the origin.
        let verts: [[f32; 2]; 3] = [[r, -r / K], [-r, -r / K], [0.0, 2.0 * r / K]];
        // Centroid distance is -r/sqrt(3); base midpoint and apex are on-surface.
        assert!((equilateral_triangle_2d([0.0, 0.0], r) - (-r / K)).abs() < 1e-5);
        assert!(equilateral_triangle_2d([0.0, -r / K], r).abs() < 1e-5);
        assert!(equilateral_triangle_2d([0.0, 2.0 * r / K], r).abs() < 1e-5);
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.3, 0.2],
            [-0.4, 0.1],
            [0.0, 1.0],
            [1.5, 0.0],
            [-1.2, -0.9],
            [0.0, -1.0],
            [0.8, 0.8],
        ];
        for p in samples {
            let got = equilateral_triangle_2d(p, r);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn regular_pentagon_2d_apothem_and_matches_polygon() {
        let apothem = 1.0f32;
        // Flat top edge sits on y = apothem; centre distance is -apothem.
        assert!(regular_pentagon_2d([0.0, apothem], apothem).abs() < 1e-6);
        assert!((regular_pentagon_2d([0.0, 0.0], apothem) - (-apothem)).abs() < 1e-6);
        let big_r = apothem / (std::f32::consts::PI / 5.0).cos();
        let mut verts = [[0.0f32; 2]; 5];
        for (k, v) in verts.iter_mut().enumerate() {
            let ang = (54.0 + 72.0 * k as f32).to_radians();
            *v = [big_r * ang.cos(), big_r * ang.sin()];
        }
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.4, 0.3],
            [-0.5, 0.2],
            [0.0, 0.9],
            [1.4, 0.0],
            [-1.0, -0.8],
            [0.6, -1.1],
            [0.9, 0.6],
        ];
        for p in samples {
            let got = regular_pentagon_2d(p, apothem);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn regular_octagon_2d_apothem_and_matches_polygon() {
        let apothem = 1.0f32;
        // Flat top edge sits on y = apothem; centre distance is -apothem.
        assert!(regular_octagon_2d([0.0, apothem], apothem).abs() < 1e-6);
        assert!((regular_octagon_2d([0.0, 0.0], apothem) - (-apothem)).abs() < 1e-6);
        let big_r = apothem / (std::f32::consts::PI / 8.0).cos();
        let mut verts = [[0.0f32; 2]; 8];
        for (k, v) in verts.iter_mut().enumerate() {
            let ang = (22.5 + 45.0 * k as f32).to_radians();
            *v = [big_r * ang.cos(), big_r * ang.sin()];
        }
        let samples: [[f32; 2]; 8] = [
            [0.0, 0.0],
            [0.5, 0.4],
            [-0.6, 0.3],
            [0.0, 0.9],
            [1.3, 0.0],
            [-1.1, -0.7],
            [0.7, -1.0],
            [0.9, 0.9],
        ];
        for p in samples {
            let got = regular_octagon_2d(p, apothem);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn hexagram_2d_matches_polygon() {
        let r = 1.0f32;
        const K: f32 = 1.732_050_8f32; // sqrt(3)
        let r_out = 2.0 * r; // outer tip radius
        let r_in = 2.0 * r / K; // inner vertex radius
        // Twelve alternating vertices: inner at 60*k deg, outer at 30 + 60*k deg.
        let mut verts = [[0.0f32; 2]; 12];
        for (k, v) in verts.iter_mut().enumerate() {
            let ang = (30.0 * k as f32).to_radians();
            let rad = if k % 2 == 0 { r_in } else { r_out };
            *v = [rad * ang.cos(), rad * ang.sin()];
        }
        // Centre distance is -2r/sqrt(3) (nearest feature is an inner vertex).
        assert!((hexagram_2d([0.0, 0.0], r) - (-r_in)).abs() < 1e-5);
        // Inner vertex (0 deg) and an outer tip (30 deg) lie on the surface.
        assert!(hexagram_2d([r_in, 0.0], r).abs() < 1e-5);
        assert!(hexagram_2d([r_out * 0.866_025_4, r_out * 0.5], r).abs() < 1e-5);
        let samples: [[f32; 2]; 10] = [
            [0.0, 0.0],
            [0.4, 0.2],
            [-0.5, 0.3],
            [0.0, 1.0],
            [1.9, 0.0],
            [-1.5, 1.0],
            [0.8, -1.4],
            [1.2, 1.2],
            [0.0, 2.1],
            [-2.2, 0.1],
        ];
        for p in samples {
            let got = hexagram_2d(p, r);
            let want = polygon_sdf2(p[0], p[1], &verts);
            assert!((got - want).abs() < 1e-5, "p={p:?} got={got} want={want}");
        }
    }

    // Independent reference for the IQ heart: build a dense boundary polyline
    // (lower flank segment + lobe arc, mirrored) from first principles, then
    // take the min point-to-segment distance with an even-odd winding sign.
    // This shares no branch structure with heart_2d's piecewise closed form.
    fn heart_reference(px: f32, py: f32) -> f32 {
        let rr = 2.0f32.sqrt() / 4.0;
        let (cx, cy) = (0.25f32, 0.75f32);
        let mut right: Vec<[f32; 2]> = Vec::new();
        // Lower-right flank: tip (0,0) -> (0.5,0.5).
        let n_line = 80;
        for i in 0..=n_line {
            let t = i as f32 / n_line as f32;
            right.push([0.5 * t, 0.5 * t]);
        }
        // Right lobe arc: centre (0.25,0.75), r = sqrt(2)/4, (0.5,0.5)->(0,1).
        let a0 = (-45.0f32).to_radians();
        let a1 = (135.0f32).to_radians();
        let n_arc = 240;
        for i in 1..=n_arc {
            let a = a0 + (a1 - a0) * (i as f32 / n_arc as f32);
            right.push([cx + rr * a.cos(), cy + rr * a.sin()]);
        }
        // Close the loop: mirror the right boundary back down the left side.
        let mut poly = right.clone();
        for q in right.iter().rev() {
            poly.push([-q[0], q[1]]);
        }
        // Unsigned distance to the closed polyline.
        let n = poly.len();
        let mut best = f32::INFINITY;
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let e = [b[0] - a[0], b[1] - a[1]];
            let w = [px - a[0], py - a[1]];
            let len2 = e[0] * e[0] + e[1] * e[1];
            let t = if len2 > 0.0 { ((e[0] * w[0] + e[1] * w[1]) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let dx = w[0] - e[0] * t;
            let dy = w[1] - e[1] * t;
            best = best.min(dx * dx + dy * dy);
        }
        // Even-odd ray crossing for the inside test (ray toward +x).
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = (poly[i][0], poly[i][1]);
            let (xj, yj) = (poly[j][0], poly[j][1]);
            if (yi > py) != (yj > py) {
                let xcross = xi + (py - yi) / (yj - yi) * (xj - xi);
                if px < xcross {
                    inside = !inside;
                }
            }
            j = i;
        }
        best.sqrt() * if inside { -1.0 } else { 1.0 }
    }

    #[test]
    fn heart_2d_matches_boundary_reference() {
        // Tip and top cusp lie on the surface; a central point is interior.
        assert!(heart_2d([0.0, 0.0]).abs() < 1e-5);
        assert!(heart_2d([0.0, 1.0]).abs() < 1e-5);
        assert!(heart_2d([0.5, 1.0]).abs() < 1e-5); // right lobe peak on the circle
        assert!(heart_2d([0.0, 0.6]) < 0.0);
        let samples: [[f32; 2]; 12] = [
            [0.0, 0.6],
            [0.3, 0.6],
            [-0.3, 0.6],
            [0.0, 0.0],
            [0.0, 1.3],
            [0.9, 0.9],
            [-0.9, 0.9],
            [0.6, 0.2],
            [-0.6, 0.2],
            [0.2, -0.3],
            [0.0, 0.9],
            [0.45, 0.95],
        ];
        for p in samples {
            let got = heart_2d(p);
            let want = heart_reference(p[0], p[1]);
            assert!((got - want).abs() < 3e-3, "p={p:?} got={got} want={want}");
        }
    }

    // Independent reference for the IQ egg: reconstruct the three-arc boundary
    // (bottom circle, cheek arc, top cap) from first-principles geometry, then
    // take the min point-to-segment distance with an even-odd winding sign.
    fn egg_reference(px: f32, py: f32, ra: f32, rb: f32) -> f32 {
        let r = ra - rb;
        let k = 3.0f32.sqrt();
        let rs = 2.0 * r + rb; // cheek radius
        let mut right: Vec<[f32; 2]> = Vec::new();
        let push_arc = |v: &mut Vec<[f32; 2]>, c: [f32; 2], rad: f32, a0: f32, a1: f32, n: usize, skip_first: bool| {
            for i in 0..=n {
                if skip_first && i == 0 {
                    continue;
                }
                let a = a0 + (a1 - a0) * (i as f32 / n as f32);
                v.push([c[0] + rad * a.cos(), c[1] + rad * a.sin()]);
            }
        };
        // Bottom arc: centre origin, radius ra, from -90deg (tip) to 0deg (widest).
        push_arc(&mut right, [0.0, 0.0], ra, (-90.0f32).to_radians(), 0.0, 120, false);
        // Cheek arc: centre (-r,0), radius rs, from 0deg to 60deg.
        push_arc(&mut right, [-r, 0.0], rs, 0.0, 60.0f32.to_radians(), 120, true);
        // Top cap: centre (0, k*r), radius rb, from 60deg to 90deg.
        push_arc(&mut right, [0.0, k * r], rb, 60.0f32.to_radians(), 90.0f32.to_radians(), 80, true);
        // Close by mirroring the right boundary down the left side.
        let mut poly = right.clone();
        for q in right.iter().rev() {
            poly.push([-q[0], q[1]]);
        }
        let n = poly.len();
        let mut best = f32::INFINITY;
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let e = [b[0] - a[0], b[1] - a[1]];
            let w = [px - a[0], py - a[1]];
            let len2 = e[0] * e[0] + e[1] * e[1];
            let t = if len2 > 0.0 { ((e[0] * w[0] + e[1] * w[1]) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let dx = w[0] - e[0] * t;
            let dy = w[1] - e[1] * t;
            best = best.min(dx * dx + dy * dy);
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = (poly[i][0], poly[i][1]);
            let (xj, yj) = (poly[j][0], poly[j][1]);
            if (yi > py) != (yj > py) {
                let xcross = xi + (py - yi) / (yj - yi) * (xj - xi);
                if px < xcross {
                    inside = !inside;
                }
            }
            j = i;
        }
        best.sqrt() * if inside { -1.0 } else { 1.0 }
    }

    #[test]
    fn egg_2d_matches_arc_reference() {
        let (ra, rb) = (1.0f32, 0.3f32);
        let k = 3.0f32.sqrt();
        let r = ra - rb;
        // Bottom tip, widest point and top cap apex are on the surface.
        assert!(egg_2d([0.0, -ra], ra, rb).abs() < 1e-5);
        assert!(egg_2d([ra, 0.0], ra, rb).abs() < 1e-5);
        assert!(egg_2d([0.0, k * r + rb], ra, rb).abs() < 1e-5);
        // Centre is one bottom-circle radius inside.
        assert!((egg_2d([0.0, 0.0], ra, rb) - (-ra)).abs() < 1e-5);
        let samples: [[f32; 2]; 12] = [
            [0.0, 0.0],
            [0.4, 0.2],
            [-0.4, 0.2],
            [0.0, -0.5],
            [0.0, 1.2],
            [0.9, 0.3],
            [-0.9, 0.3],
            [0.5, 1.0],
            [-0.5, 1.0],
            [1.3, 0.0],
            [0.2, 1.6],
            [0.0, -1.4],
        ];
        for p in samples {
            let got = egg_2d(p, ra, rb);
            let want = egg_reference(p[0], p[1], ra, rb);
            assert!((got - want).abs() < 3e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn polygon_2d_matches_fold_based_primitives() {
        // Cross-check the general polygon SDF against two independent
        // fold-based exact primitives (hexagon + equilateral triangle) and a
        // closed-form rectangle distance, exercising convex outlines.
        let apothem = 1.0f32;
        let big_r = apothem / (std::f32::consts::PI / 6.0).cos();
        let mut hexagon = [[0.0f32; 2]; 6];
        for (k, v) in hexagon.iter_mut().enumerate() {
            let ang = std::f32::consts::FRAC_PI_3 * k as f32;
            *v = [big_r * ang.cos(), big_r * ang.sin()];
        }
        let hw = 1.0f32;
        const K: f32 = 1.732_050_8f32;
        let triangle: [[f32; 2]; 3] = [[hw, -hw / K], [-hw, -hw / K], [0.0, 2.0 * hw / K]];
        // Axis-aligned rectangle, half extents (1.2, 0.7), CCW winding.
        let (ex, ey) = (1.2f32, 0.7f32);
        let rect: [[f32; 2]; 4] = [[ex, ey], [-ex, ey], [-ex, -ey], [ex, -ey]];
        let rect_ref = |p: [f32; 2]| {
            let qx = p[0].abs() - ex;
            let qy = p[1].abs() - ey;
            let ox = qx.max(0.0);
            let oy = qy.max(0.0);
            (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0)
        };
        let samples: [[f32; 2]; 10] = [
            [0.0, 0.0],
            [0.5, 0.3],
            [-0.6, 0.2],
            [1.3, 0.0],
            [0.0, 1.3],
            [0.9, 0.9],
            [-1.1, -0.4],
            [0.3, -0.9],
            [1.5, 1.0],
            [-0.2, 0.6],
        ];
        for p in samples {
            assert!(
                (polygon_2d(p, &hexagon) - regular_hexagon_2d(p, apothem)).abs() < 1e-5,
                "hexagon mismatch at {p:?}"
            );
            assert!(
                (polygon_2d(p, &triangle) - equilateral_triangle_2d(p, hw)).abs() < 1e-5,
                "triangle mismatch at {p:?}"
            );
            assert!(
                (polygon_2d(p, &rect) - rect_ref(p)).abs() < 1e-5,
                "rect mismatch at {p:?}"
            );
        }
        // Winding independence: reversing the vertex order keeps the result.
        let mut rev = hexagon;
        rev.reverse();
        for p in samples {
            assert!((polygon_2d(p, &hexagon) - polygon_2d(p, &rev)).abs() < 1e-6);
        }
        // Degenerate inputs: empty slice is +inf, a single vertex is its radius.
        assert_eq!(polygon_2d([0.0, 0.0], &[]), f32::INFINITY);
        assert!((polygon_2d([3.0, 4.0], &[[0.0, 0.0]]) - 5.0).abs() < 1e-6);
    }

    // Independent reference for the per-corner rounded box: build the exact
    // boundary (four straight edges + four quarter-circle corner arcs) from
    // first-principles geometry, then take the min point-to-segment distance
    // with an even-odd winding sign. Shares no branch structure with the
    // closed-form rounded_box_2d.
    fn rounded_box_reference(px: f32, py: f32, bx: f32, by: f32, radii: [f32; 4]) -> f32 {
        let (rtr, rbr, rtl, rbl) = (radii[0], radii[1], radii[2], radii[3]);
        let mut poly: Vec<[f32; 2]> = Vec::new();
        let push_arc = |v: &mut Vec<[f32; 2]>, c: [f32; 2], rad: f32, a0: f32, a1: f32| {
            let n = 64usize;
            for i in 0..=n {
                let a = a0 + (a1 - a0) * (i as f32 / n as f32);
                v.push([c[0] + rad * a.cos(), c[1] + rad * a.sin()]);
            }
        };
        // CCW from the bottom of the right edge.
        // Right edge bottom -> top.
        poly.push([bx, -(by - rbr)]);
        poly.push([bx, by - rtr]);
        // Top-right arc 0 -> 90.
        push_arc(&mut poly, [bx - rtr, by - rtr], rtr, 0.0, 90.0f32.to_radians());
        // Top edge.
        poly.push([-(bx - rtl), by]);
        // Top-left arc 90 -> 180.
        push_arc(&mut poly, [-(bx - rtl), by - rtl], rtl, 90.0f32.to_radians(), std::f32::consts::PI);
        // Left edge.
        poly.push([-bx, -(by - rbl)]);
        // Bottom-left arc 180 -> 270.
        push_arc(&mut poly, [-(bx - rbl), -(by - rbl)], rbl, std::f32::consts::PI, 270.0f32.to_radians());
        // Bottom edge.
        poly.push([bx - rbr, -by]);
        // Bottom-right arc 270 -> 360.
        push_arc(&mut poly, [bx - rbr, -(by - rbr)], rbr, 270.0f32.to_radians(), 360.0f32.to_radians());
        let n = poly.len();
        let mut best = f32::INFINITY;
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let e = [b[0] - a[0], b[1] - a[1]];
            let w = [px - a[0], py - a[1]];
            let len2 = e[0] * e[0] + e[1] * e[1];
            let t = if len2 > 0.0 { ((e[0] * w[0] + e[1] * w[1]) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let dx = w[0] - e[0] * t;
            let dy = w[1] - e[1] * t;
            best = best.min(dx * dx + dy * dy);
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = (poly[i][0], poly[i][1]);
            let (xj, yj) = (poly[j][0], poly[j][1]);
            if (yi > py) != (yj > py) {
                let xcross = xi + (py - yi) / (yj - yi) * (xj - xi);
                if px < xcross {
                    inside = !inside;
                }
            }
            j = i;
        }
        best.sqrt() * if inside { -1.0 } else { 1.0 }
    }

    #[test]
    fn rounded_box_2d_matches_boundary_reference() {
        let (bx, by) = (1.2f32, 0.8f32);
        let radii = [0.3f32, 0.2, 0.4, 0.1]; // TR, BR, TL, BL
        // With zero radii it collapses to a plain box distance.
        let box_ref = |p: [f32; 2]| {
            let qx = p[0].abs() - bx;
            let qy = p[1].abs() - by;
            (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0)
        };
        for p in [[0.0, 0.0], [1.5, 0.0], [0.0, 1.1], [0.5, 0.5], [-1.3, -0.9]] {
            assert!((rounded_box_2d(p, [bx, by], [0.0; 4]) - box_ref(p)).abs() < 1e-6, "box collapse at {p:?}");
        }
        // Centre is one short half-extent inside (0.8), minus nothing (corner far).
        assert!((rounded_box_2d([0.0, 0.0], [bx, by], radii) - (-by)).abs() < 1e-5);
        let samples: [[f32; 2]; 12] = [
            [0.0, 0.0],
            [1.0, 0.6],
            [-1.0, 0.6],
            [1.0, -0.6],
            [-1.0, -0.6],
            [1.4, 0.9],
            [-1.4, 0.9],
            [0.0, 0.85],
            [1.25, 0.0],
            [-1.25, 0.0],
            [0.7, -0.7],
            [-0.9, 0.75],
        ];
        for p in samples {
            let got = rounded_box_2d(p, [bx, by], radii);
            let want = rounded_box_reference(p[0], p[1], bx, by, radii);
            assert!((got - want).abs() < 3e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn vesica_segment_tips_center_and_bulge() {
        let a = [-1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let w = 0.5f32;
        // Both endpoints are tips lying on the surface.
        assert!(vesica_segment(a, a, b, w).abs() < 1e-5);
        assert!(vesica_segment(b, a, b, w).abs() < 1e-5);
        // The midpoint is the deepest interior point, one bulge half-width in.
        assert!((vesica_segment([0.0, 0.0, 0.0], a, b, w) - (-w)).abs() < 1e-5);
        // A point on the midpoint bulge lies on the surface regardless of the
        // perpendicular direction chosen.
        assert!(vesica_segment([0.0, w, 0.0], a, b, w).abs() < 1e-5);
        assert!(vesica_segment([0.0, 0.0, w], a, b, w).abs() < 1e-5);
    }

    // Independent reference: reduce to axial/radial coordinates about the axis
    // (valid because the vesica is a surface of revolution, so the nearest point
    // lies in the query's own meridian half-plane), then take the 2D signed
    // distance to the lens cross-section built from first principles as two
    // circular arcs. Shares no branch structure with the closed form.
    fn vesica_segment_reference(p: [f32; 3], a: [f32; 3], b: [f32; 3], w: f32) -> f32 {
        let c = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
        let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let l = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
        let v = [ba[0] / l, ba[1] / l, ba[2] / l];
        let pc = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let y = pc[0] * v[0] + pc[1] * v[1] + pc[2] * v[2];
        let perp = [pc[0] - y * v[0], pc[1] - y * v[1], pc[2] - y * v[2]];
        let rho = (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt();
        // Lens cross-section in the (axial, radial) plane.
        let r = 0.5 * l;
        let d = 0.5 * (r * r - w * w) / w;
        let rr = d + w; // generating-arc radius
        let a0 = (d / rr).asin(); // tip angle from each arc centre
        let span = std::f32::consts::PI - 2.0 * a0;
        let n = 128usize;
        let mut poly: Vec<[f32; 2]> = Vec::new();
        // Upper arc centred at (0, -d): tip (r,0) over (0,w) to (-r,0).
        for i in 0..=n {
            let th = a0 + span * (i as f32 / n as f32);
            poly.push([rr * th.cos(), -d + rr * th.sin()]);
        }
        // Lower arc centred at (0, d): back from (-r,0) over (0,-w) to (r,0).
        for i in 0..=n {
            let th = (std::f32::consts::PI - a0) - span * (i as f32 / n as f32);
            poly.push([rr * th.cos(), d - rr * th.sin()]);
        }
        let (px, py) = (y, rho);
        let m = poly.len();
        let mut best = f32::INFINITY;
        for i in 0..m {
            let s = poly[i];
            let e = poly[(i + 1) % m];
            let ex = e[0] - s[0];
            let ey = e[1] - s[1];
            let wx = px - s[0];
            let wy = py - s[1];
            let len2 = ex * ex + ey * ey;
            let t = if len2 > 0.0 { ((ex * wx + ey * wy) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let dx = wx - ex * t;
            let dy = wy - ey * t;
            best = best.min(dx * dx + dy * dy);
        }
        let mut inside = false;
        let mut j = m - 1;
        for i in 0..m {
            let (xi, yi) = (poly[i][0], poly[i][1]);
            let (xj, yj) = (poly[j][0], poly[j][1]);
            if (yi > py) != (yj > py) {
                let xcross = xi + (py - yi) / (yj - yi) * (xj - xi);
                if px < xcross {
                    inside = !inside;
                }
            }
            j = i;
        }
        best.sqrt() * if inside { -1.0 } else { 1.0 }
    }

    #[test]
    fn vesica_segment_matches_revolution_reference() {
        // Oriented, off-origin axis to exercise the full reduction.
        let a = [0.4f32, -0.3, 0.2];
        let b = [-0.6f32, 0.9, 0.5];
        let w = 0.35f32;
        let samples: [[f32; 3]; 12] = [
            [0.0, 0.0, 0.0],
            [0.4, -0.3, 0.2],
            [-0.6, 0.9, 0.5],
            [-0.1, 0.3, 0.35],
            [0.2, 0.1, -0.3],
            [-0.3, 0.6, 0.9],
            [0.6, -0.5, 0.1],
            [-0.1, 0.3, -0.2],
            [0.9, 0.2, 0.4],
            [-0.9, 1.2, 0.6],
            [-0.05, 0.3, 0.33],
            [0.15, 0.0, 0.5],
        ];
        for p in samples {
            let got = vesica_segment(p, a, b, w);
            let want = vesica_segment_reference(p, a, b, w);
            assert!((got - want).abs() < 3e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn circle_2d_matches_radial_distance_and_boundary() {
        let r = 1.3f32;
        // Analytic radial checks: centre, inside, on-rim and outside.
        assert!((circle_2d([0.0, 0.0], r) - (-r)).abs() < 1e-6);
        assert!((circle_2d([r, 0.0], r)).abs() < 1e-6);
        assert!((circle_2d([0.0, -r], r)).abs() < 1e-6);
        assert!((circle_2d([2.0, 0.0], r) - (2.0 - r)).abs() < 1e-6);
        // Independent reference: distance to a dense boundary polyline with an
        // inside test by radius, sharing no code with the closed form.
        let n = 512usize;
        let ring: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32 / n as f32);
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        for p in [[0.3f32, 0.2], [1.0, -0.9], [-1.6, 0.4], [0.0, 1.9], [-0.5, -0.5]] {
            let mut best = f32::INFINITY;
            for i in 0..n {
                let s = ring[i];
                let e = ring[(i + 1) % n];
                let ex = e[0] - s[0];
                let ey = e[1] - s[1];
                let wx = p[0] - s[0];
                let wy = p[1] - s[1];
                let t = ((ex * wx + ey * wy) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
                let dx = wx - ex * t;
                let dy = wy - ey * t;
                best = best.min(dx * dx + dy * dy);
            }
            let sign = if p[0] * p[0] + p[1] * p[1] < r * r { -1.0 } else { 1.0 };
            let want = best.sqrt() * sign;
            let got = circle_2d(p, r);
            assert!((got - want).abs() < 2e-3, "p={p:?} got={got} want={want}");
        }
    }

    #[test]
    fn star5_2d_matches_decagon_reference() {
        // A regular five-pointed star is a 10-gon with alternating outer/inner
        // radii; build it from first principles and cross-check sdPolygon.
        let r = 1.0f32;
        let rf = 0.45f32;
        let mut verts: Vec<[f32; 2]> = Vec::new();
        for k in 0..5 {
            let ao = (90.0 + 72.0 * k as f32).to_radians();
            verts.push([r * ao.cos(), r * ao.sin()]);
            let ai = (90.0 + 36.0 + 72.0 * k as f32).to_radians();
            verts.push([r * rf * ai.cos(), r * rf * ai.sin()]);
        }
        // Top outer tip and an inner vertex sit on the surface.
        assert!(star5_2d([0.0, r], r, rf).abs() < 1e-4);
        let ai = (90.0f32 + 36.0).to_radians();
        assert!(star5_2d([r * rf * ai.cos(), r * rf * ai.sin()], r, rf).abs() < 1e-4);
        let samples: [[f32; 2]; 14] = [
            [0.0, 0.0],
            [0.0, 0.9],
            [0.0, 1.3],
            [0.6, 0.1],
            [-0.6, 0.1],
            [0.3, -0.8],
            [-0.3, -0.8],
            [0.9, 0.9],
            [-0.9, -0.9],
            [0.2, 0.2],
            [1.2, 0.0],
            [-1.2, 0.0],
            [0.0, -1.1],
            [0.45, 0.45],
        ];
        for s in samples {
            let got = star5_2d(s, r, rf);
            let want = polygon_sdf2(s[0], s[1], &verts);
            assert!((got - want).abs() < 1e-4, "s={s:?} got={got} want={want}");
        }
    }

    #[test]
    fn pentagram_2d_matches_golden_decagon_reference() {
        let r = 1.0f32;
        let rf = (3.0f32 - 5.0f32.sqrt()) / 2.0; // 1 / phi^2
        let mut verts: Vec<[f32; 2]> = Vec::new();
        for k in 0..5 {
            let ao = (90.0 + 72.0 * k as f32).to_radians();
            verts.push([r * ao.cos(), r * ao.sin()]);
            let ai = (90.0 + 36.0 + 72.0 * k as f32).to_radians();
            verts.push([r * rf * ai.cos(), r * rf * ai.sin()]);
        }
        // Centre lies one inner radius from the nearest concave vertex.
        assert!((pentagram_2d([0.0, 0.0], r) - (-r * rf)).abs() < 1e-4);
        // Top outer tip is on the surface.
        assert!(pentagram_2d([0.0, r], r).abs() < 1e-4);
        let samples: [[f32; 2]; 12] = [
            [0.0, 0.0],
            [0.0, 0.8],
            [0.0, 1.2],
            [0.5, 0.2],
            [-0.5, 0.2],
            [0.3, -0.6],
            [-0.3, -0.6],
            [0.9, 0.9],
            [-1.1, 0.0],
            [1.1, 0.0],
            [0.0, -0.9],
            [0.25, 0.25],
        ];
        for s in samples {
            let got = pentagram_2d(s, r);
            let want = polygon_sdf2(s[0], s[1], &verts);
            assert!((got - want).abs() < 1e-4, "s={s:?} got={got} want={want}");
        }
    }

    #[test]
    fn vesica_2d_matches_two_arc_reference() {
        let r = 1.0f32;
        let off = 0.6f32;
        let b = (r * r - off * off).sqrt();
        // Analytic features: centre, both lateral cusps and the x-extent tips.
        assert!((vesica_2d([0.0, 0.0], r, off) - (-(r - off))).abs() < 1e-6);
        assert!(vesica_2d([0.0, b], r, off).abs() < 1e-6);
        assert!(vesica_2d([0.0, -b], r, off).abs() < 1e-6);
        assert!(vesica_2d([r - off, 0.0], r, off).abs() < 1e-6);
        // Independent reference: two circular arcs (centres (-off,0) and
        // (+off,0)) as a boundary polyline, inside = inside both disks.
        let a_r = b.atan2(off);
        let n = 400usize;
        let mut poly: Vec<[f32; 2]> = Vec::new();
        for i in 0..=n {
            let a = -a_r + 2.0 * a_r * (i as f32 / n as f32);
            poly.push([-off + r * a.cos(), r * a.sin()]);
        }
        let a_l = std::f32::consts::PI - a_r;
        for i in 0..=n {
            let a = a_l + 2.0 * (std::f32::consts::PI - a_l) * (i as f32 / n as f32);
            poly.push([off + r * a.cos(), r * a.sin()]);
        }
        let m = poly.len();
        let reference = |px: f32, py: f32| -> f32 {
            let mut best = f32::INFINITY;
            for i in 0..m {
                let s = poly[i];
                let e = poly[(i + 1) % m];
                let ex = e[0] - s[0];
                let ey = e[1] - s[1];
                let wx = px - s[0];
                let wy = py - s[1];
                let t = ((ex * wx + ey * wy) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
                let dx = wx - ex * t;
                let dy = wy - ey * t;
                best = best.min(dx * dx + dy * dy);
            }
            let in_a = (px + off) * (px + off) + py * py <= r * r;
            let in_b = (px - off) * (px - off) + py * py <= r * r;
            best.sqrt() * if in_a && in_b { -1.0 } else { 1.0 }
        };
        let samples: [[f32; 2]; 12] = [
            [0.0, 0.0],
            [0.2, 0.3],
            [-0.2, -0.3],
            [0.5, 0.0],
            [0.0, 0.9],
            [0.9, 0.9],
            [-0.9, 0.4],
            [0.35, -0.5],
            [1.2, 0.0],
            [0.0, 1.1],
            [-0.3, 0.7],
            [0.25, 0.25],
        ];
        for s in samples {
            let got = vesica_2d(s, r, off);
            let want = reference(s[0], s[1]);
            assert!((got - want).abs() < 5e-3, "s={s:?} got={got} want={want}");
        }
    }

    // Independent reference for `triangle_2d`: the unsigned distance is the
    // smallest point-to-segment distance over the three edges, and the sign is
    // taken from a winding-independent half-plane inside test (all three edge
    // cross products share one sign iff the point is inside). This shares no
    // logic with the IQ component-wise-min formula under test.
    fn triangle_ref(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
        fn seg(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
            let pa = [p[0] - a[0], p[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1]))
                .clamp(0.0, 1.0);
            ((pa[0] - ba[0] * h).powi(2) + (pa[1] - ba[1] * h).powi(2)).sqrt()
        }
        let d = seg(p, a, b).min(seg(p, b, c)).min(seg(p, c, a));
        let cr = |u: [f32; 2], v: [f32; 2], w: [f32; 2]| {
            (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
        };
        let c0 = cr(a, b, p);
        let c1 = cr(b, c, p);
        let c2 = cr(c, a, p);
        let inside = (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0)
            || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0);
        if inside { -d } else { d }
    }

    #[test]
    fn triangle_2d_closed_form_points() {
        // Right triangle with legs along the axes: (0,0),(4,0),(0,3).
        let a = [0.0f32, 0.0];
        let b = [4.0f32, 0.0];
        let c = [0.0f32, 3.0];
        // Centroid is interior; distance to nearest edge (the hypotenuse, line
        // 3x+4y-12=0 at distance |3*4/3+4*1-12|/5 = 0.8) -> negative 0.8.
        let g = [4.0 / 3.0, 1.0];
        assert!((triangle_2d(g, a, b, c) - (-0.8)).abs() < 1e-5);
        // Point 2 units to the left of the vertical leg is outside at distance 2.
        assert!((triangle_2d([-2.0, 1.0], a, b, c) - 2.0).abs() < 1e-6);
        // Point directly below the base is outside at its vertical distance.
        assert!((triangle_2d([1.0, -1.5], a, b, c) - 1.5).abs() < 1e-6);
        // On an edge midpoint the field is zero.
        assert!(triangle_2d([2.0, 0.0], a, b, c).abs() < 1e-6);
    }

    #[test]
    fn triangle_2d_matches_half_plane_reference() {
        // A few fixed non-degenerate triangles, both windings, sampled on a grid.
        let tris: [[[f32; 2]; 3]; 4] = [
            [[-2.0, -1.0], [3.0, -0.5], [0.5, 2.5]],
            [[0.5, 2.5], [3.0, -0.5], [-2.0, -1.0]], // reversed winding
            [[-3.0, 2.0], [-1.0, -3.0], [2.5, 0.0]],
            [[1.0, 1.0], [4.0, 1.5], [2.0, 4.0]],
        ];
        for tri in tris.iter() {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            let mut i = -40i32;
            while i <= 40 {
                let mut j = -40i32;
                while j <= 40 {
                    let p = [i as f32 * 0.15, j as f32 * 0.15];
                    let got = triangle_2d(p, a, b, c);
                    let want = triangle_ref(p, a, b, c);
                    assert!(
                        (got - want).abs() < 1e-4,
                        "triangle mismatch at {p:?}: got {got} want {want}"
                    );
                    j += 1;
                }
                i += 1;
            }
        }
    }

    #[test]
    fn box_2d_edges_corners_and_interior() {
        let b = [2.0f32, 1.0];
        // Outside past an edge: horizontal gap only.
        assert!((box_2d([5.0, 0.0], b) - 3.0).abs() < 1e-6);
        // Outside past a corner: diagonal gap (3-4-5 style).
        assert!((box_2d([2.0 + 3.0, 1.0 + 4.0], b) - 5.0).abs() < 1e-6);
        // On an edge the field is zero.
        assert!(box_2d([2.0, 0.5], b).abs() < 1e-6);
        // Interior: negative distance to the nearest edge.
        assert!((box_2d([0.0, 0.0], b) - (-1.0)).abs() < 1e-6);
        assert!((box_2d([1.5, 0.0], b) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn box_2d_matches_rounded_box_zero_radii() {
        // Sharp box must equal the rounded box with vanishing corner radii.
        let b = [1.3f32, 2.1];
        let mut i = -40i32;
        while i <= 40 {
            let mut j = -40i32;
            while j <= 40 {
                let p = [i as f32 * 0.1, j as f32 * 0.1];
                let got = box_2d(p, b);
                let want = rounded_box_2d(p, b, [0.0; 4]);
                assert!((got - want).abs() < 1e-6, "box mismatch at {p:?}: {got} vs {want}");
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn vertical_capsule_axis_caps_and_flank() {
        let h = 3.0f32;
        let r = 1.0f32;
        // Radially out from the mid-axis: distance is radial gap.
        assert!((vertical_capsule([2.0, 1.5, 0.0], h, r) - 1.0).abs() < 1e-6);
        assert!((vertical_capsule([0.0, 1.5, 2.0], h, r) - 1.0).abs() < 1e-6);
        // Above the top cap along the axis: spherical cap distance.
        assert!((vertical_capsule([0.0, h + 2.0, 0.0], h, r) - 1.0).abs() < 1e-6);
        // Below the bottom cap along the axis.
        assert!((vertical_capsule([0.0, -2.0, 0.0], h, r) - 1.0).abs() < 1e-6);
        // On the axis inside the segment: negative radius.
        assert!((vertical_capsule([0.0, 1.0, 0.0], h, r) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn vertical_capsule_matches_general_capsule() {
        // Must equal the general capsule along the (0,0,0)->(0,h,0) segment.
        let h = 2.5f32;
        let r = 0.75f32;
        let mut i = -30i32;
        while i <= 30 {
            let mut j = -30i32;
            while j <= 30 {
                let mut k = -30i32;
                while k <= 30 {
                    let p = [i as f32 * 0.2, j as f32 * 0.2, k as f32 * 0.2];
                    let got = vertical_capsule(p, h, r);
                    let want = capsule(p, [0.0, 0.0, 0.0], [0.0, h, 0.0], r);
                    assert!((got - want).abs() < 1e-5, "mismatch at {p:?}: {got} vs {want}");
                    k += 10;
                }
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn annulus_2d_band_edges_and_interior() {
        let r = 3.0f32;
        let t = 0.5f32;
        // On the mid-line: deepest interior, negative half-width.
        assert!((annulus_2d([3.0, 0.0], r, t) - (-0.5)).abs() < 1e-6);
        // On the outer and inner edges: zero.
        assert!(annulus_2d([3.5, 0.0], r, t).abs() < 1e-6);
        assert!(annulus_2d([0.0, 2.5], r, t).abs() < 1e-6);
        // Outside the outer edge and inside the hole: positive gap.
        assert!((annulus_2d([5.0, 0.0], r, t) - 1.5).abs() < 1e-6);
        assert!((annulus_2d([0.0, 0.0], r, t) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn annulus_2d_matches_two_circle_boundary_reference() {
        // Independent reference: unsigned distance is the min point-to-segment
        // distance over polyline approximations of the inner (r-t) and outer
        // (r+t) circles; sign comes from radial band membership. This shares no
        // logic with the analytic abs-of-circle formula under test.
        let r = 2.75f32;
        let t = 0.6f32;
        const N: usize = 2048;
        let circle = |rad: f32| -> Vec<[f32; 2]> {
            (0..N)
                .map(|i| {
                    let a = std::f32::consts::TAU * (i as f32) / (N as f32);
                    [rad * a.cos(), rad * a.sin()]
                })
                .collect()
        };
        let seg = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
            let pa = [p[0] - a[0], p[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1]))
                .clamp(0.0, 1.0);
            ((pa[0] - ba[0] * h).powi(2) + (pa[1] - ba[1] * h).powi(2)).sqrt()
        };
        let outer = circle(r + t);
        let inner = circle(r - t);
        let reference = |p: [f32; 2]| -> f32 {
            let mut d = f32::INFINITY;
            for poly in [&outer, &inner] {
                for i in 0..N {
                    d = d.min(seg(p, poly[i], poly[(i + 1) % N]));
                }
            }
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if rho >= r - t && rho <= r + t { -d } else { d }
        };
        let mut i = -24i32;
        while i <= 24 {
            let mut j = -24i32;
            while j <= 24 {
                let p = [i as f32 * 0.25, j as f32 * 0.25];
                let got = annulus_2d(p, r, t);
                let want = reference(p);
                assert!((got - want).abs() < 5e-3, "annulus mismatch at {p:?}: {got} vs {want}");
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn capsule_2d_closed_forms_and_brute_force() {
        let a = [-2.0f32, 0.0];
        let b = [2.0f32, 0.0];
        let r = 0.75f32;
        // Beside the shaft: perpendicular gap minus radius.
        assert!((capsule_2d([0.0, 2.0], a, b, r) - (2.0 - r)).abs() < 1e-6);
        // Past an end cap along the axis: axial gap minus radius.
        assert!((capsule_2d([4.0, 0.0], a, b, r) - (2.0 - r)).abs() < 1e-6);
        // On the shaft centre: negative radius (deep interior).
        assert!((capsule_2d([0.0, 0.0], a, b, r) - (-r)).abs() < 1e-6);
        // Independent reference: min distance to a densely sampled segment - r.
        let seg_a = [-1.0f32, -0.5];
        let seg_b = [1.5f32, 2.0];
        let reference = |p: [f32; 2]| -> f32 {
            let mut d = f32::INFINITY;
            let m = 4000i32;
            let mut i = 0i32;
            while i <= m {
                let t = i as f32 / m as f32;
                let q = [seg_a[0] + (seg_b[0] - seg_a[0]) * t, seg_a[1] + (seg_b[1] - seg_a[1]) * t];
                d = d.min(((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt());
                i += 1;
            }
            d - r
        };
        let mut gi = -20i32;
        while gi <= 20 {
            let mut gj = -20i32;
            while gj <= 20 {
                let p = [gi as f32 * 0.3, gj as f32 * 0.3];
                let got = capsule_2d(p, seg_a, seg_b, r);
                let want = reference(p);
                assert!((got - want).abs() < 2e-3, "capsule_2d mismatch at {p:?}: {got} vs {want}");
                gj += 1;
            }
            gi += 1;
        }
    }

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

    // Brute-force reference: minimum distance from `p` to a dense sampling of a
    // convex quad, splitting it into triangles (a,b,c) and (a,c,d).
    fn quad_brute_force(
        p: [f32; 3],
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
    ) -> f32 {
        tri_brute_force(p, a, b, c).min(tri_brute_force(p, a, c, d))
    }

    #[test]
    fn quad_zero_on_surface_and_vertices() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [2.0, 2.0, 0.0];
        let d = [0.0, 2.0, 0.0];
        assert!(quad_sdf(a, a, b, c, d).abs() < 1e-6);
        assert!(quad_sdf(c, a, b, c, d).abs() < 1e-6);
        // Centre of the unit square patch lies on it.
        assert!(quad_sdf([1.0, 1.0, 0.0], a, b, c, d).abs() < 1e-6);
    }

    #[test]
    fn quad_face_region_is_perpendicular_offset() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [2.0, 2.0, 0.0];
        let d = [0.0, 2.0, 0.0];
        // Straight above the centre: pure perpendicular height.
        assert!((quad_sdf([1.0, 1.0, 2.5], a, b, c, d) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn quad_vertex_region_is_vertex_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [2.0, 2.0, 0.0];
        let d = [0.0, 2.0, 0.0];
        // Diagonally past the a corner in-plane: nearest feature is vertex a.
        assert!((quad_sdf([-1.0, -1.0, 0.0], a, b, c, d) - length2([1.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn quad_matches_brute_force_general_orientation() {
        // Tilted, off-origin planar convex quad (built from two coplanar basis
        // vectors so the four corners stay exactly coplanar).
        let o = [0.2, -0.3, 0.4];
        let u = [1.3, 0.1, -0.5];
        let v = [-0.4, 1.1, 0.6];
        let a = o;
        let b = [o[0] + u[0], o[1] + u[1], o[2] + u[2]];
        let c = [o[0] + u[0] + v[0], o[1] + u[1] + v[1], o[2] + u[2] + v[2]];
        let d = [o[0] + v[0], o[1] + v[1], o[2] + v[2]];
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, -0.5],
            [0.8, 0.4, 0.9],
            [2.0, 2.0, -1.0],
            [0.3, 1.5, 1.8],
        ] {
            let got = quad_sdf(p, a, b, c, d);
            let reference = quad_brute_force(p, a, b, c, d);
            assert!(
                (got - reference).abs() < 3e-3,
                "point {p:?}: got {got}, reference {reference}"
            );
        }
    }

    // Brute-force reference for a capped-cone frustum: minimum distance from `p`
    // to a dense sampling of the lateral surface and both circular caps.
    fn frustum_brute_force(
        p: [f32; 3],
        a: [f32; 3],
        b: [f32; 3],
        ra: f32,
        rb: f32,
    ) -> f32 {
        // Orthonormal basis perpendicular to the axis a->b.
        let axis = {
            let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let seed = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = {
            let c = [
                axis[1] * seed[2] - axis[2] * seed[1],
                axis[2] * seed[0] - axis[0] * seed[2],
                axis[0] * seed[1] - axis[1] * seed[0],
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            [c[0] / l, c[1] / l, c[2] / l]
        };
        let w = [
            axis[1] * u[2] - axis[2] * u[1],
            axis[2] * u[0] - axis[0] * u[2],
            axis[0] * u[1] - axis[1] * u[0],
        ];
        let dist = |q: [f32; 3]| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        };
        let mut best = f32::INFINITY;
        const NT: usize = 240;
        const NA: usize = 240;
        for i in 0..=NA {
            let t = i as f32 / NA as f32;
            let center = [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ];
            let r = ra + (rb - ra) * t;
            for j in 0..NT {
                let ang = std::f32::consts::TAU * j as f32 / NT as f32;
                let (sa, ca) = (ang.sin(), ang.cos());
                let q = [
                    center[0] + r * (ca * u[0] + sa * w[0]),
                    center[1] + r * (ca * u[1] + sa * w[1]),
                    center[2] + r * (ca * u[2] + sa * w[2]),
                ];
                let d = dist(q);
                if d < best {
                    best = d;
                }
            }
        }
        // Interior of both caps.
        for &(center, rad) in &[(a, ra), (b, rb)] {
            for i in 0..=60 {
                let rr = rad * i as f32 / 60.0;
                for j in 0..120 {
                    let ang = std::f32::consts::TAU * j as f32 / 120.0;
                    let (sa, ca) = (ang.sin(), ang.cos());
                    let q = [
                        center[0] + rr * (ca * u[0] + sa * w[0]),
                        center[1] + rr * (ca * u[1] + sa * w[1]),
                        center[2] + rr * (ca * u[2] + sa * w[2]),
                    ];
                    let d = dist(q);
                    if d < best {
                        best = d;
                    }
                }
            }
        }
        best
    }

    #[test]
    fn capped_cone_segment_degenerates_to_capped_cylinder() {
        // Equal radii along +y from the origin: must agree with the axis-aligned
        // capped cylinder (whose segment runs y in [-h, h], centred at origin).
        // Build the same shape as a segment and compare at several queries.
        let h = 1.5f32;
        let r = 0.6f32;
        let a = [0.0, -h, 0.0];
        let b = [0.0, h, 0.0];
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.9, 1.9, 0.0],
            [0.3, -0.4, 0.2],
        ] {
            let seg = capped_cone_segment(p, a, b, r, r);
            let cyl = capped_cylinder(p, h, r);
            assert!(
                (seg - cyl).abs() < 1e-5,
                "point {p:?}: segment {seg}, cylinder {cyl}"
            );
        }
    }

    #[test]
    fn capped_cone_segment_matches_brute_force_general() {
        // A tilted, off-origin frustum with distinct radii.
        let a = [0.2, -0.3, 0.1];
        let b = [1.4, 1.1, -0.6];
        let (ra, rb) = (0.7f32, 0.3f32);
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-0.5, 0.5, 0.5],
            [0.8, 0.4, 0.2],
            [2.0, 2.0, -1.5],
            [0.6, 0.1, -0.3],
        ] {
            let got = capped_cone_segment(p, a, b, ra, rb);
            let reference = frustum_brute_force(p, a, b, ra, rb);
            // Exterior points: brute force converges from above, so allow a
            // one-sided sampling slack; magnitude must still match tightly.
            assert!(
                (got.abs() - reference).abs() < 6e-3,
                "point {p:?}: got {got}, reference {reference}"
            );
        }
    }

    // Deterministic rotation matrix (ZYX Euler, baked) used to lift axis-aligned
    // golden shapes into a general orientation for the arbitrary-endpoint tests.
    fn rot(v: [f32; 3]) -> [f32; 3] {
        // Angles ~ (0.6, -0.4, 0.9) rad; matrix elements precomputed to f64 and
        // narrowed, so the test carries no runtime trigonometry.
        const M: [[f32; 3]; 3] = [
            [0.572_540_7, -0.783_188_5, 0.242_513_7],
            [0.721_491_9, 0.340_797_2, -0.602_749_3],
            [0.389_418_3, 0.520_070_2, 0.760_184_4],
        ];
        [
            M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
        ]
    }

    #[test]
    fn round_cone_segment_degenerates_to_axis_aligned() {
        // a at origin (r1), b at (0, h, 0) (r2): must reproduce round_cone_sdf
        // point-for-point across cap, flank, and interior regions.
        let (r1, r2, h) = (1.0f32, 0.4f32, 2.0f32);
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, h, 0.0];
        for &p in &[
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, -0.5, 0.7],
        ] {
            let seg = round_cone_segment(p, a, b, r1, r2);
            let golden = round_cone_sdf(p, r1, r2, h);
            assert!(
                (seg - golden).abs() < 1e-5,
                "point {p:?}: segment {seg}, golden {golden}"
            );
        }
    }

    #[test]
    fn round_cone_segment_is_rigid_motion_invariant() {
        // Lift the canonical cone by a rotation + translation and verify the
        // segment SDF of the transformed query matches the axis-aligned golden
        // of the untransformed query (a correct Euclidean SDF is isometry
        // invariant). This proves general-orientation correctness exactly,
        // leaning on the already-verified round_cone_sdf.
        let (r1, r2, h) = (1.2f32, 0.5f32, 1.7f32);
        let tr = [0.3, -0.7, 0.4];
        let a = tr;
        let b = {
            let rb = rot([0.0, h, 0.0]);
            [rb[0] + tr[0], rb[1] + tr[1], rb[2] + tr[2]]
        };
        for &p in &[
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            [2.0, 0.5, 0.0],
            [1.0, 1.0, 0.3],
            [0.4, 1.3, -0.6],
            [-1.0, 0.2, 0.8],
        ] {
            let golden = round_cone_sdf(p, r1, r2, h);
            let rp = rot(p);
            let tp = [rp[0] + tr[0], rp[1] + tr[1], rp[2] + tr[2]];
            let seg = round_cone_segment(tp, a, b, r1, r2);
            assert!(
                (seg - golden).abs() < 2e-5,
                "point {p:?}: segment {seg}, golden {golden}"
            );
        }
    }

    // Brute-force reference: minimum distance from `p` to a dense sampling of
    // the triangle surface via barycentric coordinates. Converges to the exact
    // distance as the sampling tightens.
    fn tri_brute_force(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
        const N: usize = 400;
        let mut best = f32::INFINITY;
        for i in 0..=N {
            let u = i as f32 / N as f32;
            let jmax = N - i;
            for j in 0..=jmax {
                let v = j as f32 / N as f32;
                let w = 1.0 - u - v;
                let q = [
                    a[0] * u + b[0] * v + c[0] * w,
                    a[1] * u + b[1] * v + c[1] * w,
                    a[2] * u + b[2] * v + c[2] * w,
                ];
                let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                if d < best {
                    best = d;
                }
            }
        }
        best
    }

    #[test]
    fn triangle_zero_on_surface_and_vertices() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [0.0, 2.0, 0.0];
        // Vertices and centroid lie on the patch: distance zero.
        assert!(triangle_sdf(a, a, b, c).abs() < 1e-6);
        assert!(triangle_sdf(b, a, b, c).abs() < 1e-6);
        assert!(triangle_sdf(c, a, b, c).abs() < 1e-6);
        let centroid = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, 0.0];
        assert!(triangle_sdf(centroid, a, b, c).abs() < 1e-6);
    }

    #[test]
    fn triangle_face_region_is_perpendicular_offset() {
        // Lift the centroid straight off the xy-plane: nearest feature is the
        // face interior, so the distance is the pure perpendicular height.
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [0.0, 2.0, 0.0];
        let over_centroid = [2.0 / 3.0, 2.0 / 3.0, 1.5];
        assert!((triangle_sdf(over_centroid, a, b, c) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn triangle_vertex_region_is_vertex_distance() {
        // Past the a-vertex along -x/-y: the perpendicular feet all clamp to the
        // a vertex, so the distance is the straight line to it.
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [0.0, 2.0, 0.0];
        let p = [-1.0, -1.0, 0.0];
        assert!((triangle_sdf(p, a, b, c) - length2([1.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn triangle_matches_brute_force_general_orientation() {
        // A tilted, off-origin triangle exercises both the face and edge
        // branches against the dense surface sampling.
        let a = [0.3, -0.4, 0.1];
        let b = [1.7, 0.2, -0.6];
        let c = [-0.5, 1.3, 0.8];
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, -0.5],
            [0.6, 0.1, 0.2],
            [2.5, 2.5, -1.0],
            [0.4, 1.0, 2.0],
        ] {
            let got = triangle_sdf(p, a, b, c);
            let reference = tri_brute_force(p, a, b, c);
            assert!(
                (got - reference).abs() < 2e-3,
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

    #[test]
    fn death_star_axis_features_match_closed_forms() {
        let (ra, rb, d) = (1.0f32, 0.5f32, 0.7f32);
        // Far pole of the large sphere, opposite the bite: on the surface.
        assert!(death_star([-1.0, 0.0, 0.0], ra, rb, d).abs() < 1e-6);
        // Origin lies inside the body but inside the biting sphere's hull:
        // the crater floor governs, giving -(d - rb) = -0.2.
        assert!((death_star([0.0, 0.0, 0.0], ra, rb, d) - (-0.2)).abs() < 1e-6);
    }

    #[test]
    fn death_star_rim_branch_matches_reference() {
        let (ra, rb, d) = (1.0f32, 0.5f32, 0.7f32);
        // Points facing the crater lip take the exact distance to the rim
        // circle; values cross-checked against a brute-force surface sampler.
        assert!((death_star([2.0, 0.0, 0.0], ra, rb, d) - 1.207_122).abs() < 1e-5);
        assert!((death_star([0.9, 0.0, 0.0], ra, rb, d) - 0.464_451).abs() < 1e-5);
    }

    #[test]
    fn cone_sdf_apex_base_and_lateral_surface() {
        let (r, h) = (0.5f32, 1.0f32);
        // Straight above the apex: distance is the gap to the apex.
        assert!((cone_sdf([0.0, 0.5, 0.0], r, h) - 0.5).abs() < 1e-6);
        // Below the base disc on the axis: distance to the base plane.
        assert!((cone_sdf([0.0, -1.5, 0.0], r, h) - 0.5).abs() < 1e-6);
        // On the lateral surface (half height has half the base radius).
        assert!(cone_sdf([0.25, -0.5, 0.0], r, h).abs() < 1e-6);
        // On the base rim.
        assert!(cone_sdf([0.5, -1.0, 0.0], r, h).abs() < 1e-6);
    }

    #[test]
    fn cone_sdf_interior_is_negative_perpendicular_distance() {
        let (r, h) = (0.5f32, 1.0f32);
        // Inside, nearest feature is the slanted face: -(0.5 / sqrt(5)).
        let expected = -0.5f32 / (5.0f32).sqrt();
        assert!((cone_sdf([0.0, -0.5, 0.0], r, h) - expected).abs() < 1e-6);
    }

    #[test]
    fn line_sdf_is_perpendicular_distance_to_the_axis() {
        // Line = x axis: distance is the yz radius.
        assert!((line_sdf([0.0, 3.0, 4.0], [1.0, 0.0, 0.0]) - 5.0).abs() < 1e-6);
        // A point on the line reports zero.
        assert!(line_sdf([5.0, 0.0, 0.0], [1.0, 0.0, 0.0]).abs() < 1e-6);
        // Direction need not be unit length.
        assert!((line_sdf([1.0, 2.0, 2.0], [2.0, 0.0, 0.0]) - (8.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn rounded_cylinder_outer_silhouette_on_surface() {
        let (r, rb, h) = (0.6f32, 0.1f32, 0.6f32);
        // On the lateral surface at the equator.
        assert!(rounded_cylinder([0.6, 0.0, 0.0], r, rb, h).abs() < 1e-6);
        // On the flat top cap on the axis.
        assert!(rounded_cylinder([0.0, 0.6, 0.0], r, rb, h).abs() < 1e-6);
        // On the inner band where the fillet begins (inset by the rounding).
        assert!((rounded_cylinder([0.5, 0.5, 0.0], r, rb, h) - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn rounded_cylinder_interior_and_exterior_closed_forms() {
        let (r, rb, h) = (0.6f32, 0.1f32, 0.6f32);
        // Dead centre: deepest interior distance.
        assert!((rounded_cylinder([0.0, 0.0, 0.0], r, rb, h) - (-0.6)).abs() < 1e-6);
        // Radially outside at the equator: plain lateral gap.
        assert!((rounded_cylinder([1.0, 0.0, 0.0], r, rb, h) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn infinite_cylinder_is_radial_distance_independent_of_height() {
        // Axis on the y axis, unit radius.
        assert!((infinite_cylinder([2.0, 0.0, 0.0], [0.0, 0.0], 1.0) - 1.0).abs() < 1e-6);
        // On the lateral surface.
        assert!(infinite_cylinder([1.0, 0.0, 0.0], [0.0, 0.0], 1.0).abs() < 1e-6);
        // Interior distance is independent of y.
        assert!((infinite_cylinder([0.5, 100.0, 0.0], [0.0, 0.0], 1.0) - (-0.5)).abs() < 1e-6);
        // Offset axis shifts the measurement centre.
        assert!(infinite_cylinder([3.0, 0.0, 4.0], [3.0, 1.0], 3.0).abs() < 1e-6);
    }
}
