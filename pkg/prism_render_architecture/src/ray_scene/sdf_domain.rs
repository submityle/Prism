//! Domain and distance operators for composing signed distance fields on the
//! `CPU` golden path.
//!
//! Constructive-solid-geometry operators ([`super::sdf_csg`]) combine the
//! *values* of two fields; domain operators instead reshape a single field,
//! either by transforming the query point before it is evaluated or by
//! remapping the returned distance. Together they are the vocabulary `AAA`
//! content pipelines use to build complex implicit geometry from primitives:
//! round off edges, hollow a solid into a shell, instance a shape across a
//! lattice, or move and scale it in space.
//!
//! Point operators ([`translate`], [`repeat`], [`scale_point`]) answer "where
//! should I sample the base field to achieve this transform?", so a caller
//! evaluates the primitive at the returned point. Distance operators
//! ([`round_distance`], [`onion`], [`scale_distance`]) post-process the value
//! the primitive returns. Non-uniform spatial scaling pairs
//! [`scale_point`] with [`scale_distance`] so the result stays a valid
//! distance.
//!
//! Every operator is pure scalar or per-axis arithmetic — the only non-trivial
//! intrinsic is [`f32::round`] for lattice folding, which is permitted and
//! exact — so the whole module is transcendental-free and reproducible.

/// Rounds a shape by inflating its surface outward by `radius`, softening
/// sharp edges and corners.
///
/// Subtracts `radius` from the base distance `d`; the zero level set moves
/// outward by `radius`, which is the implicit-surface equivalent of a fillet.
pub fn round_distance(d: f32, radius: f32) -> f32 {
    d - radius
}

/// Hollows a solid into a shell of the given `thickness` centred on the
/// original surface.
///
/// Folds the distance about zero (`|d|`) and subtracts `thickness`, turning a
/// filled region into a wall of half-width `thickness` straddling the former
/// surface.
pub fn onion(d: f32, thickness: f32) -> f32 {
    d.abs() - thickness
}

/// Translates the field by `offset`: returns the point at which the base field
/// should be sampled to evaluate the shifted shape.
///
/// Evaluating a primitive at `translate(point, offset)` renders it moved by
/// `+offset` in world space.
pub fn translate(point: [f32; 3], offset: [f32; 3]) -> [f32; 3] {
    [
        point[0] - offset[0],
        point[1] - offset[1],
        point[2] - offset[2],
    ]
}

/// Folds `point` into the primitive cell of an infinite lattice with the given
/// per-axis `period`, instancing one primitive across all of space.
///
/// Each axis is mapped to `p - period * round(p / period)`, placing the result
/// in `[-period/2, period/2]`. An axis whose period is zero (or sub-normal) is
/// left unchanged, so the lattice can repeat along a subset of axes.
pub fn repeat(point: [f32; 3], period: [f32; 3]) -> [f32; 3] {
    let mut folded = point;
    for axis in 0..3 {
        if period[axis].abs() > f32::MIN_POSITIVE {
            let cell = (point[axis] / period[axis]).round();
            folded[axis] = point[axis] - period[axis] * cell;
        }
    }
    folded
}

/// Folds `point` into a *finite* lattice: like [`repeat`] but the instance
/// index on each axis is clamped to `+-limit`, so exactly `2 * limit + 1`
/// copies are placed and the field reverts to a single (edge) primitive beyond
/// the clamped range instead of tiling forever.
///
/// Each axis maps to `p - period * clamp(round(p / period), -limit, limit)`.
/// Inside the clamped band this is identical to [`repeat`]; past the last cell
/// the subtracted offset saturates, so the query point keeps growing away from
/// the final instance and the base primitive's distance grows monotonically
/// (no spurious wrapped copies). An axis whose period is zero (or sub-normal)
/// is left unchanged. This is Inigo Quilez's `opRepLim` — the operator AAA
/// procedural content uses for bounded arrays (a row of columns, a bank of
/// studs) rather than an unbounded crystal.
pub fn limited_repeat(point: [f32; 3], period: [f32; 3], limit: [f32; 3]) -> [f32; 3] {
    let mut folded = point;
    for axis in 0..3 {
        if period[axis].abs() > f32::MIN_POSITIVE {
            let cell = (point[axis] / period[axis])
                .round()
                .clamp(-limit[axis], limit[axis]);
            folded[axis] = point[axis] - period[axis] * cell;
        }
    }
    folded
}

/// Scales the field uniformly by `factor`: returns the point at which the base
/// field should be sampled to evaluate the scaled shape.
///
/// Pair with [`scale_distance`] applied to the primitive's result so the
/// returned value remains a valid (uncompressed) distance. `factor` must be
/// non-zero; a zero factor collapses the domain and is left to the caller to
/// avoid.
pub fn scale_point(point: [f32; 3], factor: f32) -> [f32; 3] {
    [point[0] / factor, point[1] / factor, point[2] / factor]
}

/// Rescales a distance sampled from a domain scaled by [`scale_point`] back
/// into world units.
///
/// Multiplies the base distance by `factor`, undoing the compression that
/// dividing the query point by `factor` introduced.
pub fn scale_distance(distance: f32, factor: f32) -> f32 {
    distance * factor
}

/// Stretches the field into a prism by carving a slab of core out of the
/// query point, giving a primitive flat caps joined by extruded sides.
///
/// Each axis subtracts `clamp(p, -h, h)`, collapsing the `[-h, h]` core to the
/// origin so the base primitive is evaluated at the squeezed point; sampling a
/// sphere through `elongate` yields a capsule, a box yields a rounded slab.
/// A zero half-extent leaves that axis unchanged.
///
/// This point transform is exact in the primitive's *exterior* but not in its
/// interior (the collapsed core maps every inside point to the primitive
/// centre). For an exact field everywhere, add [`elongate_correction`] to the
/// evaluated primitive distance: `primitive(elongate(p, h)) + elongate_correction(p, h)`.
pub fn elongate(point: [f32; 3], half_extent: [f32; 3]) -> [f32; 3] {
    [
        point[0] - point[0].clamp(-half_extent[0], half_extent[0]),
        point[1] - point[1].clamp(-half_extent[1], half_extent[1]),
        point[2] - point[2].clamp(-half_extent[2], half_extent[2]),
    ]
}

/// Interior distance correction that upgrades [`elongate`] from an
/// exterior-only transform to an exact signed field everywhere.
///
/// [`elongate`] alone evaluates the primitive at a point whose `[-h, h]` core
/// is collapsed to the origin, which is exact outside the shape but loses the
/// interior gradient. Adding this term restores it:
/// `primitive(elongate(p, h)) + elongate_correction(p, h)`. The correction is
/// `min(max(|p.x| - h.x, |p.y| - h.y, |p.z| - h.z), 0)` — the (non-positive)
/// signed distance into the inserted elongation box, and zero outside it so the
/// already-exact exterior is untouched.
///
/// This is Inigo Quilez's exact elongation: elongating a sphere of radius `r`
/// by half-extents `h` is identically a rounded box of half-extents `h` and
/// corner radius `r`, which this pairing reproduces to the bit.
pub fn elongate_correction(point: [f32; 3], half_extent: [f32; 3]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    let qz = point[2].abs() - half_extent[2];
    qx.max(qy).max(qz).min(0.0)
}

/// Two-dimensional [`elongate`]: stretches a 2D profile into a slab by carving
/// the `[-h, h]` core out of each axis of the query point.
///
/// Mirrors the 3D transform exactly — `p - clamp(p, -h, h)` per axis — and is
/// likewise exact only in the exterior. Pair it with [`elongate_2d_correction`]
/// for an exact field everywhere:
/// `profile(elongate_2d(p, h)) + elongate_2d_correction(p, h)`. Useful for
/// shaping a 2D profile before [`revolution`] or [`extrude`].
pub fn elongate_2d(point: [f32; 2], half_extent: [f32; 2]) -> [f32; 2] {
    [
        point[0] - point[0].clamp(-half_extent[0], half_extent[0]),
        point[1] - point[1].clamp(-half_extent[1], half_extent[1]),
    ]
}

/// Interior distance correction for [`elongate_2d`], the 2D companion to
/// [`elongate_correction`].
///
/// Adds the (non-positive) signed distance into the inserted elongation box:
/// `min(max(|p.x| - h.x, |p.y| - h.y), 0)`, and zero outside it. Combined with
/// [`elongate_2d`] it is exact everywhere — elongating a circle of radius `r`
/// by half-extents `h` reproduces, to float round-off, a rounded box of outer
/// half-extents `h + r` and corner radius `r`.
pub fn elongate_2d_correction(point: [f32; 2], half_extent: [f32; 2]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    qx.max(qy).min(0.0)
}

/// Rotates a 2D query point by the angle whose sine and cosine are `sin` and
/// `cos` (counter-clockwise, `R = [[cos, -sin], [sin, cos]]`).
///
/// Trigonometry is pre-baked by the caller so the runtime stays
/// transcendental-free; pass `sin.sin()`/`cos.cos()` of the desired angle (or a
/// cached pair). Rotation is an isometry, so wrapping a profile as
/// `profile(rotate_2d(p, -sin, cos))` orients it by `+angle` about the origin
/// without distorting the field. Supplying a unit `(sin, cos)` keeps distances
/// exact; a non-unit pair scales the field by the pair's magnitude.
pub fn rotate_2d(point: [f32; 2], sin: f32, cos: f32) -> [f32; 2] {
    [
        cos * point[0] - sin * point[1],
        sin * point[0] + cos * point[1],
    ]
}

/// Rotates a 3D query point about the unit axis `axis` by the angle whose sine
/// and cosine are `sin` and `cos`, using Rodrigues' rotation formula.
///
/// `v*cos + (axis x v)*sin + axis*(axis . v)*(1 - cos)`. Built only from
/// `dot`, `cross`, and multiply/add, so the runtime is transcendental-free when
/// the caller pre-bakes `(sin, cos)`. `axis` must be unit length for the result
/// to be an isometry; sampling a primitive as `primitive(rotate_axis(p, axis,
/// -sin, cos))` orients it by `+angle` about that axis without distorting the
/// field.
pub fn rotate_axis(point: [f32; 3], axis: [f32; 3], sin: f32, cos: f32) -> [f32; 3] {
    let cross = [
        axis[1] * point[2] - axis[2] * point[1],
        axis[2] * point[0] - axis[0] * point[2],
        axis[0] * point[1] - axis[1] * point[0],
    ];
    let axis_dot = axis[0] * point[0] + axis[1] * point[1] + axis[2] * point[2];
    let w = axis_dot * (1.0 - cos);
    [
        point[0] * cos + cross[0] * sin + axis[0] * w,
        point[1] * cos + cross[1] * sin + axis[1] * w,
        point[2] * cos + cross[2] * sin + axis[2] * w,
    ]
}

/// Mirrors the field across the selected coordinate planes by folding those
/// axes to their absolute value, instancing a symmetric copy of the primitive.
///
/// For each axis whose `axes` flag is set the coordinate is replaced by its
/// magnitude, so a primitive placed in the positive octant is reflected into
/// the negative side; unset axes pass through unchanged. This is the domain
/// equivalent of modelling one half and letting symmetry complete the shape.
pub fn mirror(point: [f32; 3], axes: [bool; 3]) -> [f32; 3] {
    let mut folded = point;
    for axis in 0..3 {
        if axes[axis] {
            folded[axis] = point[axis].abs();
        }
    }
    folded
}

/// Instances the field across an infinite lattice like [`repeat`], but mirrors
/// every other cell so adjacent copies meet as reflections instead of plain
/// translations, giving a seamless kaleidoscopic tiling with no visible seam at
/// the cell boundary.
///
/// Per axis the coordinate is folded to its cell-local offset
/// `p - period * round(p / period)`; when the integer cell index is odd the
/// local offset is negated, reflecting that cell. Because the fold is `C0`
/// continuous across boundaries (both sides evaluate to the same magnitude at
/// the midpoint), a primitive tiled this way joins its mirror image without a
/// crack — the operator AAA content uses for brick courses, tread plates and
/// other mirror-symmetric arrays. A zero-period axis passes through unchanged.
pub fn mirror_repeat(point: [f32; 3], period: [f32; 3]) -> [f32; 3] {
    let mut folded = point;
    for axis in 0..3 {
        if period[axis].abs() > f32::MIN_POSITIVE {
            let cell = (point[axis] / period[axis]).round();
            let mut local = point[axis] - period[axis] * cell;
            // `cell` is integral; an odd index reflects the cell so neighbours
            // mirror. `% 2.0 != 0.0` is exact for the integral values `round`
            // produces.
            if (cell % 2.0).abs() > 0.5 {
                local = -local;
            }
            folded[axis] = local;
        }
    }
    folded
}

/// Folds space across an arbitrary plane through the origin with unit `normal`,
/// reflecting everything on the plane's negative side onto its positive side so
/// a primitive modelled in one half is instanced symmetrically about the plane.
///
/// Where [`mirror`] only folds across the coordinate planes, this folds across
/// any orientation: a point whose signed distance to the plane is negative is
/// mirrored through it (`p - 2 * dot(p, n) * n`), while a point already on the
/// positive side passes through unchanged. The map is an isometry on each half,
/// so composing it with a primitive leaves the result an exact distance. With
/// `normal` equal to a basis axis it reduces to a single-axis [`mirror`].
///
/// `normal` must be unit length; a non-normalised vector scales the reflection
/// and breaks the distance property, so callers must normalise first.
pub fn fold_plane(point: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let signed = (point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2]).min(0.0);
    let k = 2.0 * signed;
    [
        point[0] - k * normal[0],
        point[1] - k * normal[1],
        point[2] - k * normal[2],
    ]
}

/// Mirror fold across an arbitrary plane `dot(p, normal) = offset`, the
/// offset-plane generalisation of [`fold_plane`] (which folds through the
/// origin).
///
/// `normal` must be unit length; `offset` is the plane's signed distance from
/// the origin along `normal`. Points already on the positive side
/// (`dot(p, normal) >= offset`) pass through unchanged, while points on the
/// negative side are reflected across the plane. The fold is a distance-
/// preserving isometry, so modelling one half of a scene and folding reproduces
/// its mirror image about any placed plane; applying it twice is idempotent.
pub fn fold_plane_offset(point: [f32; 3], normal: [f32; 3], offset: f32) -> [f32; 3] {
    let signed =
        (point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2] - offset).min(0.0);
    let k = 2.0 * signed;
    [
        point[0] - k * normal[0],
        point[1] - k * normal[1],
        point[2] - k * normal[2],
    ]
}

/// Maps a three-dimensional query point onto the two-dimensional lathe plane of
/// a solid of revolution about the `y` axis, offsetting the profile `offset`
/// units out along the radius.
///
/// This is Inigo Quilez's `opRevolution`: it returns
/// `(length(point.xz) - offset, point.y)`, the coordinates at which a 2D
/// profile should be sampled so that sweeping it around the `y` axis produces
/// the lathed solid. Pairing it with any 2D profile builds goblets, columns,
/// bottles and tori; a circular profile of radius `r` recovers an exact torus
/// of major radius `offset` and minor radius `r`. Uses a single `sqrt`, so it
/// stays transcendental-free.
pub fn revolution(point: [f32; 3], offset: f32) -> [f32; 2] {
    let radial = (point[0] * point[0] + point[2] * point[2]).sqrt();
    [radial - offset, point[1]]
}

/// Extrudes a two-dimensional signed-distance field `d2d` (measured in the
/// `xy` plane) into a three-dimensional slab of half-thickness `half_height`
/// along the `z` axis, yielding an exact 3D signed distance.
///
/// This is Inigo Quilez's `opExtrusion`: with `w = (d2d, |z| - half_height)`
/// the result is `min(max(w.x, w.y), 0) + length(max(w, 0))`. The first term
/// handles the interior (negative on both the profile and the cap), the second
/// the exterior corner where the point clears both the side wall and the cap
/// faces at once. It is the standard bridge for turning any 2D profile — a
/// polygon, star, text glyph, or arc — into a prism, and uses only `abs`,
/// `min`/`max` and a single `sqrt`, so it stays transcendental-free.
pub fn extrude(d2d: f32, z: f32, half_height: f32) -> f32 {
    let wx = d2d;
    let wy = z.abs() - half_height;
    let inside = wx.max(wy).min(0.0);
    let ox = wx.max(0.0);
    let oy = wy.max(0.0);
    let outside = (ox * ox + oy * oy).sqrt();
    inside + outside
}

/// Extrudes a two-dimensional signed-distance field `d2d` (measured in the
/// `xy` plane) into a three-dimensional slab of half-thickness `half_height`
/// along the `z` axis while rounding the rim where the side wall meets each
/// cap with fillet radius `rounding`, yielding an exact 3D signed distance.
///
/// This is the rounded-rim variant of [`extrude`]: the profile is inset by
/// `rounding` on both the planar and the axial axes before the standard
/// extrusion combine, then the result is offset back out by `rounding`. With
/// `w = (d2d + rounding, |z| - (half_height - rounding))` the distance is
/// `min(max(w.x, w.y), 0) + length(max(w, 0)) - rounding` — the same
/// inset-then-round pattern used by [`crate::ray_scene::sdf_primitives::rounded_box_2d`]
/// and [`crate::ray_scene::sdf_primitives::rounded_cylinder`]. For a solid
/// result `rounding` must not exceed `half_height`, and for a circular
/// profile of radius `r` it must not exceed `r`. Extruding a circle profile
/// `d2d = length(xy) - r` reproduces an exact rounded cylinder. Uses only
/// `abs`, `min`/`max` and a single `sqrt`, so it stays transcendental-free.
pub fn extrude_round(d2d: f32, z: f32, half_height: f32, rounding: f32) -> f32 {
    let wx = d2d + rounding;
    let wy = z.abs() - (half_height - rounding);
    let inside = wx.max(wy).min(0.0);
    let ox = wx.max(0.0);
    let oy = wy.max(0.0);
    let outside = (ox * ox + oy * oy).sqrt();
    inside + outside - rounding
}

#[cfg(test)]
mod tests {
    use super::{
        elongate, elongate_2d, elongate_2d_correction, elongate_correction, extrude, extrude_round, fold_plane, fold_plane_offset, limited_repeat, mirror, mirror_repeat, onion,
        repeat, revolution, rotate_2d, rotate_axis, round_distance, scale_distance, scale_point,
        translate,
    };
    use crate::ray_scene::sdf_primitives::{
        box_2d, capped_cylinder, circle_2d, round_box, rounded_box_2d, rounded_cylinder, sphere,
        torus,
    };

    #[test]
    fn extrude_of_a_circle_matches_a_capped_cylinder() {
        // Extruding a 2D circle of radius r along z must reproduce an exact
        // capped cylinder (whose axis runs along y): map our (x, y, z) sample
        // to the cylinder's frame by swapping the extrusion axis into y.
        let r = 0.8_f32;
        let h = 1.3_f32;
        let samples: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],   // interior centre
            [0.3, 0.2, 0.5],   // interior, off centre
            [1.5, 0.0, 0.0],   // outside the side wall
            [0.0, 0.0, 2.0],   // outside past the end cap
            [0.8, 0.0, 1.3],   // on the rim edge
            [1.1, 0.2, 1.9],   // outside the rim corner
        ];
        for p in samples {
            let d2d = (p[0] * p[0] + p[1] * p[1]).sqrt() - r;
            let got = extrude(d2d, p[2], h);
            let want = capped_cylinder([p[0], p[2], p[1]], h, r);
            assert!(
                (got - want).abs() < 1e-6,
                "mismatch at {p:?}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn extrude_interior_and_cap_distances_are_exact() {
        // A unit-circle profile extruded to half-thickness 1 along z.
        let r = 1.0_f32;
        let h = 1.0_f32;
        // Deep interior on the axis: nearest exit is the closer of the side
        // wall (distance r) and the cap (distance h) -> -min(r, h) = -1.
        assert!((extrude(-r, 0.0, h) + 1.0).abs() < 1e-6);
        // Directly beyond a cap on the axis (profile interior, z past cap).
        let z = 2.5_f32;
        assert!((extrude(-r, z, h) - (z - h)).abs() < 1e-6);
        // Straight out the side wall within the slab: pure 2D distance.
        let d2d = 0.6_f32;
        assert!((extrude(d2d, 0.0, h) - d2d).abs() < 1e-6);
        // Exterior rim corner: both terms positive -> Euclidean corner distance.
        let (dx, dz) = (0.6_f32, 0.8_f32);
        let want = (dx * dx + dz * dz).sqrt();
        assert!((extrude(dx, h + dz, h) - want).abs() < 1e-6);
    }

    #[test]
    fn extrude_round_of_a_circle_matches_a_rounded_cylinder() {
        // Extruding a 2D circle of radius R along z with rim radius r must
        // reproduce an exact rounded cylinder (whose axis runs along y): map
        // our (x, y, z) sample into the cylinder frame by swapping the
        // extrusion axis z into y, exactly as the plain-extrude test does.
        let big_r = 1.1_f32;
        let h = 1.3_f32;
        let fillet = 0.35_f32;
        let samples: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],   // interior centre
            [0.3, 0.2, 0.5],   // interior, off centre
            [1.8, 0.0, 0.0],   // outside the side wall
            [0.0, 0.0, 2.1],   // outside past the end cap
            [1.1, 0.0, 1.3],   // near the rounded rim corner
            [0.9, 0.1, 1.1],   // inside, close to the fillet
            [1.4, 0.3, 1.6],   // outside the rounded corner
            [0.5, 0.5, 0.9],   // generic interior
        ];
        for p in samples {
            let d2d = (p[0] * p[0] + p[1] * p[1]).sqrt() - big_r;
            let got = extrude_round(d2d, p[2], h, fillet);
            let want = rounded_cylinder([p[0], p[2], p[1]], big_r, fillet, h);
            assert!(
                (got - want).abs() < 1e-6,
                "mismatch at {p:?}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn extrude_round_matches_an_independent_meridian_reference() {
        // Cross-check against a brute-force signed distance computed purely in
        // the meridian plane (radial = length(xy), axial = z) to a rectangle of
        // half-extents (R - r, h - r) offset outward by r. This reference does
        // not reuse any sibling SDF, so it independently pins the formula.
        fn length2(a: f32, b: f32) -> f32 {
            (a * a + b * b).sqrt()
        }
        // Signed distance from (rad, ax) to the inset rectangle boundary, then
        // offset by the fillet radius. Interior is negative.
        fn brute(rad: f32, ax: f32, hx: f32, hy: f32, r: f32) -> f32 {
            let ar = rad.abs();
            let aa = ax.abs();
            let inside = ar <= hx && aa <= hy;
            let mut best = f32::INFINITY;
            let n = 20_000_u32;
            for i in 0..n {
                let t = (i as f32) / (n as f32) * 4.0;
                let (bx, by) = if t < 1.0 {
                    (-hx + 2.0 * hx * t, hy)
                } else if t < 2.0 {
                    (hx, hy - 2.0 * hy * (t - 1.0))
                } else if t < 3.0 {
                    (hx - 2.0 * hx * (t - 2.0), -hy)
                } else {
                    (-hx, -hy + 2.0 * hy * (t - 3.0))
                };
                let d = length2(ar - bx, aa - by);
                if d < best {
                    best = d;
                }
            }
            let sd = if inside { -best } else { best };
            sd - r
        }
        let big_r = 1.2_f32;
        let h = 1.0_f32;
        let fillet = 0.3_f32;
        let samples: [[f32; 3]; 6] = [
            [0.6, 0.4, 0.3],
            [1.5, 0.0, 0.2],
            [0.2, 0.1, 1.4],
            [1.3, 0.2, 1.1],
            [0.0, 0.0, 0.0],
            [0.9, 0.5, 0.7],
        ];
        for p in samples {
            let rad = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let d2d = rad - big_r;
            let got = extrude_round(d2d, p[2], h, fillet);
            let want = brute(rad, p[2], big_r - fillet, h - fillet, fillet);
            // Brute reference carries O(1e-4) perimeter-sampling error.
            assert!(
                (got - want).abs() < 2e-3,
                "mismatch at {p:?}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn extrude_round_closed_form_rim_cap_and_interior() {
        // Unit-circle profile (R = 1), half-thickness 1, fillet 0.25.
        let big_r = 1.0_f32;
        let h = 1.0_f32;
        let r = 0.25_f32;
        // Straight out the side wall within the slab: distance is the plain 2D
        // profile distance (the fillet only rounds the rim corner, not the
        // flat wall), so d2d outward with |z| well inside the slab -> d2d.
        let d2d = 0.5_f32;
        assert!((extrude_round(d2d, 0.0, h, r) - d2d).abs() < 1e-6);
        // Directly beyond a cap on the axis: the flat cap sits at |z| = h, so
        // distance is |z| - h regardless of the fillet when radially centred.
        let z = 2.0_f32;
        assert!((extrude_round(-big_r, z, h, r) - (z - h)).abs() < 1e-6);
        // Deep interior on the axis: nearest exit is the closer flat face,
        // min(R, h) = 1 away, so the signed distance is -1 (the fillet does not
        // reach the centre).
        assert!((extrude_round(-big_r, 0.0, h, r) + 1.0).abs() < 1e-6);
        // Exterior rounded corner: approach the rim corner (d2d = 0 at the
        // nominal radius, z = h) diagonally. The inset corner sits at
        // (0 + r, h - (h - r)) = (r, r) in w-space offsets... instead verify the
        // outward diagonal: at the exact rim point the surface distance is 0.
        let on_rim = extrude_round(0.0, h, h, r); // wx=r, wy=r -> sqrt(2)*r - r
        let want_rim = (2.0_f32).sqrt() * r - r;
        assert!((on_rim - want_rim).abs() < 1e-6);
    }

    #[test]
    fn revolution_of_a_circle_matches_a_torus() {
        // A circular profile of radius r, offset `offset` out along the radius,
        // lathed about y, is exactly a torus of major `offset`, minor r.
        let offset = 1.4_f32;
        let r = 0.5_f32;
        let samples: [[f32; 3]; 6] = [
            [2.0, 0.0, 0.0],   // outside the tube on the +x spoke
            [1.4, 0.5, 0.0],   // on the top of the tube
            [1.4, 0.0, 0.0],   // on the ring centreline (interior, -r)
            [0.0, 0.0, 1.9],   // outside on the +z spoke
            [1.0, 0.3, 1.0],   // generic off-axis point
            [0.0, 2.0, 0.0],   // on the y axis, far above
        ];
        for p in samples {
            let q = revolution(p, offset);
            let got = (q[0] * q[0] + q[1] * q[1]).sqrt() - r;
            let want = torus(p, offset, r);
            assert!(
                (got - want).abs() < 1e-6,
                "mismatch at {p:?}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn revolution_reports_radial_offset_and_height() {
        // On the +x spoke the radial coordinate is x - offset and height is y.
        let q = revolution([3.0, 0.75, 0.0], 2.0);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 0.75).abs() < 1e-6);
        // Radius is rotation-invariant: same radial coordinate on the +z spoke.
        let q2 = revolution([0.0, 0.75, 3.0], 2.0);
        assert!((q2[0] - 1.0).abs() < 1e-6);
        assert!((q2[1] - 0.75).abs() < 1e-6);
        // Inside the offset ring the radial coordinate goes negative.
        let q3 = revolution([0.5, 0.0, 0.0], 2.0);
        assert!((q3[0] + 1.5).abs() < 1e-6);
    }

    #[test]
    fn round_distance_subtracts_radius() {
        assert_eq!(round_distance(0.5, 0.2), 0.3);
        assert_eq!(round_distance(-0.1, 0.2), -0.3);
    }

    #[test]
    fn onion_folds_about_zero() {
        // Interior and exterior points at equal magnitude map identically.
        assert_eq!(onion(0.5, 0.1), 0.4);
        assert_eq!(onion(-0.5, 0.1), 0.4);
        // A point already on the surface becomes the shell half-width.
        assert_eq!(onion(0.0, 0.1), -0.1);
    }

    #[test]
    fn translate_shifts_the_query_point() {
        assert_eq!(translate([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]), [0.5, 3.0, 1.0]);
    }

    #[test]
    fn repeat_folds_into_the_centre_cell() {
        // 2.6 with period 1 lands at 2.6 - round(2.6) = 2.6 - 3 = -0.4.
        let folded = repeat([2.6, -2.6, 0.3], [1.0, 1.0, 1.0]);
        assert!((folded[0] - (-0.4)).abs() < 1e-6);
        assert!((folded[1] - 0.4).abs() < 1e-6);
        // A point already inside the first half-cell is unchanged.
        assert!((folded[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn repeat_leaves_zero_period_axes_untouched() {
        let folded = repeat([5.0, 5.0, 5.0], [2.0, 0.0, 2.0]);
        // Axis 1 has zero period: coordinate passes through verbatim.
        assert_eq!(folded[1], 5.0);
        // Axes 0 and 2 fold: 5 - 2*round(2.5) = 5 - 2*3 = -1 (round half away
        // from zero, matching `f32::round`).
        assert!((folded[0] - (-1.0)).abs() < 1e-6);
        assert!((folded[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn limited_repeat_matches_repeat_inside_the_band() {
        // Instance index round(1.4) = 1 is within the +-2 limit, so the fold is
        // identical to the unbounded lattice: 1.4 - 1 = 0.4.
        let period = [1.0, 1.0, 1.0];
        let limit = [2.0, 2.0, 2.0];
        let point = [1.4, -1.4, 0.3];
        let bounded = limited_repeat(point, period, limit);
        let unbounded = repeat(point, period);
        for axis in 0..3 {
            assert!((bounded[axis] - unbounded[axis]).abs() < 1e-6);
        }
    }

    #[test]
    fn limited_repeat_saturates_past_the_last_cell() {
        // Beyond the clamp band the subtracted instance index saturates at the
        // limit, so the fold stops wrapping: round(2.6) = 3 is clamped to 2,
        // giving 2.6 - 1*2 = 0.6 instead of the unbounded 2.6 - 3 = -0.4.
        let period = [1.0, 1.0, 1.0];
        let limit = [2.0, 2.0, 2.0];
        let folded = limited_repeat([2.6, 0.0, 0.0], period, limit);
        assert!((folded[0] - 0.6).abs() < 1e-6);
        assert!((repeat([2.6, 0.0, 0.0], period)[0] - (-0.4)).abs() < 1e-6);

        // Pushing the query point further out keeps the folded coordinate
        // growing one-for-one (the field reverts to the single edge instance),
        // so the base primitive's distance increases monotonically rather than
        // spawning spurious wrapped copies.
        let near = limited_repeat([10.0, 0.0, 0.0], period, limit)[0];
        let far = limited_repeat([11.0, 0.0, 0.0], period, limit)[0];
        assert!((near - 8.0).abs() < 1e-6);
        assert!((far - 9.0).abs() < 1e-6);
        assert!(far > near);
    }

    #[test]
    fn limited_repeat_leaves_zero_period_axes_untouched() {
        // Axis 1 has zero period and passes through verbatim; the finite lattice
        // only clamps axes that actually tile.
        let folded = limited_repeat([5.0, 7.0, 5.0], [2.0, 0.0, 2.0], [1.0, 1.0, 1.0]);
        assert_eq!(folded[1], 7.0);
        // Axes 0 and 2: round(2.5) = 3 clamped to limit 1, so 5 - 2*1 = 3.
        assert!((folded[0] - 3.0).abs() < 1e-6);
        assert!((folded[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mirror_repeat_reflects_odd_cells() {
        let period = [1.0, 1.0, 1.0];
        // Cell 0 (even): plain fold, 0.3 stays 0.3.
        assert!((mirror_repeat([0.3, 0.0, 0.0], period)[0] - 0.3).abs() < 1e-6);
        // Cell 1 (odd): local 1.4 - 1 = 0.4 is reflected to -0.4.
        assert!((mirror_repeat([1.4, 0.0, 0.0], period)[0] - (-0.4)).abs() < 1e-6);
        // Cell 3 (odd): local 2.6 - 3 = -0.4 is reflected to +0.4.
        assert!((mirror_repeat([2.6, 0.0, 0.0], period)[0] - 0.4).abs() < 1e-6);
        // Cell 2 (even): local 2.3 - 2 = 0.3 passes through.
        assert!((mirror_repeat([2.3, 0.0, 0.0], period)[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn mirror_repeat_is_continuous_across_the_seam() {
        // Approaching the cell-0/cell-1 boundary from both sides yields the same
        // folded coordinate, so a tiled primitive meets its mirror with no crack.
        let period = [1.0, 1.0, 1.0];
        let just_below = mirror_repeat([0.4999, 0.0, 0.0], period)[0];
        let just_above = mirror_repeat([0.5001, 0.0, 0.0], period)[0];
        assert!((just_below - just_above).abs() < 1e-3);
    }

    #[test]
    fn mirror_repeat_leaves_zero_period_axes_untouched() {
        let folded = mirror_repeat([1.4, 9.0, 1.4], [1.0, 0.0, 1.0]);
        assert_eq!(folded[1], 9.0);
        assert!((folded[0] - (-0.4)).abs() < 1e-6);
        assert!((folded[2] - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn fold_plane_reduces_to_axis_mirror() {
        // A unit +x normal must reproduce single-axis mirroring: the negative
        // side is reflected, the positive side passes through.
        let n = [1.0, 0.0, 0.0];
        assert_eq!(fold_plane([-3.0, 2.0, 1.0], n), [3.0, 2.0, 1.0]);
        assert_eq!(fold_plane([3.0, 2.0, 1.0], n), [3.0, 2.0, 1.0]);
    }

    #[test]
    fn fold_plane_offset_matches_fold_plane_at_zero_offset() {
        // With offset 0 the generalisation must reduce to fold_plane exactly.
        let n = [0.6, 0.8, 0.0];
        for p in [[-1.0, -2.0, 0.5], [3.0, 1.0, -1.0], [0.0, 0.0, 0.0]] {
            let a = fold_plane_offset(p, n, 0.0);
            let b = fold_plane(p, n);
            assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6 && (a[2] - b[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn fold_plane_offset_reflects_across_an_offset_plane() {
        // Plane dot(p, n) = offset with a unit diagonal normal.
        let inv = 1.0f32 / 3.0f32.sqrt();
        let n = [inv, inv, inv];
        let offset = 1.5f32;
        // A point on the positive side is left untouched.
        let keep = [2.0, 2.0, 2.0];
        let kf = fold_plane_offset(keep, n, offset);
        assert!((kf[0] - keep[0]).abs() < 1e-6 && (kf[1] - keep[1]).abs() < 1e-6 && (kf[2] - keep[2]).abs() < 1e-6);
        // A point on the negative side is mirrored: its signed distance to the
        // plane flips sign while the tangential part is preserved.
        let p = [-1.0, 0.5, -0.3];
        let f = fold_plane_offset(p, n, offset);
        let side_p = p[0] * n[0] + p[1] * n[1] + p[2] * n[2] - offset;
        let side_f = f[0] * n[0] + f[1] * n[1] + f[2] * n[2] - offset;
        assert!((side_f + side_p).abs() < 1e-6, "signed distance should negate");
        for axis in 0..3 {
            let tp = p[axis] - side_p * n[axis];
            let tf = f[axis] - side_f * n[axis];
            assert!((tp - tf).abs() < 1e-6, "tangential component preserved");
        }
        // Folding twice lands on the kept side and is then idempotent.
        let f2 = fold_plane_offset(f, n, offset);
        assert!((f2[0] - f[0]).abs() < 1e-6 && (f2[1] - f[1]).abs() < 1e-6 && (f2[2] - f[2]).abs() < 1e-6);
    }

    #[test]
    fn fold_plane_mirrors_across_a_diagonal_plane() {
        // Plane normal along the x=y diagonal. A point on the negative side is
        // reflected to the mirror position; its distance to the plane is
        // preserved (isometry), only the sign of the normal component flips.
        let inv = 1.0 / 2.0_f32.sqrt();
        let n = [inv, inv, 0.0];
        let p = [-1.0, 0.0, 0.5];
        let folded = fold_plane(p, n);
        // Reflection of (-1,0) across the line x+y=0 is (0,1).
        assert!((folded[0] - 0.0).abs() < 1e-6);
        assert!((folded[1] - 1.0).abs() < 1e-6);
        assert!((folded[2] - 0.5).abs() < 1e-6);
        // The folded point sits on the non-negative side of the plane.
        let signed = folded[0] * n[0] + folded[1] * n[1] + folded[2] * n[2];
        assert!(signed >= -1e-6);
    }

    #[test]
    fn fold_plane_is_isometric_on_the_reflected_half() {
        // Reflecting preserves lengths: the distance between two negative-side
        // points equals the distance between their folds.
        let inv = 1.0 / 3.0_f32.sqrt();
        let n = [inv, inv, inv];
        let a = [-1.0, -0.5, -0.2];
        let b = [-0.7, -0.9, -0.4];
        let fa = fold_plane(a, n);
        let fb = fold_plane(b, n);
        let d0 = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        let d1 = ((fa[0] - fb[0]).powi(2) + (fa[1] - fb[1]).powi(2) + (fa[2] - fb[2]).powi(2)).sqrt();
        assert!((d0 - d1).abs() < 1e-6, "d0={d0} d1={d1}");
    }

    #[test]
    fn elongate_collapses_the_core_and_shifts_the_rest() {
        // Outside the half-extent: shifted inward by the extent.
        let outside = elongate([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((outside[0] - 1.0).abs() < 1e-6);
        assert_eq!(outside[1], 0.0);
        // Inside the half-extent: collapses onto the slab centre.
        let inside = elongate([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(inside[0], 0.0);
    }

    #[test]
    fn elongate_correction_is_zero_outside_and_negative_inside() {
        let h = [1.0, 0.5, 0.25];
        // Fully outside the elongation box on every axis: no correction.
        assert_eq!(elongate_correction([2.0, 2.0, 2.0], h), 0.0);
        // On the box face: still zero (boundary of the exterior-exact region).
        assert!(elongate_correction([1.0, 0.0, 0.0], h).abs() < 1e-6);
        // Deep inside: correction is the (negative) distance to the nearest
        // face, i.e. min over axes of (|p| - h) clamped at zero.
        let got = elongate_correction([0.2, 0.1, 0.05], h);
        let want = (0.2_f32 - 1.0).max(0.1 - 0.5).max(0.05 - 0.25).min(0.0);
        assert!((got - want).abs() < 1e-6);
    }

    #[test]
    fn elongate_plus_correction_of_a_sphere_matches_a_rounded_box() {
        // Exact elongation (IQ): sphere(radius r) elongated by half-extents h
        // is identically a rounded box of half-extents h and corner radius r.
        // Verifies the point transform + correction over a deterministic sweep,
        // interior included, against the independently derived `round_box`.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let mut maxerr = 0.0f32;
        for _ in 0..5000 {
            let h = [next() * 2.0, next() * 2.0, next() * 2.0];
            let r = next() * 1.5 + 0.1;
            let p = [next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0];
            let got = sphere(elongate(p, h), r) + elongate_correction(p, h);
            let want = round_box(p, h, r);
            maxerr = maxerr.max((got - want).abs());
        }
        assert!(maxerr < 1e-5, "elongate-exact vs round_box maxerr = {maxerr}");
    }

    #[test]
    fn elongate_2d_matches_the_3d_transform_on_a_slice() {
        // The 2D transform must agree with the 3D one on the shared axes.
        let h2 = [1.0, 0.5];
        let h3 = [1.0, 0.5, 0.0];
        for p in [[2.0, 0.1], [0.3, 0.8], [-1.5, -0.2], [0.0, 0.0]] {
            let got = elongate_2d(p, h2);
            let want = elongate([p[0], p[1], 0.0], h3);
            assert!((got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6);
        }
        // Correction term matches too (z extent zero -> same max over x,y).
        let c2 = elongate_2d_correction([0.2, 0.1], h2);
        let want = (0.2_f32 - 1.0).max(0.1 - 0.5).min(0.0);
        assert!((c2 - want).abs() < 1e-6);
    }

    #[test]
    fn elongate_2d_plus_correction_of_a_circle_matches_a_rounded_box_2d() {
        // Exact 2D elongation: circle(radius r) elongated by half-extents h is
        // identically a rounded box of outer half-extents h + r and corner
        // radius r. Verified over a deterministic sweep against the
        // independently derived `rounded_box_2d`.
        let mut state: u32 = 0x0bad_c0de;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let mut maxerr = 0.0f32;
        for _ in 0..5000 {
            let h = [next() * 2.0, next() * 2.0];
            let r = next() * 1.5 + 0.05;
            let p = [next() * 8.0 - 4.0, next() * 8.0 - 4.0];
            let got = circle_2d(elongate_2d(p, h), r) + elongate_2d_correction(p, h);
            let want = rounded_box_2d(p, [h[0] + r, h[1] + r], [r, r, r, r]);
            maxerr = maxerr.max((got - want).abs());
        }
        assert!(maxerr < 1e-5, "elongate_2d-exact vs rounded_box_2d maxerr = {maxerr}");
    }

    #[test]
    fn rotate_2d_is_a_length_preserving_rotation() {
        // 90 degrees counter-clockwise maps +x onto +y.
        let q = rotate_2d([1.0, 0.0], 1.0, 0.0);
        assert!((q[0]).abs() < 1e-6 && (q[1] - 1.0).abs() < 1e-6);
        // Length preserved and inverse (negated sine) round-trips to identity.
        for a in [0.3_f32, 1.1, -2.4, 2.9] {
            let (s, c) = (a.sin(), a.cos());
            let p = [1.3, -0.7];
            let r = rotate_2d(p, s, c);
            let plen = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let rlen = (r[0] * r[0] + r[1] * r[1]).sqrt();
            assert!((plen - rlen).abs() < 1e-6);
            let back = rotate_2d(r, -s, c);
            assert!((back[0] - p[0]).abs() < 1e-6 && (back[1] - p[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn rotate_2d_orients_a_box_matching_a_hand_rotated_reference() {
        // A box_2d sampled through the inverse rotation is an exact rotated
        // box. Cross-check against evaluating box_2d at the point transformed
        // by an independently written rotation matrix.
        let he = [1.2, 0.5];
        for a in [0.2_f32, 0.9, -1.7] {
            let (s, c) = (a.sin(), a.cos());
            for p in [[0.4, 0.1], [2.0, -1.3], [-1.1, 0.8], [0.0, 0.0]] {
                let got = box_2d(rotate_2d(p, -s, c), he);
                // Independent reference: inverse-rotate by hand (R(-a) * p).
                let rp = [c * p[0] + s * p[1], -s * p[0] + c * p[1]];
                let qx = rp[0].abs() - he[0];
                let qy = rp[1].abs() - he[1];
                let want = ((qx.max(0.0)).powi(2) + (qy.max(0.0)).powi(2)).sqrt()
                    + qx.max(qy).min(0.0);
                assert!((got - want).abs() < 1e-6, "a={a} p={p:?} got={got} want={want}");
            }
        }
    }

    #[test]
    fn rotate_axis_matches_rotate_2d_about_z_and_preserves_length() {
        let mut state: u32 = 0x5151_a5a5;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        // About +z, the first two components must track the 2D rotation and z
        // is untouched.
        for a in [0.3_f32, 1.4, -2.1] {
            let (s, c) = (a.sin(), a.cos());
            let v = [0.7, -1.2, 0.9];
            let r3 = rotate_axis(v, [0.0, 0.0, 1.0], s, c);
            let r2 = rotate_2d([v[0], v[1]], s, c);
            assert!((r3[0] - r2[0]).abs() < 1e-6 && (r3[1] - r2[1]).abs() < 1e-6);
            assert!((r3[2] - v[2]).abs() < 1e-6);
        }
        // 90 degrees about +x maps +y onto +z.
        let q = rotate_axis([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.0);
        assert!(q[0].abs() < 1e-6 && q[1].abs() < 1e-6 && (q[2] - 1.0).abs() < 1e-6);
        // Length preservation and inverse round-trip about an arbitrary unit axis.
        for _ in 0..2000 {
            let mut axis = [next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0];
            let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if n < 1e-3 {
                continue;
            }
            axis = [axis[0] / n, axis[1] / n, axis[2] / n];
            let a = next() * 6.0 - 3.0;
            let (s, c) = (a.sin(), a.cos());
            let v = [next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0];
            let r = rotate_axis(v, axis, s, c);
            let vlen = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!((vlen - rlen).abs() < 1e-5);
            let back = rotate_axis(r, axis, -s, c);
            assert!(
                (back[0] - v[0]).abs() < 1e-5
                    && (back[1] - v[1]).abs() < 1e-5
                    && (back[2] - v[2]).abs() < 1e-5
            );
        }
    }

    #[test]
    fn mirror_folds_selected_axes() {
        let folded = mirror([-2.0, 3.0, -4.0], [true, false, true]);
        assert_eq!(folded, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn scale_point_and_distance_are_inverse_factors() {
        let p = scale_point([2.0, 4.0, 6.0], 2.0);
        assert_eq!(p, [1.0, 2.0, 3.0]);
        // Distance sampled in the compressed domain is rescaled back out.
        assert_eq!(scale_distance(0.5, 2.0), 1.0);
    }
}
