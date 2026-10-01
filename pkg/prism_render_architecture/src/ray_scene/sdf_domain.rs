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
pub fn elongate(point: [f32; 3], half_extent: [f32; 3]) -> [f32; 3] {
    [
        point[0] - point[0].clamp(-half_extent[0], half_extent[0]),
        point[1] - point[1].clamp(-half_extent[1], half_extent[1]),
        point[2] - point[2].clamp(-half_extent[2], half_extent[2]),
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

#[cfg(test)]
mod tests {
    use super::{
        elongate, extrude, fold_plane, limited_repeat, mirror, mirror_repeat, onion,
        repeat, round_distance, scale_distance, scale_point, translate,
    };
    use crate::ray_scene::sdf_primitives::capped_cylinder;

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
