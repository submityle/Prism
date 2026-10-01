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

#[cfg(test)]
mod tests {
    use super::{
        elongate, mirror, onion, repeat, round_distance, scale_distance, scale_point, translate,
    };

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
