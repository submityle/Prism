//! Constructive-solid-geometry operators on signed distances for the `CPU`
//! golden path.
//!
//! Signed distance fields compose with simple algebra: the `min` of two fields
//! is their union, the `max` is their intersection, and negating one field
//! before a `max` is a subtraction. `AAA` modeling and effects pipelines
//! (`IQ`-style ray-marched shaders, distance-field decals, procedural meshing)
//! lean on the *smooth* variants, which replace the hard `min`/`max` with a
//! polynomial blend so neighboring shapes merge with a controllable fillet
//! radius instead of a crease.
//!
//! These are pure scalar operators on distance *values* — pair them with
//! [`super::mesh_sdf_raymarch::sample_signed_distance`] to combine several
//! fields at a shared sample point. Everything is transcendental-free (only
//! `min`, `max`, `abs`, and multiplication), matching Inigo Quilez's quadratic
//! smooth-minimum, so results are reproducible across machines.

/// Union of two signed distance values: the field of the combined solid is the
/// nearer of the two surfaces.
pub fn union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Intersection of two signed distance values: a point is inside only where it
/// is inside both solids, so the farther surface governs.
pub fn intersection(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Subtraction of `b` from `a`: carves the second solid out of the first by
/// intersecting with the complement (`-b`).
pub fn subtraction(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// Smooth union with fillet radius `k`, blending the two surfaces with a
/// quadratic polynomial so they merge without a crease.
///
/// Equivalent to [`union`] as `k` approaches zero, and never larger than the
/// hard union (the blend only rounds material inward). A non-positive `k`
/// falls back to the hard [`union`].
pub fn smooth_union(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return union(a, b);
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.min(b) - h * h * k * 0.25
}

/// Smooth intersection with fillet radius `k`: the dual of [`smooth_union`],
/// rounding the shared boundary of the two solids outward.
///
/// Equivalent to [`intersection`] as `k` approaches zero, and never smaller
/// than the hard intersection. A non-positive `k` falls back to the hard
/// [`intersection`].
pub fn smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return intersection(a, b);
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.max(b) + h * h * k * 0.25
}

/// Smooth subtraction with fillet radius `k`: carves `b` out of `a` with a
/// rounded seam, equal to a [`smooth_intersection`] with the complement.
///
/// Equivalent to [`subtraction`] as `k` approaches zero. A non-positive `k`
/// falls back to the hard [`subtraction`].
pub fn smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    smooth_intersection(a, -b, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_operators_match_min_max() {
        assert_eq!(union(2.0, 5.0), 2.0);
        assert_eq!(intersection(2.0, 5.0), 5.0);
        // Subtracting the farther solid keeps the first surface.
        assert_eq!(subtraction(2.0, 5.0), 2.0);
        // Subtracting an overlapping solid exposes its (negated) back face.
        assert_eq!(subtraction(5.0, 2.0), 5.0);
        assert_eq!(subtraction(1.0, -3.0), 3.0);
    }

    #[test]
    fn smooth_union_rounds_below_hard_union() {
        // At equal distances the blend is deepest: 1 - 1*0.5*0.25 = 0.875.
        assert!((smooth_union(1.0, 1.0, 0.5) - 0.875).abs() <= 1e-6);
        // The smooth result never exceeds the hard union.
        for &(a, b) in &[(1.0f32, 3.0f32), (-2.0, 0.5), (4.0, 4.1)] {
            assert!(smooth_union(a, b, 0.5) <= union(a, b) + 1e-6);
        }
    }

    #[test]
    fn smooth_intersection_rounds_above_hard_intersection() {
        // Dual of the union case: 1 + 1*0.5*0.25 = 1.125.
        assert!((smooth_intersection(1.0, 1.0, 0.5) - 1.125).abs() <= 1e-6);
        for &(a, b) in &[(1.0f32, 3.0f32), (-2.0, 0.5), (4.0, 4.1)] {
            assert!(smooth_intersection(a, b, 0.5) >= intersection(a, b) - 1e-6);
        }
    }

    #[test]
    fn small_radius_approaches_hard_operators() {
        let k = 1e-4;
        assert!((smooth_union(1.0, 3.0, k) - union(1.0, 3.0)).abs() <= 1e-3);
        assert!((smooth_intersection(1.0, 3.0, k) - intersection(1.0, 3.0)).abs() <= 1e-3);
    }

    #[test]
    fn non_positive_radius_falls_back_to_hard() {
        assert_eq!(smooth_union(1.0, 3.0, 0.0), union(1.0, 3.0));
        assert_eq!(smooth_intersection(1.0, 3.0, -1.0), intersection(1.0, 3.0));
        assert_eq!(smooth_subtraction(1.0, 3.0, 0.0), subtraction(1.0, 3.0));
    }

    #[test]
    fn smooth_subtraction_is_intersection_with_complement() {
        let k = 0.5;
        for &(a, b) in &[(1.0f32, 2.0f32), (-1.0, 0.5), (3.0, 3.2)] {
            assert_eq!(smooth_subtraction(a, b, k), smooth_intersection(a, -b, k));
        }
    }

    #[test]
    fn blends_are_symmetric_in_their_arguments() {
        let k = 0.75;
        for &(a, b) in &[(1.0f32, 2.5f32), (-0.5, 0.5), (4.0, 4.0)] {
            assert_eq!(smooth_union(a, b, k), smooth_union(b, a, k));
            assert_eq!(smooth_intersection(a, b, k), smooth_intersection(b, a, k));
        }
    }
}
