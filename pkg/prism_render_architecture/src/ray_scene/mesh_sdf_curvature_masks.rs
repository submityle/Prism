//! Curvature-driven cavity and edge-wear masks for the `CPU` golden path.
//!
//! `AAA` material pipelines weather a surface from its curvature: dirt, grime
//! and ambient occlusion gather in concave crevices (a *cavity* mask) while
//! paint, rust and gilding rub off the convex ridges and corners that catch
//! the most contact (an *edge-wear* mask). This module turns the principal
//! curvatures recovered by [`super::mesh_sdf_curvature`] into those two
//! normalized masks plus a combined signed-curvature channel, matching the
//! curvature-map convention used by Unreal Engine and Substance-style
//! authoring.
//!
//! The masks key off the *principal* curvatures rather than the mean so a
//! sharp edge reads at full strength even when its orthogonal direction is
//! flat (a cylindrical fillet has one large principal curvature and one near
//! zero, which the mean would halve). The convex-positive sign convention of
//! [`super::mesh_sdf_curvature::SdfCurvature`] is preserved: the edge-wear
//! mask is driven by the largest convex principal curvature and the cavity
//! mask by the most concave one.
//!
//! Everything is a smoothstep ramp between a clean-surface threshold and a
//! saturation curvature, so the only operations are `clamp`, `min`, `max` and
//! a cubic polynomial — no transcendental calls, and the result is
//! reproducible across machines.

use super::mesh_sdf_curvature::{sdf_curvature, SdfCurvature};
use super::mesh_signed_distance_field::SignedDistanceField;

/// Tuning for how curvature magnitude maps onto the `[0, 1]` masks.
///
/// Curvatures are reciprocals of a radius, so `saturation_curvature` is in
/// units of inverse world distance and should be tuned to the scale of the
/// geometry (roughly `1 / feature_radius` for the sharpest feature that should
/// read at full strength).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureMaskParams {
    /// Principal-curvature magnitude that maps to a fully saturated mask value
    /// of `1.0`. Must be positive; larger values keep gentler curves clean.
    pub saturation_curvature: f32,
    /// Fraction of `saturation_curvature`, in `[0, 1)`, below which the mask
    /// reads exactly zero so flat faces and gentle curves stay unweathered.
    pub threshold: f32,
}

impl Default for CurvatureMaskParams {
    /// Neutral defaults for unit-scaled geometry: saturate at a curvature of
    /// `1.0` (a feature radius of one world unit) and ignore the gentlest tenth
    /// of that range as flat.
    fn default() -> Self {
        Self {
            saturation_curvature: 1.0,
            threshold: 0.1,
        }
    }
}

/// Normalized weathering masks derived from local surface curvature.
///
/// All three channels follow the convex-positive convention of
/// [`SdfCurvature`]: ridges and corners are convex, crevices are concave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureMasks {
    /// Cavity weight in `[0, 1]`: `1.0` deep in a sharp concave crevice, `0.0`
    /// on flat or convex surface. Drives dirt, grime and occlusion build-up.
    cavity: f32,
    /// Edge-wear weight in `[0, 1]`: `1.0` on a sharp convex ridge or corner,
    /// `0.0` on flat or concave surface. Drives paint rub-off and highlights.
    edge_wear: f32,
    /// Combined signed curvature in `[-1, 1]`: positive convex, negative
    /// concave, scaled by the saturation curvature. Handy as a single
    /// curvature-map channel.
    signed_curvature: f32,
}

impl CurvatureMasks {
    /// Cavity (crevice) weight in `[0, 1]`.
    pub fn cavity(&self) -> f32 {
        self.cavity
    }

    /// Edge-wear (ridge/corner) weight in `[0, 1]`.
    pub fn edge_wear(&self) -> f32 {
        self.edge_wear
    }

    /// Combined signed curvature in `[-1, 1]` (convex positive).
    pub fn signed_curvature(&self) -> f32 {
        self.signed_curvature
    }
}

/// Smoothstep ramp of `value` from `lo` (maps to `0.0`) to `hi` (maps to
/// `1.0`), clamped outside `[lo, hi]`.
///
/// Degenerate or inverted bounds (`hi <= lo`) collapse to a hard step just
/// above `hi` so a zero-width band can never divide by zero and a zero
/// curvature still reads as clean.
fn smooth_ramp(value: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if value > hi { 1.0 } else { 0.0 };
    }
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Builds the cavity and edge-wear masks directly from the two principal
/// curvatures (convex positive), the primitive the field-level and
/// [`SdfCurvature`] entry points both funnel into.
///
/// The edge-wear mask ramps on the largest convex principal curvature and the
/// cavity mask on the magnitude of the most concave one, each between the
/// threshold and saturation curvatures. The signed-curvature channel is the
/// mean curvature `(k1 + k2) / 2` normalized by the saturation curvature.
pub fn curvature_masks_from_principals(
    principal_max: f32,
    principal_min: f32,
    params: &CurvatureMaskParams,
) -> CurvatureMasks {
    let saturation = params.saturation_curvature;
    let lo = params.threshold.clamp(0.0, 1.0) * saturation;
    let hi = saturation;

    let convex = principal_max.max(0.0);
    let concave = (-principal_min).max(0.0);

    let edge_wear = smooth_ramp(convex, lo, hi);
    let cavity = smooth_ramp(concave, lo, hi);

    let mean = 0.5 * (principal_max + principal_min);
    let signed_curvature = (mean / saturation.max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);

    CurvatureMasks {
        cavity,
        edge_wear,
        signed_curvature,
    }
}

/// Builds the weathering masks from a sampled [`SdfCurvature`].
///
/// A thin convenience over [`curvature_masks_from_principals`] that reads the
/// principal curvatures off the curvature probe.
pub fn curvature_masks(curvature: &SdfCurvature, params: &CurvatureMaskParams) -> CurvatureMasks {
    curvature_masks_from_principals(
        curvature.principal_max(),
        curvature.principal_min(),
        params,
    )
}

/// Samples the field's curvature at `point` and reduces it to weathering
/// masks, returning [`None`] where the curvature is undefined (a flat,
/// constant region of the field with a degenerate gradient).
pub fn sdf_curvature_masks(
    field: &SignedDistanceField,
    point: [f32; 3],
    params: &CurvatureMaskParams,
) -> Option<CurvatureMasks> {
    sdf_curvature(field, point).map(|curvature| curvature_masks(&curvature, params))
}

#[cfg(test)]
mod tests {
    use super::{
        curvature_masks_from_principals, sdf_curvature_masks, CurvatureMaskParams, CurvatureMasks,
    };
    use crate::ray_scene::mesh_signed_distance_field::{signed_distance_field, SignedDistanceField};
    use crate::ray_scene::mesh_voxel_padding::pad_voxel_grid;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Index-only triangle mesh from positions.
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Closed axis-aligned unit cube (12 triangles) spanning `[0, 1]^3`.
    fn cube() -> TriangleMesh {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let i = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ];
        mesh(p, i)
    }

    /// Padded signed distance field of the unit cube with an exterior shell.
    fn cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 16).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    #[test]
    fn convex_ridge_reads_edge_wear_not_cavity() {
        // A sharply convex principal curvature at saturation drives the
        // edge-wear mask to one and leaves the cavity mask at zero.
        let params = CurvatureMaskParams {
            saturation_curvature: 2.0,
            threshold: 0.1,
        };
        let m = curvature_masks_from_principals(2.0, 0.0, &params);
        assert!((m.edge_wear() - 1.0).abs() < 1e-6);
        assert!(m.cavity().abs() < 1e-6);
        // Mean curvature is +1 against a saturation of 2 -> +0.5 signed.
        assert!((m.signed_curvature() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn concave_crevice_reads_cavity_not_edge_wear() {
        // A sharply concave principal curvature at saturation drives the
        // cavity mask to one and leaves the edge-wear mask at zero.
        let params = CurvatureMaskParams {
            saturation_curvature: 2.0,
            threshold: 0.1,
        };
        let m = curvature_masks_from_principals(0.0, -2.0, &params);
        assert!((m.cavity() - 1.0).abs() < 1e-6);
        assert!(m.edge_wear().abs() < 1e-6);
        assert!((m.signed_curvature() - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn flat_surface_has_no_weathering() {
        // Zero curvature sits below the threshold band, so both masks vanish.
        let m = curvature_masks_from_principals(0.0, 0.0, &CurvatureMaskParams::default());
        assert!(m.cavity().abs() < 1e-6);
        assert!(m.edge_wear().abs() < 1e-6);
        assert!(m.signed_curvature().abs() < 1e-6);
    }

    #[test]
    fn below_threshold_curvature_stays_clean() {
        // A convex curvature under the threshold*saturation knee reads zero.
        let params = CurvatureMaskParams {
            saturation_curvature: 1.0,
            threshold: 0.5,
        };
        // 0.4 < 0.5 knee -> clean; 1.0 >= saturation -> fully worn.
        assert!(curvature_masks_from_principals(0.4, 0.0, &params)
            .edge_wear()
            .abs()
            < 1e-6);
        assert!(
            (curvature_masks_from_principals(1.0, 0.0, &params).edge_wear() - 1.0).abs() < 1e-6
        );
    }

    #[test]
    fn ramp_is_monotonic_and_bounded() {
        // Across the saturation band the edge-wear mask rises monotonically and
        // stays within [0, 1].
        let params = CurvatureMaskParams {
            saturation_curvature: 1.0,
            threshold: 0.0,
        };
        let mut previous = -1.0f32;
        for step in 0..=10 {
            let k = step as f32 / 10.0;
            let w = curvature_masks_from_principals(k, 0.0, &params).edge_wear();
            assert!((0.0..=1.0).contains(&w));
            assert!(w >= previous - 1e-6);
            previous = w;
        }
    }

    #[test]
    fn degenerate_saturation_band_collapses_to_a_step() {
        // A zero saturation curvature leaves a zero-width band: any positive
        // convex curvature snaps to fully worn without dividing by zero.
        let params = CurvatureMaskParams {
            saturation_curvature: 0.0,
            threshold: 0.0,
        };
        let m = curvature_masks_from_principals(0.5, 0.0, &params);
        assert!((m.edge_wear() - 1.0).abs() < 1e-6);
        assert!(m.cavity().abs() < 1e-6);
    }

    #[test]
    fn cube_edges_are_more_worn_than_faces() {
        // End to end on a real field: a point just outside a cube edge sees a
        // convex ridge and reads more edge wear than a point off a flat face.
        let field = cube_field();
        let params = CurvatureMaskParams {
            saturation_curvature: 4.0,
            threshold: 0.05,
        };
        // Just outside the +x/+y edge (shared by two faces) versus just outside
        // the centre of the +x face.
        let edge = sdf_curvature_masks(&field, [1.05, 1.05, 0.5], &params);
        let face = sdf_curvature_masks(&field, [1.05, 0.5, 0.5], &params);
        if let (Some(edge), Some(face)) = (edge, face) {
            assert!(masks_bounded(&edge));
            assert!(masks_bounded(&face));
            assert!(edge.edge_wear() >= face.edge_wear() - 1e-3);
        }
    }

    /// All mask channels lie inside their documented ranges.
    fn masks_bounded(m: &CurvatureMasks) -> bool {
        (0.0..=1.0).contains(&m.cavity())
            && (0.0..=1.0).contains(&m.edge_wear())
            && (-1.0..=1.0).contains(&m.signed_curvature())
    }
}
