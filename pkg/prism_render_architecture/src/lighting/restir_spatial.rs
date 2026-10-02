//! Spatial-neighbor admissibility gate for `ReSTIR` DI reuse.
//!
//! `ReSTIR` DI's spatial pass folds a pixel's screen-space neighbors into its
//! reservoir so each pixel effectively shades against far more lights than it
//! sampled. [`super::restir_temporal::resolve_di`]'s unbiased combine keeps that
//! correct for *any* neighbor set, but blindly folding every neighbor is still a
//! quality bug: a neighbor straddling a depth silhouette or a sharp normal
//! crease holds a light that is irrelevant (or occluded) at this pixel, so its
//! sample is almost always rejected by the target re-test — wasted work that
//! inflates variance, and on the biased fast path darkens edges. Production
//! `ReSTIR` (`RTXDI`, `UE`'s reservoir lighting) therefore gates spatial
//! neighbors by geometric similarity, exactly as temporal reuse gates the
//! reprojected history.
//!
//! This module is that gate, kept as a separate `CPU` contract for the same
//! reason [`super::restir_temporal::reproject_history`] is: picking *which*
//! neighbor texels to try (the screen-space disk / spiral sampling pattern) is a
//! `GPU` addressing job the caller performs, while deciding *which of those are
//! admissible* is the part that must match the `GPU` twin bit-for-bit. The
//! caller hands us this frame's candidate neighbors (reservoir + surface) and we
//! return the geometrically compatible subset, ready to feed the resolve.
//!
//! Filtering is a deterministic function of surface geometry alone — it never
//! inspects the held light — so dropping inadmissible neighbors cannot bias the
//! estimator; it only removes high-variance, mostly-rejected sources. Pure
//! classical Monte Carlo, no neural / learned / data-driven components.

use alloc::vec::Vec;

use super::restir_temporal::{GeomReservoir, SurfaceGeometry};

/// Default relative view-depth tolerance for accepting a spatial neighbor.
/// A neighbor is rejected when `|z_n − z_c| > tol · z_c`.
pub const DEFAULT_SPATIAL_DEPTH_REL_TOLERANCE: f32 = 0.1;

/// Default minimum normal agreement (cosine) for accepting a spatial neighbor,
/// ≈25°.
pub const DEFAULT_SPATIAL_NORMAL_COS_TOLERANCE: f32 = 0.906;

/// Branchless `f32` magnitude (this crate is `no_std`; keep the reference
/// portable to the `GPU` twin rather than leaning on the `std` `f32::abs`).
fn abs_f32(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Dot product of two 3-vectors (plain multiply / add — no transcendentals).
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Tunables for the spatial-neighbor admissibility gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialParams {
    /// Relative view-depth tolerance: a neighbor is rejected when its depth
    /// differs from the center's by more than `depth_rel_tolerance · z_center`.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) between center and neighbor normals.
    pub normal_cos_tolerance: f32,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            depth_rel_tolerance: DEFAULT_SPATIAL_DEPTH_REL_TOLERANCE,
            normal_cos_tolerance: DEFAULT_SPATIAL_NORMAL_COS_TOLERANCE,
        }
    }
}

/// Whether a neighbor surface is geometrically compatible with the center pixel
/// and may be folded into its reservoir.
///
/// A neighbor is admissible when both surfaces are valid (finite, in front of
/// the camera), their view depths agree to within `params.depth_rel_tolerance`
/// (relative to the center depth), and their normals agree to within
/// `params.normal_cos_tolerance`. This is the same disocclusion-style test
/// [`super::restir_temporal::reproject_history`] applies to temporal history,
/// evaluated here between two same-frame pixels.
#[must_use]
pub fn spatial_admissible(
    center: SurfaceGeometry,
    neighbor: SurfaceGeometry,
    params: SpatialParams,
) -> bool {
    if !center.is_valid() || !neighbor.is_valid() {
        return false;
    }
    let depth_diff = abs_f32(neighbor.view_depth - center.view_depth);
    if depth_diff > params.depth_rel_tolerance * center.view_depth {
        return false;
    }
    dot3(center.normal, neighbor.normal) >= params.normal_cos_tolerance
}

/// Selects up to `max` geometrically admissible, non-empty neighbors for the
/// center pixel, preserving the caller's neighbor order.
///
/// `neighbors` is this frame's candidate neighbor set (each a finalized
/// reservoir paired with the surface it was produced on); the caller is
/// responsible for having chosen *which* screen texels those are. We keep only
/// the ones that pass [`spatial_admissible`] and still hold a sample, stopping
/// once `max` have been collected (typically `budget.spatial_neighbors`). The
/// returned reservoirs are ready to pass straight to
/// [`super::restir_temporal::resolve_di`]'s `spatial` argument.
#[must_use]
pub fn gather_admissible_neighbors(
    center: SurfaceGeometry,
    neighbors: &[GeomReservoir],
    max: usize,
    params: SpatialParams,
) -> Vec<GeomReservoir> {
    let mut out: Vec<GeomReservoir> = Vec::with_capacity(max.min(neighbors.len()));
    if max == 0 {
        return out;
    }
    for neighbor in neighbors {
        if out.len() >= max {
            break;
        }
        if neighbor.reservoir.is_empty() {
            continue;
        }
        if spatial_admissible(center, neighbor.geometry, params) {
            out.push(*neighbor);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::restir_di::{DiCandidate, DiReservoir};

    const FRONT: [f32; 3] = [0.0, 0.0, 1.0];
    const SIDE: [f32; 3] = [1.0, 0.0, 0.0];

    fn surface(depth: f32, normal: [f32; 3]) -> SurfaceGeometry {
        SurfaceGeometry::new(depth, normal)
    }

    fn occupied(light: u32) -> DiReservoir {
        let mut r = DiReservoir::empty();
        r.stream(
            DiCandidate {
                light_index: light,
                target_pdf: 1.0,
                source_pdf: 1.0,
            },
            0.5,
        );
        r.finalize();
        r
    }

    #[test]
    fn admits_matching_surface() {
        let c = surface(10.0, FRONT);
        // Within 10% depth and well inside the normal cone.
        assert!(spatial_admissible(
            c,
            surface(10.5, FRONT),
            SpatialParams::default()
        ));
    }

    #[test]
    fn rejects_depth_silhouette() {
        let c = surface(10.0, FRONT);
        // 50% deeper → across a silhouette, must be rejected.
        assert!(!spatial_admissible(
            c,
            surface(15.0, FRONT),
            SpatialParams::default()
        ));
    }

    #[test]
    fn rejects_normal_crease() {
        let c = surface(10.0, FRONT);
        // Orthogonal normal (90°) is far outside the ≈25° cone.
        assert!(!spatial_admissible(
            c,
            surface(10.0, SIDE),
            SpatialParams::default()
        ));
    }

    #[test]
    fn rejects_invalid_surface() {
        let c = surface(10.0, FRONT);
        // Background / sky neighbor (non-positive depth) never reuses.
        assert!(!spatial_admissible(
            c,
            surface(0.0, FRONT),
            SpatialParams::default()
        ));
        // A center that is itself background rejects everything.
        assert!(!spatial_admissible(
            surface(-1.0, FRONT),
            surface(10.0, FRONT),
            SpatialParams::default()
        ));
    }

    #[test]
    fn gather_filters_empty_and_inadmissible_and_preserves_order() {
        let center = surface(10.0, FRONT);
        let neighbors = [
            GeomReservoir::new(occupied(0), surface(10.2, FRONT)), // admissible
            GeomReservoir::new(DiReservoir::empty(), surface(10.0, FRONT)), // empty → skip
            GeomReservoir::new(occupied(1), surface(30.0, FRONT)), // depth mismatch → skip
            GeomReservoir::new(occupied(2), surface(9.8, FRONT)),  // admissible
            GeomReservoir::new(occupied(3), surface(10.0, SIDE)),  // normal mismatch → skip
        ];
        let kept = gather_admissible_neighbors(center, &neighbors, 8, SpatialParams::default());
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].reservoir.light_index(), 0);
        assert_eq!(kept[1].reservoir.light_index(), 2);
    }

    #[test]
    fn gather_respects_max_cap() {
        let center = surface(10.0, FRONT);
        let neighbors = [
            GeomReservoir::new(occupied(0), surface(10.1, FRONT)),
            GeomReservoir::new(occupied(1), surface(10.1, FRONT)),
            GeomReservoir::new(occupied(2), surface(10.1, FRONT)),
        ];
        let kept = gather_admissible_neighbors(center, &neighbors, 2, SpatialParams::default());
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].reservoir.light_index(), 0);
        assert_eq!(kept[1].reservoir.light_index(), 1);
        // max = 0 selects nothing.
        assert!(
            gather_admissible_neighbors(center, &neighbors, 0, SpatialParams::default()).is_empty()
        );
    }
}
