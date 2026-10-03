//! Per-level fidelity report for a level-of-detail (LOD) mesh chain.
//!
//! A collision or render cooker produces a chain of progressively coarser
//! meshes and must decide at what camera distance each level may be swapped in
//! without a visible or physical "pop". The deciding quantity is the geometric
//! error of each level relative to the original surface. This module measures
//! that error with the symmetric mesh distance
//! ([`measure_mesh_distance`](crate::collider::measure_mesh_distance)): for
//! every level it reports the Hausdorff (worst-case), mean, and root-mean-square
//! deviation from the base mesh, together with the level's triangle count. From
//! those numbers a caller derives a screen-space error bound and picks switch
//! distances, exactly as AAA pipelines do when authoring LOD transitions.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::hausdorff::{measure_mesh_distance, MeshDistanceParams};

/// A borrowed view of one LOD level's triangle mesh.
#[derive(Clone, Copy, Debug)]
pub struct LodMeshRef<'a> {
    /// The level's vertex positions.
    pub vertices: &'a [Vec3],
    /// The level's triangle indices into `vertices`.
    pub indices: &'a [[u32; 3]],
}

impl<'a> LodMeshRef<'a> {
    /// Convenience constructor.
    #[must_use]
    pub fn new(vertices: &'a [Vec3], indices: &'a [[u32; 3]]) -> Self {
        Self { vertices, indices }
    }
}

/// The measured error of a single LOD level against the base mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodLevelError {
    /// Zero-based position of this level in the supplied chain.
    pub level: usize,
    /// Number of triangles in this level.
    pub triangles: usize,
    /// Symmetric Hausdorff distance from the base surface (worst-case error).
    pub hausdorff: f32,
    /// Larger of the two directed mean distances from the base surface.
    pub mean: f32,
    /// Larger of the two directed root-mean-square distances from the base.
    pub rms: f32,
}

/// The fidelity profile of a full LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub struct LodErrorReport {
    /// Per-level error, in the order the levels were supplied.
    pub levels: Vec<LodLevelError>,
}

impl LodErrorReport {
    /// The largest Hausdorff error across all levels (the coarsest level's
    /// error for a well-ordered chain).
    #[must_use]
    pub fn max_hausdorff(&self) -> f32 {
        self.levels
            .iter()
            .map(|l| l.hausdorff)
            .fold(0.0_f32, f32::max)
    }

    /// Whether the Hausdorff error is non-decreasing as the chain coarsens.
    ///
    /// A healthy LOD chain never produces a coarser level that is *closer* to
    /// the original than a finer one; a violation flags a mis-ordered chain or
    /// a decimation bug. Chains with fewer than two levels are trivially
    /// monotonic.
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        self.levels
            .windows(2)
            .all(|w| w[1].hausdorff + 1.0e-6 >= w[0].hausdorff)
    }

    /// Number of levels in the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Whether the report has no levels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

/// Evaluates the geometric error of every LOD level against a base mesh.
///
/// For each level the symmetric mesh distance to `base` is measured and the
/// Hausdorff/mean/RMS deviations recorded. Returns `None` when the base mesh is
/// empty, when no levels are supplied, or when any single level's distance
/// measurement fails (empty level or a mesh a BVH cannot be built for). The
/// result is deterministic for a fixed seed.
#[must_use]
pub fn evaluate_lod_errors(
    base_vertices: &[Vec3],
    base_indices: &[[u32; 3]],
    levels: &[LodMeshRef<'_>],
    params: MeshDistanceParams,
) -> Option<LodErrorReport> {
    if base_vertices.is_empty() || base_indices.is_empty() || levels.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(levels.len());
    for (level, mesh) in levels.iter().enumerate() {
        let distance = measure_mesh_distance(
            base_vertices,
            base_indices,
            mesh.vertices,
            mesh.indices,
            params,
        )?;
        out.push(LodLevelError {
            level,
            triangles: mesh.indices.len(),
            hausdorff: distance.hausdorff(),
            mean: distance.max_mean(),
            rms: distance.max_rms(),
        });
    }

    Some(LodErrorReport { levels: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin as 12 outward-wound triangles.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        (verts, tris)
    }

    #[test]
    fn empty_base_or_no_levels_is_rejected() {
        let (verts, tris) = unit_cube();
        let params = MeshDistanceParams::default();
        let level = LodMeshRef::new(&verts, &tris);
        assert!(evaluate_lod_errors(&[], &[], &[level], params).is_none());
        assert!(evaluate_lod_errors(&verts, &tris, &[], params).is_none());
    }

    #[test]
    fn identical_level_has_zero_error() {
        let (verts, tris) = unit_cube();
        let params = MeshDistanceParams::default();
        let level = LodMeshRef::new(&verts, &tris);
        let report = evaluate_lod_errors(&verts, &tris, &[level], params).expect("one level");
        assert_eq!(report.len(), 1);
        assert_eq!(report.levels[0].triangles, 12);
        assert!(report.levels[0].hausdorff < 1.0e-5);
        assert!(report.is_monotonic());
    }

    #[test]
    fn coarser_levels_report_growing_error() {
        let (verts, tris) = unit_cube();
        // Three levels at increasing scale offsets: the farther a shell drifts
        // the larger its symmetric distance to the base must be.
        let l0: Vec<Vec3> = verts.iter().map(|v| *v * 1.05).collect();
        let l1: Vec<Vec3> = verts.iter().map(|v| *v * 1.25).collect();
        let l2: Vec<Vec3> = verts.iter().map(|v| *v * 1.60).collect();
        let levels = [
            LodMeshRef::new(&l0, &tris),
            LodMeshRef::new(&l1, &tris),
            LodMeshRef::new(&l2, &tris),
        ];
        let params = MeshDistanceParams {
            sample_count: 2048,
            seed: 5,
            include_vertices: true,
        };
        let report = evaluate_lod_errors(&verts, &tris, &levels, params).expect("three levels");
        assert_eq!(report.len(), 3);
        assert!(report.levels[0].hausdorff < report.levels[1].hausdorff);
        assert!(report.levels[1].hausdorff < report.levels[2].hausdorff);
        assert!(report.is_monotonic());
        assert!((report.max_hausdorff() - report.levels[2].hausdorff).abs() < 1.0e-6);
    }

    #[test]
    fn mis_ordered_chain_is_not_monotonic() {
        let (verts, tris) = unit_cube();
        let near: Vec<Vec3> = verts.iter().map(|v| *v * 1.05).collect();
        let far: Vec<Vec3> = verts.iter().map(|v| *v * 1.60).collect();
        // Coarse-then-fine ordering must be flagged as non-monotonic.
        let levels = [LodMeshRef::new(&far, &tris), LodMeshRef::new(&near, &tris)];
        let params = MeshDistanceParams {
            sample_count: 2048,
            seed: 5,
            include_vertices: true,
        };
        let report = evaluate_lod_errors(&verts, &tris, &levels, params).expect("two levels");
        assert!(!report.is_monotonic());
    }

    #[test]
    fn result_is_deterministic_for_fixed_seed() {
        let (verts, tris) = unit_cube();
        let scaled: Vec<Vec3> = verts.iter().map(|v| *v * 1.2).collect();
        let levels = [LodMeshRef::new(&scaled, &tris)];
        let params = MeshDistanceParams {
            sample_count: 512,
            seed: 42,
            include_vertices: true,
        };
        let a = evaluate_lod_errors(&verts, &tris, &levels, params).expect("run a");
        let b = evaluate_lod_errors(&verts, &tris, &levels, params).expect("run b");
        assert_eq!(a, b);
    }
}
