//! Mesh-to-mesh Hausdorff and RMS distance metrics.
//!
//! When a cooker decimates a collision mesh, builds a LOD chain, or fits a
//! convex proxy, it needs a scalar that answers "how far did the approximation
//! drift from the original surface?". The classic answer is the Hausdorff
//! distance: the largest distance from any point of one surface to the nearest
//! point of the other. The two-sided (symmetric) Hausdorff distance is the
//! larger of the two directed distances and is a true metric on compact sets.
//!
//! Exact Hausdorff distance between triangle soups is expensive, so the Metro
//! approach (Cignoni, Rocchini & Scopigno 1998) estimates it by densely
//! sampling one surface and measuring the distance from each sample to the
//! other surface via a closest-point query. This module follows that recipe:
//! area-weighted surface samples (optionally augmented with the source
//! vertices, where the extremum often lives) are projected onto the target with
//! [`MeshBvh::closest_point`](crate::collider::MeshBvh::closest_point). The
//! estimate is deterministic for a fixed seed and converges to the true
//! distance as the sample count grows. The same traversal yields the mean and
//! root-mean-square (RMS) deviation, which AAA cookers (`PhysX`, `Jolt`) report
//! alongside the peak error to gauge average fidelity.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::mesh_bvh::MeshBvh;
use crate::collider::surface_sampling::{sample_surface, SurfaceSampleParams};

/// Tuning for [`measure_mesh_distance`] and [`measure_directed_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDistanceParams {
    /// Number of area-weighted surface samples drawn from the source mesh.
    pub sample_count: usize,
    /// Seed for the deterministic area-weighted surface sampler.
    pub seed: u64,
    /// Also measure the distance at every source vertex. Surface sampling can
    /// miss sharp corners and spikes where the true extremum often sits, so
    /// including the vertices tightens the Hausdorff estimate at negligible
    /// cost for typical meshes.
    pub include_vertices: bool,
}

impl Default for MeshDistanceParams {
    /// 1024 samples with a fixed seed, source vertices included.
    fn default() -> Self {
        Self {
            sample_count: 1024,
            seed: 0x4852_4453_4446_4631,
            include_vertices: true,
        }
    }
}

/// A one-directional distance from a source surface to a target surface.
///
/// Every measurement is the Euclidean distance from a probe point on the source
/// to the nearest point anywhere on the target surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectedMeshDistance {
    /// Largest source-to-target distance seen (the directed Hausdorff estimate).
    pub max: f32,
    /// Mean source-to-target distance over all probes.
    pub mean: f32,
    /// Root-mean-square source-to-target distance over all probes.
    pub rms: f32,
    /// Number of probe points evaluated (surface samples plus any vertices).
    pub evaluated: usize,
}

/// The symmetric distance between two meshes, retaining both directed halves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshDistance {
    /// Directed distance from the first mesh to the second.
    pub forward: DirectedMeshDistance,
    /// Directed distance from the second mesh to the first.
    pub backward: DirectedMeshDistance,
}

impl MeshDistance {
    /// The symmetric (two-sided) Hausdorff distance: the larger of the two
    /// directed peak distances. This is the standard worst-case fidelity bound.
    #[must_use]
    pub fn hausdorff(&self) -> f32 {
        self.forward.max.max(self.backward.max)
    }

    /// The larger of the two directed mean distances.
    #[must_use]
    pub fn max_mean(&self) -> f32 {
        self.forward.mean.max(self.backward.mean)
    }

    /// The larger of the two directed RMS distances.
    #[must_use]
    pub fn max_rms(&self) -> f32 {
        self.forward.rms.max(self.backward.rms)
    }
}

/// Measures the directed distance from a source triangle mesh to a target
/// surface represented by a prebuilt [`MeshBvh`].
///
/// Draws `sample_count` area-weighted points on the source, optionally adds the
/// source vertices, and records the distance from each to its closest point on
/// the target. Returns `None` when the source is empty, when `sample_count` is
/// zero while `include_vertices` is `false` (no probes would exist), when
/// surface sampling fails, or when a closest-point query cannot be resolved.
/// The result is deterministic for a fixed seed.
#[must_use]
pub fn measure_directed_distance(
    source_vertices: &[Vec3],
    source_indices: &[[u32; 3]],
    target: &MeshBvh,
    params: MeshDistanceParams,
) -> Option<DirectedMeshDistance> {
    if source_vertices.is_empty() || source_indices.is_empty() {
        return None;
    }
    if params.sample_count == 0 && !params.include_vertices {
        return None;
    }

    let mut max = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut evaluated = 0_usize;

    let mut accumulate = |point: Vec3| -> Option<()> {
        let hit = target.closest_point(point)?;
        let distance = f64::from(hit.distance_sq).max(0.0).sqrt();
        if distance > max {
            max = distance;
        }
        sum += distance;
        sum_sq += distance * distance;
        evaluated += 1;
        Some(())
    };

    if params.sample_count > 0 {
        let samples = sample_surface(
            source_vertices,
            source_indices,
            SurfaceSampleParams {
                count: params.sample_count,
                seed: params.seed,
            },
        )?;
        for sample in samples {
            accumulate(sample.position)?;
        }
    }

    if params.include_vertices {
        for &vertex in source_vertices {
            accumulate(vertex)?;
        }
    }

    if evaluated == 0 {
        return None;
    }

    let count = evaluated as f64;
    Some(DirectedMeshDistance {
        max: max as f32,
        mean: (sum / count) as f32,
        rms: (sum_sq / count).sqrt() as f32,
        evaluated,
    })
}

/// Measures the symmetric distance between two triangle meshes.
///
/// Builds a [`MeshBvh`] over each mesh and evaluates the directed distance in
/// both directions (see [`measure_directed_distance`]). Returns `None` when
/// either mesh is empty, when a BVH cannot be built, or when either directed
/// measurement fails. The result is deterministic for a fixed seed; the two
/// directions use the same seed so a mesh compared against itself yields the
/// same probe set in both halves.
#[must_use]
pub fn measure_mesh_distance(
    a_vertices: &[Vec3],
    a_indices: &[[u32; 3]],
    b_vertices: &[Vec3],
    b_indices: &[[u32; 3]],
    params: MeshDistanceParams,
) -> Option<MeshDistance> {
    let a_bvh = MeshBvh::build(a_vertices, a_indices)?;
    let b_bvh = MeshBvh::build(b_vertices, b_indices)?;

    let forward = measure_directed_distance(a_vertices, a_indices, &b_bvh, params)?;
    let backward = measure_directed_distance(b_vertices, b_indices, &a_bvh, params)?;

    Some(MeshDistance { forward, backward })
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
    fn empty_inputs_are_rejected() {
        let (verts, tris) = unit_cube();
        let params = MeshDistanceParams::default();
        assert!(measure_mesh_distance(&[], &[], &verts, &tris, params).is_none());
        assert!(measure_mesh_distance(&verts, &tris, &[], &[], params).is_none());
    }

    #[test]
    fn no_probes_requested_is_rejected() {
        let (verts, tris) = unit_cube();
        let bvh = MeshBvh::build(&verts, &tris).expect("builds a bvh");
        let params = MeshDistanceParams {
            sample_count: 0,
            seed: 1,
            include_vertices: false,
        };
        assert!(measure_directed_distance(&verts, &tris, &bvh, params).is_none());
    }

    #[test]
    fn identical_meshes_have_zero_distance() {
        let (verts, tris) = unit_cube();
        let params = MeshDistanceParams::default();
        let d = measure_mesh_distance(&verts, &tris, &verts, &tris, params)
            .expect("measures identical meshes");
        assert!(d.hausdorff() < 1.0e-5, "hausdorff = {}", d.hausdorff());
        assert!(d.max_mean() < 1.0e-5, "mean = {}", d.max_mean());
        assert!(d.max_rms() < 1.0e-5, "rms = {}", d.max_rms());
        assert!(d.forward.evaluated > 0);
        assert!(d.backward.evaluated > 0);
    }

    #[test]
    fn translated_cube_distance_matches_offset() {
        let (verts, tris) = unit_cube();
        let shift = Vec3::new(0.5, 0.0, 0.0);
        let moved: Vec<Vec3> = verts.iter().map(|v| *v + shift).collect();
        let params = MeshDistanceParams {
            sample_count: 4096,
            seed: 7,
            include_vertices: true,
        };
        let d = measure_mesh_distance(&verts, &tris, &moved, &tris, params)
            .expect("measures translated cube");
        // The two unit cubes overlap after a 0.5 shift along x, so the farthest
        // a point on one face can be from the other surface is the shift
        // magnitude (a face that slid clear of its partner).
        assert!(
            (d.hausdorff() - 0.5).abs() < 0.05,
            "hausdorff = {}",
            d.hausdorff()
        );
    }

    #[test]
    fn scaled_cube_peak_hits_corner() {
        let (verts, tris) = unit_cube();
        // A cube half the size nested inside the unit cube: the farthest the big
        // cube drifts from the small one is at a corner, sqrt(3) * 0.5.
        let small: Vec<Vec3> = verts.iter().map(|v| *v * 0.5).collect();
        let params = MeshDistanceParams {
            sample_count: 4096,
            seed: 11,
            include_vertices: true,
        };
        let d = measure_mesh_distance(&verts, &tris, &small, &tris, params)
            .expect("measures nested cubes");
        let corner = (3.0_f32).sqrt() * 0.5;
        // The directed distance from the large cube to the small one peaks at a
        // corner; the reverse direction is at most 0.5 (the gap between faces).
        assert!(
            (d.forward.max - corner).abs() < 0.05,
            "forward max = {}, expected ~{}",
            d.forward.max,
            corner
        );
        assert!(
            d.backward.max <= 0.5 + 0.05,
            "backward max = {}",
            d.backward.max
        );
        assert!(
            (d.hausdorff() - corner).abs() < 0.05,
            "hausdorff = {}",
            d.hausdorff()
        );
    }

    #[test]
    fn result_is_deterministic_for_fixed_seed() {
        let (verts, tris) = unit_cube();
        let moved: Vec<Vec3> = verts
            .iter()
            .map(|v| *v + Vec3::new(0.3, 0.1, 0.0))
            .collect();
        let params = MeshDistanceParams {
            sample_count: 512,
            seed: 99,
            include_vertices: true,
        };
        let a = measure_mesh_distance(&verts, &tris, &moved, &tris, params).expect("first run");
        let b = measure_mesh_distance(&verts, &tris, &moved, &tris, params).expect("second run");
        assert_eq!(a, b);
    }
}
