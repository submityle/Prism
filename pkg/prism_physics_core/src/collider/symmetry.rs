//! Mirror-symmetry detection for triangle meshes.
//!
//! Many authored assets are built with a bilateral (mirror) symmetry, and a
//! cooker can exploit it: a symmetric collision hull can be stored once and
//! mirrored, instanced, or decimated with a symmetry constraint that keeps the
//! two halves matched. Detecting the dominant mirror plane is a classical
//! point-cloud problem solved by aligning the search to the principal axes of
//! the shape: for a bilaterally symmetric body one principal axis is normal to
//! the symmetry plane, so the three candidate planes through the centroid (one
//! per principal axis) are the only ones worth testing.
//!
//! This module draws area-weighted surface samples
//! ([`sample_surface`](crate::collider::sample_surface)), derives the principal
//! frame from their covariance via
//! [`principal_axes`](crate::collider::principal_axes), and scores each
//! candidate plane by reflecting every sample across it and measuring the
//! distance back to the surface with
//! [`MeshBvh::closest_point`](crate::collider::MeshBvh::closest_point). The
//! residual, normalised by the bounding diagonal, is the symmetry error; the
//! plane with the smallest error is the dominant mirror plane. The result is
//! deterministic for a fixed seed.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

use crate::collider::inertia::principal_axes;
use crate::collider::mesh_bvh::MeshBvh;
use crate::collider::surface_sampling::{sample_surface, SurfaceSampleParams};

/// Tuning for [`detect_mirror_symmetry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymmetryParams {
    /// Number of area-weighted surface samples used for both the principal-axis
    /// fit and the reflection test.
    pub sample_count: usize,
    /// Seed for the deterministic area-weighted surface sampler.
    pub seed: u64,
}

impl Default for SymmetryParams {
    /// 1024 samples with a fixed seed.
    fn default() -> Self {
        Self {
            sample_count: 1024,
            seed: 0x5359_4D4D_4554_5259,
        }
    }
}

/// A candidate mirror plane and how closely the mesh matches its reflection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetryPlane {
    /// Unit normal of the plane.
    pub normal: Vec3,
    /// A point on the plane (the bounding-box centre).
    pub point: Vec3,
    /// Root-mean-square distance from reflected samples back to the surface, in
    /// mesh units.
    pub rms_error: f32,
    /// Largest distance from any reflected sample back to the surface.
    pub max_error: f32,
    /// `rms_error` divided by the bounding diagonal, in `[0, 1]` for typical
    /// meshes. Scale-independent, so it is the value to threshold on.
    pub normalized_error: f32,
}

impl SymmetryPlane {
    /// A `[0, 1]` symmetry score (`1 - normalized_error`, clamped). Higher is
    /// more symmetric.
    #[must_use]
    pub fn score(&self) -> f32 {
        (1.0 - self.normalized_error).clamp(0.0, 1.0)
    }

    /// Whether the mesh is mirror-symmetric about this plane within `tolerance`
    /// (a normalised-error threshold, e.g. `0.02`).
    #[must_use]
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        self.normalized_error <= tolerance
    }
}

/// The mirror-symmetry analysis of a mesh over its three principal planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshSymmetry {
    /// The three candidate planes, one per principal axis.
    pub planes: [SymmetryPlane; 3],
    /// Index into [`planes`](MeshSymmetry::planes) of the best (lowest-error)
    /// plane.
    pub best_index: usize,
}

impl MeshSymmetry {
    /// The dominant (lowest-error) mirror plane.
    #[must_use]
    pub fn best(&self) -> SymmetryPlane {
        self.planes[self.best_index]
    }

    /// Whether the best plane is symmetric within `tolerance`.
    #[must_use]
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        self.best().is_symmetric(tolerance)
    }
}

/// Detects the dominant mirror plane of a triangle mesh.
///
/// Returns `None` when the mesh is empty, when `sample_count` is zero, when a
/// BVH cannot be built, when surface sampling fails, or when the shape is
/// degenerate (zero bounding diagonal). The analysis tests the three planes
/// through the sample centroid whose normals are the mesh's principal axes and
/// is deterministic for a fixed seed.
#[must_use]
pub fn detect_mirror_symmetry(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: SymmetryParams,
) -> Option<MeshSymmetry> {
    if vertices.is_empty() || indices.is_empty() || params.sample_count == 0 {
        return None;
    }

    let bvh = MeshBvh::build(vertices, indices)?;
    let (bmin, bmax) = bvh.local_aabb();
    let diagonal = (bmax - bmin).length();
    if !(diagonal.is_finite() && diagonal > 0.0) {
        return None;
    }

    let samples = sample_surface(
        vertices,
        indices,
        SurfaceSampleParams {
            count: params.sample_count,
            seed: params.seed,
        },
    )?;
    if samples.is_empty() {
        return None;
    }

    // Use the bounding-box centre as the plane anchor: for a mirror-symmetric
    // body the symmetry plane passes exactly through it, so this is far less
    // noisy than a Monte-Carlo sample centroid (whose error shifts the plane
    // and inflates the reflection residual).
    let n = samples.len() as f64;
    let centroid = (bmin + bmax) * 0.5;

    // Symmetric covariance of the mesh vertices about the centre. Using the
    // exact vertices (not Monte-Carlo samples) keeps the principal frame
    // noise-free: for a symmetric body the off-diagonal terms vanish and the
    // axes land exactly on the symmetry-plane normals, which a sampled
    // covariance only approximates and tilts by several degrees.
    let vcount = vertices.len() as f64;
    let (mut xx, mut yy, mut zz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut xy, mut xz, mut yz) = (0.0_f64, 0.0_f64, 0.0_f64);
    for v in vertices {
        let dx = f64::from(v.x - centroid.x);
        let dy = f64::from(v.y - centroid.y);
        let dz = f64::from(v.z - centroid.z);
        xx += dx * dx;
        yy += dy * dy;
        zz += dz * dz;
        xy += dx * dy;
        xz += dx * dz;
        yz += dy * dz;
    }
    let covariance = Mat3::from_cols(
        Vec3::new(
            (xx / vcount) as f32,
            (xy / vcount) as f32,
            (xz / vcount) as f32,
        ),
        Vec3::new(
            (xy / vcount) as f32,
            (yy / vcount) as f32,
            (yz / vcount) as f32,
        ),
        Vec3::new(
            (xz / vcount) as f32,
            (yz / vcount) as f32,
            (zz / vcount) as f32,
        ),
    );
    let frame = principal_axes(covariance);

    // Score each principal axis as a candidate mirror-plane normal.
    let mut planes = [SymmetryPlane {
        normal: Vec3::X,
        point: centroid,
        rms_error: 0.0,
        max_error: 0.0,
        normalized_error: 0.0,
    }; 3];
    let axes = [
        frame.axes.x_axis.normalize_or_zero(),
        frame.axes.y_axis.normalize_or_zero(),
        frame.axes.z_axis.normalize_or_zero(),
    ];

    for (slot, &normal) in axes.iter().enumerate() {
        // A degenerate axis cannot define a plane; mark it maximally asymmetric.
        if normal.length_squared() < 0.5 {
            planes[slot] = SymmetryPlane {
                normal: Vec3::X,
                point: centroid,
                rms_error: diagonal,
                max_error: diagonal,
                normalized_error: 1.0,
            };
            continue;
        }

        let mut sum_sq = 0.0_f64;
        let mut max = 0.0_f64;
        for s in &samples {
            // Reflect the sample across the candidate plane and measure how far
            // the reflection lands from the surface.
            let offset = (s.position - centroid).dot(normal);
            let reflected = s.position - normal * (2.0 * offset);
            if let Some(hit) = bvh.closest_point(reflected) {
                let d = f64::from(hit.distance_sq).max(0.0).sqrt();
                sum_sq += d * d;
                if d > max {
                    max = d;
                }
            }
        }
        let rms = (sum_sq / n).sqrt();
        planes[slot] = SymmetryPlane {
            normal,
            point: centroid,
            rms_error: rms as f32,
            max_error: max as f32,
            normalized_error: (rms / f64::from(diagonal)) as f32,
        };
    }

    let best_index = planes
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.normalized_error.total_cmp(&b.1.normalized_error))
        .map_or(0, |(i, _)| i);

    Some(MeshSymmetry { planes, best_index })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box of half-extents `(hx, hy, hz)` as 12 triangles.
    fn box_mesh(hx: f32, hy: f32, hz: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
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
    fn empty_or_zero_samples_rejected() {
        let (verts, tris) = box_mesh(1.0, 1.0, 1.0);
        assert!(detect_mirror_symmetry(&[], &[], SymmetryParams::default()).is_none());
        let zero = SymmetryParams {
            sample_count: 0,
            seed: 1,
        };
        assert!(detect_mirror_symmetry(&verts, &tris, zero).is_none());
    }

    #[test]
    fn distinct_extent_box_is_symmetric_about_every_principal_plane() {
        // Three distinct extents give a well-separated principal frame whose
        // axes align to the box faces, so all three principal planes are true
        // mirror planes. (An exact cube has a degenerate frame and is handled
        // by the elongated-box and best-plane paths instead.)
        let (verts, tris) = box_mesh(3.0, 2.0, 1.0);
        let sym = detect_mirror_symmetry(&verts, &tris, SymmetryParams::default())
            .expect("box has symmetry");
        for plane in &sym.planes {
            assert!(
                plane.is_symmetric(0.02),
                "plane normalized error {} too high",
                plane.normalized_error
            );
        }
        assert!(sym.is_symmetric(0.02));
        assert!(sym.best().score() > 0.95);
    }

    #[test]
    fn elongated_box_is_still_mirror_symmetric() {
        // A 3x1x1 box is mirror-symmetric about all three principal planes; the
        // detector must find a near-zero best error regardless of elongation.
        let (verts, tris) = box_mesh(3.0, 1.0, 1.0);
        let sym = detect_mirror_symmetry(&verts, &tris, SymmetryParams::default())
            .expect("box has symmetry");
        assert!(
            sym.best().is_symmetric(0.02),
            "best normalized error = {}",
            sym.best().normalized_error
        );
    }

    #[test]
    fn asymmetric_wedge_has_a_poor_plane() {
        // A tetrahedron-like wedge is far from mirror-symmetric; at least one
        // principal plane must score a substantial residual.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let sym = detect_mirror_symmetry(&verts, &tris, SymmetryParams::default())
            .expect("wedge analyses");
        let worst = sym
            .planes
            .iter()
            .map(|p| p.normalized_error)
            .fold(0.0_f32, f32::max);
        assert!(worst > 0.05, "worst normalized error = {}", worst);
    }

    #[test]
    fn result_is_deterministic_for_fixed_seed() {
        let (verts, tris) = box_mesh(2.0, 1.0, 1.5);
        let params = SymmetryParams {
            sample_count: 512,
            seed: 123,
        };
        let a = detect_mirror_symmetry(&verts, &tris, params).expect("run a");
        let b = detect_mirror_symmetry(&verts, &tris, params).expect("run b");
        assert_eq!(a, b);
    }
}
