//! Shell-thickness estimation by inward ray casting.
//!
//! Many collision assets are hollow or thin-walled, and a cooker needs to know
//! how thin before it trusts a signed-distance field, picks a voxel size, or
//! warns that a wall is too thin to collide reliably. The classic estimate
//! casts a ray *inward* from each surface point along the reverse surface
//! normal and measures the distance to the opposite wall; the distribution of
//! those distances is the local shell thickness. AAA cookers (`PhysX`, `Jolt`)
//! expose exactly this diagnostic to flag paper-thin geometry.
//!
//! Surface points are drawn with the area-weighted sampler
//! ([`sample_surface`](crate::collider::sample_surface)) and traced against a
//! [`MeshBvh`](crate::collider::MeshBvh), so the estimate is deterministic for a
//! fixed seed. This is pure triangle-soup geometry with no coupling to the
//! collision pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::mesh_bvh::MeshBvh;
use crate::collider::surface_sampling::{sample_surface, SurfaceSampleParams};

/// Minimum origin offset (as a fraction of the mesh diagonal) used to push ray
/// origins just inside the surface so they do not re-hit their own triangle.
const ORIGIN_OFFSET_FRACTION: f32 = 1.0e-4;

/// Tuning for [`estimate_shell_thickness`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellThicknessParams {
    /// Number of surface points to probe.
    pub sample_count: usize,
    /// Seed for the deterministic area-weighted surface sampler.
    pub seed: u64,
    /// Optional cap on the inward ray length. Hits beyond this distance are
    /// ignored. `None` uses the mesh's bounding-box diagonal.
    pub max_thickness: Option<f32>,
}

impl Default for ShellThicknessParams {
    /// 256 samples with a fixed seed and no explicit thickness cap.
    fn default() -> Self {
        Self {
            sample_count: 256,
            seed: 0x5151_2764_9ABC_DEF0,
            max_thickness: None,
        }
    }
}

/// A single probed surface point and the wall thickness measured there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThicknessSample {
    /// The probed surface point.
    pub position: Vec3,
    /// Distance along the inward normal to the opposite wall.
    pub thickness: f32,
}

/// The distribution of shell thickness over the probed surface points.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellThickness {
    /// Per-sample thickness readings (only samples with a valid inward hit).
    pub samples: Vec<ThicknessSample>,
    /// Smallest measured thickness (the thinnest wall found).
    pub min: f32,
    /// Largest measured thickness.
    pub max: f32,
    /// Mean measured thickness.
    pub mean: f32,
    /// Number of samples that produced a valid inward hit.
    pub evaluated: usize,
    /// Number of samples requested (including misses).
    pub requested: usize,
}

impl ShellThickness {
    /// Number of samples that produced a valid thickness reading.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.evaluated
    }

    /// Fraction of requested samples that produced a valid reading, in `[0, 1]`.
    #[must_use]
    pub fn coverage(&self) -> f32 {
        if self.requested == 0 {
            0.0
        } else {
            self.evaluated as f32 / self.requested as f32
        }
    }

    /// Whether no valid thickness readings were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Estimates the shell-thickness distribution of a triangle mesh.
///
/// Casts an inward ray from each of `sample_count` area-weighted surface points
/// and records the distance to the opposite wall. Returns `None` when the mesh
/// is empty, when `sample_count` is zero, when a BVH cannot be built, when
/// surface sampling fails, or when no sample produces a valid inward hit (for
/// example an open sheet or an outward-only cast). Assumes the input is wound so
/// surface normals point outward. The result is deterministic for a fixed seed.
#[must_use]
pub fn estimate_shell_thickness(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: ShellThicknessParams,
) -> Option<ShellThickness> {
    if vertices.is_empty() || indices.is_empty() || params.sample_count == 0 {
        return None;
    }

    let bvh = MeshBvh::build(vertices, indices)?;
    let (bmin, bmax) = bvh.local_aabb();
    let diagonal = (bmax - bmin).length();
    if !(diagonal.is_finite() && diagonal > 0.0) {
        return None;
    }
    let offset = diagonal * ORIGIN_OFFSET_FRACTION;
    let max_time = params.max_thickness.unwrap_or(diagonal).max(offset);

    let surface = sample_surface(
        vertices,
        indices,
        SurfaceSampleParams {
            count: params.sample_count,
            seed: params.seed,
        },
    )?;

    let mut samples = Vec::new();
    let mut sum = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for probe in &surface {
        let inward = (-probe.normal).normalize_or_zero();
        if inward == Vec3::ZERO {
            continue;
        }
        // Push the origin just inside so the ray does not graze its own face.
        let origin = probe.position + inward * offset;
        let Some(hit) = bvh.ray_cast(origin, inward, max_time) else {
            continue;
        };
        let thickness = hit.time + offset;
        samples.push(ThicknessSample {
            position: probe.position,
            thickness,
        });
        sum += thickness;
        min = min.min(thickness);
        max = max.max(thickness);
    }

    if samples.is_empty() {
        return None;
    }

    let evaluated = samples.len();
    Some(ShellThickness {
        samples,
        min,
        max,
        mean: sum / evaluated as f32,
        evaluated,
        requested: surface.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box with the given half-extents, outward wound and
    /// closed.
    fn box_mesh(hx: f32, hy: f32, hz: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn empty_or_zero_sample_input_is_rejected() {
        let (v, f) = box_mesh(0.5, 0.5, 0.5);
        let p = ShellThicknessParams::default();
        assert!(estimate_shell_thickness(&[], &f, p).is_none());
        assert!(estimate_shell_thickness(&v, &[], p).is_none());
        assert!(estimate_shell_thickness(
            &v,
            &f,
            ShellThicknessParams {
                sample_count: 0,
                ..p
            }
        )
        .is_none());
    }

    #[test]
    fn unit_cube_has_uniform_unit_thickness() {
        // A unit cube: every inward normal hits the opposite face at distance 1.
        let (v, f) = box_mesh(0.5, 0.5, 0.5);
        let report = estimate_shell_thickness(
            &v,
            &f,
            ShellThicknessParams {
                sample_count: 128,
                seed: 42,
                max_thickness: None,
            },
        )
        .unwrap();
        assert!(report.sample_count() > 0);
        assert!((report.min - 1.0).abs() < 1.0e-3, "min {}", report.min);
        assert!((report.max - 1.0).abs() < 1.0e-3, "max {}", report.max);
        assert!((report.mean - 1.0).abs() < 1.0e-3, "mean {}", report.mean);
    }

    #[test]
    fn thin_slab_reports_small_minimum_thickness() {
        // A 1 x 1 x 0.1 slab: the broad faces are 0.1 apart, so the thinnest
        // wall found must be about 0.1.
        let (v, f) = box_mesh(0.5, 0.5, 0.05);
        let report = estimate_shell_thickness(
            &v,
            &f,
            ShellThicknessParams {
                sample_count: 256,
                seed: 7,
                max_thickness: None,
            },
        )
        .unwrap();
        assert!(
            report.min < 0.15,
            "min thickness {} should be ~0.1",
            report.min
        );
        assert!(
            report.min > 0.05,
            "min thickness {} should exceed offset",
            report.min
        );
        // The largest wall-to-wall span is the 1-unit lateral extent.
        assert!(report.max <= 1.0 + 1.0e-3, "max {}", report.max);
    }

    #[test]
    fn coverage_is_full_for_a_closed_solid() {
        let (v, f) = box_mesh(0.5, 0.5, 0.5);
        let report = estimate_shell_thickness(
            &v,
            &f,
            ShellThicknessParams {
                sample_count: 64,
                seed: 3,
                max_thickness: None,
            },
        )
        .unwrap();
        // Every inward ray inside a closed convex solid hits the far wall.
        assert!((report.coverage() - 1.0).abs() < 1.0e-6);
        assert!(!report.is_empty());
    }

    #[test]
    fn result_is_deterministic_for_a_fixed_seed() {
        let (v, f) = box_mesh(0.5, 0.5, 0.3);
        let p = ShellThicknessParams {
            sample_count: 96,
            seed: 11,
            max_thickness: None,
        };
        let a = estimate_shell_thickness(&v, &f, p).unwrap();
        let b = estimate_shell_thickness(&v, &f, p).unwrap();
        assert_eq!(a, b);
    }
}
