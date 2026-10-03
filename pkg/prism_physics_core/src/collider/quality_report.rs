//! Cook-time mesh quality report: a single actionable gate over the lower-level
//! topology and feature-edge analyses.
//!
//! Before a mesh is turned into collision geometry (a signed-distance field, a
//! convex decomposition, a trusted inside/outside oracle) AAA cookers run a
//! quality pass and either accept the mesh or report exactly why it was
//! rejected. This module is that entry point: it composes
//! [`analyze_topology`](crate::collider::analyze_topology) and
//! [`extract_feature_edges`](crate::collider::extract_feature_edges) with cheap
//! direct counts (merged vertices, removed triangles, isolated vertices) into
//! one [`MeshQualityReport`] with convenience gates such as
//! [`is_simulation_ready`](MeshQualityReport::is_simulation_ready).
//!
//! This is standard cooking validation built from this crate\'s own primitives;
//! nothing here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashSet;

use crate::collider::feature_edges::{extract_feature_edges, FeatureEdgeParams};
use crate::collider::topology::{analyze_topology, MeshTopology};
use crate::collider::weld::{weld_mesh, WeldParams};

/// A composed description of a triangle mesh\'s fitness for collision cooking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshQualityReport {
    /// Vertices in the raw input.
    pub input_vertices: usize,
    /// Triangles in the raw input.
    pub input_triangles: usize,
    /// Vertices removed by welding near-coincident positions.
    pub merged_vertices: usize,
    /// Triangles dropped during welding (degenerate or duplicate).
    pub removed_triangles: usize,
    /// Input vertices referenced by no in-range triangle.
    pub isolated_vertices: usize,
    /// Topology diagnostics of the welded mesh.
    pub topology: MeshTopology,
    /// Crease edges at or beyond the configured dihedral threshold.
    pub crease_edges: usize,
}

impl MeshQualityReport {
    /// Whether the welded mesh is a clean closed, consistently wound 2-manifold:
    /// the precondition SDF cooking and volume integrals require.
    #[must_use]
    pub fn is_watertight_manifold(&self) -> bool {
        self.topology.is_watertight_manifold()
    }

    /// Whether the mesh is ready to cook into simulation collision geometry: a
    /// watertight manifold with no isolated vertices.
    #[must_use]
    pub fn is_simulation_ready(&self) -> bool {
        self.is_watertight_manifold() && self.isolated_vertices == 0
    }

    /// Whether welding had to remove any vertices or triangles, a hint that the
    /// authored mesh had cracks or redundant geometry.
    #[must_use]
    pub fn had_cleanup(&self) -> bool {
        self.merged_vertices > 0 || self.removed_triangles > 0
    }
}

/// Tuning for [`analyze_mesh_quality`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshQualityParams {
    /// Position tolerance used to weld near-coincident vertices.
    pub weld_epsilon: f32,
    /// Dihedral angle (radians) at or above which an edge counts as a crease.
    pub crease_angle_radians: f32,
}

impl MeshQualityParams {
    /// Builds params from a crease threshold in degrees.
    #[must_use]
    pub fn with_crease_degrees(crease_degrees: f32, weld_epsilon: f32) -> Self {
        Self {
            weld_epsilon,
            crease_angle_radians: crease_degrees.to_radians(),
        }
    }
}

impl Default for MeshQualityParams {
    /// A `1e-5` weld tolerance and a 40-degree crease threshold.
    fn default() -> Self {
        Self::with_crease_degrees(40.0, 1.0e-5)
    }
}

/// Produces a composed quality report for a triangle mesh.
///
/// Returns `None` when `vertices` or `indices` is empty, when a parameter is not
/// finite (weld tolerance must also be strictly positive), or when welding
/// leaves no non-degenerate triangle (the mesh is entirely degenerate).
#[must_use]
pub fn analyze_mesh_quality(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: MeshQualityParams,
) -> Option<MeshQualityReport> {
    if vertices.is_empty()
        || indices.is_empty()
        || !(params.weld_epsilon.is_finite() && params.weld_epsilon > 0.0)
        || !params.crease_angle_radians.is_finite()
    {
        return None;
    }

    let topology = analyze_topology(vertices, indices, params.weld_epsilon)?;

    let feature = extract_feature_edges(
        vertices,
        indices,
        FeatureEdgeParams {
            crease_angle_radians: params.crease_angle_radians,
            weld_epsilon: params.weld_epsilon,
        },
    )?;

    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: params.weld_epsilon,
            drop_duplicate_triangles: true,
        },
    )?;

    Some(MeshQualityReport {
        input_vertices: vertices.len(),
        input_triangles: indices.len(),
        merged_vertices: welded.removed_vertices,
        removed_triangles: welded.removed_triangles,
        isolated_vertices: count_isolated_vertices(vertices.len(), indices),
        topology,
        crease_edges: feature.crease_count(),
    })
}

/// Counts input vertices referenced by no in-range triangle.
fn count_isolated_vertices(vertex_count: usize, indices: &[[u32; 3]]) -> usize {
    let mut used: HashSet<u32> = HashSet::new();
    for tri in indices {
        for &idx in tri {
            if (idx as usize) < vertex_count {
                used.insert(idx);
            }
        }
    }
    vertex_count - used.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, with shared vertices.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
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
    fn empty_or_invalid_input_is_rejected() {
        let (v, f) = unit_cube();
        assert!(analyze_mesh_quality(&[], &f, MeshQualityParams::default()).is_none());
        assert!(analyze_mesh_quality(&v, &[], MeshQualityParams::default()).is_none());
        let bad_weld = MeshQualityParams {
            weld_epsilon: 0.0,
            crease_angle_radians: 0.7,
        };
        assert!(analyze_mesh_quality(&v, &f, bad_weld).is_none());
        let bad_crease = MeshQualityParams {
            weld_epsilon: 1.0e-5,
            crease_angle_radians: f32::INFINITY,
        };
        assert!(analyze_mesh_quality(&v, &f, bad_crease).is_none());
    }

    #[test]
    fn clean_cube_is_simulation_ready() {
        let (v, f) = unit_cube();
        let report = analyze_mesh_quality(&v, &f, MeshQualityParams::default()).unwrap();
        assert!(report.is_watertight_manifold());
        assert!(report.is_simulation_ready());
        assert_eq!(report.isolated_vertices, 0);
        assert_eq!(report.input_vertices, 8);
        assert_eq!(report.input_triangles, 12);
        assert_eq!(report.crease_edges, 12);
        assert!(!report.had_cleanup());
    }

    #[test]
    fn isolated_vertex_blocks_simulation_ready() {
        let (mut v, f) = unit_cube();
        v.push(Vec3::new(10.0, 10.0, 10.0)); // unreferenced
        let report = analyze_mesh_quality(&v, &f, MeshQualityParams::default()).unwrap();
        assert_eq!(report.isolated_vertices, 1);
        // Still a watertight manifold, but not simulation-ready.
        assert!(report.is_watertight_manifold());
        assert!(!report.is_simulation_ready());
    }

    #[test]
    fn open_mesh_is_not_watertight() {
        // A single triangle: three boundary edges, not closed.
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let report =
            analyze_mesh_quality(&verts, &[[0, 1, 2]], MeshQualityParams::default()).unwrap();
        assert!(!report.is_watertight_manifold());
        assert!(!report.is_simulation_ready());
        assert_eq!(report.topology.boundary_edges, 3);
    }

    #[test]
    fn duplicate_triangle_is_reported_as_cleanup() {
        let (v, mut f) = unit_cube();
        let dup = f[0];
        f.push(dup); // exact duplicate triangle
        let report = analyze_mesh_quality(&v, &f, MeshQualityParams::default()).unwrap();
        assert_eq!(report.input_triangles, 13);
        assert!(report.removed_triangles >= 1);
        assert!(report.had_cleanup());
    }

    #[test]
    fn fully_degenerate_mesh_is_rejected() {
        // Three collinear points: the only triangle has zero area.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        assert!(analyze_mesh_quality(&verts, &[[0, 1, 2]], MeshQualityParams::default()).is_none());
    }
}
