//! Removal of tiny disconnected fragments from triangle-soup collision geometry.
//!
//! Authoring exports, boolean operations and procedural fracture frequently
//! leave a mesh dusted with sub-visible specks: a few stray triangles, a loose
//! vertex fan, or a shard a thousandth the size of the main body. Feeding those
//! specks into convex decomposition or `BVH` construction wastes memory and can
//! produce spurious collision shapes. Cooking pipelines in `PhysX`, Jolt and UE
//! all run a "remove small parts" pass before decomposition.
//!
//! This module implements that pass on top of
//! [`split_connected_components`](crate::collider::split_connected_components):
//! the soup is split into shells, each shell is scored by its surface area, and
//! shells whose area is a negligible fraction of the largest shell (or whose
//! triangle count is below a floor) are dropped. The largest shell is always
//! retained so cleanup can never erase the whole mesh. The survivors are
//! concatenated back into a single vertex/index buffer.
//!
//! This is pure connectivity-and-area geometry with no coupling to the
//! collision pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::connectivity::{
    split_connected_components, ConnectivityParams, MeshComponent,
};
use crate::collider::surface_sampling::total_surface_area;

/// Parameters controlling which fragments are discarded.
#[derive(Clone, Copy, Debug)]
pub struct CleanupParams {
    /// Vertices closer than this distance are treated as the same point when
    /// grouping triangles into shells. Must be finite and non-negative; `0.0`
    /// uses purely topological (shared-index) connectivity.
    pub weld_epsilon: f32,
    /// A shell is dropped when its surface area is below this fraction of the
    /// largest shell's surface area. Must lie in `[0, 1]`. `0.0` disables the
    /// area test.
    pub min_area_fraction: f32,
    /// A shell is dropped when it has fewer than this many triangles. The
    /// largest shell is exempt so cleanup never empties the mesh.
    pub min_triangles: usize,
}

impl Default for CleanupParams {
    fn default() -> Self {
        Self {
            weld_epsilon: 1e-5,
            min_area_fraction: 1e-3,
            min_triangles: 1,
        }
    }
}

/// A triangle soup with its small fragments removed.
#[derive(Clone, Debug)]
pub struct CleanedMesh {
    /// Positions used by the surviving shells, concatenated shell by shell.
    pub vertices: Vec<Vec3>,
    /// Triangles reindexed into [`CleanedMesh::vertices`].
    pub indices: Vec<[u32; 3]>,
    /// Number of shells that were discarded.
    pub removed_components: usize,
    /// Total triangle count across the discarded shells.
    pub removed_triangles: usize,
}

impl CleanedMesh {
    /// Number of triangles retained.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Number of distinct vertices retained.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Whether the cleaned mesh carries no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Removes tiny disconnected fragments from a triangle soup.
///
/// Returns `None` when the input is empty or invalid (see
/// [`split_connected_components`](crate::collider::split_connected_components)),
/// or when `min_area_fraction` is not finite or lies outside `[0, 1]`.
///
/// The largest shell by surface area is always retained, so the result always
/// contains at least one triangle when the split succeeds.
#[must_use]
pub fn remove_small_components(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: CleanupParams,
) -> Option<CleanedMesh> {
    if !params.min_area_fraction.is_finite()
        || params.min_area_fraction < 0.0
        || params.min_area_fraction > 1.0
    {
        return None;
    }

    let components = split_connected_components(
        vertices,
        indices,
        ConnectivityParams {
            weld_epsilon: params.weld_epsilon,
        },
    )?;

    // Score every shell by surface area; a shell that is degenerate enough to
    // report no area scores zero.
    let areas: Vec<f32> = components
        .iter()
        .map(|c| total_surface_area(&c.vertices, &c.indices).unwrap_or(0.0))
        .collect();

    // The largest shell anchors the relative-area threshold and is always kept.
    let mut largest = 0usize;
    for (i, &area) in areas.iter().enumerate() {
        if area > areas[largest] {
            largest = i;
        }
    }
    let max_area = areas[largest];
    let area_floor = params.min_area_fraction * max_area;

    let mut kept: Vec<&MeshComponent> = Vec::new();
    let mut removed_components = 0usize;
    let mut removed_triangles = 0usize;
    for (i, component) in components.iter().enumerate() {
        let keep = i == largest
            || (areas[i] >= area_floor && component.triangle_count() >= params.min_triangles);
        if keep {
            kept.push(component);
        } else {
            removed_components += 1;
            removed_triangles += component.triangle_count();
        }
    }

    // Concatenate the survivors into a single buffer, offsetting indices.
    let mut out_vertices: Vec<Vec3> = Vec::new();
    let mut out_indices: Vec<[u32; 3]> = Vec::new();
    for component in kept {
        let base = out_vertices.len() as u32;
        out_vertices.extend_from_slice(&component.vertices);
        for tri in &component.indices {
            out_indices.push([tri[0] + base, tri[1] + base, tri[2] + base]);
        }
    }

    Some(CleanedMesh {
        vertices: out_vertices,
        indices: out_indices,
        removed_components,
        removed_triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit tetrahedron translated by `offset` and uniformly scaled.
    fn tetra(offset: Vec3, scale: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            offset + scale * Vec3::new(0.0, 0.0, 0.0),
            offset + scale * Vec3::new(1.0, 0.0, 0.0),
            offset + scale * Vec3::new(0.0, 1.0, 0.0),
            offset + scale * Vec3::new(0.0, 0.0, 1.0),
        ];
        let i = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        (v, i)
    }

    /// Merge several disjoint meshes into one soup with offset indices.
    fn merge(parts: &[(Vec<Vec3>, Vec<[u32; 3]>)]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        for (v, i) in parts {
            let base = verts.len() as u32;
            verts.extend_from_slice(v);
            for t in i {
                inds.push([t[0] + base, t[1] + base, t[2] + base]);
            }
        }
        (verts, inds)
    }

    #[test]
    fn rejects_invalid_params() {
        let (v, i) = tetra(Vec3::ZERO, 1.0);
        assert!(remove_small_components(
            &v,
            &i,
            CleanupParams {
                min_area_fraction: -0.1,
                ..Default::default()
            }
        )
        .is_none());
        assert!(remove_small_components(
            &v,
            &i,
            CleanupParams {
                min_area_fraction: 1.5,
                ..Default::default()
            }
        )
        .is_none());
        assert!(remove_small_components(
            &v,
            &i,
            CleanupParams {
                min_area_fraction: f32::NAN,
                ..Default::default()
            }
        )
        .is_none());
    }

    #[test]
    fn rejects_empty_input() {
        assert!(remove_small_components(&[], &[], CleanupParams::default()).is_none());
    }

    #[test]
    fn drops_a_tiny_speck_next_to_a_large_body() {
        let big = tetra(Vec3::ZERO, 10.0);
        let speck = tetra(Vec3::new(100.0, 0.0, 0.0), 0.01);
        let (v, i) = merge(&[big.clone(), speck]);
        let cleaned = remove_small_components(&v, &i, CleanupParams::default()).unwrap();
        assert_eq!(cleaned.removed_components, 1);
        assert_eq!(cleaned.removed_triangles, 4);
        // Only the big tetra's four triangles survive.
        assert_eq!(cleaned.triangle_count(), 4);
    }

    #[test]
    fn keeps_two_comparable_bodies() {
        let a = tetra(Vec3::ZERO, 5.0);
        let b = tetra(Vec3::new(50.0, 0.0, 0.0), 5.0);
        let (v, i) = merge(&[a, b]);
        let cleaned = remove_small_components(&v, &i, CleanupParams::default()).unwrap();
        assert_eq!(cleaned.removed_components, 0);
        assert_eq!(cleaned.triangle_count(), 8);
    }

    #[test]
    fn always_keeps_largest_even_with_aggressive_threshold() {
        let big = tetra(Vec3::ZERO, 10.0);
        let mid = tetra(Vec3::new(100.0, 0.0, 0.0), 3.0);
        let (v, i) = merge(&[big, mid]);
        // Demand each shell be at least as large as the biggest: only the
        // biggest can satisfy that, and it must still be retained.
        let cleaned = remove_small_components(
            &v,
            &i,
            CleanupParams {
                min_area_fraction: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(cleaned.removed_components, 1);
        assert!(cleaned.triangle_count() >= 4);
        assert!(!cleaned.is_empty());
    }

    #[test]
    fn triangle_floor_drops_small_counts() {
        let big = tetra(Vec3::ZERO, 10.0);
        let small = tetra(Vec3::new(100.0, 0.0, 0.0), 9.0);
        let (v, i) = merge(&[big, small]);
        // Both shells are comparable in area, but require >100 triangles: the
        // smaller shell is dropped by the triangle floor, the largest is kept.
        let cleaned = remove_small_components(
            &v,
            &i,
            CleanupParams {
                min_area_fraction: 0.0,
                min_triangles: 100,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(cleaned.removed_components, 1);
        assert_eq!(cleaned.triangle_count(), 4);
    }

    #[test]
    fn single_component_passes_through_unchanged() {
        let (v, i) = tetra(Vec3::ZERO, 1.0);
        let cleaned = remove_small_components(&v, &i, CleanupParams::default()).unwrap();
        assert_eq!(cleaned.removed_components, 0);
        assert_eq!(cleaned.triangle_count(), 4);
        assert_eq!(cleaned.vertex_count(), 4);
    }
}
