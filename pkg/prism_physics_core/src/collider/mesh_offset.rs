//! Signed mesh offsetting (dilation / erosion) for collision-skin generation.
//!
//! Many collision pipelines want a *skinned* version of a source mesh: a shell
//! pushed outward by a small margin so contacts resolve before the visual
//! surfaces interpenetrate, or pulled inward to recover a conservative inner
//! core. This module performs that offset implicitly. It samples a signed
//! distance field (`MeshSdf`) of the source triangles onto a regular grid,
//! shifts the iso-level by the requested `offset`, and re-extracts a watertight
//! triangle mesh with marching tetrahedra.
//!
//! Working through a distance field (rather than displacing vertices along
//! their normals) keeps the result robust on concave features and sharp
//! corners: the offset surface is the exact `offset`-level set of the distance
//! field, so self-intersections created by naive normal extrusion never arise.
//! Convex corners round off at radius `offset` and concave corners fill in,
//! matching the mathematical Minkowski dilation / erosion of the solid.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! distance-field offset is the classical level-set / Minkowski morphological
//! operation (Osher & Sethian 1988); the iso-surface extraction reuses the
//! crate's own marching-tetrahedra implementation.

use glam::Vec3;

use crate::collider::sdf::{MeshSdf, SdfBuildParams};
use crate::reconstruct::field::ScalarField;
use crate::reconstruct::marching_cubes::triangulate;

/// Default voxel count across the mesh's longest axis-aligned extent.
const DEFAULT_RESOLUTION: u32 = 48;
/// Default number of fully-exterior voxel layers kept beyond the offset
/// surface so the extracted shell is guaranteed to close.
const DEFAULT_BOUNDARY_CELLS: usize = 3;

/// Parameters controlling a signed mesh offset.
#[derive(Clone, Copy, Debug)]
pub struct MeshOffsetParams {
    /// Signed offset distance in mesh-local units. A positive value dilates the
    /// surface outward (a thicker collision skin); a negative value erodes it
    /// inward toward the medial axis.
    pub offset: f32,
    /// Target voxel count across the mesh's longest axis-aligned extent. Higher
    /// values capture finer detail at greater memory and time cost. Clamped to
    /// at least two by [`MeshOffsetParams::sanitized`].
    pub resolution: u32,
    /// Number of fully-exterior voxel layers kept beyond the offset surface so
    /// the extracted iso-surface closes into a watertight shell. Clamped to at
    /// least one by [`MeshOffsetParams::sanitized`].
    pub boundary_cells: usize,
}

impl Default for MeshOffsetParams {
    fn default() -> MeshOffsetParams {
        MeshOffsetParams {
            offset: 0.0,
            resolution: DEFAULT_RESOLUTION,
            boundary_cells: DEFAULT_BOUNDARY_CELLS,
        }
    }
}

impl MeshOffsetParams {
    /// Parameters that dilate (grow) the surface outward by `distance`. The
    /// magnitude is used, so the sign of `distance` is irrelevant.
    #[must_use]
    pub fn dilate(distance: f32) -> MeshOffsetParams {
        MeshOffsetParams {
            offset: distance.abs(),
            ..MeshOffsetParams::default()
        }
    }

    /// Parameters that erode (shrink) the surface inward by `distance`. The
    /// magnitude is used, so the sign of `distance` is irrelevant.
    #[must_use]
    pub fn erode(distance: f32) -> MeshOffsetParams {
        MeshOffsetParams {
            offset: -distance.abs(),
            ..MeshOffsetParams::default()
        }
    }

    /// Returns a copy with non-finite or out-of-range fields replaced by safe
    /// defaults: a non-finite `offset` becomes zero, `resolution` is forced to
    /// at least two nodes per axis, and `boundary_cells` to at least one.
    #[must_use]
    pub fn sanitized(self) -> MeshOffsetParams {
        MeshOffsetParams {
            offset: if self.offset.is_finite() {
                self.offset
            } else {
                0.0
            },
            resolution: self.resolution.max(2),
            boundary_cells: self.boundary_cells.max(1),
        }
    }
}

/// The result of offsetting a mesh: a watertight, edge-manifold triangle shell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OffsetMesh {
    /// Vertex positions of the offset surface, in mesh-local units.
    pub positions: Vec<Vec3>,
    /// Per-vertex outward unit normals, index-aligned with
    /// [`OffsetMesh::positions`].
    pub normals: Vec<Vec3>,
    /// Triangle vertex indices; every entry is three indices into the position
    /// and normal arrays.
    pub triangles: Vec<[u32; 3]>,
}

impl OffsetMesh {
    /// The number of vertices.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// The number of triangles.
    #[inline]
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Whether the shell holds no triangles.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// Offsets the closed triangle mesh described by `vertices` and triangle
/// `indices` by `params.offset`, returning a fresh watertight shell.
///
/// A positive offset dilates the surface outward and a negative offset erodes
/// it inward. Returns `None` when the input is empty, degenerate, or when an
/// erosion removes the entire solid (nothing survives the inward offset).
#[must_use]
pub fn offset_mesh(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: &MeshOffsetParams,
) -> Option<OffsetMesh> {
    let params = params.sanitized();
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    // The longest axis-aligned extent sets the voxel resolution.
    let mut lo = vertices[0];
    let mut hi = vertices[0];
    for &v in vertices {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let longest = (hi - lo).max_element();
    if !longest.is_finite() || longest <= 0.0 {
        return None;
    }

    let base = SdfBuildParams::from_resolution(longest, params.resolution);
    let cell = base.cell_size;
    // Pad enough to contain the dilated surface plus a fully-exterior shell so
    // the extracted iso-surface closes. Erosion needs only the exterior shell.
    let dilation = if params.offset > 0.0 {
        params.offset
    } else {
        0.0
    };
    let padding = dilation + params.boundary_cells as f32 * cell;
    let sdf = MeshSdf::from_mesh(
        vertices,
        indices,
        SdfBuildParams {
            cell_size: cell,
            padding,
        },
    )?;

    // Build the implicit field whose zero level set is the offset surface.
    // The solid interior is `signed_distance < offset`; encoding the node value
    // as `offset - signed_distance` makes marching tetrahedra (which treats
    // `value > 0` as inside) extract exactly that shell, with outward normals.
    let [nx, ny, nz] = sdf.dims();
    let mut field = ScalarField::zeros(nx, ny, nz, cell, sdf.origin());
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let distance = sdf.node_distance(i, j, k)?;
                field.add_value(i, j, k, params.offset - distance);
            }
        }
    }

    let surface = triangulate(&field, 0.0);
    if surface.is_empty() {
        return None;
    }

    let triangles: Vec<[u32; 3]> = surface
        .indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    Some(OffsetMesh {
        positions: surface.positions,
        normals: surface.normals,
        triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Unit-radius icosphere subdivided `levels` times, as outward triangles.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(p);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    /// Axis-aligned cube centred at the origin, outward-wound.
    fn cube_mesh(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = half;
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let idx = vec![
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
        (verts, idx)
    }

    /// Scales a unit mesh's vertices by `r` in place.
    fn scaled(verts: &[Vec3], r: f32) -> Vec<Vec3> {
        verts.iter().map(|&v| v * r).collect()
    }

    #[test]
    fn empty_input_returns_none() {
        let p = MeshOffsetParams::dilate(0.1);
        assert!(offset_mesh(&[], &[], &p).is_none());
        let (v, _) = cube_mesh(1.0);
        assert!(offset_mesh(&v, &[], &p).is_none());
    }

    #[test]
    fn sanitized_clamps_bad_fields() {
        let p = MeshOffsetParams {
            offset: f32::NAN,
            resolution: 0,
            boundary_cells: 0,
        }
        .sanitized();
        assert_eq!(p.offset, 0.0);
        assert_eq!(p.resolution, 2);
        assert_eq!(p.boundary_cells, 1);
    }

    #[test]
    fn sphere_dilation_moves_surface_outward() {
        let (v, i) = icosphere(3);
        let r = 2.0_f32;
        let verts = scaled(&v, r);
        let offset = 0.5_f32;
        let p = MeshOffsetParams {
            offset,
            resolution: 32,
            boundary_cells: 3,
        };
        let out = offset_mesh(&verts, &i, &p).expect("dilation produces a shell");
        assert!(!out.is_empty());
        assert_eq!(out.vertex_count(), out.normals.len());

        let expected = r + offset;
        for (idx, &pos) in out.positions.iter().enumerate() {
            let radius = pos.length();
            assert!(
                (radius - expected).abs() < 0.25,
                "vertex radius {radius} far from expected {expected}"
            );
            // Normals point outward (same hemisphere as the radial direction).
            let n = out.normals[idx];
            assert!(n.dot(pos.normalize()) > 0.5, "normal is not outward");
        }
    }

    #[test]
    fn sphere_erosion_moves_surface_inward() {
        let (v, i) = icosphere(3);
        let r = 2.0_f32;
        let verts = scaled(&v, r);
        let offset = 0.6_f32;
        let out = offset_mesh(&verts, &i, &MeshOffsetParams::erode(offset))
            .expect("erosion leaves an inner core");
        let expected = r - offset;
        for &pos in &out.positions {
            let radius = pos.length();
            assert!(
                (radius - expected).abs() < 0.25,
                "eroded radius {radius} far from expected {expected}"
            );
        }
    }

    #[test]
    fn cube_dilation_expands_bounds() {
        let (v, i) = cube_mesh(1.0);
        let offset = 0.4_f32;
        let p = MeshOffsetParams {
            offset,
            resolution: 32,
            boundary_cells: 3,
        };
        let out = offset_mesh(&v, &i, &p).expect("cube dilation produces a shell");
        let mut hi = out.positions[0];
        for &pos in &out.positions {
            hi = hi.max(pos);
        }
        // Faces move outward by `offset`; the bounding box half-extent grows to
        // roughly `1 + offset` at the face centres.
        let expected = 1.0 + offset;
        for axis in 0..3 {
            let m = hi[axis];
            assert!(
                (m - expected).abs() < 0.12,
                "axis {axis} max {m} far from expected {expected}"
            );
        }
    }

    #[test]
    fn over_erosion_vanishes() {
        let (v, i) = icosphere(2);
        let r = 1.0_f32;
        let verts = scaled(&v, r);
        // Eroding by more than the radius removes the whole solid.
        let out = offset_mesh(&verts, &i, &MeshOffsetParams::erode(2.0 * r));
        assert!(out.is_none());
    }

    #[test]
    fn offset_surface_is_watertight_manifold() {
        let (v, i) = icosphere(2);
        let verts = scaled(&v, 1.5);
        let p = MeshOffsetParams {
            offset: 0.3,
            resolution: 24,
            boundary_cells: 2,
        };
        let out = offset_mesh(&verts, &i, &p).expect("produces a shell");

        // Every index is in range.
        let n = out.vertex_count() as u32;
        for tri in &out.triangles {
            assert!(tri.iter().all(|&x| x < n));
        }

        // Every undirected edge is shared by exactly two triangles (closed,
        // edge-manifold shell).
        let mut counts: StdHashMap<(u32, u32), u32> = StdHashMap::new();
        for tri in &out.triangles {
            let e = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for &(a, b) in &e {
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        assert!(!counts.is_empty());
        for (&(a, b), &c) in &counts {
            assert_eq!(c, 2, "edge ({a},{b}) shared by {c} triangles, not 2");
        }
    }

    #[test]
    fn offset_is_deterministic() {
        let (v, i) = icosphere(2);
        let verts = scaled(&v, 1.0);
        let p = MeshOffsetParams {
            offset: 0.2,
            resolution: 20,
            boundary_cells: 2,
        };
        let a = offset_mesh(&verts, &i, &p).expect("first run");
        let b = offset_mesh(&verts, &i, &p).expect("second run");
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.triangles, b.triangles);
        assert_eq!(a.normals, b.normals);
    }
}
