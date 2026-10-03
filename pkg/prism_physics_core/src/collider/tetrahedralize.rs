//! Volumetric tetrahedral meshing of a closed surface via a lattice fill.
//!
//! Soft-body FEM, finite-volume plasticity, and reduced-order solvers all need
//! an interior tetrahedral mesh, not just a surface. This module fills the
//! inside of a watertight triangle mesh with a conforming tetrahedral lattice:
//!
//! 1. A signed distance field ([`crate::collider::sdf::MeshSdf`]) classifies
//!    points as inside (`sample < 0`) or outside.
//! 2. A regular grid over the mesh bounds is split into tetrahedra with the
//!    Freudenthal/`Kuhn` subdivision (six tets per cube sharing the main
//!    diagonal). Because the subdivision is translation invariant, the tets of
//!    neighbouring cells share faces -- the mesh is conforming with no
//!    T-junctions.
//! 3. A tetrahedron is kept when all four of its corners lie inside the surface
//!    (optionally shrunk by a margin), so every output vertex is interior.
//! 4. Each kept tetrahedron is reoriented to positive signed volume.
//!
//! The result underestimates the solid near the boundary and converges to the
//! true volume as the resolution grows -- the standard behaviour of a lattice
//! (voxel) tetrahedralisation. This is a textbook meshing technique; nothing
//! here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

use crate::collider::sdf::{MeshSdf, SdfBuildParams};

/// Largest grid resolution accepted, to bound memory and runtime.
const MAX_RESOLUTION: u32 = 160;

/// The eight corner offsets of a unit cell, indexed by `bit = i + 2*j + 4*k`.
const CORNER_OFFSET: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// The six Freudenthal tetrahedra of a cube, each as corner-bit indices. All
/// six share the main diagonal `0 -> 7`, which makes the subdivision conforming
/// across neighbouring cells.
const KUHN_TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// Parameters controlling [`tetrahedralize`].
#[derive(Clone, Copy, Debug)]
pub struct TetMeshParams {
    /// Number of cells along the mesh's longest axis. Clamped to
    /// `[1, MAX_RESOLUTION]`.
    pub resolution: u32,
    /// Extra inward margin (in mesh units) a corner must clear to count as
    /// interior. `0` keeps every strictly-interior corner; a positive value
    /// erodes the fill away from the surface.
    pub interior_margin: f32,
}

impl TetMeshParams {
    /// Builds parameters at the given lattice resolution with no extra margin.
    #[must_use]
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            interior_margin: 0.0,
        }
    }

    /// Returns a copy with the interior margin replaced.
    #[must_use]
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.interior_margin = margin;
        self
    }
}

impl Default for TetMeshParams {
    fn default() -> Self {
        Self::new(16)
    }
}

/// A tetrahedral volume mesh: shared vertices plus index quadruples.
#[derive(Clone, Debug, PartialEq)]
pub struct TetMesh {
    /// Vertex positions in mesh-local space. Every vertex lies inside the
    /// source surface.
    pub vertices: Vec<Vec3>,
    /// Tetrahedra as four vertex indices, each wound to positive signed volume
    /// (`(v1-v0) . ((v2-v0) x (v3-v0)) > 0`).
    pub tets: Vec<[u32; 4]>,
}

impl TetMesh {
    /// Number of distinct vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of tetrahedra.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.tets.len()
    }

    /// Whether the mesh carries no tetrahedra.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tets.is_empty()
    }

    /// Total enclosed volume (sum of per-tetrahedron volumes).
    #[must_use]
    pub fn total_volume(&self) -> f32 {
        let mut sum = 0.0f64;
        for t in &self.tets {
            let a = self.vertices[t[0] as usize];
            let b = self.vertices[t[1] as usize];
            let c = self.vertices[t[2] as usize];
            let d = self.vertices[t[3] as usize];
            sum += tet_vol6(a, b, c, d).abs();
        }
        (sum / 6.0) as f32
    }
}

/// Signed six-times volume `(b-a) . ((c-a) x (d-a))` accumulated in `f64`.
fn tet_vol6(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let e1 = (
        f64::from(b.x - a.x),
        f64::from(b.y - a.y),
        f64::from(b.z - a.z),
    );
    let e2 = (
        f64::from(c.x - a.x),
        f64::from(c.y - a.y),
        f64::from(c.z - a.z),
    );
    let e3 = (
        f64::from(d.x - a.x),
        f64::from(d.y - a.y),
        f64::from(d.z - a.z),
    );
    let cross = (
        e2.1 * e3.2 - e2.2 * e3.1,
        e2.2 * e3.0 - e2.0 * e3.2,
        e2.0 * e3.1 - e2.1 * e3.0,
    );
    e1.0 * cross.0 + e1.1 * cross.1 + e1.2 * cross.2
}

/// Fills the interior of a closed triangle mesh with a conforming tetrahedral
/// lattice.
///
/// Returns `None` when `vertices`/`indices` is empty, the bounds are
/// degenerate, the signed distance field cannot be built, or no tetrahedron
/// survives the interior test (e.g. the resolution is too coarse to place a
/// cell wholly inside).
#[must_use]
pub fn tetrahedralize(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: &TetMeshParams,
) -> Option<TetMesh> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let resolution = params.resolution.clamp(1, MAX_RESOLUTION);

    // Bounds of the source mesh.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let extent = hi - lo;
    let longest = extent.max_element();
    if !longest.is_finite() || longest <= 0.0 {
        return None;
    }

    // Classify interior points with a signed distance field.
    let sdf = MeshSdf::from_mesh(
        vertices,
        indices,
        SdfBuildParams::from_resolution(longest, resolution.max(8)),
    )?;

    let cell = longest / resolution as f32;
    if !cell.is_finite() || cell <= 0.0 {
        return None;
    }

    // Number of cells per axis so the lattice covers the bounds.
    let cells = |len: f32| -> usize {
        let n = (len / cell).ceil() as i64;
        n.max(1) as usize
    };
    let nx = cells(extent.x);
    let ny = cells(extent.y);
    let nz = cells(extent.z);

    let node_pos = |i: usize, j: usize, k: usize| -> Vec3 {
        lo + Vec3::new(i as f32 * cell, j as f32 * cell, k as f32 * cell)
    };

    let margin = params.interior_margin.max(0.0);
    let inside = |p: Vec3| sdf.sample(p) < -margin;

    let mut index_of: HashMap<(usize, usize, usize), u32> = HashMap::new();
    let mut out_vertices: Vec<Vec3> = Vec::new();
    let mut tets: Vec<[u32; 4]> = Vec::new();

    let intern = |key: (usize, usize, usize),
                      pos: Vec3,
                      verts: &mut Vec<Vec3>,
                      map: &mut HashMap<(usize, usize, usize), u32>|
     -> u32 {
        *map.entry(key).or_insert_with(|| {
            let id = verts.len() as u32;
            verts.push(pos);
            id
        })
    };

    for ck in 0..nz {
        for cj in 0..ny {
            for ci in 0..nx {
                // Corner node coordinates and positions for this cell.
                let mut corner_key = [(0usize, 0usize, 0usize); 8];
                let mut corner_pos = [Vec3::ZERO; 8];
                for (c, off) in CORNER_OFFSET.iter().enumerate() {
                    let key = (ci + off[0], cj + off[1], ck + off[2]);
                    corner_key[c] = key;
                    corner_pos[c] = node_pos(key.0, key.1, key.2);
                }

                for tet in &KUHN_TETS {
                    let p = [
                        corner_pos[tet[0]],
                        corner_pos[tet[1]],
                        corner_pos[tet[2]],
                        corner_pos[tet[3]],
                    ];
                    if !(inside(p[0]) && inside(p[1]) && inside(p[2]) && inside(p[3])) {
                        continue;
                    }

                    // Reorient to positive signed volume; drop numerically flat
                    // tets (should not occur on a regular lattice).
                    let vol = tet_vol6(p[0], p[1], p[2], p[3]);
                    if vol.abs() <= 0.0 {
                        continue;
                    }
                    let order = if vol < 0.0 {
                        [tet[0], tet[1], tet[3], tet[2]]
                    } else {
                        [tet[0], tet[1], tet[2], tet[3]]
                    };

                    let ids = [
                        intern(
                            corner_key[order[0]],
                            corner_pos[order[0]],
                            &mut out_vertices,
                            &mut index_of,
                        ),
                        intern(
                            corner_key[order[1]],
                            corner_pos[order[1]],
                            &mut out_vertices,
                            &mut index_of,
                        ),
                        intern(
                            corner_key[order[2]],
                            corner_pos[order[2]],
                            &mut out_vertices,
                            &mut index_of,
                        ),
                        intern(
                            corner_key[order[3]],
                            corner_pos[order[3]],
                            &mut out_vertices,
                            &mut index_of,
                        ),
                    ];
                    tets.push(ids);
                }
            }
        }
    }

    if tets.is_empty() {
        return None;
    }

    Some(TetMesh {
        vertices: out_vertices,
        tets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
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

    #[test]
    fn empty_input_returns_none() {
        let (v, i) = cube_mesh(1.0);
        assert!(tetrahedralize(&[], &i, &TetMeshParams::new(16)).is_none());
        assert!(tetrahedralize(&v, &[], &TetMeshParams::new(16)).is_none());
    }

    #[test]
    fn cube_fills_with_positive_tets_inside_bounds() {
        let (v, i) = cube_mesh(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(16)).unwrap();
        assert!(mesh.tet_count() > 0);
        assert!(mesh.vertex_count() > 0);
        // Every tet has strictly positive signed volume and every referenced
        // vertex is in range and inside the cube.
        for t in &mesh.tets {
            for &id in t {
                assert!((id as usize) < mesh.vertex_count());
            }
            let a = mesh.vertices[t[0] as usize];
            let b = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            let d = mesh.vertices[t[3] as usize];
            assert!(tet_vol6(a, b, c, d) > 0.0);
        }
        for p in &mesh.vertices {
            assert!(p.x >= -1.0001 && p.x <= 1.0001);
            assert!(p.y >= -1.0001 && p.y <= 1.0001);
            assert!(p.z >= -1.0001 && p.z <= 1.0001);
        }
        // Interior fill underestimates the true volume (8) but recovers most.
        let vol = mesh.total_volume();
        assert!(vol > 4.0 && vol <= 8.0001, "volume was {vol}");
    }

    #[test]
    fn finer_resolution_recovers_more_volume() {
        let (v, i) = cube_mesh(1.0);
        let coarse = tetrahedralize(&v, &i, &TetMeshParams::new(8)).unwrap();
        let fine = tetrahedralize(&v, &i, &TetMeshParams::new(24)).unwrap();
        assert!(fine.total_volume() >= coarse.total_volume() - 1e-3);
        // The fine mesh should be within a few percent of the true volume.
        assert!(
            fine.total_volume() > 7.0,
            "fine volume {}",
            fine.total_volume()
        );
    }

    #[test]
    fn sphere_volume_approaches_analytic() {
        let (v, i) = icosphere(3);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(24)).unwrap();
        let analytic = 4.0 / 3.0 * core::f32::consts::PI; // unit sphere
        let vol = mesh.total_volume();
        assert!(vol > 0.6 * analytic, "sphere volume {vol} vs {analytic}");
        assert!(vol <= analytic + 0.05, "sphere volume {vol} vs {analytic}");
        for p in &mesh.vertices {
            assert!(p.length() <= 1.05, "vertex outside sphere: {p:?}");
        }
    }

    #[test]
    fn margin_shrinks_the_fill() {
        let (v, i) = cube_mesh(1.0);
        let full = tetrahedralize(&v, &i, &TetMeshParams::new(16)).unwrap();
        let eroded = tetrahedralize(&v, &i, &TetMeshParams::new(16).with_margin(0.3)).unwrap();
        assert!(eroded.total_volume() < full.total_volume());
    }

    #[test]
    fn tetrahedralization_is_deterministic() {
        let (v, i) = cube_mesh(1.0);
        let a = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let b = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        assert_eq!(a, b);
    }
}
