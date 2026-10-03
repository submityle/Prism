//! Surface-conformal projection of a tetrahedral mesh boundary.
//!
//! Lattice tetrahedralisation ([`crate::collider::tetrahedralize`]) places
//! boundary vertices on a regular grid, so the recovered surface is a blocky
//! approximation of the true shape. This module snaps the boundary vertices
//! onto the zero iso-surface of a signed-distance field
//! ([`crate::collider::sdf::MeshSdf`]) using guarded Newton projection, giving
//! the volume mesh a faithful surface while leaving its interior intact.
//!
//! Each projection step is accepted only if it keeps every incident tet
//! positively oriented; otherwise the step is backtracked and, failing that,
//! skipped. This guarantees the mesh never inverts. The technique is standard
//! mesh conforming; nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::sdf::MeshSdf;
use crate::collider::tet_boundary::extract_tet_boundary;
use crate::collider::tet_quality::tet_quality;

/// Parameters controlling [`conform_tet_boundary`].
#[derive(Clone, Copy, Debug)]
pub struct TetConformParams {
    /// Number of Newton projection sweeps over the boundary vertices.
    pub iterations: u32,
    /// When `true`, reject any step that would invert an incident tet. When
    /// `false`, projection is applied unconditionally.
    pub guard: bool,
}

impl Default for TetConformParams {
    fn default() -> Self {
        Self {
            iterations: 3,
            guard: true,
        }
    }
}

/// Outcome of a conforming pass.
#[derive(Clone, Debug, PartialEq)]
pub struct TetConformResult {
    /// Updated vertex positions. Indices and tet connectivity are unchanged.
    pub vertices: Vec<Vec3>,
    /// Number of boundary vertices that moved at least once.
    pub projected: usize,
    /// Number of projection steps rejected by the inversion guard.
    pub rejected: usize,
    /// Largest absolute signed distance over the boundary vertices after
    /// projection; a measure of residual surface error.
    pub max_residual: f32,
}

/// Backtracking fractions tried, in order, when a full projection step is
/// rejected by the guard.
const BACKTRACK: [f32; 4] = [1.0, 0.5, 0.25, 0.125];

/// Projects the boundary vertices of a tetrahedral mesh onto the zero
/// iso-surface of `sdf`, keeping the interior and connectivity fixed.
///
/// Returns `None` when `tets` is empty or any tet references a vertex index
/// outside `vertices`.
#[must_use]
pub fn conform_tet_boundary(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    sdf: &MeshSdf,
    params: &TetConformParams,
) -> Option<TetConformResult> {
    let boundary = extract_tet_boundary(vertices, tets)?;

    let n = vertices.len();
    // Incident-tet lists for the boundary vertices only.
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut is_boundary = vec![false; n];
    for &vid in &boundary.boundary_vertices {
        is_boundary[vid as usize] = true;
    }
    for (ti, t) in tets.iter().enumerate() {
        for &slot in t {
            if is_boundary[slot as usize] {
                incident[slot as usize].push(ti);
            }
        }
    }

    let mut pos: Vec<Vec3> = vertices.to_vec();
    let mut projected = 0usize;
    let mut rejected = 0usize;

    // True iff every incident tet of `vid` stays positively oriented when the
    // vertex sits at `trial`.
    let guard_ok = |vid: usize, trial: Vec3, pos: &[Vec3]| -> bool {
        for &ti in &incident[vid] {
            let t = tets[ti];
            let p = |slot: u32| -> Vec3 {
                if slot as usize == vid {
                    trial
                } else {
                    pos[slot as usize]
                }
            };
            if tet_quality(p(t[0]), p(t[1]), p(t[2]), p(t[3])).inverted {
                return false;
            }
        }
        true
    };

    for _ in 0..params.iterations {
        for &vid in &boundary.boundary_vertices {
            let vid = vid as usize;
            let cur = pos[vid];
            let target = sdf.project_to_surface(cur);
            let delta = target - cur;
            if delta.length_squared() <= 0.0 {
                continue;
            }

            if !params.guard {
                pos[vid] = target;
                projected += 1;
                continue;
            }

            let mut accepted = false;
            for &alpha in &BACKTRACK {
                let trial = cur + delta * alpha;
                if guard_ok(vid, trial, &pos) {
                    pos[vid] = trial;
                    projected += 1;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                rejected += 1;
            }
        }
    }

    let mut max_residual = 0.0f32;
    for &vid in &boundary.boundary_vertices {
        max_residual = max_residual.max(sdf.sample(pos[vid as usize]).abs());
    }

    Some(TetConformResult {
        vertices: pos,
        projected,
        rejected,
        max_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::sdf::{MeshSdf, SdfBuildParams};
    use crate::collider::tet_quality::{analyze_tet_mesh_quality, TetQualityParams};
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use std::collections::HashMap as StdHashMap;

    /// Watertight unit sphere: subdivided icosahedron, vertices normalised.
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

    fn sphere_setup() -> (Vec<Vec3>, Vec<[u32; 4]>, MeshSdf) {
        let (v, i) = icosphere(2);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(16)).unwrap();
        let sdf = MeshSdf::from_mesh(&v, &i, SdfBuildParams::from_resolution(2.0, 32)).unwrap();
        (mesh.vertices, mesh.tets, sdf)
    }

    fn max_boundary_residual(vertices: &[Vec3], tets: &[[u32; 4]], sdf: &MeshSdf) -> f32 {
        let b = extract_tet_boundary(vertices, tets).unwrap();
        let mut m = 0.0f32;
        for &vid in &b.boundary_vertices {
            m = m.max(sdf.sample(vertices[vid as usize]).abs());
        }
        m
    }

    #[test]
    fn empty_tets_returns_none() {
        let (_, _, sdf) = sphere_setup();
        assert!(conform_tet_boundary(&[], &[], &sdf, &TetConformParams::default()).is_none());
    }

    #[test]
    fn out_of_range_index_returns_none() {
        let (v, _, sdf) = sphere_setup();
        assert!(conform_tet_boundary(
            &v,
            &[[0u32, 1, 2, u32::MAX]],
            &sdf,
            &TetConformParams::default()
        )
        .is_none());
    }

    #[test]
    fn projection_reduces_surface_residual() {
        let (v, tets, sdf) = sphere_setup();
        let before = max_boundary_residual(&v, &tets, &sdf);
        let out = conform_tet_boundary(&v, &tets, &sdf, &TetConformParams::default()).unwrap();
        assert!(out.projected > 0, "no boundary vertex moved");
        assert!(
            out.max_residual < before,
            "residual did not shrink: {before} -> {}",
            out.max_residual
        );
    }

    #[test]
    fn interior_vertices_are_untouched() {
        let (v, tets, sdf) = sphere_setup();
        let b = extract_tet_boundary(&v, &tets).unwrap();
        let mut is_boundary = vec![false; v.len()];
        for &vid in &b.boundary_vertices {
            is_boundary[vid as usize] = true;
        }
        let out = conform_tet_boundary(&v, &tets, &sdf, &TetConformParams::default()).unwrap();
        for (idx, (orig, now)) in v.iter().zip(&out.vertices).enumerate() {
            if !is_boundary[idx] {
                assert!((*orig - *now).length() < 1e-6, "interior vertex moved");
            }
        }
    }

    #[test]
    fn guarded_mesh_stays_sound() {
        let (v, tets, sdf) = sphere_setup();
        let out = conform_tet_boundary(&v, &tets, &sdf, &TetConformParams::default()).unwrap();
        let report =
            analyze_tet_mesh_quality(&out.vertices, &tets, &TetQualityParams::default()).unwrap();
        assert_eq!(report.inverted_count, 0, "guard allowed an inversion");
    }

    #[test]
    fn conforming_is_deterministic() {
        let (v, tets, sdf) = sphere_setup();
        let a = conform_tet_boundary(&v, &tets, &sdf, &TetConformParams::default()).unwrap();
        let b = conform_tet_boundary(&v, &tets, &sdf, &TetConformParams::default()).unwrap();
        assert_eq!(a, b);
    }
}
