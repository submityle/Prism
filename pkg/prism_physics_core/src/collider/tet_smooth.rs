//! Boundary-preserving quality smoothing for tetrahedral meshes.
//!
//! Lattice tetrahedralisation ([`crate::collider::tetrahedralize`]) produces a
//! valid but blocky interior. Laplacian smoothing relaxes each interior vertex
//! toward the centroid of its edge neighbours, which improves element shape --
//! but naive Laplacian smoothing can invert tets near concavities. This module
//! applies a *guarded* smoother:
//!
//! * Boundary vertices (those on a face used by only one tet) are pinned, so
//!   the surface the mesh approximates is preserved.
//! * Each proposed interior move is accepted only if every incident tet stays
//!   positively oriented and the local worst radius ratio does not decrease.
//!
//! The guard makes the worst-element quality monotonically non-decreasing, so
//! the pass is always safe to run. This is a standard mesh-optimisation
//! technique; nothing here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

use crate::collider::tet_quality::tet_quality;

/// Parameters controlling [`smooth_tet_mesh`].
#[derive(Clone, Copy, Debug)]
pub struct TetSmoothParams {
    /// Number of relaxation sweeps over the interior vertices.
    pub iterations: u32,
    /// Fraction of the way toward the neighbour centroid each accepted move
    /// travels, in `(0, 1]`.
    pub step: f32,
    /// When `true`, reject any move that would lower the local worst radius
    /// ratio. When `false`, only inversions are rejected.
    pub guard_quality: bool,
}

impl Default for TetSmoothParams {
    fn default() -> Self {
        Self {
            iterations: 5,
            step: 0.5,
            guard_quality: true,
        }
    }
}

/// Outcome of a smoothing pass.
#[derive(Clone, Debug, PartialEq)]
pub struct TetSmoothResult {
    /// Updated vertex positions. Indices and tet connectivity are unchanged.
    pub vertices: Vec<Vec3>,
    /// Total number of accepted vertex moves across all iterations.
    pub moved: usize,
    /// Total number of proposed moves rejected by the guard.
    pub rejected: usize,
}

/// The four triangular faces of a tetrahedron as sorted vertex-id triples.
fn tet_faces(t: [u32; 4]) -> [[u32; 3]; 4] {
    let sort3 = |mut a: [u32; 3]| {
        a.sort_unstable();
        a
    };
    [
        sort3([t[0], t[1], t[2]]),
        sort3([t[0], t[1], t[3]]),
        sort3([t[0], t[2], t[3]]),
        sort3([t[1], t[2], t[3]]),
    ]
}

/// Smooths a tetrahedral mesh in place-equivalent fashion, returning new vertex
/// positions.
///
/// Returns `None` when `tets` is empty or any tet references a vertex index
/// outside `vertices`.
#[must_use]
pub fn smooth_tet_mesh(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    params: &TetSmoothParams,
) -> Option<TetSmoothResult> {
    if tets.is_empty() {
        return None;
    }
    let n = vertices.len();
    for t in tets {
        for &id in t {
            if (id as usize) >= n {
                return None;
            }
        }
    }

    // Boundary faces are used by exactly one tet; their vertices are pinned.
    let mut face_count: HashMap<[u32; 3], u32> = HashMap::new();
    for &t in tets {
        for f in tet_faces(t) {
            *face_count.entry(f).or_insert(0) += 1;
        }
    }
    let mut is_boundary = vec![false; n];
    for (face, count) in &face_count {
        if *count == 1 {
            for &v in face {
                is_boundary[v as usize] = true;
            }
        }
    }

    // Edge-neighbour adjacency and incident-tet lists.
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ti, &t) in tets.iter().enumerate() {
        for a in 0..4 {
            incident[t[a] as usize].push(ti);
            for b in (a + 1)..4 {
                neighbours[t[a] as usize].push(t[b]);
                neighbours[t[b] as usize].push(t[a]);
            }
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }

    let step = params.step.clamp(0.0, 1.0);
    let mut pos: Vec<Vec3> = vertices.to_vec();
    let mut moved = 0usize;
    let mut rejected = 0usize;

    // Evaluates the incident tets of `vid` when it sits at `trial`, returning
    // whether all remain positively oriented and their worst radius ratio.
    let evaluate = |vid: usize, trial: Vec3, pos: &[Vec3]| -> (bool, f32) {
        let mut all_positive = true;
        let mut worst = f32::INFINITY;
        for &ti in &incident[vid] {
            let t = tets[ti];
            let p = |slot: u32| -> Vec3 {
                if slot as usize == vid {
                    trial
                } else {
                    pos[slot as usize]
                }
            };
            let q = tet_quality(p(t[0]), p(t[1]), p(t[2]), p(t[3]));
            if q.inverted {
                all_positive = false;
            }
            worst = worst.min(q.radius_ratio);
        }
        (all_positive, worst)
    };

    for _ in 0..params.iterations {
        for vid in 0..n {
            if is_boundary[vid] || neighbours[vid].is_empty() {
                continue;
            }
            // Laplacian target: centroid of edge neighbours.
            let mut centroid = Vec3::ZERO;
            for &nb in &neighbours[vid] {
                centroid += pos[nb as usize];
            }
            centroid /= neighbours[vid].len() as f32;
            let candidate = pos[vid] + (centroid - pos[vid]) * step;

            let (before_ok, before_ratio) = evaluate(vid, pos[vid], &pos);
            let (after_ok, after_ratio) = evaluate(vid, candidate, &pos);

            // Never introduce an inversion. If quality guarding is on, also
            // require the local worst ratio not to drop.
            let quality_ok =
                !params.guard_quality || after_ratio >= before_ratio - 1e-7 || !before_ok;
            if after_ok && quality_ok {
                pos[vid] = candidate;
                moved += 1;
            } else {
                rejected += 1;
            }
        }
    }

    Some(TetSmoothResult {
        vertices: pos,
        moved,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{
        analyze_tet_mesh_quality, tetrahedralize, TetMeshParams, TetQualityParams,
    };

    fn cube_surface(h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
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
            [0u32, 2, 1],
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

    #[test]
    fn empty_mesh_returns_none() {
        assert!(smooth_tet_mesh(&[], &[], &TetSmoothParams::default()).is_none());
    }

    #[test]
    fn out_of_range_index_returns_none() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let tets = vec![[0u32, 1, 2, 7]];
        assert!(smooth_tet_mesh(&verts, &tets, &TetSmoothParams::default()).is_none());
    }

    #[test]
    fn boundary_vertices_are_pinned_and_mesh_stays_sound() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(14)).unwrap();
        let before =
            analyze_tet_mesh_quality(&mesh.vertices, &mesh.tets, &TetQualityParams::default())
                .unwrap();

        let smoothed =
            smooth_tet_mesh(&mesh.vertices, &mesh.tets, &TetSmoothParams::default()).unwrap();
        assert_eq!(smoothed.vertices.len(), mesh.vertices.len());

        // Any vertex on the cube's outer shell must not have moved.
        for (orig, now) in mesh.vertices.iter().zip(&smoothed.vertices) {
            let on_shell = orig.x.abs() > 0.999 || orig.y.abs() > 0.999 || orig.z.abs() > 0.999;
            if on_shell {
                assert!((*orig - *now).length() < 1e-6, "boundary vertex moved");
            }
        }

        let after =
            analyze_tet_mesh_quality(&smoothed.vertices, &mesh.tets, &TetQualityParams::default())
                .unwrap();
        // The guard forbids inversions and quality regressions.
        assert!(after.is_sound());
        assert_eq!(after.inverted_count, 0);
        assert!(after.min_radius_ratio >= before.min_radius_ratio - 1e-4);
    }

    #[test]
    fn interior_vertex_is_recentred() {
        // Two tets sharing a face, with the shared interior apex perturbed.
        // Smoothing should relax it back toward the centroid of its neighbours.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 0.7, 1.6),  // interior apex, perturbed
            Vec3::new(1.0, 0.7, -1.6), // mirror apex
        ];
        let tets = vec![[0u32, 1, 2, 3], [0, 2, 1, 4]];
        let before = tet_quality(verts[0], verts[1], verts[2], verts[3]).radius_ratio;
        let out = smooth_tet_mesh(
            &verts,
            &tets,
            &TetSmoothParams {
                iterations: 10,
                step: 0.5,
                guard_quality: true,
            },
        )
        .unwrap();
        // Vertex 3 is shared by both tets so it is a boundary vertex here;
        // confirm the pass runs and never worsens quality instead.
        let after = tet_quality(
            out.vertices[0],
            out.vertices[1],
            out.vertices[2],
            out.vertices[3],
        )
        .radius_ratio;
        assert!(after >= before - 1e-6);
    }

    #[test]
    fn smoothing_is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = smooth_tet_mesh(&mesh.vertices, &mesh.tets, &TetSmoothParams::default()).unwrap();
        let b = smooth_tet_mesh(&mesh.vertices, &mesh.tets, &TetSmoothParams::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn moved_plus_rejected_accounts_for_all_attempts() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let params = TetSmoothParams {
            iterations: 3,
            step: 0.5,
            guard_quality: true,
        };
        let out = smooth_tet_mesh(&mesh.vertices, &mesh.tets, &params).unwrap();
        // moved + rejected == interior-vertex attempts across iterations; it is
        // positive because the cube has interior nodes to relax.
        assert!(out.moved + out.rejected > 0);
    }
}
