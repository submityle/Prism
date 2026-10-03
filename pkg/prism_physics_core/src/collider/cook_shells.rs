//! Triangle-soup collision cooking: clean, then split into ready shells.
//!
//! Downstream collision consumers (per-shell convex decomposition, `BVH`
//! builds, inertia estimation) expect clean, self-contained shells rather than
//! the raw index buffer an authoring or procedural tool emits. The canonical
//! AAA cooking order (`PhysX` `cookTriangleMesh`, `Jolt`'s mesh preparation) is
//! to *first* weld near-coincident vertices and drop degenerate/duplicate
//! faces, and only *then* separate the result into connected shells. Welding
//! before splitting matters: it fuses hairline cracks so a single logical shell
//! does not fragment, and it deduplicates vertices so each emitted shell is
//! compact.
//!
//! This module chains the two standalone passes
//! ([`weld_mesh`](crate::collider::weld_mesh) then
//! [`split_connected_components`](crate::collider::split_connected_components))
//! into one deterministic front door and reports what the clean pass removed.
//! Because welding has already merged coincident vertices into shared indices,
//! the split runs in purely topological (exact) mode. This is standard geometry
//! cooking; nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::connectivity::{
    split_connected_components, ConnectivityParams, MeshComponent,
};
use crate::collider::weld::{weld_mesh, WeldParams};

/// Parameters controlling the cook.
#[derive(Clone, Copy, Debug)]
pub struct CookShellParams {
    /// Vertices closer than this distance are welded into one before splitting.
    /// Must be finite and strictly positive.
    pub weld_epsilon: f32,
    /// When set, faces that become duplicates (same unordered vertex triple)
    /// after welding are collapsed to a single triangle.
    pub drop_duplicate_triangles: bool,
    /// Shells with fewer than this many triangles are discarded as noise. A
    /// value of `0` or `1` keeps every non-empty shell.
    pub min_triangles_per_shell: usize,
}

impl Default for CookShellParams {
    fn default() -> Self {
        Self {
            weld_epsilon: 1e-5,
            drop_duplicate_triangles: true,
            min_triangles_per_shell: 1,
        }
    }
}

/// A tally of what the cook changed, useful for logging and validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CookReport {
    /// Vertices in the input soup.
    pub input_vertices: usize,
    /// Triangles in the input soup.
    pub input_triangles: usize,
    /// Vertices remaining after welding.
    pub welded_vertices: usize,
    /// Triangles remaining after welding (before shell splitting).
    pub welded_triangles: usize,
    /// Input vertices merged away by welding.
    pub removed_vertices: usize,
    /// Input triangles dropped as degenerate or duplicate by welding.
    pub removed_triangles: usize,
    /// Connected shells kept after the minimum-size filter.
    pub shells: usize,
    /// Shells discarded by [`CookShellParams::min_triangles_per_shell`].
    pub dropped_small_shells: usize,
}

/// The cooked result: clean, self-contained shells plus a change report.
#[derive(Clone, Debug)]
pub struct CookedShells {
    /// One compact, welded shell per connected component, deterministically
    /// ordered (see [`split_connected_components`]).
    pub shells: Vec<MeshComponent>,
    /// What the cook changed.
    pub report: CookReport,
}

impl CookedShells {
    /// Number of shells produced.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shells.len()
    }

    /// Whether no shell survived the cook.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shells.is_empty()
    }
}

/// Cooks a triangle soup into clean, connected collision shells.
///
/// The soup is welded (coincident vertices merged, degenerate/duplicate faces
/// dropped) and then split into connected components; shells smaller than
/// [`CookShellParams::min_triangles_per_shell`] are discarded.
///
/// Returns `None` when the input is empty, when `weld_epsilon` is not finite
/// and strictly positive, or when no shell survives (every triangle was
/// degenerate, or every shell fell below the size threshold).
#[must_use]
pub fn cook_collision_shells(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: CookShellParams,
) -> Option<CookedShells> {
    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: params.weld_epsilon,
            drop_duplicate_triangles: params.drop_duplicate_triangles,
        },
    )?;

    // Welding already fused coincident vertices into shared indices, so an
    // exact (topological) split is both correct and avoids re-welding.
    let all_shells = split_connected_components(
        &welded.vertices,
        &welded.indices,
        ConnectivityParams::exact(),
    )?;

    let min_tris = params.min_triangles_per_shell.max(1);
    let total_shells = all_shells.len();
    let shells: Vec<MeshComponent> = all_shells
        .into_iter()
        .filter(|s| s.triangle_count() >= min_tris)
        .collect();
    if shells.is_empty() {
        return None;
    }

    let report = CookReport {
        input_vertices: vertices.len(),
        input_triangles: indices.len(),
        welded_vertices: welded.vertices.len(),
        welded_triangles: welded.indices.len(),
        removed_vertices: welded.removed_vertices,
        removed_triangles: welded.removed_triangles,
        shells: shells.len(),
        dropped_small_shells: total_shells - shells.len(),
    };

    Some(CookedShells { shells, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times.
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

    fn append_shifted(
        verts: &mut Vec<Vec3>,
        tris: &mut Vec<[u32; 3]>,
        mesh: &(Vec<Vec3>, Vec<[u32; 3]>),
        offset: Vec3,
    ) {
        let base = verts.len() as u32;
        for &v in &mesh.0 {
            verts.push(v + offset);
        }
        for t in &mesh.1 {
            tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
    }

    #[test]
    fn single_shell_cooks_to_one_component() {
        let (v, t) = icosphere(2);
        let cooked =
            cook_collision_shells(&v, &t, CookShellParams::default()).expect("cooks a shell");
        assert_eq!(cooked.len(), 1);
        assert!(!cooked.is_empty());
        assert_eq!(cooked.report.shells, 1);
        assert_eq!(cooked.report.input_triangles, t.len());
        assert_eq!(cooked.shells[0].triangle_count(), t.len());
    }

    #[test]
    fn two_separated_shells_cook_to_two() {
        let sphere = icosphere(1);
        let mut v = Vec::new();
        let mut t = Vec::new();
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(10.0, 0.0, 0.0));
        let cooked =
            cook_collision_shells(&v, &t, CookShellParams::default()).expect("cooks two shells");
        assert_eq!(cooked.len(), 2);
        assert_eq!(cooked.report.shells, 2);
    }

    #[test]
    fn duplicated_vertices_are_welded_before_splitting() {
        // Author the sphere with every position duplicated and faces pointing at
        // the second copy: without welding this would read as a torn mesh, but
        // welding fuses the copies back into one compact shell.
        let (base_v, base_t) = icosphere(1);
        let mut v: Vec<Vec3> = Vec::new();
        for &p in &base_v {
            v.push(p);
            v.push(p); // duplicate copy at the same position
        }
        // Remap each original index to its duplicate copy (odd slot).
        let t: Vec<[u32; 3]> = base_t
            .iter()
            .map(|f| [f[0] * 2 + 1, f[1] * 2 + 1, f[2] * 2 + 1])
            .collect();

        let cooked =
            cook_collision_shells(&v, &t, CookShellParams::default()).expect("welds then cooks");
        assert_eq!(cooked.len(), 1, "welding fuses the duplicated copies");
        assert_eq!(cooked.report.input_vertices, base_v.len() * 2);
        assert_eq!(cooked.report.welded_vertices, base_v.len());
        assert_eq!(cooked.report.removed_vertices, base_v.len());
        assert_eq!(cooked.shells[0].vertex_count(), base_v.len());
    }

    #[test]
    fn small_shells_are_dropped_by_the_filter() {
        let sphere = icosphere(1);
        let mut v = Vec::new();
        let mut t = Vec::new();
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        // A lone triangle far away forms a 1-triangle shell.
        let base = v.len() as u32;
        v.push(Vec3::new(50.0, 0.0, 0.0));
        v.push(Vec3::new(51.0, 0.0, 0.0));
        v.push(Vec3::new(50.0, 1.0, 0.0));
        t.push([base, base + 1, base + 2]);

        let params = CookShellParams {
            min_triangles_per_shell: 2,
            ..CookShellParams::default()
        };
        let cooked = cook_collision_shells(&v, &t, params).expect("keeps the big shell");
        assert_eq!(cooked.len(), 1);
        assert_eq!(cooked.report.dropped_small_shells, 1);
        assert_eq!(cooked.shells[0].triangle_count(), sphere.1.len());
    }

    #[test]
    fn duplicate_face_is_reported_removed() {
        let (v, mut t) = icosphere(1);
        let good = t.len();
        // Re-add the first face verbatim: a duplicate to be collapsed.
        t.push(t[0]);
        let cooked =
            cook_collision_shells(&v, &t, CookShellParams::default()).expect("cooks a shell");
        assert!(cooked.report.removed_triangles >= 1);
        assert_eq!(cooked.report.welded_triangles, good);
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        assert!(cook_collision_shells(&[], &[], CookShellParams::default()).is_none());
        let (v, t) = icosphere(0);
        let bad = CookShellParams {
            weld_epsilon: 0.0,
            ..CookShellParams::default()
        };
        assert!(cook_collision_shells(&v, &t, bad).is_none());
    }

    #[test]
    fn all_shells_below_threshold_yield_none() {
        // One tiny shell, threshold above it.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let t = vec![[0u32, 1, 2]];
        let params = CookShellParams {
            min_triangles_per_shell: 2,
            ..CookShellParams::default()
        };
        assert!(cook_collision_shells(&v, &t, params).is_none());
    }
}
