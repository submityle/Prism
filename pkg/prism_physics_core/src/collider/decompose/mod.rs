//! Approximate convex decomposition (ACD) of a concave triangle mesh.
//!
//! Dynamic rigid bodies collide fastest and most robustly against *convex*
//! shapes, but authored art is routinely concave (a chair, a wrench, a torus).
//! Production engines therefore *cook* a concave mesh into a small set of
//! convex hulls that together approximate the original solid -- Unreal's Chaos,
//! `PhysX` and Jolt all ship a variant of the V-HACD algorithm for exactly this.
//! [`convex_decompose`] is Prism's CPU cooker: it voxelizes the mesh, then
//! recursively splits the worst (most concave) part along the axis-aligned plane
//! that best reduces concavity until every part is convex enough or the hull
//! budget is exhausted, emitting one [`ConvexMeshData`] per final part.
//!
//! # Pipeline
//!
//! 1. [`voxel::VoxelGrid::voxelize`] rasterizes the mesh into a solid occupancy
//!    grid (ray-parity inside test).
//! 2. The occupied cells form the initial part. While the hull budget allows,
//!    the part with the greatest [`concavity`](concavity::concavity) above the
//!    acceptance floor is removed and cut in two by
//!    [`split::best_split`].
//! 3. Each surviving part's boundary-cell corners are fed to
//!    [`ConvexMeshData::from_points`], producing a ready-to-use convex collider.
//!
//! The whole process is deterministic: parts are selected by concavity with a
//! first-index tie-break, split planes come from a fixed candidate schedule, and
//! the hull builder is itself bit-reproducible. Identical inputs therefore yield
//! identical hull sets across runs -- a hard requirement for the engine's
//! cross-run state hashing.
//!
//! # Provenance
//!
//! The voxel-decomposition pipeline follows the publicly described V-HACD
//! approach (Khaled Mamou, *Approximate Convex Decomposition*) and standard
//! computational geometry. This module contains **no Unreal Engine source or
//! derived code**.

pub mod concavity;
pub mod params;
pub mod split;
pub mod voxel;

pub use params::DecompositionParams;

use glam::Vec3;

use crate::collider::ConvexMeshData;
use concavity::{concavity, part_corner_points, part_volume};
use split::best_split;
use voxel::VoxelGrid;

/// Candidate split planes probed per axis at each recursion step.
const MAX_PLANES_PER_AXIS: usize = 16;

/// Decomposes the triangle mesh `(vertices, triangles)` into a set of convex
/// hulls that together approximate the original solid.
///
/// Returns one [`ConvexMeshData`] per convex part. The hull count never exceeds
/// [`DecompositionParams::max_convex_hulls`]. A convex (or nearly convex) input
/// collapses to a single hull. If the mesh cannot be voxelized (empty,
/// triangle-less, or zero-extent) the function falls back to the single convex
/// hull of the raw vertices, returning an empty vector only when even that is
/// degenerate.
#[must_use]
pub fn convex_decompose(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    params: DecompositionParams,
) -> Vec<ConvexMeshData> {
    let params = params.sanitized();

    let Some(grid) = VoxelGrid::voxelize(vertices, triangles, params.resolution) else {
        return single_hull_fallback(vertices);
    };

    let root = grid.occupied_indices();
    if root.is_empty() {
        return single_hull_fallback(vertices);
    }

    let total_volume = part_volume(&root, &grid);
    if total_volume <= 0.0 || !total_volume.is_finite() {
        return single_hull_fallback(vertices);
    }

    let parts = partition_parts(&grid, root, total_volume, params);

    parts
        .iter()
        .filter_map(|cells| {
            let points = part_corner_points(cells, &grid);
            ConvexMeshData::from_points(&points)
        })
        .collect()
}

/// Runs the recursive split loop, returning the final list of voxel parts.
fn partition_parts(
    grid: &VoxelGrid,
    root: Vec<usize>,
    total_volume: f32,
    params: DecompositionParams,
) -> Vec<Vec<usize>> {
    let concavity_floor = params.max_concavity * total_volume;
    let volume_floor = params.min_volume_fraction * total_volume;
    let max_hulls = params.max_convex_hulls as usize;

    // Pending parts carry their recursion depth; `done` holds accepted parts.
    let mut pending: Vec<(Vec<usize>, u32)> = vec![(root, 0)];
    let mut done: Vec<Vec<usize>> = Vec::new();

    loop {
        if pending.len() + done.len() >= max_hulls {
            break;
        }

        // Pick the eligible pending part with the greatest concavity.
        let mut target: Option<usize> = None;
        let mut target_concavity = concavity_floor;
        for (idx, (cells, depth)) in pending.iter().enumerate() {
            if *depth >= params.max_recursion_depth || cells.len() < 2 {
                continue;
            }
            if part_volume(cells, grid) <= volume_floor {
                continue;
            }
            let c = concavity(cells, grid);
            if c > target_concavity {
                target_concavity = c;
                target = Some(idx);
            }
        }

        let Some(idx) = target else {
            break; // nothing left worth splitting
        };

        let (cells, depth) = pending.remove(idx);
        match best_split(&cells, grid, MAX_PLANES_PER_AXIS) {
            Some(result) => {
                pending.push((result.left, depth + 1));
                pending.push((result.right, depth + 1));
            }
            None => done.push(cells),
        }
    }

    done.extend(pending.into_iter().map(|(cells, _)| cells));
    done
}

/// Falls back to the single convex hull of the raw vertices.
fn single_hull_fallback(vertices: &[Vec3]) -> Vec<ConvexMeshData> {
    ConvexMeshData::from_points(vertices).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed, watertight L-shaped prism (concave) spanning the `xy` L-polygon
    /// `(0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)` extruded along `z` in `[0, 1]`.
    fn l_prism() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let xy = [
            (0.0f32, 0.0f32), // a 0
            (2.0, 0.0),       // b 1
            (2.0, 1.0),       // c 2
            (1.0, 1.0),       // d 3
            (1.0, 2.0),       // e 4
            (0.0, 2.0),       // f 5
            (0.0, 1.0),       // g 6
        ];
        let mut verts = Vec::with_capacity(14);
        for &(x, y) in &xy {
            verts.push(Vec3::new(x, y, 0.0));
        }
        for &(x, y) in &xy {
            verts.push(Vec3::new(x, y, 1.0));
        }

        // Cap triangulation of the L polygon (indices into the 7 outline verts).
        let cap: [[u32; 3]; 5] = [[0, 1, 2], [0, 2, 3], [0, 3, 6], [6, 3, 4], [6, 4, 5]];
        // Outer boundary edges, in CCW order with the colinear vertex g split in.
        let boundary: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)];

        let mut tris: Vec<[u32; 3]> = Vec::new();
        // Top cap (z = 1) uses the +7 vertices as given.
        for t in &cap {
            tris.push([t[0] + 7, t[1] + 7, t[2] + 7]);
        }
        // Bottom cap (z = 0) with reversed winding.
        for t in &cap {
            tris.push([t[0], t[2], t[1]]);
        }
        // Side walls: each boundary edge becomes two triangles.
        for &(p, q) in &boundary {
            tris.push([p, q, q + 7]);
            tris.push([p, q + 7, p + 7]);
        }
        (verts, tris)
    }

    /// A closed box mesh (outward CCW) spanning `[min, max]`.
    fn box_mesh(min: Vec3, max: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let t = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (v, t)
    }

    fn test_params() -> DecompositionParams {
        DecompositionParams {
            resolution: 24,
            max_convex_hulls: 8,
            ..DecompositionParams::default()
        }
    }

    #[test]
    fn convex_box_yields_single_hull() {
        let (v, t) = box_mesh(Vec3::splat(-1.0), Vec3::splat(1.0));
        let hulls = convex_decompose(&v, &t, test_params());
        assert_eq!(hulls.len(), 1, "a convex box must not be subdivided");
        assert!((hulls[0].volume() - 8.0).abs() < 0.2 * 8.0);
    }

    #[test]
    fn concave_l_prism_splits_into_multiple_hulls() {
        let (v, t) = l_prism();
        let hulls = convex_decompose(&v, &t, test_params());
        assert!(
            hulls.len() >= 2,
            "concave L must decompose, got {}",
            hulls.len()
        );
        assert!(hulls.len() <= test_params().max_convex_hulls as usize);
        for h in &hulls {
            assert!(h.volume() > 0.0, "every hull must be a real solid");
        }
    }

    #[test]
    fn decomposition_covers_most_of_the_volume() {
        let (v, t) = l_prism();
        let hulls = convex_decompose(&v, &t, test_params());
        let hull_volume: f32 = hulls.iter().map(|h| h.volume()).sum();
        // The L prism has solid volume 3; the convex pieces should sum to at
        // least that (parts may overlap slightly at the cut, never undershoot
        // badly).
        assert!(hull_volume >= 0.8 * 3.0, "hull volume {hull_volume}");
    }

    #[test]
    fn decomposition_is_deterministic() {
        let (v, t) = l_prism();
        let a = convex_decompose(&v, &t, test_params());
        let b = convex_decompose(&v, &t, test_params());
        assert_eq!(a.len(), b.len());
        for (ha, hb) in a.iter().zip(b.iter()) {
            assert_eq!(ha.vertices().len(), hb.vertices().len());
            assert!((ha.volume() - hb.volume()).abs() < 1e-6);
        }
    }

    #[test]
    fn hull_budget_is_respected() {
        let (v, t) = l_prism();
        let params = DecompositionParams {
            resolution: 24,
            max_convex_hulls: 3,
            max_concavity: 0.0, // force maximal splitting pressure
            ..DecompositionParams::default()
        };
        let hulls = convex_decompose(&v, &t, params);
        assert!(hulls.len() <= 3, "exceeded hull budget: {}", hulls.len());
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert!(convex_decompose(&[], &[], test_params()).is_empty());
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        // A single flat triangle cannot voxelize nor form a solid hull.
        assert!(convex_decompose(&flat, &[[0, 1, 2]], test_params()).is_empty());
    }
}
