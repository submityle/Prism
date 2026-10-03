//! Balanced domain decomposition of the tetrahedral dual graph.
//!
//! Distributing a volumetric mesh across worker threads or GPU queues wants
//! roughly equal-sized partitions that each keep neighbouring tets together, so
//! the shared faces that cross a partition boundary (and therefore need
//! synchronisation) are few. This is the classic domain-decomposition problem
//! that AAA solvers solve before dispatching per-partition work.
//!
//! Unlike graph *colouring* ([`crate::collider::tet_coloring`]), which groups
//! tets so same-colour tets never touch (for conflict-free parallel relaxation),
//! partitioning groups tets so same-partition tets *do* touch (for locality).
//!
//! The strategy is deterministic greedy region growing over the dual graph
//! ([`crate::collider::tet_adjacency`]):
//!
//! - the target sizes are fixed up front so the partitions are exactly balanced
//!   (the first `tet_count % part_count` partitions take one extra tet), then
//! - each partition is grown breadth-first from the lowest-indexed unassigned
//!   tet, absorbing face-neighbours until it reaches its target size; if a local
//!   region is exhausted first, the next lowest unassigned tet reseeds the same
//!   partition.
//!
//! Growing one compact region at a time keeps a partition's tets spatially
//! together, which keeps the boundary-face count far below an interleaved
//! assignment, while the fixed target sizes guarantee exact load balance. A
//! fixed mesh always yields the same partitions.
//!
//! This is standard mesh partitioning; nothing here is derived from Unreal
//! Engine source.

use super::tet_adjacency::{build_tet_adjacency, TetAdjacency};

/// Parameters controlling the domain decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TetPartitionParams {
    /// Desired number of partitions. The achieved count is
    /// `min(part_count, tet_count)`; a value of zero is rejected.
    pub part_count: usize,
}

impl TetPartitionParams {
    /// Requests `part_count` partitions.
    #[must_use]
    pub fn new(part_count: usize) -> TetPartitionParams {
        TetPartitionParams { part_count }
    }
}

/// A balanced assignment of each tet to a partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetPartition {
    /// `parts[t]` is the partition id of tet `t`, in `0..part_count`.
    pub parts: Vec<u32>,
    /// Number of partitions actually produced (`min(requested, tet_count)`).
    pub part_count: usize,
}

impl TetPartition {
    /// Number of tets partitioned.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.parts.len()
    }

    /// The tet count of each partition, indexed by partition id.
    #[must_use]
    pub fn sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.part_count];
        for &p in &self.parts {
            sizes[p as usize] += 1;
        }
        sizes
    }

    /// Number of shared faces whose two tets fall in different partitions (the
    /// synchronisation boundary; each such face counted once).
    #[must_use]
    pub fn boundary_face_count(&self, adjacency: &TetAdjacency) -> usize {
        let mut cut = 0usize;
        for t in 0..adjacency.tet_count() {
            for nb in adjacency.neighbours[t].iter().flatten() {
                let nb = *nb as usize;
                if t < nb && self.parts[t] != self.parts[nb] {
                    cut += 1;
                }
            }
        }
        cut
    }
}

/// Partitions the tets of a dual graph into balanced, locality-preserving
/// regions via greedy breadth-first region growing.
///
/// Returns `None` when `params.part_count` is zero.
#[must_use]
pub fn partition_tet_adjacency(
    adjacency: &TetAdjacency,
    params: &TetPartitionParams,
) -> Option<TetPartition> {
    let n = adjacency.tet_count();
    if params.part_count == 0 {
        return None;
    }
    let part_count = params.part_count.min(n);

    let base = n / part_count;
    let rem = n % part_count;

    let mut assigned = vec![false; n];
    let mut parts = vec![0u32; n];
    let mut queue: Vec<usize> = Vec::new();
    // Lowest tet index that might still be unassigned; advances monotonically.
    let mut next_seed = 0usize;

    for p in 0..part_count {
        let target = base + usize::from(p < rem);
        let mut filled = 0usize;
        queue.clear();
        let mut head = 0usize;

        while filled < target {
            if head >= queue.len() {
                // Local region exhausted: reseed from the lowest unassigned tet.
                while assigned[next_seed] {
                    next_seed += 1;
                }
                assigned[next_seed] = true;
                parts[next_seed] = p as u32;
                filled += 1;
                queue.push(next_seed);
                continue;
            }

            let t = queue[head];
            head += 1;
            for nb in adjacency.neighbours[t].iter().flatten() {
                let nb = *nb as usize;
                if !assigned[nb] {
                    assigned[nb] = true;
                    parts[nb] = p as u32;
                    filled += 1;
                    queue.push(nb);
                    if filled >= target {
                        break;
                    }
                }
            }
        }
    }

    Some(TetPartition { parts, part_count })
}

/// Builds the dual graph of the mesh and partitions it in one call.
///
/// Returns `None` under the same conditions as [`build_tet_adjacency`] (empty
/// `tets`, out-of-range index, non-manifold face) or when `params.part_count`
/// is zero.
#[must_use]
pub fn partition_tet_mesh(
    num_vertices: usize,
    tets: &[[u32; 4]],
    params: &TetPartitionParams,
) -> Option<TetPartition> {
    let adjacency = build_tet_adjacency(num_vertices, tets)?;
    partition_tet_adjacency(&adjacency, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use glam::Vec3;

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
    fn single_partition_holds_everything() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let part = partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(1))
            .unwrap();
        assert_eq!(part.part_count, 1);
        assert!(part.parts.iter().all(|&p| p == 0));
    }

    #[test]
    fn more_parts_than_tets_clamps_to_tet_count() {
        let part = partition_tet_mesh(4, &[[0u32, 1, 2, 3]], &TetPartitionParams::new(8)).unwrap();
        assert_eq!(part.part_count, 1);
        assert_eq!(part.tet_count(), 1);
    }

    #[test]
    fn partitions_are_balanced() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let part = partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(4))
            .unwrap();
        assert_eq!(part.part_count, 4);
        let sizes = part.sizes();
        let max = *sizes.iter().max().unwrap();
        let min = *sizes.iter().min().unwrap();
        assert!(max - min <= 1, "imbalance {min}..{max} exceeds 1");
    }

    #[test]
    fn coverage_and_part_ids_in_range() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let part = partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(5))
            .unwrap();
        let sizes = part.sizes();
        assert_eq!(sizes.iter().sum::<usize>(), part.tet_count());
        assert!(
            sizes.iter().all(|&s| s >= 1),
            "every partition must be used"
        );
        assert!(part.parts.iter().all(|&p| (p as usize) < part.part_count));
    }

    #[test]
    fn region_growing_cuts_far_fewer_faces_than_interleaving() {
        // A locality-preserving region growth must cut dramatically fewer faces
        // than a round-robin (interleaved) assignment, which splits almost every
        // interior face across partitions.
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let n = adj.tet_count();
        let k = 4usize;

        let grown = partition_tet_adjacency(&adj, &TetPartitionParams::new(k)).unwrap();

        // Round-robin baseline: parts[t] = t % k interleaves neighbours.
        let round_robin = TetPartition {
            parts: (0..n).map(|t| (t % k) as u32).collect(),
            part_count: k,
        };

        let grown_cut = grown.boundary_face_count(&adj);
        let rr_cut = round_robin.boundary_face_count(&adj);
        assert!(
            grown_cut * 2 < rr_cut,
            "region-grown cut {grown_cut} not far below round-robin cut {rr_cut}"
        );
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(3))
            .unwrap();
        let b = partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(3))
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn zero_parts_or_invalid_mesh_returns_none() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        assert!(
            partition_tet_mesh(mesh.vertices.len(), &mesh.tets, &TetPartitionParams::new(0))
                .is_none()
        );
        assert!(partition_tet_mesh(0, &[], &TetPartitionParams::new(2)).is_none());
        assert!(partition_tet_mesh(4, &[[0u32, 1, 2, 9]], &TetPartitionParams::new(2)).is_none());
    }
}
