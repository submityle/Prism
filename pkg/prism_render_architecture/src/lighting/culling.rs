//! View-space clustered light culling.
//!
//! Deferred and forward+ shading both need to know, per screen cluster, which
//! dynamic lights can affect it, so the lighting loop iterates only nearby
//! lights instead of every light in the scene. This module performs that
//! assignment: given the view-space bounds of each cluster (froxel) and a
//! bounding sphere per light, it builds a compact per-cluster light list in a
//! CSR layout ready for GPU upload.
//!
//! The froxel Z distribution (typically logarithmic) is decided elsewhere and
//! handed in as explicit [`ClusterBounds`], keeping this layer a pure geometric
//! overlap test with no transcendental math. Overlap uses the squared distance
//! from a light's center to the cluster AABB compared against the squared
//! radius, so no square roots are taken. Assignment is deterministic: lights
//! are tested in input order and each cluster keeps at most `max_per_cluster`
//! of them, recording any overflow so quality scaling can react.

use alloc::vec::Vec;

use super::LightHandle;

/// A light's view-space bounding sphere for culling.
#[derive(Clone, Copy, Debug)]
pub struct LightVolume {
    /// Light this volume belongs to.
    pub light: LightHandle,
    /// Sphere center in view space.
    pub center: [f32; 3],
    /// Sphere radius (light influence range) in view space.
    pub radius: f32,
}

/// A cluster's axis-aligned bounds in view space.
#[derive(Clone, Copy, Debug)]
pub struct ClusterBounds {
    /// Minimum corner in view space.
    pub min: [f32; 3],
    /// Maximum corner in view space.
    pub max: [f32; 3],
}

/// Per-cluster light lists in a compact CSR layout.
///
/// `light_indices` is the concatenation of every cluster's list; cluster `c`
/// owns the slice `light_indices[offsets[c]..offsets[c] + counts[c]]`. Indices
/// point into the `volumes` slice passed to [`assign_cluster_lights`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClusterLightAssignment {
    /// Start of each cluster's slice within `light_indices`.
    pub offsets: Vec<u32>,
    /// Number of lights assigned to each cluster.
    pub counts: Vec<u32>,
    /// Concatenated light indices for every cluster.
    pub light_indices: Vec<u32>,
    /// Number of light-cluster assignments dropped because a cluster hit its
    /// `max_per_cluster` cap. Non-zero means the cap is too small for the scene.
    pub overflow: u32,
}

impl ClusterLightAssignment {
    /// Number of clusters described by this assignment.
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.counts.len()
    }

    /// The light indices assigned to cluster `cluster`, or an empty slice when
    /// the cluster index is out of range.
    #[must_use]
    pub fn cluster_lights(&self, cluster: usize) -> &[u32] {
        let Some(&offset) = self.offsets.get(cluster) else {
            return &[];
        };
        let count = self.counts[cluster] as usize;
        let start = offset as usize;
        &self.light_indices[start..start + count]
    }
}

/// Squared distance from a point to an axis-aligned box, per axis clamped.
fn distance_sq_point_aabb(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut total = 0.0_f32;
    let mut axis = 0;
    while axis < 3 {
        let p = point[axis];
        if p < min[axis] {
            let d = min[axis] - p;
            total += d * d;
        } else if p > max[axis] {
            let d = p - max[axis];
            total += d * d;
        }
        axis += 1;
    }
    total
}

/// Returns `true` when a light's sphere overlaps a cluster's box.
#[must_use]
pub fn light_overlaps_cluster(volume: LightVolume, cluster: ClusterBounds) -> bool {
    let distance_sq = distance_sq_point_aabb(volume.center, cluster.min, cluster.max);
    distance_sq <= volume.radius * volume.radius
}

/// Builds per-cluster light lists by testing every light against every cluster.
///
/// Lights are tested in input order and appended to each overlapping cluster's
/// list until the cluster reaches `max_per_cluster`, after which further
/// overlaps for that cluster increment [`ClusterLightAssignment::overflow`]. The
/// resulting CSR layout is deterministic and ready for GPU upload.
#[must_use]
pub fn assign_cluster_lights(
    clusters: &[ClusterBounds],
    volumes: &[LightVolume],
    max_per_cluster: u32,
) -> ClusterLightAssignment {
    let mut assignment = ClusterLightAssignment {
        offsets: Vec::with_capacity(clusters.len()),
        counts: Vec::with_capacity(clusters.len()),
        light_indices: Vec::new(),
        overflow: 0,
    };

    for &cluster in clusters {
        let offset = assignment.light_indices.len() as u32;
        assignment.offsets.push(offset);
        let mut count = 0_u32;
        for (index, &volume) in volumes.iter().enumerate() {
            if !light_overlaps_cluster(volume, cluster) {
                continue;
            }
            if count >= max_per_cluster {
                assignment.overflow += 1;
                continue;
            }
            assignment.light_indices.push(index as u32);
            count += 1;
        }
        assignment.counts.push(count);
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(light: u32, center: [f32; 3], radius: f32) -> LightVolume {
        LightVolume {
            light: LightHandle(light),
            center,
            radius,
        }
    }

    fn unit_cluster(origin: [f32; 3]) -> ClusterBounds {
        ClusterBounds {
            min: origin,
            max: [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0],
        }
    }

    #[test]
    fn light_inside_cluster_overlaps() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        assert!(light_overlaps_cluster(
            volume(0, [0.5, 0.5, 0.5], 0.1),
            cluster
        ));
    }

    #[test]
    fn distant_light_does_not_overlap() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        // Center 10 units away on X, radius 1 -> no overlap.
        assert!(!light_overlaps_cluster(
            volume(0, [11.0, 0.5, 0.5], 1.0),
            cluster
        ));
    }

    #[test]
    fn reach_just_touches_face() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        // Center at x=3, box max x=1 -> gap 2; radius 2 just reaches.
        assert!(light_overlaps_cluster(
            volume(0, [3.0, 0.5, 0.5], 2.0),
            cluster
        ));
        // Radius 1.9 falls short.
        assert!(!light_overlaps_cluster(
            volume(0, [3.0, 0.5, 0.5], 1.9),
            cluster
        ));
    }

    #[test]
    fn assignment_builds_csr_slices() {
        let clusters = [
            unit_cluster([0.0, 0.0, 0.0]),
            unit_cluster([10.0, 0.0, 0.0]),
        ];
        let volumes = [
            volume(0, [0.5, 0.5, 0.5], 0.5),  // only near cluster 0
            volume(1, [10.5, 0.5, 0.5], 0.5), // only near cluster 1
            volume(2, [5.0, 0.5, 0.5], 6.0),  // large -> both clusters
        ];
        let a = assign_cluster_lights(&clusters, &volumes, 8);
        assert_eq!(a.cluster_count(), 2);
        assert_eq!(a.cluster_lights(0), &[0, 2]);
        assert_eq!(a.cluster_lights(1), &[1, 2]);
        assert_eq!(a.overflow, 0);
    }

    #[test]
    fn cap_records_overflow() {
        let clusters = [unit_cluster([0.0, 0.0, 0.0])];
        let volumes = [
            volume(0, [0.5, 0.5, 0.5], 1.0),
            volume(1, [0.5, 0.5, 0.5], 1.0),
            volume(2, [0.5, 0.5, 0.5], 1.0),
        ];
        let a = assign_cluster_lights(&clusters, &volumes, 2);
        assert_eq!(a.cluster_lights(0), &[0, 1]);
        assert_eq!(a.counts[0], 2);
        assert_eq!(a.overflow, 1);
    }

    #[test]
    fn out_of_range_cluster_is_empty() {
        let a = assign_cluster_lights(&[], &[], 4);
        assert_eq!(a.cluster_count(), 0);
        assert_eq!(a.cluster_lights(5), &[] as &[u32]);
    }

    #[test]
    fn preserves_light_input_order() {
        let clusters = [unit_cluster([0.0, 0.0, 0.0])];
        let volumes = [
            volume(7, [0.5, 0.5, 0.5], 1.0),
            volume(3, [0.5, 0.5, 0.5], 1.0),
            volume(5, [0.5, 0.5, 0.5], 1.0),
        ];
        let a = assign_cluster_lights(&clusters, &volumes, 8);
        // Indices reference the input slice positions in order, not handles.
        assert_eq!(a.cluster_lights(0), &[0, 1, 2]);
    }
}
