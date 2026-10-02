//! Deterministic leader clustering: assigns voices to a capped number of
//! representative clusters by spatial proximity and timbre similarity, and
//! merges clusters down to an object-bed budget.
//!
//! The assignment runs once per block boundary over the current member set, so
//! it is an incremental recompute in the scheduling sense (cheap, off the DSP
//! hot path) while remaining a pure function of its inputs. The policy is a
//! single deterministic pass:
//!
//! 1. Members are visited loudest-first (ties broken by voice id) so the most
//!    important voices seed clusters.
//! 2. A member joins the spatially nearest existing cluster whose centroid is
//!    within [`ClusterConfig::spatial_radius`] and whose timbre is within
//!    [`ClusterConfig::timbre_threshold`]; the centroid updates immediately so
//!    later members see the growing cluster.
//! 3. If no cluster qualifies and the cap [`ClusterConfig::max_clusters`] is not
//!    yet reached, the member seeds a new cluster; otherwise it is forced into
//!    the spatially nearest existing cluster so the cap always holds.
//!
//! [`merge_to_capacity`] further reduces an existing cluster set to a hard cap
//! by repeatedly fusing the two closest clusters, which the object-bed budget
//! uses to collapse clusters into a handful of bed channels.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements the clustering policy of design section 33 on top of the
//! [`crate::clustering::cluster`] model and [`crate::clustering::timbre`]
//! metric.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;

use crate::clustering::cluster::{Cluster, ClusterAccumulator, ClusterMember};
use crate::clustering::timbre::Timbre;

/// Tuning for [`assign`]: the cluster cap and the membership thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterConfig {
    /// Maximum number of clusters (the representative-source budget). At least
    /// one cluster is always allowed.
    pub max_clusters: usize,
    /// Maximum distance, in world units, from a cluster centroid for a member
    /// to join it on proximity grounds.
    pub spatial_radius: Sample,
    /// Maximum [`Timbre::distance`] for a member to join a cluster on timbre
    /// grounds.
    pub timbre_threshold: Sample,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { max_clusters: 8, spatial_radius: 5.0, timbre_threshold: 0.35 }
    }
}

impl ClusterConfig {
    /// Returns a copy with `max_clusters` set to at least one.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self { max_clusters: self.max_clusters.max(1), ..self }
    }
}

/// The result of an assignment pass.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterSet {
    /// The representative clusters, ordered by creation (loudest seeds first).
    pub clusters: Vec<Cluster>,
    /// For each input member, in input order, the index of its cluster in
    /// [`ClusterSet::clusters`].
    pub of_member: Vec<usize>,
}

impl ClusterSet {
    /// Number of clusters produced.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Returns `true` when no clusters were produced (empty input).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

/// Energy-weighted merge of two centroids, where each centroid is already an
/// energy-weighted mean of its own members.
fn merge_centroid(a: Vec3, ea: Sample, b: Vec3, eb: Sample) -> Vec3 {
    let total = ea + eb;
    if total > 0.0 {
        (a * ea + b * eb) / total
    } else {
        (a + b) * 0.5
    }
}

/// Energy-weighted merge of two timbres.
fn merge_timbre(a: Timbre, ea: Sample, b: Timbre, eb: Sample) -> Timbre {
    let total = ea + eb;
    if total <= 0.0 {
        return Timbre {
            brightness: (a.brightness + b.brightness) * 0.5,
            width: (a.width + b.width) * 0.5,
            flatness: (a.flatness + b.flatness) * 0.5,
        };
    }
    Timbre {
        brightness: ((a.brightness * ea + b.brightness * eb) / total).clamp(0.0, 1.0),
        width: ((a.width * ea + b.width * eb) / total).clamp(0.0, 1.0),
        flatness: ((a.flatness * ea + b.flatness * eb) / total).clamp(0.0, 1.0),
    }
}

/// Fuses cluster `b` into cluster `a`, conserving energy and preserving member
/// order (`a`'s members first, then `b`'s).
fn fuse(a: &Cluster, b: &Cluster) -> Cluster {
    let centroid = merge_centroid(a.centroid, a.total_energy, b.centroid, b.total_energy);
    let timbre = merge_timbre(a.timbre, a.total_energy, b.timbre, b.total_energy);
    let mut members = a.members.clone();
    members.extend_from_slice(&b.members);
    Cluster { centroid, total_energy: a.total_energy + b.total_energy, timbre, members }
}

/// Assigns `members` to at most `config.max_clusters` clusters by proximity and
/// timbre similarity.
///
/// The result is a deterministic function of the member set and configuration:
/// the same inputs always yield the same clusters and the same per-member
/// mapping, which makes clustering golden-testable.
#[must_use]
pub fn assign(members: &[ClusterMember], config: &ClusterConfig) -> ClusterSet {
    let config = config.sanitised();
    if members.is_empty() {
        return ClusterSet { clusters: Vec::new(), of_member: Vec::new() };
    }

    // Deterministic visit order: loudest first, ties broken by voice id, then
    // by input index so equal voices keep a stable order.
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_by(|&i, &j| {
        let a = &members[i];
        let b = &members[j];
        b.energy
            .partial_cmp(&a.energy)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.voice.cmp(&b.voice))
            .then(i.cmp(&j))
    });

    let mut accumulators: Vec<ClusterAccumulator> = Vec::new();
    // Cluster index per input member (filled out of order, hence pre-sized).
    let mut of_member = alloc_filled(members.len(), 0usize);

    for &idx in &order {
        let member = &members[idx];

        // Find the spatially nearest cluster that also passes the timbre gate.
        let mut best_gated: Option<usize> = None;
        let mut best_gated_dist = Sample::INFINITY;
        // Track the nearest cluster overall for the forced-merge fallback.
        let mut best_any: Option<usize> = None;
        let mut best_any_dist = Sample::INFINITY;

        for (c, acc) in accumulators.iter().enumerate() {
            let centroid = acc.centroid();
            let spatial = (member.position - centroid).length();
            if spatial < best_any_dist {
                best_any_dist = spatial;
                best_any = Some(c);
            }
            let timbre_dist = member.timbre.distance(&acc.timbre());
            if spatial <= config.spatial_radius
                && timbre_dist <= config.timbre_threshold
                && spatial < best_gated_dist
            {
                best_gated_dist = spatial;
                best_gated = Some(c);
            }
        }

        let target = if let Some(c) = best_gated {
            c
        } else if accumulators.len() < config.max_clusters {
            accumulators.push(ClusterAccumulator::new());
            accumulators.len() - 1
        } else {
            // Capacity reached and nothing qualified: force into the nearest.
            best_any.unwrap_or(0)
        };

        accumulators[target].push(member);
        of_member[idx] = target;
    }

    let clusters: Vec<Cluster> = accumulators.into_iter().map(ClusterAccumulator::finish).collect();
    ClusterSet { clusters, of_member }
}

/// Reduces `clusters` to at most `max_clusters` by repeatedly fusing the two
/// clusters whose centroids are closest, conserving energy at each step.
///
/// Used to collapse a representative-source set into the smaller object-bed
/// budget. The pairing is deterministic (lowest index pair wins ties). Returns
/// the input unchanged when it already fits.
#[must_use]
pub fn merge_to_capacity(mut clusters: Vec<Cluster>, max_clusters: usize) -> Vec<Cluster> {
    let cap = max_clusters.max(1);
    while clusters.len() > cap {
        let mut best = (0usize, 1usize);
        let mut best_dist = Sample::INFINITY;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d = (clusters[i].centroid - clusters[j].centroid).length();
                if d < best_dist {
                    best_dist = d;
                    best = (i, j);
                }
            }
        }
        let (i, j) = best;
        let fused = fuse(&clusters[i], &clusters[j]);
        // Remove the higher index first so the lower index stays valid.
        clusters.remove(j);
        clusters[i] = fused;
    }
    clusters
}

/// Allocates a vector of `len` copies of `value` without relying on the `vec!`
/// macro import in `no_std` builds.
fn alloc_filled(len: usize, value: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(value);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    const EPS: Sample = 1.0e-4;

    fn member(voice: usize, pos: Vec3, energy: Sample) -> ClusterMember {
        ClusterMember::new(voice, pos, energy, Timbre::neutral())
    }

    fn member_t(voice: usize, pos: Vec3, energy: Sample, t: Timbre) -> ClusterMember {
        ClusterMember::new(voice, pos, energy, t)
    }

    #[test]
    fn empty_input_yields_no_clusters() {
        let set = assign(&[], &ClusterConfig::default());
        assert!(set.is_empty());
        assert_eq!(set.of_member.len(), 0);
    }

    #[test]
    fn nearby_similar_voices_cluster_together() {
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 2.0, timbre_threshold: 0.5 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(0.5, 0.0, 0.0), 1.0),
            member(2, Vec3::new(1.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        assert_eq!(set.len(), 1);
        assert_eq!(set.clusters[0].len(), 3);
    }

    #[test]
    fn distant_voices_form_separate_clusters() {
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 2.0, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(100.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn timbre_gate_separates_dissimilar_voices() {
        let bright = Timbre { brightness: 1.0, width: 0.0, flatness: 0.0 };
        let dark = Timbre { brightness: 0.0, width: 0.0, flatness: 0.0 };
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 100.0, timbre_threshold: 0.2 };
        let members = [
            member_t(0, Vec3::new(0.0, 0.0, 0.0), 1.0, bright),
            member_t(1, Vec3::new(0.1, 0.0, 0.0), 1.0, dark),
        ];
        let set = assign(&members, &cfg);
        // Spatially adjacent but timbrally far apart -> two clusters.
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn capacity_is_never_exceeded() {
        let cfg = ClusterConfig { max_clusters: 2, spatial_radius: 1.0, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 4.0),
            member(1, Vec3::new(50.0, 0.0, 0.0), 3.0),
            member(2, Vec3::new(100.0, 0.0, 0.0), 2.0),
            member(3, Vec3::new(150.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        assert!(set.len() <= 2);
        // Every member is mapped to a valid cluster.
        for &c in &set.of_member {
            assert!(c < set.len());
        }
    }

    #[test]
    fn louder_voice_seeds_first_cluster() {
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 1.0, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(50.0, 0.0, 0.0), 9.0),
        ];
        let set = assign(&members, &cfg);
        // The louder voice (index 1) seeds cluster 0.
        assert_eq!(set.of_member[1], 0);
        assert_eq!(set.of_member[0], 1);
    }

    #[test]
    fn assignment_is_deterministic() {
        let cfg = ClusterConfig::default();
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(0.5, 0.0, 0.0), 2.0),
            member(2, Vec3::new(20.0, 0.0, 0.0), 1.5),
        ];
        let a = assign(&members, &cfg);
        let b = assign(&members, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn centroid_is_energy_weighted_after_assignment() {
        let cfg = ClusterConfig { max_clusters: 1, spatial_radius: 100.0, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(8.0, 0.0, 0.0), 3.0),
        ];
        let set = assign(&members, &cfg);
        assert_eq!(set.len(), 1);
        assert!((set.clusters[0].centroid.x - 6.0).abs() < EPS);
    }

    #[test]
    fn merge_to_capacity_reduces_count() {
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 0.1, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(1.0, 0.0, 0.0), 1.0),
            member(2, Vec3::new(2.0, 0.0, 0.0), 1.0),
            member(3, Vec3::new(3.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        assert_eq!(set.len(), 4);
        let merged = merge_to_capacity(set.clusters, 2);
        assert_eq!(merged.len(), 2);
        // Energy is conserved across the merge: total stays 4.0.
        let total: Sample = merged.iter().map(|c| c.total_energy).sum();
        assert!((total - 4.0).abs() < EPS);
    }

    #[test]
    fn merge_to_capacity_keeps_small_sets() {
        let cfg = ClusterConfig::default();
        let members = [member(0, Vec3::ZERO, 1.0)];
        let set = assign(&members, &cfg);
        let merged = merge_to_capacity(set.clusters, 4);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merged_members_are_preserved() {
        let cfg = ClusterConfig { max_clusters: 8, spatial_radius: 0.1, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 2.0),
            member(5, Vec3::new(10.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        let merged = merge_to_capacity(set.clusters, 1);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].len(), 2);
        let total: Sample = merged.iter().map(|c| c.total_energy).sum();
        assert!((total - 3.0).abs() < EPS);
    }

    #[test]
    fn zero_capacity_is_treated_as_one() {
        let cfg = ClusterConfig { max_clusters: 0, spatial_radius: 1.0, timbre_threshold: 1.0 };
        let members = [
            member(0, Vec3::new(0.0, 0.0, 0.0), 1.0),
            member(1, Vec3::new(50.0, 0.0, 0.0), 1.0),
        ];
        let set = assign(&members, &cfg);
        assert_eq!(set.len(), 1);
    }
}
