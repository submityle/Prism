//! Far-field impact clustering to bound the contact voice count.
//!
//! A crowd of bodies rattling far from the listener produces dozens of
//! impulses the ear cannot resolve individually; rendering each as its own
//! voice wastes the budget set by design section 47.1. This module collapses
//! near-simultaneous impacts that are both *far from the listener* and *close
//! to each other* into a single representative "group impact" whose energy is
//! the sum of the members (via
//! [`prism_audio_procedural::contact::cluster_to_group`]). Impacts near the
//! listener always survive individually so foreground detail is preserved. The
//! grouping is a deterministic stable-order sweep.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the clustering half of the budget/merge/cluster stage of design
//! section 47.1, downstream of [`prism_audio_procedural::contact::merge_impacts`]
//! inside [`crate::translator::ContactAudioTranslator::drain`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;

use prism_audio_procedural::contact::{cluster_to_group, ImpactEvent};
use prism_audio_core::math::Sample;

/// Configuration for far-field impact clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterConfig {
    /// World-space listener position; distance from here decides "far".
    pub listener: Vec3,
    /// Radius used both as the far-from-listener threshold and the maximum
    /// member-to-seed distance within a cluster.
    pub cluster_radius: Sample,
    /// Minimum number of members before a group is collapsed to one impact.
    pub min_cluster_size: usize,
}

impl Default for ClusterConfig {
    #[inline]
    fn default() -> Self {
        Self {
            listener: Vec3::ZERO,
            // Beyond ~20 m the far crowd is already blurred; a generous default.
            cluster_radius: 20.0,
            // Only collapse once at least three strikes pile up.
            min_cluster_size: 3,
        }
    }
}

#[inline]
fn dist_sq(a: Vec3, b: Vec3) -> Sample {
    let d = a - b;
    d.dot(d)
}

/// Collapses far, mutually close impacts into representative group impacts.
///
/// `positions` is index-aligned with `impacts` (position `i` is the world
/// location of impact `i`). An impact is a clustering candidate only when it
/// lies beyond `cluster_radius` of the listener; candidates within
/// `cluster_radius` of a seed form a group, and a group reaching
/// `min_cluster_size` is replaced by one summed-energy impact. Near impacts and
/// under-sized groups pass through unchanged. The sweep walks indices in order
/// for a deterministic, replayable result. Returns the surviving impact count.
///
/// When `positions` does not match `impacts` in length the input is left
/// untouched and the current length is returned.
pub fn cluster_far_impacts(
    impacts: &mut Vec<ImpactEvent>,
    positions: &[Vec3],
    config: &ClusterConfig,
) -> usize {
    if impacts.len() != positions.len() || impacts.len() <= 1 {
        return impacts.len();
    }
    let radius_sq = config.cluster_radius * config.cluster_radius;
    let min_size = config.min_cluster_size.max(1);

    let mut assigned = alloc::vec![false; impacts.len()];
    let mut output: Vec<ImpactEvent> = Vec::with_capacity(impacts.len());

    for seed in 0..impacts.len() {
        if assigned[seed] {
            continue;
        }
        let far = dist_sq(positions[seed], config.listener) > radius_sq;
        if !far {
            // Near the listener: keep individually, do not cluster.
            assigned[seed] = true;
            output.push(impacts[seed]);
            continue;
        }
        // Gather far, unassigned impacts within the radius of this seed.
        let mut members: Vec<usize> = Vec::new();
        members.push(seed);
        for other in (seed + 1)..impacts.len() {
            if assigned[other] {
                continue;
            }
            let other_far = dist_sq(positions[other], config.listener) > radius_sq;
            if other_far && dist_sq(positions[other], positions[seed]) <= radius_sq {
                members.push(other);
            }
        }
        if members.len() >= min_size {
            // Collapse the whole group into one representative impact.
            let group: Vec<ImpactEvent> = members.iter().map(|&i| impacts[i]).collect();
            if let Some(collapsed) = cluster_to_group(&group) {
                output.push(collapsed);
            }
            for i in members {
                assigned[i] = true;
            }
        } else {
            // Too small to collapse: keep the seed as itself.
            assigned[seed] = true;
            output.push(impacts[seed]);
        }
    }

    let count = output.len();
    *impacts = output;
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_procedural::contact::{ContactId, ContactPoint};
    use prism_audio_procedural::material::MaterialPairId;

    fn impact(contact: u64, impulse: Sample, offset: u32) -> ImpactEvent {
        ImpactEvent::new(
            ContactId(contact),
            MaterialPairId::new(0, 0),
            impulse,
            impulse,
            0.0,
            ContactPoint::new(0.5),
            offset,
            512,
        )
    }

    #[test]
    fn far_tight_cluster_collapses() {
        let mut impacts = alloc::vec![
            impact(1, 1.0, 10),
            impact(2, 2.0, 20),
            impact(3, 1.0, 30),
        ];
        let positions = alloc::vec![
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(101.0, 0.0, 0.0),
            Vec3::new(102.0, 0.0, 0.0),
        ];
        let config = ClusterConfig {
            listener: Vec3::ZERO,
            cluster_radius: 20.0,
            min_cluster_size: 3,
        };
        let n = cluster_far_impacts(&mut impacts, &positions, &config);
        assert_eq!(n, 1);
        assert_eq!(impacts.len(), 1);
        // Energy preserved: 1 + 2 + 1 = 4.
        assert!((impacts[0].impulse - 4.0).abs() < 1e-6);
        // Earliest offset kept.
        assert_eq!(impacts[0].sample_offset, 10);
    }

    #[test]
    fn far_apart_impacts_survive() {
        let mut impacts = alloc::vec![impact(1, 1.0, 10), impact(2, 1.0, 20)];
        let positions = alloc::vec![
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(100.0, 500.0, 0.0),
        ];
        let config = ClusterConfig {
            listener: Vec3::ZERO,
            cluster_radius: 20.0,
            min_cluster_size: 2,
        };
        let n = cluster_far_impacts(&mut impacts, &positions, &config);
        assert_eq!(n, 2);
    }

    #[test]
    fn near_impacts_are_not_clustered() {
        let mut impacts = alloc::vec![
            impact(1, 1.0, 10),
            impact(2, 1.0, 20),
            impact(3, 1.0, 30),
        ];
        let positions = alloc::vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let config = ClusterConfig {
            listener: Vec3::ZERO,
            cluster_radius: 20.0,
            min_cluster_size: 2,
        };
        let n = cluster_far_impacts(&mut impacts, &positions, &config);
        assert_eq!(n, 3);
    }

    #[test]
    fn undersized_far_group_survives() {
        let mut impacts = alloc::vec![impact(1, 1.0, 10), impact(2, 1.0, 20)];
        let positions = alloc::vec![
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(101.0, 0.0, 0.0),
        ];
        let config = ClusterConfig {
            listener: Vec3::ZERO,
            cluster_radius: 20.0,
            min_cluster_size: 3,
        };
        let n = cluster_far_impacts(&mut impacts, &positions, &config);
        assert_eq!(n, 2);
    }

    #[test]
    fn mismatched_lengths_are_untouched() {
        let mut impacts = alloc::vec![impact(1, 1.0, 10)];
        let positions = alloc::vec![Vec3::ZERO, Vec3::ZERO];
        let n = cluster_far_impacts(&mut impacts, &positions, &ClusterConfig::default());
        assert_eq!(n, 1);
    }
}
