//! Energy-preserving object clustering fallback.
//!
//! When a scene presents more objects than the hardware
//! [`crate::budget::ObjectBudget`] can render, the surplus must be reduced to a
//! handful of representative objects without losing acoustic energy or badly
//! distorting direction. This module performs *agglomerative* clustering down
//! to a target count: it repeatedly merges the pair of clusters that is
//! cheapest to combine, where the cost rises with the clusters' (priority-
//! weighted) energy and with their angular separation. Loud, high-priority,
//! or spatially isolated objects are therefore merged last and tend to survive
//! as discrete clusters, while quiet neighbours collapse first.
//!
//! # Energy conservation
//!
//! A cluster's energy is the sum of its members' `gain^2`, and its reported
//! `gain` is `sqrt(energy)`. Because merging only adds member energies, the
//! total energy of the output clusters equals the total energy of the input
//! objects (up to floating-point rounding). The representative direction and
//! position are the energy-weighted means of the members, so the cluster sits
//! at the perceptual centroid of the energy it carries.
//!
//! # Determinism
//!
//! All length/normalisation math routes through [`bevy_math::ops`]; pair
//! selection breaks ties by index, so the reduction is bit-reproducible and
//! order-stable across targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the object-budget clustering fallback of design section 44.2
//! (and the source-clustering idea of section 33). Driven by
//! [`crate::budget`] and consumed by [`crate::render`] and [`crate::fold`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;

use crate::object::AudioObject;

/// A representative object produced by clustering (possibly a passthrough of a
/// single input object, or a merge of several).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusteredObject {
    /// Unit representative direction (energy-weighted mean of members).
    pub direction: Vec3,
    /// Energy-weighted mean member position (metres).
    pub position: Vec3,
    /// Linear amplitude gain, `sqrt(energy)`.
    pub gain: Sample,
    /// Total acoustic energy carried by the cluster (`sum of gain^2`).
    pub energy: Sample,
    /// Highest member priority.
    pub priority: Sample,
    /// Energy-weighted mean member spread in `[0, 1]`.
    pub spread: Sample,
    /// Number of input objects merged into this cluster.
    pub member_count: usize,
    /// Whether this cluster is a single snapped object (merged clusters never
    /// snap).
    pub snap: bool,
}

/// Mutable accumulator used while merging; converted to [`ClusteredObject`] at
/// the end.
#[derive(Debug, Clone, Copy)]
struct ClusterAcc {
    energy: Sample,
    eff_energy: Sample,
    wdir: Vec3,
    wpos: Vec3,
    dir_sum: Vec3,
    pos_sum: Vec3,
    spread_wsum: Sample,
    spread_sum: Sample,
    priority: Sample,
    count: usize,
    snap: bool,
}

impl ClusterAcc {
    fn from_object(object: &AudioObject) -> Self {
        let energy = object.energy();
        let dir = object.direction().unwrap_or(Vec3::NEG_Z);
        let eff = energy * (1.0 + object.priority.max(0.0));
        Self {
            energy,
            eff_energy: eff,
            wdir: dir * energy,
            wpos: object.position * energy,
            dir_sum: dir,
            pos_sum: object.position,
            spread_wsum: object.spread * energy,
            spread_sum: object.spread,
            priority: object.priority,
            count: 1,
            snap: object.snap,
        }
    }

    fn merge(&mut self, other: &ClusterAcc) {
        self.energy += other.energy;
        self.eff_energy += other.eff_energy;
        self.wdir += other.wdir;
        self.wpos += other.wpos;
        self.dir_sum += other.dir_sum;
        self.pos_sum += other.pos_sum;
        self.spread_wsum += other.spread_wsum;
        self.spread_sum += other.spread_sum;
        self.priority = self.priority.max(other.priority);
        self.count += other.count;
        self.snap = false;
    }

    /// Unit representative direction, falling back gracefully when the energy
    /// weighting cancels or vanishes.
    fn direction(&self) -> Vec3 {
        if self.energy > 1e-12 {
            let d = self.wdir.normalize_or_zero();
            if d != Vec3::ZERO {
                return d;
            }
        }
        let d = self.dir_sum.normalize_or_zero();
        if d != Vec3::ZERO {
            d
        } else {
            Vec3::NEG_Z
        }
    }

    fn position(&self) -> Vec3 {
        if self.energy > 1e-12 {
            self.wpos / self.energy
        } else if self.count > 0 {
            self.pos_sum / (self.count as Sample)
        } else {
            Vec3::ZERO
        }
    }

    fn spread(&self) -> Sample {
        let s = if self.energy > 1e-12 {
            self.spread_wsum / self.energy
        } else if self.count > 0 {
            self.spread_sum / (self.count as Sample)
        } else {
            0.0
        };
        s.clamp(0.0, 1.0)
    }

    fn finish(&self) -> ClusteredObject {
        ClusteredObject {
            direction: self.direction(),
            position: self.position(),
            gain: ops::sqrt(self.energy.max(0.0)),
            energy: self.energy,
            priority: self.priority,
            spread: self.spread(),
            member_count: self.count,
            snap: self.snap && self.count == 1,
        }
    }
}

/// Angular separation cost in `[0, 2]`: `1 - cos(angle)` between two unit
/// directions.
fn angular_distance(a: Vec3, b: Vec3) -> Sample {
    (1.0 - a.dot(b)).clamp(0.0, 2.0)
}

/// Clusters `objects` down to at most `target` representative objects.
///
/// * `target == 0` returns an empty list (the caller is expected to fold the
///   entire scene into the bed).
/// * When `objects.len() <= target`, each object passes through as its own
///   single-member cluster (preserving its `snap` flag).
/// * Otherwise objects are agglomeratively merged until exactly `target`
///   clusters remain.
///
/// Objects with a non-finite gain or position are skipped.
#[must_use]
pub fn cluster_objects(objects: &[AudioObject], target: usize) -> Vec<ClusteredObject> {
    if target == 0 {
        return Vec::new();
    }

    let mut accs: Vec<ClusterAcc> = objects
        .iter()
        .filter(|o| o.gain.is_finite() && o.position.is_finite())
        .map(ClusterAcc::from_object)
        .collect();

    if accs.len() <= target {
        return accs.iter().map(ClusterAcc::finish).collect();
    }

    // Agglomerative merging: collapse the cheapest pair until `target` remain.
    while accs.len() > target {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_cost = Sample::INFINITY;
        for i in 0..accs.len() {
            for j in (i + 1)..accs.len() {
                let ang = angular_distance(accs[i].direction(), accs[j].direction());
                // A tiny floor keeps the ordering well-defined even when both
                // clusters carry no energy, so coincident silent objects still
                // merge before distant ones.
                let weight = accs[i].eff_energy + accs[j].eff_energy + 1e-9;
                let cost = weight * ang;
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        let merged = accs[best_j];
        accs[best_i].merge(&merged);
        accs.remove(best_j);
    }

    accs.iter().map(ClusterAcc::finish).collect()
}

/// Returns the total energy of a cluster list (`sum of energy`).
#[must_use]
pub fn total_energy(clusters: &[ClusteredObject]) -> Sample {
    clusters.iter().map(|c| c.energy).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectId;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn obj(id: u32, pos: Vec3, gain: Sample, priority: Sample) -> AudioObject {
        let mut o = AudioObject::new(ObjectId(id), pos, gain);
        o.priority = priority;
        o
    }

    #[test]
    fn passthrough_when_within_target() {
        let objects = [
            obj(0, Vec3::new(0.0, 0.0, -1.0), 0.5, 1.0),
            obj(1, Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0),
        ];
        let out = cluster_objects(&objects, 4);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].member_count, 1);
    }

    #[test]
    fn energy_is_conserved_under_merge() {
        let objects = [
            obj(0, Vec3::new(0.0, 0.0, -1.0), 0.5, 1.0),
            obj(1, Vec3::new(0.1, 0.0, -1.0), 0.5, 1.0),
            obj(2, Vec3::new(1.0, 0.0, 0.0), 0.4, 1.0),
            obj(3, Vec3::new(1.0, 0.1, 0.0), 0.4, 1.0),
            obj(4, Vec3::new(0.0, 1.0, 0.0), 0.3, 1.0),
        ];
        let input: Sample = objects.iter().map(AudioObject::energy).sum();
        let out = cluster_objects(&objects, 2);
        assert_eq!(out.len(), 2);
        assert!(close(total_energy(&out), input));
    }

    #[test]
    fn representative_direction_is_energy_weighted() {
        // Two coincident-direction objects merge to that same direction.
        let objects = [
            obj(0, Vec3::new(0.0, 0.0, -1.0), 1.0, 1.0),
            obj(1, Vec3::new(0.0, 0.0, -2.0), 0.5, 1.0),
            obj(2, Vec3::new(1.0, 0.0, 0.0), 0.2, 1.0),
        ];
        let out = cluster_objects(&objects, 2);
        // The two forward objects should merge; find the forward cluster.
        let fwd = out
            .iter()
            .find(|c| c.direction.z < -0.9)
            .expect("a forward cluster");
        assert!(close(fwd.direction.length(), 1.0));
        assert!(fwd.member_count >= 2);
    }

    #[test]
    fn loud_isolated_object_survives_as_singleton() {
        let objects = [
            obj(0, Vec3::new(1.0, 0.0, 0.0), 1.0, 5.0), // loud, high priority
            obj(1, Vec3::new(0.0, 0.0, -1.0), 0.1, 1.0),
            obj(2, Vec3::new(0.05, 0.0, -1.0), 0.1, 1.0),
            obj(3, Vec3::new(-0.05, 0.0, -1.0), 0.1, 1.0),
        ];
        let out = cluster_objects(&objects, 2);
        assert_eq!(out.len(), 2);
        let loud = out
            .iter()
            .find(|c| c.direction.x > 0.9)
            .expect("the loud object survives");
        assert_eq!(loud.member_count, 1);
    }

    #[test]
    fn zero_target_returns_empty() {
        let objects = [obj(0, Vec3::new(0.0, 0.0, -1.0), 1.0, 1.0)];
        assert!(cluster_objects(&objects, 0).is_empty());
    }

    #[test]
    fn merged_cluster_does_not_snap() {
        let mut a = obj(0, Vec3::new(0.0, 0.0, -1.0), 0.5, 1.0);
        a.snap = true;
        let mut b = obj(1, Vec3::new(0.0, 0.0, -1.01), 0.5, 1.0);
        b.snap = true;
        let out = cluster_objects(&[a, b], 1);
        assert_eq!(out.len(), 1);
        assert!(!out[0].snap);
    }

    #[test]
    fn single_snapped_object_keeps_snap() {
        let mut a = obj(0, Vec3::new(0.0, 0.0, -1.0), 0.5, 1.0);
        a.snap = true;
        let out = cluster_objects(&[a], 4);
        assert!(out[0].snap);
    }
}
