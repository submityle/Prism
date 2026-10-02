//! Cluster model: a representative virtual source standing in for several real
//! voices, with a loudness-weighted centroid and an energy-conserving downmix.
//!
//! When many quiet, nearby, similar voices would each cost a render slot, the
//! clustering stage fuses them into one [`Cluster`] whose single rendered
//! source preserves the group's perceived location and loudness:
//!
//! - **Centroid** -- the energy-weighted mean of member positions, so the
//!   loudest members pull the representative toward them (design section 33).
//! - **Total energy** -- the sum of member energies; the representative is
//!   rendered at [`Cluster::representative_amplitude`] = `sqrt(total_energy)`,
//!   which conserves acoustic energy rather than summing amplitudes (that would
//!   overshoot) or averaging them (that would lose level).
//! - **Timbre** -- the energy-weighted mean of member fingerprints, used when
//!   deciding whether further clusters may merge.
//!
//! A [`ClusterAccumulator`] builds a cluster incrementally so the assignment
//! stage can grow clusters in one deterministic pass without re-reading
//! members, and so block-boundary recomputes stay cheap.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Core data model for the source-clustering half of design section 33. Member
//! timbres come from [`crate::clustering::timbre`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;
use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::clustering::timbre::Timbre;

/// One real voice considered for clustering at a block boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterMember {
    /// Caller-defined voice identifier (for example a voice-pool slot index).
    pub voice: usize,
    /// World-space position of the voice.
    pub position: Vec3,
    /// Linear acoustic energy of the voice (non-negative; sanitised on input).
    pub energy: Sample,
    /// Spectral fingerprint of the voice.
    pub timbre: Timbre,
}

impl ClusterMember {
    /// Creates a member, clamping non-finite or negative energy to zero.
    #[must_use]
    pub fn new(voice: usize, position: Vec3, energy: Sample, timbre: Timbre) -> Self {
        let energy = if energy.is_finite() && energy > 0.0 { energy } else { 0.0 };
        let position = if position.is_finite() { position } else { Vec3::ZERO };
        Self { voice, position, energy, timbre }
    }
}

/// A representative virtual source produced by fusing one or more members.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Cluster {
    /// Energy-weighted mean position of the members.
    pub centroid: Vec3,
    /// Sum of member energies (the conserved quantity of the downmix).
    pub total_energy: Sample,
    /// Energy-weighted mean timbre of the members.
    pub timbre: Timbre,
    /// Voice identifiers of the members, in insertion order.
    pub members: Vec<usize>,
}

impl Cluster {
    /// Returns the representative render amplitude, `sqrt(total_energy)`, which
    /// conserves acoustic energy across the downmix.
    #[must_use]
    pub fn representative_amplitude(&self) -> Sample {
        ops::sqrt(self.total_energy.max(0.0))
    }

    /// Number of member voices in the cluster.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` when the cluster has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Incremental builder that accumulates member contributions into a
/// [`Cluster`].
///
/// Positions and timbres are accumulated as energy-weighted sums and normalised
/// once by [`ClusterAccumulator::finish`]. A cluster seeded only with
/// zero-energy members falls back to the arithmetic mean position so it still
/// has a sensible, finite centroid.
#[derive(Debug, Clone)]
pub struct ClusterAccumulator {
    weighted_pos: Vec3,
    plain_pos: Vec3,
    energy: Sample,
    weighted_brightness: Sample,
    weighted_width: Sample,
    weighted_flatness: Sample,
    members: Vec<usize>,
}

impl Default for ClusterAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterAccumulator {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            weighted_pos: Vec3::ZERO,
            plain_pos: Vec3::ZERO,
            energy: 0.0,
            weighted_brightness: 0.0,
            weighted_width: 0.0,
            weighted_flatness: 0.0,
            members: Vec::new(),
        }
    }

    /// Folds one member into the accumulator.
    pub fn push(&mut self, member: &ClusterMember) {
        let e = if member.energy.is_finite() && member.energy > 0.0 { member.energy } else { 0.0 };
        self.weighted_pos += member.position * e;
        self.plain_pos += member.position;
        self.energy += e;
        self.weighted_brightness += e * member.timbre.brightness;
        self.weighted_width += e * member.timbre.width;
        self.weighted_flatness += e * member.timbre.flatness;
        self.members.push(member.voice);
    }

    /// Returns the running energy total, used by the assignment stage to score
    /// candidate members before committing them.
    #[must_use]
    pub fn energy(&self) -> Sample {
        self.energy
    }

    /// Returns the current energy-weighted centroid without consuming the
    /// accumulator (falling back to the arithmetic mean for zero energy).
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let n = self.members.len();
        if self.energy > 0.0 {
            self.weighted_pos / self.energy
        } else if n > 0 {
            self.plain_pos / n as Sample
        } else {
            Vec3::ZERO
        }
    }

    /// Returns the current energy-weighted timbre.
    #[must_use]
    pub fn timbre(&self) -> Timbre {
        if self.energy > 0.0 {
            Timbre {
                brightness: (self.weighted_brightness / self.energy).clamp(0.0, 1.0),
                width: (self.weighted_width / self.energy).clamp(0.0, 1.0),
                flatness: (self.weighted_flatness / self.energy).clamp(0.0, 1.0),
            }
        } else {
            Timbre::neutral()
        }
    }

    /// Number of members accumulated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` when nothing has been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Consumes the accumulator and produces the finished cluster.
    #[must_use]
    pub fn finish(self) -> Cluster {
        let centroid = self.centroid();
        let timbre = self.timbre();
        Cluster { centroid, total_energy: self.energy, timbre, members: self.members }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-4;

    fn member(voice: usize, pos: Vec3, energy: Sample) -> ClusterMember {
        ClusterMember::new(voice, pos, energy, Timbre::neutral())
    }

    #[test]
    fn single_member_centroid_is_its_position() {
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(0, Vec3::new(1.0, 2.0, 3.0), 4.0));
        let c = acc.finish();
        assert!((c.centroid - Vec3::new(1.0, 2.0, 3.0)).length() < EPS);
        assert!((c.total_energy - 4.0).abs() < EPS);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn centroid_is_energy_weighted() {
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(0, Vec3::new(0.0, 0.0, 0.0), 1.0));
        acc.push(&member(1, Vec3::new(10.0, 0.0, 0.0), 3.0));
        let c = acc.finish();
        // Weighted toward the louder member at x=10: (0*1 + 10*3)/4 = 7.5.
        assert!((c.centroid.x - 7.5).abs() < EPS);
    }

    #[test]
    fn zero_energy_falls_back_to_arithmetic_mean() {
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(0, Vec3::new(0.0, 0.0, 0.0), 0.0));
        acc.push(&member(1, Vec3::new(4.0, 0.0, 0.0), 0.0));
        let c = acc.finish();
        assert!((c.centroid.x - 2.0).abs() < EPS);
        assert!(c.total_energy.abs() < EPS);
    }

    #[test]
    fn total_energy_is_conserved() {
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(0, Vec3::ZERO, 2.0));
        acc.push(&member(1, Vec3::ZERO, 5.0));
        acc.push(&member(2, Vec3::ZERO, 1.0));
        let c = acc.finish();
        assert!((c.total_energy - 8.0).abs() < EPS);
        // Representative amplitude is sqrt of summed energy, not sum of roots.
        assert!((c.representative_amplitude() - ops::sqrt(8.0)).abs() < EPS);
    }

    #[test]
    fn amplitude_conserves_energy_not_amplitude() {
        // Two equal-energy members: amplitude sum would be 2*sqrt(e); energy
        // conservation gives sqrt(2*e), which is strictly smaller.
        let e = 3.0;
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(0, Vec3::ZERO, e));
        acc.push(&member(1, Vec3::ZERO, e));
        let c = acc.finish();
        let energy_conserving = ops::sqrt(2.0 * e);
        let amplitude_sum = 2.0 * ops::sqrt(e);
        assert!((c.representative_amplitude() - energy_conserving).abs() < EPS);
        assert!(c.representative_amplitude() < amplitude_sum);
    }

    #[test]
    fn negative_energy_is_clamped() {
        let m = ClusterMember::new(0, Vec3::ZERO, -5.0, Timbre::neutral());
        assert!(m.energy.abs() < EPS);
    }

    #[test]
    fn members_preserve_insertion_order() {
        let mut acc = ClusterAccumulator::new();
        acc.push(&member(7, Vec3::ZERO, 1.0));
        acc.push(&member(3, Vec3::ZERO, 1.0));
        acc.push(&member(9, Vec3::ZERO, 1.0));
        let c = acc.finish();
        assert_eq!(c.members, [7, 3, 9]);
    }

    #[test]
    fn weighted_timbre_blends_members() {
        let bright = Timbre { brightness: 1.0, width: 0.0, flatness: 0.0 };
        let dark = Timbre { brightness: 0.0, width: 0.0, flatness: 0.0 };
        let mut acc = ClusterAccumulator::new();
        acc.push(&ClusterMember::new(0, Vec3::ZERO, 1.0, bright));
        acc.push(&ClusterMember::new(1, Vec3::ZERO, 1.0, dark));
        let c = acc.finish();
        assert!((c.timbre.brightness - 0.5).abs() < EPS);
    }

    #[test]
    fn empty_accumulator_is_empty() {
        let acc = ClusterAccumulator::new();
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);
        let c = acc.finish();
        assert!(c.is_empty());
    }

    #[test]
    fn running_energy_tracks_pushes() {
        let mut acc = ClusterAccumulator::new();
        assert!(acc.energy().abs() < EPS);
        acc.push(&member(0, Vec3::ZERO, 2.5));
        assert!((acc.energy() - 2.5).abs() < EPS);
    }
}
