//! Object downmix: object-to-bed matrices and object-to-binaural parameters.
//!
//! "Folding" reduces positioned objects onto a concrete delivery target. Two
//! paths are provided, matching the design's two downmix routes:
//!
//! * **Object to bed.** Each source is panned onto the channel
//!   [`crate::bed::BedLayout`] with the VBAP panner of [`crate::pan`] and then
//!   scaled by its linear gain, yielding one gain row per source. Summing the
//!   rows (times each source's mono signal) builds the bed bus. Because the
//!   panner is constant power (its per-channel gains have unit sum-of-squares),
//!   a source of gain `g` contributes exactly `g^2` of energy to the bed, so
//!   the downmix is energy preserving per source.
//! * **Object to binaural.** Each source is reduced to a direction parameter
//!   set -- `azimuth`/`elevation` (degrees), linear `gain`, and `spread` -- for
//!   a downstream head-related renderer. No HRTF data is embedded here; this
//!   module only produces the bearing/level parameters that feed one.
//!
//! # Determinism
//!
//! Angle extraction routes through [`bevy_math::ops`] (`atan2`, `hypot`), so
//! results are bit-reproducible across targets. Degrees are produced by exact
//! multiplication with `180 / PI`.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the object downmix (object-to-bed and object-to-binaural) of
//! design section 44.2. Built on [`crate::pan`] and consumed by
//! [`crate::render`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::f32::consts::PI;

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;

use crate::bed::BedLayout;
use crate::clustering::ClusteredObject;
use crate::object::AudioObject;
use crate::pan;

/// Radians-to-degrees scale factor.
const RAD_TO_DEG: Sample = 180.0 / PI;

/// Direction parameters for a single binaural source.
///
/// `azimuth` and `elevation` are in degrees in the listener-local frame:
/// azimuth is positive to the right (`+X`) and zero straight ahead (`-Z`);
/// elevation is positive upward (`+Y`). `gain` is linear amplitude and
/// `spread` in `[0, 1]` is the apparent angular size forwarded to the renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BinauralDirection {
    /// Azimuth in degrees (positive to the right, zero ahead).
    pub azimuth: Sample,
    /// Elevation in degrees (positive upward).
    pub elevation: Sample,
    /// Linear amplitude gain.
    pub gain: Sample,
    /// Apparent angular size / divergence in `[0, 1]`.
    pub spread: Sample,
}

/// Converts a unit (or near-unit) `direction` plus `gain`/`spread` into a
/// [`BinauralDirection`]. A zero-length direction maps to straight ahead
/// (azimuth and elevation zero).
#[must_use]
pub fn binaural_from_direction(direction: Vec3, gain: Sample, spread: Sample) -> BinauralDirection {
    let spread = spread.clamp(0.0, 1.0);
    // A degenerate (zero-length) direction has no bearing; map it to straight
    // ahead rather than letting atan2 of signed zeros pick an arbitrary angle.
    if direction.dot(direction) <= 1e-12 {
        return BinauralDirection {
            azimuth: 0.0,
            elevation: 0.0,
            gain,
            spread,
        };
    }
    let azimuth = ops::atan2(direction.x, -direction.z) * RAD_TO_DEG;
    let horizontal = ops::hypot(direction.x, direction.z);
    let elevation = ops::atan2(direction.y, horizontal) * RAD_TO_DEG;
    BinauralDirection {
        azimuth,
        elevation,
        gain,
        spread,
    }
}

/// Folds a single [`AudioObject`] into a per-bed-channel gain row (panned,
/// then scaled by the object's linear gain).
#[must_use]
pub fn fold_object_to_bed(object: &AudioObject, bed: BedLayout) -> Vec<Sample> {
    let mut row = pan::pan_object_to_bed(object, bed);
    for g in row.iter_mut() {
        *g *= object.gain;
    }
    row
}

/// Folds a single [`ClusteredObject`] into a per-bed-channel gain row (panned
/// by its representative direction, then scaled by its linear gain).
#[must_use]
pub fn fold_cluster_to_bed(cluster: &ClusteredObject, bed: BedLayout) -> Vec<Sample> {
    let mut row = pan::pan_cluster_to_bed(cluster, bed);
    for g in row.iter_mut() {
        *g *= cluster.gain;
    }
    row
}

/// Folds every object into the bed, returning one gain row per object.
#[must_use]
pub fn fold_objects_to_bed(objects: &[AudioObject], bed: BedLayout) -> Vec<Vec<Sample>> {
    objects.iter().map(|o| fold_object_to_bed(o, bed)).collect()
}

/// Folds every cluster into the bed, returning one gain row per cluster.
#[must_use]
pub fn fold_clusters_to_bed(clusters: &[ClusteredObject], bed: BedLayout) -> Vec<Vec<Sample>> {
    clusters
        .iter()
        .map(|c| fold_cluster_to_bed(c, bed))
        .collect()
}

/// Sums a per-source gain matrix down to a single per-channel bed gain row.
///
/// This is the column-wise accumulation of [`fold_objects_to_bed`] /
/// [`fold_clusters_to_bed`]: entry `k` is the total gain delivered to bed
/// channel `k` across all sources. `channels` sets the output width (use
/// [`BedLayout::channel_count`]); rows shorter than `channels` contribute only
/// their present entries.
#[must_use]
pub fn sum_bed_matrix(rows: &[Vec<Sample>], channels: usize) -> Vec<Sample> {
    let mut out = Vec::new();
    out.resize(channels, 0.0);
    for row in rows {
        for (acc, &g) in out.iter_mut().zip(row.iter()) {
            *acc += g;
        }
    }
    out
}

/// Folds a single [`AudioObject`] into binaural direction parameters.
#[must_use]
pub fn fold_object_to_binaural(object: &AudioObject) -> BinauralDirection {
    let dir = object.direction().unwrap_or(Vec3::NEG_Z);
    binaural_from_direction(dir, object.gain, object.spread)
}

/// Folds a single [`ClusteredObject`] into binaural direction parameters.
#[must_use]
pub fn fold_cluster_to_binaural(cluster: &ClusteredObject) -> BinauralDirection {
    binaural_from_direction(cluster.direction, cluster.gain, cluster.spread)
}

/// Folds every object into binaural direction parameters, one per object.
#[must_use]
pub fn fold_objects_to_binaural(objects: &[AudioObject]) -> Vec<BinauralDirection> {
    objects.iter().map(fold_object_to_binaural).collect()
}

/// Folds every cluster into binaural direction parameters, one per cluster.
#[must_use]
pub fn fold_clusters_to_binaural(clusters: &[ClusteredObject]) -> Vec<BinauralDirection> {
    clusters.iter().map(fold_cluster_to_binaural).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::direction_from_angles;
    use crate::object::ObjectId;

    const EPS: Sample = 1e-3;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn sum_sq(g: &[Sample]) -> Sample {
        g.iter().map(|&x| x * x).sum()
    }

    #[test]
    fn bed_fold_preserves_source_energy() {
        // A point source panned (constant power) and scaled by gain deposits
        // exactly gain^2 energy into the bed row.
        let o = AudioObject::new(ObjectId(0), direction_from_angles(20.0, 10.0), 0.5);
        let row = fold_object_to_bed(&o, BedLayout::Surround7_1_4);
        assert_eq!(row.len(), BedLayout::Surround7_1_4.channel_count());
        assert!(close(sum_sq(&row), 0.25));
    }

    #[test]
    fn bed_matrix_sum_accumulates_channels() {
        let o0 = AudioObject::new(ObjectId(0), direction_from_angles(0.0, 0.0), 1.0);
        let o1 = AudioObject::new(ObjectId(1), direction_from_angles(0.0, 0.0), 1.0);
        let rows = fold_objects_to_bed(&[o0, o1], BedLayout::Surround7_1_4);
        assert_eq!(rows.len(), 2);
        let summed = sum_bed_matrix(&rows, BedLayout::Surround7_1_4.channel_count());
        // Both sources are at the center speaker (channel 2) => it doubles.
        assert!(close(summed[2], 2.0));
    }

    #[test]
    fn binaural_recovers_right_and_up() {
        // Due right: azimuth +90, elevation 0.
        let right = binaural_from_direction(direction_from_angles(90.0, 0.0), 1.0, 0.0);
        assert!(close(right.azimuth, 90.0));
        assert!(close(right.elevation, 0.0));

        // Overhead-ish: elevation +45.
        let up = binaural_from_direction(direction_from_angles(0.0, 45.0), 1.0, 0.0);
        assert!(close(up.elevation, 45.0));
        assert!(close(up.azimuth, 0.0));
    }

    #[test]
    fn binaural_forward_is_zero_bearing() {
        let fwd = binaural_from_direction(Vec3::ZERO, 0.3, 0.0);
        assert!(close(fwd.azimuth, 0.0));
        assert!(close(fwd.elevation, 0.0));
        assert!(close(fwd.gain, 0.3));
    }

    #[test]
    fn object_binaural_carries_gain_and_spread() {
        let mut o = AudioObject::new(ObjectId(2), direction_from_angles(-30.0, 0.0), 0.8);
        o.spread = 0.5;
        let b = fold_object_to_binaural(&o);
        assert!(close(b.azimuth, -30.0));
        assert!(close(b.gain, 0.8));
        assert!(close(b.spread, 0.5));
    }

    #[test]
    fn cluster_fold_scales_by_cluster_gain() {
        let cluster = ClusteredObject {
            direction: direction_from_angles(0.0, 0.0),
            position: Vec3::NEG_Z,
            gain: 0.5,
            energy: 0.25,
            priority: 1.0,
            spread: 0.0,
            member_count: 1,
            snap: false,
        };
        let row = fold_cluster_to_bed(&cluster, BedLayout::Surround7_1_4);
        assert!(close(sum_sq(&row), 0.25));
        let b = fold_cluster_to_binaural(&cluster);
        assert!(close(b.gain, 0.5));
        assert!(close(b.azimuth, 0.0));
    }

    #[test]
    fn fold_lists_match_input_lengths() {
        let objects = [
            AudioObject::new(ObjectId(0), Vec3::NEG_Z, 1.0),
            AudioObject::new(ObjectId(1), Vec3::X, 0.5),
        ];
        assert_eq!(fold_objects_to_binaural(&objects).len(), 2);
        assert_eq!(fold_objects_to_bed(&objects, BedLayout::Stereo).len(), 2);
    }
}
