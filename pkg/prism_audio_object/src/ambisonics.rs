//! Object/bed to `AmbiX` (ACN + SN3D) Ambisonic transport encoding.
//!
//! Scene-based immersive transport encodes each source (object, cluster, or
//! bed speaker) into a set of Ambisonic channels laid out in **ACN** order with
//! **SN3D** normalisation -- the openly specified *`AmbiX`* set. The actual
//! spherical-harmonic evaluation is **reused** from
//! [`prism_audio_spatial::hoa`] (`encode_hoa` / `decode_hoa`); this module adds
//! only the object-audio glue: applying a source's linear gain to the encoded
//! coefficients and assembling per-source and per-bed-channel encoding
//! matrices for the transport stage.
//!
//! Encoding a source produces a vector of per-channel gains; multiplying the
//! source's mono signal by those gains and summing across sources yields the
//! Ambisonic bus. These functions produce the *gains* (control rate), not the
//! per-sample mix.
//!
//! # Determinism
//!
//! The reused encoder routes all harmonic/length math through
//! [`bevy_math::ops`]; the gain scaling here is pure multiplication, so results
//! are bit-reproducible across targets.
//!
//! # Provenance
//! Original work; the Ambisonic harmonics are reused from
//! `prism_audio_spatial::hoa`. No Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, Dolby, or Google Resonance Audio source or derived code; no
//! AI/ML.
//!
//! # Relationship
//! Implements the HOA/`AmbiX` transport of design section 44.2 by composing the
//! encoder of `prism_audio_spatial` (section 16). Consumed by
//! [`crate::render`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;

use prism_audio_core::math::Sample;
use prism_audio_spatial::hoa;

use crate::bed::BedLayout;
use crate::clustering::ClusteredObject;
use crate::object::AudioObject;

/// Highest Ambisonic order supported (re-exported from the spatial encoder).
pub const MAX_ORDER: usize = hoa::MAX_HOA_ORDER;

/// Returns the number of `AmbiX` channels for `order`: `(order + 1)^2`.
#[must_use]
pub fn channel_count(order: usize) -> usize {
    hoa::hoa_channel_count(order.min(MAX_ORDER))
}

/// Encodes a `direction` scaled by linear `gain` into `AmbiX` coefficients,
/// writing them into `out` (ACN order, SN3D) and returning the channel count
/// written.
///
/// `order` is clamped to [`MAX_ORDER`]; at most `(order + 1)^2` channels are
/// produced, fewer if `out` is shorter.
pub fn encode_direction(direction: Vec3, gain: Sample, order: usize, out: &mut [Sample]) -> usize {
    let written = hoa::encode_hoa(direction, order, out);
    for v in out[..written].iter_mut() {
        *v *= gain;
    }
    written
}

/// Encodes a single `AmbiX` coefficient vector for `direction` and `gain`,
/// allocating a fresh vector sized to [`channel_count`].
#[must_use]
pub fn encode_direction_vec(direction: Vec3, gain: Sample, order: usize) -> Vec<Sample> {
    let order = order.min(MAX_ORDER);
    let mut out = Vec::new();
    out.resize(channel_count(order), 0.0);
    encode_direction(direction, gain, order, &mut out);
    out
}

/// Encodes an [`AudioObject`] (direction + gain) into an `AmbiX` coefficient
/// vector. An object with no bearing encodes as omnidirectional (`W` only).
#[must_use]
pub fn encode_object(object: &AudioObject, order: usize) -> Vec<Sample> {
    match object.direction() {
        Some(dir) => encode_direction_vec(dir, object.gain, order),
        None => {
            // No bearing: emit a purely omnidirectional source (W only).
            let mut out = Vec::new();
            out.resize(channel_count(order.min(MAX_ORDER)), 0.0);
            if let Some(w) = out.first_mut() {
                *w = object.gain;
            }
            out
        }
    }
}

/// Encodes a [`ClusteredObject`] (representative direction + gain) into an
/// `AmbiX` coefficient vector.
#[must_use]
pub fn encode_cluster(cluster: &ClusteredObject, order: usize) -> Vec<Sample> {
    encode_direction_vec(cluster.direction, cluster.gain, order)
}

/// Encodes a whole cluster list into one `AmbiX` coefficient vector per cluster
/// (row-per-source transport matrix).
#[must_use]
pub fn encode_clusters(clusters: &[ClusteredObject], order: usize) -> Vec<Vec<Sample>> {
    clusters.iter().map(|c| encode_cluster(c, order)).collect()
}

/// Encodes a bed's directional speakers into one `AmbiX` coefficient vector per
/// speaker (the LFE is skipped, carrying no direction).
#[must_use]
pub fn encode_bed(bed: BedLayout, order: usize) -> Vec<Vec<Sample>> {
    bed.channels()
        .iter()
        .filter(|c| !c.is_lfe)
        .map(|c| encode_direction_vec(c.direction, 1.0, order))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::direction_from_angles;
    use crate::object::ObjectId;
    use bevy_math::ops;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn channel_count_matches_order() {
        assert_eq!(channel_count(0), 1);
        assert_eq!(channel_count(1), 4);
        assert_eq!(channel_count(3), 16);
        // Clamped above the maximum.
        assert_eq!(channel_count(9), 16);
    }

    #[test]
    fn w_channel_scales_with_gain() {
        // The zeroth-order (W) coefficient is unit under SN3D, so after gain
        // scaling it equals the gain regardless of direction.
        let coeffs = encode_direction_vec(direction_from_angles(37.0, 12.0), 0.5, 1);
        assert!(close(coeffs[0], 0.5));
    }

    #[test]
    fn encode_decode_round_trips_at_source_direction() {
        // A unit source encoded at a direction decodes back to ~its gain at a
        // coincident "speaker" direction.
        let dir = direction_from_angles(30.0, 20.0);
        let coeffs = encode_direction_vec(dir, 1.0, 3);
        let decoded = hoa::decode_hoa(&coeffs, dir, 3);
        assert!(close(decoded, 1.0));
    }

    #[test]
    fn gain_passes_through_decode() {
        let dir = direction_from_angles(-60.0, 0.0);
        let coeffs = encode_direction_vec(dir, 0.25, 2);
        let decoded = hoa::decode_hoa(&coeffs, dir, 2);
        assert!(close(decoded, 0.25));
    }

    #[test]
    fn object_without_bearing_is_omnidirectional() {
        let o = AudioObject::new(ObjectId(1), Vec3::ZERO, 0.8);
        let coeffs = encode_object(&o, 2);
        // W carries the gain; all higher channels are zero (omni).
        assert!(close(coeffs[0], 0.8));
        for &c in &coeffs[1..] {
            assert!(close(c, 0.0));
        }
    }

    #[test]
    fn encode_bed_skips_lfe_and_sizes_rows() {
        let rows = encode_bed(BedLayout::Surround7_1_4, 1);
        assert_eq!(rows.len(), BedLayout::Surround7_1_4.directional_count());
        for r in &rows {
            assert_eq!(r.len(), channel_count(1));
            // Each bed speaker is a unit source => W == 1.
            assert!(close(r[0], 1.0));
        }
    }

    #[test]
    fn encode_clusters_one_row_each() {
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
        let rows = encode_clusters(&[cluster], 1);
        assert_eq!(rows.len(), 1);
        assert!(close(rows[0][0], 0.5));
    }
}
