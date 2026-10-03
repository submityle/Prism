//! Object-to-bed amplitude panning (VBAP) with spread and snap.
//!
//! This module is the high-level panning entry point. It turns a source
//! direction (plus a `spread` width and a `snap` request) into a full
//! per-channel gain vector for a [`crate::bed::BedLayout`]:
//!
//! * [`vbap`] computes the raw VBAP gains over the layout's directional
//!   speakers;
//! * `spread` in `[0, 1]` blends those gains toward an even spread across all
//!   directional speakers, widening a point source into a diffuse one;
//! * `snap` overrides panning and places the whole source on the single
//!   nearest speaker;
//! * [`channel_layout::SpeakerArray`] scatters the result back into channel
//!   order with the LFE left silent.
//!
//! All outputs are constant-power normalised: the sum of squared channel gains
//! is one whenever any speaker is driven.
//!
//! # Determinism
//!
//! All math routes through [`bevy_math::ops`]; results are bit-reproducible.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the VBAP object-to-bed panning of design section 44.2.
//! Consumed by [`crate::fold`] and [`crate::render`].

pub mod channel_layout;
pub mod vbap;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;

use crate::bed::BedLayout;
use crate::clustering::ClusteredObject;
use crate::object::AudioObject;
use channel_layout::SpeakerArray;

/// Renormalises `gains` to constant power (sum of squares one); a near-zero
/// vector is left untouched.
fn normalize_constant_power(gains: &mut [Sample]) {
    let mut energy: Sample = 0.0;
    for &g in gains.iter() {
        energy += g * g;
    }
    if energy <= 1e-20 {
        return;
    }
    let inv = 1.0 / ops::sqrt(energy);
    for g in gains.iter_mut() {
        *g *= inv;
    }
}

/// Blends VBAP gains toward an even spread across all speakers by `spread`,
/// then renormalises to constant power. `spread == 0` is a no-op; `spread == 1`
/// is a fully diffuse (equal-power) spread.
fn apply_spread(gains: &mut [Sample], spread: Sample) {
    let spread = spread.clamp(0.0, 1.0);
    if spread <= 0.0 || gains.is_empty() {
        return;
    }
    let n = gains.len() as Sample;
    let uniform = 1.0 / ops::sqrt(n);
    for g in gains.iter_mut() {
        *g = (1.0 - spread) * *g + spread * uniform;
    }
    normalize_constant_power(gains);
}

/// Pans a source `direction` onto `bed`, returning a per-channel gain vector of
/// length [`BedLayout::channel_count`] (the LFE is always zero).
///
/// `spread` in `[0, 1]` widens the source; `snap` places it on the single
/// nearest speaker (overriding `spread`).
#[must_use]
pub fn pan_direction_to_bed(
    direction: Vec3,
    bed: BedLayout,
    spread: Sample,
    snap: bool,
) -> Vec<Sample> {
    let arr = SpeakerArray::new(bed);
    if arr.is_empty() {
        let mut full = Vec::new();
        full.resize(bed.channel_count(), 0.0);
        return full;
    }

    let mut local = Vec::new();
    local.resize(arr.len(), 0.0);

    if snap {
        if let Some(idx) = vbap::nearest_speaker(direction, arr.directions()) {
            local[idx] = 1.0;
        }
    } else {
        local = vbap::vbap_gains(direction, arr.directions());
        apply_spread(&mut local, spread);
    }

    arr.scatter(&local)
}

/// Pans an [`AudioObject`] onto `bed` using its direction, spread, and snap
/// flag. An object with no bearing (at the listener) collapses to the forward
/// direction.
#[must_use]
pub fn pan_object_to_bed(object: &AudioObject, bed: BedLayout) -> Vec<Sample> {
    let dir = object.direction().unwrap_or(Vec3::NEG_Z);
    pan_direction_to_bed(dir, bed, object.spread, object.snap)
}

/// Pans a [`ClusteredObject`] onto `bed` using its representative direction,
/// spread, and snap flag.
#[must_use]
pub fn pan_cluster_to_bed(cluster: &ClusteredObject, bed: BedLayout) -> Vec<Sample> {
    pan_direction_to_bed(cluster.direction, bed, cluster.spread, cluster.snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::direction_from_angles;
    use crate::object::ObjectId;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn sum_sq(g: &[Sample]) -> Sample {
        g.iter().map(|&x| x * x).sum()
    }

    #[test]
    fn lfe_is_never_driven() {
        let dir = direction_from_angles(0.0, 0.0);
        let g = pan_direction_to_bed(dir, BedLayout::Surround7_1_4, 0.0, false);
        let lfe = BedLayout::Surround7_1_4.lfe_index().unwrap();
        assert_eq!(g[lfe], 0.0);
        assert_eq!(g.len(), BedLayout::Surround7_1_4.channel_count());
    }

    #[test]
    fn front_center_lands_on_center_channel() {
        // 7.1.4 channel 2 is the center speaker at azimuth 0.
        let dir = direction_from_angles(0.0, 0.0);
        let g = pan_direction_to_bed(dir, BedLayout::Surround7_1_4, 0.0, false);
        assert!(g[2] > 0.99);
        assert!(close(sum_sq(&g), 1.0));
    }

    #[test]
    fn snap_puts_all_energy_on_one_speaker() {
        // A direction near but not exactly on the center.
        let dir = direction_from_angles(5.0, 2.0);
        let g = pan_direction_to_bed(dir, BedLayout::Surround7_1_4, 0.0, true);
        let driven = g.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(driven, 1);
        assert!(close(sum_sq(&g), 1.0));
    }

    #[test]
    fn spread_widens_but_keeps_constant_power() {
        let dir = direction_from_angles(0.0, 0.0);
        let point = pan_direction_to_bed(dir, BedLayout::Surround7_1_4, 0.0, false);
        let diffuse = pan_direction_to_bed(dir, BedLayout::Surround7_1_4, 1.0, false);
        let point_driven = point.iter().filter(|&&x| x > 1e-4).count();
        let diffuse_driven = diffuse.iter().filter(|&&x| x > 1e-4).count();
        assert!(diffuse_driven > point_driven);
        assert!(close(sum_sq(&diffuse), 1.0));
    }

    #[test]
    fn object_without_bearing_pans_forward() {
        let o = AudioObject::new(ObjectId(1), Vec3::ZERO, 1.0);
        let g = pan_object_to_bed(&o, BedLayout::Surround7_1_4);
        // Forward => center channel dominates.
        assert!(g[2] > 0.99);
    }
}
