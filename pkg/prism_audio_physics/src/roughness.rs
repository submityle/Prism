//! Strike-position and surface-roughness derivation for modal excitation.
//!
//! Where a body is struck and how rough the contacting surfaces are shape the
//! timbre as much as the impulse does: a strike near the rim excites high
//! modes (bright), a strike at the acoustic centre is dark, and a rough surface
//! widens the friction bandwidth. [`strike_position`] projects the contact
//! offset from the body centre onto its dominant axis and normalises it into
//! the `[0, 1]` strike coordinate the procedural modal stage expects.
//! [`surface_roughness`] derives a deterministic per-pair roughness so repeated
//! contacts of the same materials always sound identical.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the strike-position and roughness inputs of design sections
//! 47.2-47.3: the [`prism_audio_procedural::contact::ContactPoint`] feeds the
//! modal table and the roughness feeds the friction template.

use bevy_math::Vec3;

use prism_audio_procedural::contact::ContactPoint;
use prism_audio_procedural::material::MaterialPairId;
use prism_audio_core::math::Sample;

/// Projects a contact onto the body's dominant axis as a `[0, 1]` strike point.
///
/// The offset from `body_center` to `contact_point` is reduced to its largest
/// absolute component (the dominant mode axis); that distance divided by
/// `body_extent` gives `0.0` at the acoustic centre and `1.0` at the rim. A
/// non-positive or non-finite extent yields the centre (`0.0`).
#[inline]
#[must_use]
pub fn strike_position(
    contact_point: Vec3,
    body_center: Vec3,
    body_extent: Sample,
) -> ContactPoint {
    if !(body_extent.is_finite() && body_extent > 0.0) {
        return ContactPoint::new(0.0);
    }
    let offset = contact_point - body_center;
    let ax = bevy_math::ops::abs(offset.x);
    let ay = bevy_math::ops::abs(offset.y);
    let az = bevy_math::ops::abs(offset.z);
    let dominant = ax.max(ay).max(az);
    ContactPoint::new(dominant / body_extent)
}

/// Integer avalanche used to derive a stable roughness from a pair id.
#[inline]
#[must_use]
fn mix32(mut z: u32) -> u32 {
    z = z.wrapping_add(0x9e37_79b9);
    z = (z ^ (z >> 16)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

/// Returns a deterministic surface roughness in `[0, 1]` for a material pair.
///
/// A distinguishing pair (not the all-zero generic pair) is hashed into a
/// stable `[0, 1]` value so the same materials always produce the same
/// roughness. The all-zero generic pair returns the sanitised `default`.
#[inline]
#[must_use]
pub fn surface_roughness(material_pair: MaterialPairId, default: Sample) -> Sample {
    let base = if default.is_finite() {
        default.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let key = ((material_pair.lo() as u32) << 16) | material_pair.hi() as u32;
    if key == 0 {
        return base;
    }
    let hashed = mix32(key);
    // Map the full 32-bit hash into [0, 1] via exact integer-to-float division.
    hashed as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_strike_is_zero() {
        let p = strike_position(Vec3::ZERO, Vec3::ZERO, 2.0);
        assert!(p.value().abs() < 1e-6);
    }

    #[test]
    fn rim_strike_is_one() {
        let p = strike_position(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 2.0);
        assert!((p.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dominant_axis_wins() {
        // y offset is largest; extent 4 => 3/4 = 0.75.
        let p = strike_position(Vec3::new(1.0, 3.0, 2.0), Vec3::ZERO, 4.0);
        assert!((p.value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn beyond_rim_clamps_to_one() {
        let p = strike_position(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, 2.0);
        assert!((p.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_positive_extent_is_center() {
        let p = strike_position(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 0.0);
        assert!(p.value().abs() < 1e-6);
    }

    #[test]
    fn roughness_is_deterministic_and_in_range() {
        let pair = MaterialPairId::new(3, 7);
        let a = surface_roughness(pair, 0.5);
        let b = surface_roughness(pair, 0.5);
        assert!((a - b).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn generic_pair_returns_default() {
        let pair = MaterialPairId::new(0, 0);
        let r = surface_roughness(pair, 0.3);
        assert!((r - 0.3).abs() < 1e-6);
    }
}
