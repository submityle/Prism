//! Relative contact velocity and its normal/tangential decomposition.
//!
//! A rigid body's surface velocity at a world point `p` is
//! `v + omega x (p - com)`: the centre-of-mass linear velocity plus the
//! rotational contribution. The acoustically meaningful quantity at a contact
//! is the *relative* surface velocity of the two bodies, which this module
//! computes and then splits into a signed normal component (how fast the bodies
//! are approaching) and a non-negative tangential component (how fast they are
//! grazing). Those two numbers drive impact loudness and friction brightness.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the kinematic input of design section 47.1: feeds
//! [`crate::impulse`] (closing speed, reduced mass) and the sustain path of
//! [`crate::translator`] (tangential speed, normal pressure proxy).

use bevy_math::Vec3;

use crate::body::BodyAudioState;
use prism_audio_core::math::Sample;

/// Surface velocity of a body at a world-space point.
///
/// Returns `v + omega x (p - com)`, the rigid-body velocity of the material
/// point currently located at `point`.
#[inline]
#[must_use]
pub fn point_velocity(body: &BodyAudioState, point: Vec3) -> Vec3 {
    let lever = point - body.center_of_mass();
    body.linear_velocity() + body.angular_velocity().cross(lever)
}

/// Relative surface velocity at a contact point, as `v_a_at - v_b_at`.
///
/// With the physics-core normal pointing from `a` toward `b`, a positive
/// projection of this vector onto the normal means the bodies are approaching.
#[inline]
#[must_use]
pub fn relative_velocity(a: &BodyAudioState, b: &BodyAudioState, contact_point: Vec3) -> Vec3 {
    point_velocity(a, contact_point) - point_velocity(b, contact_point)
}

/// The normal/tangential split of a relative contact velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VelocitySplit {
    /// Signed speed along the contact normal.
    ///
    /// Positive means the bodies are approaching (closing); negative means they
    /// are separating. The sign follows the physics-core convention that the
    /// normal points from body `a` toward body `b` and the relative velocity is
    /// taken as `v_a - v_b`.
    pub normal_speed: Sample,
    /// Non-negative magnitude of the velocity tangent to the contact normal.
    pub tangential_speed: Sample,
}

/// Decomposes a relative velocity into signed-normal and tangential parts.
///
/// The `normal` vector is assumed unit length (as emitted by the narrow phase).
/// The tangential magnitude is computed with [`bevy_math::ops::sqrt`] for
/// deterministic cross-platform results.
#[inline]
#[must_use]
pub fn decompose(v_rel: Vec3, normal: Vec3) -> VelocitySplit {
    let normal_speed = v_rel.dot(normal);
    let tangential = v_rel - normal * normal_speed;
    let tangential_speed = bevy_math::ops::sqrt(tangential.dot(tangential));
    VelocitySplit {
        normal_speed,
        tangential_speed,
    }
}

/// Returns the positive closing speed (zero when the bodies are separating).
#[inline]
#[must_use]
pub fn closing_speed(split: VelocitySplit) -> Sample {
    split.normal_speed.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyAudioId;
    use crate::material::AudioMaterialId;

    fn body(linear: Vec3, angular: Vec3, com: Vec3) -> BodyAudioState {
        BodyAudioState::new(BodyAudioId(0), linear, angular, com, 1.0, AudioMaterialId(0))
    }

    #[test]
    fn pure_linear_closing() {
        // a moves +y toward b (normal points a->b = +y); b still.
        let a = body(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, Vec3::ZERO);
        let b = body(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0));
        let v = relative_velocity(&a, &b, Vec3::new(0.0, 1.0, 0.0));
        let split = decompose(v, Vec3::Y);
        assert!((split.normal_speed - 1.0).abs() < 1e-6);
        assert!(split.tangential_speed.abs() < 1e-6);
        assert!((closing_speed(split) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pure_tangential() {
        let a = body(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, Vec3::ZERO);
        let b = body(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let v = relative_velocity(&a, &b, Vec3::ZERO);
        let split = decompose(v, Vec3::Y);
        assert!(split.normal_speed.abs() < 1e-6);
        assert!((split.tangential_speed - 3.0).abs() < 1e-6);
        assert!(closing_speed(split).abs() < 1e-6);
    }

    #[test]
    fn separating_is_negative_normal() {
        let a = body(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, Vec3::ZERO);
        let b = body(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let v = relative_velocity(&a, &b, Vec3::ZERO);
        let split = decompose(v, Vec3::Y);
        assert!(split.normal_speed < 0.0);
        assert!(closing_speed(split).abs() < 1e-6);
    }

    #[test]
    fn angular_contributes_at_lever() {
        // omega about +z, point offset +x => surface velocity along +y.
        let a = body(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO);
        let b = body(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let v = relative_velocity(&a, &b, p);
        assert!((v.y - 1.0).abs() < 1e-6);
    }
}
