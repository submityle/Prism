//! Per-body kinematic and material snapshot the bridge consumes.
//!
//! The physics solver owns the authoritative rigid-body state; the audio bridge
//! only needs a read-only snapshot of the quantities that shape a contact
//! sound: the linear and angular velocity (to compute the relative velocity at
//! a contact point), the centre of mass (the pivot for the angular term), the
//! inverse mass (to form the reduced mass of a collision), and the acoustic
//! material (to resolve the colliding pair). [`BodyAudioState`] is a plain,
//! `Copy` value record carrying exactly that.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supplies the per-body facts for design section 47.1: consumed by
//! [`crate::kinematics`] and [`crate::impulse`] to turn physics truth into the
//! impact/sustain drive assembled by [`crate::translator`].

use bevy_math::Vec3;

use crate::material::AudioMaterialId;

/// Stable identifier of a body in the audio bridge.
///
/// Wraps an opaque 64-bit key (typically packed from the physics body handle's
/// index and generation); the bridge only compares, orders, and pairs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BodyAudioId(pub u64);

/// Read-only snapshot of a body's acoustically relevant rigid-body state.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BodyAudioState {
    id: BodyAudioId,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
    center_of_mass: Vec3,
    inverse_mass: f32,
    material: AudioMaterialId,
}

impl BodyAudioState {
    /// Builds a body snapshot from its kinematic and material facts.
    ///
    /// `inverse_mass` is `1 / mass` for a dynamic body and `0` for a static or
    /// infinitely heavy body; negative or non-finite values are clamped to `0`.
    #[inline]
    #[must_use]
    pub fn new(
        id: BodyAudioId,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        center_of_mass: Vec3,
        inverse_mass: f32,
        material: AudioMaterialId,
    ) -> Self {
        let inverse_mass = if inverse_mass.is_finite() && inverse_mass > 0.0 {
            inverse_mass
        } else {
            0.0
        };
        Self {
            id,
            linear_velocity,
            angular_velocity,
            center_of_mass,
            inverse_mass,
            material,
        }
    }

    /// Returns the body identifier.
    #[inline]
    #[must_use]
    pub fn id(&self) -> BodyAudioId {
        self.id
    }

    /// Returns the linear velocity of the centre of mass.
    #[inline]
    #[must_use]
    pub fn linear_velocity(&self) -> Vec3 {
        self.linear_velocity
    }

    /// Returns the angular velocity about the centre of mass.
    #[inline]
    #[must_use]
    pub fn angular_velocity(&self) -> Vec3 {
        self.angular_velocity
    }

    /// Returns the world-space centre of mass.
    #[inline]
    #[must_use]
    pub fn center_of_mass(&self) -> Vec3 {
        self.center_of_mass
    }

    /// Returns the inverse mass (`0` for static or infinite-mass bodies).
    #[inline]
    #[must_use]
    pub fn inverse_mass(&self) -> f32 {
        self.inverse_mass
    }

    /// Returns `true` when the body cannot move (inverse mass is zero).
    #[inline]
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.inverse_mass == 0.0
    }

    /// Returns the acoustic material of the body.
    #[inline]
    #[must_use]
    pub fn material(&self) -> AudioMaterialId {
        self.material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_round_trip() {
        let s = BodyAudioState::new(
            BodyAudioId(7),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(4.0, 5.0, 6.0),
            0.25,
            AudioMaterialId(9),
        );
        assert_eq!(s.id(), BodyAudioId(7));
        assert_eq!(s.linear_velocity(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.angular_velocity(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(s.center_of_mass(), Vec3::new(4.0, 5.0, 6.0));
        assert!((s.inverse_mass() - 0.25).abs() < 1e-6);
        assert_eq!(s.material(), AudioMaterialId(9));
        assert!(!s.is_static());
    }

    #[test]
    fn non_finite_inverse_mass_becomes_static() {
        let s = BodyAudioState::new(
            BodyAudioId(1),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            f32::NAN,
            AudioMaterialId(0),
        );
        assert!(s.is_static());
        assert!((s.inverse_mass()).abs() < 1e-6);
    }

    #[test]
    fn negative_inverse_mass_becomes_static() {
        let s = BodyAudioState::new(
            BodyAudioId(1),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            -1.0,
            AudioMaterialId(0),
        );
        assert!(s.is_static());
    }
}
