//! Shared spatial primitives: the [`Listener`], the [`Emitter`], and the
//! listener-local [`LocalSource`] that every downstream spatial DSP module
//! (attenuation, cone, Doppler, panning, air absorption, Ambisonics) consumes.
//!
//! # Coordinate convention
//!
//! World space is right-handed and matches Bevy's convention: `+X` right,
//! `+Y` up, `-Z` forward. A [`Listener`]'s [`orientation`](Listener::orientation)
//! is a unit quaternion mapping **listener-local space into world space**, so
//! resolving a world direction into the listener's frame applies the inverse
//! rotation. In listener-local space the head therefore faces `-Z`, `+X` is to
//! the right ear and `+Y` is up.
//!
//! # Determinism
//!
//! Every length/normalisation routes through [`bevy_math::ops`] (libm-backed)
//! rather than `f32` intrinsics, and quaternion/vector combinations are pure
//! multiply-adds, so localisation is bit-reproducible across targets and can be
//! golden-compared sample-for-sample.

use bevy_math::{Quat, Vec3, ops};
use prism_audio_core::math::Sample;

/// Distances below this (in metres) are treated as "at the listener": the
/// direction collapses to local forward and the radial velocity to zero,
/// avoiding a division by a near-zero length.
const COINCIDENT_EPSILON: f32 = 1.0e-6;

/// The point of reception in the world (typically the player's head/camera).
///
/// `orientation` maps listener-local space into world space; its inverse is
/// applied to world directions to express them in the listener's frame. All
/// fields are in SI units (metres, metres per second).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Listener {
    /// World-space position in metres.
    pub position: Vec3,
    /// Unit quaternion mapping listener-local space into world space.
    pub orientation: Quat,
    /// World-space velocity in metres per second (drives Doppler).
    pub velocity: Vec3,
}

impl Default for Listener {
    #[inline]
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
        }
    }
}

impl Listener {
    /// Creates a listener at `position` facing along `orientation` with the
    /// given `velocity`.
    #[must_use]
    #[inline]
    pub fn new(position: Vec3, orientation: Quat, velocity: Vec3) -> Self {
        Self { position, orientation, velocity }
    }

    /// Resolves an [`Emitter`] into this listener's local frame, producing the
    /// unit direction, distance, and radial velocity used by every spatial
    /// module.
    ///
    /// When the emitter is coincident with the listener (distance below
    /// [`COINCIDENT_EPSILON`]) the direction collapses to local forward
    /// (`-Z`) and the radial velocity to zero.
    #[must_use]
    pub fn localize(&self, emitter: &Emitter) -> LocalSource {
        let to_source = emitter.position - self.position;
        let distance_sq = to_source.dot(to_source);
        let distance = ops::sqrt(distance_sq);

        if distance <= COINCIDENT_EPSILON {
            return LocalSource {
                direction: Vec3::NEG_Z,
                distance: 0.0,
                radial_velocity: 0.0,
            };
        }

        // World-space unit direction from listener to source.
        let world_dir = to_source / distance;
        // Express it in the listener's local frame (inverse of local->world).
        let direction = self.orientation.inverse() * world_dir;

        // Rate of change of the listener-source distance. Positive means the
        // source is receding (distance increasing); negative means approaching.
        let relative_velocity = emitter.velocity - self.velocity;
        let radial_velocity = relative_velocity.dot(world_dir);

        LocalSource { direction, distance, radial_velocity }
    }
}

/// A sound source in the world.
///
/// `forward` is the source's facing direction (unit, world space) used by
/// directional-cone shaping; it defaults to `-Z`. `velocity` drives Doppler.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Emitter {
    /// World-space position in metres.
    pub position: Vec3,
    /// World-space velocity in metres per second (drives Doppler).
    pub velocity: Vec3,
    /// World-space unit facing direction (drives directional cones).
    pub forward: Vec3,
}

impl Default for Emitter {
    #[inline]
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            forward: Vec3::NEG_Z,
        }
    }
}

impl Emitter {
    /// Creates an emitter at `position` with the given `velocity` and facing
    /// `forward` (which is normalised; a degenerate `forward` falls back to
    /// `-Z`).
    #[must_use]
    #[inline]
    pub fn new(position: Vec3, velocity: Vec3, forward: Vec3) -> Self {
        Self { position, velocity, forward: normalize_or(forward, Vec3::NEG_Z) }
    }

    /// Creates a non-directional emitter (facing `-Z`) at `position` with the
    /// given `velocity`.
    #[must_use]
    #[inline]
    pub fn point(position: Vec3, velocity: Vec3) -> Self {
        Self { position, velocity, forward: Vec3::NEG_Z }
    }
}

/// An emitter resolved into a listener's local frame.
///
/// `direction` is a unit vector in listener-local space (`-Z` forward, `+X`
/// right, `+Y` up). `distance` is in metres. `radial_velocity` is the rate of
/// change of the listener-source distance in metres per second (positive =
/// receding).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LocalSource {
    /// Unit direction from listener to source in listener-local space.
    pub direction: Vec3,
    /// Distance from listener to source in metres.
    pub distance: Sample,
    /// Rate of change of distance in metres per second (positive = receding).
    pub radial_velocity: Sample,
}

impl LocalSource {
    /// Horizontal azimuth in radians: `0` straight ahead, positive toward the
    /// right ear, in `(-pi, pi]`. Computed as `atan2(x, -z)` in listener-local
    /// space.
    #[must_use]
    #[inline]
    pub fn azimuth(&self) -> Sample {
        ops::atan2(self.direction.x, -self.direction.z)
    }

    /// Elevation in radians above the horizontal plane, in `[-pi/2, pi/2]`.
    /// Computed as `atan2(y, hypot(x, z))` in listener-local space.
    #[must_use]
    #[inline]
    pub fn elevation(&self) -> Sample {
        let horizontal = ops::sqrt(
            self.direction.x * self.direction.x + self.direction.z * self.direction.z,
        );
        ops::atan2(self.direction.y, horizontal)
    }
}

/// Normalises `v`, falling back to `fallback` when `v` is too short to have a
/// well-defined direction. Deterministic (routes the length through
/// [`bevy_math::ops`]).
#[must_use]
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.dot(v);
    let len = ops::sqrt(len_sq);
    if len <= COINCIDENT_EPSILON { fallback } else { v / len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn front_source_is_forward_at_correct_distance() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);
        let local = listener.localize(&emitter);
        assert!(approx(local.distance, 5.0, 1e-5));
        // Directly ahead => local forward (-Z).
        assert!(approx(local.direction.x, 0.0, 1e-6));
        assert!(approx(local.direction.y, 0.0, 1e-6));
        assert!(approx(local.direction.z, -1.0, 1e-6));
        assert!(approx(local.azimuth(), 0.0, 1e-5));
        assert!(approx(local.elevation(), 0.0, 1e-5));
    }

    #[test]
    fn right_source_has_positive_azimuth() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let local = listener.localize(&emitter);
        assert!(approx(local.distance, 3.0, 1e-5));
        assert!(approx(local.direction.x, 1.0, 1e-6));
        assert!(approx(local.azimuth(), FRAC_PI_2, 1e-5));
        assert!(approx(local.elevation(), 0.0, 1e-5));
    }

    #[test]
    fn overhead_source_has_positive_elevation() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO);
        let local = listener.localize(&emitter);
        assert!(approx(local.elevation(), FRAC_PI_2, 1e-5));
    }

    #[test]
    fn listener_rotation_is_applied() {
        // Listener yawed 90 degrees about +Y: local->world rotates -Z (forward)
        // to -X. A source at world -X should therefore appear straight ahead.
        let listener = Listener::new(
            Vec3::ZERO,
            Quat::from_rotation_y(FRAC_PI_2),
            Vec3::ZERO,
        );
        let emitter = Emitter::point(Vec3::new(-4.0, 0.0, 0.0), Vec3::ZERO);
        let local = listener.localize(&emitter);
        assert!(approx(local.distance, 4.0, 1e-5));
        assert!(approx(local.direction.z, -1.0, 1e-5));
        assert!(approx(local.azimuth(), 0.0, 1e-4));
    }

    #[test]
    fn radial_velocity_sign_and_magnitude() {
        let listener = Listener::default();
        // Source ahead at -Z moving further away along -Z at 10 m/s => receding.
        let receding = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, -10.0));
        assert!(approx(listener.localize(&receding).radial_velocity, 10.0, 1e-4));
        // Same source moving toward the listener (+Z) => approaching (negative).
        let approaching = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 10.0));
        assert!(approx(listener.localize(&approaching).radial_velocity, -10.0, 1e-4));
        // Tangential motion (along X) has no radial component.
        let tangential = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::new(5.0, 0.0, 0.0));
        assert!(approx(listener.localize(&tangential).radial_velocity, 0.0, 1e-4));
    }

    #[test]
    fn coincident_source_is_stable() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let local = listener.localize(&emitter);
        assert_eq!(local.distance, 0.0);
        assert_eq!(local.radial_velocity, 0.0);
        assert!(approx(local.direction.z, -1.0, 1e-6));
    }

    #[test]
    fn emitter_forward_is_normalized() {
        let e = Emitter::new(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 0.0, -8.0));
        assert!(approx(e.forward.length(), 1.0, 1e-6));
        // Degenerate forward falls back to -Z.
        let d = Emitter::new(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        assert!(approx(d.forward.z, -1.0, 1e-6));
    }

    #[test]
    fn azimuth_range_behind_listener() {
        let listener = Listener::default();
        // Directly behind (+Z) => azimuth magnitude PI.
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let local = listener.localize(&emitter);
        assert!(approx(local.azimuth().abs(), PI, 1e-5));
    }
}
