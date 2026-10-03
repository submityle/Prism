//! Deterministic collision-impulse estimation from closing speed.
//!
//! When two bodies collide the solver applies an impulse that reverses their
//! closing velocity scaled by the restitution. The audio bridge does not need
//! the solver's exact value; it needs a physically faithful estimate to drive
//! loudness. For a one-dimensional elastic contact the impulse magnitude is
//! `(1 + e) * m_reduced * v_closing`, where the reduced mass
//! `m_reduced = 1 / (inv_a + inv_b)` accounts for how a light body bouncing off
//! a heavy one carries little momentum exchange. This module computes that
//! magnitude and splits it into normal/tangential parts using the velocity
//! decomposition ratios from [`crate::kinematics`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the energy source of design section 47.1-47.2: the magnitude is
//! the primary loudness driver carried by
//! [`prism_audio_procedural::contact::ImpactEvent`].

use crate::kinematics::VelocitySplit;
use prism_audio_core::math::Sample;

/// Returns the reduced mass of a two-body contact from their inverse masses.
///
/// `m_reduced = 1 / (inv_a + inv_b)`. Two static bodies (both inverse masses
/// zero) and the degenerate both-infinite case return `0`, meaning no momentum
/// is exchanged and the contact is silent.
#[inline]
#[must_use]
pub fn reduced_mass(inv_a: Sample, inv_b: Sample) -> Sample {
    let sum = inv_a.max(0.0) + inv_b.max(0.0);
    if sum > Sample::MIN_POSITIVE {
        1.0 / sum
    } else {
        0.0
    }
}

/// Estimates the collision impulse magnitude for a closing contact.
///
/// `(1 + e) * m_reduced * v_closing`, clamped to be non-negative. A separating
/// contact (`closing_normal_speed <= 0`) and a zero reduced mass both yield
/// `0`. The restitution `e` is clamped into `[0, 1]`.
#[inline]
#[must_use]
pub fn estimate_impulse(
    reduced_mass: Sample,
    closing_normal_speed: Sample,
    restitution: Sample,
) -> Sample {
    let e = if restitution.is_finite() {
        restitution.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let v = closing_normal_speed.max(0.0);
    let m = reduced_mass.max(0.0);
    ((1.0 + e) * m * v).max(0.0)
}

/// A collision impulse split into magnitude and normal/tangential components.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ImpulseEstimate {
    /// Total impulse magnitude (`kg*m/s`).
    pub magnitude: Sample,
    /// Component attributed to the normal (perpendicular) direction.
    pub normal: Sample,
    /// Component attributed to the tangential (grazing) direction.
    pub tangential: Sample,
}

impl ImpulseEstimate {
    /// Builds a full impulse estimate from a reduced mass and velocity split.
    ///
    /// The magnitude uses the closing normal speed; the normal/tangential split
    /// mirrors the velocity direction ratios, so a grazing collision carries
    /// more of its impulse in the tangential channel. When the relative speed
    /// is zero the whole impulse is attributed to the normal.
    #[inline]
    #[must_use]
    pub fn from_split(
        reduced_mass: Sample,
        split: VelocitySplit,
        restitution: Sample,
    ) -> ImpulseEstimate {
        let closing = split.normal_speed.max(0.0);
        let magnitude = estimate_impulse(reduced_mass, closing, restitution);
        let normal_component = closing.abs();
        let tangential_component = split.tangential_speed.max(0.0);
        let total = normal_component + tangential_component;
        if total > Sample::MIN_POSITIVE {
            let ratio_n = normal_component / total;
            ImpulseEstimate {
                magnitude,
                normal: magnitude * ratio_n,
                tangential: magnitude * (1.0 - ratio_n),
            }
        } else {
            ImpulseEstimate {
                magnitude,
                normal: magnitude,
                tangential: 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_mass_of_equal_bodies() {
        // Two unit masses: inv = 1 each, reduced = 0.5.
        assert!((reduced_mass(1.0, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reduced_mass_with_static_body() {
        // Dynamic (inv=0.5) against static (inv=0) => reduced = mass of dynamic.
        assert!((reduced_mass(0.5, 0.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn two_static_bodies_have_zero_reduced_mass() {
        assert!(reduced_mass(0.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn impulse_scales_with_restitution() {
        let soft = estimate_impulse(1.0, 2.0, 0.0);
        let bouncy = estimate_impulse(1.0, 2.0, 1.0);
        assert!((soft - 2.0).abs() < 1e-6);
        assert!((bouncy - 4.0).abs() < 1e-6);
    }

    #[test]
    fn separating_contact_has_zero_impulse() {
        assert!(estimate_impulse(1.0, -3.0, 0.5).abs() < 1e-6);
    }

    #[test]
    fn pure_normal_split_is_all_normal() {
        let split = VelocitySplit {
            normal_speed: 2.0,
            tangential_speed: 0.0,
        };
        let est = ImpulseEstimate::from_split(1.0, split, 0.0);
        assert!((est.magnitude - 2.0).abs() < 1e-6);
        assert!((est.normal - 2.0).abs() < 1e-6);
        assert!(est.tangential.abs() < 1e-6);
    }

    #[test]
    fn mixed_split_divides_by_ratio() {
        let split = VelocitySplit {
            normal_speed: 3.0,
            tangential_speed: 1.0,
        };
        let est = ImpulseEstimate::from_split(1.0, split, 0.0);
        // magnitude = 1 * 3 = 3; ratio_n = 3/4 => normal 2.25, tangential 0.75.
        assert!((est.magnitude - 3.0).abs() < 1e-6);
        assert!((est.normal - 2.25).abs() < 1e-6);
        assert!((est.tangential - 0.75).abs() < 1e-6);
    }

    #[test]
    fn zero_speed_attributes_to_normal() {
        let split = VelocitySplit {
            normal_speed: 0.0,
            tangential_speed: 0.0,
        };
        let est = ImpulseEstimate::from_split(1.0, split, 0.5);
        assert!(est.magnitude.abs() < 1e-6);
        assert!(est.normal.abs() < 1e-6);
        assert!(est.tangential.abs() < 1e-6);
    }
}
