//! Contact event value types fed by the physics solver into the audio engine.
//!
//! The physics step emits three kinds of contact facts, mirrored here one to
//! one: a first [`ImpactEvent`] (a discrete collision carrying an impulse,
//! its normal/tangential split, the contact point, and the colliding material
//! pair), a continuous [`SustainEvent`] (per-step relative tangential speed,
//! normal pressure, and surface roughness while two bodies stay in contact),
//! and a [`SeparationEvent`] when they part. Every event carries a
//! sample-accurate offset inside the current audio block so an impact lands on
//! the exact sample the collision happened at, avoiding frame-rate "machine
//! gun" quantisation.
//!
//! These are plain, `Copy` data records with no behaviour beyond sanitising
//! their own fields; the mapping from physical quantities to synthesis
//! excitation lives in [`crate::contact::energy_map`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the contact facts of design section 47.1; the material pair is
//! identified by [`crate::material::MaterialPairId`], and the sample offset is
//! consumed by the sample-accurate dispatch of
//! [`prism_audio_core::scheduler::EventScheduler`] at the integration layer.

use crate::material::MaterialPairId;
use prism_audio_core::math::Sample;

/// Stable identifier of a persistent contact manifold between two bodies.
///
/// The same physical contact keeps the same id across steps so the continuous
/// state machine ([`crate::continuous`]) can track its impact -> roll -> slide
/// -> separate life cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactId(pub u64);

/// Normalised position of the strike along a body's dominant mode axis.
///
/// `0.0` is the acoustic centre (anti-node of the fundamental, darkest strike)
/// and `1.0` is the rim/edge (excites high modes, brightest strike). Values are
/// clamped into `[0, 1]` on construction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactPoint(Sample);

impl ContactPoint {
    /// Builds a contact point, clamping `value` into `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn new(value: Sample) -> Self {
        Self(if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        })
    }

    /// Returns the normalised position in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn value(self) -> Sample {
        self.0
    }
}

impl Default for ContactPoint {
    #[inline]
    fn default() -> Self {
        Self(0.0)
    }
}

/// A discrete collision: a single impulse exciting the resonant bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ImpactEvent {
    /// Identifier of the owning contact manifold.
    pub contact: ContactId,
    /// Colliding material pair; selects the modal table and friction template.
    pub material_pair: MaterialPairId,
    /// Collision impulse magnitude (`kg*m/s`), the primary loudness driver.
    pub impulse: Sample,
    /// Normal (perpendicular) component of the impulse.
    pub normal: Sample,
    /// Tangential (grazing) component of the impulse.
    pub tangential: Sample,
    /// Where on the body the strike landed.
    pub point: ContactPoint,
    /// Sample offset inside the current block the impact lands at.
    pub sample_offset: u32,
}

impl ImpactEvent {
    /// Builds an impact, sanitising non-finite magnitudes to zero and clamping
    /// the sample offset below `block_frames`.
    #[inline]
    #[must_use]
    pub fn new(
        contact: ContactId,
        material_pair: MaterialPairId,
        impulse: Sample,
        normal: Sample,
        tangential: Sample,
        point: ContactPoint,
        sample_offset: u32,
        block_frames: u32,
    ) -> Self {
        let clamp = |x: Sample| if x.is_finite() { x.max(0.0) } else { 0.0 };
        Self {
            contact,
            material_pair,
            impulse: clamp(impulse),
            normal: clamp(normal),
            tangential: clamp(tangential),
            point,
            sample_offset: sample_offset.min(block_frames.saturating_sub(1)),
        }
    }
}

/// A continuous contact sample: the state of a manifold that stays in contact.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SustainEvent {
    /// Identifier of the owning contact manifold.
    pub contact: ContactId,
    /// Colliding material pair; selects the friction template.
    pub material_pair: MaterialPairId,
    /// Relative tangential speed at the contact (`m/s`); drives friction gain
    /// and brightness, and the rolling pulse rate.
    pub tangential_speed: Sample,
    /// Normal pressure holding the bodies together; scales overall level.
    pub normal_pressure: Sample,
    /// Surface roughness in `[0, 1]`; widens friction bandwidth and sets the
    /// rolling grain density.
    pub roughness: Sample,
}

impl SustainEvent {
    /// Builds a sustain sample, sanitising its fields into their valid ranges.
    #[inline]
    #[must_use]
    pub fn new(
        contact: ContactId,
        material_pair: MaterialPairId,
        tangential_speed: Sample,
        normal_pressure: Sample,
        roughness: Sample,
    ) -> Self {
        let non_neg = |x: Sample| if x.is_finite() { x.max(0.0) } else { 0.0 };
        Self {
            contact,
            material_pair,
            tangential_speed: non_neg(tangential_speed),
            normal_pressure: non_neg(normal_pressure),
            roughness: if roughness.is_finite() {
                roughness.clamp(0.0, 1.0)
            } else {
                0.5
            },
        }
    }
}

/// Emitted once when a tracked contact manifold breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SeparationEvent {
    /// Identifier of the contact manifold that broke.
    pub contact: ContactId,
}

/// The tagged union handed across the contact bus.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ContactEvent {
    /// A discrete collision.
    Impact(ImpactEvent),
    /// A continuous-contact sample.
    Sustain(SustainEvent),
    /// A contact manifold breaking.
    Separation(SeparationEvent),
}

impl ContactEvent {
    /// Returns the owning contact manifold id.
    #[inline]
    #[must_use]
    pub fn contact(&self) -> ContactId {
        match self {
            ContactEvent::Impact(e) => e.contact,
            ContactEvent::Sustain(e) => e.contact,
            ContactEvent::Separation(e) => e.contact,
        }
    }

    /// Returns the sample offset within the block (continuous and separation
    /// events are block-aligned and report `0`).
    #[inline]
    #[must_use]
    pub fn sample_offset(&self) -> u32 {
        match self {
            ContactEvent::Impact(e) => e.sample_offset,
            ContactEvent::Sustain(_) | ContactEvent::Separation(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_point_clamps() {
        assert_eq!(ContactPoint::new(2.0).value(), 1.0);
        assert_eq!(ContactPoint::new(-1.0).value(), 0.0);
        assert_eq!(ContactPoint::new(f32::NAN).value(), 0.0);
    }

    #[test]
    fn impact_sanitises_and_clamps_offset() {
        let e = ImpactEvent::new(
            ContactId(1),
            MaterialPairId::new(2, 3),
            f32::NAN,
            -4.0,
            1.0,
            ContactPoint::new(0.5),
            500,
            128,
        );
        assert_eq!(e.impulse, 0.0);
        assert_eq!(e.normal, 0.0);
        assert_eq!(e.sample_offset, 127);
    }

    #[test]
    fn sustain_sanitises_roughness() {
        let e = SustainEvent::new(ContactId(1), MaterialPairId::new(0, 0), -1.0, 2.0, 5.0);
        assert_eq!(e.tangential_speed, 0.0);
        assert_eq!(e.roughness, 1.0);
    }

    #[test]
    fn event_offset_dispatch() {
        let e = ContactEvent::Impact(ImpactEvent::new(
            ContactId(9),
            MaterialPairId::new(1, 1),
            1.0,
            1.0,
            0.0,
            ContactPoint::default(),
            64,
            128,
        ));
        assert_eq!(e.sample_offset(), 64);
        assert_eq!(e.contact(), ContactId(9));
    }
}
