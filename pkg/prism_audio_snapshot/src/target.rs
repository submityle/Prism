//! One parameter's target value together with its blending domain.
//!
//! A [`ParameterTarget`] is the atom of a snapshot: it says "parameter `id`,
//! which blends in domain `kind`, should settle at `value`". A snapshot is a
//! collection of these, keyed by parameter.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Combines [`crate::parameter::ParameterId`] and
//! [`crate::parameter::ParameterKind`] with a target [`Sample`]. Stored inside
//! [`crate::snapshot::Snapshot`] and read by [`crate::resolved`] and
//! [`crate::stack`] when resolving targets.

use prism_audio_core::math::Sample;

use crate::parameter::{ParameterId, ParameterKind};

/// A single parameter's target value and the domain it blends in.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterTarget {
    /// Identity of the targeted parameter.
    pub id: ParameterId,
    /// Blending domain of the targeted parameter.
    pub kind: ParameterKind,
    /// Target value the parameter should settle at, in `kind`'s units.
    pub value: Sample,
}

impl ParameterTarget {
    /// Builds a target for `id` of domain `kind` settling at `value`.
    #[must_use]
    pub const fn new(id: ParameterId, kind: ParameterKind, value: Sample) -> Self {
        Self { id, kind, value }
    }

    /// Returns the targeted parameter's identity.
    #[must_use]
    pub const fn id(&self) -> ParameterId {
        self.id
    }

    /// Returns the targeted parameter's blending domain.
    #[must_use]
    pub const fn kind(&self) -> ParameterKind {
        self.kind
    }

    /// Returns the target value.
    #[must_use]
    pub const fn value(&self) -> Sample {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn accessors_return_constructor_arguments() {
        let t = ParameterTarget::new(ParameterId::new(3), ParameterKind::Decibel, -6.0);
        assert_eq!(t.id(), ParameterId::new(3));
        assert_eq!(t.kind(), ParameterKind::Decibel);
        assert!((t.value() - (-6.0)).abs() < EPS);
    }
}
