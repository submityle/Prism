//! Modulation targets, expression dimensions, and the per-dimension mapping.
//!
//! Expression arrives from a controller as a handful of continuous dimensions
//! (pitch bend, pressure, timbre, velocity, and arbitrary controllers). The
//! engine, on the other side, exposes modulation targets (pitch, gain, filter
//! cutoff, and engine-specific custom slots). This module names both sides and
//! defines the [`TargetMapping`] that connects one dimension to one target with
//! a depth, an offset, and a response [`Curve`]. A mapping is pure data: its
//! [`TargetMapping::apply`] turns a dimension value into a modulation amount
//! with deterministic integer-to-float normalisation and
//! [`bevy_math::ops`]-based curve math, performing no DSP of its own.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the mapping description of design section 52 (expression to
//! modulation). The [`ModulationTarget`] values name the destinations the
//! engine's section 12 modulation routing understands; [`crate::mapping::router`]
//! uses these mappings to emit modulation writes.

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// A modulation destination inside the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ModulationTarget {
    /// Pitch, expressed as a signed offset in semitones.
    Pitch,
    /// Linear gain multiplier contribution.
    Gain,
    /// Filter cutoff contribution.
    Cutoff,
    /// An engine-specific modulation slot addressed by index.
    Custom {
        /// The custom slot index.
        slot: u16,
    },
}

/// A continuous expression dimension produced by a controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ExpressionDimension {
    /// Combined channel and per-note pitch bend, measured in semitones.
    PitchBend,
    /// Combined channel and per-note pressure, in the unipolar `[0, 1]` range.
    Pressure,
    /// Timbre (brightness), in the unipolar `[0, 1]` range.
    Timbre,
    /// Note-on velocity, in the unipolar `[0, 1]` range.
    Velocity,
    /// A specific per-voice controller, in the unipolar `[0, 1]` range.
    Controller {
        /// The controller index (`0`-`127`).
        index: u8,
    },
}

/// The response shape applied to a dimension value before scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Curve {
    /// Identity: the value passes through unchanged.
    Linear,
    /// A power curve, sign-preserving so bipolar inputs stay bipolar: the
    /// output is `sign(x) * powf(abs(x), exponent)`.
    Exponential {
        /// The exponent applied to the magnitude of the value.
        exponent: Sample,
    },
}

impl Curve {
    /// Shapes `value` according to the curve. The transformation preserves the
    /// sign of the input so a bipolar dimension (such as pitch bend) stays
    /// bipolar.
    #[must_use]
    pub fn shape(self, value: Sample) -> Sample {
        match self {
            Self::Linear => value,
            Self::Exponential { exponent } => {
                let magnitude = ops::powf(ops::abs(value), exponent);
                if value < 0.0 { -magnitude } else { magnitude }
            }
        }
    }
}

/// A mapping from one expression dimension to one modulation target.
///
/// The [`apply`](Self::apply) method shapes the dimension value with the curve,
/// scales it by `depth`, and adds `offset`, producing the modulation amount for
/// the target.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetMapping {
    /// The source expression dimension.
    pub dimension: ExpressionDimension,
    /// The destination modulation target.
    pub target: ModulationTarget,
    /// The scale applied to the shaped dimension value.
    pub depth: Sample,
    /// The constant added after scaling.
    pub offset: Sample,
    /// The response curve.
    pub curve: Curve,
}

impl TargetMapping {
    /// Creates a linear mapping with unit depth and no offset.
    #[must_use]
    pub const fn linear(dimension: ExpressionDimension, target: ModulationTarget) -> Self {
        Self {
            dimension,
            target,
            depth: 1.0,
            offset: 0.0,
            curve: Curve::Linear,
        }
    }

    /// Creates a mapping with an explicit depth, offset, and curve.
    #[must_use]
    pub const fn new(
        dimension: ExpressionDimension,
        target: ModulationTarget,
        depth: Sample,
        offset: Sample,
        curve: Curve,
    ) -> Self {
        Self {
            dimension,
            target,
            depth,
            offset,
            curve,
        }
    }

    /// Maps a dimension value to a modulation amount.
    #[must_use]
    pub fn apply(&self, value: Sample) -> Sample {
        self.offset + self.depth * self.curve.shape(value)
    }
}

/// Normalises a full-range 32-bit unipolar value to the `[0, 1]` range.
#[must_use]
pub fn normalize_u32(value: u32) -> Sample {
    (value as Sample) / (u32::MAX as Sample)
}

/// Normalises a 16-bit unipolar value (such as velocity) to the `[0, 1]` range.
#[must_use]
pub fn normalize_u16(value: u16) -> Sample {
    (value as Sample) / (u16::MAX as Sample)
}

#[cfg(test)]
mod tests {
    use super::{
        Curve, ExpressionDimension, ModulationTarget, TargetMapping, normalize_u16, normalize_u32,
    };

    const EPS: f32 = 1.0e-6;

    #[test]
    fn linear_mapping_passes_through() {
        let mapping =
            TargetMapping::linear(ExpressionDimension::PitchBend, ModulationTarget::Pitch);
        assert!((mapping.apply(2.0) - 2.0).abs() < EPS);
        assert!((mapping.apply(-2.0) + 2.0).abs() < EPS);
    }

    #[test]
    fn depth_and_offset_scale_value() {
        let mapping = TargetMapping::new(
            ExpressionDimension::Pressure,
            ModulationTarget::Gain,
            0.5,
            0.25,
            Curve::Linear,
        );
        assert!((mapping.apply(1.0) - 0.75).abs() < EPS);
        assert!((mapping.apply(0.0) - 0.25).abs() < EPS);
    }

    #[test]
    fn exponential_curve_preserves_sign() {
        let curve = Curve::Exponential { exponent: 2.0 };
        assert!((curve.shape(0.5) - 0.25).abs() < EPS);
        assert!((curve.shape(-0.5) + 0.25).abs() < EPS);
        assert!(curve.shape(0.0).abs() < EPS);
    }

    #[test]
    fn normalisation_endpoints() {
        assert!(normalize_u32(0).abs() < EPS);
        assert!((normalize_u32(u32::MAX) - 1.0).abs() < EPS);
        assert!(normalize_u16(0).abs() < EPS);
        assert!((normalize_u16(u16::MAX) - 1.0).abs() < EPS);
    }

    #[test]
    fn custom_target_round_trips() {
        let mapping = TargetMapping::linear(
            ExpressionDimension::Controller { index: 74 },
            ModulationTarget::Custom { slot: 3 },
        );
        assert_eq!(mapping.target, ModulationTarget::Custom { slot: 3 });
        assert_eq!(
            mapping.dimension,
            ExpressionDimension::Controller { index: 74 }
        );
    }
}
