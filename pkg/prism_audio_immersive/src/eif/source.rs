//! Encoder Input Format (EIF) sound sources.
//!
//! An [`EifSource`] is the declarative description of one emitter in an EIF
//! scene: what kind of signal it carries (a point object, a bed channel feed,
//! or a higher-order Ambisonic field), where it sits and which way it faces,
//! its overall gain, its frequency-dependent radiation directivity, and its
//! distance roll-off. Sources are pure authoring data; turning them into the
//! engine's runtime source nodes and spatial-parameter targets is the job of
//! [`crate::eif::import`].
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The object / channel / HOA source taxonomy and the directivity + distance
//! attenuation attributes are the publicly documented EIF source model
//! (MPEG-I Immersive Audio, ISO/IEC 23090-4, section 49.1), carried here as
//! plain scene data.
//!
//! # Relationship
//!
//! Reuses [`prism_audio_spatial::source_directivity::SourceDirectivity`] for
//! the radiation pattern and [`prism_audio_spatial::attenuation::Attenuation`]
//! for the distance curve, so an EIF source maps one-to-one onto the engine's
//! existing section 14 / section 16 source attributes. Aggregated by
//! [`crate::eif::scene::EifScene`] and consumed by [`crate::eif::import`],
//! which binds each source to a [`prism_audio_spatial::geometry::Emitter`] and
//! a [`prism_audio_spatial::spatializer::SourceDescriptor`].

use bevy_math::Vec3;

use prism_audio_core::math::Sample;
use prism_audio_spatial::attenuation::Attenuation;
use prism_audio_spatial::source_directivity::SourceDirectivity;

/// A stable identifier for a source within one EIF scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EifSourceId(pub u32);

impl EifSourceId {
    /// The underlying numeric id.
    #[inline]
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The signal class carried by an [`EifSource`].
///
/// EIF distinguishes the three ADM-aligned audio types: a single-point
/// *object*, a *channel* feed destined for a named bed loudspeaker, and a
/// scene-based *HOA* field of a given Ambisonic order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum EifSourceKind {
    /// A point object positioned freely in the scene.
    Object,
    /// A channel feed bound to a named bed loudspeaker (for example `"L"`,
    /// `"TFR"`). The label matches the engine's bed channel labels.
    Channel {
        /// The target bed loudspeaker label.
        bed_label: ChannelLabel,
    },
    /// A higher-order Ambisonic scene field of the given order.
    Hoa {
        /// The Ambisonic order (clamped into the engine's supported range by
        /// [`EifSource::hoa_order`]).
        order: u8,
    },
}

/// A fixed-capacity ASCII bed-channel label (no heap allocation, so it is
/// equally usable in `no_std` builds and copyable).
///
/// Labels are at most [`ChannelLabel::MAX_LEN`] bytes of printable ASCII; any
/// longer or non-ASCII input is truncated / sanitised on construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelLabel {
    bytes: [u8; ChannelLabel::MAX_LEN],
    len: u8,
}

impl ChannelLabel {
    /// Maximum number of bytes a label can hold.
    pub const MAX_LEN: usize = 8;

    /// Builds a label from a string slice, keeping only the leading printable
    /// ASCII bytes up to [`ChannelLabel::MAX_LEN`].
    #[must_use]
    pub fn new(label: &str) -> Self {
        let mut bytes = [0u8; Self::MAX_LEN];
        let mut len = 0usize;
        for &b in label.as_bytes() {
            if len >= Self::MAX_LEN {
                break;
            }
            if (0x20..=0x7e).contains(&b) {
                bytes[len] = b;
                len += 1;
            }
        }
        Self {
            bytes,
            len: len as u8,
        }
    }

    /// The label as a string slice (always valid ASCII, hence valid UTF-8).
    #[must_use]
    pub fn as_str(&self) -> &str {
        let end = self.len as usize;
        // The constructor only ever stores printable ASCII, which is valid
        // UTF-8, so this conversion cannot fail; fall back to the empty string
        // rather than panicking if an invariant were ever violated.
        core::str::from_utf8(&self.bytes[..end]).unwrap_or("")
    }

    /// Whether the label is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One declarative EIF source: a signal class plus its placement and shaping.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EifSource {
    /// Scene-unique identifier.
    pub id: EifSourceId,
    /// The signal class (object / channel / HOA).
    pub kind: EifSourceKind,
    /// World-space position (metres).
    pub position: Vec3,
    /// Unit forward (facing) direction used by the directivity model; collapses
    /// to `-Z` when a zero vector is supplied.
    pub forward: Vec3,
    /// Linear overall gain applied to the source signal, clamped to be
    /// non-negative.
    pub gain: Sample,
    /// Frequency-dependent radiation directivity.
    pub directivity: SourceDirectivity,
    /// Distance roll-off curve.
    pub attenuation: Attenuation,
}

impl EifSource {
    /// Builds an object source at `position` with unit gain, an omnidirectional
    /// radiation pattern, and the engine's default distance roll-off.
    #[must_use]
    pub fn object(id: EifSourceId, position: Vec3) -> Self {
        Self {
            id,
            kind: EifSourceKind::Object,
            position,
            forward: Vec3::NEG_Z,
            gain: 1.0,
            directivity: SourceDirectivity::from_sharpness([0.0; 8]),
            attenuation: Attenuation::default(),
        }
    }

    /// Builds a channel (bed feed) source bound to the loudspeaker `label`.
    #[must_use]
    pub fn channel(id: EifSourceId, label: &str, position: Vec3) -> Self {
        let mut src = Self::object(id, position);
        src.kind = EifSourceKind::Channel {
            bed_label: ChannelLabel::new(label),
        };
        src
    }

    /// Builds a scene (HOA) source of the given Ambisonic `order`.
    #[must_use]
    pub fn hoa(id: EifSourceId, order: u8, position: Vec3) -> Self {
        let mut src = Self::object(id, position);
        src.kind = EifSourceKind::Hoa { order };
        src
    }

    /// Replaces the radiation directivity, returning `self` for chaining.
    #[must_use]
    pub fn with_directivity(mut self, directivity: SourceDirectivity) -> Self {
        self.directivity = directivity;
        self
    }

    /// Replaces the distance attenuation, returning `self` for chaining.
    #[must_use]
    pub fn with_attenuation(mut self, attenuation: Attenuation) -> Self {
        self.attenuation = attenuation;
        self
    }

    /// Sets the facing direction, returning `self` for chaining.
    #[must_use]
    pub fn with_forward(mut self, forward: Vec3) -> Self {
        self.forward = forward;
        self
    }

    /// Sets the linear gain, returning `self` for chaining.
    #[must_use]
    pub fn with_gain(mut self, gain: Sample) -> Self {
        self.gain = gain;
        self
    }

    /// The sanitised linear gain (non-negative, finite).
    #[must_use]
    pub fn sanitized_gain(&self) -> Sample {
        if self.gain.is_finite() {
            self.gain.max(0.0)
        } else {
            0.0
        }
    }

    /// The unit facing direction, collapsing a zero or non-finite forward to
    /// `-Z` (the engine's default "straight ahead").
    #[must_use]
    pub fn facing(&self) -> Vec3 {
        let f = self.forward.normalize_or_zero();
        if f == Vec3::ZERO { Vec3::NEG_Z } else { f }
    }

    /// The Ambisonic order for an HOA source, clamped to the engine's supported
    /// range, or `0` for non-HOA sources.
    #[must_use]
    pub fn hoa_order(&self) -> usize {
        match self.kind {
            EifSourceKind::Hoa { order } => {
                (order as usize).min(prism_audio_spatial::hoa::MAX_HOA_ORDER)
            }
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn channel_label_sanitizes_and_truncates() {
        let l = ChannelLabel::new("TopFrontRight");
        assert_eq!(l.as_str(), "TopFront");
        assert_eq!(l.as_str().len(), ChannelLabel::MAX_LEN);
        let empty = ChannelLabel::new("");
        assert!(empty.is_empty());
    }

    #[test]
    fn builders_set_kind() {
        let obj = EifSource::object(EifSourceId(1), Vec3::ZERO);
        assert_eq!(obj.kind, EifSourceKind::Object);
        let ch = EifSource::channel(EifSourceId(2), "L", Vec3::X);
        match ch.kind {
            EifSourceKind::Channel { bed_label } => assert_eq!(bed_label.as_str(), "L"),
            _ => panic!("expected channel kind"),
        }
        let hoa = EifSource::hoa(EifSourceId(3), 2, Vec3::Y);
        assert_eq!(hoa.hoa_order(), 2);
    }

    #[test]
    fn hoa_order_clamps_to_engine_max() {
        let hoa = EifSource::hoa(EifSourceId(4), 9, Vec3::ZERO);
        assert_eq!(hoa.hoa_order(), prism_audio_spatial::hoa::MAX_HOA_ORDER);
        let obj = EifSource::object(EifSourceId(5), Vec3::ZERO);
        assert_eq!(obj.hoa_order(), 0);
    }

    #[test]
    fn facing_falls_back_to_forward_axis() {
        let s = EifSource::object(EifSourceId(6), Vec3::ZERO).with_forward(Vec3::ZERO);
        assert_eq!(s.facing(), Vec3::NEG_Z);
        let s2 = EifSource::object(EifSourceId(7), Vec3::ZERO).with_forward(Vec3::new(0.0, 0.0, 2.0));
        assert!(close(s2.facing().z, 1.0));
    }

    #[test]
    fn gain_is_sanitized() {
        let s = EifSource::object(EifSourceId(8), Vec3::ZERO).with_gain(-3.0);
        assert!(close(s.sanitized_gain(), 0.0));
        let s2 = EifSource::object(EifSourceId(9), Vec3::ZERO).with_gain(Sample::NAN);
        assert!(close(s2.sanitized_gain(), 0.0));
    }
}
