//! Unified array decode entry point with graceful degradation (design
//! section 49.3).
//!
//! This module is the front door for array rendering. It probes an
//! [`ArrayLayout`] into [`ArrayCapabilities`], picks a concrete
//! [`ArrayRenderMode`] from a requested mode and those capabilities, and falls
//! back to a fixed bed or a binaural pair when a physical array is missing or
//! too sparse. Mode selection never panics: an empty or absent array always
//! resolves to a usable fallback so object rendering can continue.
//!
//! Object point sources are always rendered with vector-base amplitude panning
//! over whichever target the decision selects (the physical array, a fallback
//! bed, or a stereo binaural pair). The all-round decoder, wave-field driver,
//! and beamformer handle their own field and position inputs through their
//! dedicated modules; this entry point routes to them by reporting the chosen
//! mode and supplies the universal object-panning path directly.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Combines only the published VBAP algorithm (Pulkki 1997) with deterministic
//! capability-based routing.
//!
//! # Relationship
//!
//! Sits above [`crate::array::vbap`], [`crate::array::allrad`],
//! [`crate::array::wfs`], and [`crate::array::beamforming`], and uses
//! `prism_audio_object::bed::BedLayout` for the fallback targets. It is the
//! array-stage analogue of the fixed downmix in design section 48: when no
//! physical array is present, rendering degrades gracefully rather than
//! stopping.

use alloc::vec::Vec;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_object::bed::BedLayout;

use crate::array::layout::ArrayLayout;
use crate::array::vbap::VbapArrayPanner;

/// The minimum directional speaker count for a layout to count as dense enough
/// for an all-round ambisonic decode.
pub const DENSE_SPEAKER_THRESHOLD: usize = 8;

/// A concrete rendering mode selected for a layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ArrayRenderMode {
    /// Vector-base amplitude panning of objects.
    Vbap,
    /// All-round ambisonic decoding of HOA scene audio.
    AllRad,
    /// Wave-field synthesis of positioned virtual sources.
    Wfs,
    /// Delay-and-sum beam steering.
    Beamforming,
}

/// A probe of what an [`ArrayLayout`] can support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayCapabilities {
    /// Whether any physical speaker is present.
    pub has_physical_array: bool,
    /// The total number of layout slots.
    pub speaker_count: usize,
    /// The number of directional (non-LFE) speakers.
    pub directional_count: usize,
    /// Whether the directional count meets [`DENSE_SPEAKER_THRESHOLD`].
    pub is_dense: bool,
    /// Whether every directional speaker lies in the horizontal plane.
    pub is_planar: bool,
}

impl ArrayCapabilities {
    /// Probes `layout`.
    #[must_use]
    pub fn from_layout(layout: &ArrayLayout) -> Self {
        let directional_count = layout.directional_count();
        Self {
            has_physical_array: !layout.is_empty(),
            speaker_count: layout.len(),
            directional_count,
            is_dense: directional_count >= DENSE_SPEAKER_THRESHOLD,
            is_planar: layout.is_planar(),
        }
    }
}

/// The outcome of mode selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderDecision {
    /// Render on the physical array with the given mode.
    Array(ArrayRenderMode),
    /// No usable array: fall back to a fixed bed layout.
    FallbackBed(BedLayout),
    /// No usable array and too few channels: fall back to a binaural pair.
    FallbackBinaural,
}

/// Chooses a [`RenderDecision`] from `capabilities` and a `requested` mode.
///
/// A missing physical array always yields [`RenderDecision::FallbackBinaural`].
/// A present but too-sparse array degrades to a lower-requirement mode or a
/// fallback bed. This function never panics.
#[must_use]
pub fn choose_mode(capabilities: ArrayCapabilities, requested: ArrayRenderMode) -> RenderDecision {
    if !capabilities.has_physical_array || capabilities.speaker_count == 0 {
        return RenderDecision::FallbackBinaural;
    }
    match requested {
        ArrayRenderMode::AllRad => {
            if capabilities.is_dense {
                RenderDecision::Array(ArrayRenderMode::AllRad)
            } else if capabilities.directional_count >= 3 {
                RenderDecision::Array(ArrayRenderMode::Vbap)
            } else {
                fallback_for(capabilities)
            }
        }
        ArrayRenderMode::Vbap => {
            if capabilities.directional_count >= 2 {
                RenderDecision::Array(ArrayRenderMode::Vbap)
            } else {
                fallback_for(capabilities)
            }
        }
        ArrayRenderMode::Wfs => {
            if capabilities.directional_count >= 2 {
                RenderDecision::Array(ArrayRenderMode::Wfs)
            } else {
                fallback_for(capabilities)
            }
        }
        ArrayRenderMode::Beamforming => {
            if capabilities.directional_count >= 2 {
                RenderDecision::Array(ArrayRenderMode::Beamforming)
            } else {
                fallback_for(capabilities)
            }
        }
    }
}

/// The fallback when an array is present but cannot serve the requested mode.
fn fallback_for(capabilities: ArrayCapabilities) -> RenderDecision {
    if capabilities.directional_count >= 2 {
        RenderDecision::FallbackBed(BedLayout::Stereo)
    } else {
        RenderDecision::FallbackBinaural
    }
}

/// A ready-to-use object renderer that applies the chosen decision.
///
/// Whatever the decision, [`ArrayRenderer::render_object`] returns a full
/// gain vector for the target channel set (physical array, fallback bed, or
/// binaural pair), so callers always get a valid, non-panicking output.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::array::layout::ArrayLayout;
/// use prism_audio_immersive::array::decode::{ArrayRenderMode, ArrayRenderer, RenderDecision};
///
/// // No physical array: rendering degrades to a binaural pair rather than failing.
/// let renderer = ArrayRenderer::new(&ArrayLayout::new(), ArrayRenderMode::Wfs);
/// assert_eq!(renderer.decision(), RenderDecision::FallbackBinaural);
/// let gains = renderer.render_object(Vec3::new(0.0, 0.0, -1.0));
/// assert_eq!(gains.len(), renderer.output_channels());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayRenderer {
    capabilities: ArrayCapabilities,
    decision: RenderDecision,
    panner: VbapArrayPanner,
}

impl ArrayRenderer {
    /// Builds a renderer for `layout` and a `requested` mode.
    #[must_use]
    pub fn new(layout: &ArrayLayout, requested: ArrayRenderMode) -> Self {
        let capabilities = ArrayCapabilities::from_layout(layout);
        let decision = choose_mode(capabilities, requested);
        let panner = match decision {
            RenderDecision::Array(_) => VbapArrayPanner::new(layout),
            RenderDecision::FallbackBed(bed) => {
                VbapArrayPanner::new(&ArrayLayout::from_bed(bed))
            }
            RenderDecision::FallbackBinaural => {
                VbapArrayPanner::new(&ArrayLayout::from_bed(BedLayout::Stereo))
            }
        };
        Self {
            capabilities,
            decision,
            panner,
        }
    }

    /// The probed capabilities.
    #[must_use]
    pub fn capabilities(&self) -> ArrayCapabilities {
        self.capabilities
    }

    /// The selected decision.
    #[must_use]
    pub fn decision(&self) -> RenderDecision {
        self.decision
    }

    /// The number of output channels this renderer produces.
    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.panner.slot_count()
    }

    /// Renders an object at `direction` to one gain per output channel.
    #[must_use]
    pub fn render_object(&self, direction: Vec3) -> Vec<Sample> {
        self.panner.gains(direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use prism_audio_object::bed::direction_from_angles;

    use crate::array::layout::{ArrayLayout, ArraySpeaker};

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn empty_layout_falls_back_to_binaural() {
        let caps = ArrayCapabilities::from_layout(&ArrayLayout::new());
        assert!(!caps.has_physical_array);
        let decision = choose_mode(caps, ArrayRenderMode::AllRad);
        assert_eq!(decision, RenderDecision::FallbackBinaural);
    }

    #[test]
    fn sparse_array_degrades_allrad_to_vbap() {
        let mut layout = ArrayLayout::new();
        for i in 0..5 {
            let azimuth = i as Sample * 72.0;
            layout.push(ArraySpeaker::from_direction(
                i,
                direction_from_angles(azimuth, 0.0),
                2.0,
                "S",
            ));
        }
        let caps = ArrayCapabilities::from_layout(&layout);
        assert!(!caps.is_dense);
        assert!(caps.directional_count >= 3);
        let decision = choose_mode(caps, ArrayRenderMode::AllRad);
        assert_eq!(decision, RenderDecision::Array(ArrayRenderMode::Vbap));
    }

    #[test]
    fn dense_array_keeps_allrad() {
        let mut layout = ArrayLayout::new();
        for i in 0..12 {
            let azimuth = i as Sample * 30.0;
            layout.push(ArraySpeaker::from_direction(
                i,
                direction_from_angles(azimuth, 0.0),
                2.0,
                "S",
            ));
        }
        let caps = ArrayCapabilities::from_layout(&layout);
        assert!(caps.is_dense);
        let decision = choose_mode(caps, ArrayRenderMode::AllRad);
        assert_eq!(decision, RenderDecision::Array(ArrayRenderMode::AllRad));
    }

    #[test]
    fn renderer_always_produces_output() {
        let renderer = ArrayRenderer::new(&ArrayLayout::new(), ArrayRenderMode::Wfs);
        assert_eq!(renderer.decision(), RenderDecision::FallbackBinaural);
        let gains = renderer.render_object(direction_from_angles(0.0, 0.0));
        assert_eq!(gains.len(), renderer.output_channels());
        let energy: Sample = gains.iter().map(|&g| g * g).sum();
        assert!(close(energy, 1.0));
    }

    #[test]
    fn physical_array_renders_on_array() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
        let renderer = ArrayRenderer::new(&layout, ArrayRenderMode::Vbap);
        assert_eq!(
            renderer.decision(),
            RenderDecision::Array(ArrayRenderMode::Vbap)
        );
        assert_eq!(renderer.output_channels(), layout.len());
    }
}
