//! Loudspeaker-array rendering for arbitrary physical layouts (design
//! section 49.3).
//!
//! This module turns the engine's source-position truth into speaker feeds for
//! irregular 2D and 3D arrays. It is deterministic classic signal processing:
//! amplitude panning, matrix decoding, and geometric delay/gain driving
//! functions, with no AI/ML and no second acoustic model.
//!
//! The submodules split the problem by concept:
//!
//! * [`layout`] - the arbitrary array geometry ([`layout::ArrayLayout`]).
//! * [`vbap`] - vector-base amplitude panning over an [`layout::ArrayLayout`].
//! * [`allrad`] - the all-round ambisonic decoder (virtual loudspeakers plus
//!   amplitude-panning resampling, dual band).
//! * [`wfs`] - wave-field-synthesis driving functions (per-speaker delay and
//!   gain for point and plane-wave virtual sources).
//! * [`beamforming`] - delay-and-sum beam steering with a tapered aperture.
//! * [`decode`] - the unified decode entry point: capability probing, mode
//!   selection, and graceful fallback to a bed or binaural output.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the published algorithms of VBAP (Pulkki 1997), the
//! all-round ambisonic decoder (Zotter and Frank 2012), wave-field synthesis
//! (Berkhout 1988), and classic delay-and-sum beamforming.
//!
//! # Relationship
//!
//! Consumes the same source-position truth as [`crate::eif`] and
//! [`crate::adm`]. Reuses `prism_audio_object::pan::vbap` for the amplitude
//! panning kernel, `prism_audio_spatial::hoa_decode` for the dual-band
//! Ambisonic decode, and `prism_audio_object::bed` for the regular layouts
//! used as bootstrap geometry and graceful-fallback targets.

pub mod allrad;
pub mod beamforming;
pub mod decode;
pub mod layout;
pub mod vbap;
pub mod wfs;

pub use allrad::{AllRadDecoder, VirtualSpeakerGrid};
pub use beamforming::{BeamTap, Beamformer};
pub use decode::{ArrayCapabilities, ArrayRenderMode, ArrayRenderer, RenderDecision};
pub use layout::{ArrayLayout, ArraySpeaker, DEFAULT_RADIUS_M, PLANAR_EPSILON};
pub use vbap::VbapArrayPanner;
pub use wfs::{WfsDriver, WfsSource, WfsSourceKind, WfsTap};
