//! Spatial audio layer for Prism's next-generation audio engine.
//!
//! This crate builds on the backend-neutral DSP kernel in
//! [`prism_audio_core`], adding the geometry and perceptual models that turn a
//! world-space sound source into signals a listener actually hears: distance
//! attenuation, directional cones, Doppler pitch shift, multi-layout panning,
//! frequency-dependent air absorption, and Ambisonic encode/decode.
//!
//! # Layering
//!
//! Everything here consumes the primitives established once in [`geometry`]:
//! the [`Listener`] (point of reception), the [`Emitter`] (a world-space
//! source), and the listener-relative [`LocalSource`] produced by
//! [`Listener::localize`]. Downstream modules operate on the `LocalSource`
//! (direction, distance, radial velocity) rather than re-deriving geometry, so
//! the coordinate convention is defined in exactly one place.
//!
//! # Coordinate convention
//!
//! World space is right-handed, matching Bevy: `+X` right, `+Y` up, `-Z`
//! forward. See [`geometry`] for the full description and the listener-local
//! frame.
//!
//! # Real-time contract
//!
//! Pure geometry/DSP math in this crate is **allocation free, lock free, and
//! panic free**, so nodes derived from it may run on a device callback thread.
//! Any authoring-time description data (poses, attenuation/cone descriptors)
//! that allocates lives outside the hot path.
//!
//! # Determinism
//!
//! All transcendental and length math routes through [`bevy_math::ops`]
//! (libm-backed) rather than `f32` intrinsics, so spatialisation is
//! bit-reproducible across targets and can be golden-compared sample-for-
//! sample. This is enforced by the workspace lints.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. All models (distance
//! attenuation curves, cone gains, the Doppler ratio, VBAP/pairwise panning,
//! ISO 9613-1 air absorption, and Ambisonic ACN/SN3D encoding) are implemented
//! from standard, publicly documented acoustics and signal-processing
//! knowledge.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod air;
pub mod ambisonics;
pub mod attenuation;
pub mod cone;
pub mod doppler;
pub mod geometry;
pub mod multi_position;
pub mod occlusion;
pub mod panner;
pub mod propagation;
pub mod rooms;
pub mod spatializer;
pub mod spread;

pub use air::{AirAbsorption, AirAbsorptionNode, AtmosphericConditions, absorption_db_per_metre};
pub use ambisonics::{
    FOA_CHANNELS, FoaEncoderNode, decode_foa, encode_foa_gains, encode_foa_sample, rotate_foa,
};
pub use attenuation::{Attenuation, DistanceModel};
pub use cone::Cone;
pub use doppler::{Doppler, SPEED_OF_SOUND_MPS, doppler_ratio};
pub use geometry::{Emitter, Listener, LocalSource};
pub use multi_position::{
    MAX_POSITIONS, MultiPositionMode, PositionInput, resolve_multi,
};
pub use occlusion::{
    NullOcclusionQuery, Occlusion, OcclusionFactors, OcclusionNode, OcclusionParams, OcclusionQuery,
};
pub use panner::{Panner, PannerNode, VbapPanner};
pub use propagation::{
    AcousticMaterial, FreeFieldBackend, MAX_PROPAGATION_PATHS, PathKind, PropagationBackend,
    PropagationPath, PropagationSummary, diffraction_cutoff_hz, diffraction_gain,
    edge_path_difference, fresnel_number, maekawa_attenuation_db, transmission_gain,
};
pub use rooms::{
    Portal, Room, RoomId, RoomNetwork, obliquity_factor, portal_coupling_gain, room_of,
};
pub use spatializer::{SourceDescriptor, SpatialParams, resolve};
pub use spread::{MAX_SPREAD_TAPS, Spread, SpreadParams, SpreadTap, compute_spread_gains, spread_taps};
