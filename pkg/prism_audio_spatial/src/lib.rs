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
pub mod diffusion_field;
pub mod doppler;
pub mod early_reflections;
pub mod geometry;
pub mod hoa;
pub mod hoa_beamform;
pub mod hoa_decode;
pub mod hoa_rotation;
pub mod material_library;
pub mod multi_position;
pub mod nfc;
pub mod occlusion;
pub mod octave_reverb;
pub mod panner;
pub mod portal_graph;
pub mod propagation;
pub mod reverb_zones;
pub mod room_acoustics;
pub mod rooms;
pub mod scattering;
pub mod spatializer;
pub mod spread;

pub use air::{AirAbsorption, AirAbsorptionNode, AtmosphericConditions, absorption_db_per_metre};
pub use ambisonics::{
    FOA_CHANNELS, FoaEncoderNode, decode_foa, encode_foa_gains, encode_foa_sample, rotate_foa,
};
pub use attenuation::{Attenuation, DistanceModel};
pub use cone::Cone;
pub use diffusion_field::{DiffusionField, MAX_ECHO_DENSITY};
pub use doppler::{Doppler, SPEED_OF_SOUND_MPS, doppler_ratio};
pub use early_reflections::{
    DEFAULT_SOUND_SPEED, EarlyReflectionRenderer, MAX_EARLY_REFLECTIONS, MAX_REFLECTION_ORDER,
    ReflectionTap, ShoeboxRoom, compute_early_reflections,
};
pub use geometry::{Emitter, Listener, LocalSource};
pub use hoa::{
    HoaEncoderNode, MAX_HOA_CHANNELS, MAX_HOA_ORDER, acn_index, decode_hoa, encode_hoa,
    hoa_channel_count,
};
pub use hoa_beamform::{BeamPattern, Beamformer, beam_gains};
pub use hoa_decode::{DecodeBand, DualBandDecoder, SpeakerLayout, max_re_gains, max_re_radius};
pub use hoa_rotation::{HoaRotationMatrix, rotate_hoa};
pub use material_library::{
    Material, MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT,
};
pub use multi_position::{
    MAX_POSITIONS, MultiPositionMode, PositionInput, resolve_multi,
};
pub use nfc::{MAX_NFC_ORDER, NfcCoeffs, NfcFilter};
pub use occlusion::{
    NullOcclusionQuery, Occlusion, OcclusionFactors, OcclusionNode, OcclusionParams, OcclusionQuery,
};
pub use octave_reverb::OctaveReverb;
pub use panner::{Panner, PannerNode, VbapPanner};
pub use portal_graph::{
    MAX_PORTAL_HOPS, MAX_PORTALS, MAX_ROOMS, MAX_ROUTED_PATHS, PortalHop, RoutedPath,
    route_portals,
};
pub use propagation::{
    AcousticMaterial, FreeFieldBackend, MAX_PROPAGATION_PATHS, PathKind, PropagationBackend,
    PropagationPath, PropagationSummary, diffraction_cutoff_hz, diffraction_gain,
    edge_path_difference, fresnel_number, maekawa_attenuation_db, transmission_gain,
};
pub use reverb_zones::{
    AuxBusId, AuxSend, MAX_AUX_SENDS, ReverbZone, ReverbZoneField, ZoneShape, source_send_gain,
};
pub use room_acoustics::{
    CRITICAL_DISTANCE_CONSTANT, RoomAcoustics, SABINE_CONSTANT, SCHROEDER_CONSTANT,
    critical_distance, eyring_rt60, mean_free_path, millington_sette_rt60, sabine_rt60,
    schroeder_frequency,
};
pub use rooms::{
    Portal, Room, RoomId, RoomNetwork, obliquity_factor, portal_coupling_gain, room_of,
};
pub use scattering::{
    ScatteringSpectrum, SurfaceScatter, diffuse_fraction, lambert_directivity, lambert_weight,
    specular_fraction,
};
pub use spatializer::{SourceDescriptor, SpatialParams, resolve};
pub use spread::{MAX_SPREAD_TAPS, Spread, SpreadParams, SpreadTap, compute_spread_gains, spread_taps};
