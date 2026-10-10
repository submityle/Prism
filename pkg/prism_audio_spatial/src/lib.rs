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

pub mod acoustic_format;
pub mod air;
pub mod ambisonics;
pub mod articulation_loss;
pub mod attenuation;
pub mod band_spectrum;
pub mod banded_propagation;
pub mod center_time;
pub mod cone;
pub mod convex_room;
pub mod diffraction;
pub mod diffusion_field;
pub mod direct_to_reverberant_ratio;
pub mod doppler;
pub mod early_reflections;
pub mod echo_criterion;
pub mod echo_density;
pub mod fresnel_transition;
pub mod geometry;
pub mod ground_effect;
pub mod hoa;
pub mod hoa_beamform;
pub mod hoa_decode;
pub mod hoa_rotation;
pub mod initial_time_delay_gap;
pub mod interaural_time_difference;
pub mod late_lateral_sound_level;
pub mod material_library;
pub mod material_spectrum;
pub mod multi_position;
pub mod multiband_spectrum;
pub mod nfc;
pub mod occlusion;
pub mod octave_reverb;
pub mod outdoor_propagation;
pub mod panner;
pub mod portal_graph;
pub mod propagation;
pub mod reflection_clustering;
pub mod reflection_directivity;
pub mod reverb_zones;
pub mod reverberant_field;
pub mod reverberation_spectrum;
pub mod room_acoustics;
pub mod room_clarity;
pub mod room_modes;
pub mod rooms;
pub mod scattering;
pub mod seat_dip_effect;
pub mod sound_strength;
pub mod source_directivity;
pub mod spatial_impression;
pub mod spatializer;
pub mod speech_transmission_index;
pub mod spread;
pub mod stage_support;
pub mod useful_to_detrimental_ratio;
pub mod utd_diffraction;

pub use acoustic_format::AcousticFormat;
pub use air::{absorption_db_per_metre, AirAbsorption, AirAbsorptionNode, AtmosphericConditions};
pub use ambisonics::{
    decode_foa, encode_foa_gains, encode_foa_sample, rotate_foa, FoaEncoderNode, FOA_CHANNELS,
};
pub use articulation_loss::{
    alcons_to_sti, articulation_loss_percent, ArticulationLoss, MAX_ALCONS,
    PEUTZ_CRITICAL_DISTANCE_CONSTANT, R_LIMIT,
};
pub use attenuation::{Attenuation, DistanceModel};
pub use band_spectrum::{
    BandGains, PROPAGATION_BAND_CENTERS, PROPAGATION_BAND_COUNT, PROPAGATION_BAND_EDGES,
    PROPAGATION_BAND_HIGH_HZ, PROPAGATION_BAND_LOW_HZ,
};
pub use banded_propagation::BandedPropagationShaper;
pub use center_time::{center_time_ms, center_time_seconds, CenterTime};
pub use cone::Cone;
pub use convex_room::{
    compute_convex_reflections, ConvexReflections, ConvexRoom, ReflectionPlane, MAX_PLANES,
};
pub use diffraction::Diffraction;
pub use diffusion_field::{DiffusionField, MAX_ECHO_DENSITY};
pub use direct_to_reverberant_ratio::{
    direct_to_reverberant_ratio_db, DirectToReverberantRatio, DIRECT_WINDOW_MS, MAX_DRR_DB,
    NO_DRR_DB,
};
pub use doppler::{doppler_ratio, Doppler, SPEED_OF_SOUND_MPS};
pub use early_reflections::{
    compute_early_reflections, EarlyReflectionRenderer, ReflectionTap, ShoeboxRoom,
    DEFAULT_SOUND_SPEED, MAX_EARLY_REFLECTIONS, MAX_REFLECTION_ORDER,
};
pub use echo_criterion::{
    echo_criterion, EchoCriterion, EchoMode, MUSIC_EXPONENT, MUSIC_THRESHOLD, MUSIC_WINDOW_MS,
    SPEECH_EXPONENT, SPEECH_THRESHOLD, SPEECH_WINDOW_MS,
};
pub use echo_density::{
    mixing_time_ms, normalized_echo_density, EchoDensityProfile, ECHO_DENSITY_WINDOW_MS,
    GAUSSIAN_EXCEEDANCE, MIXING_THRESHOLD, NO_MIXING_TIME_MS,
};
pub use fresnel_transition::{fresnel_integrals, transition, TransitionValue};
pub use geometry::{Emitter, Listener, LocalSource};
pub use ground_effect::GroundEffect;
pub use hoa::{
    acn_index, decode_hoa, encode_hoa, hoa_channel_count, HoaEncoderNode, MAX_HOA_CHANNELS,
    MAX_HOA_ORDER,
};
pub use hoa_beamform::{beam_gains, BeamPattern, Beamformer};
pub use hoa_decode::{max_re_gains, max_re_radius, DecodeBand, DualBandDecoder, SpeakerLayout};
pub use hoa_rotation::{rotate_hoa, HoaRotationMatrix};
pub use initial_time_delay_gap::{
    initial_time_delay_gap_ms, InitialTimeDelayGap, DEFAULT_REFLECTION_THRESHOLD_DB,
};
pub use interaural_time_difference::{
    InterauralTimeDifference, DEFAULT_HEAD_RADIUS_M, DEFAULT_SPEED_OF_SOUND, KUHN_HIGH_FACTOR,
    KUHN_LOW_FACTOR,
};
pub use late_lateral_sound_level::{
    late_lateral_sound_level_db, LateLateralSoundLevel, LATE_LATERAL_START_MS, NO_LATE_LATERAL_DB,
};
pub use material_library::{Material, MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};
pub use material_spectrum::{transmission_from_loss_db, BandedAcousticMaterial};
pub use multi_position::{resolve_multi, MultiPositionMode, PositionInput, MAX_POSITIONS};
pub use multiband_spectrum::{
    MultibandGains, MultibandLayout, MAX_BAND_FREQUENCY_HZ, MIN_BAND_FREQUENCY_HZ,
};
pub use nfc::{NfcCoeffs, NfcFilter, MAX_NFC_ORDER};
pub use occlusion::{
    NullOcclusionQuery, Occlusion, OcclusionFactors, OcclusionNode, OcclusionParams, OcclusionQuery,
};
pub use octave_reverb::OctaveReverb;
pub use outdoor_propagation::OutdoorPropagation;
pub use panner::{Panner, PannerNode, VbapPanner};
pub use portal_graph::{
    route_portals, PortalHop, RoutedPath, MAX_PORTALS, MAX_PORTAL_HOPS, MAX_ROOMS, MAX_ROUTED_PATHS,
};
pub use propagation::{
    diffraction_cutoff_hz, diffraction_gain, edge_path_difference, fresnel_number,
    maekawa_attenuation_db, transmission_gain, AcousticMaterial, FreeFieldBackend, PathKind,
    PropagationBackend, PropagationPath, PropagationSummary, MAX_PROPAGATION_PATHS,
};
pub use reflection_clustering::{
    cluster_taps, ReflectionCluster, ReflectionClusters, CLUSTER_COUNT,
};
pub use reflection_directivity::DirectionalEarlyReflections;
pub use reverb_zones::{
    source_send_gain, AuxBusId, AuxSend, ReverbZone, ReverbZoneField, ZoneShape, MAX_AUX_SENDS,
};
pub use reverberant_field::ReverberantField;
pub use reverberation_spectrum::{
    bass_ratio, octave_band_reverberation_times, treble_ratio, ReverberationSpectrum,
    T30_FIT_LOWER_DB, T30_FIT_UPPER_DB,
};
pub use room_acoustics::{
    critical_distance, eyring_rt60, mean_free_path, millington_sette_rt60, sabine_rt60,
    schroeder_frequency, RoomAcoustics, CRITICAL_DISTANCE_CONSTANT, SABINE_CONSTANT,
    SCHROEDER_CONSTANT,
};
pub use room_clarity::{
    center_time_s, clarity_db, definition, early_decay_time_s, energy_decay_curve,
    reverberation_time_t20_s, reverberation_time_t30_s, RoomClarity, EARLY_LATE_SPLIT_50_MS,
    EARLY_LATE_SPLIT_80_MS, EDT_LOWER_DB, EDT_UPPER_DB, MAX_CLARITY_DB, T20_LOWER_DB, T20_UPPER_DB,
    T30_LOWER_DB, T30_UPPER_DB,
};
pub use room_modes::{ModeKind, RoomMode, RoomModes, MAX_ROOM_MODES};
pub use rooms::{
    obliquity_factor, portal_coupling_gain, room_of, Portal, PortalFrame, Room, RoomId, RoomNetwork,
};
pub use scattering::{
    diffuse_fraction, lambert_directivity, lambert_weight, specular_fraction, ScatteringSpectrum,
    SurfaceScatter,
};
pub use seat_dip_effect::{
    seat_dip_attenuation_db, SeatDipEffect, SeatDipGeometry, MAX_SEAT_DIP_DB,
};
pub use sound_strength::{
    early_sound_strength_db, late_sound_strength_db, sound_strength_db, SoundStrength,
    MIN_STRENGTH_DB,
};
pub use source_directivity::{DirectivityPreset, SourceDirectivity};
pub use spatial_impression::{
    interaural_cross_correlation, lateral_energy_fraction, lateral_energy_fraction_cosine,
    SpatialImpression, EARLY_WINDOW_END_MS, IACC_MAX_LAG_MS, LF_EARLY_START_MS,
};
pub use spatializer::{resolve, SourceDescriptor, SpatialParams};
pub use speech_transmission_index::{
    speech_transmission_index, SpeechTransmissionIndex, StiRating, APPARENT_SNR_LIMIT_DB,
    MALE_ALPHA, MALE_BETA, MODULATION_COUNT, MODULATION_FREQS_HZ, OCTAVE_CENTERS_HZ, OCTAVE_COUNT,
};
pub use spread::{
    compute_spread_gains, spread_taps, Spread, SpreadParams, SpreadTap, MAX_SPREAD_TAPS,
};
pub use stage_support::{
    stage_support_early_db, stage_support_late_db, StageSupport, NO_SUPPORT_DB,
};
pub use useful_to_detrimental_ratio::{
    useful_to_detrimental_ratio_db, UsefulToDetrimental, NO_USEFUL_RATIO_DB,
};
pub use utd_diffraction::UtdWedge;
