//! HRTF binaural rendering for Prism's next-generation audio engine.
//!
//! This crate turns a mono point source into a two-channel headphone
//! (binaural) signal using measured head-related transfer functions. It sits
//! above [`prism_audio_core`] (DSP kernel, sample type, decibel math) and
//! reuses the listener/emitter geometry and azimuth/elevation convention from
//! [`prism_audio_spatial`], so HRTF selection agrees bit-for-bit with the rest
//! of the spatial pipeline.
//!
//! # Pipeline
//!
//! 1. [`dataset`] holds the in-memory [`HrtfDataset`]: a measurement grid plus
//!    per-ear HRIRs, packed for allocation-free lookup.
//! 2. [`sofa`] loads datasets from external sources via the [`HrirSource`]
//!    trait, including a decoded-[`SofaRecords`] path for SOFA/AES69 data.
//! 3. [`interpolation`] selects and blends the nearest measured HRIRs for an
//!    arbitrary direction, with inter-aural time-difference (ITD) alignment to
//!    avoid comb filtering.
//! 4. [`binaural`] convolves the dry signal with the interpolated HRIR pair
//!    ([`BinauralRenderer`]), using overlap-save block convolution and
//!    click-free crossfades on HRIR updates.
//! 5. [`nearfield`] adds sub-metre corrections: binaural parallax (a different
//!    HRIR direction per ear), per-ear inverse-distance gain, and a
//!    spherical-head proximity shadow.
//!
//! # Classic DSP only
//!
//! Everything here is deterministic, classical signal processing: SOFA/AES69
//! HRIR handling, spherical-geometry interpolation, ITD alignment, block
//! convolution, and a spherical-head near-field model. There is **no machine
//! learning, neural network, or data-driven model of any kind**.
//!
//! # Coordinate convention
//!
//! World and listener-local space are right-handed, matching Bevy and
//! [`prism_audio_spatial`]: `+X` right, `+Y` up, `-Z` forward. Azimuth is
//! `atan2(x, -z)` (front `= 0`, right positive, range `(-pi, pi]`) and
//! elevation is `atan2(y, hypot(x, z))` (range `[-pi/2, pi/2]`).
//!
//! # Real-time contract
//!
//! The hot path - direction interpolation ([`interpolation::interpolate`]),
//! block convolution ([`BinauralRenderer::process_block`]), and near-field
//! resolution ([`nearfield::resolve`]) - is **allocation free, lock free, and
//! panic free** and may run on a device callback thread. File and dataset
//! loading ([`sofa`]) allocates and runs off the audio thread, behind the
//! `std` feature where it touches the filesystem.
//!
//! # Determinism
//!
//! All transcendental and length math routes through [`bevy_math::ops`]
//! (libm-backed) rather than `f32` intrinsics, so rendering is
//! bit-reproducible across targets and can be golden-compared sample-for-
//! sample. This is enforced by the workspace lints.
//!
//! # Feature flags
//!
//! - `std` (default): enables std-backed facilities in the dependency graph
//!   and on-disk SOFA loading. The hot path stays allocation-free either way.
//! - `serialize`: derives serde (de)serialization for the plain description
//!   types (datasets, measurement grids, head/near-field parameters).
//!
//! # Examples
//!
//! Convolve a mono block with an HRIR pair (left ear a unit impulse, right ear
//! delayed by one sample):
//!
//! ```
//! use prism_audio_hrtf::BinauralRenderer;
//!
//! let mut renderer = BinauralRenderer::new(4, 8);
//! renderer.set_hrir_immediate(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
//!
//! let input = [1.0, 0.5, 0.25, 0.0];
//! let mut left = [0.0; 4];
//! let mut right = [0.0; 4];
//! let frames = renderer.process_block(&input, &mut left, &mut right);
//!
//! assert_eq!(frames, 4);
//! assert_eq!(left, input);        // left HRIR passes the signal through
//! assert_eq!(right[1], input[0]); // right HRIR delays it by one sample
//! ```
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. All models
//! (the SOFA/AES69 data conventions, inverse-distance/ITD-aligned HRIR
//! interpolation, overlap-save partitioned convolution, and the
//! spherical-head near-field model) are implemented from standard, publicly
//! documented acoustics and signal-processing knowledge.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod binaural;
pub mod calibration;
pub mod dataset;
pub mod headtracked;
pub mod hoa_binaural;
pub mod interpolation;
pub mod nearfield;
pub mod personalization;
pub mod sofa;
pub mod transaural;

pub use binaural::BinauralRenderer;
pub use calibration::{
    CalibrationState, ElevationCalibration, DEFAULT_ELEVATION_EPSILON, DEFAULT_STEP,
};
pub use dataset::{DatasetError, HrtfDataset, Measurement};
pub use headtracked::{
    local_azimuth, local_elevation, predicted_local_angles, world_to_local_direction,
    HeadLocalAngles, HeadPose, HeadTracker, MAX_PREDICTION_SECONDS,
};
pub use hoa_binaural::{HoaBinauralDecoder, VirtualSpeakerLayout, MAX_VIRTUAL_SPEAKERS};
pub use interpolation::{
    angular_distance, direction_from_angles, estimate_onset_delay, interpolate, InterpolationInfo,
    MAX_NEIGHBORS,
};
pub use nearfield::{
    resolve, HeadGeometry, NearFieldEar, NearFieldParams, NearFieldResult, DEFAULT_HEAD_RADIUS,
};
pub use personalization::{
    measured_distance_bounds, select_best, Anthropometry, HrtfCandidate, Selection,
};
pub use sofa::{
    aes69_to_local, build_dataset, HrirRecord, HrirSource, LoadError, SofaConvention, SofaRecord,
    SofaRecords,
};
pub use transaural::{
    CrosstalkCanceller, CrosstalkParams, DEFAULT_CONTRALATERAL_GAIN, DEFAULT_SOUND_SPEED,
    MIN_CROSSTALK_DELAY,
};
