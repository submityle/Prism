//! The shared control-rate uniform both acoustics kernels read.
//!
//! A single [`GeometryParams`] record is uploaded once per dispatch and bound
//! at `@binding(0)` of both the direct and reflection pipelines. It carries the
//! batch sizes the kernels loop over plus the handful of scalars the `CPU`
//! backend reads from its [`GeometricConfig`](prism_audio_geometry::GeometricConfig):
//! the audibility floor, the surface epsilon, the speed of sound, and whether a
//! shadowed direct path still leaks a transmitted arrival.
//!
//! The struct is `#[repr(C)]` plain-old-data padded to a 16-byte multiple so
//! the `std430` uniform layout each shader declares matches it field-for-field.
//!
//! # Provenance
//!
//! Original work; plain struct packing for a compute uniform; no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio
//! source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Built by [`crate::backend`] from a
//! [`GeometricConfig`](prism_audio_geometry::GeometricConfig) and the uploaded
//! [`GpuScene`](crate::GpuScene); bound by [`crate::direct`] and
//! [`crate::reflection`] and consumed by their `CPU` twins.

use bytemuck::{Pod, Zeroable};

/// Control-rate parameters shared by the direct and reflection kernels.
///
/// Mirrors the subset of
/// [`GeometricConfig`](prism_audio_geometry::GeometricConfig) the device
/// arithmetic needs, plus the batch sizes the kernels iterate over.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GeometryParams {
    /// Number of real triangles in the scene the kernels loop over.
    pub(crate) triangle_count: u32,
    /// Number of listener/emitter queries in the batch.
    pub(crate) query_count: u32,
    /// Linear-gain floor below which an arrival is dropped.
    pub(crate) min_gain: f32,
    /// Surface offset (metres) applied when re-launching rays off geometry.
    pub(crate) surface_epsilon: f32,
    /// Speed of sound (metres per second) used to convert distance to delay.
    pub(crate) speed_of_sound: f32,
    /// Non-zero when a shadowed direct path still contributes a transmitted
    /// arrival.
    pub(crate) transmission_enabled: u32,
    /// Padding to a 16-byte stride (first slot).
    pub(crate) _pad0: u32,
    /// Padding to a 16-byte stride (second slot).
    pub(crate) _pad1: u32,
}
