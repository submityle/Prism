//! Per-query inputs and per-kernel result records exchanged with the device.
//!
//! Each acoustics dispatch processes a batch of listener/emitter pairs. One
//! [`GpuQuery`] packs a single pair in the exact convention the `CPU`
//! [`Listener::localize`](prism_audio_spatial::geometry::Listener::localize)
//! uses: a world-space listener position, the listener orientation as a unit
//! quaternion (xyzw), and a world-space emitter position. The kernels return a
//! [`GpuDirectResult`] per query and, for the reflection stage, a
//! [`GpuReflectionCandidate`] per (query, triangle) pair; the host then merges
//! them exactly as the `CPU` backend does.
//!
//! Every record is `#[repr(C)]` plain-old-data with explicit padding so the
//! `std430` storage layout the shaders declare matches these structs
//! field-for-field, and a never-written result slot keeps its zero bytes, which
//! each record decodes as "no arrival".
//!
//! # Provenance
//!
//! Original work; plain struct packing for a compute buffer; no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio
//! source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Built from [`prism_audio_spatial::geometry::Listener`] and
//! [`prism_audio_spatial::geometry::Emitter`]; bound by [`crate::direct`] and
//! [`crate::reflection`], and decoded by [`crate::backend`] into the spatial
//! crate's [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)
//! set.

use bytemuck::{Pod, Zeroable};
use prism_audio_spatial::geometry::{Emitter, Listener};

/// Discriminant stored in [`GpuDirectResult::kind`] for a clear line-of-sight
/// arrival (`PathKind::Direct`).
pub(crate) const DIRECT_KIND_DIRECT: u32 = 0;

/// Discriminant stored in [`GpuDirectResult::kind`] for a transmitted arrival
/// that crossed one or more partitions (`PathKind::Transmission`).
pub(crate) const DIRECT_KIND_TRANSMISSION: u32 = 1;

/// One listener/emitter pair as the kernels see it.
///
/// The orientation is the listener's local-to-world unit quaternion stored
/// `[x, y, z, w]`; the kernels apply its inverse to a world direction to
/// express it in the listener frame, matching
/// [`Listener::localize`](prism_audio_spatial::geometry::Listener::localize).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuQuery {
    /// Listener world-space position (xyz; w padding).
    pub(crate) listener_pos: [f32; 4],
    /// Listener local-to-world unit quaternion `[x, y, z, w]`.
    pub(crate) listener_orient: [f32; 4],
    /// Emitter world-space position (xyz; w padding).
    pub(crate) emitter_pos: [f32; 4],
}

impl GpuQuery {
    /// Packs one `listener`/`emitter` pair into the device query record.
    #[must_use]
    pub(crate) fn new(listener: &Listener, emitter: &Emitter) -> GpuQuery {
        let p = listener.position;
        let q = listener.orientation;
        let e = emitter.position;
        GpuQuery {
            listener_pos: [p.x, p.y, p.z, 0.0],
            listener_orient: [q.x, q.y, q.z, q.w],
            emitter_pos: [e.x, e.y, e.z, 0.0],
        }
    }
}

/// The direct-path result the direct kernel writes for one query.
///
/// Mirrors the `CPU`
/// [`DirectResult`](prism_audio_geometry::direct_path::DirectResult): the
/// listener-local arrival direction, its delay/gain/cutoff, the base distance
/// secondary arrivals attenuate against, the obstruction/occlusion the
/// occlusion model consumes, the path kind
/// ([`DIRECT_KIND_DIRECT`]/[`DIRECT_KIND_TRANSMISSION`]), whether the arrival
/// is audible, and the normalised per-band colour the host pairs with the
/// scalar gain.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuDirectResult {
    /// Listener-local unit arrival direction (xyz; w padding).
    pub(crate) direction: [f32; 4],
    /// Propagation delay in seconds (`base_distance / speed of sound`).
    pub(crate) delay_seconds: f32,
    /// Linear gain relative to the free-field direct arrival at `base_distance`.
    pub(crate) gain: f32,
    /// Low-pass corner (Hz); the direct stage is always full band.
    pub(crate) cutoff_hz: f32,
    /// Straight-line emitter-to-listener distance (metres).
    pub(crate) base_distance: f32,
    /// Obstruction factor fed to the occlusion model, in `[0, 1]`.
    pub(crate) obstruction: f32,
    /// Occlusion factor fed to the occlusion model, in `[0, 1]`.
    pub(crate) occlusion: f32,
    /// Path kind discriminant ([`DIRECT_KIND_DIRECT`] or
    /// [`DIRECT_KIND_TRANSMISSION`]).
    pub(crate) kind: u32,
    /// Non-zero when the arrival should be rendered.
    pub(crate) audible: u32,
    /// Per-band relative colour of the arrival, normalised so the brightest
    /// band is unity (xyz = low/mid/high; w padding). Multiplying by
    /// [`gain`](Self::gain) recovers the absolute per-band transmission, so a
    /// direct arrival carries [`BandGains::UNITY`](prism_audio_spatial::BandGains)
    /// colour while a transmitted arrival keeps the surface's spectral tilt.
    pub(crate) bands: [f32; 4],
}

/// One (query, triangle) specular reflection candidate the reflection kernel
/// writes.
///
/// A `valid` of zero marks a slot the kernel rejected (wrong side, off-face,
/// blocked sub-segment, or sub-floor gain); the host skips those and merges the
/// survivors exactly as the `CPU` reflection stage does.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuReflectionCandidate {
    /// Listener-local unit reflection direction (xyz; w padding).
    pub(crate) direction: [f32; 4],
    /// Propagation delay in seconds (`path_length / speed of sound`).
    pub(crate) delay_seconds: f32,
    /// Linear reflection gain in `[0, 1]`, already spreading-scaled.
    pub(crate) gain: f32,
    /// Non-zero when this candidate is a real, audible reflection.
    pub(crate) valid: u32,
    /// Padding so the trailing `bands` vector begins on its 16-byte boundary.
    pub(crate) _pad: f32,
    /// Per-band relative colour of the reflection, normalised so the brightest
    /// band is unity (xyz = low/mid/high; w padding). Multiplying by
    /// [`gain`](Self::gain) recovers the absolute per-band specular reflection
    /// coefficient after the spreading and scattering weights are applied.
    pub(crate) bands: [f32; 4],
}
