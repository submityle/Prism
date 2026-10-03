//! `GPU`-accelerated geometric acoustics for Prism's next-generation audio
//! engine (design section 31).
//!
//! The `CPU` [`prism_audio_geometry`] crate ray-traces a triangle-mesh scene to
//! resolve the arrivals a wave takes from an emitter to a listener. For a scene
//! queried against many sources at once, that per-source trace is embarrassingly
//! parallel: this crate ports the two most expensive stages to a `wgpu` compute
//! backend while keeping the `CPU` crate the source of truth.
//!
//! # What runs on the `GPU`
//!
//! - **Direct path** ([`direct`]): one invocation per query marches the
//!   listener-to-emitter segment through the scene, folding each partition's
//!   transmission gain, and emits the direct/transmission arrival plus the
//!   occlusion it implies. This mirrors
//!   [`resolve_direct`](prism_audio_geometry::direct_path::resolve_direct)
//!   exactly.
//! - **Specular reflections** ([`reflection`]): one invocation per
//!   (query, triangle) pair runs the image-source construction and validates the
//!   bounce, mirroring
//!   [`resolve_reflections`](prism_audio_geometry::reflection_path::resolve_reflections).
//!   The host merges, de-duplicates, sorts loudest-first, and caps the survivors.
//!
//! # What stays on the `CPU` (honest division of labour)
//!
//! Edge diffraction remains the `CPU` crate's responsibility: it is a
//! least-detour search that does not parallelise cleanly per query and is a
//! small fraction of the per-source cost. This crate therefore resolves
//! `direct + reflection` on the device and defers diffraction to
//! [`prism_audio_geometry`]. The golden comparison in [`backend`] is run on
//! scenes whose direct line is clear, so the `CPU` diffraction stage is empty
//! and the two agree arrival-for-arrival.
//!
//! # Validation
//!
//! Every kernel ships with a `CPU` twin ([`direct::cpu_direct`],
//! [`reflection::cpu_reflection`]) that reproduces the device arithmetic on the
//! host; the tests assert the device result tracks its twin within a tight
//! floating-point tolerance, and that the assembled [`backend::GpuGeometryBackend`]
//! matches [`prism_audio_geometry::GeometricBackend`] on clear-line scenes.
//! `GPU` tests acquire a device through [`GpuContext::try_headless`] and skip
//! gracefully when no adapter is available.
//!
//! # Provenance
//!
//! Original work; standard `wgpu` compute and the classic image-source /
//! ray-march acoustics already implemented on the `CPU` in
//! [`prism_audio_geometry`]; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! A device-side sibling of [`prism_audio_geometry`]: it consumes the same
//! [`AcousticScene`](prism_audio_geometry::AcousticScene) and
//! [`GeometricConfig`](prism_audio_geometry::GeometricConfig), the same spatial
//! [`Listener`](prism_audio_spatial::geometry::Listener) /
//! [`Emitter`](prism_audio_spatial::geometry::Emitter), and produces the same
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath) set.

#![forbid(unsafe_code)]

extern crate alloc;

mod backend;
mod buffer;
mod context;
mod direct;
mod layout;
mod params;
mod propagation;
mod query;
mod reflection;
mod scene_upload;
mod trace;

pub use backend::{GpuGeometryBackend, ResolvedQuery};
pub use context::GpuContext;
pub use propagation::GpuPropagationBackend;
pub use scene_upload::GpuScene;
