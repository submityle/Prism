//! Geometric acoustics propagation backend for Prism's next-generation audio
//! engine.
//!
//! The spatial crate ([`prism_audio_spatial`]) defines the pluggable
//! [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//! trait and ships a [`FreeFieldBackend`](prism_audio_spatial::propagation::FreeFieldBackend)
//! that models an open field. This crate supplies the *real* geometric backend:
//! given a triangle-mesh [`scene`] with per-triangle acoustic materials, it
//! ray-traces the actual geometry to resolve the arrivals a wave takes from an
//! emitter to a listener and reports them as the same bounded
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath) set
//! the spatial voice already consumes.
//!
//! The spatial crate deliberately keeps itself free of any geometry/physics
//! dependency; this crate is that "physics-aware layer". It leans on
//! [`prism_physics_geometry`] for the triangle mesh and its
//! bounding-volume-hierarchy ray cast, and reuses the spatial crate's classic
//! acoustics helpers (transmission loss, Fresnel/Maekawa diffraction, reflection
//! coefficients) rather than inventing a parallel model.
//!
//! # Module map
//!
//! - [`config`] holds [`config::GeometricConfig`], the control-rate budget and
//!   feature switches (reflection order, ray/edge budgets, audibility floor).
//! - [`material_map`] holds [`material_map::MaterialTable`], the per-triangle
//!   acoustic-material assignment with a default fallback that leaves no hole.
//! - [`scene`] holds [`scene::AcousticScene`], the triangle mesh plus its
//!   material table, exposing BVH-accelerated ray casts and marched segment
//!   hits used by every path builder.
//! - [`direct_path`] resolves the line-of-sight arrival: BVH-accelerated
//!   transmission accumulation through intervening partitions, yielding the
//!   direct/transmission [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)
//!   and the [`OcclusionFactors`](prism_audio_spatial::occlusion::OcclusionFactors)
//!   the occlusion model consumes.
//! - [`reflection_path`] resolves first-order specular reflections with the
//!   image-source method: mirror the source across each reflector, validate the
//!   reflection point lies on the face, and check both sub-segments are clear.
//! - [`diffraction_path`] resolves edge diffraction when the direct path is
//!   shadowed: it finds the least-detour silhouette edge and applies the
//!   Maekawa barrier model via the spatial crate's Fresnel helpers.
//! - [`backend`] assembles the above into [`backend::GeometricBackend`], the
//!   [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//!   implementation that fills the caller's bounded path buffer.
//!
//! # Determinism
//!
//! Every transcendental routes through [`bevy_math::ops`] (libm-backed) exactly
//! as the spatial acoustics helpers do, so the resolved delays, gains, and
//! corners are bit-reproducible across targets and can be golden-compared.
//! The backend allocates only control-rate scratch (never on the real-time
//! audio thread) and cannot panic.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements [`prism_audio_spatial::propagation::PropagationBackend`] on top of
//! [`prism_physics_geometry`]'s triangle mesh and BVH ray cast, reusing the
//! spatial crate's `propagation`/`occlusion` acoustics. It is the real
//! geometric sibling of the spatial crate's `FreeFieldBackend` and `RoomNetwork`
//! reference backends.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod config;
pub mod diffraction_path;
pub mod direct_path;
pub mod material_map;
pub mod reflection_path;
pub mod scene;

pub use backend::GeometricBackend;
pub use config::GeometricConfig;
pub use material_map::MaterialTable;
pub use scene::{AcousticScene, SceneBuildError, SceneRayHit};
