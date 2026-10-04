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
//! - [`air_absorption`] folds frequency-dependent atmospheric air
//!   absorption (ISO 9613-1) into every resolved arrival, rolling off the
//!   highs with travelled distance; opt-in through
//!   [`config::GeometricConfig::with_air_absorption`].
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
//! - [`higher_order_reflection`] resolves the second- and higher-order specular
//!   bounces with the recursive image-source method, complementing (never
//!   duplicating) the single bounce from [`reflection_path`]; enabled by
//!   [`config::GeometricConfig::with_max_reflection_order`].
//! - [`diffraction_edges`] distils a mesh into its diffracting edges and the
//!   least-detour and wedge primitives shared by the diffraction resolvers.
//! - [`diffraction_path`] resolves edge diffraction when the direct path is
//!   shadowed: it finds the least-detour silhouette edge and applies the
//!   Maekawa barrier model via the spatial crate's Fresnel helpers.
//! - [`higher_order_diffraction`] traces multi-edge (two-or-more) sequential
//!   bends with taut-path relaxation, complementing (never duplicating) the
//!   single-edge bend; enabled by
//!   [`config::GeometricConfig::with_max_diffraction_order`].
//! - [`coupled_path`] resolves the second-order arrival that reflects off one
//!   face **and** bends over one edge, in either order, reusing the image-source
//!   and least-detour primitives so it stays consistent with the lone bounce and
//!   lone bend it extends; enabled by
//!   [`config::GeometricConfig::with_coupled_paths`].
//! - [`coupled_sequence`] traces the arbitrary-order mixed chains that
//!   interleave three or more reflections and diffractions, extending
//!   [`coupled_path`] past the order-2 coupling without re-tracing the pure
//!   reflection or diffraction sets; enabled by
//!   [`config::GeometricConfig::with_max_coupled_order`].
//! - [`source_directivity`] weights each resolved arrival by how strongly the
//!   source radiates along that arrival's departure direction (the
//!   frequency-dependent omni-to-cardioid radiation pattern), darkening and
//!   quieting off-axis arrivals; opt-in through
//!   [`config::GeometricConfig::with_source_directivity`].
//! - [`receiver_directivity`] weights each resolved arrival by how sensitively
//!   the listener hears along that arrival's direction (the same
//!   frequency-dependent omni-to-cardioid pattern used for the source, applied
//!   as a receiver pickup), the receiver-side mirror of [`source_directivity`];
//!   opt-in through [`config::GeometricConfig::with_receiver_directivity`].
//! - [`backend`] assembles the above into [`backend::GeometricBackend`], the
//!   [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//!   implementation that fills the caller's bounded path buffer.
//! - [`path_smoothing`] is the control-rate companion to [`backend`]: a
//!   stateful [`path_smoothing::PathSmoother`] the caller ticks once per
//!   query to ease each resolved arrival toward the backend's newest target
//!   with a per-parameter one-pole glide, swelling new arrivals up from
//!   silence and fading vanished ones out, so the real-time voice never hears
//!   a jumped gain, delay, or filter corner.
//! - [`doppler`] is the per-path pitch companion to [`backend`] and
//!   [`path_smoothing`]: a stateful [`doppler::PerPathDoppler`] the caller
//!   ticks once per query to difference each resolved arrival's
//!   [`delay_seconds`](prism_audio_spatial::propagation::PropagationPath::delay_seconds)
//!   across frames into a clamped, glided per-path Doppler pitch factor, so an
//!   approaching reflection rises in pitch and a receding one falls
//!   independently of the direct sound and of the delay-line length the voice
//!   renders.
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

pub mod air_absorption;
pub mod backend;
pub mod config;
pub mod coupled_path;
pub mod coupled_sequence;
pub mod diffraction_edges;
pub mod diffraction_path;
pub mod direct_path;
pub mod doppler;
pub mod higher_order_diffraction;
pub mod higher_order_reflection;
pub mod material_map;
pub mod path_smoothing;
pub mod receiver_directivity;
pub mod reflection_path;
pub mod scene;
pub mod source_directivity;

pub use backend::GeometricBackend;
pub use config::GeometricConfig;
pub use material_map::MaterialTable;
pub use scene::{AcousticScene, SceneBuildError, SceneRayHit};
