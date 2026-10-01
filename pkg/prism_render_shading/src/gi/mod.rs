//! Global-illumination CPU golden reference implementations.
//!
//! Houses backend-neutral, GPU-free numerical references for the render
//! engine's GI passes.  Every item here is a deterministic pure function with
//! unit tests; its output is the numerical reference the WESL/GPU twin passes
//! must reproduce under real-device Metal parity.
//!
//! * [`world_space`] — Lumen-style screen-probe + world-space radiance-cache
//!   pipeline, including DDGI-style Chebyshev visibility for leak suppression
//!   and spherical-Gaussian glossy reconstruction.
//! * [`screen_probe`] — screen-space probe sampling and reuse: ReSTIR GI
//!   reservoirs/GRIS and path-guided importance sampling.
//! * [`scene`] — scene-level signed-distance structures: mesh distance fields
//!   and distance-field ambient occlusion.
//! * [`occlusion`] — bent-normal and specular-occlusion reconstruction from
//!   hemispherical visibility.
//! * [`sample`] — low-discrepancy / blue-noise-like sampling (Owen-scrambled
//!   Sobol' + R2) and BSDF-domain mappings shared by every GI ray budget.
//! * [`denoise`] — firefly clamping and running-variance estimation shared by
//!   the spatio-temporal denoisers.
//! * [`light`] — many-light importance sampling: hierarchical light BVH and
//!   ReGIR world-space grid reservoirs for mega-light direct lighting.
//! * [`reflect`] — unified reflection: stochastic HiZ screen-space reflection
//!   and reflection-probe blending.
//! * [`micro`] — micro-scale occlusion: normal-map-derived micro bent normal
//!   and cavity ambient occlusion.
//! * [`material`] — material-level GI responses: thin-film iridescence and
//!   water / wet-surface reflection-refraction blending.
//! * [`world_restir`] — world-space ReSTIR spatial-hash reservoirs (SHARC-style
//!   persistent radiance reuse).
//! * [`volumetric_gi`] — froxel radiance reservoirs for participating-media
//!   scattering (Volumetric ReSTIR).
//! * [`caustics`] — photon splatting + manifold NEE for specular-to-diffuse
//!   caustic light transport.
//! * [`probe_volume`] — adaptive probe-grid irradiance + sky-visibility occlusion.

pub mod denoise;
pub mod occlusion;
pub mod sample;
pub mod scene;
pub mod screen_probe;
pub mod light;
pub mod material;
pub mod micro;
pub mod reflect;
pub mod world_restir;
pub mod volumetric_gi;
pub mod caustics;
pub mod probe_volume;
pub mod world_space;
