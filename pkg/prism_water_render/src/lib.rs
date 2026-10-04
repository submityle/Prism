//! Real-device water-surface renderer for Prism — the water subsystem's first
//! *visible* render product.
//!
//! Everything else the water engine ships in `prism_render_architecture` is
//! either a `CPU` pure-function solver or a compute twin that validates scalar
//! fields against a golden standard. Those prove the simulation is correct, but
//! none of them turns a solved surface into pixels. This crate closes that gap
//! with the smallest honest end-to-end path: it builds a real `wgpu` graphics
//! pipeline on a headless device, rasterises a Gerstner-displaced water grid
//! with a Schlick-Fresnel reflection and an analytic sun-specular highlight,
//! and reads the frame back to a portable image.
//!
//! Scope is deliberately narrow and stated plainly: this is a single offline
//! frame of a water *surface*. It does not drive the Bevy render world, the
//! `Lumen`-style global illumination, the virtual-shadow maps, the froxel
//! volumetrics or the ray-traced reflections and caustics that
//! `prism_render_scene` wires at the descriptor level. It is the first rung,
//! not the whole ladder.
//!
//! The pipeline uses only portable core `WGSL` and the guaranteed `wgpu`
//! baseline (`Rgba8Unorm` colour, `Depth32Float` depth), so it runs on any
//! Metal, Vulkan or DX12 adapter. On a host with no usable adapter the context
//! acquisition returns [`None`] and callers skip gracefully.
//!
//! Provenance: hand-written math, a dependency-free `PNG` encoder and standard
//! `wgpu` initialisation; no Unreal Engine source or derived code.

#![forbid(unsafe_code)]

extern crate alloc;

pub mod camera;
pub mod context;
pub mod png;
pub mod surface;

pub use camera::Camera;
pub use context::{block_on, GpuContext};
pub use png::encode_rgba8;
pub use surface::{render, RenderedFrame, WaterSurfaceScene};
