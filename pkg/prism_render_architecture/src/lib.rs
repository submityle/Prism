//! Contracts for Prism's Vulkan-first rendering architecture.
//!
//! ECS remains responsible for scene state, plugins, and CPU scheduling. The
//! frame graph describes GPU resource access and submission dependencies.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "Architecture modules are documented as they stabilize."
)]

extern crate alloc;

pub mod abi;
pub mod backend;
pub mod capture;
pub mod cloth;
pub mod deformation;
pub mod descriptor_heap;
pub mod diagnostics;
pub mod display;
pub mod frame_graph;
pub mod geometry;
pub mod gpu_scene;
pub mod hair;
pub mod history;
pub mod lighting;
pub mod material;
pub mod memory;
pub mod motion;
pub mod paging;
pub mod particle;
pub mod quality;
pub mod ray_scene;
pub mod shader_package;
pub mod temporal_upscale;
pub mod texture_streaming;
pub mod transparency;
pub mod view_family;
pub mod virtual_geometry;
pub mod virtual_resource;
pub mod virtual_shadow;
pub mod volumetric;
pub mod water;
pub mod work_graph;
pub mod world;

/// Version of the public architecture contracts in this package.
pub const ARCHITECTURE_VERSION: u32 = 1;
