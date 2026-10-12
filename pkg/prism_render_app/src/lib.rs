//! # `prism_render_app`
//!
//! The integration seam that wires the RHI-backed render spine
//! ([`prism_render_runtime`]) into the application shell ([`prism_app`]) as a
//! dedicated **render sub-app**. It is the in-house replacement for the host
//! role `bevy_render`'s `RenderApp` used to play: the place that owns the GPU
//! device, drives a frame graph every frame, and is fed each frame by a
//! one-way extract from the authoritative simulation world.
//!
//! ## The boundary
//!
//! The design splits the app into a main sub-app (authoritative simulation) and
//! an ordered set of secondary sub-apps, of which the render sub-app is the
//! archetype (design §9, §25.2). This crate draws that boundary explicitly:
//!
//! - **GPU state lives in the render world.** [`PrismRenderPlugin`] brings up a
//!   [`RenderRuntime`](prism_render_runtime::RenderRuntime) and installs it as a
//!   resource on the render sub-app's world, together with the offscreen frame
//!   [`RenderTarget`] and per-frame [`RenderFrameStats`]. The main world never
//!   names a device, a queue, or a texture.
//! - **The main world only publishes intent.** A [`RenderIntent`] resource on
//!   the main world is the single source of truth for the frame's size; the
//!   extract step copies it into a [`RenderRecord`] on the render world.
//! - **The render world renders.** A [`Render`] schedule holds the system that
//!   builds and executes the frame graph, and a small driver in the render
//!   sub-app's `Last` phase runs that schedule each frame.
//!
//! ## Backend-agnostic
//!
//! [`PrismRenderPlugin`] is generic over the RHI
//! [`Instance`](prism_render_driver::Instance) backend, so a real
//! app constructs it with the shipped wgpu backend while tests construct it with
//! the headless null backend — no window, no GPU, no driver. Switching backends
//! is constructing the plugin with a different `Instance`; nothing in this crate
//! changes.
//!
//! ## Example
//!
//! ```no_run
//! use prism_app::prelude::*;
//! use prism_render_app::prelude::*;
//! use prism_render_driver_wgpu::WgpuInstance;
//!
//! let mut app = App::new();
//! app.add_plugins(PrismRenderPlugin::new(WgpuInstance::new()));
//! // ... add gameplay systems ...
//! app.update(); // each update extracts and renders one frame
//! ```
//!
//! ## Scope
//!
//! This crate is the seam, not the renderer. The frame graph it records is
//! deliberately minimal — it clears the offscreen target — because the point of
//! this layer is to prove the ECS ↔ RHI wiring. Passes for geometry, materials,
//! lighting, and post-processing layer on top of the same `Render` schedule.
#![forbid(unsafe_code)]

extern crate alloc;

mod bin_queue;
mod config;
mod draw;
mod frame;
mod intent;
mod label;
mod mesh_pipeline;
mod mesh_draw;
mod pipeline_cache;
mod plugin;
mod queue;
mod record;
mod runtime_resource;
mod scene;
mod sets;
mod startup;
mod stats;
mod sync_world;
mod systems;
mod target;
mod texture_cache;
mod view_driver;
mod view_query;
mod views;

#[cfg(test)]
mod tests;

pub use bin_queue::{BinQueueOutcome, bin_indirect_args};
pub use config::RenderAppConfig;
pub use draw::{DrawFunctionId, DrawFunctions, MeshRecorder};
pub use frame::RENDER_TARGET_FORMAT;
pub use intent::{RenderIntent, RenderViewport};
pub use label::{Render, RenderApp, RenderStartup};
pub use mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider, MeshPipelineSpecializer};
pub use mesh_draw::{mesh_bin_draw, mesh_draw};
pub use pipeline_cache::{PipelineCache, PipelineKey, hash_vertex_layouts};
pub use plugin::PrismRenderPlugin;
pub use queue::{
    DrawRequest, IndirectArgs, PhaseItem, RenderQueues, phase_for_bin_class,
    phase_for_render_class,
};
pub use record::RenderRecord;
pub use runtime_resource::RenderRuntimeResource;
pub use scene::{MeshClass, SceneRegistry, VisibilityWorkList};
pub use sets::{
    RenderGraphSet, RenderPhase, RenderSystems, configure_render_graph_sets, configure_render_sets,
};
pub use startup::{GpuResourceAppExt, RenderStartupState, init_gpu_resource};
pub use stats::RenderFrameStats;
pub use sync_world::{MainEntity, MainEntityHashSet, tag_extracted_entities};
pub use target::RenderTarget;
pub use texture_cache::{GpuTexture, TextureCache, TextureKey};
pub use view_driver::{for_each_view, with_current_view};
pub use view_query::{CurrentView, ViewQuery, ViewQueryState};
pub use views::{RenderView, RenderViews};

/// Commonly used exports: the plugin, its labels, and the resources that cross
/// the main-world/render-world boundary.
pub mod prelude {
    pub use crate::bin_queue::{BinQueueOutcome, bin_indirect_args};
    pub use crate::config::RenderAppConfig;
    pub use crate::draw::{DrawFunctionId, DrawFunctions, MeshRecorder};
    pub use crate::intent::{RenderIntent, RenderViewport};
    pub use crate::label::{Render, RenderApp, RenderStartup};
    pub use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider, MeshPipelineSpecializer};
    pub use crate::pipeline_cache::{PipelineCache, PipelineKey, hash_vertex_layouts};
    pub use crate::plugin::PrismRenderPlugin;
    pub use crate::queue::{
        DrawRequest, IndirectArgs, PhaseItem, RenderQueues, phase_for_bin_class,
        phase_for_render_class,
    };
    pub use crate::record::RenderRecord;
    pub use crate::scene::{MeshClass, SceneRegistry, VisibilityWorkList};
    pub use crate::sets::{
        RenderGraphSet, RenderPhase, RenderSystems, configure_render_graph_sets,
        configure_render_sets,
    };
    pub use crate::startup::{GpuResourceAppExt, RenderStartupState, init_gpu_resource};
    pub use crate::stats::RenderFrameStats;
    pub use crate::sync_world::{MainEntity, MainEntityHashSet, tag_extracted_entities};
    pub use crate::target::RenderTarget;
    pub use crate::texture_cache::{GpuTexture, TextureCache, TextureKey};
    pub use crate::view_driver::{for_each_view, with_current_view};
    pub use crate::view_query::{CurrentView, ViewQuery, ViewQueryState};
    pub use crate::views::{RenderView, RenderViews};
}
