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
//! This crate owns the render-app seam and the executable frame-graph path. It
//! keeps backend/device state in the render world while geometry, materials,
//! lighting, and post-processing are added as declared passes on the same
//! `Render` schedule. A provider must register every response module/package
//! identity before a specialized pipeline can enter the cold-cache lifecycle;
//! missing mappings are reported as loud errors rather than silently losing
//! residency accounting.
#![forbid(unsafe_code)]

extern crate alloc;

mod bin_queue;
mod config;
mod device_loss;
mod draw;
mod frame;
mod image;
mod intent;
mod label;
mod material;
mod mesh;
mod mesh_draw;
mod mesh_pipeline;
mod mesh_pipelines;
mod mesh_source;
mod pipeline_cache;
mod pipeline_pass;
mod plugin;
mod queue;
mod queue_stage;
mod record;
mod render_asset;
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
mod image_tests;
#[cfg(test)]
mod tests;

pub use bin_queue::{bin_indirect_args, BinQueueOutcome};
pub use config::RenderAppConfig;
pub use device_loss::DeviceLossSignal;
pub use draw::{DrawFunctionId, DrawFunctions, MeshRecorder};
pub use frame::{RENDER_DEPTH_FORMAT, RENDER_TARGET_FORMAT};
pub use image::GpuImage;
pub use intent::{RenderIntent, RenderViewport};
pub use label::{Render, RenderApp, RenderStartup};
pub use material::{
    BindlessSlot, FallbackTextures, MaterialBindingLayout, MaterialParameterBuffers,
    MaterialSyncReport, MaterialTextureSlots, MaterialUploadStats, ParameterHeap, RenderMaterials,
    TextureFallback, TextureHeapStats, TextureRequest, TextureSlotHeap,
};
pub use mesh::{RenderMesh, RenderMeshBufferInfo};
pub use mesh_draw::{mesh_bin_draw, mesh_draw};
pub use mesh_pipeline::{
    full_lobe_fallback, MeshPipelineKey, MeshPipelineProvider, MeshPipelineSpecializer,
};
pub use mesh_pipelines::{
    MeshPipelines, ResidentResponseModule, ResponseModuleCache, ResponseModuleCacheError,
    ResponseModuleSpec, SpineMeshPipeline,
};
pub use pipeline_cache::{
    cold_key_from_pipeline_key, hash_vertex_layouts, PipelineBinaryCacheError, PipelineCache,
    PipelineCompiler, PipelineKey, PipelineKeyMappingError, PipelineLookupStatus,
    PipelineResolution,
};
pub use pipeline_pass::{PassDescriptor, PassDescriptorClass, PassRole, PassSetError};
pub use plugin::PrismRenderPlugin;
pub use queue::{
    phase_for_bin_class, phase_for_render_class, DrawRequest, IndirectArgs, PhaseItem, RenderQueues,
};
pub use record::RenderRecord;
pub use render_asset::{
    extract_assets, prepare_assets, prepare_extracted_assets,
    prepare_extracted_assets_with_context, register_fixed_page_streaming, register_render_assets,
    AssetSyncReport, AssetSyncStats, ExtractedAssets, FixedPageBytesUploader,
    FixedPageDeferredReplacement, FixedPageIoReport, FixedPagePool, FixedPagePoolDescriptor,
    FixedPagePoolEvent, FixedPageReclaimEvent, FixedPageRenderPayload, FixedPageRenderQueue,
    FixedPageResidencyBridge, FixedPageStreamingConfig, FixedPageStreamingGpu,
    FixedPageStreamingManager, FixedPageStreamingSource, FixedPageStreamingStats, FixedPageUpload,
    GpuPrepareBudget, PageAllocation, PagePoolError, PageReleaseTransaction,
    PageReplaceTransaction, PageResidencyBridgeError, PageUploadTransaction, PrepareAssetError,
    PrepareCancellation, PrepareContext, PrepareContextError, RenderAsset, RenderAssets,
};
pub use runtime_resource::RenderRuntimeResource;
pub use scene::{MeshClass, SceneRegistry, VisibilityWorkList};
pub use sets::{
    configure_render_graph_sets, configure_render_sets, RenderGraphSet, RenderPhase, RenderSystems,
};
pub use startup::{init_gpu_resource, GpuResourceAppExt, RenderStartupState};
pub use stats::RenderFrameStats;
pub use sync_world::{tag_extracted_entities, MainEntity, MainEntityHashSet};
pub use target::RenderTarget;
pub use texture_cache::{GpuTexture, TextureCache, TextureKey};
pub use view_driver::{for_each_view, with_current_view};
pub use view_query::{CurrentView, ViewQuery, ViewQueryState};
pub use views::{RenderView, RenderViews};

/// Commonly used exports: the plugin, its labels, and the resources that cross
/// the main-world/render-world boundary.
pub mod prelude {
    pub use crate::bin_queue::{bin_indirect_args, BinQueueOutcome};
    pub use crate::config::RenderAppConfig;
    pub use crate::draw::{DrawFunctionId, DrawFunctions, MeshRecorder};
    pub use crate::intent::{RenderIntent, RenderViewport};
    pub use crate::label::{Render, RenderApp, RenderStartup};
    pub use crate::material::{
        BindlessSlot, MaterialParameterBuffers, MaterialSyncReport, MaterialUploadStats,
        ParameterHeap, RenderMaterials, TextureFallback, TextureHeapStats, TextureRequest,
        TextureSlotHeap,
    };
    pub use crate::mesh_pipeline::{
        full_lobe_fallback, MeshPipelineKey, MeshPipelineProvider, MeshPipelineSpecializer,
    };
    pub use crate::mesh_pipelines::{
        MeshPipelines, ResidentResponseModule, ResponseModuleCache, ResponseModuleCacheError,
        ResponseModuleSpec, SpineMeshPipeline,
    };
    pub use crate::pipeline_cache::{
        cold_key_from_pipeline_key, hash_vertex_layouts, PipelineBinaryCacheError, PipelineCache,
        PipelineCompiler, PipelineKey, PipelineKeyMappingError, PipelineLookupStatus,
        PipelineResolution,
    };
    pub use crate::pipeline_pass::{PassDescriptor, PassDescriptorClass, PassRole, PassSetError};
    pub use crate::plugin::PrismRenderPlugin;
    pub use crate::queue::{
        phase_for_bin_class, phase_for_render_class, DrawRequest, IndirectArgs, PhaseItem,
        RenderQueues,
    };
    pub use crate::record::RenderRecord;
    pub use crate::render_asset::{
        extract_assets, prepare_assets, prepare_extracted_assets,
        prepare_extracted_assets_with_context, register_fixed_page_streaming,
        register_render_assets, AssetSyncReport, AssetSyncStats, ExtractedAssets,
        FixedPageBytesUploader, FixedPageDeferredReplacement, FixedPageIoReport, FixedPagePool,
        FixedPagePoolDescriptor, FixedPagePoolEvent, FixedPageReclaimEvent, FixedPageRenderPayload,
        FixedPageRenderQueue, FixedPageResidencyBridge, FixedPageStreamingConfig,
        FixedPageStreamingGpu, FixedPageStreamingManager, FixedPageStreamingSource,
        FixedPageStreamingStats, FixedPageUpload, GpuPrepareBudget, PageAllocation, PagePoolError,
        PageReleaseTransaction, PageReplaceTransaction, PageResidencyBridgeError,
        PageUploadTransaction, PrepareAssetError, PrepareCancellation, PrepareContext,
        PrepareContextError, RenderAsset, RenderAssets,
    };
    pub use crate::scene::{MeshClass, SceneRegistry, VisibilityWorkList};
    pub use crate::sets::{
        configure_render_graph_sets, configure_render_sets, RenderGraphSet, RenderPhase,
        RenderSystems,
    };
    pub use crate::startup::{init_gpu_resource, GpuResourceAppExt, RenderStartupState};
    pub use crate::stats::RenderFrameStats;
    pub use crate::sync_world::{tag_extracted_entities, MainEntity, MainEntityHashSet};
    pub use crate::target::RenderTarget;
    pub use crate::texture_cache::{GpuTexture, TextureCache, TextureKey};
    pub use crate::view_driver::{for_each_view, with_current_view};
    pub use crate::view_query::{CurrentView, ViewQuery, ViewQueryState};
    pub use crate::views::{RenderView, RenderViews};
}
