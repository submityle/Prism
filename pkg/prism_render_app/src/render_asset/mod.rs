//! Render assets: the GPU representation of a simulation-side asset.
//!
//! An asset lives once, in the main world's [`Assets`] arena or an
//! [`prism_asset::AssetServer`] typed arena, and its GPU form lives in the render world.
//! Bevy splits that into `RenderAsset` (how to
//! prepare one), `RenderAssets<A>` (the prepared table), and a prepare stage
//! that keeps the table in step with the source. Prism keeps the same three
//! pieces, with two deliberate differences documented on [`RenderAsset`]:
//!
//! - the source asset is **borrowed** while preparing (Prism's asset arena hands
//!   out `&A`; nothing moves data out of the main world), and
//! - releasing GPU objects is an explicit [`RenderAsset::unload`] call, so an
//!   implementation that owns RHI handles has one documented place to destroy
//!   them — the sync reports removals rather than leaking them.
//!
//! The table is keyed by [`AssetId`], so the identity that a render-side phase
//! item or bind group carries is the same one the simulation uses.

mod context;
mod extract;
mod lighting_virtual_residency;
#[cfg(feature = "fixed-page-feedback")]
mod page_feedback;
mod page_pool;
mod page_residency_bridge;
#[cfg(feature = "resident-page-streaming")]
mod page_resident;
mod page_streaming;
mod page_streaming_stage;
mod prepare;
mod stage;
mod table;
mod virtual_geometry_gpu;

#[cfg(test)]
mod page_pool_tests;
#[cfg(test)]
mod page_streaming_tests;
#[cfg(test)]
mod tests;

use prism_asset::Asset;
use prism_asset::AssetId;
use prism_render_driver::{RenderDevice, RenderQueue};

pub use context::{GpuPrepareBudget, PrepareCancellation, PrepareContext, PrepareContextError};
pub use extract::{extract_assets, ExtractedAssets};
pub use lighting_virtual_residency::{
    LightingPageUpload, LightingResidencySync, LightingVirtualResidencyError,
    LightingVirtualResidencyGpuAdapter,
};
#[cfg(feature = "fixed-page-feedback")]
pub use page_feedback::{
    register_fixed_page_feedback, FixedPageFeedback, FixedPageFeedbackConfig,
    FixedPageFeedbackError, FixedPageFeedbackGpu, FixedPageFeedbackSample, FixedPageQuery,
};
pub use page_pool::{
    FixedPageDeferredRelease, FixedPageDeferredReplacement, FixedPagePool, FixedPagePoolDescriptor,
    FixedPagePoolEvent, FixedPageReclaimEvent, FixedPageUpload, PageAllocation, PagePoolError,
    PageReleaseTransaction, PageReplaceTransaction, PageUploadTransaction,
};
pub use page_residency_bridge::{FixedPageResidencyBridge, PageResidencyBridgeError};
#[cfg(feature = "resident-page-streaming")]
pub use page_resident::{
    ResidentPageAllocation, ResidentPageError, ResidentPagePool, ResidentPagePublication,
};
pub use page_streaming::{
    FixedPageBytesUploader, FixedPageIoReport, FixedPageReleasePayload, FixedPageRenderPayload,
    FixedPageRenderQueue, FixedPageSlotKey, FixedPageStreamingSource,
};
pub use page_streaming_stage::{
    register_fixed_page_streaming, FixedPageStreamingConfig, FixedPageStreamingGpu,
    FixedPageStreamingStats,
};
pub use prepare::{prepare_extracted_assets_with_context, AssetSyncReport, PrepareAssetError};
pub(crate) use prepare::prepare_extracted_assets;
pub(crate) use stage::{
    invalidate_registered_render_assets, recovered_assets_pending, PrepareRecoveredAssets,
};
pub use stage::{prepare_assets, register_render_assets, AssetSyncStats};
pub use table::RenderAssets;
pub use virtual_geometry_gpu::{
    register_virtual_geometry_publication_bridge, ExtractedVirtualGeometryPublicationRequests,
    VirtualGeometryGpuBuffers, VirtualGeometryGpuPublicationStore, VirtualGeometryGpuStoreError,
    VirtualGeometryPublicationDiagnostics, VirtualGeometryPublicationRequest,
    VirtualGeometryPublicationRequests, VirtualGeometryResidencyAssetSnapshot,
    VirtualGeometryResidencySnapshot,
};

/// An asset the renderer keeps a separate GPU form of.
///
/// The source is borrowed, not consumed: the simulation world stays the single
/// owner of the authored data (design §21's one-way rule), and a prepare copies
/// what the GPU needs — a vertex buffer, a texture upload — out of it. Bevy's
/// `take_gpu_data` exists because its extract *moves* heavy payloads; Prism has
/// no such split to serve, so it is deliberately not ported until a measurement
/// asks for it.
pub trait RenderAsset: Send + Sync + 'static + Sized {
    /// The simulation-side asset this prepares from.
    ///
    /// `Clone` because the render world receives a *copy* of what changed this
    /// frame (the simulation keeps its asset; see [`extract_assets`]).
    type SourceAsset: Asset + Clone;

    /// Contextual preparation with a transactional allocation budget and
    /// cancellation checkpoint. A retry leaves `previous` authoritative, while
    /// allocations tracked by the context are rolled back before the source is
    /// queued again.
    /// budget and cancellation checkpoint.
    ///
    /// This default keeps existing implementations source-compatible. New
    /// implementations may override it and allocate through `context`; the
    /// sync layer then rolls those allocations back on retry/cancellation and
    /// unloads the old representation exactly once after a marked replacement
    /// succeeds.
    fn prepare_with_context(
        source: &Self::SourceAsset,
        id: AssetId<Self::SourceAsset>,
        context: &mut PrepareContext<'_>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>>;

    /// Releases the RHI objects `prepared` owns.
    ///
    /// Called by [`sync_assets`] when the source asset is gone (or, eventually,
    /// when the renderer reclaims it), *before* the representation is dropped —
    /// dropping a [`BufferId`](prism_render_driver::BufferId) alone would leak
    /// the GPU buffer, which is exactly what this hook prevents.
    ///
    /// The default does nothing, which is correct for a representation that owns
    /// no RHI objects (one that only borrows cached views, say).
    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        let _ = (prepared, device);
    }
}
