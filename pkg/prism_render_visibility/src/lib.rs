//! Unified multi-view visibility and GPU work generation.

#![expect(
    missing_docs,
    reason = "The visibility ABI is documented as it freezes."
)]

extern crate alloc;

mod benchmark;
mod binning;
mod culling;
mod diagnostics;
mod hzb_footprint;
mod hzb_projection;
mod lod;
mod occlusion;
mod output;
mod two_phase;
mod view;
mod work;

#[cfg(feature = "std")]
pub use benchmark::benchmark_cpu_reference;
pub use benchmark::VisibilityBenchmarkResult;
pub use binning::{
    build_view_draw_bins, DrawBinCandidate, DrawBinKey, DrawBinRange, GpuDrawBinHeader,
    ViewDrawBins, DRAW_BIN_HEADER_WORDS,
};
pub use culling::{cull_view, CullReason, VisibilityInput};
pub use diagnostics::VisibilityDiagnostics;
pub use hzb_footprint::{conservative_occluder_reverse_z, HzbFootprint};
pub use hzb_projection::{project_world_aabb, ProjectedBounds, WorldAabb};
pub use lod::{GeometryLod, GeometryLodChain, LodSelection};
pub use occlusion::{HzbPhase, HzbTest};
pub use output::{BufferRange, ViewVisibilityOutput, VisibilityFrame};
pub use two_phase::{classify_early_hzb, resolve_current_hzb};
pub use view::{GpuViewRecord, HistoryPolicy, ViewFlags, ViewHandle};
pub use work::{GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey};

#[cfg(test)]
mod tests;
