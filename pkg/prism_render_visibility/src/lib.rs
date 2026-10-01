//! Unified multi-view visibility and GPU work generation.

#![expect(
    missing_docs,
    reason = "The visibility ABI is documented as it freezes."
)]

extern crate alloc;

mod benchmark;
mod binning;
mod cull_hzb;
mod cull_two_phase;
mod culling;
mod diagnostics;
mod hzb_build;
mod hzb_footprint;
mod hzb_projection;
mod hzb_pyramid;
mod hzb_query;
mod lod;
mod occlusion;
mod occlusion_resolve;
mod output;
mod raster_phase;
mod two_phase;
mod two_phase_resolve;
mod view;
mod work;

#[cfg(feature = "std")]
pub use benchmark::benchmark_cpu_reference;
pub use benchmark::VisibilityBenchmarkResult;
pub use binning::{
    build_two_phase_view_draw_bins, build_view_draw_bins, DrawBinCandidate, DrawBinKey,
    DrawBinRange, GpuDrawBinHeader, TwoPhaseViewDrawBins, ViewDrawBins, DRAW_BIN_HEADER_WORDS,
};
pub use cull_hzb::{cull_view_with_hzb, HzbCullScene};
pub use cull_two_phase::cull_view_two_phase;
pub use culling::{cull_view, CullReason, VisibilityInput};
pub use diagnostics::VisibilityDiagnostics;
pub use hzb_build::{build_hzb_pyramid, HzbPyramidStorage, OwnedHzbMip};
pub use hzb_footprint::{conservative_occluder_reverse_z, HzbFootprint};
pub use hzb_projection::{project_world_aabb, ProjectedBounds, WorldAabb};
pub use hzb_pyramid::{HzbMip, HzbPyramid};
pub use hzb_query::{test_bounds_occluded, OcclusionQuery};
pub use lod::{GeometryLod, GeometryLodChain, LodSelection};
pub use occlusion::{HzbPhase, HzbTest};
pub use occlusion_resolve::resolve_occluded_set;
pub use output::{BufferRange, ViewVisibilityOutput, VisibilityFrame};
pub use raster_phase::{plan_two_phase_raster, raster_phase_of, RasterPhase, TwoPhaseRasterPlan};
pub use two_phase::{classify_early_hzb, resolve_current_hzb};
pub use two_phase_resolve::{resolve_two_phase_occlusion, TwoPhaseHzbInput, TwoPhaseOcclusion};
pub use view::{GpuViewRecord, HistoryPolicy, ViewFlags, ViewHandle};
pub use work::{GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey};

#[cfg(test)]
mod tests;
