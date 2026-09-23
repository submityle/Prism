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
mod lod;
mod output;
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
pub use lod::{GeometryLod, GeometryLodChain, LodSelection};
pub use output::{BufferRange, ViewVisibilityOutput, VisibilityFrame};
pub use view::{GpuViewRecord, HistoryPolicy, ViewFlags, ViewHandle};
pub use work::{GpuRenderWorkItem, RenderPassMask, VisibilityStageMask, WorkSortKey};

#[cfg(test)]
mod tests;
