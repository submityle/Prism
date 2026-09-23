//! Unified multi-view visibility and GPU work generation.

#![expect(
    missing_docs,
    reason = "The visibility ABI is documented as it freezes."
)]

extern crate alloc;

mod culling;
mod diagnostics;
mod lod;
mod output;
mod view;
mod work;

pub use culling::{cull_view, CullReason, VisibilityInput};
pub use diagnostics::VisibilityDiagnostics;
pub use lod::{GeometryLod, GeometryLodChain, LodSelection};
pub use output::{BufferRange, ViewVisibilityOutput, VisibilityFrame};
pub use view::{GpuViewRecord, HistoryPolicy, ViewFlags, ViewHandle};
pub use work::{GpuRenderWorkItem, RenderPassMask, WorkSortKey};

#[cfg(test)]
mod tests;
