//! Virtual-shadow-map page-request (`page-mark`) compute pass -- the second VSM
//! stage, downstream of the receiver-generation pass and upstream of the CPU
//! physical-page allocator.
//!
//! Wiring for the [`shaders/vsm_page_mark.wesl`] compute shader: each shadow
//! receiver produced by the receiver-generation pass selects a clip level from
//! its view distance, lands on one absolute world page, and its soft-shadow
//! filter kernel widens that footprint by whole page rings; the pass marks
//! those pages, with `atomicOr`, in a camera-snapped resident-window request
//! bitmap. It is the GPU twin of the golden
//! [`prism_render_shading::shadow::virtual_sm`] `generate_page_requests`.
//!
//! The submodules mirror the sibling receiver-generation pass one level up:
//!
//! * [`pipeline`] -- the compute pipeline and its owned group-0 layout;
//! * [`resources`] -- the per-view immediate block and persistent request
//!   bitmap, plus the `PrepareResources` system that builds them;
//! * [`bind_groups`] -- the `PrepareBindGroups` group-0 assembly;
//! * [`dispatch`] -- the `Core3d` node clearing and marking the bitmap.
//!
//! This pass does real GPU work every frame: it clears and populates a genuine
//! resident-window request bitmap from live receivers. The bitmap is copied
//! through the render-graph readback state machine, decoded by the CPU page
//! driver, and used to upload the virtual-to-physical page table. The resulting
//! residency and resident-page list are bridged in the same `PrepareResources`
//! phase to the resolve sampler and physical-atlas caster-depth pass; the only
//! intentional latency is the readback's unavoidable one-frame copy.

mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_vsm_page_mark_bind_groups;
pub(crate) use dispatch::vsm_mark_pages_pass;
pub(crate) use pipeline::init_vsm_page_mark_pipeline;
pub(crate) use resources::ViewVsmPageRequests;
pub(crate) use resources::{prepare_vsm_page_requests, VsmPageRequestBufferCache};
