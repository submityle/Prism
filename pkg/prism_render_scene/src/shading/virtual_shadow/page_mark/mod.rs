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
//! **Incremental, not a no-op.** This pass does real GPU work every frame --
//! it clears and populates a genuine resident-window request bitmap from live
//! receivers. What it does *not* yet have is a downstream GPU consumer: the
//! request bitmap is produced but not yet read back by the physical-page
//! allocator, so no page-table / physical-atlas residency is driven from it
//! yet. That allocation bridge (an LRU page-table readback feeding the resolve
//! pass) is a separate future slice. Until it lands the marked bitmap has no
//! in-crate reader outside this pass, so the plugin-facing re-exports below are
//! `dead_code`-allowed rather than trimmed.

mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_vsm_page_mark_bind_groups;
pub(crate) use dispatch::vsm_mark_pages_pass;
pub(crate) use pipeline::init_vsm_page_mark_pipeline;
pub(crate) use resources::{prepare_vsm_page_requests, VsmPageRequestBufferCache};
