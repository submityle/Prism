//! GPU virtual shadow map (UE5-style) subsystem -- scene-side twin of the CPU
//! golden [`prism_render_shading::shadow`] virtual-shadow-map reference
//! (`prism_render_shading::shadow::virtual_sm`).
//!
//! A virtual shadow map replaces a fixed-resolution shadow atlas with a sparse,
//! demand-paged one: the world is addressed through a clipmap of virtual pages,
//! but only the bounded working set the visible receivers actually touch is
//! ever backed by physical memory.  The pipeline runs in three stages, and this
//! module owns the scene-side plumbing for the two GPU shaders that bracket it:
//!
//! * **Page request** (`shaders/vsm_page_mark.wesl`) -- each shadow receiver
//!   selects a clip level from its view distance, lands on one absolute world
//!   page, and its soft-shadow filter kernel widens that footprint by whole
//!   page rings; the pass marks those pages in a camera-snapped resident-window
//!   request bitmap.  Twin of `generate_page_requests`.
//! * **Allocation** (CPU golden `allocator` / `page_table`) -- the requested
//!   pages are made resident in the physical page atlas under an LRU budget,
//!   producing the virtual-to-physical page table.
//! * **Sample** (`shaders/vsm_sample.wesl`) -- the resolve pass looks a
//!   receiver's virtual page up in the page table, maps it into the physical
//!   atlas tile and PCF-filters the stored depth.  Twin of the golden sampling
//!   path.
//!
//! Both shaders address pages byte-for-byte with the golden
//! [`prism_render_shading::ClipmapConfig`]; [`abi`] carries that layout plus the
//! per-pass driving state in `#[repr(C)]` immediate blocks kept in lockstep with
//! the WESL structs, and [`settings`] is the render-world resource mapping the
//! architecture / golden contracts onto those blocks.
//!
//! The device-side pipeline objects, buffer uploads, the physical-page
//! allocator wiring and the resolve-pass integration are registered by a later
//! slice (the render graph / plugin wiring is handled separately) and require
//! on-device validation for numerical parity against the CPU reference.  Until
//! that slice lands, the ABI and settings API below has no in-crate consumer
//! outside the layout / compilation tests, so the module and its re-exports are
//! `dead_code`-/`unused`-expected rather than trimmed.

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "VSM ABI's page-mark / sample blocks are consumed by the device-side wiring slice added separately")
)]
mod abi;
mod bind_groups;
mod dispatch;
mod extract;
mod page_mark;
mod pipeline;
mod resources;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "VSM settings' clipmap / contract accessors are consumed by the device-side wiring slice added separately")
)]
mod settings;

#[cfg(test)]
mod shader_tests;

#[expect(
    unused_imports,
    reason = "re-exported for the device-side VSM page-mark / sample wiring slice added separately"
)]
pub(crate) use abi::{
    window_slot_count, GpuVsmPageMarkParams, GpuVsmReceiver, GpuVsmReceiverGenParams,
    GpuVsmSampleParams,
    VSM_PAGE_MARK_WORKGROUP_SIZE, VSM_PAGE_UNMAPPED, VSM_SAMPLE_WORKGROUP_SIZE,
};

pub(crate) use bind_groups::prepare_vsm_receiver_gen_bind_groups;
pub(crate) use dispatch::vsm_receiver_gen_pass;
pub(crate) use extract::{extract_vsm_primary_light, VsmPrimaryLight};
pub(crate) use pipeline::init_vsm_receiver_gen_pipeline;
pub(crate) use resources::{prepare_vsm_receiver_resources, VsmReceiverBufferCache};
pub(crate) use settings::PrismVirtualShadowSettings;

pub(crate) use page_mark::{
    init_vsm_page_mark_pipeline, prepare_vsm_page_mark_bind_groups, prepare_vsm_page_requests,
    vsm_mark_pages_pass, VsmPageRequestBufferCache,
};
