//! # `prism_render_gpudriven`
//!
//! The GPU-driven culling vertical slice for Prism's RHI: the first piece of
//! the M2 milestone ("compute culling + indirect draw") built directly on the
//! frozen [`prism_render_driver`] interface.
//!
//! ## What the slice does
//!
//! 1. [`cull::CullStage`] uploads a batch of instance AABBs and a view frustum,
//!    then dispatches one compute pass. The pass classifies each AABB against
//!    the six frustum planes, appends the survivors to a visible-index list
//!    through an atomic counter, and authors the indirect-draw arguments —
//!    including the surviving instance count.
//! 2. [`draw::DrawStage`] records indexed draws that consume those GPU-authored
//!    arguments ([`draw::DrawStage::encode_indexed_indirect`] for the single
//!    draw, [`draw::DrawStage::encode_multi_indirect`] for the packed
//!    multi-draw array), so the CPU never supplies the count.
//!
//! The point of the slice is what the CPU *does not* do: it never learns how
//! many instances survive. A production caller hands the GPU-authored argument
//! buffer straight to the draw; [`cull::CullStage::run`] adds a readback only
//! so a test can prove the GPU's answer matches [`reference::cull_cpu`].
//!
//! ## Backend
//!
//! The stages are written against the RHI traits, so they are backend-agnostic.
//! The one exception is readback — [`readback::ReadBuffer`] is a backend
//! capability (the RHI is fire-and-forget), implemented here for the shipped
//! wgpu backend and used only by the GPU test and by [`cull::CullStage::run`].
//!
//! ## The frame graph
//!
//! The GPU-driven scene is declared through [`prism_render_graph`]: the frame is
//! a reset compute pass, a cull compute pass, and a raster pass, all bound to
//! resources the stages own and *imported* into the graph rather than
//! reallocated. The graph's SSA versioning orders them without a single manual
//! barrier — the cull reads the argument version the reset wrote, and the draw
//! reads the visible-list and argument versions the cull wrote.
//!
//! The naive per-object baseline deliberately stays on the direct encoder. It is
//! the comparison point that shows what the graph-driven, GPU-authored
//! submission removes, and keeping it below the graph leaves the two submission
//! paths independent.
//!
//! ## Layout
//!
//! - [`frustum`]: planes, AABBs, and the classification test the shader mirrors.
//! - [`scene`]: the deterministic test lattice and view volume.
//! - [`reference`]: the CPU cull the GPU result is checked against.
//! - [`encoding`]: the little-endian layouts shared with the WGSL shaders.
//! - [`instances`]: per-instance transform/color and the id-derived colors.
//! - [`quad`]: the shared instance quad both draw paths rasterize.
//! - [`camera`]: the orthographic view and its uniform.
//! - [`cull`] / [`draw`]: the compute cull and the coverage draw it feeds.
//! - [`scene_draw`]: the GPU-driven scene draw, declared as a render graph and
//!   reading instance storage.
//! - [`naive`]: the per-object submission baseline the benchmark compares to,
//!   kept on the direct encoder.
//! - [`gpu`]: the RHI assembly helpers every stage shares.
//! - [`readback`]: backend readback and pixel-coverage counting.

extern crate alloc;

pub mod camera;
pub mod cull;
pub mod draw;
pub mod encoding;
pub mod frustum;
pub mod gpu;
pub mod instances;
pub mod naive;
pub mod quad;
pub mod readback;
pub mod reference;
pub mod scene;
pub mod scene_draw;

pub use camera::{FrameUniform, OrthoCamera};
pub use cull::{CullOutput, CullParams, CullStage};
pub use draw::{DrawStage, INDICES_PER_INSTANCE};
pub use encoding::IndirectArgs;
pub use frustum::{aabb_inside, signed_distance, Aabb, Frustum, Plane};
pub use instances::{id_color, id_color_bytes, instance_storage_bytes, Instance};
pub use naive::NaiveStage;
pub use quad::{corner_offset, corners, QUAD_HALF_EXTENT, QUAD_VERTICES};
pub use readback::{count_lit_pixels, pixel_at, ReadBuffer};
pub use reference::{cull_cpu, projected_center_pixel};
pub use scene::{
    candidate_lattice, instance_records, lattice_instances, layer_slab, plane_view_center_frustum,
    plane_view_frustum, view_box_frustum, DEFAULT_GRID, DEFAULT_HALF_EXTENT, DEFAULT_SPACING,
};
pub use scene_draw::{IndirectMode, SceneDrawStage};

#[cfg(test)]
mod tests;
