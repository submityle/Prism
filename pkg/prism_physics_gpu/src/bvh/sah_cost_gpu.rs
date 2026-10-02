//! Real-device `wgpu` Surface Area Heuristic cost of a resident `LBVH`.
//!
//! [`GpuBvhSahCost`] evaluates the same `SAH` cost as the
//! [`lbvh_sah_cost`](crate::bvh::lbvh_sah_cost) golden twin, but reads the tree
//! straight from the device buffers a [`GpuResidentLbvh`] already holds, so the
//! per-frame rebuild-versus-refit decision ([`RefitQualityTracker`](crate::bvh::RefitQualityTracker))
//! can run without a full host round-trip: only a single reduced scalar per sum
//! (and the three-lane root box) come back.
//!
//! The kernel (`shaders/bvh_sah_cost.wgsl`) folds surface areas with a
//! single-workgroup grid-stride reduction, so the sum order is fixed and the
//! result is deterministic across launches. Because a floating-point sum groups
//! differently from the host's sequential sum, parity with the twin is checked
//! within a tight relative tolerance rather than bit-for-bit — the same
//! treatment the `CFL` reducer gets.
//!
//! # Provenance
//!
//! The Surface Area Heuristic is Goldsmith and Salmon, "Automatic Creation of
//! Object Hierarchies for Ray Tracing" (IEEE CG&A 1987); a shared-memory tree
//! reduction is a standard `GPU` technique. No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::Aabb;
use super::gpu::from_order;
use super::layout::{buffer_entry, entry};
use super::quality::{surface_area, DEFAULT_INTERSECTION_COST, DEFAULT_TRAVERSAL_COST};
use super::resident::GpuResidentLbvh;

/// Source selector matching the kernel's `Params.source`.
const SOURCE_INTERNAL: u32 = 0;
const SOURCE_LEAF: u32 = 1;

/// Bytes of one order-encoded node box corner: three `u32` lanes.
const CORNER_LANES: u64 = 3;

/// Uniform parameters shared with `Params` in `shaders/bvh_sah_cost.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements to fold.
    count: u32,
    /// Source selector: internal nodes or leaf boxes.
    source: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable `GPU` `SAH`-cost reduction pipeline.
pub struct GpuBvhSahCost {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// Layout wiring params, the four input buffers, and the output slot.
    layout: BindGroupLayout,
    /// The single-workgroup surface-area sum kernel.
    reduce: ComputePipeline,
}

impl GpuBvhSahCost {
    /// Compiles the reduction kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhSahCost {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_sah_cost"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_sah_cost.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_sah_cost_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_bvh_sah_cost_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let reduce = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_sah_cost_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reduce"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBvhSahCost {
            module,
            layout,
            reduce,
        }
    }

    /// The default-weighted `SAH` cost of `tree`, read from its resident buffers.
    ///
    /// Matches [`lbvh_sah_cost`](crate::bvh::lbvh_sah_cost): an empty tree costs
    /// `0`, a single-leaf tree (no resident buffers) costs the intersection
    /// weight, and a degenerate point root falls back to the intersection-weighted
    /// leaf count.
    #[must_use]
    pub fn sah_cost(&self, ctx: &GpuContext, tree: &GpuResidentLbvh) -> f32 {
        let num_leaves = tree.num_leaves();
        if num_leaves == 0 {
            return 0.0;
        }
        let Some(inner) = tree.buffers() else {
            // Fewer than two leaves: the lone leaf is the root, so the normalised
            // cost is exactly the intersection weight, matching the twin.
            return DEFAULT_INTERSECTION_COST;
        };

        let device = ctx.device();
        let num_internal = inner.num_internal;

        let internal_params = buffer::uniform(
            device,
            "prism_bvh_sah_cost_internal_params",
            &Params {
                count: u32::try_from(num_internal).unwrap_or(u32::MAX),
                source: SOURCE_INTERNAL,
                pad0: 0,
                pad1: 0,
            },
        );
        let leaf_params = buffer::uniform(
            device,
            "prism_bvh_sah_cost_leaf_params",
            &Params {
                count: u32::try_from(num_leaves).unwrap_or(u32::MAX),
                source: SOURCE_LEAF,
                pad0: 0,
                pad1: 0,
            },
        );

        let scalar_bytes = size_of::<u32>() as u64;
        let internal_out =
            buffer::storage_rw_zeroed(device, "prism_bvh_sah_cost_internal_out", scalar_bytes);
        let leaf_out =
            buffer::storage_rw_zeroed(device, "prism_bvh_sah_cost_leaf_out", scalar_bytes);

        let internal_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_sah_cost_internal_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &internal_params),
                entry(1, &inner.node_min),
                entry(2, &inner.node_max),
                entry(3, &inner.aabb_min),
                entry(4, &inner.aabb_max),
                entry(5, &internal_out),
            ],
        });
        let leaf_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_sah_cost_leaf_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &leaf_params),
                entry(1, &inner.node_min),
                entry(2, &inner.node_max),
                entry(3, &inner.aabb_min),
                entry(4, &inner.aabb_max),
                entry(5, &leaf_out),
            ],
        });

        let internal_stage =
            buffer::staging(device, "prism_bvh_sah_cost_internal_stage", scalar_bytes);
        let leaf_stage = buffer::staging(device, "prism_bvh_sah_cost_leaf_stage", scalar_bytes);
        // Root is internal node 0, so its three order-encoded lanes sit at the
        // front of node_min / node_max.
        let corner_bytes = CORNER_LANES * size_of::<u32>() as u64;
        let root_min_stage =
            buffer::staging(device, "prism_bvh_sah_cost_root_min_stage", corner_bytes);
        let root_max_stage =
            buffer::staging(device, "prism_bvh_sah_cost_root_max_stage", corner_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_sah_cost_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_bvh_sah_cost_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce);
            // Each reduction uses a single workgroup and a grid-stride loop.
            pass.set_bind_group(0, &internal_bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_bind_group(0, &leaf_bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        buffer::copy(&mut encoder, &internal_out, &internal_stage, scalar_bytes);
        buffer::copy(&mut encoder, &leaf_out, &leaf_stage, scalar_bytes);
        buffer::copy(&mut encoder, &inner.node_min, &root_min_stage, corner_bytes);
        buffer::copy(&mut encoder, &inner.node_max, &root_max_stage, corner_bytes);
        ctx.queue().submit([encoder.finish()]);

        let internal_sa = f32::from_bits(buffer::read_back::<u32>(ctx, &internal_stage)[0]);
        let leaf_sa = f32::from_bits(buffer::read_back::<u32>(ctx, &leaf_stage)[0]);
        let root_min = buffer::read_back::<u32>(ctx, &root_min_stage);
        let root_max = buffer::read_back::<u32>(ctx, &root_max_stage);

        let root = Aabb::new(
            glam::Vec3::new(
                from_order(root_min[0]),
                from_order(root_min[1]),
                from_order(root_min[2]),
            ),
            glam::Vec3::new(
                from_order(root_max[0]),
                from_order(root_max[1]),
                from_order(root_max[2]),
            ),
        );
        let root_sa = surface_area(&root);
        if root_sa <= 0.0 {
            return DEFAULT_INTERSECTION_COST * num_leaves as f32;
        }
        DEFAULT_TRAVERSAL_COST.mul_add(internal_sa, DEFAULT_INTERSECTION_COST * leaf_sa) / root_sa
    }
}
