//! Real-device `wgpu` incremental refit of a resident [`GpuResidentLbvh`].
//!
//! [`GpuBvhRefit`] updates a built tree's bounds for a new set of primitive
//! boxes without rebuilding its topology. It overwrites the resident
//! original-order primitive boxes in place, resets the order-encoded
//! internal-node bounds to the atomic identity with a clear kernel
//! (`shaders/bvh_refit_clear.wgsl`), then re-runs the same bottom-up bounds
//! kernel the full build uses (`shaders/bvh_bbox.wgsl`) over the unchanged
//! parent links and sorted leaf payload. The Morton, sort, and radix-tree
//! passes are skipped entirely, so a refit is far cheaper than a rebuild while
//! the tree's leaf ordering stays valid.
//!
//! Because the refit climbs the identical parent links and unions the identical
//! way as the full build, the resulting bounds are the exact componentwise
//! min/max of each node's descendant leaves, matching the
//! [`cpu_refit_lbvh`](crate::bvh::cpu_refit_lbvh) golden twin bit-for-bit (the
//! pipeline is only integer permutation plus exact min/max reductions).
//!
//! # Provenance
//!
//! Bottom-up bounds refit over the linear `BVH` of Karras, "Maximizing
//! Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (High
//! Performance Graphics 2012); the sortable-float encoding for atomic min/max is
//! the classical radix-float bit flip. No Unreal Engine source or derived code.

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
use super::resident::GpuResidentLbvh;

const WORKGROUP: u32 = 64;

/// Uniform for the clear kernel: how many internal nodes to reset.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearParams {
    num_internal: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform for the bounds kernel; layout matches `Params` in
/// `shaders/bvh_bbox.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BboxParams {
    n: u32,
    num_internal: u32,
    _pad0: u32,
    _pad1: u32,
}

/// A compiled, reusable incremental-refit pipeline pair for resident `LBVH`
/// trees.
pub struct GpuBvhRefit {
    #[expect(
        dead_code,
        reason = "kept alive so the clear pipeline it produced stays valid"
    )]
    clear_module: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the bounds pipeline it produced stays valid"
    )]
    bbox_module: ShaderModule,
    clear_layout: BindGroupLayout,
    bbox_layout: BindGroupLayout,
    clear: ComputePipeline,
    bbox: ComputePipeline,
}

impl GpuBvhRefit {
    /// Compiles the clear and bounds kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhRefit {
        let device = ctx.device();

        let clear_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_refit_clear"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_refit_clear.wgsl").into()),
        });
        let bbox_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_refit_bbox"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_bbox.wgsl").into()),
        });

        let rw = BufferBindingType::Storage { read_only: false };
        let ro = BufferBindingType::Storage { read_only: true };

        let clear_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_refit_clear_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, rw),
                buffer_entry(2, rw),
            ],
        });
        let bbox_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_refit_bbox_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, ro),
                buffer_entry(2, ro),
                buffer_entry(3, ro),
                buffer_entry(4, ro),
                buffer_entry(5, rw),
                buffer_entry(6, rw),
            ],
        });

        let clear = build_pipeline(device, &clear_layout, &clear_module, "clear_bounds");
        let bbox = build_pipeline(device, &bbox_layout, &bbox_module, "build_bounds");

        GpuBvhRefit {
            clear_module,
            bbox_module,
            clear_layout,
            bbox_layout,
            clear,
            bbox,
        }
    }

    /// Refits `tree` in place for `new_boxes`, keeping its topology.
    ///
    /// Overwrites the resident primitive boxes, resets the node bounds to the
    /// atomic identity, and re-runs the bottom-up bounds kernel. Trees with
    /// fewer than two leaves have no internal bounds and no resident buffers, so
    /// the call is a no-op for them. `new_boxes` is indexed in the original
    /// primitive order, exactly as the slice the tree was built from.
    ///
    /// # Panics
    ///
    /// Panics if `new_boxes.len()` differs from the tree's leaf count.
    pub fn refit(&self, ctx: &GpuContext, tree: &GpuResidentLbvh, new_boxes: &[Aabb]) {
        assert_eq!(
            new_boxes.len(),
            tree.num_leaves(),
            "refit box count must match the tree's leaf count"
        );
        let Some(inner) = tree.buffers() else {
            // Fewer than two leaves: no internal nodes, nothing to refit.
            return;
        };

        let device = ctx.device();
        let queue = ctx.queue();

        // Overwrite the resident primitive boxes in place (original order).
        let packed_min: Vec<[f32; 4]> = new_boxes
            .iter()
            .map(|b| [b.min.x, b.min.y, b.min.z, 0.0])
            .collect();
        let packed_max: Vec<[f32; 4]> = new_boxes
            .iter()
            .map(|b| [b.max.x, b.max.y, b.max.z, 0.0])
            .collect();
        queue.write_buffer(&inner.aabb_min, 0, bytemuck::cast_slice(&packed_min));
        queue.write_buffer(&inner.aabb_max, 0, bytemuck::cast_slice(&packed_max));

        let num_internal = inner.num_internal;
        let n = tree.num_leaves();
        let ni_u32 = u32::try_from(num_internal).unwrap_or(u32::MAX);
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);

        let clear_params = buffer::uniform(
            device,
            "prism_bvh_refit_clear_params",
            &ClearParams {
                num_internal: ni_u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let bbox_params = buffer::uniform(
            device,
            "prism_bvh_refit_bbox_params",
            &BboxParams {
                n: n_u32,
                num_internal: ni_u32,
                _pad0: 0,
                _pad1: 0,
            },
        );

        let clear_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_refit_clear_bind"),
            layout: &self.clear_layout,
            entries: &[
                entry(0, &clear_params),
                entry(1, &inner.node_min),
                entry(2, &inner.node_max),
            ],
        });
        let bbox_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_refit_bbox_bind"),
            layout: &self.bbox_layout,
            entries: &[
                entry(0, &bbox_params),
                entry(1, &inner.parent),
                entry(2, inner.sorted.values()),
                entry(3, &inner.aabb_min),
                entry(4, &inner.aabb_max),
                entry(5, &inner.node_min),
                entry(6, &inner.node_max),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_refit_encoder"),
        });
        // Reset the node bounds to identity, then refit them bottom-up. The two
        // passes are separate so every clear write is visible to the first
        // atomic accumulation.
        dispatch(
            &mut encoder,
            "prism_bvh_refit_clear_pass",
            &self.clear,
            &clear_bind,
            ni_u32.div_ceil(WORKGROUP),
        );
        dispatch(
            &mut encoder,
            "prism_bvh_refit_bbox_pass",
            &self.bbox,
            &bbox_bind,
            n_u32.div_ceil(WORKGROUP),
        );
        queue.submit([encoder.finish()]);
    }

    /// Reads a resident tree's internal-node bounds back to the host.
    ///
    /// Returns one [`Aabb`] per internal node, decoded from the order-encoded
    /// device buffers; an empty vector for a tree with fewer than two leaves.
    /// Intended for verification and debugging, not a per-frame path.
    #[must_use]
    pub fn read_internal_aabb(&self, ctx: &GpuContext, tree: &GpuResidentLbvh) -> Vec<Aabb> {
        let Some(inner) = tree.buffers() else {
            return Vec::new();
        };
        let num_internal = inner.num_internal;
        let lane_bytes = (num_internal * 3 * size_of::<u32>()) as u64;
        let device = ctx.device();
        let min_stage = buffer::staging(device, "prism_bvh_refit_min_stage", lane_bytes);
        let max_stage = buffer::staging(device, "prism_bvh_refit_max_stage", lane_bytes);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_refit_readback_encoder"),
        });
        buffer::copy(&mut encoder, &inner.node_min, &min_stage, lane_bytes);
        buffer::copy(&mut encoder, &inner.node_max, &max_stage, lane_bytes);
        ctx.queue().submit([encoder.finish()]);

        let node_min = buffer::read_back::<u32>(ctx, &min_stage);
        let node_max = buffer::read_back::<u32>(ctx, &max_stage);
        (0..num_internal)
            .map(|i| {
                let base = i * 3;
                Aabb::new(
                    glam::Vec3::new(
                        from_order(node_min[base]),
                        from_order(node_min[base + 1]),
                        from_order(node_min[base + 2]),
                    ),
                    glam::Vec3::new(
                        from_order(node_max[base]),
                        from_order(node_max[base + 1]),
                        from_order(node_max[base + 2]),
                    ),
                )
            })
            .collect()
    }
}

/// Builds a single-entry-point compute pipeline for one kernel.
#[must_use]
fn build_pipeline(
    device: &wgpu::Device,
    layout: &BindGroupLayout,
    module: &ShaderModule,
    entry_point: &str,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("prism_bvh_refit_pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("prism_bvh_refit_pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Records one block-granular dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &ComputePipeline,
    bind: &wgpu::BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups.max(1), 1, 1);
}
