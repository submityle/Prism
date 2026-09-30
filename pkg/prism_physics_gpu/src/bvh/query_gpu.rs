//! Real-device `wgpu` compute implementation of the `BVH` overlap-pair query.
//!
//! [`GpuBvhQuery`] compiles `shaders/bvh_pairs.wgsl` once and exposes
//! [`GpuBvhQuery::query`], which uploads a built [`Lbvh`]'s node arrays and
//! bounds, dispatches one invocation per leaf to traverse the hierarchy, and
//! reads the appended overlap pairs back. The result is the same candidate set
//! the [`cpu_bvh_pairs`](super::query::cpu_bvh_pairs) twin produces, reordered
//! by the device's atomic append; callers compare by sorting both.
//!
//! The tree is re-uploaded here rather than kept resident from the build: the
//! build currently reads its arrays back to the host, so this query starts from
//! that host-side [`Lbvh`]. Keeping the device buffers resident to feed the
//! query directly is a later refinement; correctness of the on-device traversal
//! is what this slice establishes.
//!
//! # Provenance
//!
//! Stackless parent-pointer traversal (Hapala et al. 2011) over the linear
//! `BVH` of Karras (2012); standard `wgpu` compute dispatch. No Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::broadphase::CandidatePair;
use crate::buffer;
use crate::context::GpuContext;

use super::config::Aabb;
use super::cpu::Lbvh;
use super::layout::{buffer_entry, entry};
use super::query::BvhQueryError;
use super::resident::GpuResidentLbvh;

/// Lanes per workgroup; must match `@workgroup_size` in `bvh_pairs.wgsl`.
const WORKGROUP: u32 = 64;

/// Uniform parameters for the query kernel. Layout matches `Params` in
/// `shaders/bvh_pairs.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of internal nodes.
    num_internal: u32,
    /// Number of leaves.
    num_leaves: u32,
    /// Encoded id of the root node.
    root: u32,
    /// Output-buffer capacity, in pairs.
    capacity: u32,
}

/// A compiled, reusable `GPU` `BVH` overlap-pair query pipeline.
pub struct GpuBvhQuery {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// Kept alive so the resident-tree pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the resident pipeline it produced stays valid"
    )]
    resident_module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
    /// Overlap-pair kernel binding a device-resident tree's buffers directly.
    resident: ComputePipeline,
}

impl GpuBvhQuery {
    /// Compiles the query kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhQuery {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_pairs"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_pairs.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_pairs_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_bvh_pairs_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_pairs_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("find_pairs"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let resident_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_pairs_resident"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_pairs_resident.wgsl").into()),
        });
        let resident = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_pairs_resident_pipeline"),
            layout: Some(&pipeline_layout),
            module: &resident_module,
            entry_point: Some("find_pairs"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBvhQuery {
            module,
            resident_module,
            layout,
            pipeline,
            resident,
        }
    }

    /// Runs the overlap-pair query over `lbvh` on device and returns the pairs.
    ///
    /// `capacity` bounds the output buffer; the query reports overflow rather
    /// than silently dropping pairs past the end.
    ///
    /// # Errors
    ///
    /// Returns [`BvhQueryError::PairCapacityExceeded`] when the device found
    /// more pairs than `capacity`, matching the `CPU` twin's overflow contract.
    pub fn query(
        &self,
        ctx: &GpuContext,
        lbvh: &Lbvh,
        capacity: u32,
    ) -> Result<Vec<CandidatePair>, BvhQueryError> {
        // Fewer than two leaves can produce no pair, and the traversal needs an
        // internal root, so return early without a dispatch.
        if lbvh.num_leaves < 2 {
            return Ok(Vec::new());
        }

        let device = ctx.device();

        let internal_min = pack_min(&lbvh.internal_aabb);
        let internal_max = pack_max(&lbvh.internal_aabb);
        let leaf_min = pack_min(&lbvh.leaf_aabb);
        let leaf_max = pack_max(&lbvh.leaf_aabb);

        let params = Params {
            num_internal: u32::try_from(lbvh.num_internal).unwrap_or(u32::MAX),
            num_leaves: u32::try_from(lbvh.num_leaves).unwrap_or(u32::MAX),
            root: lbvh.root,
            capacity,
        };

        let params_buf = buffer::uniform(device, "prism_bvh_pairs_params", &params);
        let left_buf = buffer::storage_read(device, "prism_bvh_pairs_left", &lbvh.left);
        let right_buf = buffer::storage_read(device, "prism_bvh_pairs_right", &lbvh.right);
        let parent_buf = buffer::storage_read(device, "prism_bvh_pairs_parent", &lbvh.parent);
        let internal_min_buf =
            buffer::storage_read(device, "prism_bvh_pairs_internal_min", &internal_min);
        let internal_max_buf =
            buffer::storage_read(device, "prism_bvh_pairs_internal_max", &internal_max);
        let leaf_min_buf = buffer::storage_read(device, "prism_bvh_pairs_leaf_min", &leaf_min);
        let leaf_max_buf = buffer::storage_read(device, "prism_bvh_pairs_leaf_max", &leaf_max);
        let indices_buf =
            buffer::storage_read(device, "prism_bvh_pairs_indices", &lbvh.sorted_indices);
        let count_buf = buffer::storage_rw_zeroed(device, "prism_bvh_pairs_count", 4);
        let pairs_bytes = u64::from(capacity) * 8;
        let pairs_buf = buffer::storage_rw_zeroed(device, "prism_bvh_pairs_out", pairs_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_pairs_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &left_buf),
                entry(2, &right_buf),
                entry(3, &parent_buf),
                entry(4, &internal_min_buf),
                entry(5, &internal_max_buf),
                entry(6, &leaf_min_buf),
                entry(7, &leaf_max_buf),
                entry(8, &indices_buf),
                entry(9, &count_buf),
                entry(10, &pairs_buf),
            ],
        });

        let count_stage = buffer::staging(device, "prism_bvh_pairs_count_stage", 4);
        let pairs_stage = buffer::staging(device, "prism_bvh_pairs_out_stage", pairs_bytes);

        let groups = params.num_leaves.div_ceil(WORKGROUP);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_pairs_encoder"),
        });
        dispatch(&mut encoder, &self.pipeline, &bind, groups);
        buffer::copy(&mut encoder, &count_buf, &count_stage, 4);
        buffer::copy(&mut encoder, &pairs_buf, &pairs_stage, pairs_bytes);
        ctx.queue().submit([encoder.finish()]);

        let total = buffer::read_back::<u32>(ctx, &count_stage)[0];
        if total > capacity {
            return Err(BvhQueryError::PairCapacityExceeded { capacity });
        }
        let raw = buffer::read_back::<[u32; 2]>(ctx, &pairs_stage);
        let pairs = raw
            .into_iter()
            .take(total as usize)
            .map(|[i, j]| CandidatePair::new(i, j))
            .collect();
        Ok(pairs)
    }

    /// Runs the overlap-pair query over a device-resident `lbvh` and returns the
    /// pairs, binding the built tree's device buffers directly with no host
    /// round-trip between build and traversal.
    ///
    /// The resident tree carries its internal-node bounds order-encoded (the
    /// exact `u32`s the build's bounds pass wrote); the resident kernel decodes
    /// them in-shader with the integer inverse of that encoding, so the overlap
    /// tests, and thus the pair set, match [`query`](GpuBvhQuery::query) and the
    /// [`cpu_bvh_pairs`](super::query::cpu_bvh_pairs) twin exactly.
    ///
    /// `capacity` bounds the output buffer; the query reports overflow rather
    /// than silently dropping pairs past the end. A resident tree with fewer
    /// than two leaves holds no traversable hierarchy and yields no pairs.
    ///
    /// # Errors
    ///
    /// Returns [`BvhQueryError::PairCapacityExceeded`] when the device found
    /// more pairs than `capacity`, matching the `CPU` twin's overflow contract.
    pub fn query_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        capacity: u32,
    ) -> Result<Vec<CandidatePair>, BvhQueryError> {
        // A resident tree with fewer than two leaves owns no buffers and has no
        // internal root to traverse, so there is nothing to dispatch.
        let Some(inner) = lbvh.buffers() else {
            return Ok(Vec::new());
        };

        let device = ctx.device();

        let params = Params {
            num_internal: u32::try_from(inner.num_internal).unwrap_or(u32::MAX),
            num_leaves: u32::try_from(lbvh.num_leaves()).unwrap_or(u32::MAX),
            root: inner.root,
            capacity,
        };

        let params_buf = buffer::uniform(device, "prism_bvh_pairs_resident_params", &params);
        let count_buf = buffer::storage_rw_zeroed(device, "prism_bvh_pairs_resident_count", 4);
        let pairs_bytes = u64::from(capacity) * 8;
        let pairs_buf =
            buffer::storage_rw_zeroed(device, "prism_bvh_pairs_resident_out", pairs_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_pairs_resident_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &inner.left),
                entry(2, &inner.right),
                entry(3, &inner.parent),
                entry(4, &inner.node_min),
                entry(5, &inner.node_max),
                entry(6, &inner.aabb_min),
                entry(7, &inner.aabb_max),
                entry(8, inner.sorted.values()),
                entry(9, &count_buf),
                entry(10, &pairs_buf),
            ],
        });

        let count_stage = buffer::staging(device, "prism_bvh_pairs_resident_count_stage", 4);
        let pairs_stage =
            buffer::staging(device, "prism_bvh_pairs_resident_out_stage", pairs_bytes);

        let groups = params.num_leaves.div_ceil(WORKGROUP);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_pairs_resident_encoder"),
        });
        dispatch(&mut encoder, &self.resident, &bind, groups);
        buffer::copy(&mut encoder, &count_buf, &count_stage, 4);
        buffer::copy(&mut encoder, &pairs_buf, &pairs_stage, pairs_bytes);
        ctx.queue().submit([encoder.finish()]);

        let total = buffer::read_back::<u32>(ctx, &count_stage)[0];
        if total > capacity {
            return Err(BvhQueryError::PairCapacityExceeded { capacity });
        }
        let raw = buffer::read_back::<[u32; 2]>(ctx, &pairs_stage);
        let pairs = raw
            .into_iter()
            .take(total as usize)
            .map(|[i, j]| CandidatePair::new(i, j))
            .collect();
        Ok(pairs)
    }
}

/// Packs box minimum corners into `vec4` lanes (`w` unused) for upload.
#[must_use]
fn pack_min(boxes: &[Aabb]) -> Vec<[f32; 4]> {
    boxes
        .iter()
        .map(|b| [b.min.x, b.min.y, b.min.z, 0.0])
        .collect()
}

/// Packs box maximum corners into `vec4` lanes (`w` unused) for upload.
#[must_use]
fn pack_max(boxes: &[Aabb]) -> Vec<[f32; 4]> {
    boxes
        .iter()
        .map(|b| [b.max.x, b.max.y, b.max.z, 0.0])
        .collect()
}

/// Records one dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_bvh_pairs_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
