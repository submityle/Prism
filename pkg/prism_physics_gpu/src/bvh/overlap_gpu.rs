//! Real-device `wgpu` compute implementation of the batched
//! `AABB`-versus-`BVH` overlap gather query.
//!
//! [`GpuBvhOverlap`] compiles `shaders/bvh_overlap.wgsl` once and exposes
//! [`GpuBvhOverlap::query`], which uploads a built [`Lbvh`]'s node arrays and
//! bounds plus a batch of external query boxes, dispatches one invocation per
//! query to traverse the hierarchy, and reads each query's gathered primitive
//! indices back. The result is the same per-query set the
//! [`cpu_bvh_aabb_overlap`](super::overlap::cpu_bvh_aabb_overlap) twin
//! produces, reordered within each query by the device's atomic append; callers
//! compare by sorting each query's list.
//!
//! The tree is re-uploaded here rather than kept resident from the build, in
//! step with [`GpuBvhQuery`](super::query_gpu::GpuBvhQuery): the build currently
//! reads its arrays back to the host, so this query starts from that host-side
//! [`Lbvh`]. Keeping device buffers resident across build and query is a later
//! refinement; correctness of the on-device traversal is what this slice
//! establishes.
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

use crate::buffer;
use crate::context::GpuContext;

use super::config::Aabb;
use super::cpu::Lbvh;
use super::layout::{buffer_entry, entry};
use super::overlap::OverlapQueryError;
use super::resident::GpuResidentLbvh;

/// Lanes per workgroup; must match `@workgroup_size` in `bvh_overlap.wgsl`.
const WORKGROUP: u32 = 64;

/// Uniform parameters for the overlap kernel. Layout matches `Params` in
/// `shaders/bvh_overlap.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of internal nodes.
    num_internal: u32,
    /// Number of leaves.
    num_leaves: u32,
    /// Encoded id of the root node.
    root: u32,
    /// Number of external query boxes.
    num_queries: u32,
    /// Per-query output capacity, in primitive indices.
    capacity_per_query: u32,
}

/// A compiled, reusable `GPU` batched `AABB`-versus-`BVH` overlap query.
///
/// It carries two pipelines over one shared bind-group layout: [`query`] binds
/// a host-re-uploaded [`Lbvh`], and [`query_resident`] binds a tree left
/// resident on device by the build, with no host round-trip between build and
/// traversal. The two kernels differ only in how internal-node bounds and leaf
/// boxes are laid out (plain floats versus order-encoded `u32`s read through
/// `sorted_indices`); both bind the same count and types of buffers, so one
/// layout serves both.
///
/// [`query`]: GpuBvhOverlap::query
/// [`query_resident`]: GpuBvhOverlap::query_resident
pub struct GpuBvhOverlap {
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
    /// Overlap-gather kernel binding a device-resident tree's buffers directly.
    resident: ComputePipeline,
}

impl GpuBvhOverlap {
    /// Compiles the overlap kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhOverlap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_overlap"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_overlap.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_overlap_layout"),
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
                buffer_entry(9, BufferBindingType::Storage { read_only: true }),
                buffer_entry(10, BufferBindingType::Storage { read_only: true }),
                buffer_entry(11, BufferBindingType::Storage { read_only: false }),
                buffer_entry(12, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_bvh_overlap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_overlap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("find_overlaps"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        // The resident kernel binds the same count and types of buffers (only
        // the internal-bounds and leaf-box layouts differ, both still read-only
        // storage), so it reuses the one bind-group and pipeline layout above.
        let resident_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_overlap_resident"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_overlap_resident.wgsl").into()),
        });
        let resident = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_overlap_resident_pipeline"),
            layout: Some(&pipeline_layout),
            module: &resident_module,
            entry_point: Some("find_overlaps"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBvhOverlap {
            module,
            resident_module,
            layout,
            pipeline,
            resident,
        }
    }

    /// Gathers, for every box in `queries`, the original indices of the
    /// primitives in `lbvh` whose leaf box overlaps it, on device.
    ///
    /// The returned outer vector has one entry per query, in query order; each
    /// inner vector lists the overlapping primitives' original indices in the
    /// device's atomic-append order (sort to compare against the twin). An
    /// empty tree yields an empty list for every query; an empty `queries`
    /// slice yields an empty outer vector.
    ///
    /// `capacity_per_query` bounds each query's region of the output buffer; a
    /// query overlapping more leaves than that reports overflow rather than
    /// silently truncating, matching the
    /// [`cpu_bvh_aabb_overlap`](super::overlap::cpu_bvh_aabb_overlap) contract.
    ///
    /// # Errors
    ///
    /// Returns [`OverlapQueryError::CapacityExceeded`] naming the first query
    /// whose overlap count exceeded `capacity_per_query`.
    pub fn query(
        &self,
        ctx: &GpuContext,
        lbvh: &Lbvh,
        queries: &[Aabb],
        capacity_per_query: u32,
    ) -> Result<Vec<Vec<u32>>, OverlapQueryError> {
        // An empty tree overlaps nothing (every query empty), and an empty query
        // batch has nothing to dispatch; both short-circuit without touching the
        // device.
        if lbvh.num_leaves == 0 || queries.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        let device = ctx.device();

        // A single-leaf tree has empty child and internal-bounds arrays; wgpu
        // rejects zero-length storage buffers, so pad every node array to at
        // least one element. The kernel never reads these pads: a single-leaf
        // root is encoded as a leaf, visited once, and the child/internal reads
        // only fire for internal nodes.
        let left = pad_u32(&lbvh.left);
        let right = pad_u32(&lbvh.right);
        let parent = pad_u32(&lbvh.parent);
        let internal_min = pad_vec4(&pack_min(&lbvh.internal_aabb));
        let internal_max = pad_vec4(&pack_max(&lbvh.internal_aabb));
        let leaf_min = pack_min(&lbvh.leaf_aabb);
        let leaf_max = pack_max(&lbvh.leaf_aabb);
        let query_min = pack_min(queries);
        let query_max = pack_max(queries);

        let num_queries = u32::try_from(queries.len()).unwrap_or(u32::MAX);
        let params = Params {
            num_internal: u32::try_from(lbvh.num_internal).unwrap_or(u32::MAX),
            num_leaves: u32::try_from(lbvh.num_leaves).unwrap_or(u32::MAX),
            root: lbvh.root,
            num_queries,
            capacity_per_query,
        };

        let params_buf = buffer::uniform(device, "prism_bvh_overlap_params", &params);
        let left_buf = buffer::storage_read(device, "prism_bvh_overlap_left", &left);
        let right_buf = buffer::storage_read(device, "prism_bvh_overlap_right", &right);
        let parent_buf = buffer::storage_read(device, "prism_bvh_overlap_parent", &parent);
        let internal_min_buf =
            buffer::storage_read(device, "prism_bvh_overlap_internal_min", &internal_min);
        let internal_max_buf =
            buffer::storage_read(device, "prism_bvh_overlap_internal_max", &internal_max);
        let leaf_min_buf = buffer::storage_read(device, "prism_bvh_overlap_leaf_min", &leaf_min);
        let leaf_max_buf = buffer::storage_read(device, "prism_bvh_overlap_leaf_max", &leaf_max);
        let indices_buf =
            buffer::storage_read(device, "prism_bvh_overlap_indices", &lbvh.sorted_indices);
        let query_min_buf = buffer::storage_read(device, "prism_bvh_overlap_query_min", &query_min);
        let query_max_buf = buffer::storage_read(device, "prism_bvh_overlap_query_max", &query_max);

        let counts_bytes = u64::from(num_queries) * 4;
        let counts_buf = buffer::storage_rw_zeroed(device, "prism_bvh_overlap_counts", counts_bytes);
        let hits_bytes = u64::from(num_queries) * u64::from(capacity_per_query) * 4;
        // capacity_per_query may legitimately be zero (overflow-probe); keep the
        // buffer non-empty so the bind group is valid even then.
        let hits_buf =
            buffer::storage_rw_zeroed(device, "prism_bvh_overlap_hits", hits_bytes.max(4));

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_overlap_bind"),
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
                entry(9, &query_min_buf),
                entry(10, &query_max_buf),
                entry(11, &counts_buf),
                entry(12, &hits_buf),
            ],
        });

        let counts_stage = buffer::staging(device, "prism_bvh_overlap_counts_stage", counts_bytes);
        let hits_stage = buffer::staging(device, "prism_bvh_overlap_hits_stage", hits_bytes.max(4));

        let groups = num_queries.div_ceil(WORKGROUP);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_overlap_encoder"),
        });
        dispatch(&mut encoder, &self.pipeline, &bind, groups);
        buffer::copy(&mut encoder, &counts_buf, &counts_stage, counts_bytes);
        buffer::copy(&mut encoder, &hits_buf, &hits_stage, hits_bytes.max(4));
        ctx.queue().submit([encoder.finish()]);

        let counts = buffer::read_back::<u32>(ctx, &counts_stage);
        for (query_index, &count) in counts.iter().enumerate() {
            if count > capacity_per_query {
                return Err(OverlapQueryError::CapacityExceeded {
                    capacity_per_query,
                    query: u32::try_from(query_index).unwrap_or(u32::MAX),
                });
            }
        }

        let flat = buffer::read_back::<u32>(ctx, &hits_stage);
        let cap = capacity_per_query as usize;
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(queries.len());
        for (query_index, &count) in counts.iter().enumerate() {
            let base = query_index * cap;
            let n = count as usize;
            out.push(flat[base..base + n].to_vec());
        }
        Ok(out)
    }

    /// Gathers, for every box in `queries`, the original indices of the
    /// primitives in a device-resident `lbvh` whose leaf box overlaps it,
    /// binding the built tree's device buffers directly with no host round-trip
    /// between build and traversal.
    ///
    /// Behaves exactly like [`query`](GpuBvhOverlap::query) over the same scene:
    /// the returned outer vector has one entry per query in query order, each
    /// inner vector lists the overlapping primitives' original indices in the
    /// device's atomic-append order (sort to compare against the twin), and an
    /// empty `queries` slice yields an empty outer vector. A resident tree with
    /// fewer than two leaves owns no traversable hierarchy, so every query's
    /// hit list is empty.
    ///
    /// The resident tree carries its internal-node bounds order-encoded (the
    /// exact `u32`s the build's bounds pass wrote) and its leaf boxes as the
    /// original-order primitive floats gathered through `sorted_indices`; the
    /// resident kernel decodes the former with the integer inverse of that
    /// encoding, so its overlap tests, and thus every query's hit set, match
    /// [`query`](GpuBvhOverlap::query) and the
    /// [`cpu_bvh_aabb_overlap`](super::overlap::cpu_bvh_aabb_overlap) twin
    /// exactly.
    ///
    /// `capacity_per_query` bounds each query's region of the output buffer; a
    /// query overlapping more leaves than that reports overflow rather than
    /// silently truncating, matching the twin's contract.
    ///
    /// # Errors
    ///
    /// Returns [`OverlapQueryError::CapacityExceeded`] naming the first query
    /// whose overlap count exceeded `capacity_per_query`.
    pub fn query_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        queries: &[Aabb],
        capacity_per_query: u32,
    ) -> Result<Vec<Vec<u32>>, OverlapQueryError> {
        // A resident tree with fewer than two leaves owns no buffers and
        // overlaps nothing, so every query's hit list is empty.
        let Some(inner) = lbvh.buffers() else {
            return Ok(vec![Vec::new(); queries.len()]);
        };
        // An empty query batch has nothing to dispatch.
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let device = ctx.device();

        let query_min = pack_min(queries);
        let query_max = pack_max(queries);

        let num_queries = u32::try_from(queries.len()).unwrap_or(u32::MAX);
        let params = Params {
            num_internal: u32::try_from(inner.num_internal).unwrap_or(u32::MAX),
            num_leaves: u32::try_from(lbvh.num_leaves()).unwrap_or(u32::MAX),
            root: inner.root,
            num_queries,
            capacity_per_query,
        };

        let params_buf = buffer::uniform(device, "prism_bvh_overlap_resident_params", &params);
        let query_min_buf =
            buffer::storage_read(device, "prism_bvh_overlap_resident_query_min", &query_min);
        let query_max_buf =
            buffer::storage_read(device, "prism_bvh_overlap_resident_query_max", &query_max);

        let counts_bytes = u64::from(num_queries) * 4;
        let counts_buf =
            buffer::storage_rw_zeroed(device, "prism_bvh_overlap_resident_counts", counts_bytes);
        let hits_bytes = u64::from(num_queries) * u64::from(capacity_per_query) * 4;
        // capacity_per_query may legitimately be zero (overflow-probe); keep the
        // buffer non-empty so the bind group is valid even then.
        let hits_buf = buffer::storage_rw_zeroed(
            device,
            "prism_bvh_overlap_resident_hits",
            hits_bytes.max(4),
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_overlap_resident_bind"),
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
                entry(9, &query_min_buf),
                entry(10, &query_max_buf),
                entry(11, &counts_buf),
                entry(12, &hits_buf),
            ],
        });

        let counts_stage =
            buffer::staging(device, "prism_bvh_overlap_resident_counts_stage", counts_bytes);
        let hits_stage = buffer::staging(
            device,
            "prism_bvh_overlap_resident_hits_stage",
            hits_bytes.max(4),
        );

        let groups = num_queries.div_ceil(WORKGROUP);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_overlap_resident_encoder"),
        });
        dispatch(&mut encoder, &self.resident, &bind, groups);
        buffer::copy(&mut encoder, &counts_buf, &counts_stage, counts_bytes);
        buffer::copy(&mut encoder, &hits_buf, &hits_stage, hits_bytes.max(4));
        ctx.queue().submit([encoder.finish()]);

        let counts = buffer::read_back::<u32>(ctx, &counts_stage);
        for (query_index, &count) in counts.iter().enumerate() {
            if count > capacity_per_query {
                return Err(OverlapQueryError::CapacityExceeded {
                    capacity_per_query,
                    query: u32::try_from(query_index).unwrap_or(u32::MAX),
                });
            }
        }

        let flat = buffer::read_back::<u32>(ctx, &hits_stage);
        let cap = capacity_per_query as usize;
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(queries.len());
        for (query_index, &count) in counts.iter().enumerate() {
            let base = query_index * cap;
            let n = count as usize;
            out.push(flat[base..base + n].to_vec());
        }
        Ok(out)
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

/// Returns `data`, or a single zero element when it is empty, so the upload
/// never produces a zero-length storage buffer.
#[must_use]
fn pad_u32(data: &[u32]) -> Vec<u32> {
    if data.is_empty() {
        vec![0]
    } else {
        data.to_vec()
    }
}

/// Returns `data`, or a single zero `vec4` when it is empty, so the upload
/// never produces a zero-length storage buffer.
#[must_use]
fn pad_vec4(data: &[[f32; 4]]) -> Vec<[f32; 4]> {
    if data.is_empty() {
        vec![[0.0; 4]]
    } else {
        data.to_vec()
    }
}

/// Records one dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_bvh_overlap_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
