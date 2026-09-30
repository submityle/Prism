//! Real-device `wgpu` compute implementation of the `LBVH` build.
//!
//! [`GpuLbvh`] compiles `shaders/bvh_morton.wgsl`, `shaders/bvh_tree.wgsl`, and
//! `shaders/bvh_bbox.wgsl` once and exposes [`GpuLbvh::build`], which constructs
//! the whole hierarchy on the device in a single submission: a Morton kernel
//! writes each leaf's code and identity index, the sibling [`GpuRadixSort`]
//! stably orders leaves by code entirely on device, a tree kernel derives every
//! internal node's range, split, and children, and a bounds kernel walks each
//! leaf to the root expanding every ancestor's box with atomic per-component
//! `min`/`max` over an order-preserving float encoding. Only the result buffers
//! are read back.
//!
//! The build is a pure integer permutation plus exact `min`/`max` reductions, so
//! it matches the [`cpu_build_lbvh`](super::cpu::cpu_build_lbvh) golden twin
//! bit-for-bit.
//!
//! # On-device composition
//!
//! The Morton pass produces the sort keys and payloads in device buffers handed
//! straight to [`GpuRadixSort::record_sort`], which records its passes into the
//! same encoder without a host round-trip; the tree kernel then reads the sorted
//! codes and the bounds kernel reads the sorted index payload, both left in
//! place by the sort. The Morton bind group holds strong references to the key
//! and payload buffers, so they stay valid after their handles move into the
//! sort.
//!
//! # Provenance
//!
//! The Morton-code sort plus binary radix tree and bottom-up refit is Karras,
//! "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//! (High Performance Graphics 2012); the sibling radix sort follows Blelloch
//! (1990) and Satish, Harris, Garland (2009). No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::radix::gpu::SortedBuffers;
use crate::radix::GpuRadixSort;

use super::config::{Aabb, SceneBounds};
use super::cpu::{cpu_build_lbvh, Lbvh, NO_PARENT};
use super::layout::{buffer_entry, entry};
use super::morton::cpu_inv_extent;
use super::resident::GpuResidentLbvh;

/// Lanes per workgroup for every `LBVH` kernel; must match `@workgroup_size`.
const WORKGROUP: u32 = 64;

/// Uniform parameters shared with `Params` in `shaders/bvh_morton.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MortonParams {
    /// Scene-bounds minimum corner, `x`.
    min_x: f32,
    /// Scene-bounds minimum corner, `y`.
    min_y: f32,
    /// Scene-bounds minimum corner, `z`.
    min_z: f32,
    /// Number of leaves.
    n: u32,
    /// Reciprocal of the scene-bounds extent, `x` (zero for a degenerate axis).
    inv_ext_x: f32,
    /// Reciprocal of the scene-bounds extent, `y` (zero for a degenerate axis).
    inv_ext_y: f32,
    /// Reciprocal of the scene-bounds extent, `z` (zero for a degenerate axis).
    inv_ext_z: f32,
    /// Padding to a 16-byte boundary.
    pad: u32,
}

/// Uniform parameters shared with `Params` in the tree and bounds shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TreeParams {
    /// Number of leaves.
    n: u32,
    /// Number of internal nodes (`n - 1`).
    num_internal: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
}

/// A compiled, reusable `GPU` `LBVH` builder.
pub struct GpuLbvh {
    /// Kept alive so the Morton pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    morton_module: ShaderModule,
    /// Kept alive so the tree pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    tree_module: ShaderModule,
    /// Kept alive so the bounds pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    bbox_module: ShaderModule,
    /// Layout wiring params, leaf boxes, and the key/payload outputs.
    morton_layout: BindGroupLayout,
    /// Layout wiring params, sorted codes, and the tree outputs.
    tree_layout: BindGroupLayout,
    /// Layout wiring params, parent links, sorted indices, leaf boxes, and the
    /// order-encoded node-bounds outputs.
    bbox_layout: BindGroupLayout,
    /// Writes each leaf's Morton code and identity payload.
    morton: ComputePipeline,
    /// Builds the binary radix tree over the sorted codes.
    tree: ComputePipeline,
    /// Refits internal-node bounds bottom-up.
    bbox: ComputePipeline,
    /// The sibling radix sort that orders the leaves by Morton code.
    radix: GpuRadixSort,
}

/// Every device buffer and keepalive produced by recording an `LBVH` build
/// into a command encoder, before submission.
///
/// The build is recorded once by [`GpuLbvh::record_build`]; the caller then either
/// stages these buffers for readback into a host [`Lbvh`] or keeps them resident
/// to bind directly into a query. Every buffer and bind group is owned here so
/// it outlives the submission that runs the recorded passes.
pub(crate) struct RecordedBuild {
    /// Original-order primitive box minimum corners, `vec4` lanes (`w` unused).
    pub(crate) aabb_min: Buffer,
    /// Original-order primitive box maximum corners, `vec4` lanes (`w` unused).
    pub(crate) aabb_max: Buffer,
    /// Sorted radix output: keys are leaf Morton codes, values the leaf-slot
    /// primitive indices.
    pub(crate) sorted: SortedBuffers,
    /// Left child (encoded id) per internal node.
    pub(crate) left: Buffer,
    /// Right child (encoded id) per internal node.
    pub(crate) right: Buffer,
    /// Parent (encoded id) of every node, indexed by encoded id.
    pub(crate) parent: Buffer,
    /// Order-encoded internal-node minimum bounds, three lanes per node.
    pub(crate) node_min: Buffer,
    /// Order-encoded internal-node maximum bounds, three lanes per node.
    pub(crate) node_max: Buffer,
    /// Build bind groups kept alive until the recorded passes are submitted.
    #[expect(
        dead_code,
        reason = "kept alive so the recorded dispatches keep valid bindings until submit"
    )]
    pub(crate) binds: Vec<BindGroup>,
    /// Number of leaves.
    pub(crate) num_leaves: usize,
    /// Number of internal nodes (`num_leaves - 1`).
    pub(crate) num_internal: usize,
}

impl GpuLbvh {
    /// Compiles the Morton, tree, and bounds kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLbvh {
        let device = ctx.device();

        let morton_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_morton"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_morton.wgsl").into()),
        });
        let tree_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_tree"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_tree.wgsl").into()),
        });
        let bbox_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_bbox"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_bbox.wgsl").into()),
        });

        let rw = BufferBindingType::Storage { read_only: false };
        let ro = BufferBindingType::Storage { read_only: true };

        let morton_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_morton_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, ro),
                buffer_entry(2, ro),
                buffer_entry(3, rw),
                buffer_entry(4, rw),
            ],
        });
        let tree_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_tree_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, ro),
                buffer_entry(2, rw),
                buffer_entry(3, rw),
                buffer_entry(4, rw),
            ],
        });
        let bbox_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_bbox_layout"),
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

        let morton = build_pipeline(device, &morton_layout, &morton_module, "morton");
        let tree = build_pipeline(device, &tree_layout, &tree_module, "build_tree");
        let bbox = build_pipeline(device, &bbox_layout, &bbox_module, "build_bounds");

        GpuLbvh {
            morton_module,
            tree_module,
            bbox_module,
            morton_layout,
            tree_layout,
            bbox_layout,
            morton,
            tree,
            bbox,
            radix: GpuRadixSort::new(ctx),
        }
    }

    /// Builds the complete `LBVH` over `boxes` on the device and reads it back.
    ///
    /// Records the build into one encoder, stages every result buffer, submits,
    /// and assembles a host [`Lbvh`]. Empty and single-leaf inputs have no
    /// parallel work and are built directly by the golden twin so the trivial
    /// cases stay identical. To keep the built tree on device without a host
    /// round-trip, use [`build_resident`](GpuLbvh::build_resident) instead.
    #[must_use]
    pub fn build(&self, ctx: &GpuContext, boxes: &[Aabb]) -> Lbvh {
        let n = boxes.len();
        if n <= 1 {
            return cpu_build_lbvh(boxes);
        }
        let num_internal = n - 1;
        let total_nodes = num_internal + n;

        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_encoder"),
        });
        let rec = self.record_build(device, &mut encoder, boxes);

        let key_bytes = (n * size_of::<u32>()) as u64;
        let node_link_bytes = (num_internal * size_of::<u32>()) as u64;
        let parent_bytes = (total_nodes * size_of::<u32>()) as u64;
        let node_box_bytes = (num_internal * 3 * size_of::<u32>()) as u64;

        // Stage every result buffer for readback.
        let idx_stage = buffer::staging(device, "prism_bvh_indices_stage", key_bytes);
        let codes_stage = buffer::staging(device, "prism_bvh_codes_stage", key_bytes);
        let left_stage = buffer::staging(device, "prism_bvh_left_stage", node_link_bytes);
        let right_stage = buffer::staging(device, "prism_bvh_right_stage", node_link_bytes);
        let parent_stage = buffer::staging(device, "prism_bvh_parent_stage", parent_bytes);
        let node_min_stage = buffer::staging(device, "prism_bvh_node_min_stage", node_box_bytes);
        let node_max_stage = buffer::staging(device, "prism_bvh_node_max_stage", node_box_bytes);
        buffer::copy(&mut encoder, rec.sorted.values(), &idx_stage, key_bytes);
        buffer::copy(&mut encoder, rec.sorted.keys(), &codes_stage, key_bytes);
        buffer::copy(&mut encoder, &rec.left, &left_stage, node_link_bytes);
        buffer::copy(&mut encoder, &rec.right, &right_stage, node_link_bytes);
        buffer::copy(&mut encoder, &rec.parent, &parent_stage, parent_bytes);
        buffer::copy(&mut encoder, &rec.node_min, &node_min_stage, node_box_bytes);
        buffer::copy(&mut encoder, &rec.node_max, &node_max_stage, node_box_bytes);
        ctx.queue().submit([encoder.finish()]);

        let sorted_indices = buffer::read_back::<u32>(ctx, &idx_stage);
        let sorted_codes = buffer::read_back::<u32>(ctx, &codes_stage);
        let left = buffer::read_back::<u32>(ctx, &left_stage);
        let right = buffer::read_back::<u32>(ctx, &right_stage);
        let parent = buffer::read_back::<u32>(ctx, &parent_stage);
        let node_min = buffer::read_back::<u32>(ctx, &node_min_stage);
        let node_max = buffer::read_back::<u32>(ctx, &node_max_stage);
        drop(rec);

        let internal_aabb: Vec<Aabb> = (0..num_internal)
            .map(|i| {
                let base = i * 3;
                Aabb::new(
                    Vec3::new(
                        from_order(node_min[base]),
                        from_order(node_min[base + 1]),
                        from_order(node_min[base + 2]),
                    ),
                    Vec3::new(
                        from_order(node_max[base]),
                        from_order(node_max[base + 1]),
                        from_order(node_max[base + 2]),
                    ),
                )
            })
            .collect();
        let leaf_aabb: Vec<Aabb> = sorted_indices.iter().map(|&i| boxes[i as usize]).collect();

        Lbvh {
            num_leaves: n,
            num_internal,
            root: 0,
            sorted_indices,
            sorted_codes,
            left,
            right,
            parent,
            internal_aabb,
            leaf_aabb,
        }
    }

    /// Builds the complete `LBVH` over `boxes` and keeps it resident on the
    /// device for direct query consumption, with no host round-trip.
    ///
    /// Unlike [`build`](GpuLbvh::build), no result buffer is staged or read back:
    /// the node links, order-encoded internal-node bounds, original-order
    /// primitive boxes, and sorted leaf-slot indices stay in device memory,
    /// ready to bind straight into
    /// [`query_resident`](crate::bvh::GpuBvhQuery::query_resident). Empty and
    /// single-leaf inputs have no traversable hierarchy, so they yield an empty
    /// resident tree that produces no pairs.
    #[must_use]
    pub fn build_resident(&self, ctx: &GpuContext, boxes: &[Aabb]) -> GpuResidentLbvh {
        let n = boxes.len();
        if n <= 1 {
            return GpuResidentLbvh::empty(n);
        }
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_resident_encoder"),
        });
        let rec = self.record_build(device, &mut encoder, boxes);
        ctx.queue().submit([encoder.finish()]);
        GpuResidentLbvh::from_recorded(rec)
    }

    /// Records the whole `LBVH` build over `boxes` into `encoder` without
    /// submitting or reading anything back.
    ///
    /// Returns every device buffer the build produced, together with the bind
    /// groups and the on-device sort kept alive until the caller submits
    /// `encoder`. Callers either stage those buffers for readback into a host
    /// [`Lbvh`] (see [`build`](GpuLbvh::build)) or keep them resident to feed a
    /// query directly (see [`build_resident`](GpuLbvh::build_resident)). `boxes`
    /// must hold at least two entries, the only case with parallel work; both
    /// callers special-case the trivial sizes before recording.
    pub(crate) fn record_build(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        boxes: &[Aabb],
    ) -> RecordedBuild {
        let n = boxes.len();
        let num_internal = n - 1;
        let total_nodes = num_internal + n;

        let bounds = SceneBounds::of(boxes).unwrap_or(SceneBounds {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        });
        let extent = bounds.extent();

        let packed_min: Vec<[f32; 4]> = boxes
            .iter()
            .map(|b| [b.min.x, b.min.y, b.min.z, 0.0])
            .collect();
        let packed_max: Vec<[f32; 4]> = boxes
            .iter()
            .map(|b| [b.max.x, b.max.y, b.max.z, 0.0])
            .collect();
        let aabb_min = buffer::storage_read(device, "prism_bvh_aabb_min", &packed_min);
        let aabb_max = buffer::storage_read(device, "prism_bvh_aabb_max", &packed_max);

        let key_bytes = (n * size_of::<u32>()) as u64;
        let keys_buf = buffer::storage_rw_zeroed(device, "prism_bvh_keys", key_bytes);
        let idx_buf = buffer::storage_rw_zeroed(device, "prism_bvh_indices", key_bytes);

        let node_link_bytes = (num_internal * size_of::<u32>()) as u64;
        let left_buf = buffer::storage_rw_zeroed(device, "prism_bvh_left", node_link_bytes);
        let right_buf = buffer::storage_rw_zeroed(device, "prism_bvh_right", node_link_bytes);

        let parent_init = vec![NO_PARENT; total_nodes];
        let parent_buf = buffer::storage_rw_init(device, "prism_bvh_parent", &parent_init);

        let node_lane_count = num_internal * 3;
        let node_box_bytes = (node_lane_count * size_of::<u32>()) as u64;
        let node_min_init = vec![0xffff_ffffu32; node_lane_count];
        let node_min_buf = buffer::storage_rw_init(device, "prism_bvh_node_min", &node_min_init);
        let node_max_buf = buffer::storage_rw_zeroed(device, "prism_bvh_node_max", node_box_bytes);

        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        let ni_u32 = u32::try_from(num_internal).unwrap_or(u32::MAX);
        let morton_params = buffer::uniform(
            device,
            "prism_bvh_morton_params",
            &MortonParams {
                min_x: bounds.min.x,
                min_y: bounds.min.y,
                min_z: bounds.min.z,
                n: n_u32,
                inv_ext_x: cpu_inv_extent(extent.x),
                inv_ext_y: cpu_inv_extent(extent.y),
                inv_ext_z: cpu_inv_extent(extent.z),
                pad: 0,
            },
        );
        let tree_params = buffer::uniform(
            device,
            "prism_bvh_tree_params",
            &TreeParams {
                n: n_u32,
                num_internal: ni_u32,
                pad0: 0,
                pad1: 0,
            },
        );

        // Morton pass: write per-leaf codes and identity payloads. The bind
        // group holds strong references to `keys_buf`/`idx_buf`, so they stay
        // valid after their handles move into the sort below.
        let leaf_groups = n_u32.div_ceil(WORKGROUP);
        let morton_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_morton_bind"),
            layout: &self.morton_layout,
            entries: &[
                entry(0, &morton_params),
                entry(1, &aabb_min),
                entry(2, &aabb_max),
                entry(3, &keys_buf),
                entry(4, &idx_buf),
            ],
        });
        dispatch(
            encoder,
            "prism_bvh_morton_pass",
            &self.morton,
            &morton_bind,
            leaf_groups,
        );

        // Sort leaves by Morton code on device: sorted keys are the codes the
        // tree kernel reads; sorted values are the leaf-order primitive indices.
        let sorted = self
            .radix
            .record_sort(device, encoder, keys_buf, idx_buf, n);

        // Tree pass: build the binary radix tree over the sorted codes.
        let internal_groups = ni_u32.div_ceil(WORKGROUP);
        let tree_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_tree_bind"),
            layout: &self.tree_layout,
            entries: &[
                entry(0, &tree_params),
                entry(1, sorted.keys()),
                entry(2, &left_buf),
                entry(3, &right_buf),
                entry(4, &parent_buf),
            ],
        });
        dispatch(
            encoder,
            "prism_bvh_tree_pass",
            &self.tree,
            &tree_bind,
            internal_groups,
        );

        // Bounds pass: refit internal-node boxes bottom-up.
        let bbox_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_bbox_bind"),
            layout: &self.bbox_layout,
            entries: &[
                entry(0, &tree_params),
                entry(1, &parent_buf),
                entry(2, sorted.values()),
                entry(3, &aabb_min),
                entry(4, &aabb_max),
                entry(5, &node_min_buf),
                entry(6, &node_max_buf),
            ],
        });
        dispatch(
            encoder,
            "prism_bvh_bbox_pass",
            &self.bbox,
            &bbox_bind,
            leaf_groups,
        );

        RecordedBuild {
            aabb_min,
            aabb_max,
            sorted,
            left: left_buf,
            right: right_buf,
            parent: parent_buf,
            node_min: node_min_buf,
            node_max: node_max_buf,
            binds: vec![morton_bind, tree_bind, bbox_bind],
            num_leaves: n,
            num_internal,
        }
    }
}

/// Decodes an order-preserving `u32` produced by `to_order` in
/// `shaders/bvh_bbox.wgsl` back into its original `f32`.
///
/// Inverts the sortable-float encoding: a set high bit marks an originally
/// non-negative value (clear the sign bit), otherwise the value was negative
/// (flip every bit).
#[must_use]
fn from_order(u: u32) -> f32 {
    if u & 0x8000_0000 != 0 {
        f32::from_bits(u & 0x7fff_ffff)
    } else {
        f32::from_bits(!u)
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
        label: Some("prism_bvh_pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("prism_bvh_pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Records one block-granular dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut CommandEncoder,
    label: &str,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
