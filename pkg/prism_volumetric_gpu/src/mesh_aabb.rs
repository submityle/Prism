//! `wgpu` compute twin of the mesh axis-aligned bounds reduction
//! ([`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb),
//! design §8 mesh-surface emission).
//!
//! The `CPU` golden
//! [`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb)
//! folds every vertex `position` into an axis-aligned box, taking the
//! component-wise `min` and `max` across all three axes, and returns
//! `(Vec3::ZERO, Vec3::ZERO)` for an empty mesh. [`GpuMeshAabb`] is the
//! on-device twin: the device reduces each contiguous block of vertices into a
//! partial `(min, max)` and the host folds the handful of per-block partials
//! into the final box, so a passing real-device parity test is direct evidence
//! the ported reduction computes the same bounds the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of vertex positions, the kernel reproduces the component-wise
//! `min` / `max` reduction the golden performs. Both `min` and `max` are
//! associative and commutative, and each simply *selects* one of its two `f32`
//! inputs — it performs no arithmetic — so the reduction is bit-exact
//! regardless of the fold order. The device therefore reduces each block in any
//! intra-block order and the host folds the per-block partials in block order,
//! and the selected extreme is the identical bit pattern the sequential golden
//! fold selects. This two-level split — device block reduction plus a small
//! host final fold — mirrors the host coarse aggregation the stream-compaction
//! twin [`gpu_compact`](crate::gpu_compact) uses.
//!
//! # Correctness model
//!
//! Because `min` and `max` select an input rather than compute a new value,
//! the `CPU` and `GPU` results are bit-identical, and the parity test asserts
//! an exact match on the raw `f32` bit patterns (an integer `u32` comparison of
//! `f32::to_bits`), with no tolerance. There is no transcendental call and no
//! fused multiply-add to perturb a low mantissa bit, so there is no
//! `ULP`-boundary degenerate region to avoid.
//!
//! # Degenerate inputs
//!
//! An empty mesh never dispatches: the host short-circuits and returns
//! `(Vec3::ZERO, Vec3::ZERO)`, exactly as the golden does, since a storage
//! buffer cannot be zero-sized. Every dispatched block owns at least one vertex
//! (the block count is `div_ceil(count, block_size)`), so a block's partial is
//! always seeded by a real position and the `+inf` / `-inf` sentinels never
//! leak into a result. `WGSL` has no `f32` infinity literal, so the sentinels
//! are built by `bitcast` from the `IEEE-754` bit patterns `0x7F800000`
//! (`+inf`) and `0xFF800000` (`-inf`).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the `min` and `max`
//! built-ins, unsigned integer index arithmetic and comparisons, and `bitcast`
//! — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `sqrt` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The only loop is the intra-block reduction, whose bound
//! is the block size, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission::Mesh::aabb`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 本仓孪生约定；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Default vertices reduced per block when a query requests `0`. One block per
/// vertex keeps the device fold shallow while still exercising the host final
/// aggregation across many blocks.
///
/// Provenance: 本仓孪生约定；无第三方引擎源码或衍生代码。
const DEFAULT_BLOCK_SIZE: u32 = 64;

/// The portable core-`WGSL` bounds-reduction kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `reduce` mirrors the
/// `CPU` golden
/// [`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb)
/// component-wise `min` / `max` fold; see the module documentation for the
/// algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission::Mesh::aabb`；无第三方引擎源码或衍生代码。
const MESH_AABB_WGSL: &str = r#"
// mesh_aabb twin: one thread per block reduces a contiguous run of vertex
// positions into a partial (min, max) box, taking the component-wise min and
// max across all three axes. The host folds the per-block partials into the
// final box. Both min and max select one of their inputs (no arithmetic), so
// the reduction is bit-exact regardless of fold order and agrees with the
// sequential CPU golden `particle::mesh_emission::Mesh::aabb` bit for bit. The
// kernel uses only the portable core-WGSL subset (min/max, unsigned index math
// and bitcast) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12. The +inf / -inf seeds are built by bitcast because WGSL has
// no f32 infinity literal.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::mesh_emission::Mesh::aabb;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the vertex count, the block size,
// and two pad words, matching the host `Params`.
struct Params {
    count: u32,
    block_size: u32,
    pad0: u32,
    pad1: u32,
}

// One vertex position on its own 16-byte std430 lane, matching the host
// `GpuPosition`.
struct Position {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

// One per-block partial box. 32-byte std430 stride of two 16-byte lanes (the
// min corner then the max corner), matching the host `GpuPartial`.
struct Partial {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    pad_min: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    pad_max: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<Position>;
@group(0) @binding(2) var<storage, read_write> partials: array<Partial>;

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    let block_size = max(params.block_size, 1u);
    // Number of blocks = div_ceil(count, block_size). Threads past the last
    // block return without writing.
    let num_blocks = (params.count + block_size - 1u) / block_size;
    if (block >= num_blocks) {
        return;
    }

    let start = block * block_size;
    var end = start + block_size;
    if (end > params.count) {
        end = params.count;
    }

    // Seed from +inf / -inf (bitcast, since WGSL has no f32 infinity literal).
    // Every dispatched block owns at least one vertex, so the sentinels are
    // always overwritten by a real position before the partial is written.
    let pos_inf = bitcast<f32>(0x7F800000u);
    let neg_inf = bitcast<f32>(0xFF800000u);
    var lo = vec3<f32>(pos_inf, pos_inf, pos_inf);
    var hi = vec3<f32>(neg_inf, neg_inf, neg_inf);

    // Intra-block reduction. The loop is bounded by the block size, so the
    // kernel provably terminates.
    for (var i: u32 = start; i < end; i = i + 1u) {
        let p = positions[i];
        let v = vec3<f32>(p.x, p.y, p.z);
        lo = min(lo, v);
        hi = max(hi, v);
    }

    var out: Partial;
    out.min_x = lo.x;
    out.min_y = lo.y;
    out.min_z = lo.z;
    out.pad_min = 0.0;
    out.max_x = hi.x;
    out.max_y = hi.y;
    out.max_z = hi.z;
    out.pad_max = 0.0;
    partials[block] = out;
}
"#;

/// Uniform parameters for one reduction dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`MESH_AABB_WGSL`]: the vertex `count`, the `block_size`
/// and two pad words — `16` bytes with no interior padding.
///
/// Provenance: 本模块新建的 GPU 下发参数；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of vertex positions.
    count: u32,
    /// Vertices reduced per block (per simulated `workgroup`).
    block_size: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One vertex position as uploaded. `16`-byte `std430` stride matching
/// `Position` in [`MESH_AABB_WGSL`]: the three components on one lane padded to
/// the lane boundary.
///
/// Provenance: 本模块新建的 GPU 顶点上传布局；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPosition {
    /// Position `x`.
    x: f32,
    /// Position `y`.
    y: f32,
    /// Position `z`.
    z: f32,
    /// Padding lane.
    pad: f32,
}

/// One per-block partial box as read back. `32`-byte `std430` stride matching
/// `Partial` in [`MESH_AABB_WGSL`]: the `min` corner on one `16`-byte lane, the
/// `max` corner on the next.
///
/// Provenance: 本模块新建的 GPU 分块部分结果布局；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPartial {
    /// Partial `min` `x`.
    min_x: f32,
    /// Partial `min` `y`.
    min_y: f32,
    /// Partial `min` `z`.
    min_z: f32,
    /// Padding lane.
    pad_min: f32,
    /// Partial `max` `x`.
    max_x: f32,
    /// Partial `max` `y`.
    max_y: f32,
    /// Partial `max` `z`.
    max_z: f32,
    /// Padding lane.
    pad_max: f32,
}

/// A batch of vertex positions whose axis-aligned bounds the device reduces.
///
/// `positions` lists the vertex positions in any order (the reduction is
/// order-independent); `block_size` is the number of vertices each block
/// reduces before the host folds the partials. A `block_size` of `0` is
/// replaced with `DEFAULT_BLOCK_SIZE` on dispatch, mirroring the way the
/// stream-compaction twin clamps its block size.
///
/// Provenance: 本模块新建的 GPU 批量查询类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshAabbQuery {
    /// The vertex positions to reduce.
    pub positions: Vec<Vec3>,
    /// Vertices reduced per block; `0` selects `DEFAULT_BLOCK_SIZE`.
    pub block_size: u32,
}

/// The axis-aligned bounds a reduction produces: the component-wise `min` and
/// `max` corner over every vertex position, or `(Vec3::ZERO, Vec3::ZERO)` for an
/// empty batch.
///
/// Provenance: 本模块新建的 GPU 批量结果类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeshAabbResult {
    /// The minimum corner (component-wise `min` over all positions).
    pub min: Vec3,
    /// The maximum corner (component-wise `max` over all positions).
    pub max: Vec3,
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Host final fold: reduces the per-block partials into the final box.
///
/// Folds the partial `min` corners with [`Vec3::min`] and the partial `max`
/// corners with [`Vec3::max`], seeding from the first partial. Both operations
/// select an input rather than compute a new value, so the result is the
/// identical bit pattern the sequential golden fold selects. The slice is never
/// empty when called, since a non-empty batch always yields at least one block.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission::Mesh::aabb`；无第三方引擎源码或衍生代码。
fn fold_partials(partials: &[GpuPartial]) -> GpuMeshAabbResult {
    let first = partials[0];
    let mut lo = Vec3::new(first.min_x, first.min_y, first.min_z);
    let mut hi = Vec3::new(first.max_x, first.max_y, first.max_z);
    for p in &partials[1..] {
        lo = lo.min(Vec3::new(p.min_x, p.min_y, p.min_z));
        hi = hi.max(Vec3::new(p.max_x, p.max_y, p.max_z));
    }
    GpuMeshAabbResult { min: lo, max: hi }
}

/// A compiled, reusable mesh-bounds reduction pipeline, twinning the `CPU`
/// golden
/// [`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb).
///
/// Provenance: 本模块新建的 GPU 管线封装类型；无第三方引擎源码或衍生代码。
pub struct GpuMeshAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshAabb {
    /// Compiles the bounds-reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块新建的管线构造；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_aabb_module"),
            source: ShaderSource::Wgsl(MESH_AABB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reduce"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Reduces every vertex position in `query` into its axis-aligned bounds on
    /// the device and reads the result back.
    ///
    /// The returned `(min, max)` corners equal the golden
    /// [`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb)
    /// bit for bit, because `min` and `max` select an input and never compute a
    /// new value. An empty `positions` batch returns
    /// `(Vec3::ZERO, Vec3::ZERO)` with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission::Mesh::aabb`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, query: &GpuMeshAabbQuery) -> GpuMeshAabbResult {
        let count = query.positions.len();
        if count == 0 {
            return GpuMeshAabbResult {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            };
        }

        let device = ctx.device();
        let block_size = if query.block_size == 0 {
            DEFAULT_BLOCK_SIZE
        } else {
            query.block_size
        };
        let num_blocks = (count as u32).div_ceil(block_size);

        let packed: Vec<GpuPosition> = query
            .positions
            .iter()
            .map(|p| GpuPosition {
                x: p.x,
                y: p.y,
                z: p.z,
                pad: 0.0,
            })
            .collect();
        let gpu_params = GpuParams {
            count: count as u32,
            block_size,
            pad0: 0,
            pad1: 0,
        };

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_aabb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_aabb_positions"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let partials_bytes = (num_blocks as u64) * (size_of::<GpuPartial>() as u64);
        let partials_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_aabb_partials"),
            size: partials_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let partials_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_aabb_partials_stage"),
            size: partials_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_aabb_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: partials_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_aabb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_aabb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per block, flattened to a 1-D dispatch.
            let groups = num_blocks.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partials_buf, 0, &partials_stage, 0, partials_bytes);
        ctx.queue().submit([encoder.finish()]);

        partials_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let view = partials_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let partials = bytemuck::cast_slice::<u8, GpuPartial>(&view).to_vec();
        drop(view);
        partials_stage.unmap();
        debug_assert_eq!(partials.len(), num_blocks as usize);

        fold_partials(&partials)
    }
}
