//! `wgpu` compute twin of the world-space `AABB` of an oriented bounding box,
//! from the `CPU` golden `prism_physics_core::collider::obb`'s `Obb::aabb`.
//!
//! An oriented bounding box is a center, three orthonormal axes and three
//! half-extents. Its tightest world-space axis-aligned bounding box has a
//! half-size `r` whose each component is the sum of the absolute axis
//! projections scaled by the matching half-extent, so the box is
//! `(center - r, center + r)`. This module ports that single stateless closed
//! form onto the device: one thread resolves one box, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same support
//! radius the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `Obb::aabb` for one box with explicit
//! `center`, the three axes `a0, a1, a2` and the half-extents `he`:
//!
//! * `r = |a0|·he.x + |a1|·he.y + |a2|·he.z` (component-wise absolute value,
//!   axis `a0` paired with `he.x`, `a1` with `he.y`, `a2` with `he.z`).
//! * `min = center - r`; `max = center + r`.
//!
//! There is no loop and no branch: each thread performs a fixed, bounded
//! sequence of absolute values, multiplies and adds, so the kernel provably
//! terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! Every quantity threads through absolute values, multiplies and adds, so
//! `CPU` and `GPU` are not necessarily bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous output. The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! There is no degenerate branch: an axis-aligned box (identity axes) collapses
//! the radius to the half-extents, and a zero half-extent simply contributes
//! nothing — both are the ordinary closed form, so `valid` is always `1`. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - *` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no `f32` remainder, no division and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` oriented-box world-`AABB` kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `Obb::aabb`; see the module documentation for the
/// closed form.
const OBB_WORLD_AABB_WGSL: &str = r#"
// Oriented-box world-AABB twin: one thread per box reproduces Obb::aabb. It
// uses only the portable core-WGSL subset (abs and + - * plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Every vec3 input is passed as scalar lanes and rebuilt
// inside the shader to avoid any std430 16-byte vector-alignment ambiguity.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Box center, world space.
    cx: f32, cy: f32, cz: f32,
    // Orthonormal axis 0, paired with half-extent hx.
    a0x: f32, a0y: f32, a0z: f32,
    // Orthonormal axis 1, paired with half-extent hy.
    a1x: f32, a1y: f32, a1z: f32,
    // Orthonormal axis 2, paired with half-extent hz.
    a2x: f32, a2y: f32, a2z: f32,
    // Half-extents along each local axis.
    hx: f32, hy: f32, hz: f32,
}

struct Result {
    // World-space AABB minimum corner.
    minx: f32, miny: f32, minz: f32,
    // World-space AABB maximum corner.
    maxx: f32, maxy: f32, maxz: f32,
    // Always 1; the closed form has no degenerate branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let a0 = vec3<f32>(q.a0x, q.a0y, q.a0z);
    let a1 = vec3<f32>(q.a1x, q.a1y, q.a1z);
    let a2 = vec3<f32>(q.a2x, q.a2y, q.a2z);

    // r = |a0|*hx + |a1|*hy + |a2|*hz, component-wise absolute value, in the
    // same accumulation order as the golden (a0 then a1 then a2).
    let r = abs(a0) * q.hx + abs(a1) * q.hy + abs(a2) * q.hz;
    let lo = center - r;
    let hi = center + r;

    var out: Result;
    out.minx = lo.x; out.miny = lo.y; out.minz = lo.z;
    out.maxx = hi.x; out.maxy = hi.y; out.maxz = hi.z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every `vec3` input is flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`. The
/// struct is `15` `f32` words (`60` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cx: f32,
    cy: f32,
    cz: f32,
    a0x: f32,
    a0y: f32,
    a0z: f32,
    a1x: f32,
    a1y: f32,
    a1z: f32,
    a2x: f32,
    a2y: f32,
    a2z: f32,
    hx: f32,
    hy: f32,
    hz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two `AABB` corners and the validity flag — `7` words
/// (`28` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    minx: f32,
    miny: f32,
    minz: f32,
    maxx: f32,
    maxy: f32,
    maxz: f32,
    valid: u32,
}

/// One oriented-box world-`AABB` query: the box center, three orthonormal axes
/// and the three half-extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbWorldAabbQuery {
    /// Box center, world space.
    pub center: [f32; 3],
    /// The three orthonormal box axes `a0, a1, a2`, each paired with the
    /// matching half-extent lane.
    pub axes: [[f32; 3]; 3],
    /// Half-extents along each local axis.
    pub half_extents: [f32; 3],
}

impl ObbWorldAabbQuery {
    /// Builds a query from the box center, three axes and the half-extents.
    #[must_use]
    pub fn new(center: [f32; 3], axes: [[f32; 3]; 3], half_extents: [f32; 3]) -> ObbWorldAabbQuery {
        ObbWorldAabbQuery {
            center,
            axes,
            half_extents,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `Obb::aabb` output for that box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbWorldAabbResult {
    /// The world-space `AABB` minimum corner.
    pub min: [f32; 3],
    /// The world-space `AABB` maximum corner.
    pub max: [f32; 3],
    /// Always `1`; the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`ObbWorldAabbQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ObbWorldAabbQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        a0x: q.axes[0][0],
        a0y: q.axes[0][1],
        a0z: q.axes[0][2],
        a1x: q.axes[1][0],
        a1y: q.axes[1][1],
        a1z: q.axes[1][2],
        a2x: q.axes[2][0],
        a2y: q.axes[2][1],
        a2z: q.axes[2][2],
        hx: q.half_extents[0],
        hy: q.half_extents[1],
        hz: q.half_extents[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ObbWorldAabbResult`].
fn decode_result(raw: &GpuResult) -> ObbWorldAabbResult {
    ObbWorldAabbResult {
        min: [raw.minx, raw.miny, raw.minz],
        max: [raw.maxx, raw.maxy, raw.maxz],
        valid: raw.valid,
    }
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

/// A compiled, reusable oriented-box world-`AABB` compute pipeline, twinning the
/// `CPU` golden `Obb::aabb`.
pub struct GpuObbWorldAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuObbWorldAabb {
    /// Compiles the oriented-box world-`AABB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbWorldAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_obb_world_aabb"),
            source: ShaderSource::Wgsl(OBB_WORLD_AABB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbWorldAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ObbWorldAabbResult`]
    /// per input, in order.
    ///
    /// Each continuous output matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ObbWorldAabbQuery],
    ) -> Vec<ObbWorldAabbResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_obb_world_aabb_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_obb_world_aabb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
