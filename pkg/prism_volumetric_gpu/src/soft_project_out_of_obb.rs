//! `wgpu` compute twin of the oriented-bounding-box ejection from the `CPU`
//! golden `prism_physics_core::soft::collision::body`'s `project_out_of_obb`.
//!
//! A soft body resolves penetration against a rigid oriented box by pushing any
//! interior particle out to the nearest face of the box, measured in the box's
//! own local frame. This module ports that single stateless projection onto the
//! device: one thread resolves one particle, so a passing real-device parity
//! test is direct evidence the ported kernel takes the same degenerate /
//! outside / interior branch the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `project_out_of_obb` for a single
//! particle with explicit world position `pos`, box `center`, unit orientation
//! quaternion `q` (stored `x, y, z, w`) and `half_extents` `he`. An all
//! non-positive `he` has no interior, so the point is echoed unchanged with
//! `valid = 0`. Otherwise the point is carried into the box frame with the
//! inverse (conjugate) rotation, `local = rotate(conj(q), pos - center)`. If the
//! point lies on or outside any slab (`|local.axis| >= he.axis`, which a
//! collapsed non-positive axis always satisfies) it is already outside the
//! solid and echoed with `valid = 0`. For a true interior point the penetration
//! depth per axis is `pen = he - |local|`; the axis of least penetration (ties
//! break `x` then `y` then `z`, matching the reference `<=` chain) is snapped to
//! its signed face, and the result is rotated back and recentred,
//! `out = center + rotate(q, local_out)`, with `valid = 1`.
//!
//! The quaternion rotation is the pure-arithmetic Rodrigues form
//! `rotate(q, v) = v + 2 (q.xyz x (q.xyz x v + q.w v))`, which equals `q * v`
//! for a unit quaternion; body orientations are unit, so the fixtures feed only
//! normalized quaternions. There is no loop: each thread performs a fixed,
//! bounded sequence of arithmetic, cross products and dot products, so the
//! kernel provably terminates.
//!
//! # Correctness model
//!
//! Every continuous quantity threads through multiplies, adds and cross
//! products, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous output. The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! An all non-positive `half_extents` (no interior) and any point on or outside
//! a slab are no-ops: the twin reports `valid = 0` and echoes the input
//! position. Fixtures and the sweep keep interior points away from the
//! least-penetration tie surface (the three `pen` values pairwise separated) so
//! a last-bit difference cannot flip the chosen face, and away from the slab
//! boundary so the inside / outside branch cannot flip. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `cross`, `dot`,
//! `abs`, `select`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every comparison is ordered; there is no `f32` equality, so a
//! fast-math device that folds `x == x` to `true` cannot change a branch.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body::project_out_of_obb`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` oriented-box ejection kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `project_out_of_obb` branch for branch; see the module
/// documentation for the algorithm.
const SOFT_PROJECT_OUT_OF_OBB_WGSL: &str = r#"
// Oriented-bounding-box ejection twin: one thread per particle reproduces
// project_out_of_obb. It mirrors the CPU golden branch for branch, uses only the
// portable core-WGSL subset (cross/dot/abs/select and + - * / plus unsigned
// index math), takes no optional feature, and has no loop, so it provably
// terminates. Every guard is an ordered compare + select, so a fast-math device
// cannot flip a branch.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle world position.
    px: f32, py: f32, pz: f32,
    // Box center, world space.
    cx: f32, cy: f32, cz: f32,
    // Unit orientation quaternion, stored x, y, z, w.
    qx: f32, qy: f32, qz: f32, qw: f32,
    // Half extents along the box local axes.
    hx: f32, hy: f32, hz: f32,
}

struct Result {
    // Projected (or echoed) world position.
    ox: f32, oy: f32, oz: f32,
    // 1 when the point was pushed out of the interior, 0 for a no-op.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Rotates v by the quaternion rot (stored x, y, z, w) using the pure-arithmetic
// Rodrigues form, which equals rot * v for a unit quaternion.
fn quat_rotate(rot: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let axis = vec3<f32>(rot.x, rot.y, rot.z);
    let t = cross(axis, v) + rot.w * v;
    return v + 2.0 * cross(axis, t);
}

// Conjugate of a quaternion (the inverse rotation for a unit quaternion).
fn quat_conj(rot: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-rot.x, -rot.y, -rot.z, rot.w);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let pos = vec3<f32>(q.px, q.py, q.pz);
    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let rot = vec4<f32>(q.qx, q.qy, q.qz, q.qw);
    let he = vec3<f32>(q.hx, q.hy, q.hz);

    // Default: echo the input position, valid = 0.
    var out: Result;
    out.ox = pos.x;
    out.oy = pos.y;
    out.oz = pos.z;
    out.valid = 0u;

    // No positive extent means there is no interior to project out of.
    if (he.x <= 0.0 && he.y <= 0.0 && he.z <= 0.0) {
        results[idx] = out;
        return;
    }

    // World -> local via the inverse (conjugate) rotation.
    let local = quat_rotate(quat_conj(rot), pos - center);
    let al = abs(local);
    // On or outside any slab => already outside the solid (a collapsed axis with
    // a non-positive half extent is always outside, making the box inert).
    if (al.x >= he.x || al.y >= he.y || al.z >= he.z) {
        results[idx] = out;
        return;
    }

    // Interior: push to the face of least penetration (smallest he - |local|).
    let pen = he - al;
    var local_out = local;
    if (pen.x <= pen.y && pen.x <= pen.z) {
        local_out.x = select(-he.x, he.x, local.x >= 0.0);
    } else if (pen.y <= pen.z) {
        local_out.y = select(-he.y, he.y, local.y >= 0.0);
    } else {
        local_out.z = select(-he.z, he.z, local.z >= 0.0);
    }

    let world_out = center + quat_rotate(rot, local_out);
    out.ox = world_out.x;
    out.oy = world_out.y;
    out.oz = world_out.z;
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
/// Every `vec3`/quaternion input is flattened to scalar lanes so the layout
/// never trips a `16`-byte vector-alignment rule; the kernel rebuilds each
/// vector. The struct is `13` `f32` words (`52` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    hx: f32,
    hy: f32,
    hz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the projected (or echoed) position and the validity flag — `4` words
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    ox: f32,
    oy: f32,
    oz: f32,
    valid: u32,
}

/// One oriented-box ejection query: the particle position, the box center, its
/// unit orientation quaternion (stored `x, y, z, w`) and its half extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfObbQuery {
    /// Particle world position.
    pub pos: [f32; 3],
    /// Box center, world space.
    pub center: [f32; 3],
    /// Unit orientation quaternion, stored `[x, y, z, w]`.
    pub orientation: [f32; 4],
    /// Half extents along the box local axes.
    pub half_extents: [f32; 3],
}

impl SoftProjectOutOfObbQuery {
    /// Builds a query from the particle position, box center, orientation
    /// quaternion (`[x, y, z, w]`) and half extents.
    #[must_use]
    pub fn new(
        pos: [f32; 3],
        center: [f32; 3],
        orientation: [f32; 4],
        half_extents: [f32; 3],
    ) -> SoftProjectOutOfObbQuery {
        SoftProjectOutOfObbQuery {
            pos,
            center,
            orientation,
            half_extents,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `project_out_of_obb` output for that particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfObbResult {
    /// The projected position when the point was interior, otherwise the input
    /// position echoed unchanged.
    pub out: [f32; 3],
    /// `1` when the point was pushed out of the interior, `0` for the
    /// degenerate / already-outside no-op branch.
    pub valid: u32,
}

/// Encodes one [`SoftProjectOutOfObbQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SoftProjectOutOfObbQuery) -> GpuQuery {
    GpuQuery {
        px: q.pos[0],
        py: q.pos[1],
        pz: q.pos[2],
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        qx: q.orientation[0],
        qy: q.orientation[1],
        qz: q.orientation[2],
        qw: q.orientation[3],
        hx: q.half_extents[0],
        hy: q.half_extents[1],
        hz: q.half_extents[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftProjectOutOfObbResult`].
fn decode_result(raw: &GpuResult) -> SoftProjectOutOfObbResult {
    SoftProjectOutOfObbResult {
        out: [raw.ox, raw.oy, raw.oz],
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

/// A compiled, reusable oriented-box ejection compute pipeline, twinning the
/// `CPU` golden `project_out_of_obb`.
pub struct GpuSoftProjectOutOfObb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftProjectOutOfObb {
    /// Compiles the oriented-box ejection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftProjectOutOfObb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb"),
            source: ShaderSource::Wgsl(SOFT_PROJECT_OUT_OF_OBB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftProjectOutOfObb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftProjectOutOfObbResult`] per input, in order.
    ///
    /// Each continuous output matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftProjectOutOfObbQuery],
    ) -> Vec<SoftProjectOutOfObbResult> {
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
            label: Some("prism_volumetric_soft_project_out_of_obb_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_bind_group"),
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
            label: Some("prism_volumetric_soft_project_out_of_obb_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_obb_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_project_out_of_obb_pass"),
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
