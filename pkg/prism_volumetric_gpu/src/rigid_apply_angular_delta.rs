//! `wgpu` compute twin of the small-rotation angular-integration kernel from the
//! `CPU` golden
//! `prism_physics_core::solver::xpbd::rigid::apply_angular_delta`.
//!
//! A rigid body's orientation is advanced by a small rotation vector `dphi`
//! (axis scaled by angle) using the first-order quaternion update
//! `q' = normalize(q + 0.5 * [dphi, 0] q)`, the same integration the free-body
//! angular integrator uses. One thread advances one independent orientation and
//! writes the renormalized quaternion plus a validity flag.
//!
//! [`GpuRigidApplyAngularDelta`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same
//! Hamilton product, the same half-step accumulation, the same squared-length
//! guard and the same `1 / sqrt(len_sq)` renormalization the reference computes,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate and
//! of `glam`: the Hamilton product `dq = [dphi, 0] * orientation`; the
//! half-step accumulation `updated = orientation + 0.5 * dq`; the squared length
//! `len_sq = dot4(updated, updated)`; the degenerate collapse rejection
//! (`len_sq <= 1e-20`, echo `orientation`, `valid = 0`); and the renormalized
//! result `updated / sqrt(len_sq)` (`valid = 1`). There is no loop: each thread
//! performs a fixed, bounded sequence of multiplies, adds, a divide and a
//! square root, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The renormalized quaternion threads through a product, a divide and a square
//! root, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add
//! the scalar reference leaves separate. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous channel. The discrete `valid` flag is compared exactly: it is `1`
//! only when the accumulated quaternion has a non-degenerate length, and `0`
//! for the echo path, in which case the orientation passes through unchanged.
//!
//! # Degenerate inputs
//!
//! An accumulated quaternion whose squared length has collapsed to (near) zero
//! has no defined direction, so when `len_sq <= 1e-20` the input orientation is
//! echoed (`valid = 0`) rather than dividing by zero. The divisor `len_sq` is
//! guarded with a unit fallback so the unselected arm cannot raise a `NaN` or an
//! infinity before the valid gate drops it. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `select`,
//! `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no float
//! `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The degeneracy test uses an ordered comparison feeding
//! `select`, which is robust under `Metal`'s fast-math (an `x == x` test would
//! be folded to `true`).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方
//! 引擎源码或衍生代码。
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

/// The portable core-`WGSL` angular-integration kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `apply_angular_delta`; see the module docs for the closed
/// form it reproduces.
const RIGID_APPLY_ANGULAR_DELTA_WGSL: &str = r#"
// Angular-integration twin: one thread advances one independent orientation by
// the first-order quaternion update `q' = normalize(q + 0.5 * [dphi, 0] q)` and
// writes the renormalized quaternion plus a validity flag. It mirrors the CPU
// golden exactly and uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Orientation quaternion (bx, by, bz, bw); need not be unit length.
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    // Small rotation vector dphi (axis scaled by angle).
    ax: f32,
    ay: f32,
    az: f32,
}

struct Soln {
    // Renormalized orientation quaternion.
    outx: f32,
    outy: f32,
    outz: f32,
    outw: f32,
    // 1 when the orientation was integrated, 0 when echoed.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Soln>;

// Squared length below which the accumulated quaternion has no defined
// direction, matching the golden threshold in apply_angular_delta.
const LEN_SQ_EPS: f32 = 1e-20;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let bx = q.bx;
    let by = q.by;
    let bz = q.bz;
    let bw = q.bw;
    let ax = q.ax;
    let ay = q.ay;
    let az = q.az;

    // Hamilton product dq = [dphi, 0] * orientation (self = omega, rhs = q).
    let dqx = ax * bw + ay * bz - az * by;
    let dqy = -ax * bz + ay * bw + az * bx;
    let dqz = ax * by - ay * bx + az * bw;
    let dqw = -(ax * bx + ay * by + az * bz);

    // Half-step accumulation.
    let ux = bx + 0.5 * dqx;
    let uy = by + 0.5 * dqy;
    let uz = bz + 0.5 * dqz;
    let uw = bw + 0.5 * dqw;

    let len_sq = ux * ux + uy * uy + uz * uz + uw * uw;
    // A collapsed quaternion has no defined direction: echo the input.
    let len_ok = len_sq > LEN_SQ_EPS;

    // Guard the divisor so the unselected arm cannot raise a NaN or infinity
    // before the valid gate drops it.
    let safe_len_sq = select(1.0, len_sq, len_ok);
    let inv_len = 1.0 / sqrt(safe_len_sq);

    let nx = ux * inv_len;
    let ny = uy * inv_len;
    let nz = uz * inv_len;
    let nw = uw * inv_len;

    var out: Soln;
    out.outx = select(bx, nx, len_ok);
    out.outy = select(by, ny, len_ok);
    out.outz = select(bz, nz, len_ok);
    out.outw = select(bw, nw, len_ok);
    out.valid = select(0u, 1u, len_ok);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
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
/// Seven `f32` give a fixed `28`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `28` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    ax: f32,
    ay: f32,
    az: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Soln` struct.
/// Four `f32` plus one `u32` give a fixed `20`-byte stride with no pad, since
/// the struct alignment is `4` and `20` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    outx: f32,
    outy: f32,
    outz: f32,
    outw: f32,
    valid: u32,
}

/// One query for the angular-integration twin: the orientation quaternion (any
/// scale) and the small rotation vector `dphi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyAngularDeltaQuery {
    /// `x` of the orientation quaternion.
    pub bx: f32,
    /// `y` of the orientation quaternion.
    pub by: f32,
    /// `z` of the orientation quaternion.
    pub bz: f32,
    /// `w` of the orientation quaternion.
    pub bw: f32,
    /// `x` of the small rotation vector `dphi`.
    pub ax: f32,
    /// `y` of the small rotation vector `dphi`.
    pub ay: f32,
    /// `z` of the small rotation vector `dphi`.
    pub az: f32,
}

impl RigidApplyAngularDeltaQuery {
    /// Builds a query from the orientation quaternion (any scale) and the small
    /// rotation vector `dphi`.
    ///
    /// The quaternion and rotation vector are grouped into fixed-length arrays
    /// so the constructor stays within a small, readable argument count.
    #[must_use]
    pub fn new(orientation: [f32; 4], dphi: [f32; 3]) -> RigidApplyAngularDeltaQuery {
        RigidApplyAngularDeltaQuery {
            bx: orientation[0],
            by: orientation[1],
            bz: orientation[2],
            bw: orientation[3],
            ax: dphi[0],
            ay: dphi[1],
            az: dphi[2],
        }
    }
}

/// One resolved answer for a single query: the renormalized orientation
/// quaternion and the validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyAngularDeltaResult {
    /// Resolved `x` of the orientation quaternion.
    pub outx: f32,
    /// Resolved `y` of the orientation quaternion.
    pub outy: f32,
    /// Resolved `z` of the orientation quaternion.
    pub outz: f32,
    /// Resolved `w` of the orientation quaternion.
    pub outw: f32,
    /// `1` when the orientation was integrated, `0` when echoed.
    pub valid: u32,
}

/// Encodes one [`RigidApplyAngularDeltaQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &RigidApplyAngularDeltaQuery) -> GpuQuery {
    GpuQuery {
        bx: q.bx,
        by: q.by,
        bz: q.bz,
        bw: q.bw,
        ax: q.ax,
        ay: q.ay,
        az: q.az,
    }
}

/// Decodes one `std430` [`GpuResult`] slot into a public
/// [`RigidApplyAngularDeltaResult`].
fn decode_result(r: &GpuResult) -> RigidApplyAngularDeltaResult {
    RigidApplyAngularDeltaResult {
        outx: r.outx,
        outy: r.outy,
        outz: r.outz,
        outw: r.outw,
        valid: r.valid,
    }
}

/// Builds a read-only or read-write storage-buffer bind-group-layout entry at
/// `binding`.
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

/// The on-device twin of the `CPU` golden `apply_angular_delta`: a compiled
/// compute pipeline that advances a batch of orientations by their small
/// rotation vectors.
pub struct GpuRigidApplyAngularDelta {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRigidApplyAngularDelta {
    /// Compiles the inline kernel and builds the compute pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidApplyAngularDelta {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_module"),
            source: ShaderSource::Wgsl(RIGID_APPLY_ANGULAR_DELTA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidApplyAngularDelta {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RigidApplyAngularDeltaResult`] per input, in order.
    ///
    /// The continuous channels match the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RigidApplyAngularDeltaQuery],
    ) -> Vec<RigidApplyAngularDeltaResult> {
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
            label: Some("prism_volumetric_rigid_apply_angular_delta_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_bind_group"),
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
            label: Some("prism_volumetric_rigid_apply_angular_delta_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rigid_apply_angular_delta_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rigid_apply_angular_delta_pass"),
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
