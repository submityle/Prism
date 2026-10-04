//! `wgpu` compute twin of the rigid-body world-space inverse inertia helper
//! from the `CPU` golden
//! `prism_physics_core::solver::xpbd::rigid::inv_inertia_world`.
//!
//! A rigid body stores its inverse inertia as the diagonal `inv_inertia` in the
//! body's principal frame; rotating it by the orientation quaternion `q` gives
//! the world-space inverse inertia tensor `R * diag(inv_inertia) * R^T`, where
//! `R = from_quat(q)`. This tensor is the workhorse of the `XPBD` rigid-body
//! impulse algebra (generalized inverse mass, angular impulse response). This
//! module ports that stateless, no-`RNG` closed form onto the device: one
//! compute thread resolves one body, so a passing real-device parity test is
//! direct evidence the kernel builds the same rotation matrix and performs the
//! same column-major triple product, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one body: a diagonal inverse inertia `(ix, iy, iz)` and an
//! orientation quaternion `(x, y, z, w)`. For each body the kernel reproduces
//! the reference closed form exactly, column for column:
//!
//! * `R = from_quat(q)` with the standard `glam` column-major basis;
//! * `scaled = (col0 * ix, col1 * iy, col2 * iz)` (each column scaled by its
//!   inverse-inertia component);
//! * `result = scaled * R^T`, i.e. `R * diag(inv_inertia) * R^T`, a symmetric
//!   `3x3` returned column-major.
//!
//! There is no loop and no degenerate branch: the computation is pure
//! arithmetic, so the kernel provably terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! The triple product threads through several multiply-adds, so `CPU` and `GPU`
//! are not required to be bit-exact. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the
//! nine matrix entries; the discrete `valid` flag is compared exactly. The
//! kernel has no comparisons at all, so no bare float equality and no fast-math
//! NaN sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every quaternion and inverse inertia yields
//! a well-defined tensor, so `valid` is always `1`. A zero inverse inertia
//! simply produces a zero tensor. An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *` and unsigned
//! index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no
//! `round`, no float modulo and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid::inv_inertia_world`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` world-space inverse-inertia kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `project`
/// mirrors the `CPU` golden `inv_inertia_world`; see the module documentation
/// for the algorithm.
const RIGID_INV_INERTIA_WORLD_WGSL: &str = r#"
// World-space inverse inertia twin: one thread per body reproduces the tensor
// R * diag(inv_inertia) * R^T the golden inv_inertia_world builds from a
// diagonal inverse inertia and an orientation quaternion. It mirrors the CPU
// golden operation for operation, uses only the portable core-WGSL subset
// (+ - * plus unsigned index math), takes no optional feature, and has no
// loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓
// prism_physics_core::solver::xpbd::rigid::inv_inertia_world；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of bodies in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Diagonal inverse inertia in the body's principal frame.
    ix: f32,
    iy: f32,
    iz: f32,
    // Orientation quaternion (x, y, z, w).
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
}

struct Result {
    // World-space inverse inertia tensor, column-major:
    // m0..m2 = col0, m3..m5 = col1, m6..m8 = col2.
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
    // Always 1: the computation has no degenerate branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Build the rotation matrix R = from_quat(q), glam column-major basis.
    let x2 = q.qx + q.qx;
    let y2 = q.qy + q.qy;
    let z2 = q.qz + q.qz;
    let xx = q.qx * x2;
    let xy = q.qx * y2;
    let xz = q.qx * z2;
    let yy = q.qy * y2;
    let yz = q.qy * z2;
    let zz = q.qz * z2;
    let wx = q.qw * x2;
    let wy = q.qw * y2;
    let wz = q.qw * z2;

    let col0 = vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy);
    let col1 = vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx);
    let col2 = vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy));

    // Scale each column by its inverse-inertia component.
    let s0 = col0 * q.ix;
    let s1 = col1 * q.iy;
    let s2 = col2 * q.iz;

    // result = scaled * R^T. R^T column j = R row j = (col0[j], col1[j], col2[j]).
    // result column j = s0 * rowj[0] + s1 * rowj[1] + s2 * rowj[2].
    let row0 = vec3<f32>(col0.x, col1.x, col2.x);
    let row1 = vec3<f32>(col0.y, col1.y, col2.y);
    let row2 = vec3<f32>(col0.z, col1.z, col2.z);

    let rc0 = s0 * row0.x + s1 * row0.y + s2 * row0.z;
    let rc1 = s0 * row1.x + s1 * row1.y + s2 * row1.z;
    let rc2 = s0 * row2.x + s1 * row2.y + s2 * row2.z;

    var out: Result;
    out.m0 = rc0.x;
    out.m1 = rc0.y;
    out.m2 = rc0.z;
    out.m3 = rc1.x;
    out.m4 = rc1.y;
    out.m5 = rc1.z;
    out.m6 = rc2.x;
    out.m7 = rc2.y;
    out.m8 = rc2.z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RIGID_INV_INERTIA_WORLD_WGSL`].
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
/// All components are scalar `f32` so the slot contains no `vec3` and the host
/// and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ix: f32,
    iy: f32,
    iz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The nine matrix entries are stored column-major as flat scalars; the
/// trailing `valid` word keeps the discrete flag beside them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
    valid: u32,
}

/// One query for the world-space inverse-inertia twin: a diagonal inverse
/// inertia in the body's principal frame plus the body's orientation
/// quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidInvInertiaWorldQuery {
    /// Diagonal inverse inertia `(ix, iy, iz)` in the principal frame.
    pub inv_inertia: [f32; 3],
    /// Orientation quaternion `(x, y, z, w)`.
    pub q: [f32; 4],
}

impl RigidInvInertiaWorldQuery {
    /// Builds a query from the diagonal inverse inertia and the orientation
    /// quaternion `(x, y, z, w)`.
    #[must_use]
    pub fn new(inv_inertia: [f32; 3], q: [f32; 4]) -> RigidInvInertiaWorldQuery {
        RigidInvInertiaWorldQuery { inv_inertia, q }
    }
}

/// One resolved answer for a single body: the world-space inverse inertia
/// tensor `R * diag(inv_inertia) * R^T`, stored column-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidInvInertiaWorldResult {
    /// The symmetric `3x3` world-space inverse inertia tensor, column-major:
    /// `m[0..3]` is column 0, `m[3..6]` column 1, `m[6..9]` column 2.
    pub m: [f32; 9],
    /// Always `1`: the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`RigidInvInertiaWorldQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &RigidInvInertiaWorldQuery) -> GpuQuery {
    GpuQuery {
        ix: q.inv_inertia[0],
        iy: q.inv_inertia[1],
        iz: q.inv_inertia[2],
        qx: q.q[0],
        qy: q.q[1],
        qz: q.q[2],
        qw: q.q[3],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RigidInvInertiaWorldResult`].
fn decode_result(raw: &GpuResult) -> RigidInvInertiaWorldResult {
    RigidInvInertiaWorldResult {
        m: [
            raw.m0, raw.m1, raw.m2, raw.m3, raw.m4, raw.m5, raw.m6, raw.m7, raw.m8,
        ],
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

/// A compiled, reusable world-space inverse-inertia compute pipeline, twinning
/// the `CPU` golden `inv_inertia_world`.
pub struct GpuRigidInvInertiaWorld {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRigidInvInertiaWorld {
    /// Compiles the world-space inverse-inertia kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidInvInertiaWorld {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world"),
            source: ShaderSource::Wgsl(RIGID_INV_INERTIA_WORLD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidInvInertiaWorld {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every body in `queries` and returns one
    /// [`RigidInvInertiaWorldResult`] per input, in order.
    ///
    /// Each matrix entry matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RigidInvInertiaWorldQuery],
    ) -> Vec<RigidInvInertiaWorldResult> {
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
            label: Some("prism_volumetric_rigid_inv_inertia_world_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_bind_group"),
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
            label: Some("prism_volumetric_rigid_inv_inertia_world_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rigid_inv_inertia_world_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rigid_inv_inertia_world_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
