//! `wgpu` compute twin of the `XPBD` positional-impulse apply from the `CPU`
//! golden `prism_physics_core::solver::xpbd::rigid::apply_position_impulse`.
//!
//! A positional impulse `p` (an unnormalized correction vector) applied at a
//! world lever arm `r` moves a rigid body two ways at once: the linear part
//! translates the center of mass by `inv_mass * p`, and the angular part rotates
//! the orientation by the half-angle quaternion built from the world inverse
//! inertia times `r × p`, then renormalizes. This module models one body as one
//! query — the position, orientation, inverse mass, world inverse-inertia
//! matrix, lever arm `r` and impulse `p` — and ports that update onto the
//! device: one thread resolves one body, so a passing real-device parity test is
//! direct evidence the kernel reproduces the same linear translation, the same
//! Hamilton-product angular delta and the same renormalization the reference
//! does.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the body of `apply_position_impulse`
//! together with its `apply_angular_delta` helper:
//! `new_position = position + inv_mass * p` (always updated, no guard);
//! `dphi = I (r × p)` with `I` a column-major world inverse-inertia matrix;
//! the first-order quaternion update `omega = (dphi, 0)`,
//! `dq = omega ⊗ orientation` (Hamilton product),
//! `updated = orientation + 0.5 * dq`, and
//! `new_orientation = len_sq > 1e-20 ? updated / sqrt(len_sq) : orientation`.
//!
//! # Correctness model
//!
//! The outputs are the updated position (three `f32`) and orientation (four
//! `f32`), compared with an absolute-or-relative tolerance because they are
//! continuous, plus a discrete `valid` word compared exactly. The position is
//! *always* updated, so `valid` reflects only whether the orientation could be
//! renormalized: `1` when `len_sq > 1e-20`, otherwise `0` and the orientation is
//! echoed unchanged.
//!
//! The degenerate guard is an ordered compare (`len_sq > 1e-20`) and the result
//! is chosen with `select`, so there is no bare `f32` equality anywhere in the
//! kernel. The divisor is guarded with `select(1.0, len_sq, ok)` so the
//! not-taken arm never forms an `inf`/`NaN` even on a `Metal` fast-math driver
//! that folds `x == x` to `true`.
//!
//! # Degenerate inputs
//!
//! A collapsed orientation update (`len_sq <= 1e-20`) echoes the input
//! orientation with `valid = 0` while still committing the linear translation.
//! An empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `sqrt`, and `+ - *` on scalar `f32` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder, no bare `f32` equality
//! and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every vector, matrix and quaternion is flattened to scalar `f32`
//! lanes in the storage buffers and recomposed by hand (columns `col0`, `col1`,
//! `col2`), so no vector or matrix alignment rule can perturb the `std430`
//! stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` positional-impulse apply kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the body of the `CPU` golden `apply_position_impulse`; see the module
/// documentation for the algorithm.
const RIGID_APPLY_POSITION_IMPULSE_WGSL: &str = r#"
// Positional-impulse apply twin: one thread per query reproduces the body of
// apply_position_impulse. The linear part always translates the center of mass
// by inv_mass * p; the angular part rotates the orientation by the half-angle
// quaternion built from I * (r x p) via a Hamilton product, then renormalizes
// when the squared length clears 1e-20 (otherwise the orientation is echoed).
// It uses only the portable core-WGSL subset (ordered compares, select, sqrt,
// + - * on scalar f32), takes no optional feature and has no loop.

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Center-of-mass position.
    px: f32,
    py: f32,
    pz: f32,
    // Orientation quaternion (x, y, z, w).
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    // Inverse mass.
    inv_mass: f32,
    // World inverse-inertia matrix, column-major: col0, col1, col2.
    i0: f32,
    i1: f32,
    i2: f32,
    i3: f32,
    i4: f32,
    i5: f32,
    i6: f32,
    i7: f32,
    i8: f32,
    // World lever arm r.
    rx: f32,
    ry: f32,
    rz: f32,
    // Positional impulse p.
    ipx: f32,
    ipy: f32,
    ipz: f32,
}

struct Result {
    // Updated position.
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    // Updated orientation (x, y, z, w).
    new_qx: f32,
    new_qy: f32,
    new_qz: f32,
    new_qw: f32,
    // 1 when the orientation was renormalized, 0 when it was echoed.
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

    let position = vec3<f32>(q.px, q.py, q.pz);
    let lever = vec3<f32>(q.rx, q.ry, q.rz);
    let impulse = vec3<f32>(q.ipx, q.ipy, q.ipz);

    // Linear part: always committed, no guard.
    let new_position = position + q.inv_mass * impulse;

    // Angular part: dphi = I * (r x p), I column-major.
    let rp = cross(lever, impulse);
    let col0 = vec3<f32>(q.i0, q.i1, q.i2);
    let col1 = vec3<f32>(q.i3, q.i4, q.i5);
    let col2 = vec3<f32>(q.i6, q.i7, q.i8);
    let dphi = col0 * rp.x + col1 * rp.y + col2 * rp.z;

    let ax = dphi.x;
    let ay = dphi.y;
    let az = dphi.z;
    let bx = q.qx;
    let by = q.qy;
    let bz = q.qz;
    let bw = q.qw;

    // dq = omega (x) orientation, omega = (ax, ay, az, 0), Hamilton product.
    let dqx = ax * bw + ay * bz - az * by;
    let dqy = -ax * bz + ay * bw + az * bx;
    let dqz = ax * by - ay * bx + az * bw;
    let dqw = -(ax * bx + ay * by + az * bz);

    let ux = bx + 0.5 * dqx;
    let uy = by + 0.5 * dqy;
    let uz = bz + 0.5 * dqz;
    let uw = bw + 0.5 * dqw;

    let len_sq = ux * ux + uy * uy + uz * uz + uw * uw;
    // Ordered guard; no bare x == x so a fast-math driver cannot fold it.
    let ok = len_sq > 1e-20;
    // Guarded divisor: the not-taken arm divides by 1.0, never by ~0.
    let safe_len_sq = select(1.0, len_sq, ok);
    let inv_len = 1.0 / sqrt(safe_len_sq);

    let normalized = vec4<f32>(ux, uy, uz, uw) * inv_len;
    let echoed = vec4<f32>(bx, by, bz, bw);
    // select(false_value, true_value, condition).
    let new_orientation = select(echoed, normalized, ok);

    var res: Result;
    res.new_px = new_position.x;
    res.new_py = new_position.y;
    res.new_pz = new_position.z;
    res.new_qx = new_orientation.x;
    res.new_qy = new_orientation.y;
    res.new_qz = new_orientation.z;
    res.new_qw = new_orientation.w;
    res.valid = select(0u, 1u, ok);
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RIGID_APPLY_POSITION_IMPULSE_WGSL`].
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
/// All twenty-three lanes are scalar `f32`, so the layout is a flat `92`-byte
/// stride with alignment `4` and no internal padding, and a batch of two or more
/// packs contiguously with no vector or matrix alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    inv_mass: f32,
    i0: f32,
    i1: f32,
    i2: f32,
    i3: f32,
    i4: f32,
    i5: f32,
    i6: f32,
    i7: f32,
    i8: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    ipx: f32,
    ipy: f32,
    ipz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three position words, four orientation words and one `valid` word
/// give a flat `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    new_qx: f32,
    new_qy: f32,
    new_qz: f32,
    new_qw: f32,
    valid: u32,
}

/// One query for the positional-impulse apply twin: a body's position
/// (`px`, `py`, `pz`), orientation quaternion (`qx`, `qy`, `qz`, `qw`), inverse
/// mass `inv_mass`, world inverse-inertia matrix (`i0`..`i8`, column-major),
/// world lever arm (`rx`, `ry`, `rz`) and positional impulse
/// (`ipx`, `ipy`, `ipz`).
///
/// The whole twinned per-body update is driven by this one tuple, so a single
/// query exercises the linear translation, the Hamilton-product angular delta
/// and the renormalization guard at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyPositionImpulseQuery {
    /// Position, x component.
    pub px: f32,
    /// Position, y component.
    pub py: f32,
    /// Position, z component.
    pub pz: f32,
    /// Orientation quaternion, x component.
    pub qx: f32,
    /// Orientation quaternion, y component.
    pub qy: f32,
    /// Orientation quaternion, z component.
    pub qz: f32,
    /// Orientation quaternion, w component.
    pub qw: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// World inverse-inertia matrix, column `0` row `0`.
    pub i0: f32,
    /// World inverse-inertia matrix, column `0` row `1`.
    pub i1: f32,
    /// World inverse-inertia matrix, column `0` row `2`.
    pub i2: f32,
    /// World inverse-inertia matrix, column `1` row `0`.
    pub i3: f32,
    /// World inverse-inertia matrix, column `1` row `1`.
    pub i4: f32,
    /// World inverse-inertia matrix, column `1` row `2`.
    pub i5: f32,
    /// World inverse-inertia matrix, column `2` row `0`.
    pub i6: f32,
    /// World inverse-inertia matrix, column `2` row `1`.
    pub i7: f32,
    /// World inverse-inertia matrix, column `2` row `2`.
    pub i8: f32,
    /// World lever arm, x component.
    pub rx: f32,
    /// World lever arm, y component.
    pub ry: f32,
    /// World lever arm, z component.
    pub rz: f32,
    /// Positional impulse, x component.
    pub ipx: f32,
    /// Positional impulse, y component.
    pub ipy: f32,
    /// Positional impulse, z component.
    pub ipz: f32,
}

impl RigidApplyPositionImpulseQuery {
    /// Builds a query from the position, orientation, inverse mass, column-major
    /// world inverse-inertia matrix, lever arm and impulse, in field order.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the flat scalar std430 layout one lane at a time"
    )]
    pub fn new(
        px: f32,
        py: f32,
        pz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
        inv_mass: f32,
        i0: f32,
        i1: f32,
        i2: f32,
        i3: f32,
        i4: f32,
        i5: f32,
        i6: f32,
        i7: f32,
        i8: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        ipx: f32,
        ipy: f32,
        ipz: f32,
    ) -> RigidApplyPositionImpulseQuery {
        RigidApplyPositionImpulseQuery {
            px,
            py,
            pz,
            qx,
            qy,
            qz,
            qw,
            inv_mass,
            i0,
            i1,
            i2,
            i3,
            i4,
            i5,
            i6,
            i7,
            i8,
            rx,
            ry,
            rz,
            ipx,
            ipy,
            ipz,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `apply_position_impulse` per-body update.
///
/// (`new_px`, `new_py`, `new_pz`) is the translated position (always updated).
/// (`new_qx`, `new_qy`, `new_qz`, `new_qw`) is the orientation after the angular
/// delta and renormalization. `valid` is `1` when the orientation was
/// renormalized and `0` when the update collapsed and the input orientation was
/// echoed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApplyPositionImpulseResult {
    /// Updated position, x component.
    pub new_px: f32,
    /// Updated position, y component.
    pub new_py: f32,
    /// Updated position, z component.
    pub new_pz: f32,
    /// Updated orientation, x component.
    pub new_qx: f32,
    /// Updated orientation, y component.
    pub new_qy: f32,
    /// Updated orientation, z component.
    pub new_qz: f32,
    /// Updated orientation, w component.
    pub new_qw: f32,
    /// `1` when the orientation was renormalized, `0` when it was echoed.
    pub valid: u32,
}

/// Encodes one [`RigidApplyPositionImpulseQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &RigidApplyPositionImpulseQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        qx: q.qx,
        qy: q.qy,
        qz: q.qz,
        qw: q.qw,
        inv_mass: q.inv_mass,
        i0: q.i0,
        i1: q.i1,
        i2: q.i2,
        i3: q.i3,
        i4: q.i4,
        i5: q.i5,
        i6: q.i6,
        i7: q.i7,
        i8: q.i8,
        rx: q.rx,
        ry: q.ry,
        rz: q.rz,
        ipx: q.ipx,
        ipy: q.ipy,
        ipz: q.ipz,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RigidApplyPositionImpulseResult`].
fn decode_result(raw: &GpuResult) -> RigidApplyPositionImpulseResult {
    RigidApplyPositionImpulseResult {
        new_px: raw.new_px,
        new_py: raw.new_py,
        new_pz: raw.new_pz,
        new_qx: raw.new_qx,
        new_qy: raw.new_qy,
        new_qz: raw.new_qz,
        new_qw: raw.new_qw,
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

/// A compiled, reusable positional-impulse apply compute pipeline, twinning the
/// `CPU` golden `apply_position_impulse`.
pub struct GpuRigidApplyPositionImpulse {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRigidApplyPositionImpulse {
    /// Compiles the positional-impulse apply kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidApplyPositionImpulse {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse"),
            source: ShaderSource::Wgsl(RIGID_APPLY_POSITION_IMPULSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidApplyPositionImpulse {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RigidApplyPositionImpulseResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RigidApplyPositionImpulseQuery],
    ) -> Vec<RigidApplyPositionImpulseResult> {
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
            label: Some("prism_volumetric_rigid_apply_position_impulse_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_bind_group"),
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
            label: Some("prism_volumetric_rigid_apply_position_impulse_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rigid_apply_position_impulse_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rigid_apply_position_impulse_pass"),
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
