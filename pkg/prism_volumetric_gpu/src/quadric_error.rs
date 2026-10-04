//! `wgpu` compute twin of the quadric-error evaluation from the `CPU` golden
//! `prism_physics_core::collider::quadric::Quadric::error`.
//!
//! A Garland--Heckbert error quadric is the symmetric `4x4` matrix `Q` stored as
//! its ten distinct coefficients. Evaluated at a point `v = (x, y, z, 1)`, the
//! quadratic form `v^T Q v` is the summed squared distance from `v` to the
//! planes the quadric accumulates, clamped at zero so round-off never yields a
//! spuriously negative cost. One thread resolves one independent query.
//!
//! [`GpuQuadricError`] is the on-device twin; a passing real-device parity test
//! is direct evidence the ported kernel reproduces the same ordered sum of
//! products the reference computes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate and
//! of `glam`, the exact ordered expansion of `v^T Q v`:
//!
//! ```text
//! e = a2*x*x
//!   + 2*ab*x*y + 2*ac*x*z + 2*ad*x
//!   + b2*y*y
//!   + 2*bc*y*z + 2*bd*y
//!   + c2*z*z
//!   + 2*cd*z
//!   + d2
//! error = max(e, 0)
//! ```
//!
//! The additive chain is emitted in the exact same order as the golden so a
//! fused multiply-add or a reassociation on the device cannot drift the result.
//! There is no loop and no division: each thread runs a fixed, bounded sequence
//! of multiply-adds, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The error is a continuous quantity, compared with an absolute-or-relative
//! tolerance in the parity test. `valid` is always `1` (the quadratic form is
//! total — every quadric and point yields a definite, non-negative error) and is
//! compared exactly. There is no degenerate branch.
//!
//! # Degenerate inputs
//!
//! There is no division and no square root, so there is no degenerate guard. The
//! zero quadric reports zero error everywhere; the `max(e, 0)` clamp absorbs any
//! small negative round-off. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `+ - *` — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no float `%`, and no `u64`
//! / `i64` / `f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! clamp uses the `max` builtin rather than a bare float comparison, which is
//! robust under `Metal`'s fast-math.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric`；无第三方
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

/// The portable core-`WGSL` quadric-error kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `error`; see the module docs for the closed form it reproduces.
const QUADRIC_ERROR_WGSL: &str = r#"
// Quadric-error twin: one thread resolves one independent query. It expands the
// quadratic form v^T Q v in the exact same term order as the CPU golden, then
// clamps at zero. It uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The ten distinct symmetric quadric coefficients, in golden order.
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
    // Query point.
    vx: f32,
    vy: f32,
    vz: f32,
}

struct Verdict {
    // The clamped quadratic form max(v^T Q v, 0).
    error: f32,
    // 1 always: the quadratic form is total.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Verdict>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let x = q.vx;
    let y = q.vy;
    let z = q.vz;

    // Ordered expansion of v^T Q v, matching the golden term by term.
    let e = q.a2 * x * x
        + 2.0 * q.ab * x * y
        + 2.0 * q.ac * x * z
        + 2.0 * q.ad * x
        + q.b2 * y * y
        + 2.0 * q.bc * y * z
        + 2.0 * q.bd * y
        + q.c2 * z * z
        + 2.0 * q.cd * z
        + q.d2;

    var out: Verdict;
    out.error = max(e, 0.0);
    out.valid = 1u;
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
/// Thirteen `f32` give a fixed `52`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `52` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Verdict`
/// struct. One `f32` and one `u32` give a fixed `8`-byte stride with no pad,
/// since the struct alignment is `4` and `8` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    error: f32,
    valid: u32,
}

/// One query for the quadric-error twin: the ten distinct symmetric quadric
/// coefficients (in golden order) and the point at which to evaluate the
/// quadratic form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricErrorQuery {
    /// Coefficient `a2` (row 0, col 0).
    pub a2: f32,
    /// Coefficient `ab` (row 0, col 1).
    pub ab: f32,
    /// Coefficient `ac` (row 0, col 2).
    pub ac: f32,
    /// Coefficient `ad` (row 0, col 3).
    pub ad: f32,
    /// Coefficient `b2` (row 1, col 1).
    pub b2: f32,
    /// Coefficient `bc` (row 1, col 2).
    pub bc: f32,
    /// Coefficient `bd` (row 1, col 3).
    pub bd: f32,
    /// Coefficient `c2` (row 2, col 2).
    pub c2: f32,
    /// Coefficient `cd` (row 2, col 3).
    pub cd: f32,
    /// Coefficient `d2` (row 3, col 3).
    pub d2: f32,
    /// `x` of the query point.
    pub vx: f32,
    /// `y` of the query point.
    pub vy: f32,
    /// `z` of the query point.
    pub vz: f32,
}

impl QuadricErrorQuery {
    /// Builds a query from the ten quadric coefficients (in golden order) and
    /// the query point.
    ///
    /// The coefficients and the point are grouped into fixed-length arrays so
    /// the constructor stays within a small, readable argument count.
    #[must_use]
    pub fn new(coefficients: [f32; 10], point: [f32; 3]) -> QuadricErrorQuery {
        QuadricErrorQuery {
            a2: coefficients[0],
            ab: coefficients[1],
            ac: coefficients[2],
            ad: coefficients[3],
            b2: coefficients[4],
            bc: coefficients[5],
            bd: coefficients[6],
            c2: coefficients[7],
            cd: coefficients[8],
            d2: coefficients[9],
            vx: point[0],
            vy: point[1],
            vz: point[2],
        }
    }
}

/// One resolved verdict for a single query: the clamped quadratic form and the
/// validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricErrorResult {
    /// The clamped quadratic form `max(v^T Q v, 0)`.
    pub error: f32,
    /// `1` always: the quadratic form is total.
    pub valid: u32,
}

/// Encodes one [`QuadricErrorQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricErrorQuery) -> GpuQuery {
    GpuQuery {
        a2: q.a2,
        ab: q.ab,
        ac: q.ac,
        ad: q.ad,
        b2: q.b2,
        bc: q.bc,
        bd: q.bd,
        c2: q.c2,
        cd: q.cd,
        d2: q.d2,
        vx: q.vx,
        vy: q.vy,
        vz: q.vz,
    }
}

/// Decodes one `std430` [`GpuResult`] slot into a public [`QuadricErrorResult`].
fn decode_result(r: &GpuResult) -> QuadricErrorResult {
    QuadricErrorResult {
        error: r.error,
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

/// The on-device twin of the `CPU` golden `error`: a compiled compute pipeline
/// that evaluates a batch of quadric-error queries.
pub struct GpuQuadricError {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricError {
    /// Compiles the inline kernel and builds the compute pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricError {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_error_module"),
            source: ShaderSource::Wgsl(QUADRIC_ERROR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_error_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_error_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_error_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricError {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`QuadricErrorResult`]
    /// per input, in order.
    ///
    /// The `error` matches the reference within tolerance and `valid` matches
    /// exactly. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuadricErrorQuery],
    ) -> Vec<QuadricErrorResult> {
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
            label: Some("prism_volumetric_quadric_error_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_error_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_error_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_error_bind_group"),
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
            label: Some("prism_volumetric_quadric_error_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_error_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_error_pass"),
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
