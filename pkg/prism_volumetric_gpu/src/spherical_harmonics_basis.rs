//! `wgpu` compute twin of the `CPU` golden `prism_math::spherical` real
//! spherical-harmonics basis family: the transcendental-free polynomial basis
//! `basis3`/`basis2`, the dot-product reconstruction `Sh3::eval`, and the
//! clamped-cosine convolution `Sh3::convolve_cosine`, dispatched behind an
//! integer selector so one thread resolves one query.
//!
//! This is the *basis evaluation* family and is distinct from the already
//! twinned `spherical_harmonics_rotate` module, which rotates coefficient
//! vectors. Here every output is a direct polynomial of the (assumed unit)
//! direction and the input coefficients, with no transcendental anywhere.
//!
//! A passing real-device parity test is direct evidence the ported kernel
//! computes the same algebra the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The `op_id` selects which golden operator the thread reproduces for its
//! direction `dir` and 16 input coefficients `coeffs`:
//!
//! * `op_id == 0` mirrors `basis3`: the 16 real `SH` basis values for bands
//!   `0..=3`, returned in `out[0..16]`.
//! * `op_id == 1` mirrors `basis2`: the first nine basis values (bands
//!   `0..=2`) in `out[0..9]`, with `out[9..16]` cleared to zero.
//! * `op_id == 2` mirrors `Sh3::eval`: the scalar dot product
//!   `dot(coeffs, basis3(dir))` in `out[0]`, with the rest cleared.
//! * `op_id == 3` mirrors `Sh3::convolve_cosine`: each band scaled by its
//!   clamped-cosine multiplier (`COS_A0` for band 0, `COS_A1` for band 1,
//!   `COS_A2` for band 2, and exactly `0` for band 3), in `out[0..16]`.
//!
//! Any `op_id > 3` yields `valid = 0` with a cleared `out`.
//!
//! # Correctness model
//!
//! The golden evaluates every operator in `f32`, and the kernel reproduces the
//! same `f32` operator order; `CPU` and `GPU` are therefore not bit-exact (a
//! `GPU` may contract a multiply-add). Each valid `out` entry is compared with
//! an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. Band 3 of the convolution is exactly zero
//! and is asserted as such.
//!
//! # Degenerate inputs
//!
//! There is no numeric degeneracy in this family: every operator is a finite
//! polynomial of finite inputs. The only invalid case is an out-of-range
//! `op_id` (greater than `3`), which yields `valid = 0` with a cleared `out`.
//! An empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `select`,
//! ordered compares and unsigned index arithmetic — with no transcendental, no
//! `round`, no `f32` remainder and no banned wide types, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. Operator and validity selection use
//! integer `==`/`<=` on the `u32` id; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::spherical`；无第三方引擎源码或衍生代码。
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

/// Operator id for `basis3` (16 basis values for bands `0..=3`).
pub const OP_BASIS3: u32 = 0;
/// Operator id for `basis2` (first nine basis values; the rest cleared).
pub const OP_BASIS2: u32 = 1;
/// Operator id for `Sh3::eval` (scalar dot product in `out[0]`).
pub const OP_EVAL_SH3: u32 = 2;
/// Operator id for `Sh3::convolve_cosine` (per-band clamped-cosine scaling).
pub const OP_CONVOLVE_COSINE_SH3: u32 = 3;

/// The portable core-`WGSL` spherical-harmonics basis kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the four `CPU` golden operators selected by `op_id`; see the module
/// documentation for the closed forms.
const SPHERICAL_HARMONICS_BASIS_WGSL: &str = r#"
// Spherical-harmonics basis twin: one thread per query reproduces one of four
// prism_math::spherical operators selected by op_id. It uses only the portable
// core-WGSL subset (+ - * /, select, ordered compares plus unsigned index
// math); there is no transcendental, no round, no f32 remainder and no banned
// wide type. Operator and validity selection use integer == / <= on the u32
// id; there is no bare f32 equality.
// Provenance: 孪生自本仓 prism_math::spherical。

const K0: f32 = 0.28209479;
const K1: f32 = 0.48860251;
const K2A: f32 = 1.0925484;
const K2B: f32 = 0.31539157;
const K2C: f32 = 0.5462742;
const K3A: f32 = 0.5900436;
const K3B: f32 = 2.8906114;
const K3C: f32 = 0.4570458;
const K3D: f32 = 0.3731763;
const COS_A0: f32 = 3.14159265;
const COS_A1: f32 = 2.0943951;
const COS_A2: f32 = 0.78539816;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Input coefficients for the eval and convolution operators.
    coeffs: array<f32, 16>,
    // Direction (assumed unit length) for the basis and eval operators.
    dx: f32,
    dy: f32,
    dz: f32,
    // Which golden operator to reproduce.
    op_id: u32,
}

struct Result {
    // Flattened result: 16 basis/convolution entries, or a scalar in o[0] for
    // eval with the rest zero.
    o: array<f32, 16>,
    // 1 when the operator id is known (<= 3), else 0.
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
    let x = q.dx;
    let y = q.dy;
    let z = q.dz;
    let op_id = q.op_id;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;

    // basis3: the 16 real SH basis values, in exact golden order.
    var basis: array<f32, 16>;
    basis[0] = K0;
    basis[1] = K1 * y;
    basis[2] = K1 * z;
    basis[3] = K1 * x;
    basis[4] = K2A * x * y;
    basis[5] = K2A * y * z;
    basis[6] = K2B * (3.0 * z2 - 1.0);
    basis[7] = K2A * x * z;
    basis[8] = K2C * (x2 - y2);
    basis[9] = K3A * y * (3.0 * x2 - y2);
    basis[10] = K3B * x * y * z;
    basis[11] = K3C * y * (5.0 * z2 - 1.0);
    basis[12] = K3D * z * (5.0 * z2 - 3.0);
    basis[13] = K3C * x * (5.0 * z2 - 1.0);
    basis[14] = 0.5 * K3B * z * (x2 - y2);
    basis[15] = K3A * x * (x2 - 3.0 * y2);

    // eval: sequential dot product of coeffs and basis3, matching the golden
    // iterator sum order.
    var acc = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc = acc + q.coeffs[i] * basis[i];
    }

    let valid_bool = op_id <= 3u;

    var out: Result;
    for (var i = 0u; i < 16u; i = i + 1u) {
        // Per-band clamped-cosine multiplier for the convolution operator.
        var mult = 0.0;
        mult = select(mult, COS_A0, i == 0u);
        mult = select(mult, COS_A1, (i >= 1u) && (i <= 3u));
        mult = select(mult, COS_A2, (i >= 4u) && (i <= 8u));

        var val = 0.0;
        // op 0: basis3 -> all 16 entries.
        val = select(val, basis[i], op_id == 0u);
        // op 1: basis2 -> first nine entries, rest zero.
        val = select(val, select(0.0, basis[i], i < 9u), op_id == 1u);
        // op 2: eval -> scalar in o[0].
        val = select(val, select(0.0, acc, i == 0u), op_id == 2u);
        // op 3: convolve_cosine -> per-band scaling.
        val = select(val, q.coeffs[i] * mult, op_id == 3u);

        out.o[i] = select(0.0, val, valid_bool);
    }
    out.valid = select(0u, 1u, valid_bool);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the 16 input coefficients, the three direction components and the operator
/// id. The `array<f32, 16>` member has element stride `4`, so this maps
/// byte-for-byte onto the `[f32; 16]` field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coeffs: [f32; 16],
    dx: f32,
    dy: f32,
    dz: f32,
    op_id: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the 16 flattened output entries and the validity flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    o: [f32; 16],
    valid: u32,
}

/// One spherical-harmonics basis query: an operator selector, a direction and
/// the 16 input coefficients used by the eval and convolution operators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalHarmonicsBasisQuery {
    /// Which golden operator to reproduce (see the `OP_*` constants).
    pub op_id: u32,
    /// The direction (assumed unit length) for the basis and eval operators.
    pub dir: [f32; 3],
    /// The 16 input coefficients for `eval` and `convolve_cosine`; ignored by
    /// the basis operators.
    pub coeffs: [f32; 16],
}

impl SphericalHarmonicsBasisQuery {
    /// Builds a query selecting `op_id` for direction `dir` and `coeffs`.
    #[must_use]
    pub fn new(op_id: u32, dir: [f32; 3], coeffs: [f32; 16]) -> SphericalHarmonicsBasisQuery {
        SphericalHarmonicsBasisQuery { op_id, dir, coeffs }
    }
}

/// One resolved answer for a single query, mirroring the selected golden
/// operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalHarmonicsBasisResult {
    /// The flattened result: 16 basis/convolution entries, or a scalar in
    /// `out[0]` for `eval` with the rest zero. Cleared to zero when invalid.
    pub out: [f32; 16],
    /// `1` when the operator id is known (`<= 3`), else `0`.
    pub valid: u32,
}

/// Encodes one [`SphericalHarmonicsBasisQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SphericalHarmonicsBasisQuery) -> GpuQuery {
    GpuQuery {
        coeffs: q.coeffs,
        dx: q.dir[0],
        dy: q.dir[1],
        dz: q.dir[2],
        op_id: q.op_id,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SphericalHarmonicsBasisResult`].
fn decode_result(raw: &GpuResult) -> SphericalHarmonicsBasisResult {
    SphericalHarmonicsBasisResult {
        out: raw.o,
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

/// A compiled, reusable spherical-harmonics basis compute pipeline, twinning
/// the four `CPU` golden `prism_math::spherical` basis operators.
pub struct GpuSphericalHarmonicsBasis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphericalHarmonicsBasis {
    /// Compiles the spherical-harmonics basis kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphericalHarmonicsBasis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis"),
            source: ShaderSource::Wgsl(SPHERICAL_HARMONICS_BASIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphericalHarmonicsBasis {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SphericalHarmonicsBasisResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each `out` entry to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SphericalHarmonicsBasisQuery],
    ) -> Vec<SphericalHarmonicsBasisResult> {
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
            label: Some("prism_volumetric_spherical_harmonics_basis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_bind_group"),
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
            label: Some("prism_volumetric_spherical_harmonics_basis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_basis_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spherical_harmonics_basis_pass"),
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
