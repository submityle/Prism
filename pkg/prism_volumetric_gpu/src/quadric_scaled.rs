//! `wgpu` compute twin of the quadric scalar-scaling helper from the `CPU`
//! golden `prism_physics_core::collider::quadric::Quadric::scaled`.
//!
//! A quadric error metric packs the symmetric `4x4` matrix of a plane (or an
//! accumulation of planes) into the ten upper-triangular coefficients
//! `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`. Scaling a quadric by a scalar `s`
//! simply multiplies every coefficient by `s`; this is the operation used when
//! weighting a face quadric by its area before accumulation. This module ports
//! that stateless, no-`RNG`, branch-free closed form onto the device: one
//! compute thread resolves one quadric, so a passing real-device parity test is
//! direct evidence the kernel reproduces the exact per-coefficient multiply,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one quadric (its ten coefficients) plus a scalar `scale`. The
//! kernel reproduces the reference closed form coefficient for coefficient:
//!
//! * `a2' = a2*s`, `ab' = ab*s`, `ac' = ac*s`, `ad' = ad*s`;
//! * `b2' = b2*s`, `bc' = bc*s`, `bd' = bd*s`;
//! * `c2' = c2*s`, `cd' = cd*s`, `d2' = d2*s`;
//!
//! with `s = scale`.
//!
//! There is no division and no branch: the computation is pure multiplication,
//! so the kernel provably terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! Each coefficient is a single multiply, so `CPU` and `GPU` are not required
//! to be bit-exact. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the ten coefficients; the
//! discrete `valid` flag is compared exactly. The kernel has no comparisons at
//! all, so no bare float equality and no fast-math `NaN` sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every quadric and every scalar yields ten
//! well-defined coefficients, so `valid` is always `1`. A `scale` of `0` yields
//! the zero quadric and `scale` of `1` the identity. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `*` on `f32` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no float modulo and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::scaled`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` quadric-scaling kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `scale_quadric`
/// mirrors the `CPU` golden `Quadric::scaled`; see the module documentation for
/// the closed form.
const QUADRIC_SCALED_WGSL: &str = r#"
// Quadric-scaling twin: one thread per quadric multiplies each of the ten
// upper-triangular coefficients the golden Quadric::scaled scales by a scalar.
// It mirrors the CPU golden coefficient for coefficient, uses only the portable
// core-WGSL subset (f32 multiply plus unsigned index math), takes no optional
// feature, and has no loop, so the kernel provably terminates. There is no
// division and no branch, so no float equality or fast-math sentinel is
// involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::quadric::Quadric::scaled；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of quadrics in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ten upper-triangular quadric coefficients in the golden's order.
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
    // The scalar multiplier.
    scale: f32,
    // Padding word to a 16-byte-friendly stride.
    pad0: f32,
}

struct Result {
    // Ten scaled coefficients in the golden's order.
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
    // Always 1: the closed form has no degenerate branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn scale_quadric(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let s = q.scale;

    var out: Result;
    out.a2 = q.a2 * s;
    out.ab = q.ab * s;
    out.ac = q.ac * s;
    out.ad = q.ad * s;
    out.b2 = q.b2 * s;
    out.bc = q.bc * s;
    out.bd = q.bd * s;
    out.c2 = q.c2 * s;
    out.cd = q.cd * s;
    out.d2 = q.d2 * s;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUADRIC_SCALED_WGSL`].
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
/// The ten coefficients and the scalar are scalar `f32`, with one trailing pad
/// word to a `48`-byte, `16`-byte-friendly stride, so the host and device agree
/// on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coeffs: [f32; 10],
    scale: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The ten scaled coefficients are stored as flat scalars in the
/// golden's order; the trailing `valid` word keeps the discrete flag beside
/// them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    coeffs: [f32; 10],
    valid: u32,
}

/// One query for the quadric-scaling twin: the ten quadric coefficients and the
/// scalar multiplier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricScaledQuery {
    /// The ten coefficients, in the golden's order:
    /// `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
    pub coeffs: [f32; 10],
    /// The scalar multiplier applied to every coefficient.
    pub scale: f32,
}

impl QuadricScaledQuery {
    /// Builds a query from the ten coefficients and the scalar multiplier.
    #[must_use]
    pub fn new(coeffs: [f32; 10], scale: f32) -> QuadricScaledQuery {
        QuadricScaledQuery { coeffs, scale }
    }
}

/// One resolved answer for a single quadric: the ten scaled coefficients in the
/// golden's order, plus the `valid` flag (always `1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricScaledResult {
    /// The ten scaled coefficients, in the golden's order:
    /// `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
    pub coeffs: [f32; 10],
    /// Always `1`: the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`QuadricScaledQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricScaledQuery) -> GpuQuery {
    GpuQuery {
        coeffs: q.coeffs,
        scale: q.scale,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuadricScaledResult`].
fn decode_result(raw: &GpuResult) -> QuadricScaledResult {
    QuadricScaledResult {
        coeffs: raw.coeffs,
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

/// A compiled, reusable quadric-scaling compute pipeline, twinning the `CPU`
/// golden `Quadric::scaled`.
pub struct GpuQuadricScaled {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricScaled {
    /// Compiles the quadric-scaling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricScaled {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_scaled"),
            source: ShaderSource::Wgsl(QUADRIC_SCALED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_scaled_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_scaled_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_scaled_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("scale_quadric"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricScaled {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`QuadricScaledResult`]
    /// per input, in order.
    ///
    /// Each coefficient matches the reference to the module's tolerance and the
    /// `valid` flag matches exactly. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuadricScaledQuery],
    ) -> Vec<QuadricScaledResult> {
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
            label: Some("prism_volumetric_quadric_scaled_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_scaled_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_scaled_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_scaled_bind_group"),
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
            label: Some("prism_volumetric_quadric_scaled_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_scaled_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_scaled_pass"),
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
