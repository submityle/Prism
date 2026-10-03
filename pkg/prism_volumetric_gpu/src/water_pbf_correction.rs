//! `wgpu` compute twin of the Position-Based Fluids artificial-pressure and
//! position-correction numerics
//! ([`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
//! and
//! [`position_correction`](prism_render_architecture::water::pbf::position_correction)).
//!
//! Position-Based Fluids (Macklin & Müller) nudge each particle so the
//! `SPH`-estimated density returns to rest. Two of the per-particle numerics in
//! that solve are stateless, closed-form, and fixed-width: the
//! artificial-pressure term `s_corr` that fights particle clustering, and the
//! final position correction `delta_p` assembled from a bounded neighbour list.
//! Both port cleanly to the device, so a passing real-device parity run is
//! direct evidence the ported kernel honours the same `Poly6` ratio and
//! gradient-weighted sum the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! One thread computes one particle's query, producing both quantities:
//!
//! - `s_corr` mirrors
//!   [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure):
//!   `k = max(ap_k, 0)`; a near-zero `k` short-circuits to `0`. The reference
//!   kernel is evaluated at `dq = clamp(delta_q, 0, 1) * h`; a near-zero
//!   `reference = poly6(dq * dq, h)` also short-circuits to `0`. Otherwise
//!   `ratio = poly6(r_squared, h) / reference`, `power = ratio^n` by repeated
//!   squaring, and the term is `-k * power`.
//! - `corr` mirrors
//!   [`position_correction`](prism_render_architecture::water::pbf::position_correction):
//!   a non-positive `rest_density` short-circuits to the zero vector; otherwise
//!   `(1 / rest_density) * sum_j (lambda_i + lambda_j + scorr_j) * gradient_j`
//!   over the bounded neighbour list.
//!
//! The [`poly6`](prism_render_architecture::water::pbf::poly6) kernel is
//! reproduced inline exactly: `0` for a non-positive `h` or for `r^2` at or
//! beyond `h^2`, else `315 / (64 * PI * h^9) * (h^2 - r^2)^3`.
//!
//! # What stays on the host
//!
//! The spatial-hash binning, the variable-length neighbour gather, the density
//! estimate, the `lambda` solve, and the solver-iteration schedule are
//! stateful, variable-length host passes. The host supplies the already-solved
//! `lambda_i`, each neighbour's `lambda_j`, `s_corr`, and `Spiky` gradient, and
//! a neighbour count bounded by [`MAX_NEIGHBORS`]; the device only folds the
//! fixed-width arithmetic.
//!
//! # Correctness model
//!
//! Every result is a short sequence of multiplies, adds, guarded divides, and a
//! fixed repeated-squaring loop — no transcendental — so the `CPU` and `GPU`
//! agree to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`, `/`,
//! `abs`, `min`, `max`, `clamp`, bit shifts on `u32`, and a bounded loop — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `smoothstep`, and no `round`. The degenerate legs return early, so a divide
//! by zero is never observed. Each thread performs a bounded sequence of
//! arithmetic, so the kernel provably terminates. No optional device feature is
//! required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。
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

/// Fixed upper bound on the neighbours one query may carry, matching the
/// `array<Neighbor, 8>` member of the `WGSL` `Query` struct. The host clamps a
/// particle's neighbour count to this before encoding.
pub const MAX_NEIGHBORS: usize = 8;

/// The inlined `WGSL` twin of
/// [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
/// and
/// [`position_correction`](prism_render_architecture::water::pbf::position_correction),
/// including the [`poly6`](prism_render_architecture::water::pbf::poly6) kernel
/// and the integer-power repeated-squaring, computed with only `+`, `-`, `*`,
/// `/`, `max`, `clamp`, `u32` bit shifts, and a bounded loop.
const WATER_PBF_CORRECTION_WGSL: &str = r#"
// Twin of water::pbf::{artificial_pressure, position_correction}. poly6 and the
// integer power (repeated squaring) are reproduced verbatim; no transcendental,
// only polynomial arithmetic and guarded divides.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；无第三方引擎源码或衍生代码。

// Shared epsilon matching the golden `water::EPS`.
const EPS: f32 = 1.0e-6;
// Matches the golden `water::PI` (`core::f32::consts::PI`); the long decimal
// rounds to the same f32 bit pattern.
const PI: f32 = 3.14159265358979;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Neighbor {
    // Neighbour's scaling factor lambda_j.
    lambda_j: f32,
    // Artificial-pressure term s_corr for this pair.
    scorr: f32,
    // Spiky gradient grad W_ij for this pair.
    grad_x: f32,
    grad_y: f32,
    grad_z: f32,
}

struct Query {
    // Squared neighbour distance for the artificial-pressure term.
    r_squared: f32,
    // Artificial-pressure strength k.
    ap_k: f32,
    // Smoothing radius h.
    ap_h: f32,
    // Fraction of h for the reference kernel.
    ap_delta_q: f32,
    // Artificial-pressure exponent n.
    ap_n: u32,
    // This particle's scaling factor lambda_i.
    lambda_i: f32,
    // Target rest density rho_0.
    rest_density: f32,
    // Number of valid neighbours in `neighbors` (<= 8).
    neighbor_count: u32,
    // Fixed-capacity neighbour list.
    neighbors: array<Neighbor, 8>,
}

struct Result {
    // Artificial-pressure correction s_corr.
    s_corr: f32,
    // Position correction delta_p.
    corr_x: f32,
    corr_y: f32,
    corr_z: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Poly6 smoothing kernel W(r, h) from the squared distance. Zero for a
// non-positive h or for r^2 at or beyond h^2, else 315/(64*PI*h^9)*(h^2-r^2)^3.
fn poly6(r_squared: f32, h: f32) -> f32 {
    if (h <= EPS) {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = max(r_squared, 0.0);
    if (r2 >= h2) {
        return 0.0;
    }
    let h9 = h2 * h2 * h2 * h2 * h;
    let coeff = 315.0 / (64.0 * PI * h9);
    let d = h2 - r2;
    return coeff * d * d * d;
}

// Integer power by repeated squaring, matching the golden `powi`.
fn powi(base: f32, exp: u32) -> f32 {
    var result: f32 = 1.0;
    var b: f32 = base;
    var e: u32 = exp;
    loop {
        if (e == 0u) {
            break;
        }
        if ((e & 1u) == 1u) {
            result = result * b;
        }
        e = e >> 1u;
        if (e > 0u) {
            b = b * b;
        }
    }
    return result;
}

// Artificial-pressure correction s_corr = -k * (poly6(r^2) / poly6(dq^2))^n.
fn artificial_pressure(r_squared: f32, ap_k: f32, h: f32, delta_q: f32, n: u32) -> f32 {
    let k = max(ap_k, 0.0);
    if (k <= EPS) {
        return 0.0;
    }
    let dq = clamp(delta_q, 0.0, 1.0) * h;
    let reference = poly6(dq * dq, h);
    if (reference <= EPS) {
        return 0.0;
    }
    let ratio = poly6(r_squared, h) / reference;
    let power = powi(ratio, max(n, 1u));
    return -k * power;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.s_corr = artificial_pressure(q.r_squared, q.ap_k, q.ap_h, q.ap_delta_q, q.ap_n);

    var cx: f32 = 0.0;
    var cy: f32 = 0.0;
    var cz: f32 = 0.0;
    if (q.rest_density <= EPS) {
        out.corr_x = 0.0;
        out.corr_y = 0.0;
        out.corr_z = 0.0;
    } else {
        for (var j: u32 = 0u; j < q.neighbor_count; j = j + 1u) {
            let nb = q.neighbors[j];
            let weight = q.lambda_i + nb.lambda_j + nb.scorr;
            cx = cx + nb.grad_x * weight;
            cy = cy + nb.grad_y * weight;
            cz = cz + nb.grad_z * weight;
        }
        let inv = 1.0 / q.rest_density;
        out.corr_x = cx * inv;
        out.corr_y = cy * inv;
        out.corr_z = cz * inv;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_PBF_CORRECTION_WGSL`].
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

/// `repr(C)` `std430` layout of one neighbour contribution, matching the `WGSL`
/// `Neighbor` struct: `lambda_j`, `scorr`, and the three gradient components
/// (five `f32`, `20` bytes, alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNeighbor {
    /// Neighbour's scaling factor `lambda_j`.
    lambda_j: f32,
    /// Artificial-pressure term `s_corr` for this pair.
    scorr: f32,
    /// `Spiky` gradient x component.
    grad_x: f32,
    /// `Spiky` gradient y component.
    grad_y: f32,
    /// `Spiky` gradient z component.
    grad_z: f32,
}

/// `repr(C)` `std430` layout of one correction query, matching the `WGSL`
/// `Query` struct: the artificial-pressure operands, the position-correction
/// operands, the neighbour count, and the fixed-capacity neighbour list.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Squared neighbour distance for the artificial-pressure term.
    r_squared: f32,
    /// Artificial-pressure strength `k`.
    ap_k: f32,
    /// Smoothing radius `h`.
    ap_h: f32,
    /// Fraction of `h` for the reference kernel.
    ap_delta_q: f32,
    /// Artificial-pressure exponent `n`.
    ap_n: u32,
    /// This particle's scaling factor `lambda_i`.
    lambda_i: f32,
    /// Target rest density `rho_0`.
    rest_density: f32,
    /// Number of valid neighbours in `neighbors`.
    neighbor_count: u32,
    /// Fixed-capacity neighbour list.
    neighbors: [GpuNeighbor; MAX_NEIGHBORS],
}

/// `repr(C)` `std430` layout of one correction result, matching the `WGSL`
/// `Result` struct: the artificial-pressure term and the three position
/// correction components.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Artificial-pressure correction `s_corr`.
    s_corr: f32,
    /// Position correction x component.
    corr_x: f32,
    /// Position correction y component.
    corr_y: f32,
    /// Position correction z component.
    corr_z: f32,
}

/// One neighbour's contribution to the position correction, mirroring the
/// golden [`NeighborContribution`](prism_render_architecture::water::pbf::NeighborContribution).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfNeighbor {
    /// The neighbour's scaling factor `lambda_j`.
    pub lambda_j: f32,
    /// Artificial-pressure term `s_corr` for this pair.
    pub scorr: f32,
    /// `Spiky` gradient x component.
    pub grad_x: f32,
    /// `Spiky` gradient y component.
    pub grad_y: f32,
    /// `Spiky` gradient z component.
    pub grad_z: f32,
}

/// One correction query to run on the device, mirroring the inputs of
/// [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
/// and
/// [`position_correction`](prism_render_architecture::water::pbf::position_correction).
///
/// A single thread produces both the artificial-pressure term and the position
/// correction; see the module documentation for the arithmetic. The neighbour
/// list is truncated to [`MAX_NEIGHBORS`] on encode.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterPbfCorrectionQuery {
    /// Squared neighbour distance for the artificial-pressure term.
    pub r_squared: f32,
    /// Artificial-pressure strength `k`.
    pub ap_k: f32,
    /// Smoothing radius `h`.
    pub ap_h: f32,
    /// Fraction of `h` for the reference kernel.
    pub ap_delta_q: f32,
    /// Artificial-pressure exponent `n`.
    pub ap_n: u32,
    /// This particle's scaling factor `lambda_i`.
    pub lambda_i: f32,
    /// Target rest density `rho_0`.
    pub rest_density: f32,
    /// Neighbour contributions for the position correction.
    pub neighbors: Vec<WaterPbfNeighbor>,
}

/// One resolved correction result, mirroring the golden artificial-pressure
/// term and position correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfCorrectionResult {
    /// Artificial-pressure correction `s_corr`.
    pub s_corr: f32,
    /// Position correction x component.
    pub corr_x: f32,
    /// Position correction y component.
    pub corr_y: f32,
    /// Position correction z component.
    pub corr_z: f32,
}

/// Encodes one [`WaterPbfCorrectionQuery`] into its `std430` [`GpuQuery`] slot,
/// truncating the neighbour list to [`MAX_NEIGHBORS`] and zero-filling the
/// unused slots.
fn encode_query(q: &WaterPbfCorrectionQuery) -> GpuQuery {
    let mut neighbors = [GpuNeighbor::zeroed(); MAX_NEIGHBORS];
    let count = q.neighbors.len().min(MAX_NEIGHBORS);
    for (slot, nb) in neighbors.iter_mut().zip(q.neighbors.iter()).take(count) {
        *slot = GpuNeighbor {
            lambda_j: nb.lambda_j,
            scorr: nb.scorr,
            grad_x: nb.grad_x,
            grad_y: nb.grad_y,
            grad_z: nb.grad_z,
        };
    }
    GpuQuery {
        r_squared: q.r_squared,
        ap_k: q.ap_k,
        ap_h: q.ap_h,
        ap_delta_q: q.ap_delta_q,
        ap_n: q.ap_n,
        lambda_i: q.lambda_i,
        rest_density: q.rest_density,
        neighbor_count: count as u32,
        neighbors,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterPbfCorrectionResult`].
fn decode_result(raw: &GpuResult) -> WaterPbfCorrectionResult {
    WaterPbfCorrectionResult {
        s_corr: raw.s_corr,
        corr_x: raw.corr_x,
        corr_y: raw.corr_y,
        corr_z: raw.corr_z,
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

/// A compiled, reusable `PBF` correction compute pipeline, twinning the numeric
/// core of the `CPU` golden
/// [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
/// and
/// [`position_correction`](prism_render_architecture::water::pbf::position_correction).
pub struct GpuWaterPbfCorrection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfCorrection {
    /// Compiles the `PBF` correction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfCorrection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_correction"),
            source: ShaderSource::Wgsl(WATER_PBF_CORRECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfCorrection {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one
    /// [`WaterPbfCorrectionResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfCorrectionQuery],
    ) -> Vec<WaterPbfCorrectionResult> {
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
            label: Some("prism_volumetric_water_pbf_correction_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_bind_group"),
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
            label: Some("prism_volumetric_water_pbf_correction_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_correction_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_correction_pass"),
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
