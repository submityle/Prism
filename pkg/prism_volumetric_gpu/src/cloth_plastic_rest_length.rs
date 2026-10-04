//! `wgpu` compute twin of the plastic rest-length creep scalar kernel from the
//! `CPU` golden `prism_physics_core::soft::damage::plasticity::plastic_rest_length`.
//!
//! Plasticity lets an over-stretched (or over-compressed) cloth distance edge
//! permanently creep its rest length toward the current length once the signed
//! strain leaves a yield band, capturing wrinkles and sag while a bounded
//! residual elastic strain is retained. Per edge the golden sanitises the
//! plastic parameters, rejects degenerate rest lengths and edges inside the
//! yield band, moves the rest length by `creep` times the beyond-yield excess
//! strain, floors it to a numerical epsilon, then clamps so the residual
//! elastic strain magnitude never exceeds `max_strain`. One thread solves one
//! query.
//!
//! [`GpuClothPlasticRestLength`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same
//! sanitisation, the same yield-band and degeneracy rejections, the same creep
//! arithmetic and the same residual cap the reference computes, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate: the
//! `PlasticParams::sanitized` rule (`yield_strain` and `max_strain` forced
//! non-negative with a `NaN` mapped to `0`, `creep` clamped to `[0, 1]` with a
//! `NaN` mapped to `0`); the degenerate rest-length rejection
//! (`rest_length <= EPS_REST`); the yield-band rejection
//! (`|strain| <= yield_strain`); the creep update
//! `new_rest = rest * (1 + creep * excess)` with `excess = strain - sign *
//! yield_strain`; the epsilon floor; the residual cap
//! `new_rest = length / (1 + residual_sign * max_strain)` when
//! `|residual| > max_strain`; and the final strictly-positive rest-length
//! acceptance. There is no loop: each thread performs a fixed, bounded sequence
//! of multiplies, adds, divides, clamps and selects, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! The new rest length threads through subtracts, divides and multiplies, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the
//! continuous `new_rest` channel. The discrete `valid` flag is compared exactly:
//! it is `1` only when the edge creeps to a strictly positive rest length and
//! `0` for every rejection (degenerate rest, inside the yield band, or a floored
//! collapse).
//!
//! # Degenerate inputs
//!
//! A rest length at or below `EPS_REST = 1e-9` is inert (`valid = 0`); an edge
//! whose signed strain lies within the sanitised yield band is left untouched
//! (`valid = 0`); and a creep update that collapses to the epsilon floor without
//! being lifted by the residual cap is rejected (`valid = 0`). The strain
//! denominator is guarded with a unit fallback when the rest length is
//! degenerate so the unselected arm cannot raise an infinity, and the residual
//! denominator is always evaluated on a rest length at or above the epsilon
//! floor. The residual-cap denominator `1 + residual_sign * max_strain` is
//! theoretically zero only when `residual_sign = -1` and `max_strain = 1`; the
//! sweep keeps `max_strain` well away from `1` so no real device ever divides by
//! zero. An empty query batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `select`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`,
//! no float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The `NaN` tests avoid bare float equality against a
//! literal by using the self-inequality `x == x`, which behaves correctly for
//! `NaN`; signs are chosen with ordered comparisons feeding `select`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::plasticity`；无第三方
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

/// The portable core-`WGSL` plastic rest-length creep kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `plastic_rest_length` composed with
/// `PlasticParams::sanitized`; see the module documentation.
const CLOTH_PLASTIC_REST_LENGTH_WGSL: &str = r#"
// Plastic rest-length creep twin: one thread per query sanitises the plastic
// parameters, rejects degenerate / in-band edges, moves the rest length by the
// creep fraction of the beyond-yield excess strain, floors it, then clamps the
// residual elastic strain. It mirrors the CPU golden exactly and uses only the
// portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Current rest length of the distance edge.
    rest_length: f32,
    // Current separation of the edge endpoints.
    length: f32,
    // Raw painted yield strain; sanitised non-negative before use.
    yield_strain: f32,
    // Raw painted creep fraction; sanitised into [0, 1] before use.
    creep: f32,
    // Raw painted residual-strain cap; sanitised non-negative before use.
    max_strain: f32,
}

struct Result {
    // Plastically crept rest length, or the epsilon placeholder when rejected.
    new_rest: f32,
    // 1 when the edge crept to a strictly positive rest length, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Rest length at or below which an edge is inert, matching the golden EPS_REST.
const EPS_REST: f32 = 1.0e-9;

// Forces a value non-negative, mapping a NaN to 0, matching the golden
// clamp_nonneg used by PlasticParams::sanitized. The guard is a single ordered
// comparison against zero: a NaN is unordered so `v >= 0.0` is false, which is
// robust even under Metal's fast-math (which may fold the IEEE self-inequality
// `v == v` to a constant). A finite negative also fails the test and maps to 0;
// a non-negative value (incl. +inf) passes through, matching the golden.
fn clamp_nonneg(v: f32) -> f32 {
    return select(0.0, v, v >= 0.0);
}

// Sanitises the creep fraction into [0, 1], mapping a NaN to 0, matching the
// golden creep sanitiser. Uses only ordered comparisons (NaN-safe under Metal
// fast-math): a NaN or negative input fails `c >= 0.0` and floors to 0, then an
// upper `min` caps at 1 (so +inf maps to 1, matching the golden clamp).
fn sanitize_creep(c: f32) -> f32 {
    let lo = select(0.0, c, c >= 0.0);
    return min(lo, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let rest = q.rest_length;
    let length = q.length;
    let yield_strain = clamp_nonneg(q.yield_strain);
    let creep = sanitize_creep(q.creep);
    let max_strain = clamp_nonneg(q.max_strain);

    // A degenerate rest length makes the edge inert; guard the strain divisor so
    // the unselected arm cannot raise an infinity before the valid gate drops it.
    let rest_ok = rest > EPS_REST;
    let safe_rest = select(1.0, rest, rest_ok);
    let strain = (length - rest) / safe_rest;

    // Edges whose signed strain stays within the yield band are left untouched.
    let beyond = abs(strain) > yield_strain;

    let sign = select(-1.0, 1.0, strain >= 0.0);
    let excess = strain - sign * yield_strain;
    // Move the rest length by `creep` fraction of the excess strain.
    var new_rest = rest * (1.0 + creep * excess);
    // Floor to the numerical epsilon so the residual divisor stays positive.
    new_rest = select(new_rest, EPS_REST, new_rest <= EPS_REST);

    // Clamp so the residual elastic strain magnitude stays within max_strain.
    let residual = (length - new_rest) / new_rest;
    let residual_sign = select(-1.0, 1.0, residual >= 0.0);
    let capped = length / (1.0 + residual_sign * max_strain);
    new_rest = select(new_rest, capped, abs(residual) > max_strain);

    // Accept only a strictly positive rest length that cleared the degeneracy
    // and yield-band gates.
    let accepted = rest_ok && beyond && (new_rest > EPS_REST);

    var out: Result;
    out.new_rest = select(EPS_REST, new_rest, accepted);
    out.valid = select(0u, 1u, accepted);
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
/// Five `f32` give a fixed `20`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `20` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    rest_length: f32,
    length: f32,
    yield_strain: f32,
    creep: f32,
    max_strain: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
/// One `f32` plus one `u32` give a fixed `8`-byte stride with no pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_rest: f32,
    valid: u32,
}

/// One query for the plastic rest-length creep twin: the current rest length,
/// the current edge length, and the raw painted plastic parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothPlasticRestLengthQuery {
    /// Current rest length of the distance edge.
    pub rest_length: f32,
    /// Current separation of the edge endpoints.
    pub length: f32,
    /// Raw painted yield strain; sanitised non-negative before use.
    pub yield_strain: f32,
    /// Raw painted creep fraction; sanitised into `[0, 1]` before use.
    pub creep: f32,
    /// Raw painted residual-strain cap; sanitised non-negative before use.
    pub max_strain: f32,
}

impl ClothPlasticRestLengthQuery {
    /// Builds a query from the rest length, edge length and raw plastic
    /// parameters.
    #[must_use]
    pub fn new(
        rest_length: f32,
        length: f32,
        yield_strain: f32,
        creep: f32,
        max_strain: f32,
    ) -> ClothPlasticRestLengthQuery {
        ClothPlasticRestLengthQuery {
            rest_length,
            length,
            yield_strain,
            creep,
            max_strain,
        }
    }
}

/// One resolved answer for a single query: the crept rest length and the
/// validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothPlasticRestLengthResult {
    /// Plastically crept rest length, or the epsilon placeholder when rejected.
    pub new_rest: f32,
    /// `1` when the edge crept to a strictly positive rest length, else `0`.
    pub valid: u32,
}

/// Encodes one [`ClothPlasticRestLengthQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothPlasticRestLengthQuery) -> GpuQuery {
    GpuQuery {
        rest_length: q.rest_length,
        length: q.length,
        yield_strain: q.yield_strain,
        creep: q.creep,
        max_strain: q.max_strain,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothPlasticRestLengthResult`].
fn decode_result(raw: &GpuResult) -> ClothPlasticRestLengthResult {
    ClothPlasticRestLengthResult {
        new_rest: raw.new_rest,
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

/// A compiled, reusable plastic rest-length creep compute pipeline, twinning the
/// `CPU` golden `plastic_rest_length` composed with `PlasticParams::sanitized`.
pub struct GpuClothPlasticRestLength {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothPlasticRestLength {
    /// Compiles the plastic rest-length creep kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothPlasticRestLength {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length"),
            source: ShaderSource::Wgsl(CLOTH_PLASTIC_REST_LENGTH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothPlasticRestLength {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothPlasticRestLengthResult`] per input, in order.
    ///
    /// The continuous `new_rest` channel matches the reference to within the
    /// tolerance documented on this module; the `valid` flag matches exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothPlasticRestLengthQuery],
    ) -> Vec<ClothPlasticRestLengthResult> {
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
            label: Some("prism_volumetric_cloth_plastic_rest_length_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_bind_group"),
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
            label: Some("prism_volumetric_cloth_plastic_rest_length_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_plastic_rest_length_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_plastic_rest_length_pass"),
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
