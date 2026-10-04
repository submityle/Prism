//! `wgpu` compute twin of the generalized Hoek–Brown yield-surface closed
//! forms, from the `CPU` golden
//! `prism_physics_core::collider::tet_fem_hoek_brown_plasticity`'s
//! `HoekBrownModel` bracket/yield helpers.
//!
//! The rock-mechanics Hoek–Brown criterion bounds shear strength by a *curved*
//! surface in the `(σ₁, σ₃)` principal-stress plane (tension positive). This
//! module ports the stateless per-point helpers the return map leans on — the
//! confinement bracket, the yield value, its major-stress derivative, the
//! dilation bracket and the plastic-potential slope — onto the device, one
//! thread per query. A passing real-device parity test is direct evidence the
//! ported kernel evaluates the same surface the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, for one `(σ_ci, m_b, s, a, m_g)` model
//! and one `(σ₁, σ₃)` stress pair:
//!
//! * `raw_bracket = s − m_b · σ₁ / σ_ci` (the un-clamped confinement bracket).
//! * `bracket = max(raw_bracket, MIN_BRACKET)` with `MIN_BRACKET = 1e-9`.
//! * `yield_value = (σ₁ − σ₃) − σ_ci · bracket^a`.
//! * `dyield_dmajor = 1 + a · m_b · bracket^(a−1)`.
//! * `bracket_g = max(s − m_g · σ₁ / σ_ci, MIN_BRACKET)`.
//! * `flow_major = a · m_g · bracket_g^(a−1)`.
//!
//! A query is `valid` only when the model lies in the golden `HoekBrownModel`
//! range: `σ_ci` finite and `> 0`, `m_b` finite and `> 0`, `s` finite with
//! `0 < s ≤ 1`, `a` finite with `0 < a ≤ 1`, and `m_g` finite with
//! `0 ≤ m_g ≤ m_b`. An invalid model yields all-zero outputs with `valid = 0`.
//!
//! # Correctness model
//!
//! The golden evaluates the fractional powers in `f64`; `WGSL` has no `f64`, so
//! the kernel uses `f32` `pow`. The host oracle mirrors the golden exactly in
//! `f64` and casts to `f32`, and every continuous output is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) which absorbs the
//! single-precision power evaluation. The discrete `valid` flag is compared
//! exactly; the parity sweep keeps the model comfortably inside its valid range
//! so the validity decision cannot be flipped by round-off, and keeps
//! `raw_bracket` well above `MIN_BRACKET` so the clamp knee and the
//! `bracket^(a−1)` divergence as `bracket → 0` are avoided.
//!
//! # Degenerate inputs
//!
//! An out-of-range or non-finite model is a no-op: `valid = 0` and all six
//! continuous outputs are `0`. The division by `σ_ci` is fed through a `select`
//! guard so the un-taken (invalid) branch never divides by zero, and the
//! bracket is floored at `MIN_BRACKET` so the fractional power stays finite at
//! and past the tensile apex. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `max`, `pow`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `f64`, no `sin`,
//! `cos`, `tan`, `exp`, `log`, no `round`, no `f32` remainder and no `sqrt`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, and the range checks are ordered `>`/`<=`
//! compares; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_hoek_brown_plasticity`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Hoek–Brown yield-surface kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `HoekBrownModel` bracket/yield helpers; see the
/// module documentation for the closed forms.
const HOEK_BROWN_YIELD_SURFACE_WGSL: &str = r#"
// Hoek–Brown yield-surface twin: one thread per query reproduces the model's
// bracket, yield value, major derivative, dilation bracket and flow slope. It
// uses only the portable core-WGSL subset (abs, max, pow, + - * /, select plus
// unsigned index math), has no loop and no branch, so it provably terminates.
// Finiteness is an ordered abs < 3.0e38 compare (rejecting infinities and NaN)
// and the range checks are ordered > / <= compares, both fed to select; there
// is no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Intact uniaxial compressive strength sigma_ci (> 0).
    sigma_ci: f32,
    // Reduced broken-mass constant m_b (> 0).
    m_b: f32,
    // Rock-mass constant s in (0, 1].
    s: f32,
    // Curvature exponent a in (0, 1].
    a: f32,
    // Dilation constant m_g in [0, m_b].
    m_g: f32,
    // Major principal stress sigma1 (tension positive).
    sigma1: f32,
    // Minor principal stress sigma3 (tension positive).
    sigma3: f32,
    // Padding word to a 32-byte stride.
    pad0: f32,
}

struct Result {
    // Clamped confinement bracket max(raw_bracket, MIN_BRACKET).
    bracket: f32,
    // Un-clamped confinement bracket s - m_b*sigma1/sigma_ci.
    raw_bracket: f32,
    // Yield value (sigma1 - sigma3) - sigma_ci * bracket^a.
    yield_value: f32,
    // Yield derivative w.r.t. major stress 1 + a*m_b*bracket^(a-1).
    dyield_dmajor: f32,
    // Clamped dilation bracket max(s - m_g*sigma1/sigma_ci, MIN_BRACKET).
    bracket_g: f32,
    // Plastic-potential slope a*m_g*bracket_g^(a-1).
    flow_major: f32,
    // 1 when the model is in the valid Hoek–Brown range, else 0.
    valid: u32,
    // Padding word to a 32-byte stride.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const MIN_BRACKET: f32 = 1e-9;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let sci = q.sigma_ci;
    let mb = q.m_b;
    let s = q.s;
    let a = q.a;
    let mg = q.m_g;
    let s1 = q.sigma1;
    let s3 = q.sigma3;

    // Validity mirrors the golden HoekBrownModel::new range check. Finiteness
    // via ordered abs < 3.0e38 (rejects +/-inf and NaN), plus the parameter
    // ranges as ordered > / <= compares. No bare f32 equality anywhere.
    let finite = (abs(sci) < FINITE_LIMIT)
        && (abs(mb) < FINITE_LIMIT)
        && (abs(s) < FINITE_LIMIT)
        && (abs(a) < FINITE_LIMIT)
        && (abs(mg) < FINITE_LIMIT);
    let in_range = (sci > 0.0)
        && (mb > 0.0)
        && (s > 0.0) && (s <= 1.0)
        && (a > 0.0) && (a <= 1.0)
        && (mg >= 0.0) && (mg <= mb);
    let ok = finite && in_range;

    // Guard the sigma_ci divisor so the un-taken (invalid) branch never divides
    // by zero; when ok sigma_ci is strictly positive.
    let denom = select(1.0, sci, ok);

    let raw = s - mb * s1 / denom;
    let bracket = max(raw, MIN_BRACKET);
    let raw_g = s - mg * s1 / denom;
    let bracket_g = max(raw_g, MIN_BRACKET);

    // Fractional powers: double precision in the golden, single here; tolerance and
    // the clamp floor (bracket >= MIN_BRACKET > 0) keep them finite.
    let yv = (s1 - s3) - sci * pow(bracket, a);
    let dyv = 1.0 + a * mb * pow(bracket, a - 1.0);
    let fm = a * mg * pow(bracket_g, a - 1.0);

    var out: Result;
    out.bracket = select(0.0, bracket, ok);
    out.raw_bracket = select(0.0, raw, ok);
    out.yield_value = select(0.0, yv, ok);
    out.dyield_dmajor = select(0.0, dyv, ok);
    out.bracket_g = select(0.0, bracket_g, ok);
    out.flow_major = select(0.0, fm, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
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
/// five model parameters and the two principal stresses, padded to `8` `f32`
/// words (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sigma_ci: f32,
    m_b: f32,
    s: f32,
    a: f32,
    m_g: f32,
    sigma1: f32,
    sigma3: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: six continuous outputs, the validity flag and a padding word — `8`
/// words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    bracket: f32,
    raw_bracket: f32,
    yield_value: f32,
    dyield_dmajor: f32,
    bracket_g: f32,
    flow_major: f32,
    valid: u32,
    pad0: u32,
}

/// One Hoek–Brown yield-surface query: the five model parameters and the two
/// principal stresses (tension positive).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownYieldSurfaceQuery {
    /// Intact uniaxial compressive strength `σ_ci` (`> 0`).
    pub sigma_ci: f32,
    /// Reduced broken-mass constant `m_b` (`> 0`).
    pub m_b: f32,
    /// Rock-mass constant `s ∈ (0, 1]`.
    pub s: f32,
    /// Curvature exponent `a ∈ (0, 1]`.
    pub a: f32,
    /// Dilation constant `m_g ∈ [0, m_b]`.
    pub m_g: f32,
    /// Major principal stress `σ₁` (tension positive).
    pub sigma1: f32,
    /// Minor principal stress `σ₃` (tension positive).
    pub sigma3: f32,
}

impl HoekBrownYieldSurfaceQuery {
    /// Builds a query from the five model parameters and the two principal
    /// stresses.
    #[must_use]
    pub fn new(
        sigma_ci: f32,
        m_b: f32,
        s: f32,
        a: f32,
        m_g: f32,
        sigma1: f32,
        sigma3: f32,
    ) -> HoekBrownYieldSurfaceQuery {
        HoekBrownYieldSurfaceQuery {
            sigma_ci,
            m_b,
            s,
            a,
            m_g,
            sigma1,
            sigma3,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `HoekBrownModel` bracket/yield helpers for that model and stress pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownYieldSurfaceResult {
    /// Clamped confinement bracket `max(raw_bracket, 1e-9)`, or `0` if invalid.
    pub bracket: f32,
    /// Un-clamped confinement bracket `s − m_b·σ₁/σ_ci`, or `0` if invalid.
    pub raw_bracket: f32,
    /// Yield value `(σ₁ − σ₃) − σ_ci·bracket^a`, or `0` if invalid.
    pub yield_value: f32,
    /// Yield derivative `1 + a·m_b·bracket^(a−1)`, or `0` if invalid.
    pub dyield_dmajor: f32,
    /// Clamped dilation bracket `max(s − m_g·σ₁/σ_ci, 1e-9)`, or `0` if invalid.
    pub bracket_g: f32,
    /// Plastic-potential slope `a·m_g·bracket_g^(a−1)`, or `0` if invalid.
    pub flow_major: f32,
    /// `1` when the model is in the valid Hoek–Brown range, else `0`.
    pub valid: u32,
}

/// Encodes one [`HoekBrownYieldSurfaceQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &HoekBrownYieldSurfaceQuery) -> GpuQuery {
    GpuQuery {
        sigma_ci: q.sigma_ci,
        m_b: q.m_b,
        s: q.s,
        a: q.a,
        m_g: q.m_g,
        sigma1: q.sigma1,
        sigma3: q.sigma3,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HoekBrownYieldSurfaceResult`].
fn decode_result(raw: &GpuResult) -> HoekBrownYieldSurfaceResult {
    HoekBrownYieldSurfaceResult {
        bracket: raw.bracket,
        raw_bracket: raw.raw_bracket,
        yield_value: raw.yield_value,
        dyield_dmajor: raw.dyield_dmajor,
        bracket_g: raw.bracket_g,
        flow_major: raw.flow_major,
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

/// A compiled, reusable Hoek–Brown yield-surface compute pipeline, twinning the
/// `CPU` golden `HoekBrownModel` bracket/yield helpers.
pub struct GpuHoekBrownYieldSurface {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHoekBrownYieldSurface {
    /// Compiles the Hoek–Brown yield-surface kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHoekBrownYieldSurface {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface"),
            source: ShaderSource::Wgsl(HOEK_BROWN_YIELD_SURFACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHoekBrownYieldSurface {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HoekBrownYieldSurfaceResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and every continuous
    /// output to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HoekBrownYieldSurfaceQuery],
    ) -> Vec<HoekBrownYieldSurfaceResult> {
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
            label: Some("prism_volumetric_hoek_brown_yield_surface_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_bind_group"),
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
            label: Some("prism_volumetric_hoek_brown_yield_surface_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hoek_brown_yield_surface_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hoek_brown_yield_surface_pass"),
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
