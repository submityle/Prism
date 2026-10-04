//! `wgpu` compute twin of the Stomakhin snow hardening scale of the `CPU`
//! golden `prism_physics_core::collider::tet_fem_snow_plasticity::hardened_lame`.
//!
//! Compacted snow stiffens: both Lamé parameters are multiplied by the
//! hardening factor `exp(ξ · (1 − J_p))`, where `ξ ≥ 0` is the model hardening
//! coefficient and `J_p = det Fₚ` is the plastic volume ratio. The exponent is
//! clamped to `[-40, 40]` so an extreme (near-singular) plastic gradient cannot
//! overflow the exponential.
//!
//! This module ports that single stateless derivation onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same scaled parameters the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `hardened_lame` with `J_p` supplied
//! directly as a scalar (the host computes the determinant upstream):
//!
//! * If any of `ξ`, `J_p`, `base_mu`, `base_lambda` is non-finite, the result
//!   is invalid (`valid = 0`, all three outputs `0`).
//! * Otherwise `exponent = ξ · (1 − J_p)`,
//!   `factor = exp(clamp(exponent, -40, 40))`, `mu = base_mu · factor` and
//!   `lambda = base_lambda · factor`.
//!
//! # Correctness model
//!
//! The golden computes the exponent in `f64` and takes `exp` in `f64` before
//! narrowing the factor to `f32`; the kernel evaluates the same closed form in
//! `f32` with the `WGSL` built-in `exp`. The two therefore agree on the same
//! function but need not be bit-exact, so each continuous output is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The
//! discrete `valid` flag is compared exactly; the parity test keeps inputs
//! finite and well away from the overflow knees so the two sides agree.
//!
//! # Degenerate inputs
//!
//! A non-finite `ξ`, `J_p`, `base_mu` or `base_lambda` yields `valid = 0` with
//! all outputs `0`. The exponent clamp bounds the exponential so the valid
//! branch never overflows; the invalid branch is discarded through a `select`
//! so a `NaN`/`Inf` never leaks into an output. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `exp`, `+ - *`, `select` and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested
//! with the ordered compare `abs(x) < 3.0e38` (which rejects both infinities
//! and `NaN`) rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_snow_plasticity`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` snow hardening kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `hardened_lame`; see the module documentation for the closed
/// form.
const SNOW_HARDENED_LAME_WGSL: &str = r#"
// Snow hardening twin: one thread per query reproduces the Stomakhin hardening
// scale of hardened_lame. It uses only the portable core-WGSL subset (abs,
// clamp, exp, + - *, select plus unsigned index math), takes no optional
// feature, and has no loop and no branch, so it provably terminates. Finiteness
// is an ordered abs < 3.0e38 compare (rejecting infinities and NaN) fed to
// select; there is no f32 equality.

struct Params {
    // Number of queries in the input and output buffers.
    count: u32,
    // Padding to a 16-byte uniform block.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Model hardening coefficient xi (>= 0).
    hardening: f32,
    // Plastic volume ratio J_p = det(Fp).
    plastic_volume_ratio: f32,
    // Base first Lame parameter mu.
    base_mu: f32,
    // Base second Lame parameter lambda.
    base_lambda: f32,
}

struct Result {
    // Scaled first Lame parameter base_mu * factor when valid, else 0.
    mu: f32,
    // Scaled second Lame parameter base_lambda * factor when valid, else 0.
    lambda: f32,
    // The hardening multiplier exp(clamp(exponent, -40, 40)) when valid, else 0.
    factor: f32,
    // 1 when every input is finite, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let xi = q.hardening;
    let jp = q.plastic_volume_ratio;
    let base_mu = q.base_mu;
    let base_lambda = q.base_lambda;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let finite = (abs(xi) < FINITE_LIMIT)
        && (abs(jp) < FINITE_LIMIT)
        && (abs(base_mu) < FINITE_LIMIT)
        && (abs(base_lambda) < FINITE_LIMIT);

    // Golden operator order: exponent = xi * (1 - J_p), clamped before exp.
    let one_minus = 1.0 - jp;
    let exponent = xi * one_minus;
    let factor_raw = exp(clamp(exponent, -40.0, 40.0));

    var out: Result;
    out.factor = select(0.0, factor_raw, finite);
    out.mu = select(0.0, base_mu * factor_raw, finite);
    out.lambda = select(0.0, base_lambda * factor_raw, finite);
    out.valid = select(0u, 1u, finite);
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
/// the hardening coefficient, plastic volume ratio and the two base Lamé
/// parameters packed as `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    hardening: f32,
    plastic_volume_ratio: f32,
    base_mu: f32,
    base_lambda: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two scaled Lamé parameters, the hardening factor and the
/// validity flag — `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    mu: f32,
    lambda: f32,
    factor: f32,
    valid: u32,
}

/// One snow hardening query: the hardening coefficient, plastic volume ratio
/// and the two base Lamé parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowHardenedLameQuery {
    /// Model hardening coefficient `ξ` (expected non-negative).
    pub hardening: f32,
    /// Plastic volume ratio `J_p = det Fₚ`.
    pub plastic_volume_ratio: f32,
    /// Base first Lamé parameter `μ`.
    pub base_mu: f32,
    /// Base second Lamé parameter `λ`.
    pub base_lambda: f32,
}

impl SnowHardenedLameQuery {
    /// Builds a query from the hardening coefficient, plastic volume ratio and
    /// the two base Lamé parameters.
    #[must_use]
    pub fn new(
        hardening: f32,
        plastic_volume_ratio: f32,
        base_mu: f32,
        base_lambda: f32,
    ) -> SnowHardenedLameQuery {
        SnowHardenedLameQuery {
            hardening,
            plastic_volume_ratio,
            base_mu,
            base_lambda,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `hardened_lame` derivation for that parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowHardenedLameResult {
    /// The scaled first Lamé parameter `base_mu · factor` when valid, else `0`.
    pub mu: f32,
    /// The scaled second Lamé parameter `base_lambda · factor` when valid,
    /// else `0`.
    pub lambda: f32,
    /// The hardening multiplier `exp(clamp(ξ·(1−J_p), -40, 40))` when valid,
    /// else `0`.
    pub factor: f32,
    /// `true` when every input is finite, else `false`.
    pub valid: bool,
}

/// Encodes one [`SnowHardenedLameQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SnowHardenedLameQuery) -> GpuQuery {
    GpuQuery {
        hardening: q.hardening,
        plastic_volume_ratio: q.plastic_volume_ratio,
        base_mu: q.base_mu,
        base_lambda: q.base_lambda,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SnowHardenedLameResult`].
fn decode_result(raw: &GpuResult) -> SnowHardenedLameResult {
    SnowHardenedLameResult {
        mu: raw.mu,
        lambda: raw.lambda,
        factor: raw.factor,
        valid: raw.valid != 0,
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

/// A compiled, reusable snow hardening compute pipeline, twinning the `CPU`
/// golden `hardened_lame`.
pub struct GpuSnowHardenedLame {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSnowHardenedLame {
    /// Compiles the snow hardening kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSnowHardenedLame {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame"),
            source: ShaderSource::Wgsl(SNOW_HARDENED_LAME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSnowHardenedLame {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SnowHardenedLameResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each continuous
    /// output to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SnowHardenedLameQuery],
    ) -> Vec<SnowHardenedLameResult> {
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
            label: Some("prism_volumetric_snow_hardened_lame_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_bind_group"),
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
            label: Some("prism_volumetric_snow_hardened_lame_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_snow_hardened_lame_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_snow_hardened_lame_pass"),
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
