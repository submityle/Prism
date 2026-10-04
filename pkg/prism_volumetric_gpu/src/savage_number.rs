//! `wgpu` compute twin of the Savage-number closed form, from the `CPU` golden
//! `prism_physics_core::collider::savage_number`'s `SavageNumber::from_state`.
//!
//! The Savage (`SavageNumber`) number `N_sav = ρ_s · d² · γ̇² / P` is the
//! dimensionless ratio of collisional (inertial) grain stress to the
//! quasi-static confining pressure, used to classify granular flow into a
//! frictional/quasi-static regime (`FrictionalQuasiStatic`) versus a
//! collisional/grain-inertia regime (`Collisional`) about the
//! Savage-Hutter critical value `0.1`. This module ports that single
//! stateless closed form onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same Savage number and regime the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `from_state` plus `regime` for one
//! granular state with explicit `solid_density`, `grain_diameter`,
//! `shear_rate` and `normal_stress`:
//!
//! * If any input is non-finite, or `ρ_s <= 0`, `d <= 0`, `γ̇ < 0` or
//!   `P <= 0`, the state is invalid (`valid = 0`, `savage_number = 0`,
//!   `regime = 0`).
//! * Otherwise `savage_number = ρ_s · d · d · γ̇ · γ̇ / P`, evaluated in
//!   exactly the golden operator order; a non-finite result (overflow) is also
//!   invalid.
//! * The regime is `FrictionalQuasiStatic` (encoded `0`) when
//!   `savage_number < 0.1`, else `Collisional` (encoded `1`); the threshold is
//!   half-open so exactly `0.1` is `Collisional`.
//!
//! # Correctness model
//!
//! The continuous arithmetic (four multiplies and one division) threads through
//! operators that a `GPU` may contract, so `CPU` and `GPU` are not necessarily
//! bit-exact; the valid `savage_number` scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `regime` and `valid` flags are compared exactly; the parity test keeps the
//! random sweep away from the `N_sav ≈ 0.1` regime knee so round-off cannot
//! flip the classification.
//!
//! # Degenerate inputs
//!
//! A non-finite input or an out-of-range value yields `valid = 0` with the
//! other outputs zeroed. The divisor `P` is fed through a `select` guard so the
//! un-taken (invalid) branch never divides by zero. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, and the regime split with an ordered
//! `>= 0.1`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::savage_number`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Savage-number kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `SavageNumber::from_state` plus `regime`; see the module
/// documentation for the closed form.
const SAVAGE_NUMBER_WGSL: &str = r#"
// Savage-number twin: one thread per query reproduces from_state + regime. It
// uses only the portable core-WGSL subset (abs, + - * /, select plus unsigned
// index math), takes no optional feature, and has no loop and no branch beyond
// the bounds guard, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN), range checks are ordered
// compares, and the regime split is an ordered >= 0.1; all fed to select.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Solid grain density rho_s.
    solid_density: f32,
    // Grain diameter d.
    grain_diameter: f32,
    // Shear rate gamma_dot.
    shear_rate: f32,
    // Confining normal stress P.
    normal_stress: f32,
}

struct Result {
    // Savage number rho_s*d*d*gamma*gamma/P when valid, else 0.
    savage_number: f32,
    // Regime code: 0 = FrictionalQuasiStatic, 1 = Collisional.
    regime: u32,
    // 1 when inputs are finite and in range and N_sav is finite, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const SAVAGE_CRITICAL: f32 = 0.1;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let rho = q.solid_density;
    let d = q.grain_diameter;
    let gamma = q.shear_rate;
    let p = q.normal_stress;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let finite = (abs(rho) < FINITE_LIMIT)
        && (abs(d) < FINITE_LIMIT)
        && (abs(gamma) < FINITE_LIMIT)
        && (abs(p) < FINITE_LIMIT);
    // Golden range gates: rho_s > 0, d > 0, gamma >= 0, P > 0.
    let ranges = (rho > 0.0) && (d > 0.0) && (gamma >= 0.0) && (p > 0.0);

    // Guard the divisor so the un-taken (invalid) branch never divides by zero;
    // when ranges hold P is strictly positive.
    let denom = select(1.0, p, ranges);
    // Golden operator order: rho * d * d * gamma * gamma, then divide by P.
    let n_sav = rho * d * d * gamma * gamma / denom;
    let n_finite = abs(n_sav) < FINITE_LIMIT;
    let ok = finite && ranges && n_finite;

    var out: Result;
    out.savage_number = select(0.0, n_sav, ok);
    out.regime = select(0u, 1u, ok && (n_sav >= SAVAGE_CRITICAL));
    out.valid = select(0u, 1u, ok);
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
/// four `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    solid_density: f32,
    grain_diameter: f32,
    shear_rate: f32,
    normal_stress: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the Savage number, the regime code and the validity flag — `3`
/// words (`12` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    savage_number: f32,
    regime: u32,
    valid: u32,
}

/// One Savage-number query: the granular state scalars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavageNumberQuery {
    /// Solid grain density `ρ_s`.
    pub solid_density: f32,
    /// Grain diameter `d`.
    pub grain_diameter: f32,
    /// Shear rate `γ̇`.
    pub shear_rate: f32,
    /// Confining normal stress `P`.
    pub normal_stress: f32,
}

impl SavageNumberQuery {
    /// Builds a query from the solid density, grain diameter, shear rate and
    /// confining normal stress.
    #[must_use]
    pub fn new(
        solid_density: f32,
        grain_diameter: f32,
        shear_rate: f32,
        normal_stress: f32,
    ) -> SavageNumberQuery {
        SavageNumberQuery {
            solid_density,
            grain_diameter,
            shear_rate,
            normal_stress,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `SavageNumber::from_state` plus `regime` output for that granular state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavageNumberResult {
    /// The Savage number `ρ_s d² γ̇² / P` when valid, else `0`.
    pub savage_number: f32,
    /// Regime code: `0` = `FrictionalQuasiStatic`, `1` = `Collisional`.
    pub regime: u32,
    /// `1` when inputs are finite and in range and the result is finite, else
    /// `0`.
    pub valid: u32,
}

/// Encodes one [`SavageNumberQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SavageNumberQuery) -> GpuQuery {
    GpuQuery {
        solid_density: q.solid_density,
        grain_diameter: q.grain_diameter,
        shear_rate: q.shear_rate,
        normal_stress: q.normal_stress,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SavageNumberResult`].
fn decode_result(raw: &GpuResult) -> SavageNumberResult {
    SavageNumberResult {
        savage_number: raw.savage_number,
        regime: raw.regime,
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

/// A compiled, reusable Savage-number compute pipeline, twinning the `CPU`
/// golden `SavageNumber::from_state`.
pub struct GpuSavageNumber {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSavageNumber {
    /// Compiles the Savage-number kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSavageNumber {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_savage_number"),
            source: ShaderSource::Wgsl(SAVAGE_NUMBER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_savage_number_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_savage_number_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_savage_number_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSavageNumber {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SavageNumberResult`]
    /// per input, in order.
    ///
    /// The `regime` and `valid` flags match the reference exactly and the
    /// `savage_number` scalar to the module's tolerance. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SavageNumberQuery],
    ) -> Vec<SavageNumberResult> {
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
            label: Some("prism_volumetric_savage_number_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_savage_number_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_savage_number_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_savage_number_bind_group"),
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
            label: Some("prism_volumetric_savage_number_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_savage_number_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_savage_number_pass"),
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
