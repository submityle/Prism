//! `wgpu` compute twin of the Hertzian effective-modulus closed form, from the
//! `CPU` golden `prism_physics_core::collider::hertz_contact`'s `HertzModel::new`.
//!
//! For two elastic bodies of the same material with Young's modulus `E` and
//! Poisson ratio `ν`, the effective contact modulus is
//! `E* = E / (2 · (1 − ν²))`. This module ports that single stateless closed
//! form onto the device: one thread resolves one query, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same effective modulus the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the modulus validity gate and value of
//! `HertzModel::new` for one `(young_modulus, poisson_ratio)` pair:
//!
//! * The pair is valid only when `young_modulus` and `poisson_ratio` are both
//!   finite, `young_modulus > 0`, and `ν ∈ [0, 0.5)` (ordered `0 <= ν` and
//!   `ν < 0.5`), and the resulting `E*` is itself finite and `> 0`.
//! * When valid, `E* = young_modulus / (2 · (1 − ν²))`, evaluated in exactly
//!   the golden operator order (`ν·ν` first, then `1 − ν²`, then `2·(…)`, then
//!   the division). When invalid, `E* = 0`.
//!
//! # Correctness model
//!
//! The continuous arithmetic (two multiplies, one subtract and one division)
//! threads through operators that a `GPU` may contract, so `CPU` and `GPU` are
//! not necessarily bit-exact; the valid `effective_modulus` scalar is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The
//! discrete `valid` flag is compared exactly; the parity sweep keeps `ν` well
//! below `0.5` and `young_modulus` strictly positive so the validity decision
//! cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a non-positive `young_modulus`, or `ν` outside `[0, 0.5)`
//! yields `valid = 0` with `effective_modulus = 0`. When valid, `ν < 0.5` gives
//! `1 − ν² > 0.75`, so the divisor `2·(1 − ν²) > 1.5`; the kernel still feeds
//! the divisor through a `select` guard so the un-taken (invalid) branch never
//! evaluates a division by zero. An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and validity with ordered `> 0`, `>= 0` and
//! `< 0.5`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` effective-modulus kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `HertzModel::new`; see the module documentation for the
/// closed form.
const HERTZ_EFFECTIVE_MODULUS_WGSL: &str = r#"
// Effective-modulus twin: one thread per query reproduces the E* value and the
// validity gate of HertzModel::new. It uses only the portable core-WGSL subset
// (abs, + - * /, select plus unsigned index math), takes no optional feature,
// and has no loop and no branch, so it provably terminates. Finiteness is an
// ordered abs < 3.0e38 compare (rejecting infinities and NaN) and validity a
// set of ordered compares, all fed to select. No bare f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Young's modulus E of the shared material.
    young_modulus: f32,
    // Poisson ratio of the shared material.
    poisson_ratio: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Effective modulus E = young/(2*(1-nu*nu)) when valid, else 0.
    effective_modulus: f32,
    // 1 when the inputs and the result pass the validity gate, else 0.
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
    let young = q.young_modulus;
    let nu = q.poisson_ratio;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). Young strictly positive, nu in [0, 0.5).
    let finite = (abs(young) < FINITE_LIMIT) && (abs(nu) < FINITE_LIMIT);
    let base = finite && (young > 0.0) && (nu >= 0.0) && (nu < 0.5);

    // Guard the divisor so the un-taken (invalid) branch never divides by zero;
    // when base holds, nu < 0.5 gives 1 - nu*nu > 0.75 so denom > 1.5.
    let denom_raw = 2.0 * (1.0 - nu * nu);
    let denom = select(1.0, denom_raw, base);
    // Golden operator order: nu*nu, then 1 - nu*nu, then 2*(...), then divide.
    let e = young / denom;
    let ok = base && (abs(e) < FINITE_LIMIT) && (e > 0.0);

    var out: Result;
    out.effective_modulus = select(0.0, e, ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The two scalars are padded to `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    young_modulus: f32,
    poisson_ratio: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the effective modulus and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    effective_modulus: f32,
    valid: u32,
}

/// One effective-modulus query: the material Young's modulus and Poisson ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzEffectiveModulusQuery {
    /// Young's modulus `E` of the shared material.
    pub young_modulus: f32,
    /// Poisson ratio `ν` of the shared material.
    pub poisson_ratio: f32,
}

impl HertzEffectiveModulusQuery {
    /// Builds a query from the Young's modulus and Poisson ratio.
    #[must_use]
    pub fn new(young_modulus: f32, poisson_ratio: f32) -> HertzEffectiveModulusQuery {
        HertzEffectiveModulusQuery {
            young_modulus,
            poisson_ratio,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `HertzModel::new` effective-modulus output for that material pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzEffectiveModulusResult {
    /// The effective modulus `E = young / (2·(1 − ν²))` when valid, else `0`.
    pub effective_modulus: f32,
    /// `1` when the inputs and the result pass the validity gate, else `0`.
    pub valid: u32,
}

/// Encodes one [`HertzEffectiveModulusQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &HertzEffectiveModulusQuery) -> GpuQuery {
    GpuQuery {
        young_modulus: q.young_modulus,
        poisson_ratio: q.poisson_ratio,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HertzEffectiveModulusResult`].
fn decode_result(raw: &GpuResult) -> HertzEffectiveModulusResult {
    HertzEffectiveModulusResult {
        effective_modulus: raw.effective_modulus,
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

/// A compiled, reusable effective-modulus compute pipeline, twinning the `CPU`
/// golden `HertzModel::new`.
pub struct GpuHertzEffectiveModulus {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHertzEffectiveModulus {
    /// Compiles the effective-modulus kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHertzEffectiveModulus {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus"),
            source: ShaderSource::Wgsl(HERTZ_EFFECTIVE_MODULUS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHertzEffectiveModulus {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HertzEffectiveModulusResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the
    /// `effective_modulus` scalar to the module's tolerance. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HertzEffectiveModulusQuery],
    ) -> Vec<HertzEffectiveModulusResult> {
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
            label: Some("prism_volumetric_hertz_effective_modulus_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_bind_group"),
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
            label: Some("prism_volumetric_hertz_effective_modulus_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hertz_effective_modulus_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hertz_effective_modulus_pass"),
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
