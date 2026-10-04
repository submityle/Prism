//! `wgpu` compute twin of the stress-invariant / Lode decomposition closed
//! form, from the `CPU` golden
//! `prism_physics_core::collider::stress_invariants`'s
//! `StressInvariants::from_principal_stresses` and its derived getters.
//!
//! A triple of principal stresses is decomposed into the standard
//! soil-mechanics / plasticity state descriptors in `p`–`q`–`theta` space: the
//! mean (hydrostatic) stress `p`, the second deviatoric invariant `J2`, the
//! third deviatoric invariant `J3`, the von Mises equivalent stress `q`, the
//! octahedral shear stress `tau_oct`, the deviatoric Frobenius norm, and the
//! shape descriptors — the Lode–Nadai parameter `mu_L` and the Lode angle
//! `theta`. One thread resolves one query, so a passing real-device parity test
//! is direct evidence the ported kernel computes the same invariants the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `from_principal_stresses` followed by
//! every getter for one principal-stress triple:
//!
//! * If any component is non-finite, the triple is invalid (`valid = 0`) and
//!   every output is `0`.
//! * Otherwise the three stresses are sorted descending `sigma1 >= sigma2 >=
//!   sigma3`, then `p = (sigma1 + sigma2 + sigma3) / 3`, the deviatoric
//!   stresses `s_i = sigma_i - p`, `J2 = (1/2)(s1^2 + s2^2 + s3^2)`,
//!   `J3 = s1 s2 s3`, `q = sqrt(3 J2)`, `tau_oct = sqrt(2 J2 / 3)` and the
//!   deviatoric norm `sqrt(2 J2)`.
//! * The Lode parameter `mu_L = (2 sigma2 - sigma1 - sigma3) / (sigma1 -
//!   sigma3)` and Lode angle `theta = atan(mu_L / sqrt(3))` are only defined
//!   away from the hydrostatic axis: when the span `sigma1 - sigma3 <= 1e-9`
//!   they are undefined, flagged by `lode_valid = 0` with both outputs `0`.
//!
//! # Correctness model
//!
//! The golden evaluates the Lode angle in `f64` (`atan` of an `f64` quotient)
//! and casts to `f32`; the kernel uses the portable `f32` `atan` builtin, so
//! the two sides are not bit-exact. Each continuous output is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `valid` and `lode_valid` flags are compared exactly. The parity sweep keeps
//! the span `sigma1 - sigma3` well clear of the `1e-9` hydrostatic knee so the
//! `lode_valid` decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite component yields `valid = 0` with all outputs `0`. A
//! hydrostatic (or near-hydrostatic) state keeps `valid = 1` but reports
//! `lode_valid = 0` and `0` for the Lode parameter and angle. The Lode divisor
//! is fed through a `select` guard so the un-taken branch never divides by
//! zero. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `atan`, `+ - * /`, `select` and unsigned index arithmetic — with no
//! `u64`/`i64`/`f64`, no `round`, no `f32` remainder, and no bare `f32`
//! equality, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! descending sort of three is an explicit `min`/`max` compare-exchange with no
//! loop; finiteness is the ordered compare `abs(x) < 3.0e38` (which rejects
//! both infinities and `NaN`).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::stress_invariants`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` stress-invariant kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `StressInvariants`; see the module documentation for the
/// closed forms.
const STRESS_INVARIANTS_WGSL: &str = r#"
// Stress-invariant twin: one thread per query reproduces the full
// StressInvariants decomposition. It uses only the portable core-WGSL subset
// (abs, min, max, sqrt, atan, + - * /, select plus unsigned index math) and
// has no loop and no branch beyond the bounds guard, so it provably
// terminates. The descending sort of three is an explicit min/max
// compare-exchange; finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) fed to select, never a bare equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The three principal stresses in arbitrary order.
    s0: f32,
    s1: f32,
    s2: f32,
    // Padding word to a 16-byte-friendly stride.
    pad0: f32,
}

struct Invariants {
    // Mean (hydrostatic) stress p = (s1 + s2 + s3) / 3.
    mean_stress: f32,
    // Second deviatoric invariant J2.
    j2: f32,
    // Third deviatoric invariant J3.
    j3: f32,
    // von Mises equivalent stress q = sqrt(3 J2).
    von_mises: f32,
    // Octahedral shear stress sqrt(2 J2 / 3).
    octahedral_shear: f32,
    // Deviatoric Frobenius norm sqrt(2 J2).
    deviatoric_norm: f32,
    // Lode-Nadai parameter, 0 when the state is hydrostatic.
    lode_parameter: f32,
    // Lode angle in radians, 0 when the state is hydrostatic.
    lode_angle: f32,
    // 1 when the Lode descriptors are defined (non-hydrostatic), else 0.
    lode_valid: u32,
    // 1 when every input component is finite, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Invariants>;

const FINITE_LIMIT: f32 = 3.0e38;
const HYDROSTATIC_EPS: f32 = 1.0e-9;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let x0 = q.s0;
    let x1 = q.s1;
    let x2 = q.s2;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let ok = (abs(x0) < FINITE_LIMIT) && (abs(x1) < FINITE_LIMIT) && (abs(x2) < FINITE_LIMIT);

    // Descending sort of three via min/max compare-exchange: a >= b >= c.
    let hi0 = max(x0, x1);
    let lo0 = min(x0, x1);
    let a = max(hi0, x2);
    let t = min(hi0, x2);
    let b = max(lo0, t);
    let c = min(lo0, t);

    let p = (a + b + c) / 3.0;
    let d0 = a - p;
    let d1 = b - p;
    let d2 = c - p;
    let j2 = 0.5 * (d0 * d0 + d1 * d1 + d2 * d2);
    let j3 = d0 * d1 * d2;
    // j2 >= 0, so these roots are well defined.
    let von_mises = sqrt(3.0 * j2);
    let octahedral_shear = sqrt(2.0 * j2 / 3.0);
    let deviatoric_norm = sqrt(2.0 * j2);

    // Lode descriptors: the hydrostatic span a - c <= eps is undefined. Guard
    // the divisor so the un-taken branch never divides by zero.
    let span = a - c;
    let lode_ok = ok && (span > HYDROSTATIC_EPS);
    let denom = select(1.0, span, lode_ok);
    let mu = (2.0 * b - a - c) / denom;
    let theta = atan(mu / sqrt(3.0));

    var out: Invariants;
    out.mean_stress = select(0.0, p, ok);
    out.j2 = select(0.0, j2, ok);
    out.j3 = select(0.0, j3, ok);
    out.von_mises = select(0.0, von_mises, ok);
    out.octahedral_shear = select(0.0, octahedral_shear, ok);
    out.deviatoric_norm = select(0.0, deviatoric_norm, ok);
    out.lode_parameter = select(0.0, mu, lode_ok);
    out.lode_angle = select(0.0, theta, lode_ok);
    out.lode_valid = select(0u, 1u, lode_ok);
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
/// The three principal stresses are padded to `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    s0: f32,
    s1: f32,
    s2: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Invariants`
/// struct: eight continuous `f32` outputs followed by the two `u32` flags —
/// `10` words (`40` bytes), all `4`-byte aligned with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    mean_stress: f32,
    j2: f32,
    j3: f32,
    von_mises: f32,
    octahedral_shear: f32,
    deviatoric_norm: f32,
    lode_parameter: f32,
    lode_angle: f32,
    lode_valid: u32,
    valid: u32,
}

/// One stress-invariant query: the three principal stresses in arbitrary order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressInvariantsQuery {
    /// The three principal stresses; the kernel sorts them descending.
    pub principal: [f32; 3],
}

impl StressInvariantsQuery {
    /// Builds a query from a principal-stress triple.
    #[must_use]
    pub fn new(principal: [f32; 3]) -> StressInvariantsQuery {
        StressInvariantsQuery { principal }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `StressInvariants` getters for that principal-stress triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressInvariantsResult {
    /// Mean (hydrostatic) stress `p`.
    pub mean_stress: f32,
    /// Second deviatoric invariant `J2`.
    pub j2: f32,
    /// Third deviatoric invariant `J3`.
    pub j3: f32,
    /// von Mises equivalent stress `q = sqrt(3 J2)`.
    pub von_mises: f32,
    /// Octahedral shear stress `sqrt(2 J2 / 3)`.
    pub octahedral_shear: f32,
    /// Deviatoric Frobenius norm `sqrt(2 J2)`.
    pub deviatoric_norm: f32,
    /// Lode-Nadai parameter, `0` when hydrostatic.
    pub lode_parameter: f32,
    /// Lode angle in radians, `0` when hydrostatic.
    pub lode_angle: f32,
    /// `1` when the Lode descriptors are defined (non-hydrostatic), else `0`.
    pub lode_valid: u32,
    /// `1` when every input component is finite, else `0`.
    pub valid: u32,
}

/// Encodes one [`StressInvariantsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &StressInvariantsQuery) -> GpuQuery {
    GpuQuery {
        s0: q.principal[0],
        s1: q.principal[1],
        s2: q.principal[2],
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`StressInvariantsResult`].
fn decode_result(raw: &GpuResult) -> StressInvariantsResult {
    StressInvariantsResult {
        mean_stress: raw.mean_stress,
        j2: raw.j2,
        j3: raw.j3,
        von_mises: raw.von_mises,
        octahedral_shear: raw.octahedral_shear,
        deviatoric_norm: raw.deviatoric_norm,
        lode_parameter: raw.lode_parameter,
        lode_angle: raw.lode_angle,
        lode_valid: raw.lode_valid,
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

/// A compiled, reusable stress-invariant compute pipeline, twinning the `CPU`
/// golden `StressInvariants`.
pub struct GpuStressInvariants {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStressInvariants {
    /// Compiles the stress-invariant kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStressInvariants {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_stress_invariants"),
            source: ShaderSource::Wgsl(STRESS_INVARIANTS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_stress_invariants_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_stress_invariants_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_stress_invariants_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStressInvariants {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`StressInvariantsResult`] per input, in order.
    ///
    /// The `valid` and `lode_valid` flags match the reference exactly and the
    /// continuous scalars to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[StressInvariantsQuery],
    ) -> Vec<StressInvariantsResult> {
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
            label: Some("prism_volumetric_stress_invariants_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_stress_invariants_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_stress_invariants_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_stress_invariants_bind_group"),
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
            label: Some("prism_volumetric_stress_invariants_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_stress_invariants_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_stress_invariants_pass"),
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
