//! `wgpu` compute twin of the capillary-bridge peak-adhesion closed form, from
//! the `CPU` golden
//! `prism_physics_core::collider::capillary_bridge`'s
//! `CapillaryBridgeModel::max_force`.
//!
//! When two wet grains touch, the pendular liquid bridge between them pulls
//! with its maximum magnitude `F₀ = 2π·R·γ·cos θ`, where `R` is the reduced
//! radius `2·rₐ·r_b / (rₐ + r_b)`, `γ` the liquid surface tension and `θ` the
//! solid–liquid contact angle. This module ports that single stateless closed
//! form onto the device: one thread resolves one query, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same peak force
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `max_force` for one radius pair with an
//! explicit surface tension and contact angle:
//!
//! * The pair is invalid (`valid = 0`, `force = 0`) unless every input is finite
//!   and `rₐ > 0`, `r_b > 0`, `γ > 0` and `θ ∈ [0, π/2]`. This folds together the
//!   `reduced_radius` gate (finite, strictly positive radii) and the
//!   `CapillaryBridgeModel` construction gate (`γ > 0`, `θ ∈ [0, π/2]`).
//! * Otherwise `R = 2·rₐ·r_b / (rₐ + r_b)` is formed in the golden operator
//!   order (`2 * a * b` first, then the division by `a + b`), and
//!   `force = 2π·R·γ·cos θ` with `π = std::f32::consts::PI`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through several multiplies, a division and
//! a cosine, so `CPU` and `GPU` are not necessarily bit-exact; moreover the
//! golden evaluates `cos θ` in `f64` and narrows to `f32`, while the kernel uses
//! the native `f32` `cos`, so the two agree only to a tolerance. The valid
//! `force` scalar is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity test keeps random inputs strictly inside the valid region so the
//! validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a non-positive radius or tension, or a contact angle
//! outside `[0, π/2]` yields `valid = 0` with `force = 0`. When both radii are
//! positive their sum is strictly positive, so the division is well defined; the
//! kernel still feeds the divisor through a `select` guard so the un-taken
//! (invalid) branch never evaluates a division by zero, and the final `force` is
//! `select`-masked to `0` so no `NaN` from an invalid cosine can escape. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `cos`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `sin`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and validity with ordered `> 0` and range
//! compares; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` peak-adhesion kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `CapillaryBridgeModel::max_force`; see the module documentation
/// for the closed form.
const CAPILLARY_MAX_FORCE_WGSL: &str = r#"
// Peak-adhesion twin: one thread per query reproduces max_force. It uses only
// the portable core-WGSL subset (abs, cos, + - * /, select plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN), validity ordered > 0 and range compares, both fed to
// select. The golden narrows an f64 cosine to f32; this kernel uses the native
// f32 cos, so the two agree to the parity tolerance.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Radius of the first sphere.
    radius_a: f32,
    // Radius of the second sphere.
    radius_b: f32,
    // Liquid surface tension gamma (N/m).
    surface_tension: f32,
    // Solid-liquid contact angle theta (radians).
    contact_angle: f32,
}

struct Result {
    // Peak adhesion 2*PI*R*gamma*cos(theta) when valid, else 0.
    force: f32,
    // 1 when all inputs are finite and in the valid region, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = PI / 2.0;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let ra = q.radius_a;
    let rb = q.radius_b;
    let gamma = q.surface_tension;
    let theta = q.contact_angle;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let finite = (abs(ra) < FINITE_LIMIT) && (abs(rb) < FINITE_LIMIT)
        && (abs(gamma) < FINITE_LIMIT) && (abs(theta) < FINITE_LIMIT);
    // reduced_radius gate (positive radii) plus model gate (positive tension,
    // contact angle in [0, PI/2]).
    let region = (ra > 0.0) && (rb > 0.0) && (gamma > 0.0)
        && (theta >= 0.0) && (theta <= HALF_PI);
    let ok = finite && region;

    // Guard the divisor so the un-taken (invalid) branch never divides by zero;
    // when ok the sum of two positive radii is strictly positive.
    let sum = ra + rb;
    let denom = select(1.0, sum, ok);
    // Golden operator order: 2 * a * b first, then divide by (a + b).
    let reduced = 2.0 * ra * rb / denom;
    let cos_theta = cos(theta);
    let f = 2.0 * PI * reduced * gamma * cos_theta;

    var out: Result;
    out.force = select(0.0, f, ok);
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
/// two radii, the surface tension and the contact angle — `4` `f32` words
/// (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    radius_a: f32,
    radius_b: f32,
    surface_tension: f32,
    contact_angle: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the peak force and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    force: f32,
    valid: u32,
}

/// One peak-adhesion query: the two sphere radii, the liquid surface tension
/// and the solid–liquid contact angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryMaxForceQuery {
    /// Radius of the first sphere.
    pub radius_a: f32,
    /// Radius of the second sphere.
    pub radius_b: f32,
    /// Liquid surface tension `γ` (N/m).
    pub surface_tension: f32,
    /// Solid–liquid contact angle `θ` (radians).
    pub contact_angle: f32,
}

impl CapillaryMaxForceQuery {
    /// Builds a query from the two radii, the surface tension and the contact
    /// angle.
    #[must_use]
    pub fn new(
        radius_a: f32,
        radius_b: f32,
        surface_tension: f32,
        contact_angle: f32,
    ) -> CapillaryMaxForceQuery {
        CapillaryMaxForceQuery {
            radius_a,
            radius_b,
            surface_tension,
            contact_angle,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CapillaryBridgeModel::max_force` output for that configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryMaxForceResult {
    /// The peak adhesion `2π·R·γ·cos θ` when valid, else `0`.
    pub force: f32,
    /// `1` when every input is finite and inside the valid region, else `0`.
    pub valid: u32,
}

/// Encodes one [`CapillaryMaxForceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CapillaryMaxForceQuery) -> GpuQuery {
    GpuQuery {
        radius_a: q.radius_a,
        radius_b: q.radius_b,
        surface_tension: q.surface_tension,
        contact_angle: q.contact_angle,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`CapillaryMaxForceResult`].
fn decode_result(raw: &GpuResult) -> CapillaryMaxForceResult {
    CapillaryMaxForceResult {
        force: raw.force,
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

/// A compiled, reusable peak-adhesion compute pipeline, twinning the `CPU`
/// golden `CapillaryBridgeModel::max_force`.
pub struct GpuCapillaryMaxForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapillaryMaxForce {
    /// Compiles the peak-adhesion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapillaryMaxForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capillary_max_force"),
            source: ShaderSource::Wgsl(CAPILLARY_MAX_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capillary_max_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capillary_max_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capillary_max_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapillaryMaxForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CapillaryMaxForceResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `force` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CapillaryMaxForceQuery],
    ) -> Vec<CapillaryMaxForceResult> {
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
            label: Some("prism_volumetric_capillary_max_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capillary_max_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capillary_max_force_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capillary_max_force_bind_group"),
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
            label: Some("prism_volumetric_capillary_max_force_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capillary_max_force_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capillary_max_force_pass"),
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
