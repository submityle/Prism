//! `wgpu` compute twin of the granular free-surface-flow Froude number, from
//! the `CPU` golden `prism_physics_core::collider::granular_froude`'s
//! `GranularFroude` constructors `from_flow` and `from_inclined_flow`.
//!
//! The Froude number compares inertial (advective) speed with the shallow
//! gravity-wave speed and is the central dimensionless number of free-surface
//! granular flow (chutes, avalanches, dam breaks):
//!
//! ```text
//! Fr = u / sqrt(g_eff * h)
//! ```
//!
//! where `u` is the depth-averaged flow speed, `h` the flowing-layer depth and
//! `g_eff` the effective normal gravity (`g` on a flat base, `g * cos(theta)`
//! on a chute inclined by `theta`). This module ports that single stateless
//! closed form onto the device: one thread resolves one query, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same Froude number the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces one of the two constructors, selected
//! by the `inclined` flag:
//!
//! * `inclined == 0` mirrors `from_flow`: `g_eff = g`, `theta` is ignored.
//! * `inclined != 0` mirrors `from_inclined_flow`: `theta` must be finite and
//!   `cos(theta) > 0`, and `g_eff = g * cos(theta)`.
//!
//! In both paths the shared `from_effective` validity rules apply: every input
//! must be finite, with `u >= 0`, `h > 0`, `g > 0` and `g_eff > 0`; the wave
//! speed `sqrt(g_eff * h)` must be finite and strictly positive; and the ratio
//! `Fr = u / wave_speed` must be finite. Any violation yields `valid = 0` with
//! `froude = 0` and `regime = 0`.
//!
//! The regime classification uses the half-open threshold `1.0`: `Fr < 1`
//! classifies as `Subcritical` (regime `0`) and `Fr >= 1` as `Supercritical`
//! (regime `1`), matching `FroudeRegime`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract,
//! and the inclined path evaluates `cos` natively in `f32` on the device while
//! the golden evaluates it in `f64` (`f64::from(theta).cos() as f32`); `CPU`
//! and `GPU` are therefore not bit-exact. The valid `froude` scalar is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`), which
//! absorbs the native-`cos` difference. The discrete `regime` and `valid` flags
//! are compared exactly; the parity sweep keeps `Fr` out of a narrow band
//! around the critical knee `Fr = 1` so the regime decision cannot flip under
//! round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a non-positive `h`, `g` or `g_eff`, a negative `u`, or
//! an inclined query with `cos(theta) <= 0` (i.e. `|theta| >= pi/2`) yields
//! `valid = 0` with cleared outputs. Every divisor and the radicand are fed
//! through a `select` guard, so the un-taken (invalid) branch never divides by
//! zero nor takes a square root of a negative value. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `cos`, `+ - * /`, `select` and unsigned index arithmetic — with no `sin`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no banned
//! wide types, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! Finiteness is tested with the ordered compare `abs(x) < 3.0e38` (which
//! rejects both infinities and `NaN`) rather than a bare `x == x`, and validity
//! with ordered `> 0`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_froude`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Froude-number kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `GranularFroude` constructors; see the module documentation for
/// the closed form.
const GRANULAR_FROUDE_WGSL: &str = r#"
// Granular Froude twin: one thread per query reproduces GranularFroude. It uses
// only the portable core-WGSL subset (abs, sqrt, cos, + - * /, select plus
// unsigned index math), has no loop and no data-dependent branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and validity ordered > 0 compares, all fed to select.
// Provenance: 孪生自本仓 prism_physics_core::collider::granular_froude。

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Depth-averaged flow speed u.
    speed: f32,
    // Flowing-layer depth h.
    flow_depth: f32,
    // Gravitational acceleration g.
    gravity: f32,
    // Chute inclination theta (radians); only read when inclined != 0.
    slope_angle: f32,
    // 0 selects the flat from_flow path; non-zero the inclined path.
    inclined: u32,
    // Padding words to an 8-word (32-byte) stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Froude number u / sqrt(g_eff * h) when valid, else 0.
    froude: f32,
    // 0 = Subcritical (Fr < 1), 1 = Supercritical (Fr >= 1); 0 when invalid.
    regime: u32,
    // 1 when the query produces a finite, well-defined Froude number, else 0.
    valid: u32,
    // Padding word to a 4-word (16-byte) stride.
    pad0: u32,
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
    let u = q.speed;
    let h = q.flow_depth;
    let g = q.gravity;
    let theta = q.slope_angle;
    let is_inclined = q.inclined != 0u;

    // Shared from_effective base checks: all finite, u >= 0, h > 0, g > 0.
    let u_ok = (abs(u) < FINITE_LIMIT) && (u >= 0.0);
    let h_ok = (abs(h) < FINITE_LIMIT) && (h > 0.0);
    let g_ok = (abs(g) < FINITE_LIMIT) && (g > 0.0);
    let base_ok = u_ok && h_ok && g_ok;

    // Inclined path: theta finite and cos(theta) > 0; cos evaluated natively.
    let theta_finite = abs(theta) < FINITE_LIMIT;
    let cos_theta = cos(theta);
    let incl_ok = theta_finite && (cos_theta > 0.0);
    // Effective normal gravity: g (flat) or g * cos(theta) (inclined).
    let g_eff = select(g, g * cos_theta, is_inclined);
    // The inclined branch adds its own validity requirement; flat adds none.
    let path_ok = select(true, incl_ok, is_inclined);

    let g_eff_ok = base_ok && path_ok && (abs(g_eff) < FINITE_LIMIT) && (g_eff > 0.0);

    // Wave speed sqrt(g_eff * h); guard the radicand so the invalid branch
    // never takes the root of a non-positive or non-finite value.
    let radicand = g_eff * h;
    let radicand_ok = g_eff_ok && (abs(radicand) < FINITE_LIMIT) && (radicand > 0.0);
    let radicand_safe = select(1.0, radicand, radicand_ok);
    let wave = sqrt(radicand_safe);
    let wave_ok = radicand_ok && (abs(wave) < FINITE_LIMIT) && (wave > 0.0);

    // Froude ratio; guard the divisor the same way.
    let wave_safe = select(1.0, wave, wave_ok);
    let froude = u / wave_safe;
    let ok = wave_ok && (abs(froude) < FINITE_LIMIT);

    // Half-open regime: Fr < 1 -> Subcritical (0), Fr >= 1 -> Supercritical (1).
    let regime = select(1u, 0u, froude < 1.0);

    var out: Result;
    out.froude = select(0.0, froude, ok);
    out.regime = select(0u, regime, ok);
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
/// four `f32` flow parameters, the `inclined` flag and padding to an 8-word
/// (32-byte) stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    speed: f32,
    flow_depth: f32,
    gravity: f32,
    slope_angle: f32,
    inclined: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the Froude number, the regime code, the validity flag and a padding
/// word — `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    froude: f32,
    regime: u32,
    valid: u32,
    pad0: u32,
}

/// One Froude-number query: a flow state and a flat/inclined selector.
///
/// Build a flat-base query with [`GranularFroudeQuery::from_flow`] or an
/// inclined-chute query with [`GranularFroudeQuery::from_inclined`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularFroudeQuery {
    /// Depth-averaged flow speed `u`.
    pub speed: f32,
    /// Flowing-layer depth `h`.
    pub flow_depth: f32,
    /// Gravitational acceleration `g`.
    pub gravity: f32,
    /// Chute inclination `theta` (radians); ignored when `inclined == 0`.
    pub slope_angle: f32,
    /// `0` selects the flat `from_flow` path; non-zero the inclined path.
    pub inclined: u32,
}

impl GranularFroudeQuery {
    /// Builds a flat-base query mirroring `GranularFroude::from_flow`.
    #[must_use]
    pub fn from_flow(speed: f32, flow_depth: f32, gravity: f32) -> GranularFroudeQuery {
        GranularFroudeQuery {
            speed,
            flow_depth,
            gravity,
            slope_angle: 0.0,
            inclined: 0,
        }
    }

    /// Builds an inclined-chute query mirroring
    /// `GranularFroude::from_inclined_flow`, with `slope_angle` in radians.
    #[must_use]
    pub fn from_inclined(
        speed: f32,
        flow_depth: f32,
        gravity: f32,
        slope_angle: f32,
    ) -> GranularFroudeQuery {
        GranularFroudeQuery {
            speed,
            flow_depth,
            gravity,
            slope_angle,
            inclined: 1,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GranularFroude` output for that flow state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularFroudeResult {
    /// The Froude number `u / sqrt(g_eff * h)` when valid, else `0`.
    pub froude: f32,
    /// `0` = `Subcritical` (`Fr < 1`), `1` = `Supercritical` (`Fr >= 1`); `0`
    /// when invalid.
    pub regime: u32,
    /// `1` when the query produces a finite, well-defined Froude number, else
    /// `0`.
    pub valid: u32,
}

/// Encodes one [`GranularFroudeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GranularFroudeQuery) -> GpuQuery {
    GpuQuery {
        speed: q.speed,
        flow_depth: q.flow_depth,
        gravity: q.gravity,
        slope_angle: q.slope_angle,
        inclined: q.inclined,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GranularFroudeResult`].
fn decode_result(raw: &GpuResult) -> GranularFroudeResult {
    GranularFroudeResult {
        froude: raw.froude,
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

/// A compiled, reusable Froude-number compute pipeline, twinning the `CPU`
/// golden `GranularFroude` constructors.
pub struct GpuGranularFroude {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularFroude {
    /// Compiles the Froude-number kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularFroude {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_froude"),
            source: ShaderSource::Wgsl(GRANULAR_FROUDE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_froude_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_froude_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_froude_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularFroude {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`GranularFroudeResult`]
    /// per input, in order.
    ///
    /// The `regime` and `valid` flags match the reference exactly and the
    /// `froude` scalar to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularFroudeQuery],
    ) -> Vec<GranularFroudeResult> {
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
            label: Some("prism_volumetric_granular_froude_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_froude_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_froude_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_froude_bind_group"),
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
            label: Some("prism_volumetric_granular_froude_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_froude_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_froude_pass"),
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
