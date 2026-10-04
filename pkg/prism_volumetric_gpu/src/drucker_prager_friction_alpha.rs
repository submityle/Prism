//! `wgpu` compute twin of the Drucker–Prager friction-coefficient closed form,
//! from the `CPU` golden
//! `prism_physics_core::collider::tet_fem_drucker_prager_plasticity`'s
//! `DruckerPragerModel::from_friction_angle`.
//!
//! A granular or cohesive-frictional material has an internal friction angle
//! `φ` (in degrees). The standard Drucker–Prager/Mohr–Coulomb match turns that
//! angle into the base friction coefficient
//! `α = √(2/3) · 2 sinφ / (3 − sinφ)`. This module ports that scalar onto the
//! device: one thread resolves one query, so a passing real-device parity test
//! is direct evidence the ported kernel computes the same coefficient the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the `α` branch of
//! `from_friction_angle` for one friction angle `angle_degrees`:
//!
//! * `s = sin(angle_degrees · π / 180)`,
//! * `α = √(2/3) · 2 · s / (3 − s)`.
//!
//! The validity gate mirrors the golden guard: the result is valid only when
//! `angle_degrees` is finite and strictly between `0` and `90` degrees. An
//! invalid query reports `valid = false`, `alpha = 0`.
//!
//! # Correctness model
//!
//! The golden evaluates the radians conversion, the sine and the `√(2/3)`
//! factor in `f64` (`f64::from(angle).to_radians().sin()`,
//! `(2.0_f64 / 3.0).sqrt()`) and only casts the final `α` to `f32`, while the
//! kernel uses `f32` `sin` and `sqrt`, so `CPU` and `GPU` are not bit-exact;
//! the valid `alpha` scalar is compared with an `abs <= 1e-4 || rel <= 1e-3`
//! tolerance (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared
//! exactly; the parity sweep keeps the angle well inside `(0, 90)` so the
//! `f64`/`f32` numerical gap cannot flip the validity decision.
//!
//! # Degenerate inputs
//!
//! A non-finite angle, or an angle outside the open `(0, 90)` interval, yields
//! `valid = false` with `alpha = 0`. The `3 − s` divisor is fed through a
//! `select` guard so an un-taken (invalid) branch never divides by a value it
//! should not. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset plus the `sin` and `sqrt`
//! builtins the closed form requires; it avoids `cos`, `tan`, `exp`, `log`,
//! `pow`, `round`, `f32` remainder and bare `f32` equality. Finiteness is
//! tested with the ordered compare `abs(x) < 3.0e38` (which rejects both
//! infinities and `NaN`) rather than a bare `x == x`, and validity with ordered
//! `> 0` and `< 90`; there is no `f32` equality anywhere, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_drucker_prager_plasticity`；无第三方引擎源码或衍生代码。
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

/// The portable Drucker–Prager friction-coefficient kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `α` branch of the `CPU` golden
/// `DruckerPragerModel::from_friction_angle`; see the module documentation for
/// the closed form.
const DRUCKER_PRAGER_FRICTION_ALPHA_WGSL: &str = r#"
// Drucker-Prager friction-coefficient twin: one thread per query reproduces the
// alpha branch of from_friction_angle. It uses the portable core-WGSL subset
// plus the sin and sqrt builtins the closed form requires. There is no loop and
// no branch, so it provably terminates. Finiteness is an ordered abs < 3.0e38
// compare (rejecting infinities and NaN) and validity ordered > 0 and < 90
// compares, all fed to select; there is no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Internal friction angle in degrees.
    angle_degrees: f32,
    // Padding words to a 16-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Base friction coefficient alpha when valid, else 0.
    alpha: f32,
    // 1 when the angle is finite and strictly in (0, 90), else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
// pi / 180, the degrees-to-radians factor.
const DEG_TO_RAD: f32 = 0.017453292519943295;
// sqrt(2 / 3), the Drucker-Prager/Mohr-Coulomb match factor.
const SQRT_TWO_THIRDS: f32 = 0.816496580927726;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let angle = q.angle_degrees;

    // Validity gate matching the golden guard: angle finite and strictly in the
    // open interval (0, 90) degrees.
    let angle_finite = abs(angle) < FINITE_LIMIT;
    let angle_ok = (angle > 0.0) && (angle < 90.0);
    let valid = angle_finite && angle_ok;

    let s = sin(angle * DEG_TO_RAD);
    // Guard the divisor so the un-taken (invalid) branch never divides by a
    // value it should not; when valid, s is in (0, 1) so (3 - s) is positive.
    let raw_denom = 3.0 - s;
    let denom = select(1.0, raw_denom, valid);
    let computed = SQRT_TWO_THIRDS * 2.0 * s / denom;

    var out: Result;
    out.alpha = select(0.0, computed, valid);
    out.valid = select(0u, 1u, valid);
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
/// the friction angle in degrees, padded to `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    angle_degrees: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the friction coefficient and the validity flag — `2` words
/// (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    alpha: f32,
    valid: u32,
}

/// One Drucker–Prager friction query: the internal friction angle in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerFrictionAlphaQuery {
    /// Internal friction angle `φ` in degrees.
    pub angle_degrees: f32,
}

impl DruckerPragerFrictionAlphaQuery {
    /// Builds a query from the internal friction angle in degrees.
    #[must_use]
    pub fn new(angle_degrees: f32) -> DruckerPragerFrictionAlphaQuery {
        DruckerPragerFrictionAlphaQuery { angle_degrees }
    }
}

/// One resolved answer for a single query, mirroring the `α` branch of the
/// reference `DruckerPragerModel::from_friction_angle` for that angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerFrictionAlphaResult {
    /// The base friction coefficient `α` when valid, else `0`.
    pub alpha: f32,
    /// `true` when the angle is finite and strictly in the open interval
    /// `(0, 90)` degrees, else `false`.
    pub valid: bool,
}

/// Encodes one [`DruckerPragerFrictionAlphaQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &DruckerPragerFrictionAlphaQuery) -> GpuQuery {
    GpuQuery {
        angle_degrees: q.angle_degrees,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`DruckerPragerFrictionAlphaResult`], turning the `u32` flag into a `bool`.
fn decode_result(raw: &GpuResult) -> DruckerPragerFrictionAlphaResult {
    DruckerPragerFrictionAlphaResult {
        alpha: raw.alpha,
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

/// A compiled, reusable Drucker–Prager friction-coefficient compute pipeline,
/// twinning the `α` branch of the `CPU` golden
/// `DruckerPragerModel::from_friction_angle`.
pub struct GpuDruckerPragerFrictionAlpha {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDruckerPragerFrictionAlpha {
    /// Compiles the friction-coefficient kernel on `ctx`.
    ///
    /// The kernel uses the portable core-`WGSL` subset plus the `sin` and
    /// `sqrt` builtins, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDruckerPragerFrictionAlpha {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha"),
            source: ShaderSource::Wgsl(DRUCKER_PRAGER_FRICTION_ALPHA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDruckerPragerFrictionAlpha {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`DruckerPragerFrictionAlphaResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `alpha` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DruckerPragerFrictionAlphaQuery],
    ) -> Vec<DruckerPragerFrictionAlphaResult> {
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
            label: Some("prism_volumetric_drucker_prager_friction_alpha_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_bind_group"),
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
            label: Some("prism_volumetric_drucker_prager_friction_alpha_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_drucker_prager_friction_alpha_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_drucker_prager_friction_alpha_pass"),
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
