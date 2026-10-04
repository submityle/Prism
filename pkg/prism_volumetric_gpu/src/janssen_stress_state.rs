//! `wgpu` compute twin of the Janssen silo stress-state closed form, from the
//! `CPU` golden `prism_physics_core::collider::janssen_pressure`'s
//! `JanssenProfile` depth getters.
//!
//! A granular column confined by frictional walls does not grow hydrostatic
//! with depth: friction progressively screens the overburden, so the vertical
//! stress saturates exponentially toward an asymptote `σ_∞` with a
//! characteristic depth `z_c`. This module ports the profile's four
//! depth-dependent getters onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same stresses the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel evaluates, in the golden operator order, the four
//! depth getters directly from the profile constants supplied by the query:
//!
//! * `vertical_stress = σ_∞ · (1 − exp(−depth / z_c))`, clamped to `0` for a
//!   non-positive or non-finite depth.
//! * `horizontal_stress = k_ratio · vertical_stress`.
//! * `wall_shear_stress = wall_friction · horizontal_stress`
//!   (`= μ_w · K · σ_v`).
//! * `screening_fraction = clamp(1 − vertical_stress / (ρ · g · depth), 0, 1)`,
//!   zero when the depth is non-positive or the hydrostatic reference
//!   `ρ · g · depth` is non-positive.
//!
//! The profile is constructed from silo geometry so `z_c` and `σ_∞` are
//! positive and finite invariants; the query carries them directly rather than
//! re-deriving them from radius and friction.
//!
//! # Correctness model
//!
//! The golden evaluates the saturation factor in `f64` because `exp` is banned
//! on `f32` there; this kernel uses the `WGSL` built-in `exp` in `f32`, and the
//! host oracle mirrors the golden by evaluating `exp` in `f64`. The four stress
//! scalars are therefore not bit-exact and are compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the sweep keeps
//! the ratio `depth / z_c` in `[0.05, 8]` so the `f32`/`f64` `exp` gap stays
//! within tolerance.
//!
//! # Degenerate inputs
//!
//! A non-positive or non-finite `depth` (or a non-positive/non-finite `z_c`)
//! yields all-zero stresses through the gates, with no separate validity flag;
//! a non-positive hydrostatic reference additionally zeroes only the screening
//! fraction. The characteristic-depth divisor and the hydrostatic divisor are
//! each guarded by a `select` that substitutes `1` when the corresponding gate
//! is closed, so no `inf`/`NaN` survives into the discarded branch. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `exp`, `clamp`,
//! `select`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `log`, `pow`, `sqrt`, no `round` and no `f32` remainder, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::janssen_pressure`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Janssen stress-state kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `JanssenProfile` depth getters; see the module
/// documentation for the closed form.
const JANSSEN_STRESS_STATE_WGSL: &str = r#"
// Janssen stress-state twin: one thread per query evaluates the vertical,
// horizontal and wall-shear stresses plus the friction screening fraction from
// the profile constants and the depth. It uses only the portable core-WGSL
// subset (abs, exp, clamp, select, + - * / plus unsigned index math), has no
// loop and no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; no bare
// f32 equality anywhere. The divisors are select-guarded so no inf/NaN survives
// into the discarded degenerate branch.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Characteristic Janssen depth z_c.
    characteristic_depth: f32,
    // Saturation (asymptotic) vertical stress sigma_inf.
    saturation_vertical: f32,
    // Lateral pressure coefficient K.
    k_ratio: f32,
    // Wall friction coefficient mu_w.
    wall_friction: f32,
    // Bulk density rho.
    bulk_density: f32,
    // Gravitational acceleration g.
    gravity: f32,
    // Depth below the free surface.
    depth: f32,
    // Padding to a 16-byte-friendly stride.
    pad: f32,
}

struct Result {
    // Vertical stress sigma_v(depth).
    vertical_stress: f32,
    // Horizontal (wall-normal) stress sigma_h(depth).
    horizontal_stress: f32,
    // Wall shear stress tau_w(depth).
    wall_shear_stress: f32,
    // Friction screening fraction in [0, 1].
    screening_fraction: f32,
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

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let depth_finite = abs(q.depth) < FINITE_LIMIT;
    let depth_ok = depth_finite && (q.depth > 0.0);
    let zc_finite = abs(q.characteristic_depth) < FINITE_LIMIT;
    let zc_ok = zc_finite && (q.characteristic_depth > 0.0);
    let v_ok = depth_ok && zc_ok;

    // Guard the characteristic-depth divisor: substitute 1 when the gate is
    // closed so the discarded branch never divides by zero or by inf.
    let zc_den = select(1.0, q.characteristic_depth, v_ok);
    let ratio = q.depth / zc_den;
    let factor = 1.0 - exp(-ratio);
    let sv_raw = q.saturation_vertical * factor;
    let sv = select(0.0, sv_raw, v_ok);
    let sh = q.k_ratio * sv;
    let tw = q.wall_friction * sh;

    // Screening: zero unless both depth is valid and the hydrostatic reference
    // rho*g*depth is positive. The divisor is select-guarded.
    let hydro = q.bulk_density * q.gravity * q.depth;
    let screen_ok = v_ok && (hydro > 0.0);
    let hydro_den = select(1.0, hydro, screen_ok);
    let screened_raw = clamp(1.0 - sv / hydro_den, 0.0, 1.0);
    let screening = select(0.0, screened_raw, screen_ok);

    var out: Result;
    out.vertical_stress = sv;
    out.horizontal_stress = sh;
    out.wall_shear_stress = tw;
    out.screening_fraction = screening;
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
/// the six profile constants, the depth and a padding word — `8` `f32` words
/// (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    characteristic_depth: f32,
    saturation_vertical: f32,
    k_ratio: f32,
    wall_friction: f32,
    bulk_density: f32,
    gravity: f32,
    depth: f32,
    pad: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the four depth-dependent stresses — `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    vertical_stress: f32,
    horizontal_stress: f32,
    wall_shear_stress: f32,
    screening_fraction: f32,
}

/// One Janssen stress-state query: the profile's characteristic depth,
/// saturation vertical stress, lateral pressure coefficient, wall friction,
/// bulk density, gravity and the depth to evaluate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenStressStateQuery {
    /// Characteristic Janssen depth `z_c`.
    pub characteristic_depth: f32,
    /// Saturation (asymptotic) vertical stress `sigma_inf`.
    pub saturation_vertical: f32,
    /// Lateral pressure coefficient `K`.
    pub k_ratio: f32,
    /// Wall friction coefficient `mu_w`.
    pub wall_friction: f32,
    /// Bulk density `rho`.
    pub bulk_density: f32,
    /// Gravitational acceleration `g`.
    pub gravity: f32,
    /// Depth below the free surface.
    pub depth: f32,
}

impl JanssenStressStateQuery {
    /// Builds a query from the profile constants and the depth to evaluate.
    #[must_use]
    pub fn new(
        characteristic_depth: f32,
        saturation_vertical: f32,
        k_ratio: f32,
        wall_friction: f32,
        bulk_density: f32,
        gravity: f32,
        depth: f32,
    ) -> JanssenStressStateQuery {
        JanssenStressStateQuery {
            characteristic_depth,
            saturation_vertical,
            k_ratio,
            wall_friction,
            bulk_density,
            gravity,
            depth,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference profile's
/// four depth getters at that depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenStressStateResult {
    /// Vertical stress `sigma_v(depth)`, zero for a non-positive depth.
    pub vertical_stress: f32,
    /// Horizontal (wall-normal) stress `sigma_h(depth) = K * sigma_v`.
    pub horizontal_stress: f32,
    /// Wall shear stress `tau_w(depth) = mu_w * sigma_h`.
    pub wall_shear_stress: f32,
    /// Friction screening fraction in `[0, 1]`.
    pub screening_fraction: f32,
}

/// Encodes one [`JanssenStressStateQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &JanssenStressStateQuery) -> GpuQuery {
    GpuQuery {
        characteristic_depth: q.characteristic_depth,
        saturation_vertical: q.saturation_vertical,
        k_ratio: q.k_ratio,
        wall_friction: q.wall_friction,
        bulk_density: q.bulk_density,
        gravity: q.gravity,
        depth: q.depth,
        pad: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`JanssenStressStateResult`].
fn decode_result(raw: &GpuResult) -> JanssenStressStateResult {
    JanssenStressStateResult {
        vertical_stress: raw.vertical_stress,
        horizontal_stress: raw.horizontal_stress,
        wall_shear_stress: raw.wall_shear_stress,
        screening_fraction: raw.screening_fraction,
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

/// A compiled, reusable Janssen stress-state compute pipeline, twinning the
/// `CPU` golden `JanssenProfile` depth getters.
pub struct GpuJanssenStressState {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuJanssenStressState {
    /// Compiles the Janssen stress-state kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuJanssenStressState {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_janssen_stress_state"),
            source: ShaderSource::Wgsl(JANSSEN_STRESS_STATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuJanssenStressState {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`JanssenStressStateResult`] per input, in order.
    ///
    /// Each stress scalar matches the reference to the module's tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[JanssenStressStateQuery],
    ) -> Vec<JanssenStressStateResult> {
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
            label: Some("prism_volumetric_janssen_stress_state_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_bind_group"),
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
            label: Some("prism_volumetric_janssen_stress_state_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_janssen_stress_state_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_janssen_stress_state_pass"),
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
