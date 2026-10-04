//! `wgpu` compute twin of the Janssen silo pressure profile, from the `CPU`
//! golden `prism_physics_core::collider::janssen_pressure`'s `JanssenProfile`.
//!
//! A deep granular column does not pile up hydrostatically: wall friction
//! diverts part of the overburden into the walls, so the vertical stress
//! saturates with depth instead of growing linearly. Janssen's closed form
//! captures this as `σ_v(z) = σ_∞ · (1 − exp(−z / z_c))` with characteristic
//! depth `z_c = R / (μ_w · K)` and saturation stress `σ_∞ = ρ · g · z_c`. This
//! module ports that single stateless closed form onto the device: one thread
//! resolves one query, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same stresses the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Only `JanssenProfile` is twinned. The `CPU` golden derives the hydraulic
//! radius `R` from a `SiloCrossSection`; this twin instead takes `R` directly
//! as a scalar input and does not port the cross-section geometry or the
//! variable-length `sample` helper. For each query the kernel reproduces, for a
//! single depth:
//!
//! * Construction validity: `ρ`, `g`, `μ_w` and `K` must each be finite and
//!   `> 0`; then `z_c = R / (μ_w · K)` must be finite and `> 0`, and
//!   `σ_∞ = ρ · g · z_c` must be finite. Any failure yields `valid = 0` with
//!   all stresses `0`.
//! * `characteristic_depth = z_c`, `saturation_vertical = σ_∞`,
//!   `saturation_horizontal = K · σ_∞`, `saturation_wall_shear = μ_w · K · σ_∞`.
//! * `vertical_stress`: if `depth` is non-finite or `<= 0` it is `0`; otherwise
//!   `σ_v(z) = σ_∞ · (1 − exp(−z / z_c))`.
//! * `horizontal_stress = K · σ_v(z)` and `wall_shear_stress = μ_w · σ_h(z)`.
//! * `screening_fraction`: if `depth` is non-finite or `<= 0` it is `0`;
//!   otherwise with `hydrostatic = ρ · g · depth` (also `0` when non-positive),
//!   `clamp(1 − σ_v(z) / hydrostatic, 0, 1)`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` are not necessarily bit-exact; each valid scalar is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The
//! discrete `valid` flag is compared exactly; the parity sweep keeps every
//! material parameter strictly positive and finite, and keeps `μ_w · K` away
//! from zero, so the validity decision and `z_c` cannot be flipped by round-off.
//!
//! The reference computes the saturation factor `1 − exp(−z / z_c)` in `f64`;
//! the device evaluates `exp` in `f32`. The host oracle mirrors the reference
//! exactly in `f64`, and the tolerance absorbs the `f32`-device versus
//! `f64`-host difference.
//!
//! # Degenerate inputs
//!
//! A non-finite or non-positive material parameter, or a non-finite or
//! non-positive `z_c` or `σ_∞`, yields `valid = 0` with all stresses `0`. A
//! non-positive or non-finite `depth` leaves the profile valid but zeroes the
//! depth-dependent stresses and the screening fraction. Every divisor
//! (`μ_w · K`, `z_c` and the hydrostatic stress) is fed through a `select`
//! guard so an un-taken branch never divides by zero. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `exp`,
//! `clamp`, `+ - * /`, `select` and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested
//! with the ordered compare `abs(x) < 3.0e38` (which rejects both infinities
//! and `NaN`) rather than a bare `x == x`, and validity with ordered `> 0`;
//! there is no `f32` equality anywhere.
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

/// The portable core-`WGSL` Janssen-profile kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `JanssenProfile`; see the module documentation for the closed
/// form.
const JANSSEN_PRESSURE_PROFILE_WGSL: &str = r#"
// Janssen pressure-profile twin: one thread per query reproduces the construction
// validity gate plus the saturation stresses, the depth-dependent stresses and
// the screening fraction. It uses only the portable core-WGSL subset (abs, exp,
// clamp, + - * /, select plus unsigned index math) and has no loop and no branch
// other than the out-of-range early return, so it provably terminates.
// Finiteness is an ordered abs < 3.0e38 compare (rejecting infinities and NaN)
// and validity an ordered > 0 compare, both fed to select. Every divisor is
// guarded so an un-taken branch never divides by zero.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Hydraulic radius R.
    hydraulic_radius: f32,
    // Bulk density rho.
    bulk_density: f32,
    // Gravitational acceleration g.
    gravity: f32,
    // Wall friction coefficient mu_w.
    wall_friction: f32,
    // Lateral-to-vertical stress ratio K.
    k_ratio: f32,
    // Depth below the free surface.
    depth: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Characteristic Janssen depth z_c when valid, else 0.
    characteristic_depth: f32,
    // Saturation vertical stress sigma_inf when valid, else 0.
    saturation_vertical: f32,
    // Saturation horizontal stress K*sigma_inf when valid, else 0.
    saturation_horizontal: f32,
    // Saturation wall shear mu_w*K*sigma_inf when valid, else 0.
    saturation_wall_shear: f32,
    // Vertical stress sigma_v(z) when valid and depth > 0, else 0.
    vertical_stress: f32,
    // Horizontal stress K*sigma_v(z).
    horizontal_stress: f32,
    // Wall shear stress mu_w*sigma_h(z).
    wall_shear_stress: f32,
    // Screening fraction clamp(1 - sigma_v/hydrostatic, 0, 1).
    screening_fraction: f32,
    // 1 when the profile constructs, else 0.
    valid: u32,
    // Padding word.
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
    let radius = q.hydraulic_radius;
    let rho = q.bulk_density;
    let grav = q.gravity;
    let mu = q.wall_friction;
    let k = q.k_ratio;
    let depth = q.depth;

    // Material parameters must each be finite and strictly positive. Finiteness
    // via ordered abs < 3.0e38 (rejects +/-inf and NaN); no bare f32 equality.
    let rho_ok = (abs(rho) < FINITE_LIMIT) && (rho > 0.0);
    let grav_ok = (abs(grav) < FINITE_LIMIT) && (grav > 0.0);
    let mu_ok = (abs(mu) < FINITE_LIMIT) && (mu > 0.0);
    let k_ok = (abs(k) < FINITE_LIMIT) && (k > 0.0);
    let mat_ok = rho_ok && grav_ok && mu_ok && k_ok;

    // z_c = R / (mu_w * K); when mat_ok the product is strictly positive.
    let muk = mu * k;
    let muk_safe = select(1.0, muk, mat_ok);
    let z_c = radius / muk_safe;
    let z_c_ok = mat_ok && (abs(z_c) < FINITE_LIMIT) && (z_c > 0.0);

    // sigma_inf = rho * g * z_c; must stay finite.
    let sat_v = rho * grav * z_c;
    let valid = z_c_ok && (abs(sat_v) < FINITE_LIMIT);

    let sat_h = k * sat_v;
    let sat_shear = mu * sat_h;

    // Depth-dependent stresses: zero for non-finite or non-positive depth.
    let depth_ok = (abs(depth) < FINITE_LIMIT) && (depth > 0.0);
    let z_c_div = select(1.0, z_c, z_c_ok);
    let ratio = depth / z_c_div;
    let factor = 1.0 - exp(-ratio);
    let apply_depth = valid && depth_ok;
    let sigma_v = select(0.0, sat_v * factor, apply_depth);
    let sigma_h = select(0.0, k * sigma_v, apply_depth);
    let tau_w = select(0.0, mu * sigma_h, apply_depth);

    // Screening fraction against the frictionless hydrostatic stress.
    let hydrostatic = rho * grav * depth;
    let hydro_ok = depth_ok && (hydrostatic > 0.0);
    let hydro_safe = select(1.0, hydrostatic, hydro_ok);
    let screened = clamp(1.0 - sigma_v / hydro_safe, 0.0, 1.0);
    let screening = select(0.0, screened, valid && hydro_ok);

    var out: Result;
    out.characteristic_depth = select(0.0, z_c, valid);
    out.saturation_vertical = select(0.0, sat_v, valid);
    out.saturation_horizontal = select(0.0, sat_h, valid);
    out.saturation_wall_shear = select(0.0, sat_shear, valid);
    out.vertical_stress = sigma_v;
    out.horizontal_stress = sigma_h;
    out.wall_shear_stress = tau_w;
    out.screening_fraction = screening;
    out.valid = select(0u, 1u, valid);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The six inputs are padded to `8` `f32` words (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    hydraulic_radius: f32,
    bulk_density: f32,
    gravity: f32,
    wall_friction: f32,
    k_ratio: f32,
    depth: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the four saturation quantities, the three depth-dependent stresses,
/// the screening fraction, the validity flag and one padding word — `10` words
/// (`40` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    characteristic_depth: f32,
    saturation_vertical: f32,
    saturation_horizontal: f32,
    saturation_wall_shear: f32,
    vertical_stress: f32,
    horizontal_stress: f32,
    wall_shear_stress: f32,
    screening_fraction: f32,
    valid: u32,
    pad0: u32,
}

/// One Janssen-profile query: the hydraulic radius, the material parameters and
/// one depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenPressureProfileQuery {
    /// Hydraulic radius `R`.
    pub hydraulic_radius: f32,
    /// Bulk density `ρ`.
    pub bulk_density: f32,
    /// Gravitational acceleration `g`.
    pub gravity: f32,
    /// Wall friction coefficient `μ_w`.
    pub wall_friction: f32,
    /// Lateral-to-vertical stress ratio `K`.
    pub k_ratio: f32,
    /// Depth below the free surface.
    pub depth: f32,
}

impl JanssenPressureProfileQuery {
    /// Builds a query from the hydraulic radius, the material parameters and a
    /// depth.
    #[must_use]
    pub fn new(
        hydraulic_radius: f32,
        bulk_density: f32,
        gravity: f32,
        wall_friction: f32,
        k_ratio: f32,
        depth: f32,
    ) -> JanssenPressureProfileQuery {
        JanssenPressureProfileQuery {
            hydraulic_radius,
            bulk_density,
            gravity,
            wall_friction,
            k_ratio,
            depth,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `JanssenProfile` output for that configuration and depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenPressureProfileResult {
    /// Characteristic Janssen depth `z_c = R / (μ_w K)` when valid, else `0`.
    pub characteristic_depth: f32,
    /// Saturation vertical stress `σ_∞ = ρ g z_c` when valid, else `0`.
    pub saturation_vertical: f32,
    /// Saturation horizontal stress `K σ_∞` when valid, else `0`.
    pub saturation_horizontal: f32,
    /// Saturation wall shear stress `μ_w K σ_∞` when valid, else `0`.
    pub saturation_wall_shear: f32,
    /// Vertical stress `σ_v(z)` when valid and `depth > 0`, else `0`.
    pub vertical_stress: f32,
    /// Horizontal stress `K σ_v(z)`.
    pub horizontal_stress: f32,
    /// Wall shear stress `μ_w σ_h(z)`.
    pub wall_shear_stress: f32,
    /// Screening fraction `clamp(1 − σ_v(z) / (ρ g z), 0, 1)`.
    pub screening_fraction: f32,
    /// `1` when the profile constructs, else `0`.
    pub valid: u32,
}

/// Encodes one [`JanssenPressureProfileQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &JanssenPressureProfileQuery) -> GpuQuery {
    GpuQuery {
        hydraulic_radius: q.hydraulic_radius,
        bulk_density: q.bulk_density,
        gravity: q.gravity,
        wall_friction: q.wall_friction,
        k_ratio: q.k_ratio,
        depth: q.depth,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`JanssenPressureProfileResult`].
fn decode_result(raw: &GpuResult) -> JanssenPressureProfileResult {
    JanssenPressureProfileResult {
        characteristic_depth: raw.characteristic_depth,
        saturation_vertical: raw.saturation_vertical,
        saturation_horizontal: raw.saturation_horizontal,
        saturation_wall_shear: raw.saturation_wall_shear,
        vertical_stress: raw.vertical_stress,
        horizontal_stress: raw.horizontal_stress,
        wall_shear_stress: raw.wall_shear_stress,
        screening_fraction: raw.screening_fraction,
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

/// A compiled, reusable Janssen-profile compute pipeline, twinning the `CPU`
/// golden `JanssenProfile`.
pub struct GpuJanssenPressureProfile {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuJanssenPressureProfile {
    /// Compiles the Janssen-profile kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuJanssenPressureProfile {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile"),
            source: ShaderSource::Wgsl(JANSSEN_PRESSURE_PROFILE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuJanssenPressureProfile {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`JanssenPressureProfileResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the continuous scalars
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[JanssenPressureProfileQuery],
    ) -> Vec<JanssenPressureProfileResult> {
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
            label: Some("prism_volumetric_janssen_pressure_profile_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_bind_group"),
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
            label: Some("prism_volumetric_janssen_pressure_profile_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_janssen_pressure_profile_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_janssen_pressure_profile_pass"),
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
