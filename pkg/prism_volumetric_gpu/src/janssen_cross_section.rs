//! `wgpu` compute twin of the Janssen silo cross-section constants, from the
//! `CPU` golden `prism_physics_core::collider::janssen_pressure`'s
//! `SiloCrossSection` plus the depth-independent part of `JanssenProfile::new`.
//!
//! Granular material in a tall silo screens its own overburden through wall
//! friction, so the vertical stress saturates with depth. The depth-independent
//! "constants" of Janssen's profile are the hydraulic radius `R`, the
//! characteristic depth `z_c = R / (μ_w K)`, and the saturation stresses
//! `σ_∞ = ρ g z_c`, `σ_h∞ = K σ_∞`, `τ_w∞ = μ_w σ_h∞`. This module ports that
//! single stateless derivation onto the device: one thread resolves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same constants the reference does, not merely that the shader
//! compiles.
//!
//! This twin reproduces only the depth-independent constants; the
//! `exp`-dependent stress-at-depth samplers of `JanssenProfile` are out of
//! scope.
//!
//! # What is twinned
//!
//! For each query the kernel reduces the cross-section to its hydraulic radius
//! `R` and then evaluates the Janssen constants:
//!
//! * The `shape_kind` selects `R`: `0` hydraulic uses `R = param0`; `1`
//!   circular uses `R = 0.5 · radius` with `radius = param0`; `2` rectangular
//!   uses `R = (width · depth) / (2 · (width + depth))` with `width = param0`,
//!   `depth = param1`. The relevant shape parameters must be finite and `> 0`.
//! * The material parameters `bulk_density`, `gravity`, `wall_friction` and
//!   `k_ratio` must each be finite and `> 0`.
//! * `characteristic_depth = R / (wall_friction · k_ratio)` must be finite and
//!   `> 0`; `saturation_vertical = bulk_density · gravity · characteristic_depth`
//!   must be finite.
//! * `saturation_horizontal = k_ratio · saturation_vertical` and
//!   `saturation_wall_shear = wall_friction · saturation_horizontal`.
//!
//! If any gate fails the pair is invalid (`valid = false`, all outputs `0`).
//!
//! # Correctness model
//!
//! The reference evaluates these constants in `f32`, so the kernel reproduces
//! the same `f32` operator order. `CPU` and `GPU` need not be bit-exact (a `GPU`
//! may contract a multiply-add), so each valid scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly; the parity test keeps random parameters
//! well inside the valid band so the validity decision cannot be flipped by
//! round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite or non-positive shape or material parameter, a non-positive
//! hydraulic radius, or a non-finite/non-positive characteristic depth yields
//! `valid = false` with all outputs `0`. The divisors `2 (width + depth)` and
//! `wall_friction · k_ratio` are guarded with `select` so the unselected branch
//! never produces an infinity or `NaN`. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no transcendental call, no
//! `round` and no `f32` remainder, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. Finiteness is tested with the ordered compare `abs(x) < 3.0e38`
//! (which rejects both infinities and `NaN`) rather than a bare `x == x`, and
//! the range gates use ordered compares; there is no `f32` equality anywhere.
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

/// Shape kind: hydraulic radius supplied directly.
pub const SHAPE_HYDRAULIC: u32 = 0;
/// Shape kind: circular silo of inner `radius = param0`.
pub const SHAPE_CIRCULAR: u32 = 1;
/// Shape kind: rectangular silo of `width = param0`, `depth = param1`.
pub const SHAPE_RECTANGULAR: u32 = 2;

/// The portable core-`WGSL` Janssen cross-section kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden; see the module documentation for the closed form.
const JANSSEN_CROSS_SECTION_WGSL: &str = r#"
// Janssen cross-section twin: one thread per query reduces a silo cross-section
// to its hydraulic radius and evaluates the depth-independent Janssen
// constants. It uses only the portable core-WGSL subset (abs, + - * /, select
// plus unsigned index math), takes no optional feature, and has no loop, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and the range gates are ordered compares, all fed to
// select; the two divisors are guarded so no unselected branch yields inf/NaN.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Shape selector: 0 hydraulic, 1 circular, 2 rectangular.
    shape_kind: u32,
    // Padding word to keep the following scalars 4-byte packed.
    pad0: u32,
    // Shape parameter 0 (hydraulic R, circular radius, or rectangular width).
    param0: f32,
    // Shape parameter 1 (rectangular depth; unused by other shapes).
    param1: f32,
    // Bulk density rho.
    bulk_density: f32,
    // Gravitational acceleration g.
    gravity: f32,
    // Wall friction coefficient mu_w.
    wall_friction: f32,
    // Lateral-to-vertical stress ratio K.
    k_ratio: f32,
}

struct Result {
    // Hydraulic radius R when valid, else 0.
    hydraulic_radius: f32,
    // Characteristic depth z_c when valid, else 0.
    characteristic_depth: f32,
    // Saturation vertical stress sigma_inf when valid, else 0.
    saturation_vertical: f32,
    // Saturation horizontal stress K*sigma_inf when valid, else 0.
    saturation_horizontal: f32,
    // Saturation wall shear stress mu_w*K*sigma_inf when valid, else 0.
    saturation_wall_shear: f32,
    // 1 when every gate passes, else 0.
    valid: u32,
    // Padding words to a 16-byte-friendly stride.
    pad0: u32,
    pad1: u32,
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

    let kind = q.shape_kind;
    let is_hydraulic = kind == 0u;
    let is_circular = kind == 1u;
    let is_rectangular = kind == 2u;

    // Shape-parameter finiteness/positivity via ordered compares.
    let p0_ok = (abs(q.param0) < FINITE_LIMIT) && (q.param0 > 0.0);
    let p1_ok = (abs(q.param1) < FINITE_LIMIT) && (q.param1 > 0.0);

    // Rectangular hydraulic radius with a guarded divisor so the unselected
    // branch cannot emit inf/NaN.
    let width = q.param0;
    let depth = q.param1;
    let rect_sum = width + depth;
    let rect_denom = 2.0 * rect_sum;
    let rect_denom_ok = rect_denom > 0.0;
    let rect_denom_safe = select(1.0, rect_denom, rect_denom_ok);
    let rect_radius = (width * depth) / rect_denom_safe;

    // Select the hydraulic radius and the shape validity for this kind.
    var radius = 0.0;
    var shape_ok = false;
    if (is_hydraulic) {
        radius = q.param0;
        shape_ok = p0_ok;
    } else if (is_circular) {
        radius = 0.5 * q.param0;
        shape_ok = p0_ok;
    } else if (is_rectangular) {
        radius = rect_radius;
        shape_ok = p0_ok && p1_ok;
    }

    // from_hydraulic_radius re-checks the derived R.
    let radius_ok = shape_ok && (abs(radius) < FINITE_LIMIT) && (radius > 0.0);

    // Material parameters.
    let rho = q.bulk_density;
    let g = q.gravity;
    let mu = q.wall_friction;
    let k = q.k_ratio;
    let rho_ok = (abs(rho) < FINITE_LIMIT) && (rho > 0.0);
    let g_ok = (abs(g) < FINITE_LIMIT) && (g > 0.0);
    let mu_ok = (abs(mu) < FINITE_LIMIT) && (mu > 0.0);
    let k_ok = (abs(k) < FINITE_LIMIT) && (k > 0.0);
    let material_ok = rho_ok && g_ok && mu_ok && k_ok;

    // z_c = R / (mu * K) with a guarded divisor.
    let muk = mu * k;
    let muk_safe = select(1.0, muk, muk > 0.0);
    let z_c = radius / muk_safe;
    let z_c_ok = (abs(z_c) < FINITE_LIMIT) && (z_c > 0.0);

    // sigma_inf = rho * g * z_c.
    let sigma_v = rho * g * z_c;
    let sigma_v_ok = abs(sigma_v) < FINITE_LIMIT;

    let ok = radius_ok && material_ok && z_c_ok && sigma_v_ok;

    let sigma_h = k * sigma_v;
    let tau_w = mu * sigma_h;

    var out: Result;
    out.hydraulic_radius = select(0.0, radius, ok);
    out.characteristic_depth = select(0.0, z_c, ok);
    out.saturation_vertical = select(0.0, sigma_v, ok);
    out.saturation_horizontal = select(0.0, sigma_h, ok);
    out.saturation_wall_shear = select(0.0, tau_w, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
    out.pad1 = 0u;
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
/// the shape selector, two shape parameters and four material parameters,
/// padded to `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    shape_kind: u32,
    pad0: u32,
    param0: f32,
    param1: f32,
    bulk_density: f32,
    gravity: f32,
    wall_friction: f32,
    k_ratio: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the five Janssen constants, the validity flag and padding to `8`
/// words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    hydraulic_radius: f32,
    characteristic_depth: f32,
    saturation_vertical: f32,
    saturation_horizontal: f32,
    saturation_wall_shear: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One Janssen cross-section query: the shape selector, its shape parameters,
/// and the four material parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenCrossSectionQuery {
    /// Shape selector: [`SHAPE_HYDRAULIC`], [`SHAPE_CIRCULAR`] or
    /// [`SHAPE_RECTANGULAR`].
    pub shape_kind: u32,
    /// Shape parameter 0: hydraulic `R`, circular `radius`, or rectangular
    /// `width`.
    pub param0: f32,
    /// Shape parameter 1: rectangular `depth`; ignored by other shapes.
    pub param1: f32,
    /// Bulk density `ρ`.
    pub bulk_density: f32,
    /// Gravitational acceleration `g`.
    pub gravity: f32,
    /// Wall friction coefficient `μ_w`.
    pub wall_friction: f32,
    /// Lateral-to-vertical stress ratio `K`.
    pub k_ratio: f32,
}

impl JanssenCrossSectionQuery {
    /// Builds a hydraulic-radius query from `R` and the material parameters.
    #[must_use]
    pub fn hydraulic(
        hydraulic_radius: f32,
        bulk_density: f32,
        gravity: f32,
        wall_friction: f32,
        k_ratio: f32,
    ) -> JanssenCrossSectionQuery {
        JanssenCrossSectionQuery {
            shape_kind: SHAPE_HYDRAULIC,
            param0: hydraulic_radius,
            param1: 0.0,
            bulk_density,
            gravity,
            wall_friction,
            k_ratio,
        }
    }

    /// Builds a circular-silo query from the inner `radius` and the material
    /// parameters.
    #[must_use]
    pub fn circular(
        radius: f32,
        bulk_density: f32,
        gravity: f32,
        wall_friction: f32,
        k_ratio: f32,
    ) -> JanssenCrossSectionQuery {
        JanssenCrossSectionQuery {
            shape_kind: SHAPE_CIRCULAR,
            param0: radius,
            param1: 0.0,
            bulk_density,
            gravity,
            wall_friction,
            k_ratio,
        }
    }

    /// Builds a rectangular-silo query from `width`, `depth` and the material
    /// parameters.
    #[must_use]
    pub fn rectangular(
        width: f32,
        depth: f32,
        bulk_density: f32,
        gravity: f32,
        wall_friction: f32,
        k_ratio: f32,
    ) -> JanssenCrossSectionQuery {
        JanssenCrossSectionQuery {
            shape_kind: SHAPE_RECTANGULAR,
            param0: width,
            param1: depth,
            bulk_density,
            gravity,
            wall_friction,
            k_ratio,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference Janssen
/// cross-section constants for that silo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JanssenCrossSectionResult {
    /// Hydraulic radius `R` when valid, else `0`.
    pub hydraulic_radius: f32,
    /// Characteristic depth `z_c = R / (μ_w K)` when valid, else `0`.
    pub characteristic_depth: f32,
    /// Saturation vertical stress `σ_∞ = ρ g z_c` when valid, else `0`.
    pub saturation_vertical: f32,
    /// Saturation horizontal stress `K σ_∞` when valid, else `0`.
    pub saturation_horizontal: f32,
    /// Saturation wall shear stress `μ_w K σ_∞` when valid, else `0`.
    pub saturation_wall_shear: f32,
    /// `true` when every gate passes, else `false`.
    pub valid: bool,
}

/// Encodes one [`JanssenCrossSectionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &JanssenCrossSectionQuery) -> GpuQuery {
    GpuQuery {
        shape_kind: q.shape_kind,
        pad0: 0,
        param0: q.param0,
        param1: q.param1,
        bulk_density: q.bulk_density,
        gravity: q.gravity,
        wall_friction: q.wall_friction,
        k_ratio: q.k_ratio,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`JanssenCrossSectionResult`].
fn decode_result(raw: &GpuResult) -> JanssenCrossSectionResult {
    JanssenCrossSectionResult {
        hydraulic_radius: raw.hydraulic_radius,
        characteristic_depth: raw.characteristic_depth,
        saturation_vertical: raw.saturation_vertical,
        saturation_horizontal: raw.saturation_horizontal,
        saturation_wall_shear: raw.saturation_wall_shear,
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

/// A compiled, reusable Janssen cross-section compute pipeline, twinning the
/// `CPU` golden `prism_physics_core::collider::janssen_pressure` constants.
pub struct GpuJanssenCrossSection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuJanssenCrossSection {
    /// Compiles the Janssen cross-section kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuJanssenCrossSection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_janssen_cross_section"),
            source: ShaderSource::Wgsl(JANSSEN_CROSS_SECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuJanssenCrossSection {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`JanssenCrossSectionResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the five constants to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[JanssenCrossSectionQuery],
    ) -> Vec<JanssenCrossSectionResult> {
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
            label: Some("prism_volumetric_janssen_cross_section_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_bind_group"),
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
            label: Some("prism_volumetric_janssen_cross_section_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_janssen_cross_section_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_janssen_cross_section_pass"),
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
