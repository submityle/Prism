//! `wgpu` compute twin of the full Hertzian contact-force resolver from the
//! `CPU` golden `prism_physics_core::collider::hertz_contact`'s
//! `evaluate_hertz_contact`.
//!
//! Two elastic grains in contact develop a nonlinear normal force that
//! stiffens with penetration (`F_elastic = (4/3)·E*·√R*·δ^{3/2}`), a viscous
//! approach-damping term `−γₙ·v_n` clamped so the contact never pulls, and a
//! tangential viscous friction `γ_t·‖v_t‖` capped at the Coulomb limit
//! `μ·F_n`. This module ports that stateless, no-`RNG` closed form onto the
//! device: one compute thread resolves one contact, so a passing real-device
//! parity test is direct evidence the kernel reproduces the same branch
//! structure and arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries the material parameters (`effective_modulus`,
//! `normal_damping`, `tangential_damping`, `friction`), the contact axis
//! `axis` (from `a` toward `b`, normalized defensively), the penetration
//! `overlap`, the reduced contact radius `effective_radius`, and the relative
//! velocity `rel_vel = v_b − v_a`. For each the kernel reproduces the golden:
//!
//! * `axis_len = ‖axis‖`; the contact is degenerate when `axis_len ≤ EPS`, or
//!   `overlap` is non-finite or `≤ 0`, or `effective_radius` is non-finite or
//!   `≤ 0`. A degenerate contact yields a zero force, `overlap.max(0)`, zero
//!   magnitudes and `sliding = false`.
//! * otherwise `n = axis / axis_len`, `v_n = rel_vel · n`,
//!   `elastic = (4/3)·E*·√R*·δ^{3/2}`,
//!   `normal_magnitude = max(elastic − γₙ·v_n, 0)`,
//!   `normal_force = normal_magnitude · n`;
//! * `v_t = rel_vel − v_n·n`, `speed_t = ‖v_t‖`; when `speed_t > EPS`,
//!   `viscous = γ_t·speed_t`, `coulomb = μ·normal_magnitude`,
//!   `sliding = viscous ≥ coulomb`, `magnitude = min(viscous, coulomb)`,
//!   `tangential_force = −magnitude · v_t/speed_t`; else all tangential terms
//!   are zero and `sliding = false`;
//! * `force_on_b = normal_force + tangential_force`.
//!
//! # Correctness model
//!
//! The golden evaluates the elastic `δ^{3/2}` term in `f64` and narrows the
//! result to `f32` (`r_eff.sqrt()`, `delta.powf(1.5)`), keeping the fractional
//! power off the `f32` fast-math path, while the kernel uses `f32` `sqrt` and
//! `pow`, so `CPU` and `GPU` are not bit-exact. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! each continuous component; the discrete `sliding` flag is compared exactly.
//! The sweep keeps every input well clear of the degeneracy and sliding knees
//! so the `f64`/`f32` round-off cannot flip a discrete decision.
//!
//! # Degenerate inputs
//!
//! A near-zero axis, a non-positive or non-finite `overlap`, or a non-positive
//! or non-finite `effective_radius` yields `force_on_b = 0`,
//! `overlap = overlap.max(0)`, zero magnitudes and `sliding = false`. Every
//! device-side branch uses an ordered comparison (`abs(x) < 3.0e38`, `x > 0`,
//! `x >= y`) feeding `select`; the normalizing divisions are guarded so the
//! unselected arm never produces an infinity or `NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset plus the `sqrt`, `pow`,
//! `length` and `dot` builtins the closed form requires; it avoids `sin`,
//! `cos`, `tan`, `exp`, `log`, `round`, float modulo and bare `f32` equality,
//! and uses no reserved identifier, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`.
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

/// The portable Hertz contact-force kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `evaluate_hertz_contact`; see the module documentation for the
/// closed form.
const HERTZ_CONTACT_FORCE_WGSL: &str = r#"
// Hertz contact-force twin: one thread per contact reproduces the force the
// golden evaluate_hertz_contact builds from the material parameters, the
// contact axis, the penetration, the reduced radius and the relative velocity.
// It uses the portable core-WGSL subset plus sqrt, pow, length and dot. There
// is no loop and no branch, so it provably terminates. Every degenerate test
// is an ordered compare (abs < 3.0e38, x > 0, x >= y) feeding select, never a
// bare float equality or a fast-math NaN sentinel; the normalizing divisions
// are guarded so the unselected arm never produces an infinity or NaN.
//
// Provenance: 孪生自本仓 prism_physics_core::collider::hertz_contact；
// 无第三方引擎源码或衍生代码。

// f32::EPSILON, the near-zero threshold the golden uses for the axis and the
// tangential speed.
const EPS: f32 = 1.1920929e-7;
// Finiteness bound: abs(x) < FINITE_LIMIT rejects both infinities and NaN.
const FINITE_LIMIT: f32 = 3.0e38;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Material parameters of the Hertz law.
    effective_modulus: f32,
    normal_damping: f32,
    tangential_damping: f32,
    friction: f32,
    // Penetration depth and reduced contact radius.
    overlap: f32,
    effective_radius: f32,
    pad0: f32,
    pad1: f32,
    // Contact axis (from a toward b), padded to a 16-byte group.
    ax: f32,
    ay: f32,
    az: f32,
    pad2: f32,
    // Relative velocity v_b - v_a, padded to a 16-byte group.
    rvx: f32,
    rvy: f32,
    rvz: f32,
    pad3: f32,
}

struct Result {
    // Force on body b, padded to a 16-byte group.
    fbx: f32,
    fby: f32,
    fbz: f32,
    pad0: f32,
    // Clamped penetration and the two force magnitudes.
    overlap: f32,
    normal_magnitude: f32,
    tangential_magnitude: f32,
    // 1 when the tangential response is at the Coulomb (sliding) limit.
    sliding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let axis = vec3<f32>(q.ax, q.ay, q.az);
    let rel_vel = vec3<f32>(q.rvx, q.rvy, q.rvz);

    // Degeneracy gate mirroring the golden guard.
    let axis_len = length(axis);
    let axis_ok = axis_len > EPS;
    let overlap_finite = abs(q.overlap) < FINITE_LIMIT;
    let overlap_pos = q.overlap > 0.0;
    let radius_finite = abs(q.effective_radius) < FINITE_LIMIT;
    let radius_pos = q.effective_radius > 0.0;
    let non_degenerate = axis_ok && overlap_finite && overlap_pos && radius_finite && radius_pos;

    // Normalize the axis; the denominator is guarded so the unselected arm is
    // finite.
    let safe_len = select(1.0, axis_len, axis_ok);
    let n = axis / safe_len;
    let v_n = dot(rel_vel, n);

    // Hertz elastic penalty. The sqrt and pow arguments are guarded positive so
    // the unselected (degenerate) arm never produces an infinity or NaN.
    let safe_radius = select(1.0, q.effective_radius, radius_finite && radius_pos);
    let safe_overlap = select(1.0, q.overlap, overlap_finite && overlap_pos);
    let elastic = (4.0 / 3.0) * q.effective_modulus * sqrt(safe_radius) * pow(safe_overlap, 1.5);
    let normal_magnitude = max(elastic - q.normal_damping * v_n, 0.0);
    let normal_force = normal_magnitude * n;

    // Tangential: viscous friction opposing slip, capped at the Coulomb limit.
    let v_t = rel_vel - v_n * n;
    let speed_t = length(v_t);
    let slip = speed_t > EPS;
    let safe_speed = select(1.0, speed_t, slip);
    let t_hat = v_t / safe_speed;
    let viscous = q.tangential_damping * speed_t;
    let coulomb = q.friction * normal_magnitude;
    let sliding_pred = viscous >= coulomb;
    let magnitude = min(viscous, coulomb);
    let zero3 = vec3<f32>(0.0, 0.0, 0.0);
    let tangential_force = select(zero3, -magnitude * t_hat, slip);
    let tangential_magnitude = select(0.0, magnitude, slip);
    let sliding_out = slip && sliding_pred;

    // Clamp the penetration like the golden overlap.max(0.0); a non-finite
    // overlap collapses to 0 so the field stays comparable.
    let overlap_clamped = max(select(0.0, q.overlap, overlap_finite), 0.0);

    let final_force = select(zero3, normal_force + tangential_force, non_degenerate);
    let final_normal = select(0.0, normal_magnitude, non_degenerate);
    let final_tangential = select(0.0, tangential_magnitude, non_degenerate);
    let final_sliding = non_degenerate && sliding_out;

    var out: Result;
    out.fbx = final_force.x;
    out.fby = final_force.y;
    out.fbz = final_force.z;
    out.pad0 = 0.0;
    out.overlap = overlap_clamped;
    out.normal_magnitude = final_normal;
    out.tangential_magnitude = final_tangential;
    out.sliding = select(0u, 1u, final_sliding);
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
/// All components are scalar `f32` so the slot contains no `vec3` and the host
/// and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    effective_modulus: f32,
    normal_damping: f32,
    tangential_damping: f32,
    friction: f32,
    overlap: f32,
    effective_radius: f32,
    pad0: f32,
    pad1: f32,
    ax: f32,
    ay: f32,
    az: f32,
    pad2: f32,
    rvx: f32,
    rvy: f32,
    rvz: f32,
    pad3: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The force vector is stored as flat scalars; the trailing `sliding`
/// word keeps the discrete flag beside the continuous quantities.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    fbx: f32,
    fby: f32,
    fbz: f32,
    pad0: f32,
    overlap: f32,
    normal_magnitude: f32,
    tangential_magnitude: f32,
    sliding: u32,
}

/// One query for the Hertz contact-force twin: the material parameters, the
/// contact axis, the penetration, the reduced contact radius, and the relative
/// velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzContactForceQuery {
    /// Effective contact modulus `E*`.
    pub effective_modulus: f32,
    /// Normal viscous damping `γₙ`.
    pub normal_damping: f32,
    /// Tangential viscous damping `γ_t`.
    pub tangential_damping: f32,
    /// Coulomb friction coefficient `μ`.
    pub friction: f32,
    /// Contact axis, pointing from `a` toward `b` (normalized defensively).
    pub axis: [f32; 3],
    /// Penetration depth `δ` (positive while overlapping).
    pub overlap: f32,
    /// Reduced contact radius `R*`.
    pub effective_radius: f32,
    /// Relative velocity `v_b − v_a`.
    pub rel_vel: [f32; 3],
}

impl HertzContactForceQuery {
    /// Builds a query from the material parameters and contact geometry.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "flat scalar constructor mirrors the golden evaluate_hertz_contact signature"
    )]
    pub fn new(
        effective_modulus: f32,
        normal_damping: f32,
        tangential_damping: f32,
        friction: f32,
        axis: [f32; 3],
        overlap: f32,
        effective_radius: f32,
        rel_vel: [f32; 3],
    ) -> HertzContactForceQuery {
        HertzContactForceQuery {
            effective_modulus,
            normal_damping,
            tangential_damping,
            friction,
            axis,
            overlap,
            effective_radius,
            rel_vel,
        }
    }
}

/// One resolved contact, mirroring the golden `ContactForce` for that pair.
/// When the contact is degenerate the force and magnitudes are zero,
/// `overlap = overlap.max(0)`, and `sliding` is `false`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzContactForceResult {
    /// Force on body `b` (body `a` receives its negation).
    pub force_on_b: [f32; 3],
    /// Clamped penetration depth `overlap.max(0)`.
    pub overlap: f32,
    /// Normal force magnitude `max(elastic − γₙ·v_n, 0)`.
    pub normal_magnitude: f32,
    /// Tangential force magnitude `min(viscous, Coulomb)`.
    pub tangential_magnitude: f32,
    /// `true` when the tangential response is at the Coulomb (sliding) limit.
    pub sliding: bool,
}

/// Encodes one [`HertzContactForceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HertzContactForceQuery) -> GpuQuery {
    GpuQuery {
        effective_modulus: q.effective_modulus,
        normal_damping: q.normal_damping,
        tangential_damping: q.tangential_damping,
        friction: q.friction,
        overlap: q.overlap,
        effective_radius: q.effective_radius,
        pad0: 0.0,
        pad1: 0.0,
        ax: q.axis[0],
        ay: q.axis[1],
        az: q.axis[2],
        pad2: 0.0,
        rvx: q.rel_vel[0],
        rvy: q.rel_vel[1],
        rvz: q.rel_vel[2],
        pad3: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HertzContactForceResult`], turning the `u32` flag into a `bool`.
fn decode_result(raw: &GpuResult) -> HertzContactForceResult {
    HertzContactForceResult {
        force_on_b: [raw.fbx, raw.fby, raw.fbz],
        overlap: raw.overlap,
        normal_magnitude: raw.normal_magnitude,
        tangential_magnitude: raw.tangential_magnitude,
        sliding: raw.sliding != 0,
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

/// A compiled, reusable Hertz contact-force compute pipeline, twinning the
/// `CPU` golden `evaluate_hertz_contact`.
pub struct GpuHertzContactForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHertzContactForce {
    /// Compiles the Hertz contact-force kernel on `ctx`.
    ///
    /// The kernel uses the portable core-`WGSL` subset plus the `sqrt`, `pow`,
    /// `length` and `dot` builtins, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHertzContactForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hertz_contact_force"),
            source: ShaderSource::Wgsl(HERTZ_CONTACT_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHertzContactForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every contact in `queries` and returns one
    /// [`HertzContactForceResult`] per input, in order.
    ///
    /// Each continuous component matches the reference to within the tolerance
    /// documented on this module; the `sliding` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HertzContactForceQuery],
    ) -> Vec<HertzContactForceResult> {
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
            label: Some("prism_volumetric_hertz_contact_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_bind_group"),
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
            label: Some("prism_volumetric_hertz_contact_force_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hertz_contact_force_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hertz_contact_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per contact, flattened to a 1-D dispatch.
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
