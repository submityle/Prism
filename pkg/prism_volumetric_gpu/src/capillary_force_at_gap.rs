//! `wgpu` compute twin of the capillary-bridge attractive-force closed form,
//! from the `CPU` golden
//! `prism_physics_core::collider::capillary_bridge`'s
//! `CapillaryBridgeModel::force_at_gap`.
//!
//! A pendular liquid bridge between two wet grains *pulls* them together across
//! a surface gap `H`, surviving until the gap exceeds a rupture distance. This
//! module ports the Rabinovich et al. (2005) force magnitude onto the device:
//! one thread resolves one query, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same force the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `force_at_gap` for one grain pair with
//! explicit radii `radius_a`, `radius_b` and bridge parameters surface tension
//! `gamma`, contact angle `theta` and liquid volume `V`, at surface separation
//! `gap`:
//!
//! * The rupture distance is `H_rupture = (1 + 0.5 * theta) * V^(1/3)`.
//! * The reduced radius is `R = 2 * radius_a * radius_b / (radius_a + radius_b)`.
//! * The contact adhesion is `F0 = 2 * PI * R * gamma * cos(theta)`.
//! * At contact (`gap <= 0`) the magnitude is `F0`; otherwise with
//!   `sep = gap`, `inner = 1 + 2 * V / (PI * R * sep * sep)`,
//!   `d_sp = 0.5 * sep * (-1 + sqrt(inner))`, the force is
//!   `F0 / (1 + sep / (2 * d_sp))`.
//!
//! The validity gate follows the golden `force_at_gap` exactly: the result is
//! valid only when `gap` is finite, `gap <= H_rupture`, and both radii are
//! finite and strictly positive. The golden `force_at_gap` does *not* re-gate
//! `gamma`, `theta` or `V` (those are validated only when a model is built), so
//! neither does the twin. An invalid query reports `valid = 0`, `magnitude = 0`.
//!
//! # Correctness model
//!
//! The golden takes the cube root and cosine in `f64`
//! (`(V as f64).cbrt() as f32`, `(theta as f64).cos() as f32`) while the kernel
//! uses `f32` `pow(V, 1/3)` and `cos`, so `CPU` and `GPU` are not bit-exact; the
//! valid `magnitude` scalar is compared with an `abs <= 1e-4 || rel <= 1e-3`
//! tolerance (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared
//! exactly; the parity sweep keeps `gap` well clear of the rupture knee so the
//! `f64`/`f32` numerical gap cannot flip the validity decision.
//!
//! # Degenerate inputs
//!
//! A non-finite `gap`, a `gap` past rupture, or a non-finite / non-positive
//! radius yields `valid = 0` with `magnitude = 0`. Every divisor
//! (`radius_a + radius_b`, `PI * R * sep * sep`, `2 * d_sp` and the final
//! `1 + sep / (2 * d_sp)`) is fed through a `select` guard so an un-taken branch
//! never evaluates a division by zero. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset plus the `sqrt`, `cos` and
//! `pow` builtins the closed form requires; it avoids `sin`, `tan`, `exp`,
//! `log`, `round`, `f32` remainder and bare `f32` equality. Finiteness is
//! tested with the ordered compare `abs(x) < 3.0e38` (which rejects both
//! infinities and `NaN`) rather than a bare `x == x`, and validity with ordered
//! `> 0` / `<=`; there is no `f32` equality anywhere, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
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

/// The portable capillary force-at-gap kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `CapillaryBridgeModel::force_at_gap`; see the module
/// documentation for the closed form.
const CAPILLARY_FORCE_AT_GAP_WGSL: &str = r#"
// Force-at-gap twin: one thread per query reproduces force_at_gap. It uses the
// portable core-WGSL subset plus the sqrt, cos and pow builtins the Rabinovich
// closed form requires. There is no loop and no branch, so it provably
// terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and validity ordered > 0 / <= compares, all fed to
// select; there is no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Radius of the first grain.
    radius_a: f32,
    // Radius of the second grain.
    radius_b: f32,
    // Surface separation (gap) between the grains.
    gap: f32,
    // Liquid surface tension gamma.
    surface_tension: f32,
    // Solid-liquid contact angle theta, radians.
    contact_angle: f32,
    // Liquid bridge volume V.
    liquid_volume: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Attractive force magnitude when valid, else 0.
    magnitude: f32,
    // 1 when gap finite, gap <= rupture and both radii finite & > 0, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const PI: f32 = 3.1415927;
const ONE_THIRD: f32 = 0.33333334;
const TINY: f32 = 1.0e-30;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let ra = q.radius_a;
    let rb = q.radius_b;
    let gap = q.gap;
    let gamma = q.surface_tension;
    let theta = q.contact_angle;
    let vol = q.liquid_volume;

    // Rupture distance H_rupture = (1 + 0.5*theta) * V^(1/3). Guard the pow base
    // so a non-positive volume never produces NaN in the un-taken branch; valid
    // bridges carry V > 0.
    let v_pos = vol > 0.0;
    let v_base = select(1.0, vol, v_pos);
    let cube_root = pow(v_base, ONE_THIRD);
    let rupture = (1.0 + 0.5 * theta) * cube_root;

    // Validity gate matching the golden force_at_gap: gap finite, gap within
    // rupture, both radii finite and strictly positive.
    let gap_finite = abs(gap) < FINITE_LIMIT;
    let within_rupture = gap <= rupture;
    let radius_ok = (abs(ra) < FINITE_LIMIT) && (abs(rb) < FINITE_LIMIT) && (ra > 0.0) && (rb > 0.0);
    let valid = gap_finite && within_rupture && radius_ok;

    // Reduced radius 2*a*b/(a+b); guard the divisor for the invalid branch.
    let sum = ra + rb;
    let sum_denom = select(1.0, sum, radius_ok);
    let reduced = 2.0 * ra * rb / sum_denom;

    // Contact adhesion F0 = 2*PI*R*gamma*cos(theta).
    let cos_theta = cos(theta);
    let f0 = 2.0 * PI * reduced * gamma * cos_theta;

    // Overlapping cores (gap < 0) clamp to contact.
    let sep = max(gap, 0.0);
    let in_contact = sep <= 0.0;

    // Embracing distance d_sp = 0.5*sep*(-1 + sqrt(1 + 2V/(PI*R*sep*sep))).
    let pirs = PI * reduced * sep * sep;
    let pirs_ok = abs(pirs) > TINY;
    let pirs_denom = select(1.0, pirs, pirs_ok);
    let inner_raw = 1.0 + 2.0 * vol / pirs_denom;
    let inner_ok = inner_raw > 0.0;
    let inner = select(1.0, inner_raw, inner_ok);
    let embracing = 0.5 * sep * (-1.0 + sqrt(inner));
    let two_emb = 2.0 * embracing;
    let two_emb_ok = abs(two_emb) > TINY;
    let two_emb_denom = select(1.0, two_emb, two_emb_ok);
    let ratio = sep / two_emb_denom;
    let final_raw = 1.0 + ratio;
    let final_ok = abs(final_raw) > TINY;
    let final_denom = select(1.0, final_raw, final_ok);
    let f_embrace = f0 / final_denom;

    // At contact return F0 directly, else the embracing form.
    let magnitude = select(f_embrace, f0, in_contact);

    var out: Result;
    out.magnitude = select(0.0, magnitude, valid);
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
/// the two radii, the gap and the three bridge parameters, padded to `8` `f32`
/// words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    radius_a: f32,
    radius_b: f32,
    gap: f32,
    surface_tension: f32,
    contact_angle: f32,
    liquid_volume: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the force magnitude and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    magnitude: f32,
    valid: u32,
}

/// One force-at-gap query: the grain-pair radii, surface separation and the
/// pendular-bridge parameters (surface tension, contact angle, liquid volume).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryForceAtGapQuery {
    /// Radius of the first grain.
    pub radius_a: f32,
    /// Radius of the second grain.
    pub radius_b: f32,
    /// Surface separation (gap) between the grains.
    pub gap: f32,
    /// Liquid surface tension `gamma`.
    pub surface_tension: f32,
    /// Solid-liquid contact angle `theta`, in radians.
    pub contact_angle: f32,
    /// Liquid bridge volume `V`.
    pub liquid_volume: f32,
}

impl CapillaryForceAtGapQuery {
    /// Builds a query from the radii, gap and bridge parameters.
    #[must_use]
    pub fn new(
        radius_a: f32,
        radius_b: f32,
        gap: f32,
        surface_tension: f32,
        contact_angle: f32,
        liquid_volume: f32,
    ) -> CapillaryForceAtGapQuery {
        CapillaryForceAtGapQuery {
            radius_a,
            radius_b,
            gap,
            surface_tension,
            contact_angle,
            liquid_volume,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CapillaryBridgeModel::force_at_gap` output for that configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryForceAtGapResult {
    /// The attractive force magnitude when valid, else `0`.
    pub magnitude: f32,
    /// `1` when the query is valid (gap finite, within rupture, radii positive),
    /// else `0`.
    pub valid: u32,
}

/// Encodes one [`CapillaryForceAtGapQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CapillaryForceAtGapQuery) -> GpuQuery {
    GpuQuery {
        radius_a: q.radius_a,
        radius_b: q.radius_b,
        gap: q.gap,
        surface_tension: q.surface_tension,
        contact_angle: q.contact_angle,
        liquid_volume: q.liquid_volume,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CapillaryForceAtGapResult`].
fn decode_result(raw: &GpuResult) -> CapillaryForceAtGapResult {
    CapillaryForceAtGapResult {
        magnitude: raw.magnitude,
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

/// A compiled, reusable force-at-gap compute pipeline, twinning the `CPU`
/// golden `CapillaryBridgeModel::force_at_gap`.
pub struct GpuCapillaryForceAtGap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapillaryForceAtGap {
    /// Compiles the force-at-gap kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the `sqrt`,
    /// `cos` and `pow` builtins the closed form requires, so no optional device
    /// feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapillaryForceAtGap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap"),
            source: ShaderSource::Wgsl(CAPILLARY_FORCE_AT_GAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapillaryForceAtGap {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CapillaryForceAtGapResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `magnitude` scalar
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CapillaryForceAtGapQuery],
    ) -> Vec<CapillaryForceAtGapResult> {
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
            label: Some("prism_volumetric_capillary_force_at_gap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_bind_group"),
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
            label: Some("prism_volumetric_capillary_force_at_gap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capillary_force_at_gap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capillary_force_at_gap_pass"),
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
