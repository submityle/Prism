//! `wgpu` compute twin of the stateless isometric-bending `XPBD` projection
//! inside the cloth bending contract
//! ([`project_bending`](prism_render_architecture::cloth::bending::project_bending)).
//!
//! The `CPU` golden
//! [`project_bending`](prism_render_architecture::cloth::bending::project_bending)
//! delegates to
//! [`project_isometric_bending`](prism_render_architecture::cloth::bending)
//! and performs one compliant `XPBD` step over a four-vertex hinge stencil
//! `[edge0, edge1, apex_a, apex_b]`: it accumulates the bend vector
//! `S = Σ wᵢ · xᵢ`, forms the energy `E = ½ · scale · |S|²`, and distributes a
//! mass-weighted correction `Δxᵢ = inv_massᵢ · Δλ · scale · wᵢ · S` to the free
//! particles, returning the pre-step energy.
//!
//! [`GpuClothBendingProject`] is the on-device twin of that closed-form step.
//! One thread solves one hinge: it reproduces the reference's `S` accumulation,
//! the flat/degenerate short-circuit, the pinned/compliance denominator guard,
//! and the per-vertex position update, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same correction the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one hinge the twin reproduces, per query: the four updated particle
//! positions and the scalar pre-step `energy`. The host pre-resolves the hinge
//! `vertices` into four stencil slots, each carrying a resolved `position`, an
//! `inverse_mass`, and a `valid` flag marking whether the index was in range —
//! matching the reference's `positions.get` / `unwrap_or(0.0)` out-of-range
//! handling. An in-range pinned slot (`inverse_mass <= 0`) still contributes to
//! `S` yet never moves, exactly as the reference does.
//!
//! # What stays on the host
//!
//! The variable-length particle slice, the `vertices` index resolution, the
//! structure-of-arrays conversion
//! ([`project_bending`](prism_render_architecture::cloth::bending::project_bending)
//! calls the physics-engine `to_soa` adapter), and any multi-constraint
//! Gauss-Seidel sweep remain host work; the device sees only four pre-resolved
//! stencil slots per hinge. Aliased stencils (one particle index in two slots)
//! are a host concern and are not enqueued: the twin models four distinct slots.
//!
//! # Correctness model
//!
//! The `energy` and the updated positions thread through only `+ - * /` with no
//! `sqrt` and no transcendental, so the `CPU` and `GPU` agree to within the
//! documented tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`). The
//! degenerate branches — `dt <= 0`, a flat stencil (`|S|² <= 1e-12`), and a
//! non-positive denominator (all-pinned, zero compliance) — are reproduced
//! exactly so a fixture that lands in any of them leaves the positions
//! unchanged just as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `+ - * /`, a
//! fixed four-iteration bounded loop and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`, no `sqrt`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::bending`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` isometric-bending projection kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`project_bending`](prism_render_architecture::cloth::bending::project_bending)
/// closed-form step; see the module documentation for the algorithm.
const CLOTH_BENDING_PROJECT_WGSL: &str = r#"
// Isometric-bending XPBD projection twin: one thread projects one four-vertex
// hinge stencil, mirroring the CPU golden `cloth::bending::project_bending`
// closed form with only max and + - * /. It owns no particle slice, no vertex
// index resolution and no multi-constraint sweep; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::bending；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of hinges in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Four stencil-slot positions, flattened as [x0,y0,z0, x1,y1,z1, ...].
    pos: array<f32, 12>,
    // Per-slot inverse mass; a non-positive value pins that slot.
    inv_mass: array<f32, 4>,
    // Per-slot bend-Laplacian weight aligned with the stencil.
    weight: array<f32, 4>,
    // Per-slot in-range flag (1 contributes to S, 0 is an out-of-range slot).
    valid: array<u32, 4>,
    // Area-derived energy scale.
    scale: f32,
    // XPBD compliance (clamped non-negative to mirror the reference value()).
    compliance: f32,
    // Substep time; a non-positive dt is a no-op.
    dt: f32,
    pad0: u32,
}

struct Result {
    // Four updated stencil-slot positions, flattened like Query.pos.
    out_pos: array<f32, 12>,
    // Pre-step bending energy E = 0.5 * scale * |S|^2.
    energy: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    var q = queries[idx];

    // Default: positions unchanged, zero energy. Overwritten below as branches
    // compute each quantity, matching the reference early returns.
    var out: Result;
    out.out_pos = q.pos;
    out.energy = 0.0;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    let compliance = max(q.compliance, 0.0);
    let dt = q.dt;

    // dt <= 0: the reference returns 0 and leaves positions untouched.
    if (dt <= 0.0) {
        results[idx] = out;
        return;
    }

    // S = Σ wᵢ · xᵢ over in-range slots (pinned slots still contribute).
    var sx = 0.0;
    var sy = 0.0;
    var sz = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (q.valid[i] != 0u) {
            let w = q.weight[i];
            let base = i * 3u;
            sx = sx + w * q.pos[base];
            sy = sy + w * q.pos[base + 1u];
            sz = sz + w * q.pos[base + 2u];
        }
    }
    let s_len_sq = sx * sx + sy * sy + sz * sz;

    // Flat or degenerate stencil (|S|^2 <= EPS_LEN_SQ): no-op, zero energy.
    if (s_len_sq <= 0.000000000001) {
        results[idx] = out;
        return;
    }
    let energy = 0.5 * q.scale * s_len_sq;
    out.energy = energy;

    // Denominator Σ inv_massᵢ · |gradᵢ|² (gradᵢ = scale·wᵢ·S) over free slots.
    var sum_w_grad = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let inv_mass = q.inv_mass[i];
        if (inv_mass > 0.0) {
            let grad_scalar = q.scale * q.weight[i];
            sum_w_grad = sum_w_grad + inv_mass * grad_scalar * grad_scalar * s_len_sq;
        }
    }
    let alpha_tilde = compliance / (dt * dt);
    let denom = sum_w_grad + alpha_tilde;

    // Non-positive denominator (all-pinned, zero compliance): energy returned,
    // positions unchanged.
    if (denom <= 0.0) {
        results[idx] = out;
        return;
    }
    let d_lambda = -energy / denom;

    // Δxᵢ = inv_massᵢ · Δλ · scale · wᵢ · S for the free slots only; S is the
    // original accumulated vector, computed once above.
    for (var i = 0u; i < 4u; i = i + 1u) {
        let inv_mass = q.inv_mass[i];
        if (inv_mass > 0.0) {
            let coeff = inv_mass * d_lambda * q.scale * q.weight[i];
            let base = i * 3u;
            out.out_pos[base] = q.pos[base] + sx * coeff;
            out.out_pos[base + 1u] = q.pos[base + 1u] + sy * coeff;
            out.out_pos[base + 2u] = q.pos[base + 2u] + sz * coeff;
        }
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the hinge count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`CLOTH_BENDING_PROJECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid hinges in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one hinge query, matching the `WGSL` `Query`
/// struct: four flattened slot positions, per-slot inverse masses, weights and
/// in-range flags, the `scale`, `compliance` and `dt`, plus one pad word to a
/// `112`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Four stencil-slot positions flattened as `[x0, y0, z0, x1, ...]`.
    pos: [f32; 12],
    /// Per-slot inverse mass; non-positive pins the slot.
    inv_mass: [f32; 4],
    /// Per-slot bend-Laplacian weight.
    weight: [f32; 4],
    /// Per-slot in-range flag (`1` contributes to `S`, `0` out-of-range).
    valid: [u32; 4],
    /// Area-derived energy scale.
    scale: f32,
    /// `XPBD` compliance (clamped non-negative on device).
    compliance: f32,
    /// Substep time.
    dt: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one hinge result, matching the `WGSL` `Result`
/// struct: four flattened updated positions, the pre-step `energy`, and three
/// pad words to a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Four updated stencil-slot positions flattened like `GpuQuery::pos`.
    out_pos: [f32; 12],
    /// Pre-step bending energy `0.5 * scale * |S|^2`.
    energy: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One hinge query for the isometric-bending projection twin: four pre-resolved
/// stencil slots plus the hinge `scale`, `compliance` and substep `dt`.
///
/// The host resolves the reference hinge `vertices` into four slots. Each slot
/// carries its world-space `positions`, its `inverse_masses` (non-positive pins
/// the slot), and a `valid` flag that is `false` for an out-of-range vertex
/// index — reproducing the reference's `positions.get` / `unwrap_or(0.0)`
/// out-of-range handling. `weights` are the bend-Laplacian stencil aligned with
/// the slots. An in-range pinned slot still contributes to the bend vector `S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothBendingProjectQuery {
    /// The four stencil-slot world positions `[edge0, edge1, apex_a, apex_b]`.
    pub positions: [[f32; 3]; 4],
    /// Per-slot inverse mass; a non-positive value pins the slot.
    pub inverse_masses: [f32; 4],
    /// Per-slot in-range flag; `false` is an out-of-range vertex index.
    pub valid: [bool; 4],
    /// Per-slot bend-Laplacian weights aligned with `positions`.
    pub weights: [f32; 4],
    /// Area-derived energy `scale`.
    pub scale: f32,
    /// `XPBD` compliance; clamped non-negative to mirror the reference.
    pub compliance: f32,
    /// Substep time `dt`; a non-positive value makes the projection a no-op.
    pub dt: f32,
}

/// One resolved hinge projection, mirroring the reference
/// [`project_bending`](prism_render_architecture::cloth::bending::project_bending)
/// outcome: the four updated stencil-slot positions and the pre-step `energy`.
///
/// `positions` holds the solved slot positions; out-of-range or pinned slots are
/// returned unchanged. `energy` is the pre-step bending energy
/// `0.5 * scale * |S|²` the reference returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothBendingProjectResult {
    /// The four updated stencil-slot world positions.
    pub positions: [[f32; 3]; 4],
    /// The pre-step bending energy `0.5 * scale * |S|²`.
    pub energy: f32,
}

/// Encodes one [`ClothBendingProjectQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothBendingProjectQuery) -> GpuQuery {
    let mut pos = [0.0f32; 12];
    for (slot, p) in q.positions.iter().enumerate() {
        let base = slot * 3;
        pos[base] = p[0];
        pos[base + 1] = p[1];
        pos[base + 2] = p[2];
    }
    let mut valid = [0u32; 4];
    for (word, flag) in valid.iter_mut().zip(q.valid.iter()) {
        *word = u32::from(*flag);
    }
    GpuQuery {
        pos,
        inv_mass: q.inverse_masses,
        weight: q.weights,
        valid,
        scale: q.scale,
        compliance: q.compliance,
        dt: q.dt,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothBendingProjectResult`],
/// unflattening the four slot positions.
fn decode_result(raw: &GpuResult) -> ClothBendingProjectResult {
    let mut positions = [[0.0f32; 3]; 4];
    for (slot, out) in positions.iter_mut().enumerate() {
        let base = slot * 3;
        out[0] = raw.out_pos[base];
        out[1] = raw.out_pos[base + 1];
        out[2] = raw.out_pos[base + 2];
    }
    ClothBendingProjectResult {
        positions,
        energy: raw.energy,
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

/// A compiled, reusable isometric-bending projection compute pipeline, twinning
/// the `CPU` golden
/// [`project_bending`](prism_render_architecture::cloth::bending::project_bending).
pub struct GpuClothBendingProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothBendingProject {
    /// Compiles the isometric-bending projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothBendingProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_bending_project"),
            source: ShaderSource::Wgsl(CLOTH_BENDING_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothBendingProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every hinge in `queries` and returns one
    /// [`ClothBendingProjectResult`] per input, in order.
    ///
    /// The updated positions and the `energy` equal the reference to within the
    /// tolerance documented on this module. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothBendingProjectQuery],
    ) -> Vec<ClothBendingProjectResult> {
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
            label: Some("prism_volumetric_cloth_bending_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_bind_group"),
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
            label: Some("prism_volumetric_cloth_bending_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_bending_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_bending_project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per hinge, flattened to a 1-D dispatch.
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
