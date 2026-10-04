//! `wgpu` compute twin of the isometric-bending projection from the `CPU`
//! golden `prism_physics_core::soft::constraint::isometric_bending`'s
//! `project_isometric_bending`.
//!
//! A cloth or thin-shell soft body resists folding by coupling the four
//! particles of a bending stencil and driving its weighted sum `S = Σ wᵢ·xᵢ`
//! back toward zero (the flat configuration). This module ports that single
//! stateless `XPBD` projection onto the device: one thread resolves one
//! stencil, so a passing real-device parity test is direct evidence the ported
//! kernel takes the same degenerate / projecting branch the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! The golden operates over raw particle indices; this twin is made
//! self-contained by passing the four stencil vertices directly (equivalent to
//! `vertices = [0, 1, 2, 3]`, so every `positions.get` succeeds). For each
//! query the kernel reproduces `project_isometric_bending` for a single stencil
//! with explicit positions `p0..p3`, inverse masses `w0..w3`, cotangent weights
//! `wt0..wt3`, the bending `scale`, `compliance` and the substep `dt`:
//!
//! * `dt <= 0` is a no-op.
//! * `S = p0·wt0 + p1·wt1 + p2·wt2 + p3·wt3`; `s_len_sq = dot(S, S)`.
//! * `s_len_sq <= EPS_LEN_SQ` (a flat / degenerate stencil) is a no-op.
//! * `energy = ½·scale·s_len_sq`.
//! * `sum_w_grad = Σ_i ( wᵢ > 0 ? wᵢ·(scale·wtᵢ)²·s_len_sq : 0 )`.
//! * `alpha_tilde = compliance / (dt·dt)`; `denom = sum_w_grad + alpha_tilde`.
//! * `denom <= 0` is a no-op.
//! * `d_lambda = -energy / denom`; each free corner (`wᵢ > 0`) moves by
//!   `S · (wᵢ·d_lambda·scale·wtᵢ)` while pinned corners stay put.
//!
//! There is no loop: each thread performs a fixed, bounded sequence of
//! multiplies, adds, a dot product and a guarded division, so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, a dot product and a guarded
//! division, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous output. The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! Three branches collapse to the same no-op — `dt <= 0`, a flat stencil whose
//! `s_len_sq <= EPS_LEN_SQ`, and `denom <= 0` (a fully pinned stencil) — in
//! which case the twin reports `valid = 0` and echoes the input positions
//! unchanged. Fixtures and the sweep keep configurations away from the
//! `s_len_sq == EPS_LEN_SQ` and `denom == 0` surfaces so a last-bit difference
//! cannot flip the branch. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Every
//! comparison is ordered; there is no `f32` equality, and the guarded division
//! substitutes a safe denominator on the no-op arm so no `inf` / `nan` is
//! produced.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::isometric_bending`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` isometric-bending projection kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `project_isometric_bending` branch for
/// branch; see the module documentation for the algorithm.
const SOFT_ISOMETRIC_BENDING_PROJECT_WGSL: &str = r#"
// Isometric-bending projection twin: one thread per bending stencil reproduces
// project_isometric_bending. It mirrors the CPU golden branch for branch, uses
// only the portable core-WGSL subset (dot/select and + - * / plus unsigned
// index math), takes no optional feature, and has no loop, so it provably
// terminates. The bending stiffness scale is named bend_scale to avoid any
// reserved-identifier risk.

const EPS_LEN_SQ: f32 = 1e-12;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Stencil positions, world space.
    p0x: f32, p0y: f32, p0z: f32,
    p1x: f32, p1y: f32, p1z: f32,
    p2x: f32, p2y: f32, p2z: f32,
    p3x: f32, p3y: f32, p3z: f32,
    // Inverse masses of the four stencil vertices (zero pins a vertex).
    w0: f32, w1: f32, w2: f32, w3: f32,
    // Cotangent bending weights.
    wt0: f32, wt1: f32, wt2: f32, wt3: f32,
    // Bending stiffness scale.
    bend_scale: f32,
    // Compliance (inverse stiffness); already non-negative.
    compliance: f32,
    // Substep timestep.
    dt: f32,
}

struct Result {
    // Updated stencil positions.
    n0x: f32, n0y: f32, n0z: f32,
    n1x: f32, n1y: f32, n1z: f32,
    n2x: f32, n2y: f32, n2z: f32,
    n3x: f32, n3y: f32, n3z: f32,
    // 1 when the projection ran, 0 for a degenerate no-op branch.
    valid: u32,
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
    let p0 = vec3<f32>(q.p0x, q.p0y, q.p0z);
    let p1 = vec3<f32>(q.p1x, q.p1y, q.p1z);
    let p2 = vec3<f32>(q.p2x, q.p2y, q.p2z);
    let p3 = vec3<f32>(q.p3x, q.p3y, q.p3z);

    // S = Sum wt_i * p_i (every stencil vertex contributes, including pinned).
    let s = p0 * q.wt0 + p1 * q.wt1 + p2 * q.wt2 + p3 * q.wt3;
    let s_len_sq = dot(s, s);
    let energy = 0.5 * q.bend_scale * s_len_sq;

    // Mass-weighted gradient denominator; grad_i = bend_scale * wt_i * S, so
    // |grad_i|^2 = (bend_scale*wt_i)^2 * s_len_sq. Pinned (w_i <= 0) contribute
    // nothing, matched with an ordered compare + select.
    let g0 = q.bend_scale * q.wt0;
    let g1 = q.bend_scale * q.wt1;
    let g2 = q.bend_scale * q.wt2;
    let g3 = q.bend_scale * q.wt3;
    let t0 = select(0.0, q.w0 * g0 * g0 * s_len_sq, q.w0 > 0.0);
    let t1 = select(0.0, q.w1 * g1 * g1 * s_len_sq, q.w1 > 0.0);
    let t2 = select(0.0, q.w2 * g2 * g2 * s_len_sq, q.w2 > 0.0);
    let t3 = select(0.0, q.w3 * g3 * g3 * s_len_sq, q.w3 > 0.0);
    let sum_w_grad = t0 + t1 + t2 + t3;

    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let denom = sum_w_grad + alpha_tilde;

    // All three no-op branches collapse to valid == false: dt <= 0, a flat
    // stencil (s_len_sq <= EPS_LEN_SQ) or a non-positive denominator.
    let run = (q.dt > 0.0) && (s_len_sq > EPS_LEN_SQ) && (denom > 0.0);

    // Guard the division so the unselected arm never forms inf / nan.
    let safe_denom = select(1.0, denom, denom > 0.0);
    let d_lambda = -energy / safe_denom;

    // Per-vertex displacement, zero on the no-op arm or for pinned vertices.
    let d0 = select(vec3<f32>(0.0, 0.0, 0.0), s * (q.w0 * d_lambda * g0), run && (q.w0 > 0.0));
    let d1 = select(vec3<f32>(0.0, 0.0, 0.0), s * (q.w1 * d_lambda * g1), run && (q.w1 > 0.0));
    let d2 = select(vec3<f32>(0.0, 0.0, 0.0), s * (q.w2 * d_lambda * g2), run && (q.w2 > 0.0));
    let d3 = select(vec3<f32>(0.0, 0.0, 0.0), s * (q.w3 * d_lambda * g3), run && (q.w3 > 0.0));

    let np0 = p0 + d0;
    let np1 = p1 + d1;
    let np2 = p2 + d2;
    let np3 = p3 + d3;

    var out: Result;
    out.n0x = np0.x; out.n0y = np0.y; out.n0z = np0.z;
    out.n1x = np1.x; out.n1y = np1.y; out.n1z = np1.z;
    out.n2x = np2.x; out.n2y = np2.y; out.n2z = np2.z;
    out.n3x = np3.x; out.n3y = np3.y; out.n3z = np3.z;
    out.valid = select(0u, 1u, run);
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
/// Every `vec3` input is flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`. The
/// struct is `23` `f32` words (`92` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    p0x: f32,
    p0y: f32,
    p0z: f32,
    p1x: f32,
    p1y: f32,
    p1z: f32,
    p2x: f32,
    p2y: f32,
    p2z: f32,
    p3x: f32,
    p3y: f32,
    p3z: f32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    wt0: f32,
    wt1: f32,
    wt2: f32,
    wt3: f32,
    bend_scale: f32,
    compliance: f32,
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the four updated stencil positions and the validity flag — `13`
/// words (`52` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    n0x: f32,
    n0y: f32,
    n0z: f32,
    n1x: f32,
    n1y: f32,
    n1z: f32,
    n2x: f32,
    n2y: f32,
    n2z: f32,
    n3x: f32,
    n3y: f32,
    n3z: f32,
    valid: u32,
}

/// One isometric-bending projection query: the four stencil positions, their
/// inverse masses, the cotangent bending weights, the bending scale, compliance
/// and the substep timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftIsometricBendingProjectQuery {
    /// Stencil positions `p0..p3`, world space.
    pub positions: [[f32; 3]; 4],
    /// Inverse masses `w0..w3`; a non-positive value pins the corresponding
    /// vertex.
    pub inverse_masses: [f32; 4],
    /// Cotangent bending weights `wt0..wt3`.
    pub weights: [f32; 4],
    /// Bending stiffness scale.
    pub scale: f32,
    /// Compliance (inverse stiffness); expected non-negative.
    pub compliance: f32,
    /// Substep timestep; a non-positive value is a no-op.
    pub dt: f32,
}

impl SoftIsometricBendingProjectQuery {
    /// Builds a query from the four stencil vertices, inverse masses, bending
    /// weights, scale, compliance and timestep.
    #[must_use]
    pub fn new(
        positions: [[f32; 3]; 4],
        inverse_masses: [f32; 4],
        weights: [f32; 4],
        scale: f32,
        compliance: f32,
        dt: f32,
    ) -> SoftIsometricBendingProjectQuery {
        SoftIsometricBendingProjectQuery {
            positions,
            inverse_masses,
            weights,
            scale,
            compliance,
            dt,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `project_isometric_bending` output for that stencil.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftIsometricBendingProjectResult {
    /// The updated stencil positions: the projected values when the pass ran,
    /// otherwise the input positions echoed unchanged.
    pub positions: [[f32; 3]; 4],
    /// `1` when the projection ran, `0` for a degenerate no-op branch
    /// (`dt <= 0`, a flat stencil, or `denom <= 0`).
    pub valid: u32,
}

/// Encodes one [`SoftIsometricBendingProjectQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &SoftIsometricBendingProjectQuery) -> GpuQuery {
    GpuQuery {
        p0x: q.positions[0][0],
        p0y: q.positions[0][1],
        p0z: q.positions[0][2],
        p1x: q.positions[1][0],
        p1y: q.positions[1][1],
        p1z: q.positions[1][2],
        p2x: q.positions[2][0],
        p2y: q.positions[2][1],
        p2z: q.positions[2][2],
        p3x: q.positions[3][0],
        p3y: q.positions[3][1],
        p3z: q.positions[3][2],
        w0: q.inverse_masses[0],
        w1: q.inverse_masses[1],
        w2: q.inverse_masses[2],
        w3: q.inverse_masses[3],
        wt0: q.weights[0],
        wt1: q.weights[1],
        wt2: q.weights[2],
        wt3: q.weights[3],
        bend_scale: q.scale,
        compliance: q.compliance,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftIsometricBendingProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftIsometricBendingProjectResult {
    SoftIsometricBendingProjectResult {
        positions: [
            [raw.n0x, raw.n0y, raw.n0z],
            [raw.n1x, raw.n1y, raw.n1z],
            [raw.n2x, raw.n2y, raw.n2z],
            [raw.n3x, raw.n3y, raw.n3z],
        ],
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

/// A compiled, reusable isometric-bending projection compute pipeline,
/// twinning the `CPU` golden `project_isometric_bending`.
pub struct GpuSoftIsometricBendingProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftIsometricBendingProject {
    /// Compiles the isometric-bending projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftIsometricBendingProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project"),
            source: ShaderSource::Wgsl(SOFT_ISOMETRIC_BENDING_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftIsometricBendingProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftIsometricBendingProjectResult`] per input, in order.
    ///
    /// Each continuous output matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftIsometricBendingProjectQuery],
    ) -> Vec<SoftIsometricBendingProjectResult> {
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
            label: Some("prism_volumetric_soft_isometric_bending_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_bind_group"),
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
            label: Some("prism_volumetric_soft_isometric_bending_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_isometric_bending_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_isometric_bending_project_pass"),
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
