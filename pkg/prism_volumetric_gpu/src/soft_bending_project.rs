//! `wgpu` compute twin of the three-particle `XPBD` bending-projection kernel
//! from the `CPU` golden
//! `prism_physics_core::soft::constraint::bending::project_bending`.
//!
//! A bending constraint couples three particles `a`, `center`, `b` and resists
//! the joint at `center` folding away from the midpoint `M = (a + b) / 2` of its
//! neighbours. The scalar constraint is `C = |center - M| - rest_offset`, with
//! gradients `+n` on `center` and `-n/2` on each of `a` and `b`, where
//! `n = (center - M) / |center - M|`. The compliant `XPBD` denominator is
//! therefore `w_center + (w_a + w_b) / 4 + alpha_tilde`. One thread projects one
//! independent constraint and writes the three updated positions plus the
//! updated Lagrange multiplier.
//!
//! [`GpuSoftBendingProject`] is the on-device twin; a passing real-device parity
//! test is direct evidence the ported kernel reproduces the same denominator,
//! the same midpoint and normal, the same multiplier update and the same two
//! degeneracy rejections the reference computes, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate: the
//! inverse-mass denominator `denom_mass = w_center + 0.25 * (w_a + w_b)`; the
//! all-pinned rejection (`denom_mass <= 0`, pass-through, `valid = 0`); the
//! midpoint `M = (a + b) * 0.5`, the offset `delta = center - M` and its length;
//! the coincident rejection (`length < EPSILON`, pass-through, `valid = 0`); the
//! normal `delta / length`; the constraint value `c = length - rest_offset`; the
//! compliance term `alpha_tilde = compliance / (dt * dt)`; the multiplier delta
//! `delta_lambda = (-c - alpha_tilde * lambda) / (denom_mass + alpha_tilde)`; the
//! three position updates (`center += n * (delta_lambda * w_center)`,
//! `a -= n * (delta_lambda * w_a * 0.5)`, `b -= n * (delta_lambda * w_b * 0.5)`);
//! and the returned `lambda + delta_lambda`. There is no loop: each thread
//! performs a fixed, bounded sequence of multiplies, adds, divides and selects,
//! so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The updated positions and multiplier thread through subtracts, divides and
//! multiplies, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous channel. The discrete `valid` flag is compared exactly: it
//! is `1` only when the constraint projects (a positive denominator and a
//! non-degenerate offset) and `0` for either rejection, in which case the three
//! positions and the multiplier pass through unchanged.
//!
//! # Degenerate inputs
//!
//! When every endpoint is pinned the denominator `denom_mass` is zero or
//! negative and the constraint is inert (`valid = 0`, pass-through). When
//! `center` sits on the midpoint the offset length falls below
//! `EPSILON = 1.1920929e-7` and the normal is undefined, so the constraint is
//! inert (`valid = 0`, pass-through). Both divisors are guarded with a unit
//! fallback so the unselected arm cannot raise an infinity before the valid gate
//! drops it. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `clamp`, `select`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! `round`, no float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The degeneracy tests use ordered comparisons
//! feeding `select`, which are robust under `Metal`'s fast-math.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::bending`；无第三方
//! 引擎源码或衍生代码。
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

/// The portable core-`WGSL` three-particle bending-projection kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `project_bending`; see the module
/// documentation.
const SOFT_BENDING_PROJECT_WGSL: &str = r#"
// Three-particle XPBD bending projection twin: one thread projects one
// independent constraint, resolving the fold at `center` toward the midpoint of
// its neighbours by one compliant Lagrange step, and writes the three updated
// positions plus the updated multiplier. It mirrors the CPU golden exactly and
// uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Position of neighbour particle `a`.
    pax: f32,
    pay: f32,
    paz: f32,
    // Position of the joint particle `center`.
    pcx: f32,
    pcy: f32,
    pcz: f32,
    // Position of neighbour particle `b`.
    pbx: f32,
    pby: f32,
    pbz: f32,
    // Inverse masses of `a`, `center`, `b`.
    wa: f32,
    wc: f32,
    wb: f32,
    // Rest offset of `center` from the midpoint of `a` and `b`.
    rest_offset: f32,
    // Compliance (inverse stiffness); 0 is perfectly rigid.
    compliance: f32,
    // Accumulated Lagrange multiplier for the current substep.
    lambda: f32,
    // Substep duration.
    dt: f32,
}

struct Soln {
    // Updated position of neighbour particle `a`.
    new_pax: f32,
    new_pay: f32,
    new_paz: f32,
    // Updated position of the joint particle `center`.
    new_pcx: f32,
    new_pcy: f32,
    new_pcz: f32,
    // Updated position of neighbour particle `b`.
    new_pbx: f32,
    new_pby: f32,
    new_pbz: f32,
    // Updated Lagrange multiplier.
    new_lambda: f32,
    // 1 when the constraint projected, 0 when it was inert.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Soln>;

// Offset length below which the normal is undefined, matching the golden
// crate::math::scalar::EPSILON.
const EPSILON: f32 = 1.1920929e-7;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let pa = vec3<f32>(q.pax, q.pay, q.paz);
    let pc = vec3<f32>(q.pcx, q.pcy, q.pcz);
    let pb = vec3<f32>(q.pbx, q.pby, q.pbz);

    // Gradient magnitudes: |grad center| = 1, |grad a| = |grad b| = 1/2.
    let denom_mass = q.wc + 0.25 * (q.wa + q.wb);
    let mass_ok = denom_mass > 0.0;

    let midpoint = (pa + pb) * 0.5;
    let delta = pc - midpoint;
    let length = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    let length_ok = length >= EPSILON;

    // Guard the normal divisor so the unselected arm cannot raise an infinity
    // before the valid gate drops it.
    let safe_length = select(1.0, length, length_ok);
    let normal = delta / safe_length;

    let c = length - q.rest_offset;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let delta_lambda = (-c - alpha_tilde * q.lambda) / (denom_mass + alpha_tilde);

    let new_pc = pc + normal * (delta_lambda * q.wc);
    let new_pa = pa - normal * (delta_lambda * q.wa * 0.5);
    let new_pb = pb - normal * (delta_lambda * q.wb * 0.5);
    let new_lambda = q.lambda + delta_lambda;

    let accepted = mass_ok && length_ok;

    // Pass through the inputs unchanged on either rejection.
    let out_pa = select(pa, new_pa, accepted);
    let out_pc = select(pc, new_pc, accepted);
    let out_pb = select(pb, new_pb, accepted);
    let out_lambda = select(q.lambda, new_lambda, accepted);

    var out: Soln;
    out.new_pax = out_pa.x;
    out.new_pay = out_pa.y;
    out.new_paz = out_pa.z;
    out.new_pcx = out_pc.x;
    out.new_pcy = out_pc.y;
    out.new_pcz = out_pc.z;
    out.new_pbx = out_pb.x;
    out.new_pby = out_pb.y;
    out.new_pbz = out_pb.z;
    out.new_lambda = out_lambda;
    out.valid = select(0u, 1u, accepted);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
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
/// Sixteen `f32` give a fixed `64`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `64` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    pax: f32,
    pay: f32,
    paz: f32,
    pcx: f32,
    pcy: f32,
    pcz: f32,
    pbx: f32,
    pby: f32,
    pbz: f32,
    wa: f32,
    wc: f32,
    wb: f32,
    rest_offset: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Soln` struct.
/// Ten `f32` plus one `u32` give a fixed `44`-byte stride with no pad, since the
/// struct alignment is `4` and `44` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_pax: f32,
    new_pay: f32,
    new_paz: f32,
    new_pcx: f32,
    new_pcy: f32,
    new_pcz: f32,
    new_pbx: f32,
    new_pby: f32,
    new_pbz: f32,
    new_lambda: f32,
    valid: u32,
}

/// One query for the three-particle bending-projection twin: the three particle
/// positions, their inverse masses, the rest offset, compliance, the current
/// Lagrange multiplier and the substep duration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftBendingProjectQuery {
    /// `x` of neighbour particle `a`.
    pub pax: f32,
    /// `y` of neighbour particle `a`.
    pub pay: f32,
    /// `z` of neighbour particle `a`.
    pub paz: f32,
    /// `x` of the joint particle `center`.
    pub pcx: f32,
    /// `y` of the joint particle `center`.
    pub pcy: f32,
    /// `z` of the joint particle `center`.
    pub pcz: f32,
    /// `x` of neighbour particle `b`.
    pub pbx: f32,
    /// `y` of neighbour particle `b`.
    pub pby: f32,
    /// `z` of neighbour particle `b`.
    pub pbz: f32,
    /// Inverse mass of neighbour particle `a`.
    pub wa: f32,
    /// Inverse mass of the joint particle `center`.
    pub wc: f32,
    /// Inverse mass of neighbour particle `b`.
    pub wb: f32,
    /// Rest offset of `center` from the midpoint of `a` and `b`.
    pub rest_offset: f32,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: f32,
    /// Accumulated Lagrange multiplier for the current substep.
    pub lambda: f32,
    /// Substep duration.
    pub dt: f32,
}

impl SoftBendingProjectQuery {
    /// Builds a query from the three particle positions (`a`, `center`, `b`),
    /// their inverse masses in the same order, the rest offset, compliance, the
    /// current multiplier and the substep duration.
    ///
    /// Positions and inverse masses are grouped into fixed-length arrays so the
    /// constructor stays within a small, readable argument count.
    #[must_use]
    pub fn new(
        positions: [[f32; 3]; 3],
        inverse_masses: [f32; 3],
        rest_offset: f32,
        compliance: f32,
        lambda: f32,
        dt: f32,
    ) -> SoftBendingProjectQuery {
        let [pa, pc, pb] = positions;
        let [wa, wc, wb] = inverse_masses;
        SoftBendingProjectQuery {
            pax: pa[0],
            pay: pa[1],
            paz: pa[2],
            pcx: pc[0],
            pcy: pc[1],
            pcz: pc[2],
            pbx: pb[0],
            pby: pb[1],
            pbz: pb[2],
            wa,
            wc,
            wb,
            rest_offset,
            compliance,
            lambda,
            dt,
        }
    }
}

/// One resolved answer for a single query: the three updated positions, the
/// updated Lagrange multiplier and the validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftBendingProjectResult {
    /// Updated `x` of neighbour particle `a`.
    pub new_pax: f32,
    /// Updated `y` of neighbour particle `a`.
    pub new_pay: f32,
    /// Updated `z` of neighbour particle `a`.
    pub new_paz: f32,
    /// Updated `x` of the joint particle `center`.
    pub new_pcx: f32,
    /// Updated `y` of the joint particle `center`.
    pub new_pcy: f32,
    /// Updated `z` of the joint particle `center`.
    pub new_pcz: f32,
    /// Updated `x` of neighbour particle `b`.
    pub new_pbx: f32,
    /// Updated `y` of neighbour particle `b`.
    pub new_pby: f32,
    /// Updated `z` of neighbour particle `b`.
    pub new_pbz: f32,
    /// Updated Lagrange multiplier.
    pub new_lambda: f32,
    /// `1` when the constraint projected, `0` when it was inert.
    pub valid: u32,
}

/// Encodes one [`SoftBendingProjectQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SoftBendingProjectQuery) -> GpuQuery {
    GpuQuery {
        pax: q.pax,
        pay: q.pay,
        paz: q.paz,
        pcx: q.pcx,
        pcy: q.pcy,
        pcz: q.pcz,
        pbx: q.pbx,
        pby: q.pby,
        pbz: q.pbz,
        wa: q.wa,
        wc: q.wc,
        wb: q.wb,
        rest_offset: q.rest_offset,
        compliance: q.compliance,
        lambda: q.lambda,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SoftBendingProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftBendingProjectResult {
    SoftBendingProjectResult {
        new_pax: raw.new_pax,
        new_pay: raw.new_pay,
        new_paz: raw.new_paz,
        new_pcx: raw.new_pcx,
        new_pcy: raw.new_pcy,
        new_pcz: raw.new_pcz,
        new_pbx: raw.new_pbx,
        new_pby: raw.new_pby,
        new_pbz: raw.new_pbz,
        new_lambda: raw.new_lambda,
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

/// A compiled, reusable three-particle bending-projection compute pipeline,
/// twinning the `CPU` golden `project_bending`.
pub struct GpuSoftBendingProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftBendingProject {
    /// Compiles the bending-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftBendingProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_bending_project"),
            source: ShaderSource::Wgsl(SOFT_BENDING_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_bending_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_bending_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_bending_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftBendingProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftBendingProjectResult`] per input, in order.
    ///
    /// The continuous channels match the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftBendingProjectQuery],
    ) -> Vec<SoftBendingProjectResult> {
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
            label: Some("prism_volumetric_soft_bending_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_bending_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_bending_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_bending_project_bind_group"),
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
            label: Some("prism_volumetric_soft_bending_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_bending_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_bending_project_pass"),
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
