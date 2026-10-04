//! `wgpu` compute twin of the tetrahedral volume-preservation projection from
//! the `CPU` golden `prism_physics_core::soft::constraint::volume`'s
//! `TetraVolumeConstraint::project`.
//!
//! A volumetric soft body resists squashing and inflation by coupling the four
//! particles of each tetrahedron and driving its signed volume back to a rest
//! value. This module ports that single stateless `XPBD` projection onto the
//! device: one thread resolves one tetrahedron, so a passing real-device parity
//! test is direct evidence the ported kernel takes the same degenerate /
//! projecting branch the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `TetraVolumeConstraint::project` for a
//! single tetrahedron with explicit corner positions `p0..p3`, inverse masses
//! `w0..w3`, `rest_volume`, `compliance`, the accumulated multiplier `lambda`
//! and the substep `dt`. With `e1 = p1 - p0`, `e2 = p2 - p0`, `e3 = p3 - p0`
//! the volume gradients are `grad1 = cross(e2, e3) / 6`,
//! `grad2 = cross(e3, e1) / 6`, `grad3 = cross(e1, e2) / 6` and
//! `grad0 = -(grad1 + grad2 + grad3)`. The effective mass is
//! `denom_mass = w0 |grad0|^2 + w1 |grad1|^2 + w2 |grad2|^2 + w3 |grad3|^2`
//! (each squared length via `dot(v, v)`). When `denom_mass <= 0` the projection
//! is a no-op. Otherwise `volume = dot(e1, cross(e2, e3)) / 6`,
//! `c = volume - rest_volume`, `alpha_tilde = compliance / (dt * dt)`,
//! `delta_lambda = (-c - alpha_tilde * lambda) / (denom_mass + alpha_tilde)`,
//! `new_lambda = lambda + delta_lambda` and each corner moves by
//! `grad_i * (delta_lambda * w_i)`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, cross products and dot products, so
//! the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, cross products, dot
//! products and a guarded division, so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate, and the
//! gradient magnitudes can be large. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous output. The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A tetrahedron whose effective mass `denom_mass` is at or below zero — a
//! fully pinned cell (`w0..w3` all zero) or a collapsed / coplanar shape with
//! vanishing gradients — is a no-op: the twin reports `valid = 0`, echoes the
//! input positions and leaves `lambda` unchanged. Fixtures and the sweep keep
//! configurations away from the `denom_mass == 0` surface so a last-bit
//! difference cannot flip the branch. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `cross`, `dot`,
//! `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every comparison is ordered; there is no `f32` equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::volume`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` tetrahedral volume-projection kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `TetraVolumeConstraint::project` branch for
/// branch; see the module documentation for the algorithm.
const SOFT_TETRA_VOLUME_PROJECT_WGSL: &str = r#"
// Tetrahedral volume-preservation twin: one thread per tetrahedron reproduces
// TetraVolumeConstraint::project. It mirrors the CPU golden branch for branch,
// uses only the portable core-WGSL subset (cross/dot/select/sqrt and + - * /
// plus unsigned index math), takes no optional feature, and has no loop, so it
// provably terminates.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Corner positions, world space.
    p0x: f32, p0y: f32, p0z: f32,
    p1x: f32, p1y: f32, p1z: f32,
    p2x: f32, p2y: f32, p2z: f32,
    p3x: f32, p3y: f32, p3z: f32,
    // Inverse masses of the four corners (zero pins a corner).
    w0: f32, w1: f32, w2: f32, w3: f32,
    // Target signed volume.
    rest_volume: f32,
    // Compliance (inverse stiffness); already non-negative.
    compliance: f32,
    // Accumulated Lagrange multiplier for the current substep.
    lambda: f32,
    // Substep timestep.
    dt: f32,
}

struct Result {
    // Updated corner positions.
    n0x: f32, n0y: f32, n0z: f32,
    n1x: f32, n1y: f32, n1z: f32,
    n2x: f32, n2y: f32, n2z: f32,
    n3x: f32, n3y: f32, n3z: f32,
    // Updated Lagrange multiplier.
    new_lambda: f32,
    // 1 when the projection ran, 0 for the degenerate no-op branch.
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

    // Default: a no-op that echoes the input positions and lambda, valid = 0.
    var out: Result;
    out.n0x = p0.x; out.n0y = p0.y; out.n0z = p0.z;
    out.n1x = p1.x; out.n1y = p1.y; out.n1z = p1.z;
    out.n2x = p2.x; out.n2y = p2.y; out.n2z = p2.z;
    out.n3x = p3.x; out.n3y = p3.y; out.n3z = p3.z;
    out.new_lambda = q.lambda;
    out.valid = 0u;

    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let e3 = p3 - p0;
    let grad1 = cross(e2, e3) / 6.0;
    let grad2 = cross(e3, e1) / 6.0;
    let grad3 = cross(e1, e2) / 6.0;
    let grad0 = -(grad1 + grad2 + grad3);
    let denom_mass = q.w0 * dot(grad0, grad0)
        + q.w1 * dot(grad1, grad1)
        + q.w2 * dot(grad2, grad2)
        + q.w3 * dot(grad3, grad3);
    if (denom_mass <= 0.0) {
        results[idx] = out;
        return;
    }

    let volume = dot(e1, cross(e2, e3)) / 6.0;
    let c = volume - q.rest_volume;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let delta_lambda = (-c - alpha_tilde * q.lambda) / (denom_mass + alpha_tilde);
    out.new_lambda = q.lambda + delta_lambda;

    let np0 = p0 + grad0 * (delta_lambda * q.w0);
    let np1 = p1 + grad1 * (delta_lambda * q.w1);
    let np2 = p2 + grad2 * (delta_lambda * q.w2);
    let np3 = p3 + grad3 * (delta_lambda * q.w3);
    out.n0x = np0.x; out.n0y = np0.y; out.n0z = np0.z;
    out.n1x = np1.x; out.n1y = np1.y; out.n1z = np1.z;
    out.n2x = np2.x; out.n2y = np2.y; out.n2z = np2.z;
    out.n3x = np3.x; out.n3y = np3.y; out.n3z = np3.z;
    out.valid = 1u;
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
    rest_volume: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the four updated corner positions, the updated multiplier and the
/// validity flag — `14` words (`56` bytes).
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
    new_lambda: f32,
    valid: u32,
}

/// One tetrahedral volume-projection query: the four corner positions, their
/// inverse masses, the rest volume, compliance, the accumulated multiplier and
/// the substep timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftTetraVolumeProjectQuery {
    /// Corner positions `p0..p3`, world space.
    pub positions: [[f32; 3]; 4],
    /// Inverse masses `w0..w3`; a zero pins the corresponding corner.
    pub inverse_masses: [f32; 4],
    /// Target signed volume.
    pub rest_volume: f32,
    /// Compliance (inverse stiffness); expected non-negative.
    pub compliance: f32,
    /// Accumulated Lagrange multiplier for the current substep.
    pub lambda: f32,
    /// Substep timestep (expected strictly positive).
    pub dt: f32,
}

impl SoftTetraVolumeProjectQuery {
    /// Builds a query from the four corners, inverse masses, rest volume,
    /// compliance, multiplier and timestep.
    #[must_use]
    pub fn new(
        positions: [[f32; 3]; 4],
        inverse_masses: [f32; 4],
        rest_volume: f32,
        compliance: f32,
        lambda: f32,
        dt: f32,
    ) -> SoftTetraVolumeProjectQuery {
        SoftTetraVolumeProjectQuery {
            positions,
            inverse_masses,
            rest_volume,
            compliance,
            lambda,
            dt,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `TetraVolumeConstraint::project` output for that tetrahedron.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftTetraVolumeProjectResult {
    /// The updated corner positions: the projected values when the pass ran,
    /// otherwise the input positions echoed unchanged.
    pub positions: [[f32; 3]; 4],
    /// The updated Lagrange multiplier, unchanged on the no-op branch.
    pub new_lambda: f32,
    /// `1` when the projection ran, `0` for the degenerate no-op branch
    /// (`denom_mass <= 0`).
    pub valid: u32,
}

/// Encodes one [`SoftTetraVolumeProjectQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SoftTetraVolumeProjectQuery) -> GpuQuery {
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
        rest_volume: q.rest_volume,
        compliance: q.compliance,
        lambda: q.lambda,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftTetraVolumeProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftTetraVolumeProjectResult {
    SoftTetraVolumeProjectResult {
        positions: [
            [raw.n0x, raw.n0y, raw.n0z],
            [raw.n1x, raw.n1y, raw.n1z],
            [raw.n2x, raw.n2y, raw.n2z],
            [raw.n3x, raw.n3y, raw.n3z],
        ],
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

/// A compiled, reusable tetrahedral volume-projection compute pipeline,
/// twinning the `CPU` golden `TetraVolumeConstraint::project`.
pub struct GpuSoftTetraVolumeProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftTetraVolumeProject {
    /// Compiles the tetrahedral volume-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftTetraVolumeProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project"),
            source: ShaderSource::Wgsl(SOFT_TETRA_VOLUME_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftTetraVolumeProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftTetraVolumeProjectResult`] per input, in order.
    ///
    /// Each continuous output matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftTetraVolumeProjectQuery],
    ) -> Vec<SoftTetraVolumeProjectResult> {
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
            label: Some("prism_volumetric_soft_tetra_volume_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_bind_group"),
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
            label: Some("prism_volumetric_soft_tetra_volume_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_tetra_volume_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_tetra_volume_project_pass"),
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
