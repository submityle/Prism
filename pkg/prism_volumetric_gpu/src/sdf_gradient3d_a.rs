//! `wgpu` compute twin of four analytic signed-distance *gradient* primitives
//! of the `CPU` golden path — `sphere_gradient`, `box_gradient`,
//! `torus_gradient` and `plane_gradient` in
//! `prism_render_architecture::ray_scene::sdf_primitives`.
//!
//! Ray-marching procedural geometry needs not only each primitive's signed
//! distance but its *gradient*: the analytic surface normal (the unit gradient
//! of the distance field) used for shading and for the first-order step along a
//! ray. The reference derives these in closed form with zero finite-difference
//! error. [`GpuSdfGradient3dA`] is the on-device twin: each thread reads one
//! query point plus the shapes' parameters and writes the four gradient
//! vectors, reproducing the reference closed forms with only `sqrt`, `abs`,
//! `min`, `max`, `select`, sign selection, products and quotients — never a
//! numerical central difference.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfGradient3dAQuery`] — a query `point`, the
//! `plane_gradient` `plane_normal`, the `box_gradient` `box_half_extent` and
//! the `torus_gradient` `torus_major_radius`/`torus_minor_radius` — and writes
//! one [`SdfGradient3dAResult`] holding the four gradient vectors
//! `sphere_gradient`, `box_gradient`, `torus_gradient` and `plane_gradient`.
//!
//! The `sphere_gradient` kernel returns the outward radial direction
//! `point / |point|`, and the zero vector at the degenerate centre. The
//! `box_gradient` kernel forms the per-axis overshoot `max(|p| - b, 0)`: when
//! its length is positive it normalises that overshoot and restores each axis'
//! sign (exterior); otherwise the nearest face is the least-negative axis and
//! the gradient is that signed unit axis (interior). The `torus_gradient`
//! kernel reduces the point to `(rho - major, p.y)` with `rho = |p.xz|`, scales
//! the planar component radially and returns the `+y` axis on the degenerate
//! ring / central-axis fallbacks. The `plane_gradient` kernel returns the
//! constant plane normal unchanged.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms, the ray-marcher
//! that evaluates them along a ray, and higher-level shading all stay on the
//! host; the device sees only the four stateless, fixed-width gradient
//! evaluations, one query at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The sphere, box and torus gradients thread through `sqrt` and quotients, so
//! the `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a
//! few units in the last place from the scalar reference. The parity test
//! asserts each component within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! tight enough to catch a wrong port yet loose enough to admit a legal
//! last-place difference. Fixtures stay clear of the box edges/corners (the
//! measure-zero creases where the normal is genuinely undefined), the sphere
//! centre and the torus ring/central axis, where the `CPU` and `GPU` could pick
//! different sides of a branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `select`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`/`ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The exact-zero
//! degeneracy checks of the reference are expressed as ordered `<= 0.0`
//! comparisons (every quantity compared is a non-negative length), never a bare
//! float equality. It runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` analytic gradient kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `sphere_gradient`, `box_gradient`, `torus_gradient` and
/// `plane_gradient`, one query per thread.
const SDF_GRADIENT3D_A_WGSL: &str = r#"
// Analytic signed-distance gradients twinned from the CPU golden path. Each
// thread reads one query and writes four gradient vectors using only sqrt, abs,
// min, max, select, sign selection, products and quotients. The domain/CSG
// operators and the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point components.
    px: f32,
    py: f32,
    pz: f32,
    // Plane normal (already unit length; the plane gradient is this constant).
    nx: f32,
    ny: f32,
    nz: f32,
    // Box half extent per axis.
    bx: f32,
    by: f32,
    bz: f32,
    // Torus ring (major) radius and tube (minor) radius.
    major: f32,
    minor: f32,
    pad0: f32,
}

struct Gradients {
    // Sphere gradient (outward radial unit normal; zero at the centre).
    sgx: f32,
    sgy: f32,
    sgz: f32,
    // Box gradient.
    bgx: f32,
    bgy: f32,
    bgz: f32,
    // Torus gradient.
    tgx: f32,
    tgy: f32,
    tgz: f32,
    // Plane gradient (the constant plane normal).
    pgx: f32,
    pgy: f32,
    pgz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Gradients>;

// Reproduces the reference `f32::signum`: +1 for positive (and +0), -1 for
// negative. Expressed with an ordered comparison so no bare float equality
// appears; fixtures avoid the exact-zero inputs where this differs from a
// three-way sign.
fn signum_like(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Gradients;

    // --- Sphere gradient: outward radial unit normal, zero at the centre. ---
    let s_len = sqrt(q.px * q.px + q.py * q.py + q.pz * q.pz);
    if (s_len <= 0.0) {
        out.sgx = 0.0;
        out.sgy = 0.0;
        out.sgz = 0.0;
    } else {
        out.sgx = q.px / s_len;
        out.sgy = q.py / s_len;
        out.sgz = q.pz / s_len;
    }

    // --- Box gradient: normalised signed overshoot outside, nearest face in. ---
    let qx = abs(q.px) - q.bx;
    let qy = abs(q.py) - q.by;
    let qz = abs(q.pz) - q.bz;
    let mx = max(qx, 0.0);
    let my = max(qy, 0.0);
    let mz = max(qz, 0.0);
    let b_len = sqrt(mx * mx + my * my + mz * mz);
    if (b_len > 0.0) {
        // Exterior: normalise the overshoot and restore each axis' sign.
        out.bgx = signum_like(q.px) * mx / b_len;
        out.bgy = signum_like(q.py) * my / b_len;
        out.bgz = signum_like(q.pz) * mz / b_len;
    } else if (qx >= qy && qx >= qz) {
        // Interior: nearest face is the least-negative axis (largest q_i).
        out.bgx = signum_like(q.px);
        out.bgy = 0.0;
        out.bgz = 0.0;
    } else if (qy >= qz) {
        out.bgx = 0.0;
        out.bgy = signum_like(q.py);
        out.bgz = 0.0;
    } else {
        out.bgx = 0.0;
        out.bgy = 0.0;
        out.bgz = signum_like(q.pz);
    }

    // --- Torus gradient: radial planar component plus tube cross-section. ---
    let rho = sqrt(q.px * q.px + q.pz * q.pz);
    let t_qx = rho - q.major;
    let t_qy = q.py;
    let t_l = sqrt(t_qx * t_qx + t_qy * t_qy);
    if (t_l <= 0.0) {
        out.tgx = 0.0;
        out.tgy = 1.0;
        out.tgz = 0.0;
    } else if (rho <= 0.0) {
        // On the symmetry axis the planar direction is undefined.
        out.tgx = 0.0;
        out.tgy = signum_like(t_qy);
        out.tgz = 0.0;
    } else {
        let radial = (t_qx / t_l) / rho;
        out.tgx = radial * q.px;
        out.tgy = t_qy / t_l;
        out.tgz = radial * q.pz;
    }

    // --- Plane gradient: the constant plane normal, unchanged. ---
    out.pgx = q.nx;
    out.pgy = q.ny;
    out.pgz = q.nz;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_GRADIENT3D_A_WGSL`].
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
/// the point components, the plane normal, the box half extent and the torus
/// radii plus one pad word to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Plane normal `x`.
    nx: f32,
    /// Plane normal `y`.
    ny: f32,
    /// Plane normal `z`.
    nz: f32,
    /// Box half extent `x`.
    bx: f32,
    /// Box half extent `y`.
    by: f32,
    /// Box half extent `z`.
    bz: f32,
    /// Torus ring (major) radius.
    major: f32,
    /// Torus tube (minor) radius.
    minor: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Gradients`
/// struct: the four gradient vectors at a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Sphere gradient `x`.
    sgx: f32,
    /// Sphere gradient `y`.
    sgy: f32,
    /// Sphere gradient `z`.
    sgz: f32,
    /// Box gradient `x`.
    bgx: f32,
    /// Box gradient `y`.
    bgy: f32,
    /// Box gradient `z`.
    bgz: f32,
    /// Torus gradient `x`.
    tgx: f32,
    /// Torus gradient `y`.
    tgy: f32,
    /// Torus gradient `z`.
    tgz: f32,
    /// Plane gradient `x`.
    pgx: f32,
    /// Plane gradient `y`.
    pgy: f32,
    /// Plane gradient `z`.
    pgz: f32,
}

/// One query for the analytic gradient twin: the query `point` plus the
/// `plane_gradient`, `box_gradient` and `torus_gradient` shape parameters.
///
/// `point` is the evaluation position shared by the sphere, box and torus
/// gradients; `plane_normal` is the (unit) plane normal returned verbatim by
/// `plane_gradient`; `box_half_extent` is the box's per-axis half extent;
/// `torus_major_radius`/`torus_minor_radius` are the torus ring and tube radii
/// (the gradient is independent of the tube radius, which is carried for
/// parity with the reference signature).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient3dAQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Plane normal `[x, y, z]` (already unit length).
    pub plane_normal: [f32; 3],
    /// Box half extent `[x, y, z]` (non-negative per axis).
    pub box_half_extent: [f32; 3],
    /// Torus ring (major) radius.
    pub torus_major_radius: f32,
    /// Torus tube (minor) radius (unused by the gradient; carried for parity).
    pub torus_minor_radius: f32,
}

impl SdfGradient3dAQuery {
    /// Builds a query from the point and the plane, box and torus parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        plane_normal: [f32; 3],
        box_half_extent: [f32; 3],
        torus_major_radius: f32,
        torus_minor_radius: f32,
    ) -> SdfGradient3dAQuery {
        SdfGradient3dAQuery {
            point,
            plane_normal,
            box_half_extent,
            torus_major_radius,
            torus_minor_radius,
        }
    }
}

/// One resolved query of the analytic gradient twin: the `sphere_gradient`,
/// `box_gradient`, `torus_gradient` and `plane_gradient` vectors at the query
/// point.
///
/// Each is the exact analytic surface normal (unit gradient of the signed
/// distance) of its primitive, matching the reference with zero
/// finite-difference error; `sphere_gradient` is the zero vector at the
/// degenerate centre and `torus_gradient` falls back to `+y` on the degenerate
/// ring / central axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient3dAResult {
    /// Sphere gradient `[x, y, z]`.
    pub sphere_gradient: [f32; 3],
    /// Box gradient `[x, y, z]`.
    pub box_gradient: [f32; 3],
    /// Torus gradient `[x, y, z]`.
    pub torus_gradient: [f32; 3],
    /// Plane gradient `[x, y, z]`.
    pub plane_gradient: [f32; 3],
}

/// Encodes one [`SdfGradient3dAQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfGradient3dAQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        nx: q.plane_normal[0],
        ny: q.plane_normal[1],
        nz: q.plane_normal[2],
        bx: q.box_half_extent[0],
        by: q.box_half_extent[1],
        bz: q.box_half_extent[2],
        major: q.torus_major_radius,
        minor: q.torus_minor_radius,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfGradient3dAResult`].
fn decode_result(raw: &GpuResult) -> SdfGradient3dAResult {
    SdfGradient3dAResult {
        sphere_gradient: [raw.sgx, raw.sgy, raw.sgz],
        box_gradient: [raw.bgx, raw.bgy, raw.bgz],
        torus_gradient: [raw.tgx, raw.tgy, raw.tgz],
        plane_gradient: [raw.pgx, raw.pgy, raw.pgz],
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

/// A compiled, reusable analytic gradient compute pipeline, twinning the `CPU`
/// golden `sphere_gradient`, `box_gradient`, `torus_gradient` and
/// `plane_gradient` of `prism_render_architecture::ray_scene::sdf_primitives`.
pub struct GpuSdfGradient3dA {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfGradient3dA {
    /// Compiles the analytic gradient kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfGradient3dA {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a"),
            source: ShaderSource::Wgsl(SDF_GRADIENT3D_A_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfGradient3dA {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfGradient3dAResult`]
    /// per input, in order.
    ///
    /// The gradient components match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfGradient3dAQuery],
    ) -> Vec<SdfGradient3dAResult> {
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
            label: Some("prism_volumetric_sdf_gradient3d_a_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_bind_group"),
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
            label: Some("prism_volumetric_sdf_gradient3d_a_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_a_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_gradient3d_a_pass"),
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
