//! `wgpu` compute twin of the point-in-tetrahedron containment and 3D
//! barycentric-coordinate contract
//! ([`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron),
//! particle design §8.2, §11).
//!
//! The `CPU` golden
//! [`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron)
//! owns the small, verifiable geometry several particle stages share: expressing
//! a query point as a convex blend of a tetrahedron's four corners
//! ([`barycentric_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron::barycentric_in_tetrahedron)),
//! and deciding whether those four weights place the point inside the solid cell
//! ([`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron::point_in_tetrahedron)).
//! Both are built from the signed-volume orientation determinant
//! ([`orient3d`](prism_render_architecture::particle::point_in_tetrahedron::orient3d),
//! the scalar triple product `dot(b - a, cross(c - a, d - a))`).
//! [`GpuPointInTetrahedron`] is the on-device twin: one thread solves one
//! `(tetrahedron, point)` query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same weights and classifies the same
//! degenerate (coplanar) case the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the four barycentric weights `[b0, b1, b2, b3]` (with a
//! validity flag mirroring the reference [`Option`], left at zero for a
//! degenerate cell), and the inside/outside containment flag. The weight for
//! corner `i` is the ratio `Vi / V`, where `V` is the whole tetrahedron's signed
//! volume and `Vi` is the signed volume of the sub-tetrahedron that replaces
//! corner `i` with the query point.
//!
//! # Correctness model
//!
//! The validity flag and the inside flag are discrete classifications built from
//! `f32` magnitude comparisons, so for inputs clear of the degeneracy and
//! boundary thresholds the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on both. The four weights thread through multiplies,
//! adds and one guarded division, so `CPU` and `GPU` are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every weight, tight enough to
//! catch a genuinely wrong port (a dropped term, a swapped corner, a wrong
//! sub-volume) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A degenerate (coplanar or coincident) tetrahedron would divide the weights by
//! a near-zero total signed volume; the kernel checks the total against
//! [`DEGENERATE_EPS`] and reports an invalid flag (weights left at zero) instead,
//! matching the reference [`None`]. A degenerate cell is never inside, so its
//! containment flag is always `false`. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`, `cross`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no loop: each thread performs a fixed, bounded sequence of arithmetic, so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_in_tetrahedron`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` point-in-tetrahedron kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron)
/// branch for branch; see the module documentation for the algorithm.
const POINT_IN_TETRAHEDRON_WGSL: &str = r#"
// Point-in-tetrahedron twin: one thread per query reproduces the four
// barycentric weights (each a ratio of scalar triple products) and the
// inside/outside containment flag. It mirrors the CPU golden
// `particle::point_in_tetrahedron` branch for branch, uses only the portable
// core-WGSL subset (abs/dot/cross and + - * / plus unsigned index math), needs
// no sqrt and no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::point_in_tetrahedron；无第三方
// 引擎源码或衍生代码。

// Magnitude below which the tetrahedron's total signed volume is treated as
// zero, marking the cell as degenerate (coplanar / coincident) so no interior
// can be defined. Matches the reference `DEGENERATE_EPS`; the compare rule used
// instead of an f32 `==`.
const DEGENERATE_EPS: f32 = 1.0e-6;

// Slack applied to each barycentric weight when classifying containment: a
// weight is accepted as non-negative when it is at least -BOUNDARY_EPS, so a
// point on a face, edge or vertex still counts as inside. Matches the reference
// `BOUNDARY_EPS`.
const BOUNDARY_EPS: f32 = 1.0e-5;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Tetrahedron corners t0, t1, t2, t3 (each a vec3 with a trailing pad lane),
    // and the query point p.
    t0: vec3<f32>,
    pad0: f32,
    t1: vec3<f32>,
    pad1: f32,
    t2: vec3<f32>,
    pad2: f32,
    t3: vec3<f32>,
    pad3: f32,
    p: vec3<f32>,
    pad_p: f32,
}

struct Result {
    // Barycentric weights [b0, b1, b2, b3] (zero when invalid).
    bary: vec4<f32>,
    // 1 when the tetrahedron was non-degenerate, 0 otherwise.
    valid: u32,
    // 1 when p lies inside the tetrahedron (boundary included), 0 otherwise.
    inside: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scalar triple product dot(b - a, cross(c - a, d - a)): six times the signed
// volume of the tetrahedron (a, b, c, d), i.e. the orientation determinant.
// Mirrors the reference `orient3d`.
fn orient3d(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    return dot(b - a, cross(c - a, d - a));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let t0 = q.t0;
    let t1 = q.t1;
    let t2 = q.t2;
    let t3 = q.t3;
    let p = q.p;

    // Whole-tetrahedron signed volume. A near-zero total marks a degenerate
    // (coplanar) cell: set valid = 0 and leave the weights at zero rather than
    // dividing by ~zero, mirroring the reference `None`.
    var bary: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var valid: u32 = 0u;
    var inside: u32 = 0u;
    let total = orient3d(t0, t1, t2, t3);
    if (abs(total) >= DEGENERATE_EPS) {
        let inv = 1.0 / total;
        // Replacing corner i with p yields the signed sub-volume opposite i;
        // its ratio to the whole volume is the barycentric weight bi.
        let b0 = orient3d(p, t1, t2, t3) * inv;
        let b1 = orient3d(t0, p, t2, t3) * inv;
        let b2 = orient3d(t0, t1, p, t3) * inv;
        let b3 = orient3d(t0, t1, t2, p) * inv;
        bary = vec4<f32>(b0, b1, b2, b3);
        valid = 1u;

        // Containment: every weight at least -BOUNDARY_EPS admits the boundary
        // while rejecting points clearly outside.
        if (b0 >= -BOUNDARY_EPS && b1 >= -BOUNDARY_EPS && b2 >= -BOUNDARY_EPS && b3 >= -BOUNDARY_EPS) {
            inside = 1u;
        }
    }

    var out: Result;
    out.bary = bary;
    out.valid = valid;
    out.inside = inside;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`POINT_IN_TETRAHEDRON_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Tetrahedron corner `t0`.
    t0: [f32; 3],
    /// Pad lane after `t0`.
    pad0: f32,
    /// Tetrahedron corner `t1`.
    t1: [f32; 3],
    /// Pad lane after `t1`.
    pad1: f32,
    /// Tetrahedron corner `t2`.
    t2: [f32; 3],
    /// Pad lane after `t2`.
    pad2: f32,
    /// Tetrahedron corner `t3`.
    t3: [f32; 3],
    /// Pad lane after `t3`.
    pad3: f32,
    /// Query point `p`.
    p: [f32; 3],
    /// Pad lane after `p`.
    pad_p: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Barycentric weights `[b0, b1, b2, b3]` (zero when invalid).
    bary: [f32; 4],
    /// `1` when the tetrahedron was non-degenerate, `0` otherwise.
    valid: u32,
    /// `1` when `p` lies inside the tetrahedron (boundary included), `0`
    /// otherwise.
    inside: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One query for the point-in-tetrahedron twin: a tetrahedron's four corners and
/// a query point.
///
/// The barycentric solve and the containment test are both derived from the same
/// signed volumes, so a single query exercises every twinned answer at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointInTetrahedronQuery {
    /// Tetrahedron corners in `t0`, `t1`, `t2`, `t3` order.
    pub tet: [[f32; 3]; 4],
    /// Query point tested against the tetrahedron.
    pub point: [f32; 3],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointInTetrahedronResult {
    /// Barycentric weights `[b0, b1, b2, b3]`, or [`None`] for a degenerate
    /// tetrahedron, matching
    /// [`barycentric_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron::barycentric_in_tetrahedron).
    pub barycentric: Option<[f32; 4]>,
    /// Containment flag, matching
    /// [`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron::point_in_tetrahedron).
    pub inside: bool,
}

/// Encodes one [`PointInTetrahedronQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &PointInTetrahedronQuery) -> GpuQuery {
    GpuQuery {
        t0: q.tet[0],
        pad0: 0.0,
        t1: q.tet[1],
        pad1: 0.0,
        t2: q.tet[2],
        pad2: 0.0,
        t3: q.tet[3],
        pad3: 0.0,
        p: q.point,
        pad_p: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PointInTetrahedronResult`],
/// turning the validity flag back into an [`Option`].
fn decode_result(raw: &GpuResult) -> PointInTetrahedronResult {
    PointInTetrahedronResult {
        barycentric: if raw.valid == 0 { None } else { Some(raw.bary) },
        inside: raw.inside != 0,
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

/// A compiled, reusable point-in-tetrahedron compute pipeline, twinning the
/// `CPU` golden
/// [`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron).
pub struct GpuPointInTetrahedron {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPointInTetrahedron {
    /// Compiles the point-in-tetrahedron kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPointInTetrahedron {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron"),
            source: ShaderSource::Wgsl(POINT_IN_TETRAHEDRON_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPointInTetrahedron {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`PointInTetrahedronResult`] per input, in order.
    ///
    /// The validity and inside flags equal the reference exactly for inputs clear
    /// of the degeneracy and boundary thresholds; the weights match to within the
    /// tolerance documented on this module. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PointInTetrahedronQuery],
    ) -> Vec<PointInTetrahedronResult> {
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
            label: Some("prism_volumetric_point_in_tetrahedron_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_bind_group"),
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
            label: Some("prism_volumetric_point_in_tetrahedron_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_point_in_tetrahedron_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_point_in_tetrahedron_pass"),
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
