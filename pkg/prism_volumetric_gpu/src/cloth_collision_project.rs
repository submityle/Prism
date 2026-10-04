//! `wgpu` compute twin of the cloth position-based-dynamics collision
//! projections from the `CPU` golden `prism_render_architecture::cloth::collision`
//! (which forwards to `prism_physics_core::soft::collision`).
//!
//! A cloth solver keeps a garment off the skinned body and in front of its
//! painted backstops by projecting each particle out of a handful of analytic
//! colliders every iteration. This module ports the four stateless, closed-form
//! position projections — sphere, half-space, oriented box and backstop plane —
//! onto the device: one thread resolves one `(kind, position, parameters)`
//! query. The Gauss-Seidel `resolve_*` sweeps that take `&mut` slices are
//! deliberately *not* twinned, since they are stateful batch scatters.
//!
//! [`GpuClothCollisionProject`] is the on-device twin: a passing real-device
//! parity test is direct evidence the kernel takes the same early-out and
//! push-direction branch the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! * `kind = 0` — `project_out_of_sphere`: an interior point is pushed radially
//!   to the sphere surface; a point already outside, a non-positive radius, and
//!   a point coincident with the center (nudged out along `+Y`) are handled
//!   exactly as the reference.
//! * `kind = 1` — `project_out_of_half_space`: a point on the infeasible side
//!   `normal.dot(pos) < offset` is moved onto the plane along the (unnormalized)
//!   normal; a (near) zero normal is inert.
//! * `kind = 2` — `project_out_of_obb`: a point inside all three local slabs is
//!   pushed to the face of least penetration; the quaternion maps box-local
//!   axes to world space, so its conjugate takes the point into the local
//!   frame. An all-non-positive box is inert.
//! * `kind = 3` — `apply_backstop`: a point sunk more than `distance` behind the
//!   anchor along the unit normal is pushed back onto the limiting plane; a
//!   (near) zero normal is inert.
//!
//! The quaternion rotation uses the exact `glam` `Quat::mul_vec3` form
//! `v*(w*w - b.b) + b*(2 v.b) + (b x v)*(2 w)`, matched byte for byte on host
//! and device so the oriented-box path agrees to tolerance.
//!
//! # Correctness model
//!
//! Every coordinate threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every projected coordinate; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A point already outside its collider, a non-positive sphere radius, an
//! all-non-positive box and a (near) zero plane/backstop normal all fall back
//! to the identity exactly as the reference, so no path yields `NaN`. A query
//! `kind` outside `0..=3` passes the position through unchanged with
//! `valid = 0`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `select`, `sqrt`, `abs`, `dot`, `cross`, `+ - * /` and unsigned
//! index arithmetic — with no transcendental, `round`, `copysign`, `pow` or
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::collision`；无第三方引擎源码或衍生代码。
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

/// Query kind selecting the sphere projection.
pub const KIND_SPHERE: u32 = 0;
/// Query kind selecting the half-space projection.
pub const KIND_HALF_SPACE: u32 = 1;
/// Query kind selecting the oriented-box projection.
pub const KIND_OBB: u32 = 2;
/// Query kind selecting the backstop-plane projection.
pub const KIND_BACKSTOP: u32 = 3;

/// The portable core-`WGSL` cloth collision-projection kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// dispatches on `kind` to the four `CPU` golden projections branch for branch;
/// see the module documentation for the algorithm.
const CLOTH_COLLISION_PROJECT_WGSL: &str = r#"
// Cloth collision-projection twin: one thread per query reproduces one of the
// four position projections (sphere, half-space, oriented box, backstop plane)
// that cloth::collision forwards to prism_physics_core::soft::collision.
// Provenance: 孪生自本仓 prism_render_architecture::cloth::collision；无第三方引擎源码或衍生代码。

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    kind: u32,
    px: f32,
    py: f32,
    pz: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    c0: f32,
    c1: f32,
    c2: f32,
    s0: f32,
    pad0: u32,
}

struct Result {
    px: f32,
    py: f32,
    pz: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const EPS_LEN_SQ: f32 = 1e-12;

// glam Quat conjugate: negate the vector part, keep the scalar part.
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// glam Quat::mul_vec3 exact form: v*(w*w - b.b) + b*(2 v.b) + (b x v)*(2 w).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let scalar = q.w;
    let axis = vec3<f32>(q.x, q.y, q.z);
    let axis_sq = dot(axis, axis);
    return v * (scalar * scalar - axis_sq)
        + axis * (dot(v, axis) * 2.0)
        + cross(axis, v) * (scalar * 2.0);
}

fn project_out_of_sphere(pos: vec3<f32>, center: vec3<f32>, radius: f32) -> vec3<f32> {
    if (radius <= 0.0) {
        return pos;
    }
    let delta = pos - center;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= radius * radius) {
        return pos;
    }
    if (dist_sq <= EPS_LEN_SQ) {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = delta * (1.0 / sqrt(dist_sq));
    return center + dir * radius;
}

fn project_out_of_half_space(pos: vec3<f32>, normal: vec3<f32>, offset: f32) -> vec3<f32> {
    let len_sq = dot(normal, normal);
    if (len_sq <= EPS_LEN_SQ) {
        return pos;
    }
    let signed = dot(normal, pos) - offset;
    if (signed >= 0.0) {
        return pos;
    }
    let t = -signed / len_sq;
    return pos + normal * t;
}

fn project_out_of_obb(
    pos: vec3<f32>,
    center: vec3<f32>,
    orientation: vec4<f32>,
    half_extents: vec3<f32>,
) -> vec3<f32> {
    if (half_extents.x <= 0.0 && half_extents.y <= 0.0 && half_extents.z <= 0.0) {
        return pos;
    }
    let local = quat_rotate(quat_conj(orientation), pos - center);
    let al = abs(local);
    if (al.x >= half_extents.x || al.y >= half_extents.y || al.z >= half_extents.z) {
        return pos;
    }
    let pen = half_extents - al;
    var local_out = local;
    if (pen.x <= pen.y && pen.x <= pen.z) {
        local_out.x = select(-half_extents.x, half_extents.x, local.x >= 0.0);
    } else if (pen.y <= pen.z) {
        local_out.y = select(-half_extents.y, half_extents.y, local.y >= 0.0);
    } else {
        local_out.z = select(-half_extents.z, half_extents.z, local.z >= 0.0);
    }
    return center + quat_rotate(orientation, local_out);
}

fn apply_backstop(
    pos: vec3<f32>,
    origin: vec3<f32>,
    normal: vec3<f32>,
    distance: f32,
) -> vec3<f32> {
    let len_sq = dot(normal, normal);
    if (len_sq <= EPS_LEN_SQ) {
        return pos;
    }
    let unit = normal * (1.0 / sqrt(len_sq));
    let s = dot(unit, pos - origin);
    let min_s = -distance;
    if (s < min_s) {
        return pos + unit * (min_s - s);
    }
    return pos;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let pos = vec3<f32>(q.px, q.py, q.pz);
    let a = vec3<f32>(q.a0, q.a1, q.a2);
    let b = vec4<f32>(q.b0, q.b1, q.b2, q.b3);
    let c = vec3<f32>(q.c0, q.c1, q.c2);
    let s0 = q.s0;

    var out_pos = pos;
    var valid = 0u;
    if (q.kind == 0u) {
        out_pos = project_out_of_sphere(pos, a, s0);
        valid = 1u;
    } else if (q.kind == 1u) {
        out_pos = project_out_of_half_space(pos, a, s0);
        valid = 1u;
    } else if (q.kind == 2u) {
        out_pos = project_out_of_obb(pos, a, b, c);
        valid = 1u;
    } else if (q.kind == 3u) {
        out_pos = apply_backstop(pos, a, vec3<f32>(b.x, b.y, b.z), s0);
        valid = 1u;
    }

    var out: Result;
    out.px = out_pos.x;
    out.py = out_pos.y;
    out.pz = out_pos.z;
    out.valid = valid;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_COLLISION_PROJECT_WGSL`].
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
/// Every input is flattened to a scalar lane so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3`/`vec4`. The
/// trailing `pad0` rounds the slot to a `16`-word (`64`-byte) stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    kind: u32,
    px: f32,
    py: f32,
    pz: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    c0: f32,
    c1: f32,
    c2: f32,
    s0: f32,
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    px: f32,
    py: f32,
    pz: f32,
    valid: u32,
}

/// One cloth collision-projection query: a `kind` selector plus the position
/// and the parameters for the selected collider.
///
/// The parameter lanes are reused per `kind`: `a` is the sphere/box center, the
/// half-space normal or the backstop anchor; `b` is the box orientation
/// quaternion `(x, y, z, w)` or the backstop normal in its first three lanes;
/// `c` is the box half-extents; and `s` is the sphere radius, the half-space
/// offset or the backstop distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothCollisionProjectQuery {
    /// Which projection to apply (`0` sphere, `1` half-space, `2` box, `3`
    /// backstop); any other value passes the position through with `valid = 0`.
    pub kind: u32,
    /// The particle position to project.
    pub pos: [f32; 3],
    /// Primary vector parameter (center / normal / anchor).
    pub a: [f32; 3],
    /// Secondary four-lane parameter (box orientation quaternion, or backstop
    /// normal in the first three lanes).
    pub b: [f32; 4],
    /// Tertiary vector parameter (box half-extents).
    pub c: [f32; 3],
    /// Scalar parameter (radius / offset / distance).
    pub s: f32,
}

impl ClothCollisionProjectQuery {
    /// Builds a sphere projection query from the center and radius.
    #[must_use]
    pub fn sphere(pos: [f32; 3], center: [f32; 3], radius: f32) -> ClothCollisionProjectQuery {
        ClothCollisionProjectQuery {
            kind: KIND_SPHERE,
            pos,
            a: center,
            b: [0.0; 4],
            c: [0.0; 3],
            s: radius,
        }
    }

    /// Builds a half-space projection query from the (unnormalized) plane normal
    /// and offset.
    #[must_use]
    pub fn half_space(pos: [f32; 3], normal: [f32; 3], offset: f32) -> ClothCollisionProjectQuery {
        ClothCollisionProjectQuery {
            kind: KIND_HALF_SPACE,
            pos,
            a: normal,
            b: [0.0; 4],
            c: [0.0; 3],
            s: offset,
        }
    }

    /// Builds an oriented-box projection query from the center, orientation
    /// quaternion `(x, y, z, w)` and half-extents.
    #[must_use]
    pub fn obb(
        pos: [f32; 3],
        center: [f32; 3],
        orientation: [f32; 4],
        half_extents: [f32; 3],
    ) -> ClothCollisionProjectQuery {
        ClothCollisionProjectQuery {
            kind: KIND_OBB,
            pos,
            a: center,
            b: orientation,
            c: half_extents,
            s: 0.0,
        }
    }

    /// Builds a backstop projection query from the anchor, (unnormalized) normal
    /// and distance.
    #[must_use]
    pub fn backstop(
        pos: [f32; 3],
        origin: [f32; 3],
        normal: [f32; 3],
        distance: f32,
    ) -> ClothCollisionProjectQuery {
        ClothCollisionProjectQuery {
            kind: KIND_BACKSTOP,
            pos,
            a: origin,
            b: [normal[0], normal[1], normal[2], 0.0],
            c: [0.0; 3],
            s: distance,
        }
    }
}

/// One resolved projection, mirroring the reference projection output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothCollisionProjectResult {
    /// The projected particle position.
    pub pos: [f32; 3],
    /// `1` when a recognized `kind` was applied, `0` for an unknown `kind`
    /// (position passed through unchanged).
    pub valid: u32,
}

/// Encodes one [`ClothCollisionProjectQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &ClothCollisionProjectQuery) -> GpuQuery {
    GpuQuery {
        kind: q.kind,
        px: q.pos[0],
        py: q.pos[1],
        pz: q.pos[2],
        a0: q.a[0],
        a1: q.a[1],
        a2: q.a[2],
        b0: q.b[0],
        b1: q.b[1],
        b2: q.b[2],
        b3: q.b[3],
        c0: q.c[0],
        c1: q.c[1],
        c2: q.c[2],
        s0: q.s,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`ClothCollisionProjectResult`].
fn decode_result(raw: &GpuResult) -> ClothCollisionProjectResult {
    ClothCollisionProjectResult {
        pos: [raw.px, raw.py, raw.pz],
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

/// A compiled, reusable cloth collision-projection compute pipeline, twinning
/// the four `CPU` golden position projections.
pub struct GpuClothCollisionProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothCollisionProject {
    /// Compiles the cloth collision-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothCollisionProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_collision_project"),
            source: ShaderSource::Wgsl(CLOTH_COLLISION_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothCollisionProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothCollisionProjectResult`] per input, in order.
    ///
    /// Each projected coordinate matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothCollisionProjectQuery],
    ) -> Vec<ClothCollisionProjectResult> {
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
            label: Some("prism_volumetric_cloth_collision_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_bind_group"),
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
            label: Some("prism_volumetric_cloth_collision_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_collision_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_collision_project_pass"),
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
