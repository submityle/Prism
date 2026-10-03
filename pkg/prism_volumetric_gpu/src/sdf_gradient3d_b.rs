//! `wgpu` compute twin of four analytic signed-distance *gradient* primitives
//! of the `CPU` golden path
//! ([`capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capsule_gradient),
//! [`capped_cylinder_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capped_cylinder_gradient),
//! [`octahedron_gradient`](prism_render_architecture::ray_scene::sdf_primitives::octahedron_gradient)
//! and
//! [`vertical_capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::vertical_capsule_gradient)).
//!
//! Sphere tracing needs the *analytic* surface normal (unit gradient) of each
//! implicit primitive rather than a sampled central difference. The reference
//! derives four closed-form gradients: [`capsule_gradient`] for an
//! arbitrary-segment capsule, [`capped_cylinder_gradient`] for a `y`-axis
//! capped cylinder, [`octahedron_gradient`] for the exact octahedron, and
//! [`vertical_capsule_gradient`] for an upright `+y` capsule. Each stays
//! transcendental-free, using only `sqrt`, `abs`, `min`, `max`, `clamp`,
//! products and quotients. [`GpuSdfGradient3dB`] is the on-device twin: each
//! thread reads one point plus all four shapes' parameters and writes all four
//! gradient vectors, reproducing the reference closed forms operation for
//! operation.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfGradient3dBQuery`] — a query `point` plus the
//! capsule endpoints (`capsule_a`, `capsule_b`), the capped-cylinder
//! (`half_height`, `radius`), the octahedron (`oct_radius`) and the vertical
//! capsule (`height`) — and writes one [`SdfGradient3dBResult`] holding the
//! four gradient vectors. The capsule and vertical capsule normalise the offset
//! from the clamped nearest skeleton point; the capped cylinder maps the planar
//! box gradient in the `(radial, |y|)` frame back to `3D`; the octahedron folds
//! into the positive octant, picks the dominant face and normalises the folded
//! offset, then un-permutes and re-signs.
//!
//! # What stays on the host
//!
//! The signed-distance *values*, the domain and `CSG` operators that compose
//! these atoms, and the ray-marcher that walks a ray all stay on the host; the
//! device sees only the four stateless, fixed-width gradient evaluations, one
//! query at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every gradient threads through a `sqrt`-based `length` and a divide, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! each component within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The gradient
//! is discontinuous on the measure-zero skeleton/fold creases (the capsule
//! axis, the cylinder central axis and interior medial seam, the octahedron
//! octant and coordinate planes); fixtures stay a safe margin from those creases
//! so a last-place difference never picks a different branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The sign of a value is
//! recovered with an ordered comparison rather than the built-in `sign` so it
//! matches the reference branch exactly on the tested domain. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` analytic-gradient kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capsule_gradient),
/// [`capped_cylinder_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capped_cylinder_gradient),
/// [`octahedron_gradient`](prism_render_architecture::ray_scene::sdf_primitives::octahedron_gradient)
/// and
/// [`vertical_capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::vertical_capsule_gradient).
const SDF_GRADIENT3D_B_WGSL: &str = r#"
// Analytic signed-distance gradient twin: one thread computes one query point's
// capsule, capped-cylinder, octahedron and vertical-capsule unit gradients,
// mirroring the CPU golden `ray_scene::sdf_primitives::{capsule_gradient,
// capped_cylinder_gradient, octahedron_gradient, vertical_capsule_gradient}`
// with only sqrt, abs, min, max, clamp, products and quotients. The distance
// values, CSG operators and the ray-marcher stay on the host.
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
    // Capsule endpoint a.
    ax: f32,
    ay: f32,
    az: f32,
    // Capsule endpoint b.
    bx: f32,
    by: f32,
    bz: f32,
    // Capped-cylinder half height and radius (y axis).
    half_height: f32,
    radius: f32,
    // Octahedron vertex radius.
    oct_radius: f32,
    // Vertical-capsule height along +y.
    height: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Gradients {
    // Capsule unit gradient.
    cg_x: f32,
    cg_y: f32,
    cg_z: f32,
    // Capped-cylinder unit gradient.
    yg_x: f32,
    yg_y: f32,
    yg_z: f32,
    // Octahedron unit gradient.
    og_x: f32,
    og_y: f32,
    og_z: f32,
    // Vertical-capsule unit gradient.
    vg_x: f32,
    vg_y: f32,
    vg_z: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Gradients>;

// Euclidean length of a 3-vector.
fn length3(v: vec3<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Euclidean length of a 2-vector.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Dot product of two 3-vectors, summed left to right to match the reference.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 1 / sqrt(3): the octahedron central-slab face-normal component.
const OCT_C: f32 = 0.57735026;

// Smallest positive normal f32, matching the reference `f32::MIN_POSITIVE`
// guard on the capsule's squared segment length.
const MIN_POSITIVE: f32 = 1.17549435e-38;

// Gradient of length(u) in the rotated octant frame for a folded triple q.
fn oct_grad_q(q: vec3<f32>, radius: f32) -> vec3<f32> {
    let k = clamp(0.5 * (q.z - q.y + radius), 0.0, radius);
    let u = vec3<f32>(q.x, q.y - radius + k, q.z - k);
    let l = length3(u);
    if (l > 0.0) {
        return u / l;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let point = vec3<f32>(q.px, q.py, q.pz);

    var out: Gradients;

    // --- Capsule gradient (arbitrary segment a-b) ---
    let a = vec3<f32>(q.ax, q.ay, q.az);
    let b = vec3<f32>(q.bx, q.by, q.bz);
    let pa = point - a;
    let ba = b - a;
    let ba_len_sq = dot3(ba, ba);
    var h = 0.0;
    if (ba_len_sq > MIN_POSITIVE) {
        h = clamp(dot3(pa, ba) / ba_len_sq, 0.0, 1.0);
    }
    let closest = pa - ba * h;
    let cl = length3(closest);
    if (cl > 0.0) {
        out.cg_x = closest.x / cl;
        out.cg_y = closest.y / cl;
        out.cg_z = closest.z / cl;
    } else {
        out.cg_x = 0.0;
        out.cg_y = 0.0;
        out.cg_z = 0.0;
    }

    // --- Capped-cylinder gradient (y axis) ---
    let radial = length2(q.px, q.pz);
    let dx = radial - q.radius;
    let dy = abs(q.py) - q.half_height;
    let mr = max(dx, 0.0);
    let mh = max(dy, 0.0);
    let ll = length2(mr, mh);
    var gr = 0.0;
    var gh = 0.0;
    if (ll > 0.0) {
        gr = mr / ll;
        gh = mh / ll;
    } else if (dx >= dy) {
        gr = 1.0;
        gh = 0.0;
    } else {
        gr = 0.0;
        gh = 1.0;
    }
    var sy = 1.0;
    if (q.py < 0.0) {
        sy = -1.0;
    }
    if (radial > 0.0) {
        out.yg_x = gr * q.px / radial;
        out.yg_y = gh * sy;
        out.yg_z = gr * q.pz / radial;
    } else {
        out.yg_x = 0.0;
        out.yg_y = gh * sy;
        out.yg_z = 0.0;
    }

    // --- Octahedron gradient (vertex radius) ---
    let orad = q.oct_radius;
    var osign = vec3<f32>(1.0, 1.0, 1.0);
    if (q.px < 0.0) {
        osign.x = -1.0;
    }
    if (q.py < 0.0) {
        osign.y = -1.0;
    }
    if (q.pz < 0.0) {
        osign.z = -1.0;
    }
    let op = vec3<f32>(abs(q.px), abs(q.py), abs(q.pz));
    let m = op.x + op.y + op.z - orad;
    var gp = vec3<f32>(0.0, 0.0, 0.0);
    if (3.0 * op.x < m) {
        gp = oct_grad_q(vec3<f32>(op.x, op.y, op.z), orad);
    } else if (3.0 * op.y < m) {
        let g = oct_grad_q(vec3<f32>(op.y, op.z, op.x), orad);
        gp = vec3<f32>(g.z, g.x, g.y);
    } else if (3.0 * op.z < m) {
        let g = oct_grad_q(vec3<f32>(op.z, op.x, op.y), orad);
        gp = vec3<f32>(g.y, g.z, g.x);
    } else {
        gp = vec3<f32>(OCT_C, OCT_C, OCT_C);
    }
    out.og_x = gp.x * osign.x;
    out.og_y = gp.y * osign.y;
    out.og_z = gp.z * osign.z;

    // --- Vertical-capsule gradient (+y axis) ---
    let qy = q.py - clamp(q.py, 0.0, q.height);
    let v = vec3<f32>(q.px, qy, q.pz);
    let vl = length3(v);
    if (vl > 0.0) {
        out.vg_x = v.x / vl;
        out.vg_y = v.y / vl;
        out.vg_z = v.z / vl;
    } else {
        out.vg_x = 0.0;
        out.vg_y = 0.0;
        out.vg_z = 0.0;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_GRADIENT3D_B_WGSL`].
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
/// the point components plus all four shapes' parameters and three pad words to
/// a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Capsule endpoint `a` component `x`.
    ax: f32,
    /// Capsule endpoint `a` component `y`.
    ay: f32,
    /// Capsule endpoint `a` component `z`.
    az: f32,
    /// Capsule endpoint `b` component `x`.
    bx: f32,
    /// Capsule endpoint `b` component `y`.
    by: f32,
    /// Capsule endpoint `b` component `z`.
    bz: f32,
    /// Capped-cylinder half height.
    half_height: f32,
    /// Capped-cylinder radius.
    radius: f32,
    /// Octahedron vertex radius.
    oct_radius: f32,
    /// Vertical-capsule height.
    height: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Gradients`
/// struct: the four `3`-component gradient vectors packed as twelve scalars to
/// a `48`-byte stride (already a `16`-byte multiple, so no pad word is needed).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Capsule gradient `x`.
    cg_x: f32,
    /// Capsule gradient `y`.
    cg_y: f32,
    /// Capsule gradient `z`.
    cg_z: f32,
    /// Capped-cylinder gradient `x`.
    yg_x: f32,
    /// Capped-cylinder gradient `y`.
    yg_y: f32,
    /// Capped-cylinder gradient `z`.
    yg_z: f32,
    /// Octahedron gradient `x`.
    og_x: f32,
    /// Octahedron gradient `y`.
    og_y: f32,
    /// Octahedron gradient `z`.
    og_z: f32,
    /// Vertical-capsule gradient `x`.
    vg_x: f32,
    /// Vertical-capsule gradient `y`.
    vg_y: f32,
    /// Vertical-capsule gradient `z`.
    vg_z: f32,
}

/// One query for the analytic-gradient twin: the query `point` plus the
/// capsule, capped-cylinder, octahedron and vertical-capsule shape parameters.
///
/// `point` is the evaluation position; `capsule_a`/`capsule_b` are the
/// [`capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capsule_gradient)
/// endpoints; `half_height`/`radius` describe the
/// [`capped_cylinder_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capped_cylinder_gradient);
/// `oct_radius` describes the
/// [`octahedron_gradient`](prism_render_architecture::ray_scene::sdf_primitives::octahedron_gradient);
/// `height` describes the
/// [`vertical_capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::vertical_capsule_gradient).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient3dBQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Capsule endpoint `a`.
    pub capsule_a: [f32; 3],
    /// Capsule endpoint `b`.
    pub capsule_b: [f32; 3],
    /// Capped-cylinder half height.
    pub half_height: f32,
    /// Capped-cylinder radius.
    pub radius: f32,
    /// Octahedron vertex radius.
    pub oct_radius: f32,
    /// Vertical-capsule height.
    pub height: f32,
}

impl SdfGradient3dBQuery {
    /// Builds a query from the point and all four shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        capsule_a: [f32; 3],
        capsule_b: [f32; 3],
        half_height: f32,
        radius: f32,
        oct_radius: f32,
        height: f32,
    ) -> SdfGradient3dBQuery {
        SdfGradient3dBQuery {
            point,
            capsule_a,
            capsule_b,
            half_height,
            radius,
            oct_radius,
            height,
        }
    }
}

/// One resolved query of the analytic-gradient twin: the four unit gradients at
/// the query point.
///
/// `capsule_grad` is
/// [`capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capsule_gradient);
/// `capped_cylinder_grad` is
/// [`capped_cylinder_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capped_cylinder_gradient);
/// `octahedron_grad` is
/// [`octahedron_gradient`](prism_render_architecture::ray_scene::sdf_primitives::octahedron_gradient);
/// `vertical_capsule_grad` is
/// [`vertical_capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::vertical_capsule_gradient).
/// Each is unit length away from the measure-zero skeleton/fold creases, where
/// the reference returns the zero vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient3dBResult {
    /// Capsule unit gradient `[x, y, z]`.
    pub capsule_grad: [f32; 3],
    /// Capped-cylinder unit gradient `[x, y, z]`.
    pub capped_cylinder_grad: [f32; 3],
    /// Octahedron unit gradient `[x, y, z]`.
    pub octahedron_grad: [f32; 3],
    /// Vertical-capsule unit gradient `[x, y, z]`.
    pub vertical_capsule_grad: [f32; 3],
}

/// Encodes one [`SdfGradient3dBQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfGradient3dBQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        ax: q.capsule_a[0],
        ay: q.capsule_a[1],
        az: q.capsule_a[2],
        bx: q.capsule_b[0],
        by: q.capsule_b[1],
        bz: q.capsule_b[2],
        half_height: q.half_height,
        radius: q.radius,
        oct_radius: q.oct_radius,
        height: q.height,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfGradient3dBResult`].
fn decode_result(raw: &GpuResult) -> SdfGradient3dBResult {
    SdfGradient3dBResult {
        capsule_grad: [raw.cg_x, raw.cg_y, raw.cg_z],
        capped_cylinder_grad: [raw.yg_x, raw.yg_y, raw.yg_z],
        octahedron_grad: [raw.og_x, raw.og_y, raw.og_z],
        vertical_capsule_grad: [raw.vg_x, raw.vg_y, raw.vg_z],
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

/// A compiled, reusable analytic-gradient compute pipeline, twinning the `CPU`
/// golden
/// [`capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capsule_gradient),
/// [`capped_cylinder_gradient`](prism_render_architecture::ray_scene::sdf_primitives::capped_cylinder_gradient),
/// [`octahedron_gradient`](prism_render_architecture::ray_scene::sdf_primitives::octahedron_gradient)
/// and
/// [`vertical_capsule_gradient`](prism_render_architecture::ray_scene::sdf_primitives::vertical_capsule_gradient).
pub struct GpuSdfGradient3dB {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfGradient3dB {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfGradient3dB {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_module"),
            source: ShaderSource::Wgsl(SDF_GRADIENT3D_B_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfGradient3dB {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfGradient3dBResult`] per input, in order.
    ///
    /// The gradients match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfGradient3dBQuery],
    ) -> Vec<SdfGradient3dBResult> {
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
            label: Some("prism_volumetric_sdf_gradient3d_b_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_bind_group"),
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
            label: Some("prism_volumetric_sdf_gradient3d_b_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_gradient3d_b_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_gradient3d_b_pass"),
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
