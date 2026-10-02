//! `wgpu` compute twin of the 3D oriented-bounding-box vs oriented-bounding-box
//! boolean intersection golden
//! ([`obb_obb_sat_3d`](prism_render_architecture::particle::obb_obb_sat_3d),
//! design §10, §13).
//!
//! Several particle stages need a cheap, exact yes/no answer to "do these two
//! oriented boxes overlap?": an emitter-volume broadphase, a bounds-cluster
//! overlap merge and a light-proxy-vs-particle-bounds cull. The `CPU` golden
//! [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects)
//! owns that math via the Separating Axis Theorem (`SAT`); [`GpuObbSat3d`] is
//! the on-device twin that runs one thread per `OBB` pair and reproduces the
//! same per-pair boolean verdict, in input order. A passing real-device parity
//! test is therefore direct evidence the ported kernel folds the same fifteen
//! candidate axes, the same degenerate-axis skip and the same contact slack the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference axis for axis. Two convex polytopes are
//! disjoint if and only if some axis separates their projections; for a pair of
//! boxes it suffices to test fifteen candidates: the three face normals of box
//! `a`, the three face normals of box `b`, and the nine pairwise cross products
//! of one edge direction from each box. On every candidate axis each box
//! projects to a symmetric interval whose half-width is the sum of each
//! half-extent times the absolute dot of that box axis with the candidate axis
//! (the shared
//! [`projected_radius`](prism_render_architecture::particle::obb_obb_sat_3d::Obb::projected_radius)),
//! and the centers project to a signed offset whose magnitude is compared
//! against the summed half-widths plus
//! [`CONTACT_EPS`](prism_render_architecture::particle::obb_obb_sat_3d::CONTACT_EPS)
//! slack (the shared
//! [`separated_on_axis`](prism_render_architecture::particle::obb_obb_sat_3d::separated_on_axis)).
//! If any axis leaves a positive gap the boxes are disjoint; if none separates
//! them they intersect. When two edges are parallel their cross product is
//! (near) zero and cannot separate the boxes on its own — that direction is
//! already covered by the face normals — so such a degenerate axis (squared
//! length at or below
//! [`DEGENERATE_AXIS_EPS`](prism_render_architecture::particle::obb_obb_sat_3d::DEGENERATE_AXIS_EPS))
//! is skipped rather than normalized, exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `cross`, `+ - * /` and one `sqrt` to normalize a non-degenerate
//! edge-cross axis (mirroring the reference's single `f32::sqrt`) — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `smoothstep` or optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no `f32`
//! equality: the degenerate-axis test and the separation test compare against
//! the module epsilons.
//!
//! # Correctness model
//!
//! Each pair is a fixed, non-reorderable sequence of fifteen axis tests, so
//! `CPU` and `GPU` evaluate the same closed form in the same associativity.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The verdict is nevertheless an *exact* boolean match for pairs placed
//! clear of the contact boundary, which the named fixtures guarantee and the
//! random batch keeps away from the tie band.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::obb_obb_sat_3d`；
//! standard Gottschalk `OBBTree` fifteen-axis `SAT` overlap test plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::obb_obb_sat_3d::{intersects, Obb};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete intersect code written by the kernel for a lane whose two boxes
/// overlap (or exactly touch): matches the host `== 1` decode in
/// [`decode_result`]. A direct `f32` equality is forbidden, so the kernel emits
/// an integer flag rather than a sentinel float.
const CODE_INTERSECT: u32 = 1;

/// The portable core-`WGSL` `OBB`-`OBB` `SAT` kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects)
/// axis-for-axis; see the module documentation for the algorithm.
const OBB_OBB_SAT_3D_WGSL: &str = r#"
// OBB-OBB SAT twin: one thread per (box_a, box_b) pair tests the fifteen SAT
// candidate axes (three face normals per box plus nine edge-cross axes), skips a
// degenerate (near-parallel) edge-cross axis and reports a single intersect
// flag. It mirrors the CPU golden `particle::obb_obb_sat_3d` axis for axis, uses
// only the portable core-WGSL subset (abs/min/max/dot/cross, + - * / and one
// sqrt to normalize a non-degenerate edge-cross axis), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Gottschalk OBBTree fifteen-axis SAT overlap test; no
// third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One oriented box: center, three orthonormal local axes and the three
// half-extents (packed in the xyz lanes of `half`). Each field is a vec4 so the
// storage array needs no manual vec3 alignment arithmetic.
struct Obb {
    center: vec4<f32>,
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
    axis_z: vec4<f32>,
    half: vec4<f32>,
}

// One query: the ordered pair of boxes to test. 160-byte std430 stride matching
// the host `GpuQuery` (two 80-byte boxes).
struct Query {
    a: Obb,
    b: Obb,
}

// One result. 16-byte std430 stride matching the host `GpuResult`: the intersect
// flag as 0u/1u plus three pad words.
struct Result {
    intersects: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared-length threshold below which an edge-cross axis is treated as
// degenerate (the two edges are parallel) and skipped, mirroring the reference
// `DEGENERATE_AXIS_EPS`.
const DEGENERATE_AXIS_EPS: f32 = 1.0e-8;

// Separation slack added to the summed projection radii, mirroring the reference
// `CONTACT_EPS`: a pair is disjoint only when the center offset exceeds the
// radius sum by more than this, so an exact face/edge contact counts as an
// intersection rather than flickering from rounding noise.
const CONTACT_EPS: f32 = 1.0e-5;

// Half-width of this box's projection onto `axis`: the sum of each half-extent
// times the absolute alignment of its axis with `axis`. Mirrors the reference
// `Obb::projected_radius`.
fn projected_radius(box_in: Obb, axis: vec3<f32>) -> f32 {
    return box_in.half.x * abs(dot(box_in.axis_x.xyz, axis))
        + box_in.half.y * abs(dot(box_in.axis_y.xyz, axis))
        + box_in.half.z * abs(dot(box_in.axis_z.xyz, axis));
}

// Returns true when the two boxes have a positive gap along `axis`, i.e. `axis`
// is a separating axis. `axis` must be unit length for the `CONTACT_EPS` slack
// to carry a consistent world-space meaning. Mirrors `separated_on_axis`.
fn separated_on_axis(a: Obb, b: Obb, axis: vec3<f32>) -> bool {
    let center_offset = abs(dot(b.center.xyz - a.center.xyz, axis));
    let radius_sum = projected_radius(a, axis) + projected_radius(b, axis);
    return center_offset > radius_sum + CONTACT_EPS;
}

// Unit-length copy. Callers guarantee the vector is longer than
// `DEGENERATE_AXIS_EPS` (squared); a shorter vector is returned unchanged so no
// division by zero and no NaN can escape. Mirrors `Vec3::normalized`.
fn normalized(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= DEGENERATE_AXIS_EPS) {
        return v;
    }
    return v * (1.0 / sqrt(len_sq));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let a = queries[idx].a;
    let b = queries[idx].b;

    var a_axes = array<vec3<f32>, 3>(a.axis_x.xyz, a.axis_y.xyz, a.axis_z.xyz);
    var b_axes = array<vec3<f32>, 3>(b.axis_x.xyz, b.axis_y.xyz, b.axis_z.xyz);

    var hit = true;

    // Six face-normal axes: three from each box. These are already unit length.
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (separated_on_axis(a, b, a_axes[i])) {
            hit = false;
        }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (separated_on_axis(a, b, b_axes[i])) {
            hit = false;
        }
    }

    // Nine edge-cross axes: one edge direction from each box. A parallel pair
    // yields a near-zero cross product that cannot separate on its own and is
    // skipped (covered by the face normals above) rather than divided by ~0.
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let axis = cross(a_axes[i], b_axes[j]);
            if (dot(axis, axis) <= DEGENERATE_AXIS_EPS) {
                continue;
            }
            if (separated_on_axis(a, b, normalized(axis))) {
                hit = false;
            }
        }
    }

    var res: Result;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    if (hit) {
        res.intersects = 1u;
    } else {
        res.intersects = 0u;
    }
    results[idx] = res;
}
"#;

/// One `OBB`-`OBB` intersection query: the ordered pair of oriented boxes to
/// test.
///
/// Mirrors a single reference
/// [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects)
/// call. Carrying both boxes per query lets one dispatch test many distinct
/// pairs. Derives only `Clone`, `Copy` and `Debug`: the reference `Obb` it
/// carries derives no `PartialEq`, so neither can this wrapper.
#[derive(Clone, Copy, Debug)]
pub struct ObbSat3dQuery {
    /// The first oriented box of the pair.
    pub a: Obb,
    /// The second oriented box of the pair.
    pub b: Obb,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `intersects` is the per-pair boolean: `true` when the boxes overlap or
/// exactly touch, `false` when some `SAT` axis separates them. Mirrors the
/// reference
/// [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects).
/// Derives [`Eq`] because it carries only a boolean, no `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObbSat3dResult {
    /// Whether the two oriented boxes intersect (overlap or exactly touch).
    pub intersects: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`OBB_OBB_SAT_3D_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One oriented box as uploaded. `80`-byte `std430` stride matching `Obb` in the
/// shader: the center, three local unit axes and the half-extents, each padded
/// to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuObb {
    /// Box center in `xyz`; the `w` lane is unused padding.
    center: [f32; 4],
    /// Local `x` axis in `xyz`; the `w` lane is unused padding.
    axis_x: [f32; 4],
    /// Local `y` axis in `xyz`; the `w` lane is unused padding.
    axis_y: [f32; 4],
    /// Local `z` axis in `xyz`; the `w` lane is unused padding.
    axis_z: [f32; 4],
    /// Per-axis half-extents in `xyz`; the `w` lane is unused padding.
    half: [f32; 4],
}

impl GpuObb {
    /// Packs a reference [`Obb`] into the `std430` upload layout.
    fn from_obb(obb: &Obb) -> GpuObb {
        let c = obb.center;
        let ax = obb.axis_x;
        let ay = obb.axis_y;
        let az = obb.axis_z;
        GpuObb {
            center: [c.x, c.y, c.z, 0.0],
            axis_x: [ax.x, ax.y, ax.z, 0.0],
            axis_y: [ay.x, ay.y, ay.z, 0.0],
            axis_z: [az.x, az.y, az.z, 0.0],
            half: [obb.half_x, obb.half_y, obb.half_z, 0.0],
        }
    }
}

/// One query as uploaded. `160`-byte `std430` stride matching `Query` in the
/// shader: the ordered pair of `80`-byte boxes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The first box of the pair.
    a: GpuObb,
    /// The second box of the pair.
    b: GpuObb,
}

impl GpuQuery {
    /// Packs an [`ObbSat3dQuery`] into the `std430` upload layout.
    fn from_query(query: &ObbSat3dQuery) -> GpuQuery {
        GpuQuery {
            a: GpuObb::from_obb(&query.a),
            b: GpuObb::from_obb(&query.b),
        }
    }
}

/// One result as read back. `16`-byte `std430` stride matching `Result` in the
/// shader: the intersect flag as `0`/`1` plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Intersect flag (`1` = overlap or touch).
    intersects: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Maps one kernel `Result` lane back to the host [`ObbSat3dResult`].
fn decode_result(raw: &GpuResult) -> ObbSat3dResult {
    ObbSat3dResult {
        intersects: raw.intersects == CODE_INTERSECT,
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// point so callers (and the parity test) can pin the twin lane for lane.
///
/// Returns the per-pair boolean from
/// [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects).
#[must_use]
pub fn cpu_reference(query: &ObbSat3dQuery) -> bool {
    intersects(&query.a, &query.b)
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

/// A compiled, reusable `OBB`-`OBB` `SAT` intersection pipeline.
pub struct GpuObbSat3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuObbSat3d {
    /// Compiles the `OBB`-`OBB` `SAT` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbSat3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d"),
            source: ShaderSource::Wgsl(OBB_OBB_SAT_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbSat3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`ObbSat3dResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects)
    /// evaluated on `q.a` and `q.b`. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ObbSat3dQuery]) -> Vec<ObbSat3dResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_obb_obb_sat_3d_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_obb_obb_sat_3d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
