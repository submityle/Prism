//! `wgpu` compute twin of the 3D convex-hull construction contract
//! ([`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d) owns
//! the small, verifiable boundary contract several particle stages share:
//! turning an unordered slice of 3D points into the triangular faces of their
//! convex hull with the incremental method
//! ([`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d::convex_hull_3d)),
//! each face expressed as a triple of vertex indices into the input slice with a
//! consistent outward winding, built on the signed-volume predicate
//! ([`orient3d`](prism_render_architecture::particle::convex_hull_3d::orient3d))
//! and reported alongside its outward normal
//! ([`face_normal`](prism_render_architecture::particle::convex_hull_3d::face_normal))
//! and plane distance
//! ([`hull_signed_distance`](prism_render_architecture::particle::convex_hull_3d::hull_signed_distance)).
//! [`GpuConvexHull3d`] is the on-device twin: one thread builds one hull, so a
//! passing real-device parity test is direct evidence the ported kernel seeds,
//! inserts and stitches the same way the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-set answer the reference computes is reproduced for a batch of
//! independent point sets: the face count, the exact outward-wound vertex-index
//! triples (same seed tetrahedron, same incremental insertion order, same
//! visible-face deletion, same horizon-edge collection and same stitch order, so
//! the face list matches the reference `Vec` face for face and element for
//! element including winding), and each face's `cross`-product normal
//! ([`face_normal`](prism_render_architecture::particle::convex_hull_3d::face_normal)).
//! Each set carries up to [`MAX_POINTS`] points in a fixed `std430` slot plus its
//! actual point count, and the kernel reads only the leading `point_count` lanes.
//!
//! # Correctness model
//!
//! The face count and the index triples are discrete classifications, so for
//! inputs clear of the coplanarity and coincidence thresholds the `CPU` and
//! `GPU` take every `orient3d` sign branch identically and the parity test
//! asserts an exact `==` on the count and on every `u32` index. The face normals
//! thread through multiplies and subtractions, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each normal channel,
//! tight enough to catch a genuinely wrong port (a dropped visible face, a
//! swapped horizon edge, a flipped winding) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The kernel reproduces the reference's explicit degenerate handling: fewer
//! than four distinct points, a fully collinear set and a fully coplanar set all
//! lack hull volume and yield a zero face count. Points that coincide within
//! [`HULL_EPS`] are collapsed to their first occurrence before the seed is
//! chosen, so an index never names a duplicate, and interior points are naturally
//! skipped because they can see no face. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `cross`,
//! `dot`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, not even a `sqrt` (the twin
//! reports unnormalized `cross`-product normals), and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Every loop is bounded
//! by [`MAX_POINTS`] or [`MAX_FACES`], so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::convex_hull_3d`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Fixed per-set capacity: the maximum number of points one query may carry.
/// Each uploaded [`GpuPointSet`] reserves this many `vec3` lanes, and the host
/// rejects any [`ConvexHull3dQuery`] with more points than this.
pub const MAX_POINTS: usize = 16;

/// Fixed per-set face capacity. A convex polytope on `V` vertices has exactly
/// `2 * V - 4` triangular faces, so [`MAX_POINTS`] vertices admit at most this
/// many faces; the fixed output slot reserves exactly that bound.
pub const MAX_FACES: usize = 2 * MAX_POINTS - 4;

/// The portable core-`WGSL` 3D convex-hull kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d) branch
/// for branch; see the module documentation for the algorithm.
const CONVEX_HULL_3D_WGSL: &str = r#"
// 3D convex-hull twin: one thread per point-set collapses coincident points,
// chooses a non-degenerate seed tetrahedron, orients its four faces outward
// against the seed centroid, then inserts every point by deleting the faces it
// can see (orient3d above HULL_EPS), collecting the horizon loop of edges that
// border exactly one deleted face, and stitching one new outward-wound face to
// each horizon edge. It mirrors the CPU golden `particle::convex_hull_3d` branch
// for branch, uses only the portable core-WGSL subset (abs/min/cross/dot,
// + - * / and unsigned index math) with no sqrt and no transcendental call, and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
// Every loop is bounded by the fixed point and face capacities, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::convex_hull_3d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a signed volume, a squared edge length or a coordinate
// difference is treated as zero. Matches the reference `HULL_EPS`; the compare
// rule used instead of an f32 `==`.
const HULL_EPS: f32 = 1.0e-6;

// Fixed per-set point capacity, mirroring the host `MAX_POINTS`. Every point
// loop clamps its bound to this so a malformed count can never index past the
// fixed arrays.
const MAX_POINTS: u32 = 16u;

// Fixed per-set face capacity (2 * MAX_POINTS - 4), the exact face count of a
// convex polytope on MAX_POINTS vertices.
const MAX_FACES: u32 = 28u;

// Fixed working-edge capacity: three directed edges per face, enough for every
// visible face at once.
const MAX_EDGES: u32 = 84u;

struct Params {
    // Number of point-sets in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct PointSet {
    // The actual point count for this set; three pad words follow so the fixed
    // array starts at a 16-byte-aligned, vec3-friendly offset. Lanes at or past
    // `count` are never read.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    points: array<vec3<f32>, 16>,
}

struct Hull {
    // Number of valid faces; only the leading `face_count` lanes are meaningful.
    face_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Outward-wound vertex-index triples into the original point slice.
    faces: array<vec3<u32>, 28>,
    // The matching per-face cross-product normals (unnormalized).
    normals: array<vec3<f32>, 28>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sets: array<PointSet>;
@group(0) @binding(2) var<storage, read_write> results: array<Hull>;

// Six times the signed volume of the tetrahedron a, b, c, d, i.e. the scalar
// triple product ((b - a) x (c - a)) . (d - a); mirrors the reference
// `orient3d`. Strictly positive when d is on the outward (normal) side.
fn orient3d(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    return dot(cross(ab, ac), ad);
}

// Two points coincide when all three axes differ by at most HULL_EPS; mirrors
// the reference `points_equal`.
fn points_equal(a: vec3<f32>, b: vec3<f32>) -> bool {
    return abs(a.x - b.x) <= HULL_EPS && abs(a.y - b.y) <= HULL_EPS && abs(a.z - b.z) <= HULL_EPS;
}

// Builds a face a, b, c (working indices with coordinates pa, pb, pc) whose
// outward normal points away from the interior reference; mirrors the reference
// `make_face`. When the reference lies on the positive side the winding is
// flipped by swapping the last two vertices.
fn make_face(
    pa: vec3<f32>,
    pb: vec3<f32>,
    pc: vec3<f32>,
    interior: vec3<f32>,
    a: u32,
    b: u32,
    c: u32,
) -> vec3<u32> {
    let o = orient3d(pa, pb, pc, interior);
    if (o > 0.0) {
        return vec3<u32>(a, c, b);
    }
    return vec3<u32>(a, b, c);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var out: Hull;
    out.face_count = 0u;

    let n = min(sets[idx].count, MAX_POINTS);

    // Collapse coincident points, keeping the first occurrence and remembering
    // its original lane for the final index remap.
    var dpts: array<vec3<f32>, 16>;
    var orig: array<u32, 16>;
    var dcount: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let p = sets[idx].points[i];
        var dup = false;
        for (var j: u32 = 0u; j < dcount; j = j + 1u) {
            if (points_equal(dpts[j], p)) {
                dup = true;
            }
        }
        if (!dup) {
            dpts[dcount] = p;
            orig[dcount] = i;
            dcount = dcount + 1u;
        }
    }

    // Fewer than four distinct points have no 3D hull volume.
    if (dcount < 4u) {
        results[idx] = out;
        return;
    }

    // Seed tetrahedron: the first edge, the first point off that edge, and the
    // first point off the resulting plane.
    let i0 = 0u;
    let i1 = 1u;
    let base = dpts[i1] - dpts[i0];
    var i2 = 0u;
    var found2 = false;
    for (var k: u32 = 2u; k < dcount; k = k + 1u) {
        let cr = cross(base, dpts[k] - dpts[i0]);
        if (!found2 && dot(cr, cr) > HULL_EPS * HULL_EPS) {
            i2 = k;
            found2 = true;
        }
    }
    if (!found2) {
        results[idx] = out;
        return;
    }
    var i3 = 0u;
    var found3 = false;
    for (var m: u32 = 2u; m < dcount; m = m + 1u) {
        if (!found3 && m != i2 && abs(orient3d(dpts[i0], dpts[i1], dpts[i2], dpts[m])) > HULL_EPS) {
            i3 = m;
            found3 = true;
        }
    }
    if (!found3) {
        results[idx] = out;
        return;
    }

    // The seed centroid is a strictly interior reference for every winding.
    let interior = (dpts[i0] + dpts[i1] + dpts[i2] + dpts[i3]) * 0.25;

    var faces: array<vec3<u32>, 28>;
    faces[0] = make_face(dpts[i0], dpts[i1], dpts[i2], interior, i0, i1, i2);
    faces[1] = make_face(dpts[i0], dpts[i1], dpts[i3], interior, i0, i1, i3);
    faces[2] = make_face(dpts[i0], dpts[i2], dpts[i3], interior, i0, i2, i3);
    faces[3] = make_face(dpts[i1], dpts[i2], dpts[i3], interior, i1, i2, i3);
    var fcount: u32 = 4u;

    for (var i: u32 = 0u; i < dcount; i = i + 1u) {
        let p = dpts[i];

        // Which existing faces does this point lie strictly outside of?
        var vis: array<bool, 28>;
        var any_visible = false;
        for (var f: u32 = 0u; f < fcount; f = f + 1u) {
            let fv = faces[f];
            let seen = orient3d(dpts[fv.x], dpts[fv.y], dpts[fv.z], p) > HULL_EPS;
            vis[f] = seen;
            any_visible = any_visible || seen;
        }
        if (any_visible) {
            // Directed boundary edges of the visible region.
            var eu: array<u32, 84>;
            var ev: array<u32, 84>;
            var ecount: u32 = 0u;
            for (var f: u32 = 0u; f < fcount; f = f + 1u) {
                if (vis[f]) {
                    let fv = faces[f];
                    eu[ecount] = fv.x;
                    ev[ecount] = fv.y;
                    ecount = ecount + 1u;
                    eu[ecount] = fv.y;
                    ev[ecount] = fv.z;
                    ecount = ecount + 1u;
                    eu[ecount] = fv.z;
                    ev[ecount] = fv.x;
                    ecount = ecount + 1u;
                }
            }

            // A horizon edge is a directed edge whose reverse is absent, i.e. the
            // edge borders exactly one visible face.
            var hu: array<u32, 84>;
            var hv: array<u32, 84>;
            var hcount: u32 = 0u;
            for (var e: u32 = 0u; e < ecount; e = e + 1u) {
                let u = eu[e];
                let v = ev[e];
                var is_shared = false;
                for (var e2: u32 = 0u; e2 < ecount; e2 = e2 + 1u) {
                    if (eu[e2] == v && ev[e2] == u) {
                        is_shared = true;
                    }
                }
                if (!is_shared) {
                    hu[hcount] = u;
                    hv[hcount] = v;
                    hcount = hcount + 1u;
                }
            }

            // Retain only the faces the point cannot see, preserving their order.
            var kcount: u32 = 0u;
            for (var f: u32 = 0u; f < fcount; f = f + 1u) {
                if (!vis[f]) {
                    faces[kcount] = faces[f];
                    kcount = kcount + 1u;
                }
            }
            fcount = kcount;

            // Stitch the point to every horizon edge in collection order.
            for (var h: u32 = 0u; h < hcount; h = h + 1u) {
                faces[fcount] = make_face(
                    dpts[hu[h]],
                    dpts[hv[h]],
                    dpts[i],
                    interior,
                    hu[h],
                    hv[h],
                    i,
                );
                fcount = fcount + 1u;
            }
        }
    }

    // Remap working indices to the original lanes and emit each face normal.
    out.face_count = fcount;
    for (var f: u32 = 0u; f < fcount; f = f + 1u) {
        let fv = faces[f];
        out.faces[f] = vec3<u32>(orig[fv.x], orig[fv.y], orig[fv.z]);
        let a = dpts[fv.x];
        let b = dpts[fv.y];
        let c = dpts[fv.z];
        out.normals[f] = cross(b - a, c - a);
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`CONVEX_HULL_3D_WGSL`]: the point-set count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid point-sets.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One point-set as uploaded. `272`-byte `std430` stride matching `PointSet` in
/// the shader: the actual point count, three pad words lifting the array to its
/// `16`-byte-aligned offset, and a fixed array of up to [`MAX_POINTS`] `vec3`
/// lanes stored on `16`-byte slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPointSet {
    /// Number of valid points in `points`.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word that lifts `points` to its `16`-byte-aligned offset.
    pad2: u32,
    /// The point-set; lanes at or past `count` are unused and each `vec3` sits on
    /// a `16`-byte slot with a trailing pad lane.
    points: [[f32; 4]; MAX_POINTS],
}

/// One hull as read back. `912`-byte `std430` stride matching `Hull` in the
/// shader: the face count with three pad words, then the fixed `vec3<u32>` face
/// array and the fixed `vec3<f32>` normal array, each element on a `16`-byte
/// slot with a trailing pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHullRaw {
    /// Number of valid faces.
    face_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Outward-wound vertex-index triples, each on a `16`-byte slot.
    faces: [[u32; 4]; MAX_FACES],
    /// Per-face `cross`-product normals, each on a `16`-byte slot.
    normals: [[f32; 4]; MAX_FACES],
}

/// One convex-hull query: an independent point-set whose convex hull is solved
/// by a single thread.
///
/// Mirrors a single reference
/// [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d::convex_hull_3d)
/// call on `points`. The slice may hold up to [`MAX_POINTS`] points; fewer than
/// four distinct points, a collinear set and a coplanar set all yield an empty
/// hull. Holds `f32` geometry, so it is not hashable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvexHull3dQuery {
    /// The point-set to hull; at most [`MAX_POINTS`] points.
    pub points: Vec<[f32; 3]>,
}

/// The resolved convex hull for one query, read back from the kernel.
///
/// `face_count` names how many leading entries of `faces` and `face_normals` are
/// meaningful. Each face is an outward-wound triple of vertex indices into the
/// original query `points`, matching the reference
/// [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d::convex_hull_3d)
/// face list face for face and element for element, and each normal matches the
/// reference
/// [`face_normal`](prism_render_architecture::particle::convex_hull_3d::face_normal).
/// Holds `f32` geometry, so it derives only [`PartialEq`] (no `Eq` / `Hash`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuHull3d {
    /// Number of valid faces in `faces` and `face_normals`.
    pub face_count: u32,
    /// Outward-wound vertex-index triples into the original query `points`; only
    /// the leading `face_count` entries are meaningful.
    pub faces: [[u32; 3]; MAX_FACES],
    /// Per-face `cross`-product normals (unnormalized); only the leading
    /// `face_count` entries are meaningful.
    pub face_normals: [[f32; 3]; MAX_FACES],
}

impl GpuPointSet {
    /// Packs a [`ConvexHull3dQuery`] into the `std430` upload layout.
    ///
    /// # Panics
    ///
    /// Panics when `query.points` holds more than [`MAX_POINTS`] points, since
    /// the fixed per-set upload lane budget cannot carry them.
    fn from_query(query: &ConvexHull3dQuery) -> GpuPointSet {
        assert!(
            query.points.len() <= MAX_POINTS,
            "ConvexHull3dQuery holds {} points, exceeding MAX_POINTS ({MAX_POINTS})",
            query.points.len()
        );
        let mut points = [[0.0_f32; 4]; MAX_POINTS];
        for (lane, &p) in query.points.iter().enumerate() {
            points[lane] = [p[0], p[1], p[2], 0.0];
        }
        GpuPointSet {
            count: query.points.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            points,
        }
    }
}

/// Decodes one packed [`GpuHullRaw`] into the public [`GpuHull3d`], copying the
/// leading `face_count` faces and normals and dropping each `16`-byte slot's pad
/// lane.
fn decode_result(raw: &GpuHullRaw) -> GpuHull3d {
    let count = (raw.face_count as usize).min(MAX_FACES);
    let mut faces = [[0_u32; 3]; MAX_FACES];
    let mut face_normals = [[0.0_f32; 3]; MAX_FACES];
    for f in 0..count {
        faces[f] = [raw.faces[f][0], raw.faces[f][1], raw.faces[f][2]];
        face_normals[f] = [raw.normals[f][0], raw.normals[f][1], raw.normals[f][2]];
    }
    GpuHull3d {
        face_count: count as u32,
        faces,
        face_normals,
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

/// A compiled, reusable 3D convex-hull compute pipeline, twinning the `CPU`
/// golden
/// [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d).
pub struct GpuConvexHull3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConvexHull3d {
    /// Compiles the 3D convex-hull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConvexHull3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_convex_hull_3d"),
            source: ShaderSource::Wgsl(CONVEX_HULL_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConvexHull3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the convex hull of every point-set in `queries`, returning one
    /// [`GpuHull3d`] per query in input order.
    ///
    /// The returned hull for query `q` mirrors the reference
    /// [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d::convex_hull_3d)
    /// evaluated on `q.points`: the face count and the outward-wound vertex-index
    /// triples match exactly for inputs clear of the coplanarity and coincidence
    /// thresholds, and each face normal matches the reference
    /// [`face_normal`](prism_render_architecture::particle::convex_hull_3d::face_normal)
    /// to within the tolerance documented on this module. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics when any query holds more than [`MAX_POINTS`] points.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[ConvexHull3dQuery]) -> Vec<GpuHull3d> {
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
        let gpu_sets: Vec<GpuPointSet> = queries.iter().map(GpuPointSet::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuHullRaw>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let sets_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_sets"),
            contents: bytemuck::cast_slice(&gpu_sets),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_convex_hull_3d_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_convex_hull_3d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per point-set, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuHullRaw>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
