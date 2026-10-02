//! `wgpu` compute twin of the 2D convex-hull construction and hull-metric
//! contract
//! ([`convex_hull_2d`](prism_render_architecture::particle::convex_hull_2d),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`convex_hull_2d`](prism_render_architecture::particle::convex_hull_2d) owns
//! the small, verifiable boundary contract several particle stages share:
//! turning an unordered slice of 2D points into their convex hull as a `CCW`
//! (counter-clockwise) vertex ring with Andrew's monotone chain
//! ([`convex_hull`](prism_render_architecture::particle::convex_hull_2d::convex_hull)),
//! plus the scalar metrics that describe that hull — the shoelace area
//! ([`hull_area`](prism_render_architecture::particle::convex_hull_2d::hull_area)),
//! the closed-edge perimeter
//! ([`hull_perimeter`](prism_render_architecture::particle::convex_hull_2d::hull_perimeter)),
//! the all-pairs squared diameter
//! ([`diameter_squared`](prism_render_architecture::particle::convex_hull_2d::diameter_squared)),
//! and the convex-winding predicate
//! ([`is_convex_ccw`](prism_render_architecture::particle::convex_hull_2d::is_convex_ccw)),
//! each built on the signed-area turn test
//! ([`cross2`](prism_render_architecture::particle::convex_hull_2d::cross2)).
//! [`GpuConvexHull2d`] is the on-device twin: one thread builds one hull, so a
//! passing real-device parity test is direct evidence the ported kernel sorts,
//! sweeps and measures the same way the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-set answer the reference computes is reproduced for a batch of
//! independent point sets: the ordered hull ring itself (identical vertex
//! sequence, since the kernel replays the same sort-then-sweep), the hull vertex
//! count, the shoelace area, the closed perimeter, the squared diameter and the
//! convex-`CCW` flag. Each set carries up to [`MAX_POINTS`] points in a fixed
//! `std430` slot plus its actual point count, and the kernel reads only the
//! leading `point_count` lanes.
//!
//! # Correctness model
//!
//! The hull vertex count and the convex flag are discrete classifications, so
//! for inputs clear of the collinearity and coincidence thresholds the `CPU` and
//! `GPU` agree exactly and the parity test asserts an exact `==` on the count and
//! on the flag. The vertex coordinates are copied verbatim from the sorted
//! input, so they match the reference to the same tolerance the continuous
//! metrics do. The area, perimeter and squared diameter thread through
//! multiplies, adds and (for the perimeter) a `sqrt`, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous
//! quantity, tight enough to catch a genuinely wrong port (a dropped turn test, a
//! swapped chain, a missing dedup) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The kernel reproduces the reference's explicit degenerate handling: an empty
//! set yields an empty hull with zero metrics; a single distinct point yields
//! that point; two distinct points yield a segment (zero area, perimeter equal to
//! the single edge length); and three-or-more collinear points collapse to the
//! two extreme endpoints, since every interior turn fails the strict
//! left-turn test against [`CMP_EPS`]. Coincident points within [`CMP_EPS`] are
//! deduplicated before the sweep, exactly as the reference collapses exact
//! duplicates. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `+ - * /`, the unsigned `%` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every loop is bounded by [`MAX_POINTS`], so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::convex_hull_2d`；无第三方引擎源码或衍生代码。
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
/// Each uploaded [`GpuPointSet`] reserves this many `vec2` lanes, and the host
/// rejects any [`ConvexHull2dQuery`] with more points than this.
pub const MAX_POINTS: usize = 32;

/// The portable core-`WGSL` 2D convex-hull kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`convex_hull_2d`](prism_render_architecture::particle::convex_hull_2d) branch
/// for branch; see the module documentation for the algorithm.
const CONVEX_HULL_2D_WGSL: &str = r#"
// 2D convex-hull twin: one thread per point-set sorts the set by (x ascending,
// then y ascending), collapses coincident points, builds the lower and upper
// monotone chains (popping any vertex that is not a strict left turn against
// CMP_EPS) and concatenates them into a CCW ring, then measures the shoelace
// area, the closed perimeter, the all-pairs squared diameter and the convex
// winding flag. It mirrors the CPU golden `particle::convex_hull_2d` branch for
// branch, uses only the portable core-WGSL subset (abs/min/max/sqrt, + - * /,
// unsigned % and unsigned index math) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. Every loop is bounded by MAX_POINTS, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::convex_hull_2d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a cross product, a coordinate difference or a turn is
// treated as zero. Matches the reference `CMP_EPS`; the compare rule used
// instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Fixed per-set capacity, mirroring the host `MAX_POINTS`. Every loop clamps its
// bound to this so a malformed count can never index past the fixed arrays.
const MAX_POINTS: u32 = 32u;

struct Params {
    // Number of point-sets in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct PointSet {
    // The actual point count for this set; a pad word follows so the fixed array
    // starts at a 16-byte-friendly, vec2-aligned offset. Lanes at or past
    // `count` are never read.
    count: u32,
    pad0: u32,
    points: array<vec2<f32>, 32>,
}

struct Hull {
    // The CCW hull ring; only the leading `hull_count` lanes are meaningful.
    hull: array<vec2<f32>, 32>,
    // Shoelace area of the ring (0.0 for fewer than three vertices).
    area: f32,
    // Closed perimeter of the ring (single edge length for a segment).
    perimeter: f32,
    // Largest squared distance between any two hull vertices.
    diameter_squared: f32,
    // Number of vertices in `hull`.
    hull_count: u32,
    // 1u when the ring is convex and wound CCW, 0u otherwise.
    is_convex_ccw: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sets: array<PointSet>;
@group(0) @binding(2) var<storage, read_write> results: array<Hull>;

// Twice the signed area of triangle `o, a, b`, i.e. the 2D cross product
// `(a - o) x (b - o)`; mirrors the reference `cross2`. Strictly positive for a
// left (CCW) turn, strictly negative for a right turn, within CMP_EPS of zero
// when the three points are collinear.
fn cross2(o: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Two points coincide when both axes differ by at most CMP_EPS; mirrors the
// reference `points_equal`.
fn points_equal(a: vec2<f32>, b: vec2<f32>) -> bool {
    return abs(a.x - b.x) <= CMP_EPS && abs(a.y - b.y) <= CMP_EPS;
}

// Euclidean distance between two 2D points; mirrors the reference `distance`.
fn distance2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Strict (x then y) less-than used by the stable insertion sort. Points whose
// x coordinates are within CMP_EPS are treated as sharing an x key and tie-break
// on y, mirroring the reference `total_cmp` ordering on the well-separated
// fixtures the twin targets.
fn point_less(a: vec2<f32>, b: vec2<f32>) -> bool {
    if (abs(a.x - b.x) > CMP_EPS) {
        return a.x < b.x;
    }
    return a.y < b.y;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let n = min(sets[idx].count, MAX_POINTS);

    // Local working copy of the input lanes.
    var pts: array<vec2<f32>, 32>;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        pts[i] = sets[idx].points[i];
    }

    var out: Hull;
    out.area = 0.0;
    out.perimeter = 0.0;
    out.diameter_squared = 0.0;
    out.hull_count = 0u;
    out.is_convex_ccw = 0u;
    out.pad0 = 0u;

    var hull: array<vec2<f32>, 32>;
    var hc: u32 = 0u;

    if (n <= 1u) {
        // The reference returns a 0-or-1 point set verbatim, with no sort or
        // dedup.
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            hull[i] = pts[i];
        }
        hc = n;
    } else {
        // Stable insertion sort by (x ascending, then y ascending).
        for (var i: u32 = 1u; i < n; i = i + 1u) {
            let key = pts[i];
            var j: i32 = i32(i) - 1;
            while (j >= 0 && point_less(key, pts[u32(j)])) {
                pts[u32(j + 1)] = pts[u32(j)];
                j = j - 1;
            }
            pts[u32(j + 1)] = key;
        }

        // Collapse runs of coincident points, keeping the first of each run.
        var m: u32 = 0u;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            if (m == 0u || !points_equal(pts[i], pts[m - 1u])) {
                pts[m] = pts[i];
                m = m + 1u;
            }
        }

        if (m <= 2u) {
            // A single surviving point or a segment is already the hull.
            for (var i: u32 = 0u; i < m; i = i + 1u) {
                hull[i] = pts[i];
            }
            hc = m;
        } else {
            // Lower chain: sweep in sorted order, popping any vertex whose turn
            // is not a strict left turn.
            var lower: array<vec2<f32>, 32>;
            var ll: u32 = 0u;
            for (var i: u32 = 0u; i < m; i = i + 1u) {
                let p = pts[i];
                while (ll >= 2u && cross2(lower[ll - 2u], lower[ll - 1u], p) <= CMP_EPS) {
                    ll = ll - 1u;
                }
                lower[ll] = p;
                ll = ll + 1u;
            }

            // Upper chain: sweep in reverse order with the same pop rule.
            var upper: array<vec2<f32>, 32>;
            var ul: u32 = 0u;
            for (var i: u32 = 0u; i < m; i = i + 1u) {
                let p = pts[m - 1u - i];
                while (ul >= 2u && cross2(upper[ul - 2u], upper[ul - 1u], p) <= CMP_EPS) {
                    ul = ul - 1u;
                }
                upper[ul] = p;
                ul = ul + 1u;
            }

            // Drop each chain's shared endpoint, then concatenate lower + upper.
            ll = ll - 1u;
            ul = ul - 1u;
            for (var i: u32 = 0u; i < ll; i = i + 1u) {
                hull[i] = lower[i];
            }
            for (var i: u32 = 0u; i < ul; i = i + 1u) {
                hull[ll + i] = upper[i];
            }
            hc = ll + ul;
        }
    }

    // Shoelace area: zero for a ring of fewer than three vertices.
    if (hc >= 3u) {
        var twice_signed: f32 = 0.0;
        for (var i: u32 = 0u; i < hc; i = i + 1u) {
            let cur = hull[i];
            let nxt = hull[(i + 1u) % hc];
            twice_signed = twice_signed + (cur.x * nxt.y - nxt.x * cur.y);
        }
        out.area = abs(twice_signed * 0.5);
    }

    // Perimeter: a segment uses its single edge length; three-or-more vertices
    // sum the closed loop.
    if (hc == 2u) {
        out.perimeter = distance2d(hull[0], hull[1]);
    } else if (hc >= 3u) {
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < hc; i = i + 1u) {
            sum = sum + distance2d(hull[i], hull[(i + 1u) % hc]);
        }
        out.perimeter = sum;
    }

    // Squared diameter: the largest squared distance between any two vertices.
    var best: f32 = 0.0;
    for (var i: u32 = 0u; i < hc; i = i + 1u) {
        for (var j: u32 = i + 1u; j < hc; j = j + 1u) {
            let dx = hull[i].x - hull[j].x;
            let dy = hull[i].y - hull[j].y;
            best = max(best, dx * dx + dy * dy);
        }
    }
    out.diameter_squared = best;

    // Convex-CCW predicate: a ring needs three vertices and every consecutive
    // turn must be left or collinear (cross product at least -CMP_EPS).
    var convex: u32 = 0u;
    if (hc >= 3u) {
        convex = 1u;
        for (var i: u32 = 0u; i < hc; i = i + 1u) {
            let o = hull[i];
            let a = hull[(i + 1u) % hc];
            let b = hull[(i + 2u) % hc];
            if (cross2(o, a, b) < -CMP_EPS) {
                convex = 0u;
            }
        }
    }
    out.is_convex_ccw = convex;

    for (var i: u32 = 0u; i < hc; i = i + 1u) {
        out.hull[i] = hull[i];
    }
    out.hull_count = hc;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`CONVEX_HULL_2D_WGSL`]: the point-set count and three pad words —
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

/// One point-set as uploaded. `264`-byte `std430` stride matching `PointSet` in
/// the shader: the actual point count, one pad word and a fixed array of up to
/// [`MAX_POINTS`] `vec2` lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPointSet {
    /// Number of valid points in `points`.
    count: u32,
    /// Padding word that lifts `points` to its `vec2`-aligned offset.
    pad0: u32,
    /// The point-set; lanes at or past `count` are unused.
    points: [[f32; 2]; MAX_POINTS],
}

/// One hull as read back. `280`-byte `std430` stride matching `Hull` in the
/// shader: the fixed `vec2` ring, the three continuous metrics, the vertex
/// count, the convex flag and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHull2d {
    /// The `CCW` hull ring; only the leading `hull_count` lanes are meaningful.
    hull: [[f32; 2]; MAX_POINTS],
    /// Shoelace area of the ring.
    area: f32,
    /// Closed perimeter of the ring.
    perimeter: f32,
    /// Largest squared distance between any two hull vertices.
    diameter_squared: f32,
    /// Number of vertices in `hull`.
    hull_count: u32,
    /// `1` when the ring is convex and wound `CCW`, `0` otherwise.
    is_convex_ccw: u32,
    /// Padding word.
    pad0: u32,
}

/// One convex-hull query: an independent point-set whose convex hull and hull
/// metrics are solved by a single thread.
///
/// Mirrors a single reference
/// [`convex_hull`](prism_render_architecture::particle::convex_hull_2d::convex_hull)
/// call on `points`, together with the metrics the reference derives from that
/// hull. The slice may hold up to [`MAX_POINTS`] points; an empty slice yields an
/// empty hull with zero metrics. Holds `f32` geometry, so it is not hashable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvexHull2dQuery {
    /// The point-set to hull; at most [`MAX_POINTS`] points.
    pub points: Vec<[f32; 2]>,
}

/// The resolved convex hull and its metrics for one query, read back from the
/// kernel.
///
/// `hull` is the `CCW` vertex ring, truncated to the vertices the kernel kept;
/// it matches the reference
/// [`convex_hull`](prism_render_architecture::particle::convex_hull_2d::convex_hull)
/// vertex sequence. The metrics mirror
/// [`hull_area`](prism_render_architecture::particle::convex_hull_2d::hull_area),
/// [`hull_perimeter`](prism_render_architecture::particle::convex_hull_2d::hull_perimeter),
/// [`diameter_squared`](prism_render_architecture::particle::convex_hull_2d::diameter_squared)
/// and
/// [`is_convex_ccw`](prism_render_architecture::particle::convex_hull_2d::is_convex_ccw).
/// Holds `f32` geometry, so it derives only [`PartialEq`] (no `Eq` / `Hash`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvexHull2dResult {
    /// The `CCW` hull ring, open (the first vertex is not repeated at the end).
    pub hull: Vec<[f32; 2]>,
    /// Shoelace area of the hull ring.
    pub area: f32,
    /// Closed perimeter of the hull ring.
    pub perimeter: f32,
    /// Largest squared distance between any two hull vertices.
    pub diameter_squared: f32,
    /// `true` when the hull ring is convex and wound `CCW`.
    pub is_convex_ccw: bool,
}

impl GpuPointSet {
    /// Packs a [`ConvexHull2dQuery`] into the `std430` upload layout.
    ///
    /// # Panics
    ///
    /// Panics when `query.points` holds more than [`MAX_POINTS`] points, since
    /// the fixed per-set upload lane budget cannot carry them.
    fn from_query(query: &ConvexHull2dQuery) -> GpuPointSet {
        assert!(
            query.points.len() <= MAX_POINTS,
            "ConvexHull2dQuery holds {} points, exceeding MAX_POINTS ({MAX_POINTS})",
            query.points.len()
        );
        let mut points = [[0.0_f32; 2]; MAX_POINTS];
        for (lane, &p) in query.points.iter().enumerate() {
            points[lane] = p;
        }
        GpuPointSet {
            count: query.points.len() as u32,
            pad0: 0,
            points,
        }
    }
}

/// Decodes one packed [`GpuHull2d`] into the public [`ConvexHull2dResult`],
/// truncating the fixed ring to the kept vertex count and turning the convex
/// flag back into a [`bool`].
fn decode_result(raw: &GpuHull2d) -> ConvexHull2dResult {
    let hull_count = (raw.hull_count as usize).min(MAX_POINTS);
    ConvexHull2dResult {
        hull: raw.hull[..hull_count].to_vec(),
        area: raw.area,
        perimeter: raw.perimeter,
        diameter_squared: raw.diameter_squared,
        is_convex_ccw: raw.is_convex_ccw != 0,
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

/// A compiled, reusable 2D convex-hull compute pipeline, twinning the `CPU`
/// golden
/// [`convex_hull_2d`](prism_render_architecture::particle::convex_hull_2d).
pub struct GpuConvexHull2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConvexHull2d {
    /// Compiles the 2D convex-hull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConvexHull2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_convex_hull_2d"),
            source: ShaderSource::Wgsl(CONVEX_HULL_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConvexHull2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the convex hull of every point-set in `queries`, returning one
    /// [`ConvexHull2dResult`] per query in input order.
    ///
    /// The returned hull for query `q` mirrors the reference
    /// [`convex_hull`](prism_render_architecture::particle::convex_hull_2d::convex_hull)
    /// evaluated on `q.points`, together with the reference metrics derived from
    /// that hull: the vertex count and the convex flag match exactly for inputs
    /// clear of the collinearity and coincidence thresholds, and the coordinates
    /// and continuous metrics match to within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics when any query holds more than [`MAX_POINTS`] points.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ConvexHull2dQuery],
    ) -> Vec<ConvexHull2dResult> {
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

        let out_bytes = (queries.len() * size_of::<GpuHull2d>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let sets_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_sets"),
            contents: bytemuck::cast_slice(&gpu_sets),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_convex_hull_2d_bind_group"),
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
            label: Some("prism_volumetric_convex_hull_2d_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_convex_hull_2d_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuHull2d>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
