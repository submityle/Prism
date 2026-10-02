//! `wgpu` compute twin of the clustered-forward light-sampling numeric contract
//! ([`light_clustered`](prism_render_architecture::particle::light_clustered),
//! particle design §8.3 *Scene* category, consumed by the §17 lighting
//! closures).
//!
//! The `CPU` golden
//! [`light_clustered`](prism_render_architecture::particle::light_clustered)
//! owns the closed-form, transcendental-free math a clustered-forward shader
//! needs per particle: the unit-interval clamp
//! ([`clamp01`](prism_render_architecture::particle::light_clustered::clamp01)),
//! the Hermite blend
//! ([`smoothstep`](prism_render_architecture::particle::light_clustered::smoothstep)),
//! the windowed inverse-square distance falloff
//! ([`distance_attenuation`](prism_render_architecture::particle::light_clustered::distance_attenuation)),
//! the cone falloff
//! ([`spot_attenuation`](prism_render_architecture::particle::light_clustered::spot_attenuation)),
//! the froxel lookups
//! ([`ClusterGrid::depth_slice`](prism_render_architecture::particle::light_clustered::ClusterGrid::depth_slice),
//! [`ClusterGrid::cluster_coord`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_coord),
//! [`ClusterGrid::linear_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::linear_index),
//! [`ClusterGrid::cluster_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_index)),
//! the punctual-light algebra
//! ([`PunctualLight::is_active`](prism_render_architecture::particle::light_clustered::PunctualLight::is_active),
//! [`PunctualLight::range_squared`](prism_render_architecture::particle::light_clustered::PunctualLight::range_squared)),
//! and the single-light diffuse response
//! ([`shade_point_light`](prism_render_architecture::particle::light_clustered::shade_point_light)).
//! [`GpuLightClustered`] is the on-device twin: one thread resolves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! evaluates the same closed form the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-element numeric primitive is reproduced through a tagged
//! [`LightClusteredQuery`]: one variant per reference routine. The froxel
//! lookups are twinned by budgeting the grid's monotonic depth-boundary array
//! into a fixed-length slot (at most [`MAX_SLICE_BOUNDARIES`] edges) the host
//! fills before dispatch, mirroring how the reference scans its boundary slice.
//!
//! # Correctness model
//!
//! Every term is a polynomial or rational function of its inputs with at most
//! one `sqrt` (the robust normalize inside the shade routine), so `CPU` and
//! `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous lane; the boolean active flag, the
//! integer slice / tile / cluster indices, and the valid / out-of-frustum flags
//! are compared exactly.
//!
//! # Degenerate inputs
//!
//! Every division is guarded exactly as the reference guards it: a non-positive
//! range yields a zero falloff, a near-equal `smoothstep` edge pair collapses to
//! a hard step, a zero-depth or zero-slope froxel column reports no cluster
//! rather than dividing, and a coincident light / surface or a zero normal
//! shades to black rather than producing `NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `min`, `max`, `floor`, `dot`, `sqrt` and `+ - * /` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no builtin `smoothstep`
//! (the Hermite blend is hand-expanded as `t * t * (3 - 2 * t)`), and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only loop is bounded by [`MAX_SLICE_BOUNDARIES`], so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! # Honest host boundary
//!
//! The variable-length container logic is deliberately *not* twinned, because it
//! is host-side allocation / indexing rather than per-element kernel math:
//! [`ClusterGrid::linear`](prism_render_architecture::particle::light_clustered::ClusterGrid::linear)
//! and
//! [`ClusterGrid::with_boundaries`](prism_render_architecture::particle::light_clustered::ClusterGrid::with_boundaries)
//! build the boundary `Vec`,
//! [`ClusterLightList::from_per_cluster`](prism_render_architecture::particle::light_clustered::ClusterLightList::from_per_cluster)
//! and
//! [`ClusterLightList::lights_for`](prism_render_architecture::particle::light_clustered::ClusterLightList::lights_for)
//! pack and read a flat index buffer, and
//! [`accumulate_lighting`](prism_render_architecture::particle::light_clustered::accumulate_lighting)
//! gathers over a variable-length cluster slice. The host feeds the twinned
//! froxel lookups a fixed-length boundary budget instead.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::light_clustered::{
    ClusterCoord, LightKind, PunctualLight,
};
use prism_render_architecture::particle::Vec3;
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
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Upper bound on the depth-boundary edges one froxel-lookup query carries. The
/// host budgets the grid's monotonic
/// [`ClusterGrid::slice_boundaries`](prism_render_architecture::particle::light_clustered::ClusterGrid::slice_boundaries)
/// into at most this many edges before dispatch, so a grid with up to `32`
/// depth slices (`33` edges) is twinnable without a variable-length device
/// array.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
pub const MAX_SLICE_BOUNDARIES: usize = 33;

/// The portable core-`WGSL` clustered-light-sampling kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// branches on a per-query op code into the `CPU` golden
/// [`light_clustered`](prism_render_architecture::particle::light_clustered)
/// terms; see the module documentation for the algorithm.
const LIGHT_CLUSTERED_WGSL: &str = r#"
// Clustered-forward light-sampling twin: one thread per query reproduces one
// reference numeric routine selected by `op`. It mirrors the CPU golden
// `particle::light_clustered` term for term, uses only the portable core-WGSL
// subset (abs/clamp/min/max/floor/dot/sqrt and + - * /), needs no transcendental
// call (the Hermite blend is hand-expanded, not the builtin), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. The only
// loop is bounded by MAX_SLICE_BOUNDARIES, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::light_clustered；
// 无第三方引擎源码或衍生代码。

// Shared tolerance, matching the reference EPS (division guard and comparison
// slack).
const EPS: f32 = 1.0e-6;
// Squared-length floor for a degenerate direction, matching EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Largest depth-slice count the budgeted boundary array can describe.
const MAX_SLICES: u32 = 32u;
// Light-kind codes shared with the host encoder.
const KIND_POINT: u32 = 0u;
const KIND_SPOT: u32 = 1u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Light / geometry vectors; each vec3 carries a trailing pad lane to stay
    // 16-byte aligned on device.
    light_position: vec3<f32>,
    pad_lp: f32,
    light_direction: vec3<f32>,
    pad_ld: f32,
    light_color_intensity: vec3<f32>,
    pad_lc: f32,
    surface_pos: vec3<f32>,
    pad_sp: f32,
    normal: vec3<f32>,
    pad_nm: f32,
    view_pos: vec3<f32>,
    pad_vp: f32,
    // Budgeted monotonic depth-slice boundaries for the froxel lookups.
    boundaries: array<f32, 33>,
    // Scalar inputs; a field a given op does not name is ignored.
    x: f32,
    edge0: f32,
    edge1: f32,
    distance_squared: f32,
    range: f32,
    cos_angle: f32,
    cos_inner: f32,
    cos_outer: f32,
    slope_x: f32,
    slope_y: f32,
    depth_z: f32,
    // Integer inputs / controls.
    boundary_count: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    slice_count: u32,
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
    kind: u32,
    op: u32,
    pad_q0: f32,
    pad_q1: f32,
    pad_q2: f32,
}

struct Result {
    // Up to four scalar outputs; the interpretation depends on the query op. For
    // the Option-returning froxel lookups, `a` is a 1.0 / 0.0 valid flag and
    // `b` (optionally `c`, `d`) carry the integer payload as exact f32. For
    // shade_point_light, a.b.c are the radiance xyz. Every single-valued op
    // leaves its answer in `a`.
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

struct OptU32 {
    valid: bool,
    value: u32,
}

struct OptCoord {
    valid: bool,
    x: u32,
    y: u32,
    z: u32,
}

// Clamps `x` to the unit interval; mirrors `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hand-expanded Hermite blend; mirrors `smoothstep` without the banned builtin.
// A near-equal edge pair collapses to a hard step at edge1 (denominator
// guarded), matching the reference.
fn smoothstep_poly(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if (abs(denom) <= EPS) {
        if (x >= edge1) {
            return 1.0;
        }
        return 0.0;
    }
    let t = clamp01((x - edge0) / denom);
    return t * t * (3.0 - 2.0 * t);
}

// Windowed inverse-square distance falloff; mirrors `distance_attenuation`.
fn distance_attenuation(distance_squared: f32, range: f32) -> f32 {
    if (range <= EPS) {
        return 0.0;
    }
    let r2 = range * range;
    if (distance_squared >= r2) {
        return 0.0;
    }
    let ratio2 = distance_squared / r2;
    let ratio4 = ratio2 * ratio2;
    let window = clamp01(1.0 - ratio4);
    let window_sq = window * window;
    let inv_sq = 1.0 / (distance_squared + EPS);
    return window_sq * inv_sq;
}

// Cone angular falloff; mirrors `spot_attenuation`.
fn spot_attenuation(cos_angle: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    return smoothstep_poly(cos_outer, cos_inner, cos_angle);
}

// Robust normalize; mirrors `Vec3::normalize_or_zero` (EPS_LEN_SQ threshold).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Single-light diffuse response; mirrors `shade_point_light`.
fn shade_point_light(q: Query) -> vec3<f32> {
    let zero = vec3<f32>(0.0, 0.0, 0.0);
    // is_active: non-positive range contributes nothing.
    if (!(q.range > EPS)) {
        return zero;
    }
    let to_light = q.light_position - q.surface_pos;
    let dist_sq = dot(to_light, to_light);
    let range_sq = q.range * q.range;
    if (dist_sq >= range_sq) {
        return zero;
    }
    let l = normalize_or_zero(to_light);
    if (dot(l, l) <= EPS) {
        return zero;
    }
    let n = normalize_or_zero(q.normal);
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= EPS) {
        return zero;
    }
    let atten = distance_attenuation(dist_sq, q.range);
    if (atten <= EPS) {
        return zero;
    }
    var spot = 1.0;
    if (q.kind == KIND_SPOT) {
        let cos_angle = dot(q.light_direction, l * -1.0);
        spot = spot_attenuation(cos_angle, q.cos_inner, q.cos_outer);
    }
    if (spot <= EPS) {
        return zero;
    }
    return q.light_color_intensity * (n_dot_l * atten * spot);
}

// Depth slice of a view depth against the budgeted boundary array; mirrors
// `ClusterGrid::depth_slice`.
fn depth_slice_scan(q: Query, z: f32) -> OptU32 {
    var r: OptU32;
    r.valid = false;
    r.value = 0u;
    let bcount = q.boundary_count;
    if (bcount < 2u) {
        return r;
    }
    let near = q.boundaries[0];
    let far = q.boundaries[bcount - 1u];
    if (z < near - EPS || z > far + EPS) {
        return r;
    }
    let slices = bcount - 1u;
    var i = 0u;
    loop {
        if (i >= slices || i >= MAX_SLICES) {
            break;
        }
        let lo = q.boundaries[i];
        let hi = q.boundaries[i + 1u];
        if (z >= lo - EPS && z <= hi + EPS) {
            r.valid = true;
            r.value = i;
            return r;
        }
        i = i + 1u;
    }
    // Numerically at/just past the far edge: clamp to the last slice.
    r.valid = true;
    r.value = slices - 1u;
    return r;
}

// Normalized screen coordinate to tile index; mirrors `ClusterGrid::ndc_to_tile`.
fn ndc_to_tile(ndc: f32, count: u32) -> OptU32 {
    var r: OptU32;
    r.valid = false;
    r.value = 0u;
    if (ndc < -1.0 - EPS || ndc > 1.0 + EPS) {
        return r;
    }
    let countf = f32(count);
    let scaled = (ndc + 1.0) * 0.5 * countf;
    let tile = floor(scaled);
    r.valid = true;
    if (tile < 0.0) {
        r.value = 0u;
    } else if (tile >= countf) {
        r.value = count - 1u;
    } else {
        r.value = u32(tile);
    }
    return r;
}

// View-space position to cluster coordinate; mirrors `ClusterGrid::cluster_coord`.
fn cluster_coord(q: Query) -> OptCoord {
    var r: OptCoord;
    r.valid = false;
    r.x = 0u;
    r.y = 0u;
    r.z = 0u;
    let z = q.view_pos.z;
    let ds = depth_slice_scan(q, z);
    if (!ds.valid) {
        return r;
    }
    let half_w = z * q.slope_x;
    let half_h = z * q.slope_y;
    if (half_w <= EPS || half_h <= EPS) {
        return r;
    }
    let ndc_x = q.view_pos.x / half_w;
    let ndc_y = q.view_pos.y / half_h;
    let tx = ndc_to_tile(ndc_x, q.tile_count_x);
    if (!tx.valid) {
        return r;
    }
    let ty = ndc_to_tile(ndc_y, q.tile_count_y);
    if (!ty.valid) {
        return r;
    }
    r.valid = true;
    r.x = tx.value;
    r.y = ty.value;
    r.z = ds.value;
    return r;
}

// Flattens a cluster coordinate to a linear index; mirrors
// `ClusterGrid::linear_index`.
fn linear_index(cx: u32, cy: u32, cz: u32, tcx: u32, tcy: u32, scount: u32) -> OptU32 {
    var r: OptU32;
    r.valid = false;
    r.value = 0u;
    if (cx >= tcx || cy >= tcy || cz >= scount) {
        return r;
    }
    let per_slice = tcx * tcy;
    r.valid = true;
    r.value = cz * per_slice + cy * tcx + cx;
    return r;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.a = 0.0;
    out.b = 0.0;
    out.c = 0.0;
    out.d = 0.0;

    if (q.op == 0u) {
        out.a = clamp01(q.x);
    } else if (q.op == 1u) {
        out.a = smoothstep_poly(q.edge0, q.edge1, q.x);
    } else if (q.op == 2u) {
        out.a = distance_attenuation(q.distance_squared, q.range);
    } else if (q.op == 3u) {
        out.a = spot_attenuation(q.cos_angle, q.cos_inner, q.cos_outer);
    } else if (q.op == 4u) {
        out.a = q.range * q.range;
    } else if (q.op == 5u) {
        // is_active: range > EPS, encoded as 1.0 / 0.0.
        if (q.range > EPS) {
            out.a = 1.0;
        } else {
            out.a = 0.0;
        }
    } else if (q.op == 6u) {
        let ds = depth_slice_scan(q, q.depth_z);
        if (ds.valid) {
            out.a = 1.0;
            out.b = f32(ds.value);
        }
    } else if (q.op == 7u) {
        let cc = cluster_coord(q);
        if (cc.valid) {
            out.a = 1.0;
            out.b = f32(cc.x);
            out.c = f32(cc.y);
            out.d = f32(cc.z);
        }
    } else if (q.op == 8u) {
        let li = linear_index(q.coord_x, q.coord_y, q.coord_z, q.tile_count_x, q.tile_count_y, q.slice_count);
        if (li.valid) {
            out.a = 1.0;
            out.b = f32(li.value);
        }
    } else if (q.op == 9u) {
        let cc = cluster_coord(q);
        if (cc.valid) {
            let scount = q.boundary_count - 1u;
            let li = linear_index(cc.x, cc.y, cc.z, q.tile_count_x, q.tile_count_y, scount);
            if (li.valid) {
                out.a = 1.0;
                out.b = f32(li.value);
            }
        }
    } else {
        // shade_point_light: diffuse radiance xyz.
        let rad = shade_point_light(q);
        out.a = rad.x;
        out.b = rad.y;
        out.c = rad.z;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`LIGHT_CLUSTERED_WGSL`].
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
/// on device; the trailing pad words fill the final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Light world/view-space position.
    light_position: [f32; 3],
    /// Pad lane after `light_position`.
    pad_lp: f32,
    /// Light cone axis (already normalized by the reference constructor).
    light_direction: [f32; 3],
    /// Pad lane after `light_direction`.
    pad_ld: f32,
    /// Pre-multiplied radiance (`color * intensity`).
    light_color_intensity: [f32; 3],
    /// Pad lane after `light_color_intensity`.
    pad_lc: f32,
    /// Shaded surface position.
    surface_pos: [f32; 3],
    /// Pad lane after `surface_pos`.
    pad_sp: f32,
    /// Surface normal (normalized robustly on device).
    normal: [f32; 3],
    /// Pad lane after `normal`.
    pad_nm: f32,
    /// View-space position for the froxel lookups.
    view_pos: [f32; 3],
    /// Pad lane after `view_pos`.
    pad_vp: f32,
    /// Budgeted monotonic depth-slice boundaries.
    boundaries: [f32; MAX_SLICE_BOUNDARIES],
    /// Scalar input for `clamp01` / `smoothstep`.
    x: f32,
    /// Lower `smoothstep` edge.
    edge0: f32,
    /// Upper `smoothstep` edge.
    edge1: f32,
    /// Squared distance for the distance falloff.
    distance_squared: f32,
    /// Light range (also drives `range_squared` and `is_active`).
    range: f32,
    /// Alignment cosine for the spot falloff.
    cos_angle: f32,
    /// Inner cone cosine (full brightness at/above).
    cos_inner: f32,
    /// Outer cone cosine (zero brightness at/below).
    cos_outer: f32,
    /// Per-axis frustum slope along screen X.
    slope_x: f32,
    /// Per-axis frustum slope along screen Y.
    slope_y: f32,
    /// Standalone view depth for the depth-slice op.
    depth_z: f32,
    /// Number of live boundary edges (`slice_count + 1`).
    boundary_count: u32,
    /// Tile columns along screen X.
    tile_count_x: u32,
    /// Tile rows along screen Y.
    tile_count_y: u32,
    /// Depth-slice count for the standalone linear-index op.
    slice_count: u32,
    /// Cluster coordinate X for the linear-index op.
    coord_x: u32,
    /// Cluster coordinate Y for the linear-index op.
    coord_y: u32,
    /// Cluster coordinate Z for the linear-index op.
    coord_z: u32,
    /// Light-kind code (`0` point, `1` spot).
    kind: u32,
    /// Op classification code (`0..=10`).
    op: u32,
    /// Padding lane.
    pad_q0: f32,
    /// Padding lane.
    pad_q1: f32,
    /// Padding lane.
    pad_q2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: up to four scalar outputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First output lane (single answer / valid flag / radiance x).
    a: f32,
    /// Second output lane.
    b: f32,
    /// Third output lane.
    c: f32,
    /// Fourth output lane.
    d: f32,
}

/// One tagged query selecting which reference routine the kernel evaluates.
///
/// There is one variant per twinned `CPU` golden routine; a field a variant does
/// not name is ignored. The discriminant order matches the `u32` op codes the
/// kernel branches on.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightClusteredQuery {
    /// The unit-interval clamp, matching
    /// [`clamp01`](prism_render_architecture::particle::light_clustered::clamp01).
    Clamp01 {
        /// Value to clamp.
        x: f32,
    },
    /// The Hermite blend, matching
    /// [`smoothstep`](prism_render_architecture::particle::light_clustered::smoothstep).
    Smoothstep {
        /// Lower edge.
        edge0: f32,
        /// Upper edge.
        edge1: f32,
        /// Interpolant.
        x: f32,
    },
    /// The windowed inverse-square distance falloff, matching
    /// [`distance_attenuation`](prism_render_architecture::particle::light_clustered::distance_attenuation).
    DistanceAttenuation {
        /// Squared distance to the light.
        distance_squared: f32,
        /// Light influence range.
        range: f32,
    },
    /// The cone angular falloff, matching
    /// [`spot_attenuation`](prism_render_architecture::particle::light_clustered::spot_attenuation).
    SpotAttenuation {
        /// Alignment cosine `dot(cone_axis, light_to_point)`.
        cos_angle: f32,
        /// Inner cone cosine.
        cos_inner: f32,
        /// Outer cone cosine.
        cos_outer: f32,
    },
    /// The squared influence radius, matching
    /// [`PunctualLight::range_squared`](prism_render_architecture::particle::light_clustered::PunctualLight::range_squared).
    RangeSquared {
        /// Light influence range.
        range: f32,
    },
    /// The usable-range test, matching
    /// [`PunctualLight::is_active`](prism_render_architecture::particle::light_clustered::PunctualLight::is_active).
    IsActive {
        /// Light influence range.
        range: f32,
    },
    /// The depth-slice lookup, matching
    /// [`ClusterGrid::depth_slice`](prism_render_architecture::particle::light_clustered::ClusterGrid::depth_slice).
    DepthSlice {
        /// Budgeted monotonic depth-slice boundaries.
        boundaries: [f32; MAX_SLICE_BOUNDARIES],
        /// Number of live boundary edges (`slice_count + 1`).
        boundary_count: u32,
        /// View depth to classify.
        z: f32,
    },
    /// The cluster-coordinate lookup, matching
    /// [`ClusterGrid::cluster_coord`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_coord).
    ClusterCoord {
        /// Tile columns along screen X.
        tile_count_x: u32,
        /// Tile rows along screen Y.
        tile_count_y: u32,
        /// Per-axis frustum slope along screen X.
        slope_x: f32,
        /// Per-axis frustum slope along screen Y.
        slope_y: f32,
        /// Budgeted monotonic depth-slice boundaries.
        boundaries: [f32; MAX_SLICE_BOUNDARIES],
        /// Number of live boundary edges (`slice_count + 1`).
        boundary_count: u32,
        /// View-space position to classify.
        view_pos: Vec3,
    },
    /// The coordinate-flattening lookup, matching
    /// [`ClusterGrid::linear_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::linear_index).
    LinearIndex {
        /// Tile columns along screen X.
        tile_count_x: u32,
        /// Tile rows along screen Y.
        tile_count_y: u32,
        /// Depth-slice count.
        slice_count: u32,
        /// Cluster coordinate to flatten.
        coord: ClusterCoord,
    },
    /// The position-to-index convenience, matching
    /// [`ClusterGrid::cluster_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_index).
    ClusterIndex {
        /// Tile columns along screen X.
        tile_count_x: u32,
        /// Tile rows along screen Y.
        tile_count_y: u32,
        /// Per-axis frustum slope along screen X.
        slope_x: f32,
        /// Per-axis frustum slope along screen Y.
        slope_y: f32,
        /// Budgeted monotonic depth-slice boundaries.
        boundaries: [f32; MAX_SLICE_BOUNDARIES],
        /// Number of live boundary edges (`slice_count + 1`).
        boundary_count: u32,
        /// View-space position to classify.
        view_pos: Vec3,
    },
    /// The single-light diffuse response, matching
    /// [`shade_point_light`](prism_render_architecture::particle::light_clustered::shade_point_light).
    ShadePointLight {
        /// The punctual light to shade with.
        light: PunctualLight,
        /// The shaded surface position.
        surface_pos: Vec3,
        /// The surface normal (normalized robustly on device).
        normal: Vec3,
    },
}

impl LightClusteredQuery {
    /// Returns the `u32` op code the kernel branches on for this variant.
    #[must_use]
    const fn code(&self) -> u32 {
        match self {
            LightClusteredQuery::Clamp01 { .. } => 0,
            LightClusteredQuery::Smoothstep { .. } => 1,
            LightClusteredQuery::DistanceAttenuation { .. } => 2,
            LightClusteredQuery::SpotAttenuation { .. } => 3,
            LightClusteredQuery::RangeSquared { .. } => 4,
            LightClusteredQuery::IsActive { .. } => 5,
            LightClusteredQuery::DepthSlice { .. } => 6,
            LightClusteredQuery::ClusterCoord { .. } => 7,
            LightClusteredQuery::LinearIndex { .. } => 8,
            LightClusteredQuery::ClusterIndex { .. } => 9,
            LightClusteredQuery::ShadePointLight { .. } => 10,
        }
    }
}

/// One resolved answer, tagged to match the query variant that produced it.
///
/// Each variant carries exactly the term(s) the matching [`LightClusteredQuery`]
/// variant selects.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightClusteredResult {
    /// The clamped value.
    Clamp01 {
        /// `clamp(x, 0, 1)`.
        value: f32,
    },
    /// The Hermite blend value.
    Smoothstep {
        /// `t * t * (3 - 2 * t)` on the clamped, normalized parameter.
        value: f32,
    },
    /// The distance falloff.
    DistanceAttenuation {
        /// Windowed inverse-square attenuation.
        value: f32,
    },
    /// The cone falloff.
    SpotAttenuation {
        /// Angular attenuation.
        value: f32,
    },
    /// The squared influence radius.
    RangeSquared {
        /// `range * range`.
        value: f32,
    },
    /// The usable-range test.
    IsActive {
        /// Whether the light has a strictly positive range.
        active: bool,
    },
    /// The depth-slice lookup.
    DepthSlice {
        /// The slice index, or `None` when outside the near/far range.
        slice: Option<u32>,
    },
    /// The cluster-coordinate lookup.
    ClusterCoord {
        /// The cluster coordinate, or `None` when outside the frustum.
        coord: Option<ClusterCoord>,
    },
    /// The coordinate-flattening lookup.
    LinearIndex {
        /// The linear cluster index, or `None` when any component is out of range.
        index: Option<u32>,
    },
    /// The position-to-index convenience.
    ClusterIndex {
        /// The linear cluster index, or `None` when outside the frustum.
        index: Option<u32>,
    },
    /// The single-light diffuse response.
    ShadePointLight {
        /// The accumulated radiance `radiance * n_dot_l * falloff`.
        radiance: [f32; 3],
    },
}

/// Encodes one [`LightClusteredQuery`] into its `std430` [`GpuQuery`] slot,
/// zeroing the lanes the chosen op does not read.
fn encode_query(query: &LightClusteredQuery) -> GpuQuery {
    let mut gpu = GpuQuery {
        light_position: [0.0; 3],
        pad_lp: 0.0,
        light_direction: [0.0; 3],
        pad_ld: 0.0,
        light_color_intensity: [0.0; 3],
        pad_lc: 0.0,
        surface_pos: [0.0; 3],
        pad_sp: 0.0,
        normal: [0.0; 3],
        pad_nm: 0.0,
        view_pos: [0.0; 3],
        pad_vp: 0.0,
        boundaries: [0.0; MAX_SLICE_BOUNDARIES],
        x: 0.0,
        edge0: 0.0,
        edge1: 0.0,
        distance_squared: 0.0,
        range: 0.0,
        cos_angle: 0.0,
        cos_inner: 0.0,
        cos_outer: 0.0,
        slope_x: 0.0,
        slope_y: 0.0,
        depth_z: 0.0,
        boundary_count: 0,
        tile_count_x: 0,
        tile_count_y: 0,
        slice_count: 0,
        coord_x: 0,
        coord_y: 0,
        coord_z: 0,
        kind: 0,
        op: query.code(),
        pad_q0: 0.0,
        pad_q1: 0.0,
        pad_q2: 0.0,
    };
    match *query {
        LightClusteredQuery::Clamp01 { x } => {
            gpu.x = x;
        }
        LightClusteredQuery::Smoothstep { edge0, edge1, x } => {
            gpu.edge0 = edge0;
            gpu.edge1 = edge1;
            gpu.x = x;
        }
        LightClusteredQuery::DistanceAttenuation {
            distance_squared,
            range,
        } => {
            gpu.distance_squared = distance_squared;
            gpu.range = range;
        }
        LightClusteredQuery::SpotAttenuation {
            cos_angle,
            cos_inner,
            cos_outer,
        } => {
            gpu.cos_angle = cos_angle;
            gpu.cos_inner = cos_inner;
            gpu.cos_outer = cos_outer;
        }
        LightClusteredQuery::RangeSquared { range } | LightClusteredQuery::IsActive { range } => {
            gpu.range = range;
        }
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z,
        } => {
            gpu.boundaries = boundaries;
            gpu.boundary_count = boundary_count;
            gpu.depth_z = z;
        }
        LightClusteredQuery::ClusterCoord {
            tile_count_x,
            tile_count_y,
            slope_x,
            slope_y,
            boundaries,
            boundary_count,
            view_pos,
        }
        | LightClusteredQuery::ClusterIndex {
            tile_count_x,
            tile_count_y,
            slope_x,
            slope_y,
            boundaries,
            boundary_count,
            view_pos,
        } => {
            gpu.tile_count_x = tile_count_x;
            gpu.tile_count_y = tile_count_y;
            gpu.slope_x = slope_x;
            gpu.slope_y = slope_y;
            gpu.boundaries = boundaries;
            gpu.boundary_count = boundary_count;
            gpu.view_pos = [view_pos.x, view_pos.y, view_pos.z];
        }
        LightClusteredQuery::LinearIndex {
            tile_count_x,
            tile_count_y,
            slice_count,
            coord,
        } => {
            gpu.tile_count_x = tile_count_x;
            gpu.tile_count_y = tile_count_y;
            gpu.slice_count = slice_count;
            gpu.coord_x = coord.x;
            gpu.coord_y = coord.y;
            gpu.coord_z = coord.z;
        }
        LightClusteredQuery::ShadePointLight {
            light,
            surface_pos,
            normal,
        } => {
            gpu.light_position = [light.position.x, light.position.y, light.position.z];
            gpu.light_direction = [light.direction.x, light.direction.y, light.direction.z];
            gpu.light_color_intensity = [
                light.color_intensity.x,
                light.color_intensity.y,
                light.color_intensity.z,
            ];
            gpu.range = light.range;
            gpu.cos_inner = light.cos_inner;
            gpu.cos_outer = light.cos_outer;
            gpu.kind = kind_code(light.kind);
            gpu.surface_pos = [surface_pos.x, surface_pos.y, surface_pos.z];
            gpu.normal = [normal.x, normal.y, normal.z];
        }
    }
    gpu
}

/// Maps a [`LightKind`] to the `u32` code the kernel and encoder share.
#[must_use]
const fn kind_code(kind: LightKind) -> u32 {
    match kind {
        LightKind::Point => 0,
        LightKind::Spot => 1,
    }
}

/// Reconstructs an `Option<u32>` from a valid flag and an exact-integer f32
/// payload.
#[must_use]
fn decode_opt_u32(valid: f32, payload: f32) -> Option<u32> {
    if valid > 0.5 {
        Some(payload.round() as u32)
    } else {
        None
    }
}

/// Decodes one packed [`GpuResult`] into the public [`LightClusteredResult`],
/// selecting the fields the originating `query` variant produced.
fn decode_result(query: &LightClusteredQuery, raw: &GpuResult) -> LightClusteredResult {
    match query {
        LightClusteredQuery::Clamp01 { .. } => LightClusteredResult::Clamp01 { value: raw.a },
        LightClusteredQuery::Smoothstep { .. } => LightClusteredResult::Smoothstep { value: raw.a },
        LightClusteredQuery::DistanceAttenuation { .. } => {
            LightClusteredResult::DistanceAttenuation { value: raw.a }
        }
        LightClusteredQuery::SpotAttenuation { .. } => {
            LightClusteredResult::SpotAttenuation { value: raw.a }
        }
        LightClusteredQuery::RangeSquared { .. } => {
            LightClusteredResult::RangeSquared { value: raw.a }
        }
        LightClusteredQuery::IsActive { .. } => LightClusteredResult::IsActive {
            active: raw.a > 0.5,
        },
        LightClusteredQuery::DepthSlice { .. } => LightClusteredResult::DepthSlice {
            slice: decode_opt_u32(raw.a, raw.b),
        },
        LightClusteredQuery::ClusterCoord { .. } => {
            let coord = if raw.a > 0.5 {
                Some(ClusterCoord {
                    x: raw.b.round() as u32,
                    y: raw.c.round() as u32,
                    z: raw.d.round() as u32,
                })
            } else {
                None
            };
            LightClusteredResult::ClusterCoord { coord }
        }
        LightClusteredQuery::LinearIndex { .. } => LightClusteredResult::LinearIndex {
            index: decode_opt_u32(raw.a, raw.b),
        },
        LightClusteredQuery::ClusterIndex { .. } => LightClusteredResult::ClusterIndex {
            index: decode_opt_u32(raw.a, raw.b),
        },
        LightClusteredQuery::ShadePointLight { .. } => LightClusteredResult::ShadePointLight {
            radiance: [raw.a, raw.b, raw.c],
        },
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

/// A compiled, reusable clustered-light-sampling compute pipeline, twinning the
/// `CPU` golden
/// [`light_clustered`](prism_render_architecture::particle::light_clustered).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
pub struct GpuLightClustered {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLightClustered {
    /// Compiles the clustered-light-sampling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLightClustered {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_light_clustered"),
            source: ShaderSource::Wgsl(LIGHT_CLUSTERED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_light_clustered_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_light_clustered_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_light_clustered_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLightClustered {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`LightClusteredResult`] per input, in order.
    ///
    /// Each result matches the `CPU` golden within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[LightClusteredQuery],
    ) -> Vec<LightClusteredResult> {
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
            label: Some("prism_volumetric_light_clustered_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_light_clustered_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_light_clustered_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_light_clustered_bind_group"),
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
            label: Some("prism_volumetric_light_clustered_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_light_clustered_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_light_clustered_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
