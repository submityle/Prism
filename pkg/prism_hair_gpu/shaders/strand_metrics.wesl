// Per-strand geometry-metrics twin: evaluates on-device the same arc length and
// transcendental-free turning (curvature proxy) the CPU goldens
// `strand_arc_length` and `strand_curvature` compute, one thread per strand.
//
// The density/decimation stage ranks render strands for continuous LOD thinning
// by how long and how curly each one is: a longer, curlier strand carries more
// silhouette and must survive to a coarser tier than a short, straight one. The
// CPU goldens for those two per-strand scalars are
// `prism_render_architecture::hair::decimation::{strand_arc_length,
// strand_curvature}`; this kernel mirrors their bodies value-for-value over a
// strand-major fixed-stride point pool (the `ResampledGroom` layout), so a
// passing real-device parity test is direct evidence the ported kernel computes
// the same metrics as the reference, not merely that it compiles.
//
// Arc length sums the segment lengths in the same root->tip order; a strand of
// fewer than two points yields 0. Curvature sums `1 - dot(t_in, t_out)` over the
// interior vertices, with each tangent normalized by the same zero-length guard
// (`normalize_or` falling back to the zero vector when the squared length is at
// or below `f32::EPSILON`), so coincident control points contribute a defined
// (zero-tangent) term rather than a divide-by-zero; a strand of fewer than three
// points yields 0. Both accumulate in the reference's ascending index order.
//
// It uses only the portable core-WGSL subset (sqrt, dot, multiply/add) - no exp,
// no pow, no optional feature - so this twin runs unmodified on Metal, Vulkan
// and DX12. It is not bit-exact with the scalar CPU reference: a GPU may fuse a
// multiply-add, perturbing the low mantissa bits, so the parity test asserts a
// per-component tolerance rather than exact equality.
//
// Provenance: standard polyline arc-length / turning-angle geometry metrics plus
// wgpu compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of strands, bounding the dispatch.
    strand_count: u32,
    // Control points per strand (the fixed `ResampledGroom` stride).
    points_per_strand: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Strand-major control points, stride 3 f32 (xyz); strand `s` owns the
// `points_per_strand` points starting at `s * points_per_strand`.
@group(0) @binding(1) var<storage, read> points: array<f32>;
// Per-strand output, stride 2 f32: (arc_length, curvature).
@group(0) @binding(2) var<storage, read_write> metrics: array<f32>;

// The interpolation `Vec3::normalize_or` zero-length threshold: `f32::EPSILON`
// (2^-23) bit-for-bit, so the compare splits exactly as the reference.
fn eps() -> f32 {
    return bitcast<f32>(0x34000000u);
}

// Mirrors `Vec3::normalize_or(fallback)`: unit vector unless the squared length
// is at or below `f32::EPSILON`, in which case the fallback is returned.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= eps()) {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

fn load_point(base: u32, i: u32) -> vec3<f32> {
    let b = base + 3u * i;
    return vec3<f32>(points[b], points[b + 1u], points[b + 2u]);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let n = params.points_per_strand;
    let base = s * n * 3u;

    // Arc length: sum of segment lengths (0 for fewer than two points).
    var arc_length = 0.0;
    if (n >= 2u) {
        for (var i: u32 = 1u; i < n; i = i + 1u) {
            let d = load_point(base, i) - load_point(base, i - 1u);
            arc_length = arc_length + length(d);
        }
    }

    // Curvature: sum of `1 - dot(t_in, t_out)` over interior vertices (0 for
    // fewer than three points), each tangent guarded by `normalize_or`.
    var curvature = 0.0;
    if (n >= 3u) {
        let zero = vec3<f32>(0.0, 0.0, 0.0);
        for (var i: u32 = 1u; i < n - 1u; i = i + 1u) {
            let t_in = normalize_or(load_point(base, i) - load_point(base, i - 1u), zero);
            let t_out = normalize_or(load_point(base, i + 1u) - load_point(base, i), zero);
            curvature = curvature + (1.0 - dot(t_in, t_out));
        }
    }

    let out = 2u * s;
    metrics[out] = arc_length;
    metrics[out + 1u] = curvature;
}
