// Per-control-point forward/backward-difference strand tangent twin: evaluates
// on-device the same unit tangents the CPU golden `strand_tangents` produces,
// one thread per control point across a flattened batch of strands.
//
// Ribbon/card expansion and anisotropic hair shading read a per-point tangent;
// the guide-to-render interpolator and the rotation-minimizing frame transport
// both seed from these tangents (design frames.rs). The CPU golden
// `strand_tangents` (in `prism_render_architecture::hair::frames`) walks each
// strand polyline: a strand with a single point (or none) takes a fixed
// fallback direction; an interior/root point takes the forward difference to the
// next point; the tip takes the backward difference from the previous point;
// each raw direction is then normalized, falling back to the fixed direction
// when the segment is (numerically) zero length.
//
// This differs from the sibling `frames.wesl` twin, which runs one thread per
// strand and transports a full orthonormal frame (tangent + normal + bitangent)
// along each strand with the double-reflection method. Here each thread owns one
// control point and emits only its tangent, so the batch is embarrassingly
// parallel over the flattened point stream with no per-strand serial transport.
//
// Every guard is reproduced bit-for-bit from the golden: the single-point /
// empty-strand fallback (`FALLBACK_TANGENT`), the forward-vs-backward branch at
// the tip, and `Vec3::normalize_or`'s zero-length guard (which compares the
// squared length against `f32::EPSILON`, not the solver's `EPS_LEN_SQ`). The
// kernel uses only compares, subtract, `dot` and a `sqrt` + reciprocal multiply
// in the portable core-WGSL subset - no `exp`, `pow` or optional feature - so
// this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard finite-difference polyline tangents plus a `wgpu`
// compute dispatch; no Unreal Engine source or derived code.

// Fallback tangent for degenerate strands, matching `frames::FALLBACK_TANGENT`:
// an arbitrary but fixed unit direction so a tangent is always defined.
const FALLBACK_TANGENT: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

// Matches `f32::EPSILON` exactly (2^-23). `Vec3::normalize_or` treats a vector
// whose squared length is <= this as zero-length and returns the fallback.
const F32_EPSILON: f32 = 0x1p-23f;

struct Params {
    // Number of control points in `points` / `meta`, and of 3-f32 slots in
    // `out_tangents`.
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened control points, 3 f32 per point in strand-concatenated order.
@group(0) @binding(1) var<storage, read> points: array<f32>;
// Per control point: `.x` is the global point index of its strand's first
// point (the strand offset), `.y` is the strand's control-point count.
@group(0) @binding(2) var<storage, read> strand_meta: array<vec2<u32>>;
// Per control point: 3 f32 (unit tangent).
@group(0) @binding(3) var<storage, read_write> out_tangents: array<f32>;

// Loads control point `idx` from the flattened `points` buffer.
fn load_point(idx: u32) -> vec3<f32> {
    let b = 3u * idx;
    return vec3<f32>(points[b], points[b + 1u], points[b + 2u]);
}

// Unit vector along `v`, or `fallback` when `v` is (numerically) zero length,
// mirroring `Vec3::normalize_or` (`len_sq <= f32::EPSILON` guard, then
// `v * (1 / sqrt(len_sq))`).
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= F32_EPSILON) {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= params.point_count) {
        return;
    }
    let offset = strand_meta[g].x;
    let count = strand_meta[g].y;
    // Local index of this point within its strand (`i` in the golden).
    let i = g - offset;
    var raw: vec3<f32>;
    if (count <= 1u) {
        // Single-point (or empty) strand: fixed fallback direction.
        raw = FALLBACK_TANGENT;
    } else if (i + 1u < count) {
        // Interior/root point: forward difference to the next point.
        raw = load_point(g + 1u) - load_point(g);
    } else {
        // Tip: backward difference from the previous point.
        raw = load_point(g) - load_point(g - 1u);
    }
    let t = normalize_or(raw, FALLBACK_TANGENT);
    let b = 3u * g;
    out_tangents[b] = t.x;
    out_tangents[b + 1u] = t.y;
    out_tangents[b + 2u] = t.z;
}
