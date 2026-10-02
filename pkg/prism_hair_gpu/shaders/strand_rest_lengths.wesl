// Per-edge strand rest-length twin: evaluates on-device the same per-segment
// rest lengths the CPU golden `strand_rest_lengths` produces, one thread per
// edge across a flattened batch of strands.
//
// The guide XPBD solver (design dynamics.rs) stores one *rest length* per edge
// of a resampled strand - the distance between consecutive control points - as
// the target its edge-length constraint relaxes toward. The CPU golden
// `strand_rest_lengths` (in `prism_render_architecture::hair::groom_import`)
// emits one length per edge for every strand with at least two control points
// (`count - 1` entries) and an empty list for a strand shorter than two points.
// This crate is the on-device twin that precomputes exactly that rest-length
// buffer so a passing real-device parity test is direct evidence the ported
// kernel computes the same lengths as the reference, not merely that its shader
// compiles.
//
// This is deliberately a different dispatch from the sibling `strand_metrics`
// twin, which runs one thread per strand and *sums* the segment lengths into a
// single scalar arc length. Here each thread owns one edge and emits that
// edge's bare length, so the output has `count - 1` entries per strand (not one)
// and the batch is embarrassingly parallel over a flattened edge stream with no
// per-strand reduction.
//
// The length is `sqrt(dot(d, d))` of the edge vector `d = p1 - p0`, matching
// `Vec3::length` exactly (no zero-length guard: a degenerate edge yields `0`).
// The kernel uses only subtract, `dot` and `sqrt` in the portable core-WGSL
// subset - no `exp`, `pow` or optional feature - so this twin runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard polyline per-segment lengths plus a `wgpu` compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of edges in `edge_p0` / `out_lengths`.
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened control points, 3 f32 per point in strand-concatenated order.
@group(0) @binding(1) var<storage, read> points: array<f32>;
// Per edge: the global point index of the edge's first endpoint (`p0`); the
// second endpoint is `p0 + 1`, always within the same strand.
@group(0) @binding(2) var<storage, read> edge_p0: array<u32>;
// Per edge: one f32 (rest length).
@group(0) @binding(3) var<storage, read_write> out_lengths: array<f32>;

// Loads control point `idx` from the flattened `points` buffer.
fn load_point(idx: u32) -> vec3<f32> {
    let b = 3u * idx;
    return vec3<f32>(points[b], points[b + 1u], points[b + 2u]);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= params.edge_count) {
        return;
    }
    let p0 = edge_p0[e];
    let d = load_point(p0 + 1u) - load_point(p0);
    out_lengths[e] = sqrt(dot(d, d));
}
