// Projective-Dynamics local edge-projection twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::projective_global::local_project_edge`. One
// thread per edge projects a distance (edge-length) constraint: it recenters the
// edge on its midpoint and places the two endpoints symmetrically about that
// midpoint so the separation equals the (non-negative) rest length. This is the
// per-constraint *local* step of Projective Dynamics (Bouaziz 2014) / the
// position-based distance projection; the global solve consuming a batch of
// these targets runs elsewhere.
//
// Each endpoint and the rest length are sanitised in-shader bit-faithfully to
// the golden: a non-finite position component collapses to `0`, and a negative
// or non-finite rest length collapses to `0`. A degenerate (near-zero) edge has
// no direction, so it is split along the canonical `+x` axis instead of
// producing a NaN.
//
// The midpoint, halving and endpoint add/sub are exact (0.5 is a power of two),
// but the projection scales the edge vector by `rest / length`, a divide the GPU
// may round differently from the scalar reference. The twin is therefore matched
// against a tolerance (abs_diff < 1e-4 or rel_diff < 1e-3), not bit-for-bit.
//
// The kernel uses only `sqrt` (via `length`), a divide and single add/sub/mul in
// the portable core-WGSL subset - no `exp`, `pow` or optional device feature -
// so the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: Projective Dynamics / position-based distance projection
// (Bouaziz 2014) plus wgpu compute dispatch; no Unreal Engine source or derived
// code.

struct Params {
    // Number of edges in `edges` / `out_points`.
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per edge, stride 7 f32: xi.x, xi.y, xi.z, xj.x, xj.y, xj.z, rest.
@group(0) @binding(1) var<storage, read> edges: array<f32>;
// Per edge, stride 6 f32: new_xi.x, new_xi.y, new_xi.z, new_xj.x, new_xj.y, new_xj.z.
@group(0) @binding(2) var<storage, read_write> out_points: array<f32>;

// Vectors shorter than this are treated as having no well-defined direction
// (the golden's `EPS_LEN`).
const EPS_LEN: f32 = 1.0e-12;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Force a non-finite coordinate to 0 (the golden's `sanitize_finite`).
fn sanitize_finite(v: f32) -> f32 {
    return select(0.0, v, finite_scalar(v));
}

fn sanitize_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sanitize_finite(p.x), sanitize_finite(p.y), sanitize_finite(p.z));
}

// Clamp a rest length to a finite, non-negative value (the golden's
// `sanitize_nonneg`): negative or non-finite values collapse to 0.
fn sanitize_nonneg(v: f32) -> f32 {
    return select(0.0, max(v, 0.0), finite_scalar(v));
}

// Project a direction vector onto the sphere of radius `rest`. A near-zero input
// has no direction, so it maps to `rest` along `+x` to stay finite (the golden's
// `project_to_length`).
fn project_to_length(v: vec3<f32>, rest: f32) -> vec3<f32> {
    let len = length(v);
    if (len > EPS_LEN) {
        return v * (rest / len);
    }
    return vec3<f32>(rest, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.edge_count) {
        return;
    }
    let si = 7u * i;
    // Sanitise the raw edge exactly as the golden before projecting.
    let a = sanitize_point(vec3<f32>(edges[si], edges[si + 1u], edges[si + 2u]));
    let b = sanitize_point(vec3<f32>(edges[si + 3u], edges[si + 4u], edges[si + 5u]));
    let rest = sanitize_nonneg(edges[si + 6u]);

    let mid = (a + b) * 0.5;
    let dir = project_to_length(a - b, rest);
    let half = dir * 0.5;
    let xi = mid + half;
    let xj = mid - half;

    let oi = 6u * i;
    out_points[oi] = xi.x;
    out_points[oi + 1u] = xi.y;
    out_points[oi + 2u] = xi.z;
    out_points[oi + 3u] = xj.x;
    out_points[oi + 4u] = xj.y;
    out_points[oi + 5u] = xj.z;
}
