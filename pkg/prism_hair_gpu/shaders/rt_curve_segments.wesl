// Strand-polyline to Linear-Swept-Spheres (LSS) segment conversion twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::rt_curve::strand_to_lss`. One thread per
// output segment turns a strand polyline (vertices, each with a radius) into the
// per-segment LSS primitives a downstream ray-tracing BLAS build registers.
//
// Adjacent vertices form one segment each, so `n` vertices yield `n - 1`
// segments in input order; segment `i` sweeps the sanitised sphere at vertex
// `i` to the sanitised sphere at vertex `i + 1`. The kernel does no arithmetic
// beyond the per-component sanitiser - it copies sanitised positions and radii
// into endpoint pairs - so there is no multiply to fuse and the twin is
// *bit-exact* with the scalar reference (the parity test compares raw bit
// patterns, not a tolerance).
//
// Each position component and radius is sanitised in-shader bit-faithfully to
// the golden's `CurveVertex::sanitized`: a non-finite position component
// collapses to `0`, and a negative or non-finite radius collapses to `0`, so a
// stray NaN can never poison a segment.
//
// Provenance: hardware ray-traced curve / LSS strand primitive BLAS contract
// (RTX DXR / OptiX LSS) plus wgpu compute dispatch; no Unreal Engine source or
// derived code.

struct Params {
    // Number of output segments (= max(0, vertex_count - 1)).
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per vertex, stride 4 f32: position.x, position.y, position.z, radius.
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
// Per segment, stride 8 f32: a.x, a.y, a.z, radius_a, b.x, b.y, b.z, radius_b.
@group(0) @binding(2) var<storage, read_write> segments: array<f32>;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Force a non-finite coordinate to 0 (the golden's `sanitize_coord`).
fn sanitize_coord(v: f32) -> f32 {
    return select(0.0, v, finite_scalar(v));
}

// Clamp a radius to a finite, non-negative value (the golden's
// `sanitize_radius`): negative or non-finite radii collapse to 0.
fn sanitize_radius(r: f32) -> f32 {
    return select(0.0, r, finite_scalar(r) && r >= 0.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.segment_count) {
        return;
    }
    // Segment i joins vertex i and vertex i + 1.
    let v0 = 4u * i;
    let v1 = 4u * (i + 1u);

    let ax = sanitize_coord(vertices[v0]);
    let ay = sanitize_coord(vertices[v0 + 1u]);
    let az = sanitize_coord(vertices[v0 + 2u]);
    let ra = sanitize_radius(vertices[v0 + 3u]);

    let bx = sanitize_coord(vertices[v1]);
    let by = sanitize_coord(vertices[v1 + 1u]);
    let bz = sanitize_coord(vertices[v1 + 2u]);
    let rb = sanitize_radius(vertices[v1 + 3u]);

    let oi = 8u * i;
    segments[oi] = ax;
    segments[oi + 1u] = ay;
    segments[oi + 2u] = az;
    segments[oi + 3u] = ra;
    segments[oi + 4u] = bx;
    segments[oi + 5u] = by;
    segments[oi + 6u] = bz;
    segments[oi + 7u] = rb;
}
