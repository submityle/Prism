// Linear-Swept-Spheres (LSS) per-segment conservative AABB twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::rt_curve::lss_segment_aabb`. One thread per
// LSS segment emits the conservative axis-aligned bounding box a downstream
// ray-tracing BLAS build registers for that swept-sphere primitive.
//
// The newest ray-tracing hardware (RTX via DXR / OptiX) expresses a hair strand
// segment as a Linear Swept Spheres / curve primitive the acceleration
// structure traverses directly, so real strands no longer need a proxy mesh.
// Each segment sweeps a sphere of radius `radius_a` at `a` to one of radius
// `radius_b` at `b`; its conservative box is the union of the two endpoint
// spheres' boxes. A swept-sphere endpoint box needs no square root - each axis
// is just the centre plus or minus its radius - so the whole kernel is
// `min`/`max` and single add/sub, with no multiply to fuse and no transcendental
// step.
//
// Because there is no multiply-add the compiler could contract, the kernel is
// *bit-exact* with the scalar reference (the parity test compares raw bit
// patterns, not a tolerance). Each endpoint and radius is sanitised in-shader
// bit-faithfully to the golden's `sanitized`: a non-finite position component
// collapses to `0`, and a negative or non-finite radius collapses to `0`, so a
// stray NaN can never poison a box.
//
// Provenance: hardware ray-traced curve / LSS strand primitive BLAS contract
// (RTX DXR / OptiX LSS) plus wgpu compute dispatch; no Unreal Engine source or
// derived code.

struct Params {
    // Number of LSS segments in `segments` / `out_boxes`.
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per segment, stride 8 f32: a.x, a.y, a.z, radius_a, b.x, b.y, b.z, radius_b.
@group(0) @binding(1) var<storage, read> segments: array<f32>;
// Per segment, stride 6 f32: min.x, min.y, min.z, max.x, max.y, max.z.
@group(0) @binding(2) var<storage, read_write> out_boxes: array<f32>;

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

fn sanitize_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sanitize_coord(p.x), sanitize_coord(p.y), sanitize_coord(p.z));
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
    let si = 8u * i;
    // Sanitise the raw segment exactly as the golden `LssSegment::sanitized`.
    let a = sanitize_point(vec3<f32>(segments[si], segments[si + 1u], segments[si + 2u]));
    let ra = sanitize_radius(segments[si + 3u]);
    let b = sanitize_point(vec3<f32>(segments[si + 4u], segments[si + 5u], segments[si + 6u]));
    let rb = sanitize_radius(segments[si + 7u]);

    // Endpoint boxes: centre +/- radius on each axis (no sqrt, no multiply).
    let box_a_min = a - vec3<f32>(ra, ra, ra);
    let box_a_max = a + vec3<f32>(ra, ra, ra);
    let box_b_min = b - vec3<f32>(rb, rb, rb);
    let box_b_max = b + vec3<f32>(rb, rb, rb);

    // Union: per-axis min of the mins, max of the maxes.
    let lo = min(box_a_min, box_b_min);
    let hi = max(box_a_max, box_b_max);

    let oi = 6u * i;
    out_boxes[oi] = lo.x;
    out_boxes[oi + 1u] = lo.y;
    out_boxes[oi + 2u] = lo.z;
    out_boxes[oi + 3u] = hi.x;
    out_boxes[oi + 4u] = hi.y;
    out_boxes[oi + 5u] = hi.z;
}
