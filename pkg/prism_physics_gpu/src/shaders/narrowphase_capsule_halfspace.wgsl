// Capsule-versus-halfspace narrow-phase contact kernel.
//
// One invocation per (capsule, plane) candidate couple. Each invocation reads a
// capsule (segment endpoints p0, p1 and a swept radius) and a plane (unit
// outward normal and offset), runs the same affine-support overlap test and
// manifold construction the CPU twin runs (`narrowphase/capsule_halfspace.rs`),
// and writes one manifold slot: the plane's outward normal (the push-out
// direction), a live point count, and up to two contact points, one per
// penetrating axis endpoint. A couple whose capsule floats clear or only grazes
// the surface writes a zeroed slot with count = 0, so the output index stays
// aligned with the input couple index.
//
// Geometry: a plane is a unit outward normal `n` and offset `d`; the surface is
// `dot(n, x) = d` and the solid occupies `dot(n, x) <= d`. The signed distance
// `dot(n, x) - d` is affine along the capsule axis, so its minimum is always at
// an endpoint: each endpoint `p_i` behaves like a sphere of radius `rc` centred
// there, with centre signed distance `s_i = dot(n, p_i) - d`. That end
// penetrates when `s_i < rc`, by `depth_i = rc - s_i`, and the contact point is
// `p_i` projected onto the surface, `p_i - n * s_i`.
//
// Two-point manifold: a capsule lying flat on a plane rests on a segment, so
// both penetrating ends are reported (count = 2); a capsule on end or a
// degenerate zero-length capsule reports one. This is the manifold a solver
// needs to hold a horizontal capsule still, not a stub.
//
// Degenerate segment: when `dot(ab, ab) <= SEG_EPS2` the capsule is a sphere, so
// only the p0 endpoint is tested and the two coincident ends never emit a
// duplicate point. The threshold matches the CPU constant so both paths collapse
// on the same capsules.
//
// The arithmetic is bit-for-bit with the twin: the endpoint dot products, the
// strict `s_i < rc` rejections, and the surface projections, with no square root
// or reciprocal on this path, so the parity test matches the point count exactly
// and the normal, positions, and depths to within the tightest float tolerance.
//
// Provenance: textbook capsule-versus-halfspace (affine support) collision
// manifold. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// One capsule: (p0.xyz, radius) then (p1.xyz, pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
};

// One output manifold: (normal.xyz, count) then up to four (position.xyz, depth)
// points. Slots beyond `count` are zeroed.
struct Manifold {
    normal_count: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

// Squared-length threshold below which the capsule axis is a single point.
const SEG_EPS2: f32 = 1.0e-12;

@group(0) @binding(0) var<uniform> params: Params;
// Capsules, two vec4s each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Planes: xyz outward unit normal, w offset.
@group(0) @binding(2) var<storage, read> planes: array<vec4<f32>>;
// Candidate couples, one (capsule, plane) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one slot per couple.
@group(0) @binding(4) var<storage, read_write> manifolds: array<Manifold>;

@compute @workgroup_size(64)
fn narrowphase_capsule_halfspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let couple = pairs[i];
    let cap = capsules[couple.x];
    let plane = planes[couple.y];

    let p0 = cap.p0_radius.xyz;
    let rc = cap.p0_radius.w;
    let p1 = cap.p1_pad.xyz;
    let n = plane.xyz;
    let d = plane.w;

    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // A degenerate (zero-length) axis is a sphere: test only p0.
    let ab = p1 - p0;
    let degenerate = dot(ab, ab) <= SEG_EPS2;

    var count = 0u;
    var pts: array<vec4<f32>, 2>;
    pts[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    pts[1] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // Endpoint p0: signed distance of its centre, strict penetration test.
    let s0 = dot(n, p0) - d;
    if (s0 < rc) {
        pts[count] = vec4<f32>(p0 - n * s0, rc - s0);
        count = count + 1u;
    }

    // Endpoint p1, unless the capsule collapsed to a single point.
    if (!degenerate) {
        let s1 = dot(n, p1) - d;
        if (s1 < rc) {
            pts[count] = vec4<f32>(p1 - n * s1, rc - s1);
            count = count + 1u;
        }
    }

    if (count == 0u) {
        // Neither end reaches the surface: no penetration, leave the slot zeroed.
        manifolds[i] = out;
        return;
    }

    out.normal_count = vec4<f32>(n, f32(count));
    out.p0 = pts[0];
    out.p1 = pts[1];
    manifolds[i] = out;
}
