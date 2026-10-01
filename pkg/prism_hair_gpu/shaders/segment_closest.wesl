// Per-pair closest-point-between-segments twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::barrier_contact::segment_segment_closest`.
// For each pair of segments `[p1, q1]` and `[p2, q2]` it returns the closest
// point on each segment plus the distance between them, following the clamped
// parametric solve from Ericson, *Real-Time Collision Detection*: the segment
// parameters `s, t` are clamped to `[0, 1]` so the result lies on the segments,
// and degenerate (zero-length) segments are handled without dividing by zero.
//
// This closed-form closest-pair query is the geometric kernel the strand
// self-collision / barrier contact pass evaluates for every candidate
// segment-segment pair, so the on-device twin must match the scalar reference
// branch for branch.
//
// # Why one thread per pair
//
// Each pair's result depends only on its own four endpoints, so this is
// embarrassingly parallel: one thread owns one pair, reads its four points and
// writes its two closest points plus the scalar distance. The host lays the
// pairs out flat (four `vec4` lanes per pair) so threads never alias.
//
// Points travel through the storage buffer as `vec4<f32>` in `(x, y, z, 0)`
// lane order; the outputs likewise carry the distance in `out_c1`'s `w` lane.
//
// # Correctness model
//
// The reference sanitizes its inputs (replacing non-finite components with 0);
// the host supplies finite endpoints, so the kernel omits that guard but keeps
// the `EPS_LEN_SQ` degeneracy thresholds and the `EPS_DENOM` parallel guard
// verbatim. The solve is a single closed-form evaluation (a handful of
// dot/clamp/divide/multiply-add plus one `sqrt` for the distance), with no
// chained recurrence, so the only `CPU` vs `GPU` divergence is legal
// fused-multiply-add contraction; parity is asserted to `abs < 1e-4` or
// `rel < 1e-3`. Test pairs stay clear of the branch boundaries so the branch
// taken is identical on both sides.
//
// # Portability
//
// The kernel uses only `dot`, `clamp`, `length`, divide and multiply-add in the
// portable core-`WGSL` subset — no `exp`, `pow` or optional device feature — so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Ericson segment-segment closest-point construction plus
// a `wgpu` compute dispatch; no Unreal Engine source or derived code.

const EPS_LEN_SQ: f32 = 1.0e-24;
const EPS_DENOM: f32 = 1.0e-12;

// Shared parameters for one dispatch: the pair count padded to one `16`-byte
// uniform slot.
struct Params {
    pair_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flat segment endpoints: pair `i` occupies lanes `4 * i .. 4 * i + 4` as
// `(p1, q1, p2, q2)`, each `(x, y, z, 0)`.
@group(0) @binding(1) var<storage, read> segs: array<vec4<f32>>;
// Per pair: closest point on segment 1 in `xyz`, the distance in `w`.
@group(0) @binding(2) var<storage, read_write> out_c1: array<vec4<f32>>;
// Per pair: closest point on segment 2 in `xyz` (`w` is 0).
@group(0) @binding(3) var<storage, read_write> out_c2: array<vec4<f32>>;

// `clamp01` mirrors the reference `finite_or(x, 0).clamp(0, 1)`; inputs are
// finite here, so it reduces to a plain clamp.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.pair_count) {
        return;
    }
    let base = i * 4u;
    let p1 = segs[base].xyz;
    let q1 = segs[base + 1u].xyz;
    let p2 = segs[base + 2u].xyz;
    let q2 = segs[base + 3u].xyz;

    let d1 = q1 - p1; // direction and length of segment 1
    let d2 = q2 - p2; // direction and length of segment 2
    let r = p1 - p2;
    let a = dot(d1, d1); // squared length of segment 1
    let e = dot(d2, d2); // squared length of segment 2
    let f = dot(d2, r);

    let seg1_degenerate = a <= EPS_LEN_SQ;
    let seg2_degenerate = e <= EPS_LEN_SQ;

    var s: f32 = 0.0;
    var t: f32 = 0.0;
    if (seg1_degenerate && seg2_degenerate) {
        // Both segments are points.
        s = 0.0;
        t = 0.0;
    } else if (seg1_degenerate) {
        // First segment is a point: only slide along the second.
        s = 0.0;
        t = clamp01(f / e);
    } else {
        let c = dot(d1, r);
        if (seg2_degenerate) {
            // Second segment is a point: only slide along the first.
            s = clamp01(-c / a);
            t = 0.0;
        } else {
            // General non-degenerate case.
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            var s0: f32 = 0.0;
            if (abs(denom) > EPS_DENOM) {
                s0 = clamp01((b * f - c * e) / denom);
            } else {
                // Parallel segments: pick an arbitrary point on segment 1 and
                // let the clamp on t below place the closest point.
                s0 = 0.0;
            }
            // Compute t for this s, then re-clamp s if t left [0, 1].
            let t0 = (b * s0 + f) / e;
            if (t0 < 0.0) {
                s = clamp01(-c / a);
                t = 0.0;
            } else if (t0 > 1.0) {
                s = clamp01((b - c) / a);
                t = 1.0;
            } else {
                s = s0;
                t = t0;
            }
        }
    }

    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    let dist = length(c1 - c2);

    out_c1[i] = vec4<f32>(c1.x, c1.y, c1.z, dist);
    out_c2[i] = vec4<f32>(c2.x, c2.y, c2.z, 0.0);
}
