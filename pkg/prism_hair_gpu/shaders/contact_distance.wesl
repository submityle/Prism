// Point/plane contact distance primitives twin.
//
// Device mirror of the two closed-form distance primitives in the `CPU` golden
// `prism_render_architecture::hair::barrier_contact`:
//
//   * `point_plane_signed_distance(p, plane_point, plane_normal)` — the signed
//     distance from `p` to the plane through `plane_point` oriented by the unit
//     `plane_normal`; positive on the side the normal points toward, negative
//     behind it, and exactly `0` when the normal is (numerically) zero and so
//     has no orientation.
//   * `point_point_distance(a, b)` — the Euclidean distance between two points.
//
// These are the plane- and point-collider distance queries the hair barrier
// contact solver evaluates per contact candidate; this twin lifts them out as a
// standalone per-query gather so any consumer (plane colliders, point proximity)
// can reuse the identical device arithmetic.
//
// Each query packs nine `f32`: the query point `p` (`xyz`), the plane/other
// point `pp` (`xyz`), and the plane normal `nn` (`xyz`). The output row is
// `vec4(signed_plane_distance, point_point_distance, 0, 0)`, where the second
// lane is the plain distance between `p` and `pp`.
//
// # Why one thread per query
//
// Each query reads only its own nine inputs and writes its own output row, with
// no shared mutable state, so this is embarrassingly parallel: one thread owns
// one query.
//
// # Correctness model
//
// The reference sanitizes every input vector (non-finite components -> 0) and
// guards a (numerically) zero plane normal with the squared-length threshold
// `EPS_LEN_SQ = 1e-24`, returning `0` for the signed distance in that case. The
// host supplies finite inputs, so the kernel keeps only the zero-normal guard
// verbatim and omits the per-component finite guards. Both primitives are a
// single closed-form evaluation (a dot and a reciprocal square root for the
// plane distance, one subtract-and-length for the point distance) with no
// chained recurrence, so the only `CPU` vs `GPU` divergence is legal
// fused-multiply-add contraction and correctly rounded `sqrt`/division; parity
// is asserted to `abs < 1e-4` or `rel < 1e-3`. The zero-normal signed-distance
// branch yields a bit-exact `0` on both sides.
//
// # Portability
//
// The kernel uses only subtract/dot/mul/divide/sqrt in the portable core-`WGSL`
// subset — no `exp`, `pow`, `log` or optional device feature — so the twin runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard analytic point/plane distance plus a `wgpu` compute
// dispatch; no Unreal Engine source or derived code.

// Just the query count; the per-query data lives in the `queries` storage
// buffer. A single `u32` fills (and is padded to) the `16`-byte uniform slot.
struct Params {
    query_count: u32,
}

// Mirrors `barrier_contact::EPS_LEN_SQ`: the squared-length threshold below
// which a direction is treated as the zero vector (no orientation).
const EPS_LEN_SQ: f32 = 1.0e-24;

@group(0) @binding(0) var<uniform> params: Params;
// Flat per-query inputs, nine `f32` per query: `p.xyz`, `pp.xyz`, `nn.xyz`.
@group(0) @binding(1) var<storage, read> queries: array<f32>;
// Per query: `(signed_plane_distance, point_point_distance, 0, 0)`.
@group(0) @binding(2) var<storage, read_write> out_vals: array<vec4<f32>>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.query_count) {
        return;
    }
    let base_i = i * 9u;
    let p = vec3<f32>(queries[base_i], queries[base_i + 1u], queries[base_i + 2u]);
    let pp = vec3<f32>(queries[base_i + 3u], queries[base_i + 4u], queries[base_i + 5u]);
    let nn = vec3<f32>(queries[base_i + 6u], queries[base_i + 7u], queries[base_i + 8u]);

    // Signed plane distance: normalize the plane normal (zero -> no
    // orientation -> distance 0), then project `p - pp` onto it.
    let len_sq = dot(nn, nn);
    var signed_dist: f32 = 0.0;
    if (len_sq > EPS_LEN_SQ) {
        let n = nn * (1.0 / sqrt(len_sq));
        signed_dist = dot(p - pp, n);
    }

    // Plain Euclidean distance between the two points.
    let delta = p - pp;
    let point_dist = sqrt(dot(delta, delta));

    out_vals[i] = vec4<f32>(signed_dist, point_dist, 0.0, 0.0);
}
