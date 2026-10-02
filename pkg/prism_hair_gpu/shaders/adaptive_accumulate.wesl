// Compute twin of Prism's adaptive transmittance curve *accumulation*
// (prism_render_architecture::hair::adaptive_transmittance::accumulate): the
// build side that turns one light ray/texel's strand samples into the full
// (uncompressed) transmittance curve, as opposed to the sibling
// `adaptive_transmittance` twin, which is the receiver-query *lookup* on an
// already-compressed curve.
//
// The golden, per ray, re-sanitises each sample, stably sorts it by light-space
// depth, folds every sample sharing a depth (within DEPTH_EPS) into one control
// node, and stores at each node the running composite transmittance
// `T = product(1 - alpha)` of every sample at or in front of it, clamped to
// `0..=1` at each node boundary. The result is monotone non-increasing in
// transmittance and sorted by depth.
//
// The *sort* and the same-depth *grouping* are host/sequential concerns (the
// golden documents the device-side build as separate scheduling), so the host
// re-sanitises, sorts and groups, then uploads each ray's node list as a
// contiguous run of (alpha_start, alpha_count) descriptors into a shared,
// per-node-grouped `alphas` pool. This kernel owns exactly the per-ray running
// product: one thread per ray walks its nodes in depth order, composites each
// node's alphas into the running product and clamps at the node boundary,
// writing one transmittance per global node. The node depths are host
// passthrough and are reattached on the host after readback.
//
// # Correctness model
//
// The composite is a closed-form running product with no transcendental call,
// so CPU and GPU evaluate the same arithmetic in the same order. They are not
// bit-exact: a GPU may fuse the `running * (1 - alpha)` multiply-add the scalar
// reference leaves separate, perturbing the low mantissa bits by a few ULP. The
// parity test asserts a per-value tolerance (`abs_diff < 1e-4` or
// `rel_diff < 1e-3`) on transmittance; node depths and node counts match
// exactly because they are host passthrough.
//
// # Portability
//
// The kernel uses only subtract, multiply and `clamp` in the portable
// core-WGSL subset -- no `exp`, `pow`, `sin` or optional device feature -- so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard alpha-composite running product (per-ray depth sort then
// cumulative `product(1 - alpha)`) plus a wgpu compute dispatch; no Unreal
// Engine source or derived code.

// Just the ray count; the per-ray and per-node layout lives in the storage
// buffers. A single `u32` fills (and is padded to) the `16`-byte uniform slot.
struct Params {
    ray_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Sanitised, depth-sorted alpha values, grouped per node, concatenated across
// every ray.
@group(0) @binding(1) var<storage, read> alphas: array<f32>;
// Per node: two `u32` `(alpha_start, alpha_count)` into `alphas`.
@group(0) @binding(2) var<storage, read> nodes: array<u32>;
// Per ray: two `u32` `(node_start, node_count)` into `nodes`.
@group(0) @binding(3) var<storage, read> rays: array<u32>;
// One running composite transmittance per global node, node-index aligned.
@group(0) @binding(4) var<storage, read_write> out_t: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.ray_count) {
        return;
    }
    let node_start = rays[r * 2u];
    let node_count = rays[r * 2u + 1u];

    // Running composite transmittance resets per ray and only ever decreases.
    var running: f32 = 1.0;
    for (var k = 0u; k < node_count; k = k + 1u) {
        let node = node_start + k;
        let a_start = nodes[node * 2u];
        let a_count = nodes[node * 2u + 1u];
        for (var j = 0u; j < a_count; j = j + 1u) {
            running = running * (1.0 - alphas[a_start + j]);
        }
        running = clamp(running, 0.0, 1.0);
        out_t[node] = running;
    }
}
