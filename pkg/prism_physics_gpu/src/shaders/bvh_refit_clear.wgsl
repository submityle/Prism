// LBVH refit bounds-reset kernel, device pass.
//
// One invocation per internal node. It resets that node's three order-encoded
// bound lanes to the identity the atomic min/max accumulation expects: every
// minimum lane to 0xffffffff (the largest order-encoded value, so any later
// atomicMin wins) and every maximum lane to 0 (the smallest, so any later
// atomicMax wins). After this pass, re-running the bottom-up bounds kernel in
// shaders/bvh_bbox.wgsl over the same topology refits the whole tree in place.
//
// Running this before build_bounds is exactly how the full build primes those
// buffers at creation (node_min initialised to 0xffffffff, node_max to 0); a
// refit reuses the resident buffers, so it must re-prime them explicitly.
//
// Provenance: standard clear pass over the sortable-float order encoding used by
// shaders/bvh_bbox.wgsl. No Unreal Engine source or derived code.

struct Params {
    // Number of internal nodes (num_leaves - 1).
    num_internal: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Per-internal-node minimum bounds, order-encoded, 3 lanes (x, y, z) per node.
@group(0) @binding(1) var<storage, read_write> node_min: array<u32>;
// Per-internal-node maximum bounds, order-encoded, 3 lanes (x, y, z) per node.
@group(0) @binding(2) var<storage, read_write> node_max: array<u32>;

@compute @workgroup_size(64)
fn clear_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if (node >= params.num_internal) {
        return;
    }
    let base = node * 3u;
    node_min[base + 0u] = 0xffffffffu;
    node_min[base + 1u] = 0xffffffffu;
    node_min[base + 2u] = 0xffffffffu;
    node_max[base + 0u] = 0u;
    node_max[base + 1u] = 0u;
    node_max[base + 2u] = 0u;
}
