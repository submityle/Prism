// LBVH bottom-up bounds kernel, device pass.
//
// One invocation per leaf. Each leaf walks the parent links from its own parent
// up to the root and, at every ancestor, expands that node's bounds to include
// the leaf's box. The expansion is a per-component atomic min/max over an
// order-preserving encoding of the float bits, so a node's bounds become the
// exact componentwise min/max of every descendant leaf box regardless of the
// order threads arrive; there is no dependency on another node's writes, which
// keeps the pass correct across workgroups without a device-scope fence. The
// result is the exact componentwise min/max the `climb_bounds` golden twin in
// `src/bvh/cpu.rs` computes, so parity is bit-for-bit.
//
// Leaves are addressed in the encoded index space (leaf `leaf` has encoded id
// `num_internal + leaf`); a leaf's box is its primitive's box gathered through
// the sorted index payload.
//
// Provenance: bottom-up bounds refit over the linear BVH of Karras, "Maximizing
// Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (High
// Performance Graphics 2012); the sortable-float encoding for atomic min/max is
// the classical radix-float bit flip. No Unreal Engine source or derived code.

struct Params {
    // Number of leaves.
    n: u32,
    // Number of internal nodes (n - 1).
    num_internal: u32,
    _pad0: u32,
    _pad1: u32,
};

const NO_PARENT: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: Params;
// Parent (encoded id) of every node, indexed by encoded id.
@group(0) @binding(1) var<storage, read> parent: array<u32>;
// Leaf-order primitive indices (the sorted radix payload).
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
// Primitive box minimum corners in original order; xyz used.
@group(0) @binding(3) var<storage, read> aabb_min: array<vec4<f32>>;
// Primitive box maximum corners in original order; xyz used.
@group(0) @binding(4) var<storage, read> aabb_max: array<vec4<f32>>;
// Per-internal-node minimum bounds, order-encoded, 3 lanes (x, y, z) per node.
@group(0) @binding(5) var<storage, read_write> node_min: array<atomic<u32>>;
// Per-internal-node maximum bounds, order-encoded, 3 lanes (x, y, z) per node.
@group(0) @binding(6) var<storage, read_write> node_max: array<atomic<u32>>;

// Maps a float's bits to a u32 whose unsigned order matches the float's numeric
// order (for non-NaN values): flip all bits when negative, else set the sign
// bit. Monotonic, so atomicMin/atomicMax select the exact extreme float.
fn to_order(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

@compute @workgroup_size(64)
fn build_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf = gid.x;
    if (leaf >= params.n) {
        return;
    }
    let prim = sorted_indices[leaf];
    let lo = aabb_min[prim].xyz;
    let hi = aabb_max[prim].xyz;
    let omin = vec3<u32>(to_order(lo.x), to_order(lo.y), to_order(lo.z));
    let omax = vec3<u32>(to_order(hi.x), to_order(hi.y), to_order(hi.z));

    var node = parent[params.num_internal + leaf];
    loop {
        if (node == NO_PARENT) {
            break;
        }
        let base = node * 3u;
        atomicMin(&node_min[base + 0u], omin.x);
        atomicMin(&node_min[base + 1u], omin.y);
        atomicMin(&node_min[base + 2u], omin.z);
        atomicMax(&node_max[base + 0u], omax.x);
        atomicMax(&node_max[base + 1u], omax.y);
        atomicMax(&node_max[base + 2u], omax.z);
        node = parent[node];
    }
}
