// Batched AABB-vs-BVH overlap gather kernel, device-resident variant.
//
// Identical in traversal and output to `bvh_overlap.wgsl`, but it binds a tree
// left resident in device memory by the build (see `src/bvh/resident.rs`)
// instead of host-re-uploaded node arrays. Two things differ from the host
// variant, exactly as in `bvh_pairs_resident.wgsl`:
//
//   1. Internal-node bounds arrive order-encoded (the raw `u32`s the build's
//      bounds pass wrote), three lanes (x, y, z) per node, and are decoded
//      in-shader with the integer inverse of that encoding (`from_order`). The
//      decode is pure bit arithmetic, so the recovered float is bit-for-bit the
//      builder's input and the overlap tests match the host variant and the
//      `cpu_bvh_aabb_overlap` twin exactly.
//   2. Leaf boxes are the original-order primitive arrays; a leaf slot indexes
//      them through `sorted_indices` rather than reading a pre-gathered
//      leaf-slot-order array.
//
// As in the host variant, one invocation handles one external query box, walks
// the whole hierarchy from the root with the stackless parent-pointer scheme,
// and appends every overlapping leaf's original primitive index to the query's
// fixed region of a flat output buffer under a per-query atomic counter. The
// per-query hit set equals the twin's; only the intra-query append order
// differs, so a sorted per-query comparison is bit-for-bit.
//
// Provenance: stackless parent-pointer BVH traversal of Hapala et al.,
// "Efficient Stack-less BVH Traversal for Ray Tracing" (2011), over the linear
// BVH of Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
// and k-d Trees" (High Performance Graphics 2012). No Unreal Engine source or
// derived code.

struct Params {
    // Number of internal nodes (num_leaves - 1; at least one here, since a
    // resident tree with fewer than two leaves owns no buffers and is never
    // dispatched).
    num_internal: u32,
    // Number of leaves (mesh primitives in the tree).
    num_leaves: u32,
    // Encoded id of the root node.
    root: u32,
    // Number of external query boxes.
    num_queries: u32,
    // Per-query output capacity, in primitive indices.
    capacity_per_query: u32,
};

const NO_PARENT: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: Params;
// Left child (encoded id) of each internal node.
@group(0) @binding(1) var<storage, read> left: array<u32>;
// Right child (encoded id) of each internal node.
@group(0) @binding(2) var<storage, read> right: array<u32>;
// Parent (encoded id) of every node, indexed by encoded id.
@group(0) @binding(3) var<storage, read> parent: array<u32>;
// Order-encoded internal-node minimum bounds, three lanes (x, y, z) per node.
@group(0) @binding(4) var<storage, read> node_min_enc: array<u32>;
// Order-encoded internal-node maximum bounds, three lanes (x, y, z) per node.
@group(0) @binding(5) var<storage, read> node_max_enc: array<u32>;
// Primitive box minimum corners in original order; xyz used.
@group(0) @binding(6) var<storage, read> aabb_min: array<vec4<f32>>;
// Primitive box maximum corners in original order; xyz used.
@group(0) @binding(7) var<storage, read> aabb_max: array<vec4<f32>>;
// Leaf-slot primitive indices (the sorted radix payload).
@group(0) @binding(8) var<storage, read> sorted_indices: array<u32>;
// External query box minimum corners, one per query; xyz used.
@group(0) @binding(9) var<storage, read> query_min: array<vec4<f32>>;
// External query box maximum corners, one per query; xyz used.
@group(0) @binding(10) var<storage, read> query_max: array<vec4<f32>>;
// Per-query hit counts (may exceed capacity on overflow, flagged host-side).
@group(0) @binding(11) var<storage, read_write> counts: array<atomic<u32>>;
// Flat output: query q owns slots [q*capacity_per_query, (q+1)*capacity_per_query).
@group(0) @binding(12) var<storage, read_write> hits: array<u32>;

// Inverse of the bounds pass' `to_order`: maps an order-preserving u32 back to
// its original float bits. A set high bit marks an originally non-negative
// value (clear the sign bit); otherwise the value was negative (flip all bits).
fn from_order(u: u32) -> f32 {
    if ((u & 0x80000000u) != 0u) {
        return bitcast<f32>(u & 0x7fffffffu);
    }
    return bitcast<f32>(~u);
}

// Whether an encoded id refers to a leaf rather than an internal node.
fn is_leaf(encoded: u32) -> bool {
    return encoded >= params.num_internal;
}

// Minimum corner of an encoded node's box.
fn node_min(encoded: u32) -> vec3<f32> {
    if (encoded < params.num_internal) {
        let base = encoded * 3u;
        return vec3<f32>(
            from_order(node_min_enc[base + 0u]),
            from_order(node_min_enc[base + 1u]),
            from_order(node_min_enc[base + 2u]),
        );
    }
    let prim = sorted_indices[encoded - params.num_internal];
    return aabb_min[prim].xyz;
}

// Maximum corner of an encoded node's box.
fn node_max(encoded: u32) -> vec3<f32> {
    if (encoded < params.num_internal) {
        let base = encoded * 3u;
        return vec3<f32>(
            from_order(node_max_enc[base + 0u]),
            from_order(node_max_enc[base + 1u]),
            from_order(node_max_enc[base + 2u]),
        );
    }
    let prim = sorted_indices[encoded - params.num_internal];
    return aabb_max[prim].xyz;
}

// Whether boxes [amin, amax] and [bmin, bmax] overlap, boundaries inclusive.
fn overlap(amin: vec3<f32>, amax: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    return amin.x <= bmax.x && bmin.x <= amax.x
        && amin.y <= bmax.y && bmin.y <= amax.y
        && amin.z <= bmax.z && bmin.z <= amax.z;
}

@compute @workgroup_size(64)
fn find_overlaps(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.num_queries) {
        return;
    }

    let qmin = query_min[q].xyz;
    let qmax = query_max[q].xyz;
    let base = q * params.capacity_per_query;

    // Stackless traversal: `prev` is the node we came from, `cur` the node we
    // are visiting. Comparing `prev` to `cur`'s parent and children tells us
    // whether this is a first visit (descend) or a return (go to the sibling or
    // back up), so no explicit stack is needed.
    var prev: u32 = NO_PARENT;
    var cur: u32 = params.root;
    loop {
        if (cur == NO_PARENT) {
            break;
        }
        let p = parent[cur];
        var next: u32;
        if (prev == p) {
            // First arrival at `cur` (descended from its parent).
            if (overlap(node_min(cur), node_max(cur), qmin, qmax)) {
                if (is_leaf(cur)) {
                    let leaf_slot = cur - params.num_internal;
                    let slot = atomicAdd(&counts[q], 1u);
                    if (slot < params.capacity_per_query) {
                        hits[base + slot] = sorted_indices[leaf_slot];
                    }
                    next = p;
                } else {
                    // Box overlaps: descend into the left child.
                    next = left[cur];
                }
            } else {
                // Box misses: prune this subtree, return to the parent.
                next = p;
            }
        } else if (prev == left[cur]) {
            // Returned from the left child; visit the right child next.
            next = right[cur];
        } else {
            // Returned from the right child; both children done, go up.
            next = p;
        }
        prev = cur;
        cur = next;
    }
}
