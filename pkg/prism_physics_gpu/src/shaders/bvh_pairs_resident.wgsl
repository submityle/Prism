// LBVH overlap-pair broad-phase query kernel over a device-resident tree.
//
// Identical traversal to `bvh_pairs.wgsl`, but the tree is consumed straight
// from the buffers the build left resident on device, with no readback and no
// re-upload. Two things differ from the host-fed kernel:
//
//   * Internal-node bounds arrive order-encoded (the exact u32s the bounds pass
//     wrote with atomic min/max), so this kernel decodes them in-shader with
//     `from_order`, the exact integer inverse of the bounds pass' `to_order`.
//     The decode is pure bit manipulation, so the resulting f32 is bit-for-bit
//     the value the host `from_order` produces; the overlap comparisons are
//     therefore identical to the host-fed kernel and to the CPU golden twin.
//   * Leaf boxes are gathered from the original-order primitive boxes through
//     the sorted leaf-slot index payload, rather than from a pre-gathered
//     leaf-slot-order array. `leaf_box(slot)` is `aabb[sorted_indices[slot]]`.
//
// One invocation per leaf. The invocation walks the whole hierarchy from the
// root and appends every leaf whose box overlaps its own and whose leaf slot is
// strictly greater than its own, so the smaller slot owns each unordered pair
// and every pair is emitted exactly once. Emitted values are original primitive
// indices (gathered through `sorted_indices`), matching `cpu_bvh_pairs`.
//
// The traversal is stackless: each step reads the current node's parent link
// and decides the next node from where it came, using O(1) registers, so a
// degenerate O(n)-deep tree cannot overflow a fixed stack.
//
// Nodes use the encoded index space of the builder: an id below `num_internal`
// is that internal node; an id at or above it is leaf `id - num_internal`.
//
// Provenance: stackless parent-pointer BVH traversal of Hapala et al.,
// "Efficient Stack-less BVH Traversal for Ray Tracing" (2011), over the linear
// BVH of Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
// and k-d Trees" (High Performance Graphics 2012); the sortable-float bit
// encoding is the classical radix-float bit flip. No Unreal Engine source or
// derived code.

struct Params {
    // Number of internal nodes (num_leaves - 1).
    num_internal: u32,
    // Number of leaves (input primitives).
    num_leaves: u32,
    // Encoded id of the root node.
    root: u32,
    // Output-buffer capacity, in pairs.
    capacity: u32,
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
// Total number of emitted pairs (may exceed capacity on overflow).
@group(0) @binding(9) var<storage, read_write> pair_count: atomic<u32>;
// Emitted pairs; slots at or beyond capacity are dropped.
@group(0) @binding(10) var<storage, read_write> pairs: array<vec2<u32>>;

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
fn find_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let my_slot = gid.x;
    if (my_slot >= params.num_leaves) {
        return;
    }

    let my_prim = sorted_indices[my_slot];
    let qmin = aabb_min[my_prim].xyz;
    let qmax = aabb_max[my_prim].xyz;

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
                    let other_slot = cur - params.num_internal;
                    if (other_slot > my_slot) {
                        let slot = atomicAdd(&pair_count, 1u);
                        if (slot < params.capacity) {
                            pairs[slot] = vec2<u32>(sorted_indices[my_slot], sorted_indices[other_slot]);
                        }
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
