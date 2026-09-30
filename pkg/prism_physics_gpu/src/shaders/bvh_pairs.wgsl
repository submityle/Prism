// LBVH overlap-pair broad-phase query kernel, device pass.
//
// One invocation per leaf. The invocation's query box is that leaf's box; it
// walks the whole hierarchy from the root and appends, to a shared output
// buffer, every leaf whose box overlaps the query box and whose leaf slot is
// strictly greater than the querying slot. The strict-greater rule makes the
// smaller slot the sole owner of each unordered pair, so every overlapping pair
// is emitted exactly once. The result set matches the `cpu_bvh_pairs` golden
// twin in `src/bvh/query.rs` exactly (the append order differs, but each pair
// is canonicalised host-side, so a sorted comparison is bit-for-bit).
//
// The traversal is stackless: instead of a per-thread stack (a degenerate
// Karras tree is O(n) deep, so any fixed stack could overflow), each step reads
// the current node's parent link and decides the next node from where it came,
// using O(1) registers and correctly handling any depth.
//
// Nodes use the encoded index space of the builder: an id below `num_internal`
// is that internal node; an id at or above it is leaf `id - num_internal`.
//
// Provenance: stackless parent-pointer BVH traversal of Hapala et al.,
// "Efficient Stack-less BVH Traversal for Ray Tracing" (2011), over the linear
// BVH of Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
// and k-d Trees" (High Performance Graphics 2012). No Unreal Engine source or
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
// Internal-node box minimum corners, indexed by internal id; xyz used.
@group(0) @binding(4) var<storage, read> internal_min: array<vec4<f32>>;
// Internal-node box maximum corners, indexed by internal id; xyz used.
@group(0) @binding(5) var<storage, read> internal_max: array<vec4<f32>>;
// Leaf box minimum corners in leaf-slot order; xyz used.
@group(0) @binding(6) var<storage, read> leaf_min: array<vec4<f32>>;
// Leaf box maximum corners in leaf-slot order; xyz used.
@group(0) @binding(7) var<storage, read> leaf_max: array<vec4<f32>>;
// Leaf-slot primitive indices (the sorted radix payload).
@group(0) @binding(8) var<storage, read> sorted_indices: array<u32>;
// Total number of emitted pairs (may exceed capacity on overflow).
@group(0) @binding(9) var<storage, read_write> pair_count: atomic<u32>;
// Emitted pairs; slots at or beyond capacity are dropped.
@group(0) @binding(10) var<storage, read_write> pairs: array<vec2<u32>>;

// Whether an encoded id refers to a leaf rather than an internal node.
fn is_leaf(encoded: u32) -> bool {
    return encoded >= params.num_internal;
}

// Minimum corner of an encoded node's box.
fn node_min(encoded: u32) -> vec3<f32> {
    if (encoded < params.num_internal) {
        return internal_min[encoded].xyz;
    }
    return leaf_min[encoded - params.num_internal].xyz;
}

// Maximum corner of an encoded node's box.
fn node_max(encoded: u32) -> vec3<f32> {
    if (encoded < params.num_internal) {
        return internal_max[encoded].xyz;
    }
    return leaf_max[encoded - params.num_internal].xyz;
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

    let qmin = leaf_min[my_slot].xyz;
    let qmax = leaf_max[my_slot].xyz;

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
