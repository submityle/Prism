// LBVH closest-hit ray query kernel over a device-resident tree.
//
// One invocation per ray. Each ray walks the whole hierarchy from the root with
// the same stackless parent-pointer traversal the overlap-pair kernel uses
// (`bvh_pairs_resident.wgsl`), but instead of appending overlaps it keeps the
// nearest leaf box the ray enters. A subtree is pruned when the ray misses its
// node box or enters it no nearer than the best hit found so far, so the search
// cost tracks genuine intersections rather than the whole tree.
//
// The tree buffers are consumed straight from device memory, exactly as the
// resident overlap query consumes them: internal-node bounds arrive
// order-encoded and are decoded in-shader with `from_order` (the integer
// inverse of the bounds pass' `to_order`), and leaf boxes are gathered from the
// original-order primitive boxes through the sorted leaf-slot index payload,
// `leaf_box(slot) = aabb[sorted_indices[slot]]`.
//
// The ray-box test is the branch-free slab method of Williams et al., "An
// Efficient and Robust Ray-Box Intersection Algorithm" (2005): reciprocate the
// direction per component (a zero component yields a signed infinity, matching
// the CPU twin's `1.0 / 0.0`), map each slab to an entry and exit distance, take
// the largest entry and smallest exit across axes, and clamp the entry to zero
// so an origin inside the box hits at distance zero. Component-wise min/max are
// NaN-robust, so a zero-direction axis whose origin lies outside its slab
// collapses to an empty interval and the box is missed, bit-for-bit with the
// twin. Only the reciprocal is inexact (up to 2.5 ULP), so the parity test
// matches the primitive index exactly and the distance within a tolerance.
//
// Nodes use the builder's encoded index space: an id below `num_internal` is
// that internal node; an id at or above it is leaf `id - num_internal`.
//
// Provenance: stackless parent-pointer BVH traversal of Hapala et al.,
// "Efficient Stack-less BVH Traversal for Ray Tracing" (2011), over the linear
// BVH of Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
// and k-d Trees" (High Performance Graphics 2012), with Williams et al. (2005)
// slab intersection. No Unreal Engine source or derived code.

struct Params {
    // Number of internal nodes (num_leaves - 1).
    num_internal: u32,
    // Number of leaves (input primitives).
    num_leaves: u32,
    // Encoded id of the root node.
    root: u32,
    // Number of rays queued in the ray buffer.
    num_rays: u32,
};

const NO_PARENT: u32 = 0xffffffffu;
const NO_PRIM: u32 = 0xffffffffu;

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
// Rays, two vec4 per ray: [2i] = (origin.xyz, t_max), [2i+1] = (dir.xyz, _).
@group(0) @binding(9) var<storage, read> rays: array<vec4<f32>>;
// Closest hits, one per ray: (bitcast<u32>(t), prim); miss => (inf bits, NO_PRIM).
@group(0) @binding(10) var<storage, read_write> hits: array<vec2<u32>>;

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

// Ray-box slab entry distance. Returns the clamped entry parameter; the caller
// treats it as a hit only when `hit` is set (entry within exit and `t_max`).
struct Slab {
    hit: bool,
    t: f32,
};

fn slab_enter(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, t_max: f32) -> Slab {
    let inv = vec3<f32>(1.0, 1.0, 1.0) / d;
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let ts = min(t0, t1);
    let tb = max(t0, t1);
    let t_near = max(max(max(ts.x, ts.y), ts.z), 0.0);
    let t_far = min(min(tb.x, tb.y), tb.z);
    var s: Slab;
    s.t = t_near;
    s.hit = (t_near <= t_far) && (t_near <= t_max);
    return s;
}

@compute @workgroup_size(64)
fn raycast_closest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= params.num_rays) {
        return;
    }

    let origin = rays[2u * ray].xyz;
    let ray_t_max = rays[2u * ray].w;
    let dir = rays[2u * ray + 1u].xyz;

    // Best hit so far. `best_t` doubles as the pruning cap: a node entered no
    // nearer than `best_t` cannot contain a closer leaf, so its subtree is
    // skipped. Initialising it to the ray's `t_max` bounds the search to the
    // ray's extent.
    var best_t: f32 = ray_t_max;
    var best_prim: u32 = NO_PRIM;

    // Stackless traversal: `prev` is the node we came from, `cur` the node we
    // are visiting; comparing `prev` to `cur`'s parent and children tells first
    // visits (descend) apart from returns (sibling or go up).
    var prev: u32 = NO_PARENT;
    var cur: u32 = params.root;
    loop {
        if (cur == NO_PARENT) {
            break;
        }
        let par = parent[cur];
        var next: u32;
        if (prev == par) {
            // First arrival at `cur`: test the ray against its box, capped by
            // the best hit so far so farther subtrees prune away.
            let s = slab_enter(origin, dir, node_min(cur), node_max(cur), best_t);
            if (s.hit) {
                if (is_leaf(cur)) {
                    // Strict improvement keeps the first primitive on ties,
                    // matching the CPU twin's `<` selection.
                    if (s.t < best_t) {
                        best_t = s.t;
                        best_prim = sorted_indices[cur - params.num_internal];
                    }
                    next = par;
                } else {
                    next = left[cur];
                }
            } else {
                next = par;
            }
        } else if (prev == left[cur]) {
            next = right[cur];
        } else {
            next = par;
        }
        prev = cur;
        cur = next;
    }

    if (best_prim == NO_PRIM) {
        // Miss: store +inf so a host reading the raw bits maps it to no hit.
        hits[ray] = vec2<u32>(0x7f800000u, NO_PRIM);
    } else {
        hits[ray] = vec2<u32>(bitcast<u32>(best_t), best_prim);
    }
}
