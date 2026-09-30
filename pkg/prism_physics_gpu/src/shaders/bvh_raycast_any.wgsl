// LBVH any-hit ray query kernel over a device-resident tree.
//
// One invocation per ray. Each ray walks the hierarchy with the same stackless
// parent-pointer traversal the closest-hit kernel uses, but it only needs to
// know whether *any* primitive is hit, so it writes 1 and returns the instant
// the first leaf box is entered within the ray's extent. This is the shadow /
// occlusion probe of PhysX any-hit and Unreal's blocking-hit line trace: the
// early-out makes an unobstructed query stop at the first crossing instead of
// finding the nearest one.
//
// A subtree is pruned when the ray misses its node box within t_max. Unlike the
// closest kernel there is no best-distance cap, since existence, not nearness,
// is what the query reports. The tree buffers, bounds decoding, and leaf-box
// gather are identical to `bvh_raycast.wgsl`.
//
// Nodes use the builder's encoded index space: an id below `num_internal` is
// that internal node; an id at or above it is leaf `id - num_internal`.
//
// Provenance: stackless parent-pointer BVH traversal of Hapala et al. (2011)
// over the linear BVH of Karras (2012), with Williams et al. (2005) slab
// intersection. No Unreal Engine source or derived code.

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
// Any-hit flags, one per ray: 1 when the ray hits some primitive, else 0.
@group(0) @binding(10) var<storage, read_write> hits: array<u32>;

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

// Whether the ray from `o` along `d` enters box [bmin, bmax] within [0, t_max].
// Branch-free slab test, bit-for-bit with `bvh_raycast.wgsl`.
fn slab_hit(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, t_max: f32) -> bool {
    let inv = vec3<f32>(1.0, 1.0, 1.0) / d;
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let ts = min(t0, t1);
    let tb = max(t0, t1);
    let t_near = max(max(max(ts.x, ts.y), ts.z), 0.0);
    let t_far = min(min(tb.x, tb.y), tb.z);
    return (t_near <= t_far) && (t_near <= t_max);
}

@compute @workgroup_size(64)
fn raycast_any(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= params.num_rays) {
        return;
    }

    let origin = rays[2u * ray].xyz;
    let t_max = rays[2u * ray].w;
    let dir = rays[2u * ray + 1u].xyz;

    // Stackless traversal identical to the closest kernel; the only difference
    // is the early return on the first leaf hit.
    var prev: u32 = NO_PARENT;
    var cur: u32 = params.root;
    loop {
        if (cur == NO_PARENT) {
            break;
        }
        let par = parent[cur];
        var next: u32;
        if (prev == par) {
            if (slab_hit(origin, dir, node_min(cur), node_max(cur), t_max)) {
                if (is_leaf(cur)) {
                    hits[ray] = 1u;
                    return;
                }
                next = left[cur];
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
}
