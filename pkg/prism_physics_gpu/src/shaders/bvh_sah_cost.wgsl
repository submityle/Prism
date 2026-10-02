// BVH Surface Area Heuristic (SAH) cost reduction.
//
// Computes, on device, the two surface-area sums the host needs to form the SAH
// cost of a resident LBVH: the summed surface area of every internal node and
// of every leaf box. One compute entry point runs with a single workgroup and a
// grid-stride loop so the fold order is fixed and the result is deterministic
// across launches; the host divides by the root surface area to finish the
// normalised cost.
//
// The reduction mirrors the cpu_sah_cost / lbvh_sah_cost golden twin. The fold
// is a floating-point sum, whose grouping differs from the host's sequential
// sum, so parity is checked within a tight relative tolerance rather than
// bit-for-bit (as with the CFL reducer).
//
// Node bounds are order-encoded (the monotonic bit mapping the build's
// atomicMin/atomicMax use); from_order inverts it exactly. Leaf boxes are the
// original-order primitive boxes in plain f32 vec4 lanes.
//
// Provenance: the Surface Area Heuristic is Goldsmith and Salmon (1987) and a
// shared-memory tree reduction is a standard GPU technique. No Unreal Engine
// source or derived code.

struct Params {
    // Number of elements to fold (internal nodes, or leaf boxes).
    count: u32,
    // Source selector: 0 = order-encoded internal nodes, 1 = leaf vec4 boxes.
    source: u32,
    // Padding to a 16-byte boundary.
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Order-encoded internal-node minimum bounds, three lanes per node.
@group(0) @binding(1) var<storage, read> node_min: array<u32>;
// Order-encoded internal-node maximum bounds, three lanes per node.
@group(0) @binding(2) var<storage, read> node_max: array<u32>;
// Original-order primitive box minimum corners; xyz = corner, w unused.
@group(0) @binding(3) var<storage, read> aabb_min: array<vec4<f32>>;
// Original-order primitive box maximum corners; xyz = corner, w unused.
@group(0) @binding(4) var<storage, read> aabb_max: array<vec4<f32>>;
// Single output slot: the reduced surface-area sum as its f32 bit pattern.
@group(0) @binding(5) var<storage, read_write> out_sum: array<u32>;

// Inverts to_order: recovers the float whose order-encoded bits are u.
fn from_order(u: u32) -> f32 {
    if ((u & 0x80000000u) != 0u) {
        return bitcast<f32>(u & 0x7fffffffu);
    }
    return bitcast<f32>(~u);
}

// Surface area of a box: 2*(dx*dy + dy*dz + dz*dx), clamping degenerate extents
// to zero so the result is non-negative, matching the host surface_area.
fn surface_area(lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let d = max(hi - lo, vec3<f32>(0.0, 0.0, 0.0));
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Grid-stride accumulation: each lane folds every 256th element, giving a
    // fixed per-lane order independent of the dispatch size.
    var acc: f32 = 0.0;
    var i = lid.x;
    loop {
        if (i >= params.count) {
            break;
        }
        var area: f32;
        if (params.source == 0u) {
            let base = i * 3u;
            let lo = vec3<f32>(
                from_order(node_min[base + 0u]),
                from_order(node_min[base + 1u]),
                from_order(node_min[base + 2u]),
            );
            let hi = vec3<f32>(
                from_order(node_max[base + 0u]),
                from_order(node_max[base + 1u]),
                from_order(node_max[base + 2u]),
            );
            area = surface_area(lo, hi);
        } else {
            area = surface_area(aabb_min[i].xyz, aabb_max[i].xyz);
        }
        acc = acc + area;
        i = i + 256u;
    }
    scratch[lid.x] = acc;
    workgroupBarrier();

    // Halving tree reduction over the 256 lanes; stride is uniform so every lane
    // reaches each barrier.
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        out_sum[0] = bitcast<u32>(scratch[0]);
    }
}
