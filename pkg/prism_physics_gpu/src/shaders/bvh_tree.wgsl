// LBVH binary radix tree kernel, device pass.
//
// One invocation per internal node (there are n - 1 of them for n leaves). Each
// node independently determines the contiguous range of sorted leaves it
// covers, the split point inside that range, and its two children, following
// Karras' construction. Nodes are addressed in one encoded index space: an id
// below `num_internal` is an internal node; an id at or above it is leaf
// `id - num_internal`. Each internal node writes its children's parent links;
// every child has exactly one parent, so those writes never race. This mirrors
// `karras_children` in `src/bvh/cpu.rs` exactly, including the index tiebreak in
// `delta` that keeps the tree well defined under duplicate Morton codes.
//
// Provenance: binary radix tree of Karras, "Maximizing Parallelism in the
// Construction of BVHs, Octrees, and k-d Trees" (High Performance Graphics
// 2012). No Unreal Engine source or derived code.

struct Params {
    // Number of leaves.
    n: u32,
    // Number of internal nodes (n - 1).
    num_internal: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Morton codes in ascending sorted order.
@group(0) @binding(1) var<storage, read> codes: array<u32>;
// Output: left child (encoded id) of each internal node.
@group(0) @binding(2) var<storage, read_write> left: array<u32>;
// Output: right child (encoded id) of each internal node.
@group(0) @binding(3) var<storage, read_write> right: array<u32>;
// Output: parent (encoded id) of every node, indexed by encoded id.
@group(0) @binding(4) var<storage, read_write> parent: array<u32>;

fn isign(x: i32) -> i32 {
    if (x > 0) {
        return 1;
    }
    if (x < 0) {
        return -1;
    }
    return 0;
}

// Length of the common Morton-code prefix of leaves i and j, or -1 when j is
// out of range. Equal codes fall back to the index prefix so every pair has a
// distinct delta.
fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) {
        return -1;
    }
    let ki = codes[u32(i)];
    let kj = codes[u32(j)];
    if (ki == kj) {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(ki ^ kj));
}

@compute @workgroup_size(64)
fn build_tree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let num_internal = i32(params.num_internal);
    if (i >= num_internal) {
        return;
    }
    let n = i32(params.n);

    // Direction of the range this node covers (+1 forwards, -1 backwards).
    let d = isign(delta(i, i + 1, n) - delta(i, i - 1, n));

    // Upper bound on the range length, then binary search for its far end.
    let delta_min = delta(i, i - d, n);
    var l_max = 2;
    while (delta(i, i + l_max * d, n) > delta_min) {
        l_max = l_max * 2;
    }
    var l = 0;
    var t = l_max / 2;
    while (t >= 1) {
        if (delta(i, i + (l + t) * d, n) > delta_min) {
            l = l + t;
        }
        t = t / 2;
    }
    let j = i + l * d;

    // Binary search for the split position inside the range.
    let delta_node = delta(i, j, n);
    var s = 0;
    var ts = (l + 1) / 2;
    loop {
        if (delta(i, i + (s + ts) * d, n) > delta_node) {
            s = s + ts;
        }
        if (ts == 1) {
            break;
        }
        ts = (ts + 1) / 2;
    }
    let gamma = i + s * d + min(d, 0);

    // A child is a leaf when its side of the split reaches the range end.
    let range_lo = min(i, j);
    let range_hi = max(i, j);
    var lc = u32(gamma);
    if (range_lo == gamma) {
        lc = u32(num_internal + gamma);
    }
    var rc = u32(gamma + 1);
    if (range_hi == gamma + 1) {
        rc = u32(num_internal + gamma + 1);
    }
    left[u32(i)] = lc;
    right[u32(i)] = rc;
    parent[lc] = u32(i);
    parent[rc] = u32(i);
}
