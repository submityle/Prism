// LBVH Morton-code kernel, device pass.
//
// One invocation per leaf primitive. Each thread reads its axis-aligned box,
// takes the centroid, quantises it into a [0, 1023] integer per axis relative
// to the scene bounds, and interleaves the three coordinates into a 30-bit
// Morton (Z-order) code written as the radix sort key, plus its own leaf index
// as the payload the sort carries alongside. The expansion, quantisation, and
// interleave mirror `cpu_morton_code` in `src/bvh/morton.rs` exactly, so the
// emitted keys match the CPU twin bit-for-bit.
//
// Provenance: bit-interleaved Morton/Z-order indexing feeding the linear BVH of
// Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and
// k-d Trees" (High Performance Graphics 2012). No Unreal Engine source or
// derived code.

struct Params {
    // Scene-bounds minimum corner.
    min_x: f32,
    min_y: f32,
    min_z: f32,
    // Number of leaves.
    n: u32,
    // Reciprocal of each scene-bounds axis extent (0 for a degenerate axis).
    // Computed on the host: WGSL division carries a 2.5-ULP tolerance, so the
    // kernel multiplies by this correctly-rounded reciprocal to stay bit-for-bit
    // identical to the CPU twin.
    inv_ext_x: f32,
    inv_ext_y: f32,
    inv_ext_z: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Leaf box minimum corners; xyz used, w unused.
@group(0) @binding(1) var<storage, read> aabb_min: array<vec4<f32>>;
// Leaf box maximum corners; xyz used, w unused.
@group(0) @binding(2) var<storage, read> aabb_max: array<vec4<f32>>;
// Output: per-leaf 30-bit Morton code (the sort key).
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
// Output: per-leaf identity payload the sort carries with the key.
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;

fn expand_bits_10(v: u32) -> u32 {
    var x = v & 0x000003ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn quantize(c: f32, lo: f32, inv_extent: f32) -> u32 {
    let norm = (c - lo) * inv_extent;
    return u32(clamp(floor(norm * 1024.0), 0.0, 1023.0));
}

@compute @workgroup_size(64)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let centre = (aabb_min[i].xyz + aabb_max[i].xyz) * 0.5;
    let qx = quantize(centre.x, params.min_x, params.inv_ext_x);
    let qy = quantize(centre.y, params.min_y, params.inv_ext_y);
    let qz = quantize(centre.z, params.min_z, params.inv_ext_z);
    keys[i] = expand_bits_10(qx) | (expand_bits_10(qy) << 1u) | (expand_bits_10(qz) << 2u);
    indices[i] = i;
}
