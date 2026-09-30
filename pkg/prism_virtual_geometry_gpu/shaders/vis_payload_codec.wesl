// Vis-buffer payload codec twin: packs a (cluster_id, triangle_id) pair into
// the shipping 32-bit visibility payload and unpacks it back, mirroring the CPU
// golden const fns in `prism_render_architecture::virtual_geometry`
// (`software_raster.rs`):
//
//   pack_cluster_triangle(cluster_id, triangle_id)
//       = (cluster_id << CLUSTER_TRIANGLE_BITS) | (triangle_id & TRIANGLE_ID_MASK)
//   cluster_of(payload)  = payload >> CLUSTER_TRIANGLE_BITS
//   triangle_of(payload) = payload & TRIANGLE_ID_MASK
//
// with CLUSTER_TRIANGLE_BITS = 7 (so TRIANGLE_ID_MASK = 127): the low 7 bits
// hold the triangle index within a cluster (up to MAX_CLUSTER_TRIANGLES = 128),
// the high 25 bits hold the cluster id. The software rasterizer writes exactly
// this payload beside the depth key, so a GPU-driven rasterizer that emits the
// same visibility word must agree with the reference bit-for-bit here. This
// twin isolates that pack/unpack ABI - a real divergence point because it is
// shift-and-mask arithmetic - rather than validating it only as a sub-step
// buried inside the full raster kernel.
//
// It composites only the 32-bit payload word (no 64-bit `(depth << 32) |
// payload` key), so it uses only the portable core-WGSL subset - integer
// shifts, ORs and ANDs - and runs unmodified on Metal, Vulkan and DX12 with no
// optional feature. The 64-bit `pack_vis` / `vis_depth` / `vis_payload` word is
// deliberately out of scope: it needs SHADER_INT64 and is covered by the
// payload-raster twin instead.
//
// The result is exact: integer shift and mask are defined identically on every
// backend, so the packed word and both unpacked fields match the reference
// index-for-index with no tolerance.
//
// Provenance: Prism's own vis-buffer payload bit layout; no Unreal Engine
// source or derived code.

struct Params {
    // Number of valid inputs in `inputs` and outputs in `outputs`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One (cluster_id, triangle_id) pair to encode. 8-byte stride.
struct PackInput {
    cluster_id: u32,
    triangle_id: u32,
}

// One codec result: the packed payload plus both fields unpacked back out.
// 12-byte payload padded to a 16-byte stride so the host struct matches.
struct CodecOutput {
    packed: u32,
    cluster: u32,
    triangle: u32,
    pad0: u32,
}

// Must match the CPU golden `CLUSTER_TRIANGLE_BITS` and its derived mask.
const CLUSTER_TRIANGLE_BITS: u32 = 7u;
const TRIANGLE_ID_MASK: u32 = 127u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<PackInput>;
@group(0) @binding(2) var<storage, read_write> outputs: array<CodecOutput>;

@compute @workgroup_size(64)
fn codec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let in = inputs[i];
    let packed = (in.cluster_id << CLUSTER_TRIANGLE_BITS) | (in.triangle_id & TRIANGLE_ID_MASK);
    var out: CodecOutput;
    out.packed = packed;
    out.cluster = packed >> CLUSTER_TRIANGLE_BITS;
    out.triangle = packed & TRIANGLE_ID_MASK;
    out.pad0 = 0u;
    outputs[i] = out;
}
