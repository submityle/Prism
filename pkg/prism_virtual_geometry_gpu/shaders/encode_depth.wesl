// Prism virtual-geometry vis-buffer depth-key twin: encodes a reversed-Z NDC
// depth into the u32 compositing key, mirroring the CPU golden
// `encode_depth` (in `prism_render_architecture::virtual_geometry`):
// `bitcast<u32>(clamp(depth, 0.0, 1.0))`.
//
// The vis-buffer resolves the nearest surface with `atomicMax` over this key.
// For non-negative floats the IEEE bit pattern is monotonic, so `1.0` (nearest)
// is the largest key and `0.0` (farthest, also the cleared value) is `0`. The
// clamp keeps a stray negative depth from ever winning the atomicMax. The
// vis-buffer raster twins already encode depth inline; this kernel isolates the
// encode so the clamp + bitcast contract is diffed to the bit on its own.
//
// Numerical model: the encode is bit-exact. `clamp` is comparison + select
// (no arithmetic, no rounding), and `bitcast` copies the bit pattern, so on any
// finite non-negative input the loaded value is returned unchanged, a strictly
// negative input clamps to `+0.0`, and a value above `1.0` (including `+inf`)
// clamps to `1.0` - identical on the sequential CPU golden. `NaN`, `-0.0` and
// subnormals are excluded from the parity domain: WGSL `min`/`max` NaN handling
// and the `max(-0.0, +0.0)` sign are backend-indeterminate and GPUs may flush
// subnormals to zero, so the golden's contract is only exercised on the finite,
// normal, non-negative-zero domain the rasterizer feeds.
// The kernel uses only the portable core-WGSL subset - no optional feature - so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard reversed-Z depth-key encode for atomicMax vis-buffer
// compositing; no Unreal Engine source or derived code.

struct Params {
    // Number of depths in `inputs` / keys in `keys`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;

@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    // bitcast<u32>(clamp(depth, 0, 1)) - mirrors encode_depth exactly.
    keys[i] = bitcast<u32>(clamp(inputs[i], 0.0, 1.0));
}
