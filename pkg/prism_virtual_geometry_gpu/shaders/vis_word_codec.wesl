// 64-bit vis-buffer word codec twin: packs a (depth_key, payload) pair into the
// shipping 64-bit visibility word and unpacks it back, mirroring the CPU golden
// const fns in `prism_render_architecture::virtual_geometry` (`software_raster.rs`):
//
//   pack_vis(depth_key, payload) = (u64(depth_key) << 32) | u64(payload)
//   vis_depth(packed)   = u32(packed >> 32)
//   vis_payload(packed)  = u32(packed & 0xFFFF_FFFF)
//
// The software rasterizer composites exactly this word through a 64-bit
// `atomicMax`: the reversed-Z depth key in the high 32 bits (so the nearest
// surface wins the max) and the payload in the low 32 bits (carried along for
// free). The payload-raster twin validates the *atomic composite* of that word,
// but only within a one-ULP depth tolerance and never runs the pack/unpack on
// device in isolation. This kernel closes that gap: it runs the pure 64-bit
// shift-and-mask codec per thread and is asserted bit-exact against the
// reference, so the 64-bit word ABI a GPU-driven rasterizer must emit is
// pinned down on its own.
//
// It needs the SHADER_INT64 `u64` type (no atomics), so - like the payload
// raster - the host gates compilation on device support and skips gracefully
// where 64-bit integers are unavailable. This is the 64-bit sibling of the
// portable 32-bit `vis_payload_codec`.
//
// The result is exact: 64-bit shift, OR and mask are defined identically on
// every backend that supports them, so the packed word and both unpacked
// fields match the reference index-for-index with no tolerance.
//
// Provenance: Prism's own vis-buffer word bit layout; no Unreal Engine source
// or derived code.

struct Params {
    // Number of valid inputs in `inputs` and outputs in `outputs`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One (depth_key, payload) pair to encode. 8-byte stride.
struct PackInput {
    depth_key: u32,
    payload: u32,
}

// One codec result: the packed 64-bit word plus both fields unpacked back out.
// `u64` forces 8-byte alignment; 8 + 4 + 4 = 16-byte stride, no trailing pad.
struct CodecOutput {
    packed: u64,
    depth: u32,
    payload: u32,
}

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
    let packed: u64 = (u64(in.depth_key) << 32u) | u64(in.payload);
    var out: CodecOutput;
    out.packed = packed;
    out.depth = u32(packed >> 32u);
    // Low 32 bits: a truncating u64->u32 cast, equivalent to the golden's
    // `(packed & 0xFFFF_FFFF) as u32`. A `& 0xFFFF_FFFF` literal here would be
    // u32-typed and naga rejects mixing it with the u64 operand.
    out.payload = u32(packed);
    outputs[i] = out;
}
