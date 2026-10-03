// GDeflate warp-interleave de-interleave (reversible 32-bit word transpose).
//
// One invocation produces exactly one *linear* output word. The GDeflate
// container stores a single standard DEFLATE word stream transposed into
// lane-major order: logical word `r * LANES + lane` lives at lane-major
// position `lane * rounds + r`. Reversing that transpose is a pure permutation
// of whole 32-bit words, so the recovered linear stream is the ordinary DEFLATE
// bitstream the host inflate core then decodes.
//
// Given the linear output word index `logical`, this kernel recovers
// `lane = logical % LANES`, `r = logical / LANES`, and gathers from the stored
// source word `lane * rounds + r`. No bit is reinterpreted — a whole word is
// moved — so device output equals the CPU golden de-interleave bit-for-bit
// (exact equality, not tolerance). No floating point is involved anywhere.
//
// Provenance: the public GDeflate/DirectStorage format description only. No
// Unreal Engine or NVIDIA GDeflate source or derived code.

const LANES: u32 = 32u;

struct Params {
    // Number of linear output words to produce (== source word count).
    word_count: u32,
    // Interleave rounds across the 32 lanes (`word_count / LANES`).
    rounds: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Lane-major (warp-interleaved) source words.
@group(0) @binding(1) var<storage, read> src: array<u32>;
// Recovered linear (de-interleaved) output words.
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let logical = gid.x;
    if (logical >= params.word_count) {
        return;
    }
    let lane = logical % LANES;
    let r = logical / LANES;
    let stored = lane * params.rounds + r;
    dst[logical] = src[stored];
}
