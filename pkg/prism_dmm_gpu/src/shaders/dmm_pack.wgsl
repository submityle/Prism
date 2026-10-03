// Displaced-micro-map raw 11-bit bitstream packing kernel.
//
// One invocation produces one output u32 word of the raw little-endian 11-bit
// displacement stream. Code `j` occupies global bit positions
// `j * 11 .. j * 11 + 11`; within that field bit `i` (bit 0 least significant)
// lands on global bit `j * 11 + i`, and global bit `p` lives in byte `p / 8`
// at bit `p % 8`. Four little-endian bytes form one u32 word, so output word
// `w` gathers global bits `w * 32 .. w * 32 + 32`: for each local bit, the
// global position `p` decodes to code index `p / 11` and in-code bit `p % 11`,
// and the extracted bit is OR-ed into the word at its local position. Reading
// each word back as little-endian bytes reproduces the CPU byte layout exactly.
//
// This mirrors `prism_dmm::pack_unorm11` element for element. Provenance:
// classical bit packing; no Unreal Engine source and no AI/ML.

// Significant bits per unorm displacement code.
const BITS_PER_CODE: u32 = 11u;
// Mask selecting the low 11 bits of a code.
const CODE_MASK: u32 = 0x7FFu;
// Bits per output word.
const BITS_PER_WORD: u32 = 32u;

struct PackParams {
    // Number of 11-bit codes to pack.
    count: u32,
    // Number of output u32 words.
    word_count: u32,
    // Padding to a 4-word uniform block.
    pad0: u32,
    // Padding to a 4-word uniform block.
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_words: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if w >= params.word_count {
        return;
    }

    var acc = 0u;
    for (var bit = 0u; bit < BITS_PER_WORD; bit += 1u) {
        let p = w * BITS_PER_WORD + bit;
        let j = p / BITS_PER_CODE;
        if j < params.count {
            let i = p % BITS_PER_CODE;
            let code = codes[j] & CODE_MASK;
            acc |= ((code >> i) & 1u) << bit;
        }
    }
    out_words[w] = acc;
}
