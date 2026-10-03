// Opacity-micromap bit-packing kernel.
//
// One invocation packs one 32-bit output word of the DXR micromap byte layout.
// For the four-state (2-bit) format a word holds 16 micro-triangles at bit
// offsets `r * 2`; for the two-state (1-bit) format a word holds 32
// micro-triangles at bit offset `r`. The low-to-high within-byte order matches
// `VK_EXT_opacity_micromap` and DXR 1.2.
//
// This mirrors `prism_micromap::omm::pack` so the host CPU golden and this GPU
// twin produce byte-identical data. Provenance: classical bit packing; no
// Unreal Engine source and no AI/ML.

struct PackParams {
    format: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> states: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_words: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let count = params.count;

    if params.format == 1u {
        // Four-state: 16 entries of 2 bits per word.
        let word_count = (count + 15u) / 16u;
        if w >= word_count {
            return;
        }
        var acc = 0u;
        for (var r = 0u; r < 16u; r += 1u) {
            let i = w * 16u + r;
            if i < count {
                let code = states[i] & 3u;
                acc |= code << (r * 2u);
            }
        }
        out_words[w] = acc;
    } else {
        // Two-state: 32 entries of 1 bit per word.
        let word_count = (count + 31u) / 32u;
        if w >= word_count {
            return;
        }
        var acc = 0u;
        for (var r = 0u; r < 32u; r += 1u) {
            let i = w * 32u + r;
            if i < count {
                let bit = states[i] & 1u;
                acc |= bit << r;
            }
        }
        out_words[w] = acc;
    }
}
