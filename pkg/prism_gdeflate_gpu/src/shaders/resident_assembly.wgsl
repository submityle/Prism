struct Params {
    source_offset: u32,
    destination_offset: u32,
    byte_len: u32,
    word_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.word_count { return; }
    let word = params.destination_offset / 4u + id.x;
    let first_byte = word * 4u;
    let end_byte = params.destination_offset + params.byte_len;
    var value = 0u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let byte = first_byte + lane;
        if byte >= params.destination_offset && byte < end_byte {
            let source_byte = params.source_offset + (byte - params.destination_offset);
            let octet = (source[source_byte / 4u] >> ((source_byte % 4u) * 8u)) & 255u;
            value |= octet << (lane * 8u);
        }
    }
    // Distinct slices have disjoint byte lanes, including shared seam words.
    // Fresh wgpu output is zeroed; untouched tail lanes stay zero.
    atomicOr(&destination[word], value);
}
