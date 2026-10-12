// Parallel copy stage for validated RFC 1951 stored blocks.
//
// The preceding de-interleave pass has recovered a linear raw-DEFLATE stream.
// The host has already validated all headers and supplied only payload ranges;
// this pass maps one invocation to one output word and assembles its bytes.
// Fixed/dynamic Huffman blocks never reach this shader.

struct Params {
    block_count: u32,
    output_words: u32,
    output_bytes: u32,
    _pad: u32,
};

struct Block {
    source_offset: u32,
    destination_offset: u32,
    length: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> blocks: array<Block>;
@group(0) @binding(3) var<storage, read_write> destination: array<u32>;

fn source_byte(byte_offset: u32) -> u32 {
    let word = source[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

fn output_byte(byte_offset: u32) -> u32 {
    // Destination ranges are contiguous and ordered by the validated host
    // plan. Binary search bounds work to log2(block_count), even for streams
    // consisting of many tiny stored blocks.
    var low = 0u;
    var high = params.block_count;
    for (var iteration = 0u; iteration < 32u && low < high; iteration = iteration + 1u) {
        let middle = low + (high - low) / 2u;
        let block = blocks[middle];
        if (byte_offset < block.destination_offset) {
            high = middle;
        } else if (byte_offset >= block.destination_offset + block.length) {
            low = middle + 1u;
        } else {
            return source_byte(block.source_offset + byte_offset - block.destination_offset);
        }
    }
    return 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_index = gid.x;
    if (word_index >= params.output_words) {
        return;
    }
    var value = 0u;
    for (var byte = 0u; byte < 4u; byte = byte + 1u) {
        let offset = word_index * 4u + byte;
        if (offset < params.output_bytes) {
            value = value | (output_byte(offset) << (byte * 8u));
        }
    }
    destination[word_index] = value;
}
