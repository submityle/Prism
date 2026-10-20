// One invocation decodes one GDeflate tile.  The tile's DEFLATE stream is
// serial by nature (LZ77 references have read-after-write dependencies), but
// the current host path dispatches tiles one at a time. This shader reads the
// lane-major GDeflate words directly; it does not materialize a linear
// intermediate.
// Fixed-only callers select params.fixed_only; the unified decoder accepts
// every non-reserved block type and constructs all dynamic tables here.

struct Params {
    source_words: u32,
    rounds: u32,
    compressed_bytes: u32,
    output_bytes: u32,
    fixed_only: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

struct Reader {
    byte_pos: u32,
    bit_buf: u32,
    bit_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
// status[0]: 0=success, 1=unsupported block type, 2=truncated input,
// 3=invalid code/symbol, 4=output size mismatch, 5=distance too far,
// 6=invalid invocation parameters. status[1] is bytes written.
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

const STATUS_OK: u32 = 0u;
const STATUS_UNSUPPORTED: u32 = 1u;
const STATUS_TRUNCATED: u32 = 2u;
const STATUS_INVALID: u32 = 3u;
const STATUS_OUTPUT: u32 = 4u;
const STATUS_DISTANCE: u32 = 5u;
const STATUS_PARAMETERS: u32 = 6u;

const LENGTH_BASE: array<u32, 29> = array<u32, 29>(
    3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u, 15u, 17u, 19u, 23u,
    27u, 31u, 35u, 43u, 51u, 59u, 67u, 83u, 99u, 115u, 131u, 163u,
    195u, 227u, 258u,
);
const LENGTH_EXTRA: array<u32, 29> = array<u32, 29>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u,
    2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 0u,
);
const DIST_BASE: array<u32, 30> = array<u32, 30>(
    1u, 2u, 3u, 4u, 5u, 7u, 9u, 13u, 17u, 25u, 33u, 49u, 65u, 97u,
    129u, 193u, 257u, 385u, 513u, 769u, 1025u, 1537u, 2049u, 3073u,
    4097u, 6145u, 8193u, 12289u, 16385u, 24577u,
);
const DIST_EXTRA: array<u32, 30> = array<u32, 30>(
    0u, 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u,
    6u, 7u, 7u, 8u, 8u, 9u, 9u, 10u, 10u, 11u, 11u, 12u, 12u, 13u, 13u,
);

const CL_ORDER: array<u32, 19> = array<u32, 19>(
    16u, 17u, 18u, 0u, 8u, 7u, 9u, 6u, 10u, 5u, 11u, 4u, 12u, 3u, 13u, 2u, 14u, 1u, 15u,
);

// Invocation-private bounded scratch. Tables 0/1/2 hold literal-length,
// distance and code-length alphabets, respectively. No host-built decode
// tables or CPU block planning are used.
struct Huffman {
    counts: array<u32, 16>,
    symbols: array<u32, 288>,
}
var<private> tables: array<Huffman, 3>;
var<private> lengths: array<u32, 320>;

fn build_table(table: u32, start: u32, count: u32, max_bits: u32) -> bool {
    for (var bits = 0u; bits <= 15u; bits = bits + 1u) {
        tables[table].counts[bits] = 0u;
    }
    for (var symbol = 0u; symbol < count; symbol = symbol + 1u) {
        let bits = lengths[start + symbol];
        if (bits > max_bits) { return false; }
        if (bits != 0u) {
            tables[table].counts[bits] = tables[table].counts[bits] + 1u;
        }
    }
    var left = 1u;
    var total = 0u;
    var offsets: array<u32, 16>;
    for (var bits = 1u; bits <= 15u; bits = bits + 1u) {
        left = left * 2u;
        let used = tables[table].counts[bits];
        if (used > left) { return false; }
        left = left - used;
        offsets[bits] = total;
        total = total + used;
    }
    // An unused distance alphabet is legal in a literals-only block. The
    // only other incomplete tree allowed is a single one-bit lit/dist code.
    if (left != 0u && !(table == 1u && total == 0u) &&
        !(table != 2u && total == 1u && tables[table].counts[1] == 1u)) {
        return false;
    }
    for (var symbol = 0u; symbol < count; symbol = symbol + 1u) {
        let bits = lengths[start + symbol];
        if (bits != 0u) {
            tables[table].symbols[offsets[bits]] = symbol;
            offsets[bits] = offsets[bits] + 1u;
        }
    }
    return true;
}

fn decode_symbol(table: u32, reader: ptr<function, Reader>, ok: ptr<function, bool>) -> u32 {
    var code = 0u;
    var first = 0u;
    var index = 0u;
    for (var bits = 1u; bits <= 15u; bits = bits + 1u) {
        code = code | read_bits(reader, 1u, ok);
        if (!*ok) { return 0u; }
        let count = tables[table].counts[bits];
        if (code >= first && code - first < count) {
            return tables[table].symbols[index + code - first];
        }
        index = index + count;
        first = (first + count) << 1u;
        code = code << 1u;
    }
    *ok = false;
    return 0u;
}

fn dynamic_tables(reader: ptr<function, Reader>, ok: ptr<function, bool>) {
    let hlit = read_bits(reader, 5u, ok) + 257u;
    let hdist = read_bits(reader, 5u, ok) + 1u;
    let hclen = read_bits(reader, 4u, ok) + 4u;
    if (!*ok || hlit > 286u) { *ok = false; return; }
    for (var i = 0u; i < 19u; i = i + 1u) { lengths[i] = 0u; }
    for (var i = 0u; i < hclen; i = i + 1u) {
        lengths[CL_ORDER[i]] = read_bits(reader, 3u, ok);
    }
    if (!*ok || !build_table(2u, 0u, 19u, 7u)) { *ok = false; return; }
    let total = hlit + hdist;
    var i = 0u;
    loop {
        if (i >= total) { break; }
        let symbol = decode_symbol(2u, reader, ok);
        if (!*ok) { return; }
        var value = symbol;
        var repeat = 1u;
        if (symbol == 16u) {
            if (i == 0u) { *ok = false; return; }
            value = lengths[i - 1u];
            repeat = read_bits(reader, 2u, ok) + 3u;
        } else if (symbol == 17u) {
            value = 0u;
            repeat = read_bits(reader, 3u, ok) + 3u;
        } else if (symbol == 18u) {
            value = 0u;
            repeat = read_bits(reader, 7u, ok) + 11u;
        }
        if (!*ok || repeat > total - i) { *ok = false; return; }
        for (var j = 0u; j < repeat; j = j + 1u) {
            lengths[i] = value;
            i = i + 1u;
        }
    }
    // Every compressed block must have an end-of-block code.
    if (lengths[256] == 0u || !build_table(0u, 0u, hlit, 15u) ||
        !build_table(1u, hlit, hdist, 15u)) { *ok = false; }
}

fn input_byte(offset: u32) -> u32 {
    let logical_word = offset / 4u;
    let lane = logical_word % 32u;
    let round = logical_word / 32u;
    let stored_word = lane * params.rounds + round;
    let word = source[stored_word];
    return (word >> ((offset % 4u) * 8u)) & 0xffu;
}

fn read_bits(reader: ptr<function, Reader>, count: u32, ok: ptr<function, bool>) -> u32 {
    loop {
        if ((*reader).bit_count >= count) {
            break;
        }
        if ((*reader).byte_pos >= params.compressed_bytes) {
            *ok = false;
            return 0u;
        }
        (*reader).bit_buf = (*reader).bit_buf | (input_byte((*reader).byte_pos) << (*reader).bit_count);
        (*reader).byte_pos = (*reader).byte_pos + 1u;
        (*reader).bit_count = (*reader).bit_count + 8u;
    }
    let mask = (1u << count) - 1u;
    let value = (*reader).bit_buf & mask;
    (*reader).bit_buf = (*reader).bit_buf >> count;
    (*reader).bit_count = (*reader).bit_count - count;
    return value;
}

// Matches the compact canonical walk used by the fixed code table. `code -
// first` is already the relative index within the current code length; the
// symbol mapping below is not a cumulative count across shorter lengths.
fn fixed_litlen(reader: ptr<function, Reader>, ok: ptr<function, bool>) -> u32 {
    var code = 0u;
    var first = 0u;
    for (var len = 1u; len <= 9u; len = len + 1u) {
        code = code | read_bits(reader, 1u, ok);
        if (!*ok) {
            return 0u;
        }
        var count = 0u;
        if (len == 7u) {
            count = 24u;
        } else if (len == 8u) {
            count = 152u;
        } else if (len == 9u) {
            count = 112u;
        }
        if (code >= first && code - first < count) {
            let relative = code - first;
            if (len == 7u) {
                return 256u + relative;
            }
            if (len == 8u) {
                if (relative < 144u) {
                    return relative;
                }
                return 280u + relative - 144u;
            }
            return 144u + relative;
        }
        first = (first + count) << 1u;
        code = code << 1u;
    }
    *ok = false;
    return 0u;
}

fn fixed_distance(reader: ptr<function, Reader>, ok: ptr<function, bool>) -> u32 {
    var code = 0u;
    for (var len = 1u; len <= 5u; len = len + 1u) {
        code = code | read_bits(reader, 1u, ok);
        if (!*ok) {
            return 0u;
        }
        if (len == 5u) {
            return code;
        }
        code = code << 1u;
    }
    *ok = false;
    return 0u;
}

fn output_byte(offset: u32) -> u32 {
    return (output[offset / 4u] >> ((offset % 4u) * 8u)) & 0xffu;
}

fn write_byte(offset: u32, value: u32) {
    let word_index = offset / 4u;
    let shift = (offset % 4u) * 8u;
    let mask = 0xffu << shift;
    output[word_index] = (output[word_index] & ~mask) | ((value & 0xffu) << shift);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x != 0u) {
        return;
    }
    status[0] = STATUS_PARAMETERS;
    status[1] = 0u;
    if (params.rounds == 0u || params.source_words != params.rounds * 32u ||
        params.compressed_bytes == 0u || params.compressed_bytes > params.source_words * 4u) {
        return;
    }

    var reader = Reader(0u, 0u, 0u);
    var written = 0u;
    var final_block = false;
    loop {
        var ok = true;
        final_block = read_bits(&reader, 1u, &ok) != 0u;
        let block_type = read_bits(&reader, 2u, &ok);
        if (!ok) {
            status[0] = STATUS_TRUNCATED;
            status[1] = written;
            return;
        }
        if (block_type == 3u || (params.fixed_only != 0u && block_type != 1u)) {
            status[0] = STATUS_UNSUPPORTED;
            status[1] = written;
            return;
        }

        if (block_type == 0u) {
            // read_bits never prefetches beyond the requested byte: discard
            // only alignment bits, then validate LEN/NLEN on the device.
            let drop = reader.bit_count % 8u;
            reader.bit_buf = reader.bit_buf >> drop;
            reader.bit_count = reader.bit_count - drop;
            let size = read_bits(&reader, 16u, &ok);
            let complement = read_bits(&reader, 16u, &ok);
            if (!ok || (size ^ complement) != 0xffffu) {
                status[0] = STATUS_INVALID;
                status[1] = written;
                return;
            }
            if (size > params.output_bytes - written) {
                status[0] = STATUS_OUTPUT;
                status[1] = written;
                return;
            }
            for (var i = 0u; i < size; i = i + 1u) {
                let value = read_bits(&reader, 8u, &ok);
                if (!ok) {
                    status[0] = STATUS_TRUNCATED;
                    status[1] = written;
                    return;
                }
                write_byte(written, value);
                written = written + 1u;
            }
            if (final_block) { break; }
            continue;
        }
        if (block_type == 2u) {
            dynamic_tables(&reader, &ok);
            if (!ok) {
                status[0] = STATUS_INVALID;
                status[1] = written;
                return;
            }
        }

        loop {
            var symbol = 0u;
            if (block_type == 1u) {
                symbol = fixed_litlen(&reader, &ok);
            } else {
                symbol = decode_symbol(0u, &reader, &ok);
            }
            if (!ok) {
                status[0] = STATUS_INVALID;
                status[1] = written;
                return;
            }
            if (symbol <= 255u) {
                if (written >= params.output_bytes) {
                    status[0] = STATUS_OUTPUT;
                    status[1] = written;
                    return;
                }
                write_byte(written, symbol);
                written = written + 1u;
                continue;
            }
            if (symbol == 256u) {
                break;
            }
            if (symbol < 257u || symbol > 285u) {
                status[0] = STATUS_INVALID;
                status[1] = written;
                return;
            }
            let length_index = symbol - 257u;
            let length = LENGTH_BASE[length_index] + read_bits(&reader, LENGTH_EXTRA[length_index], &ok);
            var distance_symbol = 0u;
            if (block_type == 1u) {
                distance_symbol = fixed_distance(&reader, &ok);
            } else {
                distance_symbol = decode_symbol(1u, &reader, &ok);
            }
            if (!ok || distance_symbol >= 30u) {
                status[0] = STATUS_INVALID;
                status[1] = written;
                return;
            }
            let distance = DIST_BASE[distance_symbol] + read_bits(&reader, DIST_EXTRA[distance_symbol], &ok);
            if (!ok) {
                status[0] = STATUS_TRUNCATED;
                status[1] = written;
                return;
            }
            if (distance > written) {
                status[0] = STATUS_DISTANCE;
                status[1] = written;
                return;
            }
            if (length > params.output_bytes - written) {
                status[0] = STATUS_OUTPUT;
                status[1] = written;
                return;
            }
            for (var copy_index = 0u; copy_index < length; copy_index = copy_index + 1u) {
                write_byte(written, output_byte(written - distance));
                written = written + 1u;
            }
        }
        if (final_block) {
            break;
        }
    }
    status[1] = written;
    if (written != params.output_bytes) {
        status[0] = STATUS_OUTPUT;
        return;
    }
    status[0] = STATUS_OK;
}
