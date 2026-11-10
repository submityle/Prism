// Raw byte gathering only: no texture loads, numeric/float/sRGB conversions,
// atomics, or cross-invocation read/modify/write of destination words.
struct Params {
    page_size: u32,
    gutter: u32,
    bytes_per_pixel: u32,
    extent: u32,
    bytes_per_row: u32,
    word_count: u32,
    padding_0: u32,
    padding_1: u32,
    // offset, cropped width, cropped height, present; 3x3 row-major.
    regions: array<vec4<u32>, 9>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination_words: array<u32>;

fn page_offset(coordinate: i32) -> i32 {
    if coordinate < 0 { return -1; }
    if coordinate >= i32(params.page_size) { return 1; }
    return 0;
}

fn rem_euclid_page(coordinate: i32) -> u32 {
    let remainder = coordinate % i32(params.page_size);
    if remainder < 0 { return u32(remainder + i32(params.page_size)); }
    return u32(remainder);
}

fn sample_byte(output_byte: u32) -> u32 {
    let output_y = output_byte / params.bytes_per_row;
    let row_byte = output_byte % params.bytes_per_row;
    if row_byte >= params.extent * params.bytes_per_pixel {
        return 0u;
    }
    let centered_x = i32(row_byte / params.bytes_per_pixel) - i32(params.gutter);
    let centered_y = i32(output_y) - i32(params.gutter);
    // Classification deliberately selects only immediate neighbors even when
    // a gutter is wider than a logical page, matching material_vt_atlas.rs.
    let index = u32((page_offset(centered_y) + 1) * 3 + page_offset(centered_x) + 1);
    var region = params.regions[index];
    var source_x: u32;
    var source_y: u32;
    if index != 4u && region.w != 0u {
        source_x = rem_euclid_page(centered_x);
        source_y = rem_euclid_page(centered_y);
    } else {
        // A missing neighbor falls back to the center on BOTH axes.
        region = params.regions[4];
        source_x = u32(clamp(centered_x, 0, i32(params.page_size) - 1));
        source_y = u32(clamp(centered_y, 0, i32(params.page_size) - 1));
    }
    source_x = min(source_x, region.y - 1u);
    source_y = min(source_y, region.z - 1u);
    let source_byte = region.x
        + (source_y * region.y + source_x) * params.bytes_per_pixel
        + row_byte % params.bytes_per_pixel;
    return (source_words[source_byte / 4u] >> ((source_byte % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    if word >= params.word_count { return; }
    let first_byte = word * 4u;
    var word_value = 0u;
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        word_value = word_value | (sample_byte(first_byte + lane) << (lane * 8u));
    }
    destination_words[word] = word_value;
}
