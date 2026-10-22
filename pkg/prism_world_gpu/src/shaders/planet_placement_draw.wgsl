struct DrawUniform {
    frame_view: mat4x4<f32>,
    target_size: vec2<f32>,
    instance_size_pixels: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> draw: DrawUniform;
@group(0) @binding(1) var<storage, read> records: array<u32>;
@group(0) @binding(2) var<storage, read> compact: array<u32>;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) rank: f32,
};

// WGSL has no portable i128/f64. Decode the producer's 4-limb two's-complement
// integer exactly as limbs, take its unsigned magnitude, then convert to f32.
// This is the sole precision boundary: millimetres become frame-local metres
// here, after the producer has already subtracted output_origin_mm in i128.
fn signed_i128_mm_to_meters(limbs_in: vec4<u32>) -> f32 {
    let negative = (limbs_in.w & 0x80000000u) != 0u;
    var magnitude = limbs_in;
    if (negative) {
        magnitude = ~magnitude;
        let a = magnitude.x + 1u;
        magnitude.y = magnitude.y + select(0u, 1u, a == 0u);
        let b = magnitude.y;
        magnitude.z = magnitude.z + select(0u, 1u, a == 0u && b == 0u);
        let c = magnitude.z;
        magnitude.w = magnitude.w + select(0u, 1u, a == 0u && b == 0u && c == 0u);
        magnitude.x = a;
    }
    let mm = f32(magnitude.x)
        + f32(magnitude.y) * 4294967296.0
        + f32(magnitude.z) * 18446744073709551616.0
        + f32(magnitude.w) * 79228162514264337593543950336.0;
    return select(mm, -mm, negative) * 0.001;
}

fn word(source_slot: u32, field: u32) -> u32 {
    return records[source_slot * 29u + field];
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    let source_slot = compact[instance_index];
    let x = signed_i128_mm_to_meters(vec4<u32>(
        word(source_slot, 6u), word(source_slot, 7u),
        word(source_slot, 8u), word(source_slot, 9u)));
    let y = signed_i128_mm_to_meters(vec4<u32>(
        word(source_slot, 10u), word(source_slot, 11u),
        word(source_slot, 12u), word(source_slot, 13u)));
    let z = signed_i128_mm_to_meters(vec4<u32>(
        word(source_slot, 14u), word(source_slot, 15u),
        word(source_slot, 16u), word(source_slot, 17u)));
    let center = draw.frame_view * vec4<f32>(x, y, z, 1.0);
    // The producer's indirect vertex_count is fixed at three; render a small
    // equilateral marker and derive its size in target pixels.
    let corners = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.57735026),
        vec2<f32>(-0.5, -0.28867513),
        vec2<f32>(0.5, -0.28867513));
    let pixel_delta = corners[vertex_index % 3u] * draw.instance_size_pixels;
    let clip_delta = vec2<f32>(
        2.0 * pixel_delta.x / draw.target_size.x,
        -2.0 * pixel_delta.y / draw.target_size.y) * center.w;
    var output: VertexOut;
    output.clip_position = vec4<f32>(center.xy + clip_delta, center.z, center.w);
    output.normal = normalize(vec3<f32>(
        bitcast<f32>(word(source_slot, 18u)),
        bitcast<f32>(word(source_slot, 19u)),
        bitcast<f32>(word(source_slot, 20u))));
    output.rank = f32(word(source_slot, 22u)) / 4294967295.0;
    return output;
}

@fragment
fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
    let normal_color = abs(normalize(input.normal));
    let rank_tint = 0.25 + 0.75 * input.rank;
    return vec4<f32>(normal_color * rank_tint, 1.0);
}
