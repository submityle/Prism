// Export only: channel blending and auxiliary rules belong exclusively to
// response_composition.wgsl. A response is five packed RGB values plus alpha.
struct ResponseOutput {
    words: array<f32, 16>,
}

@group(0) @binding(0) var<storage, read> responses: array<ResponseOutput>;
@group(0) @binding(1) var hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var visibility_ids: texture_2d<u32>;
@group(0) @binding(3) var visibility_metadata: texture_2d<u32>;

@compute @workgroup_size(8, 8)
fn response_to_hdr(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hdr);
    if id.x >= size.x || id.y >= size.y { return; }
    let index = id.y * size.x + id.x;
    if index >= arrayLength(&responses) { return; }
    let coord = vec2<i32>(id.xy);
    // Background/invalid metadata follows the same contract as composite.wesl.
    // Preserve the resolve/sky background instead of copying undefined arms.
    if textureLoad(visibility_ids, coord, 0).x == 0xffffffffu ||
        textureLoad(visibility_metadata, coord, 0).x == 0xffffffffu { return; }
    let words = responses[index].words;
    var rgb = vec3<f32>(0.0);
    for (var channel = 0u; channel < 5u; channel += 1u) {
        let offset = 3u * channel;
        rgb += vec3<f32>(words[offset], words[offset + 1u], words[offset + 2u]);
    }
    textureStore(hdr, coord, vec4<f32>(rgb, words[15]));
}
