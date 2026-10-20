// GPU object-ID debug preview only. This is not an engine material shader.
struct Preview {
    origin_extent: vec4<f32>,
    selected: vec4<u32>,
};

@group(0) @binding(0) var ids: texture_2d<u32>;
@group(0) @binding(1) var<uniform> preview: Preview;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let positions = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(positions[i], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(ids));
    let uv = (p.xy - preview.origin_extent.xy) / preview.origin_extent.zw;
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    let id = textureLoad(ids, pixel, 0);
    if all(id == vec4<u32>(0u)) {
        return vec4(0.025, 0.03, 0.045, 1.0);
    }
    let hash = id.x * 1664525u + id.y * 1013904223u;
    let rgb = vec3<u32>(hash, hash >> 8u, hash >> 16u) & vec3<u32>(255u);
    var color = vec3<f32>(0.2) + 0.65 * vec3<f32>(rgb) / 255.0;
    if all(id == preview.selected) {
        color = mix(color, vec3(1.0, 0.85, 0.25), 0.65);
    }
    return vec4(color, 1.0);
}
