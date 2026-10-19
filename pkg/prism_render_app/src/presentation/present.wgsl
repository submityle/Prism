@group(0) @binding(0) var scene: texture_2d<f32>;
const TARGET_SRGB: bool = false;

fn srgb_decode(value: vec3<f32>) -> vec3<f32> {
    let linear_segment = value / vec3<f32>(12.92);
    let power_segment = pow((value + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(power_segment, linear_segment, value <= vec3<f32>(0.04045));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle, with no vertex buffer or sampler allocation.
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let color = textureLoad(scene, vec2<i32>(pixel.xy), 0);
    if (TARGET_SRGB) {
        return vec4<f32>(srgb_decode(color.rgb), color.a);
    }
    return color;
}
