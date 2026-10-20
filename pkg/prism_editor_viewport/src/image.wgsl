@group(0) @binding(0) var image: texture_2d<f32>;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) clip: vec4<f32>,
    @location(2) @interpolate(flat) opacity: f32,
}
@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) uv: vec2<f32>,
      @location(2) clip: vec4<f32>, @location(3) opacity: f32) -> Vertex {
    return Vertex(vec4<f32>(position, 0.0, 1.0), uv, clip, opacity);
}
fn sample_premultiplied(p: vec2<i32>, extent: vec2<i32>) -> vec4<f32> {
    // sRGB textureLoad performs transfer decoding before interpolation.
    let rgba = textureLoad(image, clamp(p, vec2<i32>(0), extent - vec2<i32>(1)), 0);
    return vec4<f32>(rgba.rgb * rgba.a, rgba.a);
}
@fragment
fn fs(vertex: Vertex) -> @location(0) vec4<f32> {
    let position = vertex.position.xy;
    if any(position < vertex.clip.xy) || any(position >= vertex.clip.zw) { discard; }
    let extent = vec2<i32>(textureDimensions(image));
    let p = vertex.uv * vec2<f32>(extent) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    let a = mix(sample_premultiplied(base, extent), sample_premultiplied(base + vec2<i32>(1,0), extent), f.x);
    let b = mix(sample_premultiplied(base + vec2<i32>(0,1), extent), sample_premultiplied(base + vec2<i32>(1,1), extent), f.x);
    return mix(a, b, f.y) * vertex.opacity;
}
