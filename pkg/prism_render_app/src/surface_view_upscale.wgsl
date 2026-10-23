struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
var<immediate> params: vec4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var output: VertexOutput;
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(source));
    let position = clamp(input.uv * vec2<f32>(size) - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(size - vec2<i32>(1)));
    let lower = vec2<i32>(floor(position));
    let upper = min(lower + vec2<i32>(1), size - vec2<i32>(1));
    let weight = fract(position);
    let a = textureLoad(source, vec2<i32>(lower.x, lower.y), 0);
    let b = textureLoad(source, vec2<i32>(upper.x, lower.y), 0);
    let c = textureLoad(source, vec2<i32>(lower.x, upper.y), 0);
    let d = textureLoad(source, vec2<i32>(upper.x, upper.y), 0);
    let hdr = mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
    return vec4<f32>(hdr.rgb * exp2(params.x), hdr.a);
}
