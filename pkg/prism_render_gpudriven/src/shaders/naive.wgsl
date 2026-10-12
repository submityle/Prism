// The naive per-object baseline: one plain vertex-shaded quad per submitted
// instance.
//
// This is the submission the GPU-driven path is measured against. The CPU has
// already baked each instance's world-space corners and color into its own
// vertex block, so the shader does nothing GPU-driven at all: it projects the
// vertex it was handed and outputs the color it was handed.

struct Frame {
    resolution: vec2<f32>,
    world_min: vec2<f32>,
    world_size: vec2<f32>,
    // x = the quad's half-extent, y = indices per instance; unused here, kept
    // so both paths read the identical uniform layout.
    params: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    let pixel = (input.position - frame.world_min) / frame.world_size * frame.resolution;

    var output: VertexOutput;
    output.clip_position = vec4<f32>(
        pixel.x / frame.resolution.x * 2.0 - 1.0,
        1.0 - pixel.y / frame.resolution.y * 2.0,
        0.5,
        1.0,
    );
    output.color = input.color;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
