struct Frame {
    resolution: vec2<f32>,
    world_min: vec2<f32>,
    world_size: vec2<f32>,
    size: f32,
    _pad: f32,
    color: vec4<f32>,
}

struct Particle {
    position_life: vec4<f32>,
    velocity_pad: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> active_indices: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn project(world: vec2<f32>) -> vec4<f32> {
    let pixel = (world - frame.world_min) / frame.world_size * frame.resolution;
    return vec4<f32>(
        pixel.x / frame.resolution.x * 2.0 - 1.0,
        1.0 - pixel.y / frame.resolution.y * 2.0,
        0.5,
        1.0,
    );
}

@vertex
fn vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = particles[active_indices[instance_index]];
    let corner = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u));
    let world = particle.position_life.xy
        + (corner - vec2<f32>(0.5, 0.5)) * frame.size * 2.0;
    var output: VertexOutput;
    output.clip_position = project(world);
    output.color = frame.color;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
