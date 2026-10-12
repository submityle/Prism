// Draws the GPU-culled scene, reading every per-instance input from the GPU.
//
// The surviving instance ids live in the cull pass's `visible` list; this shader
// indexes the scene's instance storage with them to recover each survivor's
// placement and color. Nothing about the scene reaches the CPU: even the number
// of draws is read from the indirect-draw arguments the cull pass authored.
//
// Two vertex entry points cover the two indirect paths:
//
//  * `vs` is the single indirect draw. One record carries the GPU's surviving
//    count, so `instance_index` walks the visible list directly.
//  * `vs_per_draw` is the packed multi-draw path, one record per survivor. wgpu
//    does not fold a record's `first_instance` into `instance_index`, so the
//    survivor slot travels through the record's `first_index` instead: the slot
//    owns a block of `indices_per_instance` indices, so it is
//    `vertex_index / indices_per_instance` and the corner is the remainder.

struct Frame {
    resolution: vec2<f32>,
    world_min: vec2<f32>,
    world_size: vec2<f32>,
    // x = the quad's half-extent in world units, y = indices per instance.
    params: vec2<f32>,
}

struct Instance {
    center: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> visible: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

// Projects a world XY onto the target and returns its clip-space position.
fn project(world: vec2<f32>) -> vec4<f32> {
    let pixel = (world - frame.world_min) / frame.world_size * frame.resolution;
    return vec4<f32>(
        pixel.x / frame.resolution.x * 2.0 - 1.0,
        1.0 - pixel.y / frame.resolution.y * 2.0,
        0.5,
        1.0,
    );
}

// Places one survivor's quad from the scene storage and `corner_index`.
fn place(object: u32, corner_index: u32) -> VertexOutput {
    let instance = instances[object];
    // Triangle-strip corners: bit 0 selects x, bit 1 selects y.
    let corner = vec2<f32>(f32(corner_index & 1u), f32(corner_index >> 1u));
    let world = instance.center.xy
        + (corner - vec2<f32>(0.5, 0.5)) * (frame.params.x * 2.0);

    var output: VertexOutput;
    output.clip_position = project(world);
    output.color = instance.color;
    return output;
}

@vertex
fn vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    return place(visible[instance_index], vertex_index);
}

@vertex
fn vs_per_draw(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let indices = u32(frame.params.y);
    let slot = vertex_index / indices;
    let corner = vertex_index % indices;
    return place(visible[slot], corner);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
