// Draws the GPU-culled survivors.
//
// Every per-instance input comes from the GPU: the instance count lives in the
// indirect-draw arguments the cull pass authored, and the surviving instance
// ids live in the GPU-written visible list, which this shader reads as storage.
//
// Two vertex entry points cover the two indirect paths, because they address
// the visible list differently:
//
//  * `vs` is the single indirect draw. Its whole point is that one record
//    carries the GPU's surviving count, so `instance_index` walks the visible
//    list directly.
//  * `vs_per_draw` is the multi-draw path, one record per survivor. wgpu does
//    not fold a record's `first_instance` into `instance_index` (Metal's draw
//    calls have no base-instance meaning here), so the survivor slot has to
//    travel through the record's `first_index` instead. Each slot owns a block
//    of `indices_per_instance` indices in the shared index buffer, so the slot
//    is `vertex_index / indices_per_instance` and the corner is the remainder.

struct DrawParams {
    resolution: vec2<f32>,
    // The side of one instance's square cell, in pixels.
    cell_pixels: f32,
    // How many cells fit along one row; also the modulus that tiles instance
    // ids across the screen.
    cells_per_row: u32,
    // The quad's index count: the size of one slot's index block.
    indices_per_instance: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> draw_params: DrawParams;
@group(0) @binding(1) var<storage, read> visible: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

// Places one survivor's quad: `instance` is the survivor id from the GPU list,
// `corner_index` selects one of the four triangle-strip corners.
fn place(instance: u32, corner_index: u32) -> VertexOutput {
    let cell_x = f32(instance % draw_params.cells_per_row);
    let cell_y = f32(instance / draw_params.cells_per_row);

    // A half-cell-sized quad centred in the instance's cell, from a
    // triangle-strip corner index (0,0) (1,0) (0,1) (1,1).
    let corner = vec2<f32>(f32(corner_index & 1u), f32(corner_index >> 1u));
    let half_size = draw_params.cell_pixels * 0.25;
    let center = (vec2<f32>(cell_x, cell_y) + vec2<f32>(0.5, 0.5)) * draw_params.cell_pixels;
    let pixel = center + (corner - vec2<f32>(0.5, 0.5)) * (half_size * 2.0);

    var output: VertexOutput;
    output.clip_position = vec4<f32>(
        pixel.x / (draw_params.resolution.x * 0.5) - 1.0,
        1.0 - pixel.y / (draw_params.resolution.y * 0.5),
        0.5,
        1.0,
    );
    output.color = vec3<f32>(1.0, 1.0, 1.0);
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
    let indices = draw_params.indices_per_instance;
    let slot = vertex_index / indices;
    let corner = vertex_index % indices;
    return place(visible[slot], corner);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
