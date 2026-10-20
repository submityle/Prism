struct Params {
    capacity: u32,
    input_count: u32,
    back_to_front: u32,
    _pad0: u32,
    near_depth: f32,
    far_depth: f32,
    max_distance: f32,
    _pad1: f32,
    camera_position: vec4<f32>,
    _pad2: vec4<f32>,
}

struct SortItem {
    particle_index: u32,
    depth_key: u32,
}

@group(0) @binding(0) var<storage, read> compact_source: array<SortItem>;
@group(0) @binding(1) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> control: array<u32>;

@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let visible_count = min(control[0], params.capacity);
    if index >= visible_count {
        return;
    }
    visible_indices[index] = compact_source[index].particle_index;
}
