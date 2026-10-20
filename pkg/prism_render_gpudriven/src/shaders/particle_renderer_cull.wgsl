struct Frustum {
    planes: array<vec4<f32>, 6>,
}

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

struct ParticleInput {
    center_radius: vec4<f32>,
    depth: f32,
    particle_index: u32,
    flags: u32,
    reserved: u32,
}

struct SortItem {
    particle_index: u32,
    depth_key: u32,
}

@group(0) @binding(0) var<uniform> frustum: Frustum;
@group(0) @binding(1) var<storage, read> inputs: array<ParticleInput>;
@group(0) @binding(2) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cull_items: array<SortItem>;
@group(0) @binding(4) var<uniform> params: Params;

fn sort_key(depth: f32) -> u32 {
    let normalized = clamp((depth - params.near_depth) / (params.far_depth - params.near_depth), 0.0, 1.0);
    let quantized = u32(normalized * 65535.0 + 0.5);
    return select(quantized, 65535u - quantized, params.back_to_front != 0u);
}

fn sphere_visible(input: ParticleInput) -> bool {
    let center = input.center_radius.xyz;
    let radius = input.center_radius.w;
    if params.max_distance > 0.0 {
        let delta = center - params.camera_position.xyz;
        if dot(delta, delta) > (params.max_distance + radius) * (params.max_distance + radius) {
            return false;
        }
    }
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = frustum.planes[plane_index];
        if dot(plane.xyz, center) + plane.w < -radius {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(1)
fn reset() {
    atomicStore(&control[0], 0u);
    atomicStore(&control[1], 0u);
    atomicStore(&control[2], 0u);
    atomicStore(&control[3], 0u);
}

@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&cull_items) {
        return;
    }
    cull_items[index] = SortItem(0xffffffffu, 0xffffffffu);
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.input_count || !sphere_visible(inputs[index]) {
        return;
    }
    let output = atomicAdd(&control[0], 1u);
    if output < params.capacity {
        let input = inputs[index];
        cull_items[output] = SortItem(input.particle_index, sort_key(input.depth));
    }
}
