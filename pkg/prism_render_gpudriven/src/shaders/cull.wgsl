// GPU frustum culling.
//
// One invocation per instance AABB: classify it against the six frustum
// planes, append the survivors to a visible-index list, and author the
// indirect-draw arguments. The CPU uploads the candidate set and the fixed
// draw constants; it never learns how many instances survive, so the draw it
// later records is bounded by nothing but the GPU's own result.

struct Frustum {
    // Six planes as (normal.xyz, distance): a point p is inside the half-space
    // when dot(normal, p) + distance >= 0.
    planes: array<vec4<f32>, 6>,
}

struct Aabb {
    min: vec3<f32>,
    _pad0: f32,
    max: vec3<f32>,
    _pad1: f32,
}

// The indirect arguments the draw consumes. `instance_count` doubles as the
// atomic append cursor: the slot an instance reserves is its final index in the
// draw, and once every reservation has landed the field *is* the surviving
// count the draw reads back.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// One packed record per survivor, for the multi-draw-indirect path: a single
// one-instance indexed draw that reads the survivor's own index block. The slot
// is named through `first_index` rather than `first_instance` because wgpu does
// not fold `first_instance` into the vertex shader's instance index (see
// draw.wgsl).
struct DrawArgRecord {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct Params {
    // How many candidate instances were uploaded (<= the buffer's capacity).
    count: u32,
    // The geometry's index count, and where its indices start.
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
}

@group(0) @binding(0) var<uniform> frustum: Frustum;
@group(0) @binding(1) var<storage, read> instances: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> visible: array<u32>;
@group(0) @binding(3) var<storage, read_write> draw_args: DrawArgs;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> multi_args: array<DrawArgRecord>;

// The positive-vertex test: the box is outside only when even its corner
// furthest along the plane normal is behind that plane.
fn aabb_visible(box: Aabb) -> bool {
    for (var plane_index = 0u; plane_index < 6u; plane_index = plane_index + 1u) {
        let plane = frustum.planes[plane_index];
        let normal = plane.xyz;
        let extreme = vec3<f32>(
            select(box.min.x, box.max.x, normal.x >= 0.0),
            select(box.min.y, box.max.y, normal.y >= 0.0),
            select(box.min.z, box.max.z, normal.z >= 0.0),
        );
        if (dot(normal, extreme) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.count) {
        return;
    }
    if (!aabb_visible(instances[index])) {
        return;
    }

    let slot = atomicAdd(&draw_args.instance_count, 1u);
    visible[slot] = index;
    multi_args[slot] = DrawArgRecord(
        params.index_count,
        1u,
        // The survivor's index block base: slot times the block size.
        slot * params.index_count,
        params.base_vertex,
        0u,
    );
}
