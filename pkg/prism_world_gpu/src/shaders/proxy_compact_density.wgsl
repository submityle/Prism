// Geometric visibility is unchanged. This compaction variant intersects it
// with coverage, avoiding a fifth storage buffer in either compute stage.
struct CompactParams {
    camera: vec4<f32>,
    projection_scale: f32,
    max_distance: f32,
    proxy_count: u32,
    reserved: u32,
};

struct VisibilityRecord {
    visible: u32,
    lod: u32,
    error_quantized: u32,
    proxy_id: u32,
    screen_error: f32,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> params: CompactParams;
@group(0) @binding(1) var<storage, read> records: array<VisibilityRecord>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: DrawArgs;
@group(0) @binding(4) var<storage, read_write> coverage: array<f32>;

var<workgroup> local_count: atomic<u32>;
var<workgroup> output_base: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = id.x;
    if lane == 0u {
        atomicStore(&local_count, 0u);
        output_base = 0u;
    }
    workgroupBarrier();
    var visible = false;
    if index < params.proxy_count {
        if records[index].visible == 0u {
            coverage[index] = 0.0;
        } else {
            visible = coverage[index] > 0.0;
        }
    }
    var local_index = 0u;
    if visible {
        local_index = atomicAdd(&local_count, 1u);
    }
    workgroupBarrier();
    if lane == 0u {
        let count = atomicLoad(&local_count);
        if count != 0u {
            output_base = atomicAdd(&args.instance_count, count);
        }
    }
    workgroupBarrier();
    if visible {
        indices[output_base + local_index] = index;
    }
}
