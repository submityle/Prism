struct SpriteArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct IndexedArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct RouteParams {
    kind: u32,
    geometry_count: u32,
    index_count: u32,
    first_instance: u32,
    sprite_vertex_count: u32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read> control: array<u32>;
@group(0) @binding(1) var<storage, read_write> sprite_args: SpriteArgs;
@group(0) @binding(2) var<storage, read_write> indexed_args: IndexedArgs;
@group(0) @binding(3) var<uniform> route: RouteParams;

@compute @workgroup_size(1)
fn emit_args() {
    let instance_count = min(control[0], route.geometry_count);
    sprite_args.vertex_count = route.sprite_vertex_count;
    sprite_args.instance_count = select(0u, instance_count, route.kind == 0u);
    sprite_args.first_vertex = 0u;
    sprite_args.first_instance = route.first_instance;
    indexed_args.index_count = route.index_count;
    indexed_args.instance_count = select(0u, instance_count, route.kind != 0u);
    indexed_args.first_index = 0u;
    indexed_args.base_vertex = 0;
    indexed_args.first_instance = route.first_instance;
}
