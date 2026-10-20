struct SortItem {
    particle_index: u32,
    depth_key: u32,
}

struct Step {
    k: u32,
    j: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> sort_source: array<SortItem>;
@group(0) @binding(1) var<storage, read_write> sort_destination: array<SortItem>;
@group(0) @binding(2) var<uniform> step: Step;

fn item_less_or_equal(a: SortItem, b: SortItem) -> bool {
    if a.depth_key != b.depth_key {
        return a.depth_key < b.depth_key;
    }
    return a.particle_index <= b.particle_index;
}

@compute @workgroup_size(64)
fn sort_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&sort_source) {
        return;
    }
    let partner = index ^ step.j;
    let left = sort_source[index];
    let right = sort_source[partner];
    let ascending = (index & step.k) == 0u;
    let is_lower = (index & step.j) == 0u;
    let left_before_right = item_less_or_equal(left, right);
    let choose_left = (ascending == is_lower) == left_before_right;
    if choose_left {
        sort_destination[index] = left;
    } else {
        sort_destination[index] = right;
    }
}
