struct BoxSelectionParams {
    min_x: u32,
    min_y: u32,
    width: u32,
    height: u32,
    table_mask: u32,
    max_ids: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: BoxSelectionParams;
@group(0) @binding(1) var ids: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> locks: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> values: array<atomic<u32>>;
// stats[0] is the number of inserted IDs; stats[1] is the overflow flag.
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

fn hash_id(id: vec4<u32>) -> u32 {
    var hash = id.x * 0x9E3779B1u + id.y;
    hash = (hash ^ (hash >> 16u)) * 0x85EBCA6Bu;
    hash = (hash ^ id.z) * 0xC2B2AE35u;
    hash = (hash ^ id.w) * 0x27D4EB2Fu;
    return hash ^ (hash >> 15u);
}

fn same_id(base: u32, id: vec4<u32>) -> bool {
    return atomicLoad(&values[base]) == id.x &&
        atomicLoad(&values[base + 1u]) == id.y &&
        atomicLoad(&values[base + 2u]) == id.z &&
        atomicLoad(&values[base + 3u]) == id.w;
}

fn insert_id(id: vec4<u32>) {
    var slot = hash_id(id) & params.table_mask;
    var probe = 0u;
    loop {
        if (probe > params.table_mask) {
            atomicStore(&stats[1], 1u);
            return;
        }

        var state = atomicLoad(&locks[slot]);
        if (state == 0u) {
            let acquired = atomicCompareExchangeWeak(&locks[slot], 0u, 1u);
            if (acquired.exchanged) {
                let base = slot * 4u;
                atomicStore(&values[base], id.x);
                atomicStore(&values[base + 1u], id.y);
                atomicStore(&values[base + 2u], id.z);
                atomicStore(&values[base + 3u], id.w);
                let previous = atomicAdd(&stats[0], 1u);
                if (previous >= params.max_ids) {
                    atomicStore(&stats[1], 1u);
                }
                // A claimed slot is never returned to empty. This prevents a
                // later invocation from observing stale zeroed value words.
                atomicStore(&locks[slot], 2u);
                return;
            }
            state = atomicLoad(&locks[slot]);
        }
        if (state == 1u) {
            loop {
                state = atomicLoad(&locks[slot]);
                if (state != 1u) {
                    break;
                }
            }
        }
        if (state == 2u) {
            let base = slot * 4u;
            if (same_id(base, id)) {
                return;
            }
        }

        probe += 1u;
        slot = (slot + 1u) & params.table_mask;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let x = index % params.width;
    let y = index / params.width;
    let id = textureLoad(ids, vec2<i32>(
        i32(params.min_x + x),
        i32(params.min_y + y),
    ), 0);
    if (all(id == vec4<u32>(0u))) {
        return;
    }
    insert_id(id);
}
