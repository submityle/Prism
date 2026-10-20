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
// A slot contains a source pixel index + 1, never a publication lock. The
// source texture is immutable throughout this dispatch, so another invocation
// can compare all four ID words without waiting for the winner to write them.
@group(0) @binding(2) var<storage, read_write> representatives: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> values: array<vec4<u32>>;
// stats[0] is the number of inserted IDs; stats[1] is the overflow flag.
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

fn hash_id(id: vec4<u32>) -> u32 {
    var hash = id.x * 0x9E3779B1u + id.y;
    hash = (hash ^ (hash >> 16u)) * 0x85EBCA6Bu;
    hash = (hash ^ id.z) * 0xC2B2AE35u;
    hash = (hash ^ id.w) * 0x27D4EB2Fu;
    return hash ^ (hash >> 15u);
}

fn source_id(index: u32) -> vec4<u32> {
    return textureLoad(ids, vec2<i32>(
        i32(params.min_x + index % params.width),
        i32(params.min_y + index / params.width),
    ), 0);
}

fn insert_id(id: vec4<u32>, index: u32) {
    var slot = hash_id(id) & params.table_mask;
    var probe = 0u;
    loop {
        if (atomicLoad(&stats[1]) != 0u) {
            return;
        }
        if (probe > params.table_mask) {
            atomicStore(&stats[1], 1u);
            return;
        }

        let acquired = atomicCompareExchangeWeak(&representatives[slot], 0u, index + 1u);
        if (acquired.exchanged) {
            // Only the successful claimant writes this slot. Other
            // invocations read the source texture, never this output.
            values[slot] = id;
            let previous = atomicAdd(&stats[0], 1u);
            if (previous >= params.max_ids) {
                atomicStore(&stats[1], 1u);
            }
            return;
        }
        if (acquired.old_value == 0u) {
            // A weak compare-exchange may fail spuriously. Retry this slot;
            // progress never depends on a different invocation being scheduled.
            continue;
        }
        if (all(source_id(acquired.old_value - 1u) == id)) {
            return;
        }

        probe += 1u;
        slot = (slot + 1u) & params.table_mask;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.y * groups.x * 64u + gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let id = source_id(index);
    if (all(id == vec4<u32>(0u))) {
        return;
    }
    insert_id(id, index);
}
