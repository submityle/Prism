// Spatial-hash broad phase (Teschner et al. 2003), device kernel.
//
// Byte-for-byte twin of the CPU reference in `src/broadphase/cpu.rs`. The cell
// and hash math below mirror `src/broadphase/hash.rs` exactly (bitcast to u32,
// wrapping multiply by the three primes, xor, modulo), so a particle lands in
// the same bucket on both paths and the parity test can compare pair sets.
//
// Provenance: Teschner, Heidelberger, Müller, Pomeranets, Gross, VMV 2003.
// No Unreal Engine source or derived code.

struct Params {
    count: u32,
    table_size: u32,
    max_per_bucket: u32,
    pair_capacity: u32,
    cell_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bucket_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bucket_entries: array<u32>;
@group(0) @binding(4) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> pairs: array<vec2<u32>>;

fn cell_coord(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(position / params.cell_size));
}

fn hash_cell(cell: vec3<i32>) -> u32 {
    let ux = bitcast<u32>(cell.x) * 73856093u;
    let uy = bitcast<u32>(cell.y) * 19349663u;
    let uz = bitcast<u32>(cell.z) * 83492791u;
    return (ux ^ uy ^ uz) % params.table_size;
}

// Stage 1: hash every particle into its bucket.
@compute @workgroup_size(64)
fn populate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let cell = cell_coord(particles[i].xyz);
    let bucket = hash_cell(cell);
    let slot = atomicAdd(&bucket_counts[bucket], 1u);
    if (slot < params.max_per_bucket) {
        bucket_entries[bucket * params.max_per_bucket + slot] = i;
    }
}

// Stage 2: scan each particle's 3x3x3 neighbourhood and emit overlapping pairs.
@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let pi = particles[i].xyz;
    let ri = particles[i].w;
    let base = cell_coord(pi);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbour = vec3<i32>(base.x + dx, base.y + dy, base.z + dz);
                let bucket = hash_cell(neighbour);
                let count = atomicLoad(&bucket_counts[bucket]);
                let capacity = min(count, params.max_per_bucket);
                for (var slot = 0u; slot < capacity; slot = slot + 1u) {
                    let j = bucket_entries[bucket * params.max_per_bucket + slot];
                    if (j <= i) {
                        continue;
                    }
                    let pj = particles[j].xyz;
                    // Accept `j` only under its own cell so a collision-shared
                    // bucket cannot emit the pair twice.
                    let cj = cell_coord(pj);
                    if (cj.x != neighbour.x || cj.y != neighbour.y || cj.z != neighbour.z) {
                        continue;
                    }
                    let rj = particles[j].w;
                    let delta = pi - pj;
                    let radius_sum = ri + rj;
                    if (dot(delta, delta) <= radius_sum * radius_sum) {
                        let index = atomicAdd(&pair_count[0], 1u);
                        if (index < params.pair_capacity) {
                            pairs[index] = vec2<u32>(i, j);
                        }
                    }
                }
            }
        }
    }
}
