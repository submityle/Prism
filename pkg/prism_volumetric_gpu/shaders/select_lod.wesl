// Cloud-LOD twin: classifies a view distance into a discrete level-of-detail
// bucket on-device, mirroring the CPU golden `select_lod` (in
// `prism_render_architecture::volumetric::cloud_lod`).
//
// Volumetric clouds reduce raymarch cost with distance: near clouds get the
// full step count and buffer resolution, distant clouds get coarser marching,
// and horizon clouds fall back to camera-facing imposters (design section 11).
// This kernel maps a per-query view `distance` against three ascending
// thresholds to one of four buckets, deciding, deterministically, how much
// work each cloud is scheduled:
//   distance < mid_beyond       -> Near     (ordinal 0)
//   distance < far_beyond       -> Mid      (ordinal 1)
//   distance < imposter_beyond  -> Far      (ordinal 2)
//   otherwise                   -> Imposter (ordinal 3)
// The boundaries are tested in ascending order, so even a collapsed or
// inverted threshold window terminates deterministically (it just skips the
// empty bucket) and no input - including negatives or infinity - can trap. The
// CPU golden owns that logic; this kernel is the on-device twin that runs one
// thread per query and reproduces the same bucket.
//
// The result is a u32 ordinal (0 = Near, 1 = Mid, 2 = Far, 3 = Imposter),
// matching `CloudLod::rank`. It uses only the portable core-WGSL subset -
// float comparisons - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard distance-bucketed cloud LOD selection; no Unreal Engine
// source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One LOD query: the view distance plus the three ascending distance
// thresholds. Matches `GpuQuery` in `src/select_lod.rs`.
struct Query {
    distance: f32,
    mid_beyond: f32,
    far_beyond: f32,
    imposter_beyond: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// The classification, mirroring `select_lod`. Returns the bucket ordinal.
fn select_lod(distance: f32, mid_beyond: f32, far_beyond: f32, imposter_beyond: f32) -> u32 {
    if (distance < mid_beyond) {
        return 0u;
    }
    if (distance < far_beyond) {
        return 1u;
    }
    if (distance < imposter_beyond) {
        return 2u;
    }
    return 3u;
}

@compute @workgroup_size(64)
fn select_lod_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = select_lod(q.distance, q.mid_beyond, q.far_beyond, q.imposter_beyond);
}
