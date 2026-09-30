// Terrain-occlusion twin: evaluates on-device the cloud-versus-terrain
// occlusion factor, mirroring the CPU golden `terrain_occlusion` (in
// `prism_render_architecture::volumetric::coupling`).
//
// Two-way scene coupling (design section 9d) lets mountains and buildings
// punch into the cloud layer: a cloud sample at or below the terrain surface is
// fully occluded (`1`), a sample a full `TERRAIN_OCCLUSION_BAND` above the
// surface is fully clear (`0`), and the transition between them is a monotone
// non-increasing `smoothstep`. The CPU golden `terrain_occlusion` owns that
// curve; this kernel is the on-device twin that runs one thread per query and
// reproduces the same value.
//
// The smoothstep mirrors the reference's hand-rolled `math::smoothstep`
// bit-for-bit - the same collapsed-edge EPS guard and the same Hermite
// `t*t*(3-2t)` polynomial - rather than the device-native `smoothstep`, so the
// twin stays bit-close to the reference and the parity test can assert a tight
// tolerance. It contains no transcendental call at all.
//
// It uses only the portable core-WGSL subset - clamp and multiply/add - so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Hermite smoothstep terrain intersection; no Unreal
// Engine source or derived code.

const EPS: f32 = 1e-6;

// Half-height of the soft transition band above the terrain surface; matches
// `TERRAIN_OCCLUSION_BAND` in `src/terrain_occlusion.rs` and the CPU golden.
const TERRAIN_OCCLUSION_BAND: f32 = 50.0;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One terrain-occlusion query: the cloud sample altitude and the terrain
// surface height beneath it. Matches `GpuQuery` in `src/terrain_occlusion.rs`.
struct Query {
    cloud_pos_y: f32,
    terrain_height: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep between `edge0` and `edge1`, clamped to `[0, 1]`,
// mirroring `math::smoothstep`: collapsed edges degrade to a hard step at
// `edge0` instead of dividing by zero.
fn smoothstep_c(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate_s((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Terrain occlusion factor for a cloud sample at altitude `cloud_pos_y`,
// mirroring `terrain_occlusion`.
fn terrain_occlusion(cloud_pos_y: f32, terrain_height: f32) -> f32 {
    let clear = smoothstep_c(terrain_height, terrain_height + TERRAIN_OCCLUSION_BAND, cloud_pos_y);
    return saturate_s(1.0 - clear);
}

@compute @workgroup_size(64)
fn terrain_occlusion_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = terrain_occlusion(q.cloud_pos_y, q.terrain_height);
}
