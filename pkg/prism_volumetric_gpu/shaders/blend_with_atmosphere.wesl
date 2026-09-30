// Aerial-perspective blend twin: composites on-device a cloud colour with
// sampled atmosphere in-scatter, mirroring the CPU golden
// `blend_with_atmosphere` (in `prism_render_architecture::volumetric::atmosphere`).
//
// The atmosphere hookup (design section 8) fades a cloud toward the airlight it
// sits in front of and the airlight the view path accumulates. `cloud_color` is
// the cloud's out-scattered radiance, `cloud_transmittance` (0..=1) is how much
// distant sky shows through the cloud, `inscatter` is the sampled airlight, and
// `weight` (0..=1) is the distance fade. The result is two nested convex
// combinations:
//   composited = lerp(cloud_color, inscatter, saturate(cloud_transmittance))
//   result     = lerp(composited,  inscatter, saturate(weight))
// Because both steps are convex combinations, every output channel stays within
// the interval spanned by `cloud_color` and `inscatter`, so the blend is
// energy-conserving and never over-exposes beyond its brightest source. The CPU
// golden owns that math; this kernel is the on-device twin that runs one thread
// per query and reproduces the value.
//
// The `lerp` is expanded to the SAME closed form the CPU `math` module uses and
// `saturate` clamps both weights, so CPU and GPU evaluate identical algebra and
// the parity test can assert a tight tolerance. Results are `vec4` (xyz = rgb,
// w unused) so the storage buffer stays 16-byte aligned.
//
// It uses only the portable core-WGSL subset - clamp/saturate and multiply/add
// - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard aerial-perspective composite; no Unreal Engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: cloud colour packed with its transmittance, and sampled in-scatter
// packed with the distance-fade weight. Matches `GpuQuery` in
// `src/blend_with_atmosphere.rs`.
struct Query {
    // xyz = cloud_color, w = cloud_transmittance.
    cloud: vec4<f32>,
    // xyz = inscatter, w = weight.
    air: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` bit for bit.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return vec3<f32>(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    );
}

@compute @workgroup_size(64)
fn blend_with_atmosphere_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let cloud_color = q.cloud.xyz;
    let inscatter = q.air.xyz;
    let bg = saturate(q.cloud.w);
    let w = saturate(q.air.w);
    let composited = lerp3(cloud_color, inscatter, bg);
    let blended = lerp3(composited, inscatter, w);
    results[idx] = vec4<f32>(blended, 0.0);
}
