// Contrail-spread twin: evaluates on-device the cross-section spread (half-width
// proxy) of a contrail at a given age, mirroring the CPU golden
// `contrail_spread` (in `prism_render_architecture::volumetric::fog`).
//
// The unified volumetric fog model (design section 9f) widens an aircraft
// contrail over time as diffusion and wind shear smear it: the spread grows
// linearly from CONTRAIL_BASE_SIGMA at a rate of CONTRAIL_DIFFUSION_RATE per
// second of age,
//   spread = CONTRAIL_BASE_SIGMA + CONTRAIL_DIFFUSION_RATE * max(age, 0)
// Negative ages are floored at zero, so a fresh (or not-yet-born) contrail
// starts at the base sigma and the spread rises monotonically with age. The CPU
// golden `contrail_spread` owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The kernel is a single max plus a multiply-add - no transcendental call - and
// uses the SAME constants as the CPU reference, so CPU and GPU evaluate the same
// algebra and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - max and multiply/add - so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard linear diffusion spread; no Unreal Engine source or
// derived code.

// Base cross-section spread (sigma) of a fresh contrail, matching
// `CONTRAIL_BASE_SIGMA` in the CPU reference.
const CONTRAIL_BASE_SIGMA: f32 = 1.0;
// Spread growth per second of age, matching `CONTRAIL_DIFFUSION_RATE` in the CPU
// reference.
const CONTRAIL_DIFFUSION_RATE: f32 = 0.5;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One contrail-spread query: the contrail's age in seconds. Matches `GpuQuery`
// in `src/contrail_spread.rs`.
struct Query {
    age: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// The contrail cross-section spread, mirroring `contrail_spread`.
fn contrail_spread(age: f32) -> f32 {
    return CONTRAIL_BASE_SIGMA + CONTRAIL_DIFFUSION_RATE * max(age, 0.0);
}

@compute @workgroup_size(64)
fn contrail_spread_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = contrail_spread(q.age);
}
