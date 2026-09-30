// Should-fallback twin: evaluates on-device the temporal history invalidation
// rule, mirroring the CPU golden `should_fallback` (in
// `prism_render_architecture::volumetric::temporal`).
//
// The temporal reprojection (design section 10) discards a reprojected history
// sample - falling back to the freshly raymarched current frame - when the
// history was flagged invalid (for example a reprojection that lands off
// screen) or when the parallax disocclusion measure exceeds a threshold:
//   result = (!history_valid) || (disocclusion > max_disocclusion)
// This deterministic invalidation rule stops disoccluded or high-parallax
// regions from dragging stale cloud colour. The CPU golden owns that logic;
// this kernel is the on-device twin that runs one thread per query.
//
// `history_valid` is uploaded as a u32 flag (0 = invalid, non-zero = valid) and
// the result is written as a u32 (0 = keep history, 1 = fall back). It uses
// only the portable core-WGSL subset - a comparison and boolean or - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard TAA disocclusion / validity fallback rule; no Unreal
// Engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One fallback query: the history validity flag plus the parallax disocclusion
// measure and its threshold. Matches `GpuQuery` in `src/should_fallback.rs`.
struct Query {
    history_valid: u32,
    disocclusion: f32,
    max_disocclusion: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(64)
fn should_fallback_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let history_valid = q.history_valid != 0u;
    let fallback = (!history_valid) || (q.disocclusion > q.max_disocclusion);
    results[idx] = select(0u, 1u, fallback);
}
