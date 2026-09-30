// Per-groom ray-traced-reflection role twin: evaluates on-device the same
// two-gate policy decision the CPU golden
// `prism_render_architecture::hair::rt_proxy::resolve_rt_role` makes, one thread
// per groom instance.
//
// A scene may carry many groom instances, and every frame each one must be
// classified into how it participates in the ray-traced reflection pass before
// the RT backend builds its BVH: trace the real strands (hero-shot luxury),
// register a cheap LOD proxy (card/mesh shell), or skip the groom in reflections
// entirely (design §7 "RT 侧", §8 降级矩阵 "RT 反射"). That decision is a pure,
// per-instance function of the groom's LOD tier, its screen coverage and a
// shared quality policy, with no cross-instance dependency, so it maps to one
// GPU thread per groom instance.
//
// The decision is a two-gate map, mirroring the reference body exactly:
//   1. Visibility gate: a coverage below `min_coverage_for_proxy` (or a NaN
//      coverage, treated as invisible/unknown) is Excluded - too small to
//      justify any reflection cost.
//   2. Representation gate: a groom that passes is a Proxy by default; a
//      strand-based tier (Strands / ReducedStrands) is promoted to FullStrands
//      only when the policy opts strands in.
//
// Encodings shared with the host: tier 0=Strands, 1=ReducedStrands, 2=Cards,
// 3=Mesh (`is_strand_based` is `tier < 2u`); role 0=FullStrands, 1=Proxy,
// 2=Excluded; `allow_strands` is 0/1.
//
// It uses only integer arithmetic and one float compare in the portable
// core-WGSL subset - no exp, pow, sin or optional device feature - so the twin
// runs unmodified on Metal, Vulkan and DX12. There is no transcendental and no
// fused multiply-add anywhere on the path (the only float operation is the
// gate comparison), so the CPU and GPU produce the identical role code for
// every finite-coverage groom; the parity test asserts exact equality rather
// than a tolerance.
//
// Provenance: standard LOD-driven RT reflection proxy/exclusion policy plus
// wgpu compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of groom instances, bounding the dispatch.
    query_count: u32,
    // 1 when strand-based tiers are traced as real strands, 0 otherwise.
    allow_strands: u32,
    // Screen coverage at or above which a groom is worth registering in the
    // reflection BVH; below it the groom is excluded.
    min_coverage_for_proxy: f32,
    pad: u32,
}

// One groom instance: its LOD tier code and screen coverage.
struct Query {
    tier: u32,
    coverage: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// Per-groom output: the resolved RT reflection role code.
@group(0) @binding(2) var<storage, read_write> roles: array<u32>;

const ROLE_FULL_STRANDS: u32 = 0u;
const ROLE_PROXY: u32 = 1u;
const ROLE_EXCLUDED: u32 = 2u;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }

    let q = queries[idx];

    // Visibility gate: too small / off-screen / NaN never enters the BVH.
    // `coverage != coverage` is the NaN test; for the finite coverages the
    // parity test exercises it is always false, matching `f32::is_nan`.
    if (q.coverage < params.min_coverage_for_proxy || q.coverage != q.coverage) {
        roles[idx] = ROLE_EXCLUDED;
        return;
    }

    // Representation gate: strand-based tiers (tier < 2) become full strands
    // only when the policy opts in; everything else is a proxy.
    if (q.tier < 2u && params.allow_strands != 0u) {
        roles[idx] = ROLE_FULL_STRANDS;
    } else {
        roles[idx] = ROLE_PROXY;
    }
}
