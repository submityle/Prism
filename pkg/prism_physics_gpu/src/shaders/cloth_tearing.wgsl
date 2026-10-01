// Cloth tearing: per-edge break-flag evaluation for over-stretched edges.
//
// Real-device twin of `prism_physics_core::tear_flag`, run over every distance
// edge in parallel. Each thread owns one edge, samples its endpoint separation
// from the read-only position snapshot, and writes a break flag into
// `out_flags[i]` (`1u` = the edge tears, `0u` = it survives). A single atomic
// counter tallies how many edges tear, matching the sequential golden's torn
// count.
//
// Removing the torn edges (constraint-graph compaction) is a host-side step;
// this kernel only computes the flags, so edges share no mutable state and the
// pass needs neither colouring nor an inter-thread barrier.
//
// Provenance: removing a constraint whose strain exceeds a threshold is a
// standard, publicly documented position-based-dynamics technique. No Unreal
// Engine source or derived code.

// Numerical floor below which a rest length is degenerate and the edge is
// skipped (mirrors `EPS_REST` in `prism_physics_core`).
const EPS_REST: f32 = 1e-9;

// Uniform parameters; mirrors the host `Params` struct exactly (16 bytes).
// `break_strain` arrives already sanitized on the host (a `NaN` or negative
// threshold maps to +infinity, so nothing tears).
struct Params {
    particle_count: u32,
    edge_count: u32,
    break_strain: f32,
    _pad0: f32,
};

// One tearable edge; mirrors the host `ClothTearEdge` (12 bytes, stride 12).
struct Edge {
    a: u32,
    b: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<storage, read_write> out_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> torn: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.edge_count) {
        return;
    }

    let e = edges[i];
    // Default: the edge survives unless the break predicate fires below.
    out_flags[i] = 0u;

    // Out-of-range endpoints make the edge inert.
    if (e.a >= params.particle_count || e.b >= params.particle_count) {
        return;
    }
    // Degenerate rest length is skipped to avoid a divide-by-zero in the strain.
    if (e.rest_length <= EPS_REST) {
        return;
    }

    let pa = positions[e.a].xyz;
    let pb = positions[e.b].xyz;
    let length = distance(pa, pb);

    let strain = (length - e.rest_length) / e.rest_length;
    if (strain > params.break_strain) {
        out_flags[i] = 1u;
        atomicAdd(&torn, 1u);
    }
}
