// Cloth plasticity: permanent rest-length creep for over-stretched edges.
//
// Real-device twin of `prism_physics_core::plastic_rest_length`, run over every
// distance edge in parallel. Each thread owns one edge, samples its endpoint
// separation from the read-only position snapshot, and writes the (possibly
// crept) rest length into `out_rest[i]`. A single atomic counter tallies how
// many edges crept, matching the sequential golden's `modified` return.
//
// Plasticity never writes a particle position, so edges share no mutable state
// and the pass needs neither colouring nor an inter-thread barrier.
//
// Provenance: rest-length creep past a yield strain is a standard, publicly
// documented plastic-set model for position-based cloth. No Unreal Engine
// source or derived code.

// Numerical floor below which a rest length is degenerate and the edge is
// skipped (mirrors `EPS_REST` in `prism_physics_core`).
const EPS_REST: f32 = 1e-9;

// Uniform parameters; mirrors the host `Params` struct exactly (32 bytes).
// `yield_strain`, `creep`, and `max_strain` arrive already sanitized on the host.
struct Params {
    particle_count: u32,
    edge_count: u32,
    yield_strain: f32,
    creep: f32,
    max_strain: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// One plastic edge; mirrors the host `ClothPlasticEdge` (12 bytes, stride 12).
struct Edge {
    a: u32,
    b: u32,
    rest_length: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<storage, read_write> out_rest: array<f32>;
@group(0) @binding(4) var<storage, read_write> modified: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.edge_count) {
        return;
    }

    let e = edges[i];
    let rest = e.rest_length;
    // Default: the edge keeps its input rest length unless it creeps below.
    out_rest[i] = rest;

    // Out-of-range endpoints make the edge inert.
    if (e.a >= params.particle_count || e.b >= params.particle_count) {
        return;
    }
    // Degenerate rest length is skipped to avoid a divide-by-zero in the strain.
    if (rest <= EPS_REST) {
        return;
    }

    let pa = positions[e.a].xyz;
    let pb = positions[e.b].xyz;
    let length = distance(pa, pb);

    let strain = (length - rest) / rest;
    if (abs(strain) <= params.yield_strain) {
        return;
    }

    let sign = select(-1.0, 1.0, strain >= 0.0);
    let excess = strain - sign * params.yield_strain;
    // Move the rest length by `creep` fraction of the excess strain.
    var new_rest = rest * (1.0 + params.creep * excess);
    if (new_rest <= EPS_REST) {
        new_rest = EPS_REST;
    }
    // Clamp so the residual elastic strain magnitude stays within max.
    let residual = (length - new_rest) / new_rest;
    if (abs(residual) > params.max_strain) {
        let residual_sign = select(-1.0, 1.0, residual >= 0.0);
        new_rest = length / (1.0 + residual_sign * params.max_strain);
    }
    if (new_rest > EPS_REST) {
        out_rest[i] = new_rest;
        atomicAdd(&modified, 1u);
    }
}
