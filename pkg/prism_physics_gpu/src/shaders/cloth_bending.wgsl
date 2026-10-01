// GPU colour-batched cloth bending constraints.
//
// One invocation per constraint in the current colour class projects the
// compliant point-to-midpoint bending constraint, the faithful twin of
// `prism_physics_core`'s `project_bending`. Because every constraint in a
// colour class touches a disjoint set of particles, the three position writes
// never race, so the whole class runs as one compute pass; the host dispatches
// the classes in ascending order, one pass each, which is the colour-ordered
// Gauss-Seidel sweep.
//
// Provenance: the compliant XPBD bending projection is the published Müller et
// al. position-based-dynamics technique. No Unreal Engine source or derived
// code.

struct Params {
    // First constraint index (into the reordered arrays) of this colour class.
    start: u32,
    // Number of constraints in this colour class.
    count: u32,
    // Number of addressable particles (min of positions / inverse-mass lengths).
    particle_count: u32,
    // Substep time (seconds).
    dt: f32,
};

struct Bending {
    a: u32,
    center: u32,
    b: u32,
    rest_offset: f32,
    compliance: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<Bending>;
@group(0) @binding(4) var<storage, read_write> lambdas: array<f32>;

// Matches `f32::EPSILON`, the coincidence guard in the CPU golden.
const EPSILON: f32 = 1.1920929e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    if (lane >= params.count) {
        return;
    }
    let k = params.start + lane;
    let c = constraints[k];
    let ia = c.a;
    let ic = c.center;
    let ib = c.b;
    if (ia >= params.particle_count || ic >= params.particle_count || ib >= params.particle_count) {
        return;
    }

    let wa = inverse_masses[ia];
    let wc = inverse_masses[ic];
    let wb = inverse_masses[ib];
    // Gradient magnitudes: |grad center| = 1, |grad a| = |grad b| = 1/2.
    let denom_mass = wc + 0.25 * (wa + wb);
    if (denom_mass <= 0.0) {
        return;
    }

    let pa = positions[ia].xyz;
    let pc = positions[ic].xyz;
    let pb = positions[ib].xyz;
    let midpoint = (pa + pb) * 0.5;
    let delta = pc - midpoint;
    let len = length(delta);
    if (len < EPSILON) {
        return;
    }

    let normal = delta / len;
    let err = len - c.rest_offset;
    let alpha_tilde = c.compliance / (params.dt * params.dt);
    let lambda = lambdas[k];
    let delta_lambda = (-err - alpha_tilde * lambda) / (denom_mass + alpha_tilde);
    lambdas[k] = lambda + delta_lambda;

    positions[ic] = vec4<f32>(pc + normal * (delta_lambda * wc), positions[ic].w);
    positions[ia] = vec4<f32>(pa - normal * (delta_lambda * wa * 0.5), positions[ia].w);
    positions[ib] = vec4<f32>(pb - normal * (delta_lambda * wb * 0.5), positions[ib].w);
}
