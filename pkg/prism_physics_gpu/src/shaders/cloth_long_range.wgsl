// GPU colour-batched cloth long-range-attachment (LRA) leashes.
//
// One invocation per leash in the current colour class projects the one-sided
// compliant distance constraint tying a particle to a fixed anchor, the
// faithful twin of `prism_physics_core`'s `project_long_range`. Because every
// leash in a colour class touches a disjoint particle, the single position
// write never races, so the whole class runs as one compute pass; the host
// dispatches the classes in ascending order, one pass each, which is the
// colour-ordered Gauss-Seidel sweep.
//
// Provenance: the one-sided long-range-attachment leash is a published
// position-based dynamics technique (Kim et al., "Long Range Attachments"). No
// Unreal Engine source or derived code.

struct Params {
    // First leash index (into the reordered array) of this colour class.
    start: u32,
    // Number of leashes in this colour class.
    count: u32,
    // Number of addressable particles (min of positions / inverse-mass lengths).
    particle_count: u32,
    // Substep time (seconds).
    dt: f32,
};

struct Leash {
    anchor: vec3<f32>,
    max_distance: f32,
    compliance: f32,
    particle: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<Leash>;
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
    let i = c.particle;
    if (i >= params.particle_count) {
        return;
    }

    let w = inverse_masses[i];
    if (w <= 0.0) {
        return;
    }

    let pos = positions[i].xyz;
    let delta = pos - c.anchor;
    let len = length(delta);
    if (len < EPSILON) {
        return;
    }
    // One-sided: slack inside the leash sphere does nothing.
    let err = len - c.max_distance;
    if (err <= 0.0) {
        return;
    }

    let normal = delta / len;
    let alpha_tilde = c.compliance / (params.dt * params.dt);
    let lambda = lambdas[k];
    let delta_lambda = (-err - alpha_tilde * lambda) / (w + alpha_tilde);
    lambdas[k] = lambda + delta_lambda;

    positions[i] = vec4<f32>(pos + normal * (delta_lambda * w), positions[i].w);
}
