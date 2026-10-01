// GPU colour-batched cloth strain limiting (biphasic length clamp).
//
// One invocation per edge in the current colour class applies the biphasic
// length clamp, the faithful twin of `prism_physics_core`'s
// `project_strain_limit`. Because every edge in a colour class touches a
// disjoint pair of particles, the two position writes never race, so the whole
// class runs as one compute pass; the host dispatches the classes in ascending
// order, one pass each, which is the colour-ordered Gauss-Seidel sweep.
//
// The clamp is a pure geometric projection with no Lagrange multiplier, so
// unlike the compliant kernels there is no persistent lambda buffer.
//
// Provenance: biphasic strain limiting is a standard, publicly documented cloth
// technique (Provot 1995; Thomaszewski et al. 2009). No Unreal Engine source or
// derived code.

struct Params {
    // First edge index (into the reordered array) of this colour class.
    start: u32,
    // Number of edges in this colour class.
    count: u32,
    // Number of addressable particles (min of positions / inverse-mass lengths).
    particle_count: u32,
    // Padding to a 16-byte uniform; unused.
    pad: u32,
};

struct Edge {
    a: u32,
    b: u32,
    rest_length: f32,
    max_scale: f32,
    min_scale: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<Edge>;

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
    let ib = c.b;
    if (ia == ib) {
        return;
    }
    if (ia >= params.particle_count || ib >= params.particle_count) {
        return;
    }

    let wa = inverse_masses[ia];
    let wb = inverse_masses[ib];
    let w_sum = wa + wb;
    if (w_sum <= 0.0) {
        return;
    }

    let pa = positions[ia].xyz;
    let pb = positions[ib].xyz;
    let delta = pa - pb;
    let len = length(delta);
    if (len < EPSILON) {
        return;
    }

    let max_len = c.rest_length * c.max_scale;
    let min_len = c.rest_length * c.min_scale;
    // Signed length error outside the allowed band; positive means
    // overstretched, negative means over-compressed. Zero inside the band.
    var error: f32 = 0.0;
    if (len > max_len) {
        error = len - max_len;
    } else if (c.min_scale > 0.0 && len < min_len) {
        error = len - min_len;
    } else {
        return;
    }

    let direction = delta / len;
    let correction = direction * error;
    // Mass-weighted split; moving `a` toward `b` for overstretch (error > 0)
    // and apart for over-compression (error < 0) via the shared sign.
    positions[ia] = vec4<f32>(pa - correction * (wa / w_sum), positions[ia].w);
    positions[ib] = vec4<f32>(pb + correction * (wb / w_sum), positions[ib].w);
}
