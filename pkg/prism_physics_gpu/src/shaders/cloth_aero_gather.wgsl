// GPU cloth aerodynamics — phase 2 (per-vertex force gather).
//
// One invocation per particle gathers the evenly-shared forces of its incident
// triangles (written by `cloth_aero_force.wgsl`) and applies the sum as an
// external velocity increment: `velocity += gathered * (inverse_mass * dt)`.
// The incidence is a host-built `CSR` in ascending triangle order, so each
// vertex folds its faces in the exact reduction order of the sequential
// `prism_physics_core` golden — a vertex writes only its own slot, so the pass
// is race-free. Accumulating the force first and scaling once by
// `inverse_mass * dt` matches the CPU golden's arithmetic order.
//
// Provenance: standard scatter-to-vertex force accumulation for the per-face
// cloth aerodynamics model. No Unreal Engine source or derived code.

struct Params {
    drag: f32,
    lift: f32,
    air_density: f32,
    // Substep time (seconds).
    dt: f32,
    triangle_count: u32,
    // Number of addressable particles.
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(3) var<storage, read> tri_forces: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> vert_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> vert_tris: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vertex_count) {
        return;
    }
    let w = inverse_masses[v];
    if (w <= 0.0) {
        return;
    }

    let lo = vert_offsets[v];
    let hi = vert_offsets[v + 1u];
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = lo; k < hi; k = k + 1u) {
        acc = acc + tri_forces[vert_tris[k]].xyz;
    }

    let current = velocities[v];
    velocities[v] = vec4<f32>(current.xyz + acc * (w * params.dt), current.w);
}
