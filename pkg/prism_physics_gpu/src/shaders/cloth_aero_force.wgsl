// GPU cloth aerodynamics — phase 1 (per-triangle face force).
//
// One invocation per retained triangle computes the aerodynamic force on that
// face from the frozen position/velocity snapshot and the host-baked per-face
// wind, then writes the evenly-shared `force / 3` into `tri_forces[t]`. This is
// the faithful twin of `prism_physics_core`'s `triangle_aero_force`: the
// relative wind (face wind minus the mean vertex velocity) is split into a
// drag component along the unit face normal and the remaining in-plane lift
// component, and the sum is scaled by the triangle area (times the optional
// quadratic dynamic pressure). The turbulence jitter is already folded into the
// per-triangle wind on the host, so this kernel does only the floating-point
// drag/lift arithmetic. Phase 2 (`cloth_aero_gather.wgsl`) gathers these
// per-face forces onto the vertices.
//
// Provenance: the per-triangle drag/lift decomposition and the optional
// quadratic dynamic-pressure term are the standard, publicly documented cloth
// aerodynamics model. No Unreal Engine source or derived code.

struct Params {
    // Normal-direction (drag) coefficient, sanitized non-negative on the host.
    drag: f32,
    // In-plane (lift) coefficient, sanitized non-negative on the host.
    lift: f32,
    // Air density: `<= 0` selects the linear model, `> 0` the quadratic one.
    air_density: f32,
    // Substep time (seconds); unused in this pass, read by the gather pass.
    dt: f32,
    // Number of retained (in-range) triangles.
    triangle_count: u32,
    // Number of addressable particles.
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> triangles: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> triangle_winds: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> tri_forces: array<vec4<f32>>;

// Squared length below which a triangle's edge-cross is treated as degenerate.
// Matches `prism_physics_core`'s `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.triangle_count) {
        return;
    }

    let corners = triangles[t];
    let i0 = corners.x;
    let i1 = corners.y;
    let i2 = corners.z;

    let p0 = positions[i0].xyz;
    let p1 = positions[i1].xyz;
    let p2 = positions[i2].xyz;

    let cross_vec = cross(p1 - p0, p2 - p0);
    let cross_len_sq = dot(cross_vec, cross_vec);
    if (cross_len_sq <= EPS_LEN_SQ) {
        tri_forces[t] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    let area = 0.5 * sqrt(cross_len_sq);
    // Mirror glam's `normalize_or_zero`: `cross * (1 / length(cross))`. Keep the
    // explicit reciprocal square root (never `inverseSqrt`) so the device and
    // the CPU golden round identically.
    let normal = cross_vec * (1.0 / sqrt(cross_len_sq));

    let v0 = velocities[i0].xyz;
    let v1 = velocities[i1].xyz;
    let v2 = velocities[i2].xyz;
    let face_velocity = (v0 + v1 + v2) * (1.0 / 3.0);
    let relative = triangle_winds[t].xyz - face_velocity;

    let normal_component = normal * dot(relative, normal);
    let tangent_component = relative - normal_component;
    let directional = normal_component * params.drag + tangent_component * params.lift;

    var pressure = area;
    if (params.air_density > 0.0) {
        pressure = area * (0.5 * params.air_density * sqrt(dot(relative, relative)));
    }

    let force = directional * pressure;
    tri_forces[t] = vec4<f32>(force * (1.0 / 3.0), 0.0);
}
