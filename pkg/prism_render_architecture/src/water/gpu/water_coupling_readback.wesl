// Water two-way coupling read-back compute kernel
// (entry point `water_coupling_readback`).
//
// Two-way fluid/rigid coupling needs the fluid's reaction on every floating
// body: the `Archimedes` buoyancy that lifts it, the quadratic form drag that
// resists its motion, the added-mass inertia of the water it must drag along,
// and the fraction of its momentum that is stamped back into the fluid. Rather
// than reading the whole field back per frame, the renderer batches the per
// body force evaluation into one bounded read-back pass: one invocation per
// body turns its sampled state into the four coupling scalars.
//
// This pass runs the per-body force math from the `CPU` golden `coupling`
// module (`buoyancy_force`, `drag_force`, `added_mass`,
// `source_writeback_fraction`) byte-for-byte. The frame-level sub-step / batch
// schedule stays on the `CPU` (`coupling::plan_coupling`); it is a single
// per-frame scalar, not a per-body quantity, so it is not part of this kernel.
//
// ABI (matches `WaterKernel::CouplingReadback.descriptor()`: 2 storage buffers,
// 1 uniform, 0 storage textures, 0 sampled textures, 64x1x1 brick, Particle
// domain):
//   @binding(0) body_in   : array<vec4<f32>>, per-body sampled state
//                           (submerged_volume, total_volume, cross_section,
//                           rel_speed)
//   @binding(1) force_out : array<vec4<f32>>, per-body coupling forces
//                           (buoyancy, drag, added_mass, writeback_fraction)
//                           (written)
//   @binding(2) params    : CouplingReadbackParams uniform
//
// Only `+ - * /`, comparisons, `clamp` and `max` appear — no float intrinsic
// the workspace determinism policy forbids. `EPS` matches the crate's
// `water::EPS` (1e-6).

const EPS: f32 = 1e-6;

struct CouplingReadbackParams {
    fluid_density: f32,
    drag_coeff: f32,
    added_mass_coeff: f32,
    gravity: f32,
    body_count: u32,
}

@group(0) @binding(0) var<storage, read> body_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> force_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: CouplingReadbackParams;

// `Archimedes` buoyancy magnitude fluid_density * submerged_volume * gravity.
fn buoyancy_force(fluid_density: f32, submerged_volume: f32, gravity: f32) -> f32 {
    return max(fluid_density, 0.0) * max(submerged_volume, 0.0) * max(gravity, 0.0);
}

// Quadratic form drag 0.5 * drag_coeff * fluid_density * area * rel_speed^2.
fn drag_force(drag_coeff: f32, fluid_density: f32, area: f32, rel_speed: f32) -> f32 {
    let v = max(rel_speed, 0.0);
    return 0.5 * max(drag_coeff, 0.0) * max(fluid_density, 0.0) * max(area, 0.0) * v * v;
}

// Added-mass reaction added_mass_coeff * fluid_density * displaced_volume.
fn added_mass(added_mass_coeff: f32, fluid_density: f32, displaced_volume: f32) -> f32 {
    return max(added_mass_coeff, 0.0) * max(fluid_density, 0.0) * max(displaced_volume, 0.0);
}

// Momentum write-back fraction submerged_volume / total_volume clamped to 0..=1.
fn source_writeback_fraction(submerged_volume: f32, total_volume: f32) -> f32 {
    let total = max(total_volume, 0.0);
    if (total <= EPS) {
        return 0.0;
    }
    return clamp(max(submerged_volume, 0.0) / total, 0.0, 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn water_coupling_readback(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.body_count) {
        return;
    }
    let i = gid.x;
    let state = body_in[i];
    let submerged_volume = state.x;
    let total_volume = state.y;
    let cross_section = state.z;
    let rel_speed = state.w;

    // Added mass uses the submerged (displaced) volume, matching the golden
    // per-frame assembly `coupling_frame::plan_coupling_frame`.
    let buoyancy = buoyancy_force(params.fluid_density, submerged_volume, params.gravity);
    let drag = drag_force(params.drag_coeff, params.fluid_density, cross_section, rel_speed);
    let am = added_mass(params.added_mass_coeff, params.fluid_density, submerged_volume);
    let frac = source_writeback_fraction(submerged_volume, total_volume);

    force_out[i] = vec4<f32>(buoyancy, drag, am, frac);
}
