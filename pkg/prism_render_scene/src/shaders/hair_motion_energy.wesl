//! GPU per-particle motion-energy map for the sleep / activation gate.
//!
//! Compute twin of the per-element body of the CPU golden reference
//! `prism_render_architecture::hair::sleep::groom_motion_energy` — the
//! per-particle `particle_speed_squared` map whose sum is the groom's kinetic
//! proxy that drives the hysteretic sleep gate (design §6.9, §8 "休眠 groom 不占
//! 预算"). A crowd scene has far more grooms than the shared deformation budget
//! can simulate, so a groom at rest is put to sleep and stops consuming budget;
//! this kernel produces the per-particle energy that the reduction sums to
//! decide rest. `UE5` Groom and `HairWorks` gate simulation on exactly this kind
//! of motion proxy so distant or still characters cost nothing.
//!
//! One invocation owns one particle. It reads that particle's current and
//! previous position (the dynamics integrator keeps velocity implicitly as
//! `position - prev_position`), and writes the squared implicit velocity
//! `dot(v, v)` to `out_energy[p]`. Pinned particles contribute `0` because the
//! integrator holds their `prev_position` equal to `position`, exactly as the
//! CPU reduction observes.
//!
//! Host / other-kernel work: the per-groom sum over `out_energy` is a reduction
//! and stays host/other-kernel, the same split the density-LOD twins
//! (`hair_binding_metrics`, `hair_importance`) use for their groom-global
//! reductions. The hysteretic gate itself (`update_sleep`: dead-band, quiet-run
//! counter, `NaN`-keeps-awake) is a tiny per-groom scalar state machine and
//! stays host; this kernel only carries the embarrassingly-parallel per-particle
//! map that dominates the cost for crowd-sized grooms.
//!
//! Bit-faithfulness: the map mirrors the CPU exactly — `v = position -
//! prev_position` per component and `out = v.x*v.x + v.y*v.y + v.z*v.z`, summed
//! left-to-right in the same order as `Vec3::dot` so a `GPU` map feeds a host
//! reduction the same per-particle terms the CPU loop would sum. The particle
//! layout matches the shared `hair_sim` state (`positions.xyz` = position with
//! `.w` = inverse mass, `prev_positions.xyz` = previous position), so no
//! repacking is needed between the simulation and this proxy. A `GPU` is now
//! available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware.

/// Motion-energy dispatch parameters. `count` bounds the dispatch (one
/// invocation per particle across the groom's flat particle array).
struct HairMotionEnergyParams {
    count: u32,
}

// Current particle state shared with `hair_sim`: `xyz` = world-space position,
// `w` = inverse mass (`0` = pinned). Only `xyz` is read here.
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;

// Previous-substep position history for the implicit velocity (`xyz` used),
// shared with `hair_sim`.
@group(0) @binding(1) var<storage, read> prev_positions: array<vec4<f32>>;

// Per-particle squared implicit velocity, summed host-side into the groom's
// motion energy for the sleep gate.
@group(0) @binding(2) var<storage, read_write> out_energy: array<f32>;

var<immediate> params: HairMotionEnergyParams;

@compute @workgroup_size(64)
fn hair_motion_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.count {
        return;
    }

    // Implicit velocity: position - prev_position (pinned => 0).
    let v = positions[p].xyz - prev_positions[p].xyz;

    // Squared length summed left-to-right to match CPU `Vec3::dot`.
    out_energy[p] = v.x * v.x + v.y * v.y + v.z * v.z;
}
