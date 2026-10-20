struct ParticleInput {
    center_radius: vec4<f32>,
    depth: f32,
    particle_index: u32,
    flags: u32,
    reserved: u32,
}

struct AuxRecord {
    position_radius: vec4<f32>,
    payload: vec4<f32>,
}

struct Params {
    count: u32,
    mode_bits: u32,
    strength: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<ParticleInput>;
@group(0) @binding(1) var<storage, read_write> records: array<AuxRecord>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn emit_lights(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let p = particles[i];
    let radius = max(p.center_radius.w, 0.001);
    records[i] = AuxRecord(p.center_radius, vec4<f32>(params.strength, radius, f32(p.particle_index), 0.0));
}

@compute @workgroup_size(64)
fn project_decals(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let p = particles[i];
    let extent = max(p.center_radius.w, 0.001);
    records[i] = AuxRecord(p.center_radius, vec4<f32>(extent, extent, params.strength, f32(p.flags)));
}

@compute @workgroup_size(64)
fn accumulate_volumes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let p = particles[i];
    let radius = max(p.center_radius.w, 0.001);
    let density = max(params.strength, 0.0);
    let optical_depth = density * radius * 2.0;
    let opacity = 1.0 - exp(-optical_depth);
    records[i] = AuxRecord(p.center_radius, vec4<f32>(density, optical_depth, opacity, f32(p.particle_index)));
}
