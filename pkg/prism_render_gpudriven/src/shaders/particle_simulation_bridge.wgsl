struct Particle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
    radius: f32,
    alive: u32,
    particle_id: u32,
    pad: u32,
};

struct EventRecord {
    payload: vec4<f32>,
    kind: u32,
    particle_index: u32,
    pad0: u32,
    pad1: u32,
};

struct ParticleRenderInput {
    center_radius: vec4<f32>,
    depth: f32,
    particle_index: u32,
    flags: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> active_indices: array<u32>;
@group(0) @binding(2) var<storage, read> compact_events: array<EventRecord>;
@group(0) @binding(3) var<storage, read> control: array<u32>;
@group(0) @binding(4) var<storage, read_write> event_flags: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> renderer_inputs: array<ParticleRenderInput>;
@group(0) @binding(6) var<storage, read_write> validation: array<atomic<u32>>;

const INVALID_LIFETIME: u32 = 1u;
const EVENT_PRESENT: u32 = 2u;

fn finite1(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn lifetime_valid(particle: Particle) -> bool {
    let age = particle.position_age.w;
    let lifetime = particle.velocity_lifetime.w;
    return particle.alive != 0u
        && finite1(age)
        && age >= 0.0
        && finite1(lifetime)
        && lifetime >= 0.0
        && (lifetime == 0.0 || age < lifetime);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&event_flags)) {
        atomicStore(&event_flags[index], 0u);
        renderer_inputs[index] = ParticleRenderInput(vec4<f32>(0.0), 0.0, 0u, INVALID_LIFETIME, 0u);
    }
    if (index == 0u) {
        atomicStore(&validation[0], 0u);
    }
}

@compute @workgroup_size(64)
fn mark_events(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let event_count = min(control[8], arrayLength(&compact_events));
    if (index >= event_count) {
        return;
    }
    let event = compact_events[index];
    if (event.kind < 4u && event.particle_index < arrayLength(&event_flags)) {
        atomicOr(&event_flags[event.particle_index], EVENT_PRESENT);
    }
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    let active_count = min(control[2], arrayLength(&active_indices));
    if (output_index >= active_count) {
        return;
    }
    let pool_index = active_indices[output_index];
    if (pool_index >= arrayLength(&particles)) {
        atomicAdd(&validation[0], 1u);
        return;
    }
    let particle = particles[pool_index];
    let valid = lifetime_valid(particle);
    var flags = atomicLoad(&event_flags[pool_index]);
    if (!valid) {
        atomicAdd(&validation[0], 1u);
        flags = flags | INVALID_LIFETIME;
    }
    let position = select(vec3<f32>(0.0), particle.position_age.xyz, valid);
    let radius = select(0.0, max(particle.radius, 0.0), valid);
    let depth = select(0.0, position.z, valid);
    renderer_inputs[output_index] = ParticleRenderInput(
        vec4<f32>(position, radius),
        depth,
        pool_index,
        flags,
        0u,
    );
}
