// Bounded particle packets consume real water spawn records. No water shader ABI changes.
struct Packet { position: vec3<f32>, count: u32, velocity: vec3<f32>, age: f32 }
struct Packets { count: u32, dropped: u32, pad0: u32, pad1: u32, items: array<Packet> }
struct Spawn { counter: u32, pad0: u32, pad1: u32, pad2: u32, items: array<Packet> }
struct Params {
    dims: vec4<u32>, // nx, nz, admitted spawn slots, packet capacity
    domain: vec4<f32>, // world origin xyz, cell size
    tuning: vec4<f32>, // dt, packet lifetime, foam gain per particle, unused
}
@group(0) @binding(0) var<storage, read> spawn: Spawn;
@group(0) @binding(1) var<storage, read> previous_particles: Packets;
@group(0) @binding(2) var<storage, read_write> particles: Packets;
@group(0) @binding(3) var<storage, read> previous_feedback: array<f32>;
@group(0) @binding(4) var<storage, read_write> feedback: array<f32>;
@group(0) @binding(5) var<storage, read_write> foam_sources: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn apply_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < params.dims.x * params.dims.y {
        foam_sources[i] = clamp(foam_sources[i] + previous_feedback[i], 0.0, 1.0);
    }
}

fn advance(input: Packet) -> Packet {
    var p = input;
    p.velocity.y -= 9.81 * params.tuning.x;
    p.position += p.velocity * params.tuning.x;
    p.age += params.tuning.x;
    return p;
}

// One bounded compaction thread gives deterministic admission without float atomics.
@compute @workgroup_size(1)
fn simulate() {
    let capacity = min(params.dims.w, arrayLength(&particles.items));
    var n = 0u;
    var dropped = 0u;
    for (var i = 0u; i < min(previous_particles.count, capacity); i += 1u) {
        let p = previous_particles.items[i];
        // Impact packets were published exactly once by the preceding gather.
        if p.position.y <= params.domain.y || p.age >= params.tuning.y { continue; }
        particles.items[n] = advance(p);
        n += 1u;
    }
    for (var i = 0u; i < min(params.dims.z, arrayLength(&spawn.items)); i += 1u) {
        var p = spawn.items[i];
        if p.count == 0u { continue; }
        if n >= capacity { dropped += 1u; continue; }
        p.age = 0.0; // The emitter's last word is padding, not particle age.
        particles.items[n] = advance(p);
        n += 1u;
    }
    particles.count = n;
    particles.dropped = dropped;
}

@compute @workgroup_size(64)
fn publish_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.dims.x * params.dims.y { return; }
    let cell = vec2<u32>(i % params.dims.x, i / params.dims.x);
    var source = 0.0;
    for (var j = 0u; j < min(particles.count, params.dims.w); j += 1u) {
        let p = particles.items[j];
        let local = (p.position.xz - params.domain.xz) / params.domain.w;
        if p.position.y <= params.domain.y && all(local >= vec2<f32>(0.0)) &&
           all(local < vec2<f32>(f32(params.dims.x), f32(params.dims.y))) &&
           all(vec2<u32>(floor(local)) == cell) {
            source += f32(p.count) * params.tuning.z;
        }
    }
    feedback[i] = clamp(source, 0.0, 1.0);
}
