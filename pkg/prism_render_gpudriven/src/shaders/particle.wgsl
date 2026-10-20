struct Params {
    capacity: u32,
    spawn_count: u32,
    dt: f32,
    spawn_life: f32,
    gravity: vec4<f32>,
    spawn_position: vec4<f32>,
    spawn_velocity: vec4<f32>,
}

struct Particle {
    position_life: vec4<f32>,
    velocity_pad: vec4<f32>,
}

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> active_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(1)
fn reset() {
    atomicStore(&control[0], 0u);
    atomicStore(&control[1], 0u);
    atomicStore(&control[2], 0u);
    atomicStore(&control[3], 0u);
    draw_args.index_count = 4u;
    draw_args.instance_count = 0u;
    draw_args.first_index = 0u;
    draw_args.base_vertex = 0;
    draw_args.first_instance = 0u;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.capacity {
        return;
    }

    var particle = particles[index];
    var alive = particle.position_life.w > 0.0;
    if alive {
        particle.position_life.w = max(particle.position_life.w - params.dt, 0.0);
        alive = particle.position_life.w > 0.0;
        if alive {
            let velocity = particle.velocity_pad.xyz + params.gravity.xyz * params.dt;
            let position = particle.position_life.xyz + velocity * params.dt;
            particle.velocity_pad = vec4<f32>(velocity, particle.velocity_pad.w);
            particle.position_life = vec4<f32>(position, particle.position_life.w);
        } else {
            particle.position_life = vec4<f32>(0.0);
            particle.velocity_pad = vec4<f32>(0.0);
        }
    }
    particles[index] = particle;

    if alive {
        let output = atomicAdd(&control[0], 1u);
        if output < params.capacity {
            active_indices[output] = index;
        } else {
            atomicAdd(&control[2], 1u);
        }
    }
}

// Each free slot competes for a bounded spawn ticket. Attribute initialization
// is identical for every ticket in this reference slice, so atomic scheduling
// changes only active-index order, not particle state. This keeps assignment
// parallel across the full pool instead of serially scanning capacity.
@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.capacity || params.spawn_count == 0u {
        return;
    }
    if particles[index].position_life.w <= 0.0 {
        let ticket = atomicAdd(&control[1], 1u);
        if ticket < params.spawn_count {
            particles[index].position_life = vec4<f32>(params.spawn_position.xyz, params.spawn_life);
            particles[index].velocity_pad = vec4<f32>(params.spawn_velocity.xyz, 0.0);
            if params.spawn_life > 0.0 {
                let output = atomicAdd(&control[0], 1u);
                active_indices[output] = index;
            }
        }
    }
}

@compute @workgroup_size(1)
fn finalize() {
    draw_args.index_count = 4u;
    draw_args.instance_count = min(atomicLoad(&control[0]), params.capacity);
    draw_args.first_index = 0u;
    draw_args.base_vertex = 0;
    draw_args.first_instance = 0u;
}
