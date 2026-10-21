struct Params {
    capacity: u32,
    force_count: u32,
    collider_count: u32,
    event_capacity: u32,
    dt: f32,
    restitution: f32,
    friction: f32,
    spawn_lifetime: f32,
    spawn_position: vec4<f32>,
    spawn_velocity: vec4<f32>,
    spawn_count: u32,
    condition_enabled: u32,
    integrator: u32,
    condition_x: f32,
};
struct Particle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
    radius: f32,
    alive: u32,
    particle_id: u32,
    emitter_id: u32,
};
struct ForceEntry {
    kind: u32,
    enabled: u32,
    falloff: u32,
    gate_radius: f32,
    gate_weight: vec4<f32>,
    data: array<vec4<f32>, 5>,
};
struct ColliderEntry {
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data: array<vec4<f32>, 5>,
};
struct EventRecord {
    payload: vec4<f32>,
    kind: u32,
    particle_index: u32,
    pad0: u32,
    pad1: u32,
};
struct CollisionResult { position: vec3<f32>, velocity: vec3<f32>, hit: u32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> forces: array<ForceEntry>;
@group(0) @binding(3) var<storage, read> colliders: array<ColliderEntry>;
@group(0) @binding(4) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> source_events: array<EventRecord>;
@group(0) @binding(6) var<storage, read_write> compact_events: array<EventRecord>;
@group(0) @binding(7) var<storage, read_write> event_cursors: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> channel_offsets: array<u32>;
@group(0) @binding(9) var<storage, read_write> active_indices: array<u32>;

const EPS: f32 = 1.0e-6;
const FORCE_GRAVITY: u32 = 0u;
const FORCE_LINEAR_DRAG: u32 = 1u;
const FORCE_QUADRATIC_DRAG: u32 = 2u;
const FORCE_WIND: u32 = 3u;
const FORCE_VORTEX: u32 = 4u;
const FORCE_POINT: u32 = 5u;
const FORCE_LINE: u32 = 6u;
const FORCE_RADIAL: u32 = 7u;
const FORCE_ORBITAL: u32 = 8u;
const FORCE_WELL: u32 = 9u;
const FORCE_TURBULENCE: u32 = 10u;
const FORCE_SPRING: u32 = 11u;

fn finite1(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}
fn is_finite3(v: vec3<f32>) -> bool {
    return finite1(v.x) && finite1(v.y) && finite1(v.z);
}
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (!finite1(l2) || l2 <= EPS * EPS) { return vec3<f32>(0.0); }
    return v * inverseSqrt(l2);
}
fn finite3(v: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), v, vec3<bool>(finite1(v.x), finite1(v.y), finite1(v.z)));
}
fn falloff(kind: u32, ratio: f32) -> f32 {
    let t = clamp(ratio, 0.0, 1.0);
    if (kind == 1u) { return 1.0 - t; }
    if (kind == 2u) { let s = 1.0 - t; return s * s * (3.0 - 2.0 * s); }
    if (kind == 3u) { return 1.0 / (1.0 + t * t); }
    return 1.0;
}
fn point_force(pos: vec3<f32>, attractor: vec3<f32>, strength: f32, radius: f32, softening: f32, shape: u32) -> vec3<f32> {
    if (!is_finite3(pos) || !is_finite3(attractor) || !finite1(strength) || !finite1(radius)) {
        return vec3<f32>(0.0);
    }
    let delta = attractor - pos;
    let d2 = dot(delta, delta);
    let dist = sqrt(d2);
    if (!finite1(dist) || (radius > EPS && dist > radius)) { return vec3<f32>(0.0); }
    var soft = 0.0;
    if (finite1(softening)) { soft = max(softening, 0.0); }
    let dir = delta * (1.0 / sqrt(d2 + soft * soft + EPS * EPS));
    var ratio = 0.0;
    if (radius > EPS) { ratio = dist / radius; }
    return dir * (strength * falloff(shape, ratio));
}
fn noise_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}
fn hash3(p: vec3<i32>, seed: u32) -> f32 {
    var h = seed ^ 0x9e3779b9u;
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    h ^= bitcast<u32>(p.x) * 0x85ebca6bu;
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    h ^= bitcast<u32>(p.y) * 0xc2b2ae35u;
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    h ^= bitcast<u32>(p.z) * 0x27d4eb2fu;
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    return f32(h >> 8u) / 8388608.0 - 1.0;
}
fn noise(p: vec3<f32>, seed: u32) -> f32 {
    if (!is_finite3(p)) { return 0.0; }
    let cell = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = hash3(cell, seed); let n100 = hash3(cell + vec3<i32>(1,0,0), seed);
    let n010 = hash3(cell + vec3<i32>(0,1,0), seed); let n110 = hash3(cell + vec3<i32>(1,1,0), seed);
    let n001 = hash3(cell + vec3<i32>(0,0,1), seed); let n101 = hash3(cell + vec3<i32>(1,0,1), seed);
    let n011 = hash3(cell + vec3<i32>(0,1,1), seed); let n111 = hash3(cell + vec3<i32>(1,1,1), seed);
    let x00 = noise_lerp(n000, n100, u.x); let x10 = noise_lerp(n010, n110, u.x);
    let x01 = noise_lerp(n001, n101, u.x); let x11 = noise_lerp(n011, n111, u.x);
    return noise_lerp(noise_lerp(x00, x10, u.y), noise_lerp(x01, x11, u.y), u.z);
}
fn force_value(f: ForceEntry, pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    let d0 = f.data[0]; let d1 = f.data[1]; let d2 = f.data[2];
    if (f.kind == FORCE_GRAVITY) { return finite3(d0.xyz); }
    if (f.kind == FORCE_LINEAR_DRAG) {
        if (!finite1(d0.x) || d0.x <= 0.0 || !finite1(d0.y) || d0.y <= 0.0) { return vec3<f32>(0.0); }
        return vel * (-min(d0.x, 1.0 / d0.y));
    }
    if (f.kind == FORCE_QUADRATIC_DRAG) {
        let speed = length(vel);
        if (!finite1(d0.x) || d0.x <= 0.0 || speed <= EPS) { return vec3<f32>(0.0); }
        return vel * (-d0.x * speed);
    }
    if (f.kind == FORCE_WIND) {
        if (!finite1(d0.w) || d0.w <= 0.0) { return vec3<f32>(0.0); }
        return (finite3(d0.xyz) - vel) * d0.w;
    }
    if (f.kind == FORCE_VORTEX) {
        if (!is_finite3(d0.xyz) || !is_finite3(d1.xyz) || !finite1(d1.w)) { return vec3<f32>(0.0); }
        let axis = safe_normalize(d1.xyz); let offset = pos - d0.xyz;
        let radial = offset - axis * dot(offset, axis);
        let r2 = dot(radial, radial);
        if (r2 <= EPS*EPS) { return vec3<f32>(0.0); }
        return cross(axis, radial) * (d1.w / r2);
    }
    if (f.kind == FORCE_POINT || f.kind == FORCE_RADIAL) {
        let strength = select(d1.x, -d1.x, f.kind == FORCE_RADIAL);
        return point_force(pos, d0.xyz, strength, d1.y, d1.z, f.falloff);
    }
    if (f.kind == FORCE_WELL) {
        if (!is_finite3(d0.xyz) || !finite1(d1.x)) { return vec3<f32>(0.0); }
        let toward = d0.xyz - pos;
        var soft = 0.0;
        if (finite1(d1.y)) { soft = max(d1.y, 0.0); }
        return safe_normalize(toward) * (d1.x / (dot(toward, toward) + soft * soft + EPS * EPS));
    }
    if (f.kind == FORCE_LINE) {
        if (!is_finite3(d0.xyz) || !is_finite3(d1.xyz)) { return vec3<f32>(0.0); }
        let axis = safe_normalize(d1.xyz);
        if (dot(axis, axis) <= EPS * EPS) {
            return point_force(pos, d0.xyz, d2.x, d2.y, d2.z, f.falloff);
        }
        let off = pos - d0.xyz;
        let nearest = d0.xyz + axis * dot(off, axis);
        return point_force(pos, nearest, d2.x, d2.y, d2.z, f.falloff);
    }
    if (f.kind == FORCE_ORBITAL) {
        if (!is_finite3(d0.xyz) || !is_finite3(d1.xyz) || !is_finite3(d2.xyz)) { return vec3<f32>(0.0); }
        let axis = safe_normalize(d1.xyz); let offset = pos - d0.xyz;
        if (dot(axis, axis) <= EPS * EPS) { return vec3<f32>(0.0); }
        let radial = offset - axis * dot(offset,axis); let dist = length(radial);
        if (dist <= EPS) { return vec3<f32>(0.0); }
        let radial_unit = safe_normalize(radial);
        let tangent = safe_normalize(cross(axis, radial_unit));
        return tangent * d2.y + radial_unit * (-max(d2.z, 0.0) * (dist - max(d2.x, 0.0)));
    }
    if (f.kind == FORCE_TURBULENCE) {
        if (!is_finite3(d0.xyz) || !finite1(d0.w)) { return vec3<f32>(0.0); }
        let eps = select(1.0e-3,d1.z,finite1(d1.z) && d1.z>0.0); let seed = bitcast<u32>(d1.x);
        let inv = 1.0 / (2.0 * eps);
        var result=vec3<f32>(0.0); var frequency=d0.x; var amplitude=d0.y;
        for (var octave=0u; octave<bitcast<u32>(d1.y); octave+=1u) {
            let q=pos*frequency; let s=seed+octave*0x01000193u;
            // Preserve zero * amplitude too: an overflowing amplitude invalidates
            // the whole entry, rather than retaining only its earlier octaves.
            var curl = vec3<f32>(0.0);
            if (is_finite3(q)) {
            let dx=vec3<f32>(eps,0.0,0.0); let dy=vec3<f32>(0.0,eps,0.0); let dz=vec3<f32>(0.0,0.0,eps);
            let s1=s; let s2=s+0x10000001u; let s3=s+0x20000002u;
            let p3dy=(noise(q+dy,s3)-noise(q-dy,s3))*inv; let p2dz=(noise(q+dz,s2)-noise(q-dz,s2))*inv;
            let p1dz=(noise(q+dz,s1)-noise(q-dz,s1))*inv; let p3dx=(noise(q+dx,s3)-noise(q-dx,s3))*inv;
            let p2dx=(noise(q+dx,s2)-noise(q-dx,s2))*inv; let p1dy=(noise(q+dy,s1)-noise(q-dy,s1))*inv;
            curl=vec3<f32>(p3dy-p2dz,p1dz-p3dx,p2dx-p1dy);
            }
            result+=curl*amplitude;
            frequency*=d0.z; amplitude*=d0.w;
        }
        return result;
    }
    if (f.kind == FORCE_SPRING) {
        if (!is_finite3(d0.xyz) || !finite1(d1.x) || !finite1(d1.y)) { return vec3<f32>(0.0); }
        return (d0.xyz - pos) * max(d1.x, 0.0) + vel * (-max(d1.y, 0.0));
    }
    return vec3<f32>(0.0);
}
fn acceleration(pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    if (!is_finite3(pos) || !is_finite3(vel)) { return vec3<f32>(0.0); }
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < params.force_count; i += 1u) {
        let f = forces[i];
        if (f.enabled == 0u || !finite1(f.gate_weight.w) || f.gate_weight.w <= 0.0) { continue; }
        if (f.gate_radius > EPS) {
            if (!finite1(f.gate_radius) || !is_finite3(f.gate_weight.xyz)) { continue; }
            let delta = pos - f.gate_weight.xyz;
            if (dot(delta, delta) > f.gate_radius * f.gate_radius) { continue; }
        }
        let contribution = force_value(f, pos, vel) * f.gate_weight.w;
        if (is_finite3(contribution)) { sum += contribution; }
    }
    return finite3(sum);
}
fn emit_event(kind: u32, index: u32, pos: vec3<f32>, age: f32) {
    let slot = atomicAdd(&control[0], 1u);
    atomicAdd(&control[3u + kind], 1u);
    if (slot < params.event_capacity) {
        source_events[slot] = EventRecord(vec4<f32>(pos, age), kind, index, 0u, 0u);
    } else {
        atomicAdd(&control[7], 1u);
    }
}
fn collider_contact(c: ColliderEntry, pos: vec3<f32>, radius: f32) -> vec4<f32> {
    if (c.kind == 0u) {
        let n = safe_normalize(c.data[0].xyz); let penetration = radius - (dot(n,pos) + c.data[0].w);
        return vec4<f32>(n, penetration);
    }
    if (c.kind == 1u) {
        let delta = pos - c.data[0].xyz; let dist = length(delta); let penetration = c.data[0].w + radius - dist;
        return vec4<f32>(select(vec3<f32>(0.0,1.0,0.0), delta / max(dist,EPS), dist > EPS), penetration);
    }
    if (c.kind == 2u) {
        let lo = c.data[0].xyz; let hi = c.data[1].xyz; let closest = clamp(pos,lo,hi); let delta=pos-closest; let dist=length(delta);
        if (dist > EPS) { return vec4<f32>(delta/dist, radius-dist); }
        let distances = min(pos-lo, hi-pos); let axis = select(select(vec3<f32>(0.0,0.0,1.0),vec3<f32>(0.0,1.0,0.0), distances.y < distances.z),vec3<f32>(1.0,0.0,0.0), distances.x < min(distances.y,distances.z));
        let sign_axis = select(-1.0,1.0, dot(pos-(lo+hi)*0.5,axis)>=0.0);
        return vec4<f32>(axis*sign_axis, radius + min(distances.x,min(distances.y,distances.z)));
    }
    if (c.kind == 3u) {
        let a=c.data[0].xyz; let b=c.data[1].xyz; let ab=b-a; let t=clamp(dot(pos-a,ab)/max(dot(ab,ab),EPS*EPS),0.0,1.0);
        let delta=pos-(a+ab*t); let dist=length(delta); let penetration=c.data[1].w+radius-dist;
        return vec4<f32>(select(vec3<f32>(0.0,1.0,0.0),delta/max(dist,EPS),dist>EPS),penetration);
    }
    let grad=safe_normalize(c.data[0].xyz); return vec4<f32>(grad, radius-c.data[0].w);
}
fn resolve(pos_in: vec3<f32>, vel_in: vec3<f32>, radius: f32, index: u32, age: f32) -> CollisionResult {
    var pos=pos_in; var vel=vel_in; var collided=false; var best=vec4<f32>(0.0);
    for (var i=0u; i<params.collider_count; i+=1u) {
        let contact=collider_contact(colliders[i],pos,radius);
        if (contact.w > best.w && dot(contact.xyz,contact.xyz)>EPS*EPS) {
            collided=true; best=contact;
        }
    }
    if (collided) {
        let n=safe_normalize(best.xyz); pos+=n*best.w;
        let vn=dot(vel,n); let vn_after=select(vn,-clamp(params.restitution,0.0,1.0)*vn,vn<0.0);
        let tangent=vel-n*vn; let tangent_speed=length(tangent);
        let new_tangent_speed=max(tangent_speed-clamp(params.friction,0.0,1.0)*abs(vn_after-vn),0.0);
        let tangent_after=select(vec3<f32>(0.0),tangent*(new_tangent_speed/max(tangent_speed,EPS)),tangent_speed>EPS);
        vel=n*vn_after+tangent_after;
    }
    if (collided) { emit_event(2u,index,pos,age); }
    return CollisionResult(pos,vel,select(0u,1u,collided));
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    for (var i=0u;i<12u;i+=1u) { atomicStore(&control[i],0u); }
    for (var i=0u;i<4u;i+=1u) { atomicStore(&event_cursors[i],0u); }
}
@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x; if (i>=params.capacity) { return; }
    if (particles[i].alive != 0u || atomicLoad(&control[1]) >= params.spawn_count) { return; }
    let claim=atomicAdd(&control[1],1u);
    if (claim>=params.spawn_count) { return; }
    particles[i].position_age=vec4<f32>(params.spawn_position.xyz,0.0);
    particles[i].velocity_lifetime=vec4<f32>(params.spawn_velocity.xyz,params.spawn_lifetime);
    particles[i].radius=max(particles[i].radius,0.0); particles[i].alive=1u;
    emit_event(0u,i,params.spawn_position.xyz,0.0);
}
@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x; if (i>=params.capacity || particles[i].alive==0u) { return; }
    var p=particles[i].position_age.xyz; var v=particles[i].velocity_lifetime.xyz;
    let age=particles[i].position_age.w; let life=particles[i].velocity_lifetime.w; let dt=max(params.dt,0.0);
    if (dt>0.0) {
        let a1=acceleration(p,v);
        if (params.integrator==1u) { p += v*dt + a1*(0.5*dt*dt); v += a1*dt; }
        else if (params.integrator==2u) { let midv=v+a1*(0.5*dt); let midp=p+v*(0.5*dt); let a2=acceleration(midp,midv); v+=a2*dt; p+=midv*dt; }
        else { v+=a1*dt; p+=v*dt; }
    }
    let new_age=age+dt;
    if (life>0.0 && new_age>=life) {
        particles[i].position_age=vec4<f32>(p,new_age); particles[i].velocity_lifetime=vec4<f32>(v,life); particles[i].alive=0u;
        emit_event(1u,i,p,new_age); return;
    }
    let collision=resolve(p,v,particles[i].radius,i,new_age); p=collision.position; v=collision.velocity;
    if (params.condition_enabled!=0u && p.x>=params.condition_x) { emit_event(3u,i,p,new_age); }
    particles[i].position_age=vec4<f32>(p,new_age); particles[i].velocity_lifetime=vec4<f32>(v,life);
}
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x; if (i>=params.capacity || particles[i].alive==0u) { return; }
    let slot=atomicAdd(&control[2],1u); active_indices[slot]=i;
}
@compute @workgroup_size(64)
fn event_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x; let count=min(atomicLoad(&control[0]),params.event_capacity); if (i>=count) { return; }
    let e=source_events[i]; if (e.kind>=4u) { return; }
    let local=atomicAdd(&event_cursors[e.kind],1u);
    let max_per_channel=arrayLength(&compact_events)/4u;
    if (local<max_per_channel) {
        let out_index=channel_offsets[e.kind]+local;
        compact_events[out_index]=e;
        atomicAdd(&control[8],1u);
    } else { atomicAdd(&control[7],1u); }
}
