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
    pad: u32,
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
fn softened_toward(pos: vec3<f32>, target_pos: vec3<f32>, soft: f32) -> vec4<f32> {
    let delta = target_pos - pos;
    let d2 = dot(delta, delta);
    let dist = sqrt(max(d2, 0.0));
    let s = max(soft, 0.0);
    return vec4<f32>(delta * inverseSqrt(max(d2 + s * s + EPS * EPS, EPS * EPS)), dist);
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
    let cell = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = hash3(cell, seed); let n100 = hash3(cell + vec3<i32>(1,0,0), seed);
    let n010 = hash3(cell + vec3<i32>(0,1,0), seed); let n110 = hash3(cell + vec3<i32>(1,1,0), seed);
    let n001 = hash3(cell + vec3<i32>(0,0,1), seed); let n101 = hash3(cell + vec3<i32>(1,0,1), seed);
    let n011 = hash3(cell + vec3<i32>(0,1,1), seed); let n111 = hash3(cell + vec3<i32>(1,1,1), seed);
    let x00 = mix(n000, n100, u.x); let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x); let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn noise3(p: vec3<f32>, seed: u32) -> vec3<f32> {
    return vec3<f32>(noise(p, seed), noise(p + vec3<f32>(19.1, 7.7, 3.4), seed ^ 0x68bc21ebu), noise(p + vec3<f32>(5.2, 17.3, 11.8), seed ^ 0x02e5be93u));
}
fn force_value(f: ForceEntry, pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    let d0 = f.data[0]; let d1 = f.data[1]; let d2 = f.data[2];
    if (f.kind == FORCE_GRAVITY) { return d0.xyz; }
    if (f.kind == FORCE_LINEAR_DRAG) { return -vel * min(max(d0.x, 0.0), 1.0 / max(d0.y, EPS)); }
    if (f.kind == FORCE_QUADRATIC_DRAG) { return -vel * length(vel) * max(d0.x, 0.0); }
    if (f.kind == FORCE_WIND) { return (d0.xyz - vel) * d0.w; }
    if (f.kind == FORCE_VORTEX) {
        let axis = safe_normalize(d1.xyz); let offset = pos - d0.xyz;
        let radial = offset - axis * dot(offset, axis);
        let r2 = dot(radial, radial);
        if (r2 <= EPS*EPS) { return vec3<f32>(0.0); }
        return cross(axis, radial) * (d1.w / r2);
    }
    if (f.kind == FORCE_POINT || f.kind == FORCE_RADIAL || f.kind == FORCE_WELL) {
        let toward = d0.xyz - pos;
        let radial_target = select(toward, -toward, f.kind == FORCE_RADIAL);
        let dist = length(radial_target);
        if (f.kind == FORCE_WELL) { return safe_normalize(toward) * (d1.x / max(dot(toward,toward) + d1.y*d1.y + EPS*EPS, EPS)); }
        let softened = radial_target * inverseSqrt(max(dot(radial_target,radial_target) + max(d1.z, 0.0) * max(d1.z, 0.0) + EPS*EPS, EPS*EPS));
        let radius = d1.y;
        if (radius > EPS && dist > radius) { return vec3<f32>(0.0); }
        let shape = select(1.0, falloff(f.falloff, select(0.0, dist / max(radius, EPS), radius > EPS)), radius > EPS);
        if (f.kind == FORCE_RADIAL) { return softened * d1.x * shape; }
        return softened * d1.x * shape;
    }
    if (f.kind == FORCE_LINE) {
        let axis = safe_normalize(d1.xyz);
        let off = pos - d0.xyz;
        let nearest = select(d0.xyz, d0.xyz + axis * dot(off, axis), dot(axis,axis) > EPS*EPS);
        let to = nearest - pos; let dist = length(to);
        if (d2.y > EPS && dist > d2.y) { return vec3<f32>(0.0); }
        let softened = to * inverseSqrt(max(dot(to,to) + d2.z*d2.z + EPS*EPS, EPS*EPS));
        return softened * d2.x * select(1.0, falloff(f.falloff, dist / max(d2.y, EPS)), d2.y > EPS);
    }
    if (f.kind == FORCE_ORBITAL) {
        let axis = safe_normalize(d1.xyz); let offset = pos - d0.xyz;
        let radial = offset - axis * dot(offset,axis); let dist = length(radial);
        if (dist <= EPS) { return vec3<f32>(0.0); }
        let tangent = safe_normalize(cross(axis, radial));
        let radial_unit = radial / dist;
        return tangent * d2.y - radial_unit * ((dist - max(d2.x,0.0)) * max(d2.z,0.0));
    }
    if (f.kind == FORCE_TURBULENCE) {
        let eps = select(1.0e-3,d1.z,d1.z>0.0); let p = pos * d0.x; let seed = bitcast<u32>(d1.x);
        var result=vec3<f32>(0.0); var frequency=1.0; var amplitude=1.0;
        for (var octave=0u; octave<u32(max(d1.y,0.0)); octave+=1u) {
            let q=p*frequency; let s=seed+octave*0x01000193u;
            let dx=vec3<f32>(eps,0.0,0.0); let dy=vec3<f32>(0.0,eps,0.0); let dz=vec3<f32>(0.0,0.0,eps);
            let s1=s; let s2=s+0x10000001u; let s3=s+0x20000002u;
            let p3dy=(noise(q+dy,s3)-noise(q-dy,s3))/(2.0*eps); let p2dz=(noise(q+dz,s2)-noise(q-dz,s2))/(2.0*eps);
            let p1dz=(noise(q+dz,s1)-noise(q-dz,s1))/(2.0*eps); let p3dx=(noise(q+dx,s3)-noise(q-dx,s3))/(2.0*eps);
            let p2dx=(noise(q+dx,s2)-noise(q-dx,s2))/(2.0*eps); let p1dy=(noise(q+dy,s1)-noise(q-dy,s1))/(2.0*eps);
            result+=vec3<f32>(p3dy-p2dz,p1dz-p3dx,p2dx-p1dy)*(d0.y*amplitude);
            frequency*=d0.z; amplitude*=d0.w;
        }
        return result;
    }
    if (f.kind == FORCE_SPRING) { return (d0.xyz - pos) * d1.x - vel * d1.y; }
    return vec3<f32>(0.0);
}
fn acceleration(pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < params.force_count; i += 1u) {
        let f = forces[i];
        if (f.enabled == 0u || !finite1(f.gate_weight.w) || f.gate_weight.w <= 0.0 || !finite1(f.gate_radius)) { continue; }
        let gate_dist = length(pos - f.gate_weight.xyz);
        if (f.gate_radius > EPS && gate_dist > f.gate_radius) { continue; }
        sum += force_value(f, pos, vel) * f.gate_weight.w;
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
