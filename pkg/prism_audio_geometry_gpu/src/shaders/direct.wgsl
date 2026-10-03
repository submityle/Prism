// Direct line-of-sight arrival and transmission kernel.
//
// One invocation per query marches the listener-to-emitter segment through the
// scene, folding each crossed partition's transmission gain, and emits the
// direct/transmission arrival plus the occlusion it implies. This mirrors the
// CPU `resolve_direct` in `prism_audio_geometry::direct_path` arithmetic for
// arithmetic.
//
// Provenance: original work; standard wgpu compute mirroring the classic
// ray-march already implemented on the CPU; no Unreal Engine, Unity, Godot,
// Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio source or derived
// code; no AI/ML.

struct Params {
    triangle_count: u32,
    query_count: u32,
    min_gain: f32,
    surface_epsilon: f32,
    speed_of_sound: f32,
    transmission_enabled: u32,
    pad0: u32,
    pad1: u32,
};

struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    normal: vec4<f32>,
    transmission_gain: f32,
    reflection_gain: f32,
    pad0: f32,
    pad1: f32,
};

struct Query {
    listener_pos: vec4<f32>,
    listener_orient: vec4<f32>,
    emitter_pos: vec4<f32>,
};

struct DirectResult {
    direction: vec4<f32>,
    delay_seconds: f32,
    gain: f32,
    cutoff_hz: f32,
    base_distance: f32,
    obstruction: f32,
    occlusion: f32,
    kind: u32,
    audible: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<DirectResult>;

const FULL_BAND: f32 = 1.0e6;
const TRI_EPSILON: f32 = 1.0e-7;
const COINCIDENT: f32 = 1.0e-6;
const KIND_DIRECT: u32 = 0u;
const KIND_TRANSMISSION: u32 = 1u;
const MAX_MARCH_HITS: u32 = 64u;

// Conjugate of a unit quaternion equals its inverse.
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Rotates vector `v` by quaternion `q` (glam's quat * vec3 arithmetic).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    let b2 = dot(b, b);
    return v * (w * w - b2) + b * (dot(v, b) * 2.0) + cross(b, v) * (w * 2.0);
}

// Moller-Trumbore ray/triangle test. Returns (hit_flag, t): hit_flag > 0.5 and
// a valid `t` when the ray hits within [0, max_distance]. Double-sided.
fn ray_triangle(origin: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, max_distance: f32) -> vec2<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = cross(dir, edge2);
    let det = dot(edge1, pvec);
    if abs(det) < TRI_EPSILON {
        return vec2<f32>(0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = dot(tvec, pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return vec2<f32>(0.0, 0.0);
    }
    let qvec = cross(tvec, edge1);
    let v = dot(dir, qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return vec2<f32>(0.0, 0.0);
    }
    let t = dot(edge2, qvec) * inv_det;
    if t >= 0.0 && t <= max_distance {
        return vec2<f32>(1.0, t);
    }
    return vec2<f32>(0.0, 0.0);
}

struct Hit {
    valid: bool,
    t: f32,
    index: u32,
};

// Nearest surface the ray hits no farther than `max_distance`, scanning every
// triangle in index order; ties resolve to the lowest index (strict `<`).
fn first_hit(origin: vec3<f32>, dir: vec3<f32>, max_distance: f32) -> Hit {
    var out: Hit;
    out.valid = false;
    out.t = max_distance;
    out.index = 0u;
    if max_distance <= 0.0 {
        return out;
    }
    var found = false;
    var best_t = max_distance;
    var best_index = 0u;
    for (var i = 0u; i < params.triangle_count; i = i + 1u) {
        let tri = triangles[i];
        let r = ray_triangle(origin, dir, tri.a.xyz, tri.b.xyz, tri.c.xyz, max_distance);
        if r.x > 0.5 {
            if !found || r.y < best_t {
                found = true;
                best_t = r.y;
                best_index = i;
            }
        }
    }
    out.valid = found;
    out.t = best_t;
    out.index = best_index;
    return out;
}

@compute @workgroup_size(64)
fn resolve_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if q >= params.query_count {
        return;
    }
    let query = queries[q];
    let lp = query.listener_pos.xyz;
    let ep = query.emitter_pos.xyz;
    let orient = query.listener_orient;

    // localize (mirrors Listener::localize): express the world listener->source
    // direction in the listener's local frame via the inverse orientation.
    let to_source = ep - lp;
    let dist = sqrt(dot(to_source, to_source));
    var direction = vec3<f32>(0.0, 0.0, -1.0);
    var distance = 0.0;
    if dist > COINCIDENT {
        direction = quat_rotate(quat_conj(orient), to_source / dist);
        distance = dist;
    }
    let delay = distance / params.speed_of_sound;

    // March the segment, folding each partition's transmission gain exactly as
    // the CPU `march_segment` + `resolve_direct` visit does.
    let eps = max(params.surface_epsilon, 0.0);
    var transmitted = 1.0;
    var crossings = 0u;
    if dist > 0.0 {
        let dir = to_source / dist;
        var cursor = lp;
        var remaining = dist;
        for (var step_index = 0u; step_index < MAX_MARCH_HITS; step_index = step_index + 1u) {
            if remaining <= eps {
                break;
            }
            let hit = first_hit(cursor, dir, remaining);
            if !hit.valid {
                break;
            }
            transmitted = transmitted * triangles[hit.index].transmission_gain;
            crossings = crossings + 1u;
            if !(transmitted > params.min_gain) {
                break;
            }
            let step = hit.t + eps;
            cursor = cursor + dir * step;
            remaining = remaining - step;
        }
    }

    var out: DirectResult;
    out.direction = vec4<f32>(direction, 0.0);
    out.delay_seconds = delay;
    out.cutoff_hz = FULL_BAND;
    out.base_distance = distance;
    if crossings == 0u {
        out.gain = 1.0;
        out.obstruction = 0.0;
        out.occlusion = 0.0;
        out.kind = KIND_DIRECT;
        out.audible = 1u;
    } else {
        let blocked = clamp(1.0 - transmitted, 0.0, 1.0);
        out.obstruction = blocked;
        out.occlusion = blocked;
        out.kind = KIND_TRANSMISSION;
        let audible_bool = (params.transmission_enabled != 0u) && (transmitted > params.min_gain);
        if audible_bool {
            out.gain = transmitted;
            out.audible = 1u;
        } else {
            out.gain = 0.0;
            out.audible = 0u;
        }
    }
    results[q] = out;
}
