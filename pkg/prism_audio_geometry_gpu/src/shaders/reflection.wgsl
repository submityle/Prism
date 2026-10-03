// First-order specular reflection kernel (image-source method).
//
// One invocation per (query, triangle) pair mirrors the emitter across the
// candidate face, validates the bounce (same side, on-face, both legs clear,
// above the gain floor) and emits a reflection candidate. This mirrors the CPU
// `resolve_reflections` in `prism_audio_geometry::reflection_path` arithmetic
// for arithmetic; the host merges, de-duplicates, sorts, and caps the
// survivors.
//
// Provenance: original work; standard wgpu compute mirroring the classic
// image-source construction already implemented on the CPU; no Unreal Engine,
// Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio
// source or derived code; no AI/ML.

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

struct ReflectionCandidate {
    direction: vec4<f32>,
    delay_seconds: f32,
    gain: f32,
    valid: u32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<ReflectionCandidate>;

const TRI_EPSILON: f32 = 1.0e-7;
const COINCIDENT: f32 = 1.0e-6;
const F32_EPSILON: f32 = 1.1920929e-7;
const BARY_TOL: f32 = 1.0e-4;

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    let b2 = dot(b, b);
    return v * (w * w - b2) + b * (dot(v, b) * 2.0) + cross(b, v) * (w * 2.0);
}

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

// Mirrors `AcousticScene::segment_blocked`: shrinks the segment by `eps` at both
// ends so a surface an endpoint already lies on does not count as a blocker.
fn segment_blocked(seg_from: vec3<f32>, seg_to: vec3<f32>, eps: f32) -> bool {
    let delta = seg_to - seg_from;
    let len = sqrt(dot(delta, delta));
    let e = max(eps, 0.0);
    if len <= 2.0 * e {
        return false;
    }
    let dir = delta / len;
    let origin = seg_from + dir * e;
    return first_hit(origin, dir, len - 2.0 * e).valid;
}

// Coplanar barycentric containment with a small positive slack, mirroring the
// CPU `point_in_triangle`.
fn point_in_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;
    let dot00 = dot(v0, v0);
    let dot01 = dot(v0, v1);
    let dot02 = dot(v0, v2);
    let dot11 = dot(v1, v1);
    let dot12 = dot(v1, v2);
    let denom = dot00 * dot11 - dot01 * dot01;
    if abs(denom) <= F32_EPSILON {
        return false;
    }
    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    return u >= -BARY_TOL && v >= -BARY_TOL && (u + v) <= 1.0 + BARY_TOL;
}

@compute @workgroup_size(64)
fn resolve_reflection(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.query_count * params.triangle_count;
    if idx >= total {
        return;
    }
    let q = idx / params.triangle_count;
    let t = idx % params.triangle_count;

    var out: ReflectionCandidate;
    out.direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.delay_seconds = 0.0;
    out.gain = 0.0;
    out.valid = 0u;
    out.pad = 0.0;

    let query = queries[q];
    let lp = query.listener_pos.xyz;
    let ep = query.emitter_pos.xyz;
    let orient = query.listener_orient;
    let tri = triangles[t];
    let normal = tri.normal.xyz;

    // A degenerate face is stored with a zero normal and is skipped exactly as
    // the CPU `triangle_normal` -> None path does.
    if dot(normal, normal) <= 0.0 {
        results[idx] = out;
        return;
    }
    let a = tri.a.xyz;
    let b = tri.b.xyz;
    let c = tri.c.xyz;

    let d_listener = dot(lp - a, normal);
    let d_source = dot(ep - a, normal);
    if d_listener * d_source <= 0.0 {
        results[idx] = out;
        return;
    }

    let image = ep - 2.0 * d_source * normal;
    let direction_v = image - lp;
    let denom = dot(direction_v, normal);
    if abs(denom) <= F32_EPSILON {
        results[idx] = out;
        return;
    }
    let tt = -d_listener / denom;
    if !(tt > 0.0 && tt < 1.0) {
        results[idx] = out;
        return;
    }
    let point = lp + tt * direction_v;
    if !point_in_triangle(point, a, b, c) {
        results[idx] = out;
        return;
    }

    let eps = max(params.surface_epsilon, 0.0);
    if segment_blocked(lp, point, eps) || segment_blocked(point, ep, eps) {
        results[idx] = out;
        return;
    }

    let leg_in = point - lp;
    let leg_out = ep - point;
    let path_length = sqrt(dot(leg_in, leg_in)) + sqrt(dot(leg_out, leg_out));
    if path_length <= 0.0 {
        results[idx] = out;
        return;
    }
    let base_vec = ep - lp;
    let base = sqrt(dot(base_vec, base_vec));
    let spreading = clamp(base / path_length, 0.0, 1.0);
    let gain = clamp(tri.reflection_gain * spreading, 0.0, 1.0);
    if gain <= params.min_gain {
        results[idx] = out;
        return;
    }

    // localize the reflection point direction into the listener frame.
    let to_point = point - lp;
    let d = sqrt(dot(to_point, to_point));
    var ldir = vec3<f32>(0.0, 0.0, -1.0);
    if d > COINCIDENT {
        ldir = quat_rotate(quat_conj(orient), to_point / d);
    }

    out.direction = vec4<f32>(ldir, 0.0);
    out.delay_seconds = path_length / params.speed_of_sound;
    out.gain = gain;
    out.valid = 1u;
    out.pad = 0.0;
    results[idx] = out;
}
