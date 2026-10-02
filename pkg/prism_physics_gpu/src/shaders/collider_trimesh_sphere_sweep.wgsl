// Swept-sphere-versus-triangle-mesh scene query kernel: one lane sweeps the
// moving sphere against one triangle and writes the per-triangle time of impact,
// contact point, push-out normal, and a hit flag. The host reduces the
// per-triangle rows to the earliest contact, so the device result matches the
// CPU brute golden operation for operation (up to the handful of square roots
// and reciprocals' floating-point tolerance). This mirrors
// collider/trimesh_sphere_sweep.rs exactly: one face test plus three
// edge-capsule tests, with an initial-overlap short circuit.
//
// Provenance: moving-sphere / rounded-triangle reduction and the
// ray-versus-capsule and intersecting-moving-sphere tests per Ericson,
// "Real-Time Collision Detection" (2005), sections 5.3.4 and 5.5.7. No Unreal
// Engine source or derived code.

struct Params {
    // xyz: sweep start centre; w: max travel distance.
    origin_maxdist: vec4<f32>,
    // xyz: sweep direction (should be unit); w: sphere radius.
    dir_radius: vec4<f32>,
    // x: triangle count; y, z, w: padding.
    counts: vec4<u32>,
};

// Per-triangle output, two vec4 rows per triangle:
//   results[2*tri]     = (toi, point.x, point.y, point.z)
//   results[2*tri + 1] = (hit_flag, normal.x, normal.y, normal.z)
// hit_flag is 1.0 on a contact, else 0.0.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<f32>>;

const PARALLEL_EPS: f32 = 1e-8;
const ON_SURFACE_EPS2: f32 = 1e-12;

struct SweepResult {
    toi: f32,
    hit: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
};

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > ON_SURFACE_EPS2) {
        return v / sqrt(len2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Returns the point on triangle (a, b, c) closest to p (Ericson 5.1.5).
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

// Earliest intersection of the unit-direction ray with a sphere, or -1.0 on a
// miss. A start already inside the sphere (negative first root) is a miss here;
// the triangle-level initial-overlap test handles that case.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, centre: vec3<f32>, r: f32) -> f32 {
    let m = origin - centre;
    let b = dot(m, dir);
    let c = dot(m, m) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let s = -b - sqrt(disc);
    if (s >= 0.0 && s <= max_dist) {
        return s;
    }
    return -1.0;
}

// Sweep the sphere against the triangle plane; a hit requires the touch point to
// project inside the solid triangle.
fn sweep_face(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, r: f32,
              a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> SweepResult {
    var res: SweepResult;
    res.hit = 0.0;
    let raw = cross(b - a, c - a);
    if (dot(raw, raw) < PARALLEL_EPS) {
        return res;
    }
    let n = normalize(raw);
    let dist0 = dot(origin - a, n);
    let d_n = dot(dir, n);
    if (abs(d_n) < PARALLEL_EPS) {
        return res;
    }
    var target = r;
    if (dist0 < 0.0) {
        target = -r;
    }
    let s = (target - dist0) / d_n;
    if (s < 0.0 || s > max_dist) {
        return res;
    }
    let centre_s = origin + dir * s;
    let contact = centre_s - n * target;
    let q = closest_point_on_triangle(contact, a, b, c);
    if (dot(q - contact, q - contact) > ON_SURFACE_EPS2) {
        return res;
    }
    var normal = n;
    if (dist0 < 0.0) {
        normal = -n;
    }
    res.hit = 1.0;
    res.toi = s;
    res.point = contact;
    res.normal = normal;
    return res;
}

// Sweep the sphere against one edge treated as a capsule of radius r.
fn sweep_edge(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, r: f32,
              p1: vec3<f32>, p2: vec3<f32>) -> SweepResult {
    var res: SweepResult;
    res.hit = 0.0;
    let axis = p2 - p1;
    let axis_len2 = dot(axis, axis);
    var best_s = -1.0;

    if (axis_len2 > PARALLEL_EPS) {
        let axis_len = sqrt(axis_len2);
        let ax = axis / axis_len;
        let m = origin - p1;
        let dir_perp = dir - ax * dot(dir, ax);
        let m_perp = m - ax * dot(m, ax);
        let aa = dot(dir_perp, dir_perp);
        if (aa > PARALLEL_EPS) {
            let bb = 2.0 * dot(m_perp, dir_perp);
            let cc = dot(m_perp, m_perp) - r * r;
            let disc = bb * bb - 4.0 * aa * cc;
            if (disc >= 0.0) {
                let s = (-bb - sqrt(disc)) / (2.0 * aa);
                if (s >= 0.0 && s <= max_dist) {
                    let centre_s = origin + dir * s;
                    let u = dot(centre_s - p1, ax);
                    if (u >= 0.0 && u <= axis_len) {
                        best_s = s;
                    }
                }
            }
        }
    }

    for (var i: i32 = 0; i < 2; i = i + 1) {
        var cap = p1;
        var is_p1 = true;
        if (i == 1) {
            cap = p2;
            is_p1 = false;
        }
        let s = ray_sphere(origin, dir, max_dist, cap, r);
        if (s < 0.0) {
            continue;
        }
        let centre_s = origin + dir * s;
        let along = dot(centre_s - p1, axis);
        var valid = false;
        if (axis_len2 <= PARALLEL_EPS) {
            valid = true;
        } else if (is_p1) {
            valid = along <= 0.0;
        } else {
            valid = along >= axis_len2;
        }
        if (valid && (best_s < 0.0 || s < best_s)) {
            best_s = s;
        }
    }

    if (best_s < 0.0) {
        return res;
    }
    let centre_s = origin + dir * best_s;
    var t = 0.0;
    if (axis_len2 > PARALLEL_EPS) {
        t = clamp(dot(centre_s - p1, axis) / axis_len2, 0.0, 1.0);
    }
    let q = p1 + axis * t;
    res.hit = 1.0;
    res.toi = best_s;
    res.point = q;
    res.normal = safe_normalize(centre_s - q);
    return res;
}

// Keep the earlier contact, preferring the running best on an exact tie.
fn merge(best: SweepResult, cand: SweepResult) -> SweepResult {
    if (cand.hit > 0.5 && (best.hit < 0.5 || cand.toi < best.toi)) {
        return cand;
    }
    return best;
}

fn sweep_sphere_triangle(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, radius: f32,
                         a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> SweepResult {
    var res: SweepResult;
    res.hit = 0.0;

    let q0 = closest_point_on_triangle(origin, a, b, c);
    let diff0 = origin - q0;
    let d02 = dot(diff0, diff0);
    if (d02 < radius * radius) {
        var normal = safe_normalize(cross(b - a, c - a));
        if (d02 > ON_SURFACE_EPS2) {
            normal = diff0 / sqrt(d02);
        }
        res.hit = 1.0;
        res.toi = 0.0;
        res.point = q0;
        res.normal = normal;
        return res;
    }

    var best = sweep_face(origin, dir, max_dist, radius, a, b, c);
    best = merge(best, sweep_edge(origin, dir, max_dist, radius, a, b));
    best = merge(best, sweep_edge(origin, dir, max_dist, radius, b, c));
    best = merge(best, sweep_edge(origin, dir, max_dist, radius, c, a));
    return best;
}

@compute @workgroup_size(64)
fn collider_trimesh_sphere_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if (tri >= params.counts.x) {
        return;
    }

    let idx = indices[tri];
    let a = vertices[idx.x].xyz;
    let b = vertices[idx.y].xyz;
    let c = vertices[idx.z].xyz;

    let origin = params.origin_maxdist.xyz;
    let max_dist = params.origin_maxdist.w;
    let dir = params.dir_radius.xyz;
    let radius = params.dir_radius.w;

    let r = sweep_sphere_triangle(origin, dir, max_dist, radius, a, b, c);
    results[2u * tri] = vec4<f32>(r.toi, r.point.x, r.point.y, r.point.z);
    results[2u * tri + 1u] = vec4<f32>(r.hit, r.normal.x, r.normal.y, r.normal.z);
}
