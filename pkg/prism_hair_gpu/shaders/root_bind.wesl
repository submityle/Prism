// Root-binding bake twin: evaluates on-device the same per-root nearest-triangle
// scalp attachment the CPU golden
// `prism_render_architecture::hair::binding::bind_roots` computes, one thread
// per strand root.
//
// A groom is authored in a character's rest pose. Before it can ride the skinned
// scalp every frame (the `root_resolve` replay), each strand root must first be
// pinned to the surface it grew from: at import time the root is projected onto
// its closest scalp triangle and stored as a barycentric attachment plus a
// signed height along the face normal. This is the import-time bake `UE5` Groom
// builds into its binding asset and the root-skinning setup `TressFX` runs once
// per groom. This kernel is the on-device twin of exactly that bake: one thread
// per root brute-force scans every triangle for the one whose closest surface
// point is nearest (ties resolve to the lowest triangle index, matching the
// golden's first-strict-minimum scan), then records the barycentric projection
// and the signed height along that face's normal. It bakes; it does not resolve
// (the per-frame replay stays the `root_resolve` pass). Each thread owns one
// root and writes one disjoint output slot, so the pass is race-free.
//
// Buffer layout: `params` carries the root / triangle / vertex counts; `roots`
// is the authored rest-pose root positions (xyz in a vec4, w padding);
// `tri_indices` is the flattened triangle list (three vertex indices per face);
// `vertices` is the rest-pose scalp vertex pool (xyz in a vec4, w padding);
// `out_bindings` receives one `MeshBinding` per root (triangle index, three
// barycentric weights, signed height, three pad words).
//
// Degenerate inputs match the reference exactly: an empty triangle list, or a
// triangle whose vertex index is out of range, is skipped; a root that finds no
// bindable triangle resolves to the `UNBOUND` sentinel with zeroed weights; a
// zero-area face contributes a zero height. A bad root never panics and never
// poisons its neighbours.
//
// Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
// plus a brute-force nearest scan plus wgpu compute dispatch; no Unreal Engine
// source or derived code.

// `MeshBinding::UNBOUND` sentinel: this root failed to bind to any triangle.
const UNBOUND: u32 = 0xffffffffu;
// Squared-length degeneracy guard (mirrors `binding::EPS_LEN_SQ`).
const EPS_LEN_SQ: f32 = 1.0e-12;
// `f32::EPSILON`, the threshold `Vec3::normalize_or` compares squared length to.
const F32_EPSILON: f32 = 1.1920929e-7;

struct Params {
    // Number of strand roots, bounding the dispatch (one thread per root).
    root_count: u32,
    // Number of scalp triangles scanned per root.
    triangle_count: u32,
    // Number of rest-pose vertices; a corner index `>= vertex_count` skips its face.
    vertex_count: u32,
    pad0: u32,
}

// One strand root's attachment. Matches the host `GpuMeshBinding`: the triangle
// index, the three barycentric weights, the signed height, and three pad words
// so the stride is a 16-byte multiple with no `vec3` alignment hazard.
struct MeshBinding {
    triangle: u32,
    bary0: f32,
    bary1: f32,
    bary2: f32,
    height: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The closest point on a triangle plus its barycentric weights, returned by
// `closest_point_on_triangle`.
struct Closest {
    point: vec3<f32>,
    bary: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
// Authored rest-pose root positions (xyz in a vec4, w unused).
@group(0) @binding(1) var<storage, read> roots: array<vec4<f32>>;
// Flattened triangle list: three vertex indices per face at `3*triangle`.
@group(0) @binding(2) var<storage, read> tri_indices: array<u32>;
// Rest-pose scalp vertex positions (xyz in a vec4, w unused).
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
// Per-root baked attachment.
@group(0) @binding(4) var<storage, read_write> out_bindings: array<MeshBinding>;

// Bit-faithful port of `Vec3::normalize_or`: returns the unit vector, or
// `fallback` when the squared length is at or below `f32::EPSILON`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let ls = dot(v, v);
    if ls <= F32_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(ls));
}

// Bit-faithful port of `binding::closest_point_on_triangle`: the standard
// Voronoi-region test from Ericson's Real-Time Collision Detection. Returns the
// point on triangle `abc` closest to `p` plus its barycentric weights; the
// branch order and every comparison match the reference exactly so both sides
// pick the same region.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Closest {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return Closest(a, vec3<f32>(1.0, 0.0, 0.0));
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    // Vertex region B.
    if d3 >= 0.0 && d4 <= d3 {
        return Closest(b, vec3<f32>(0.0, 1.0, 0.0));
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        var v = 0.0;
        if denom > EPS_LEN_SQ {
            v = d1 / denom;
        }
        let point = a + ab * v;
        return Closest(point, vec3<f32>(1.0 - v, v, 0.0));
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    // Vertex region C.
    if d6 >= 0.0 && d5 <= d6 {
        return Closest(c, vec3<f32>(0.0, 0.0, 1.0));
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        var w = 0.0;
        if denom > EPS_LEN_SQ {
            w = d2 / denom;
        }
        let point = a + ac * w;
        return Closest(point, vec3<f32>(1.0 - w, 0.0, w));
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    let bc_num = d4 - d3;
    let bc_den = (d4 - d3) + (d5 - d6);
    if va <= 0.0 && bc_num >= 0.0 && (d5 - d6) >= 0.0 {
        var w = 0.0;
        if bc_den > EPS_LEN_SQ {
            w = bc_num / bc_den;
        }
        let point = b + (c - b) * w;
        return Closest(point, vec3<f32>(0.0, 1.0 - w, w));
    }

    // Interior face region: barycentric via the precomputed numerators.
    let denom_sum = va + vb + vc;
    if denom_sum > EPS_LEN_SQ {
        let inv = 1.0 / denom_sum;
        let v = vb * inv;
        let w = vc * inv;
        let point = a + ab * v + ac * w;
        return Closest(point, vec3<f32>(1.0 - v - w, v, w));
    }
    // Zero-area triangle: fall back to vertex A.
    return Closest(a, vec3<f32>(1.0, 0.0, 0.0));
}

// Signed height of `offset` along the face normal (`face_normal` then
// `offset.dot(n)`); a zero-area face contributes zero, matching the golden's
// `map_or(0.0, ...)`.
fn signed_height(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, offset: vec3<f32>) -> f32 {
    let n_raw = cross(b - a, c - a);
    if dot(n_raw, n_raw) > EPS_LEN_SQ {
        let n = normalize_or(n_raw, vec3<f32>(0.0, 1.0, 0.0));
        return dot(offset, n);
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= params.root_count {
        return;
    }
    let root = roots[r].xyz;

    // Brute-force nearest scan, matching the golden's first-strict-minimum loop:
    // ties resolve to the lowest triangle index because the update is `<` only.
    var found = false;
    var best_dist = 0.0;
    var best_tri = UNBOUND;
    var best_bary = vec3<f32>(0.0, 0.0, 0.0);
    var best_height = 0.0;

    for (var index = 0u; index < params.triangle_count; index = index + 1u) {
        let ti = index * 3u;
        let ia = tri_indices[ti + 0u];
        let ib = tri_indices[ti + 1u];
        let ic = tri_indices[ti + 2u];
        // `triangle_corners`: any out-of-range vertex index skips the face.
        if ia >= params.vertex_count || ib >= params.vertex_count || ic >= params.vertex_count {
            continue;
        }
        let a = vertices[ia].xyz;
        let b = vertices[ib].xyz;
        let c = vertices[ic].xyz;
        let hit = closest_point_on_triangle(root, a, b, c);
        let offset = root - hit.point;
        let dist_sq = dot(offset, offset);
        var closer = false;
        if found {
            closer = dist_sq < best_dist;
        } else {
            closer = true;
        }
        if closer {
            best_height = signed_height(a, b, c, offset);
            found = true;
            best_dist = dist_sq;
            best_tri = index;
            best_bary = hit.bary;
        }
    }

    if found {
        out_bindings[r] = MeshBinding(
            best_tri,
            best_bary.x,
            best_bary.y,
            best_bary.z,
            best_height,
            0u,
            0u,
            0u,
        );
    } else {
        out_bindings[r] = MeshBinding(UNBOUND, 0.0, 0.0, 0.0, 0.0, 0u, 0u, 0u);
    }
}
