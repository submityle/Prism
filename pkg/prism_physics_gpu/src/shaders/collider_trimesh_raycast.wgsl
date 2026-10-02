// Ray-versus-triangle-mesh scene query kernel: one lane intersects one
// triangle with the query ray via the Moller-Trumbore algorithm (1997) and
// writes the ray parameter and barycentric edge coordinates. The host reduces
// the per-triangle results to the nearest hit and finalises point, normal, and
// barycentric weights with the shared CPU rule, so the device result matches
// the CPU brute golden operation for operation (up to the single reciprocal's
// floating-point tolerance).
//
// Double-sided: a near-zero determinant (ray parallel to the triangle plane)
// misses, but either face is otherwise accepted. This mirrors
// collider/trimesh_raycast.rs exactly.
//
// Provenance: Moller-Trumbore ray/triangle intersection (1997). No Unreal
// Engine source or derived code.

struct Params {
    // xyz: ray origin, w: max travel distance.
    origin_maxdist: vec4<f32>,
    // xyz: ray direction (should be unit), w: unused.
    dir: vec4<f32>,
    // x: triangle count; y, z, w: padding.
    counts: vec4<u32>,
};

// Per-triangle output: (t, u, v, hit_flag). hit_flag is 1.0 on a hit, else 0.0.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<f32>>;

// Determinant magnitude below which the ray is parallel to the triangle plane.
const PARALLEL_EPS: f32 = 1e-8;

@compute @workgroup_size(64)
fn collider_trimesh_raycast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if (tri >= params.counts.x) {
        return;
    }

    let idx = indices[tri];
    let a = vertices[idx.x].xyz;
    let b = vertices[idx.y].xyz;
    let c = vertices[idx.z].xyz;

    let origin = params.origin_maxdist.xyz;
    let max_distance = params.origin_maxdist.w;
    let dir = params.dir.xyz;

    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = cross(dir, edge2);
    let det = dot(edge1, pvec);

    // Default to a miss; each early-out leaves this zeroed row in place.
    results[tri] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (abs(det) < PARALLEL_EPS) {
        return;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return;
    }
    let qvec = cross(tvec, edge1);
    let v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return;
    }
    let t = dot(edge2, qvec) * inv_det;
    if (t < 0.0 || t > max_distance) {
        return;
    }
    results[tri] = vec4<f32>(t, u, v, 1.0);
}
