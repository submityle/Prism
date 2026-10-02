// Closest-point-versus-triangle-mesh scene query kernel: one lane finds the
// point on one triangle nearest the query via Ericson's Voronoi-region cascade
// (Real-Time Collision Detection, 2005, section 5.1.5) and writes that point
// plus its distance to the query. The host reduces the per-triangle results to
// the nearest point and finalises distance, barycentric weights, and the
// query-facing normal with the shared CPU rule, so the device result matches
// the CPU brute golden operation for operation (up to the barycentric
// reciprocals' floating-point tolerance). This mirrors
// collider/trimesh_closest_point.rs exactly.
//
// Provenance: Ericson Voronoi-region closest point on triangle (2005). No
// Unreal Engine source or derived code.

struct Params {
    // xyz: query point; w: unused.
    query: vec4<f32>,
    // x: triangle count; y, z, w: padding.
    counts: vec4<u32>,
};

// Per-triangle output: (qx, qy, qz, distance) where q is the nearest point on
// the triangle and distance is |query - q|.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<f32>>;

// Returns the point on triangle (a, b, c) closest to p.
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

@compute @workgroup_size(64)
fn collider_trimesh_closest_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if (tri >= params.counts.x) {
        return;
    }

    let idx = indices[tri];
    let a = vertices[idx.x].xyz;
    let b = vertices[idx.y].xyz;
    let c = vertices[idx.z].xyz;

    let p = params.query.xyz;
    let q = closest_point_on_triangle(p, a, b, c);
    let dist = length(p - q);
    results[tri] = vec4<f32>(q.x, q.y, q.z, dist);
}
