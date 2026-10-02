// Sphere-versus-triangle narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads a sphere (centre and
// radius) and a triangle (three world-space vertices), runs the same closest-
// point query and manifold construction the CPU twin runs
// (`narrowphase/sphere_triangle.rs`), and writes one contact slot: a unit normal
// pointing from the triangle toward the sphere (the push-out direction), the
// penetration depth, the world contact point on the triangle, and a validity
// flag. A pair that does not penetrate writes a zeroed slot with valid = 0, so
// the output index stays aligned with the input pair index.
//
// Geometry: find the closest point `q` on the solid triangle to the sphere
// centre `p` with Ericson's Voronoi-region cascade, then inspect `diff = p - q`.
// When the centre is off the triangle (`dot(diff, diff) > COINCIDENT_EPS2`) the
// pair contacts only for a strict `dist < r`; the normal is `diff / dist` and
// depth `r - dist`, with the contact point at `q`. When the centre lies on the
// triangle, the offset is degenerate, so the normal falls back to the geometric
// face normal `normalize((b - a) x (c - a))`, the depth is the full radius, and
// the point is `q`.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and the
// handful of barycentric reciprocals in the closest-point clamps, whose WGSL
// rounding differs in the low bits; the parity test therefore matches the
// validity flag exactly and the normal, depth, and point within a tight
// tolerance.
//
// Provenance: closest-point-on-triangle is the Voronoi-region method from
// Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
// sphere manifold is textbook. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which the offset from the nearest triangle
// point to the sphere centre is treated as zero, i.e. the centre lies on the
// triangle and the face-normal fallback runs; kept identical to
// `COINCIDENT_EPS2` in `narrowphase/sphere_triangle.rs`.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// One triangle collider. Each vertex rides in the xyz of its row; the w lanes
// are unused padding to keep the triangle to three vec4s.
struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Spheres in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// Triangles, three vec4s each.
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
// Candidate pairs, one (sphere, triangle) index couple each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// `closest_point_on_triangle`, including the fixed comparison order and the
// barycentric reciprocals.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    // Vertex region outside A.
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    // Vertex region outside B.
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    // Vertex region outside C.
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    // Interior face region.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

@compute @workgroup_size(64)
fn narrowphase_sphere_triangle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let sphere = spheres[pair.x];
    let p = sphere.xyz;
    let r = sphere.w;

    let tri = triangles[pair.y];
    let a = tri.a.xyz;
    let b = tri.b.xyz;
    let c = tri.c.xyz;

    let q = closest_point_on_triangle(p, a, b, c);
    let diff = p - q;
    let d2 = dot(diff, diff);

    var out: Contact;

    if (d2 > COINCIDENT_EPS2) {
        // Centre off the triangle: nearest feature is the closest point q.
        let dist = sqrt(d2);
        if (dist >= r) {
            // Strict overlap: a grazing sphere carries no penetration.
            out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            contacts[i] = out;
            return;
        }
        let normal = diff / dist;
        let depth = r - dist;
        out.normal_depth = vec4<f32>(normal, depth);
        out.point_valid = vec4<f32>(q, 1.0);
        contacts[i] = out;
        return;
    }

    // Centre is on the triangle: fall back to the geometric face normal.
    let raw = cross(b - a, c - a);
    let len2 = dot(raw, raw);
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (len2 > COINCIDENT_EPS2) {
        normal = raw / sqrt(len2);
    }
    out.normal_depth = vec4<f32>(normal, r);
    out.point_valid = vec4<f32>(q, 1.0);
    contacts[i] = out;
}
