// Sphere-versus-heightfield narrow-phase contact kernel.
//
// One invocation per candidate (sphere, heightfield) pair. Each invocation
// reads a sphere (centre and radius) and a heightfield (a regular rows x cols
// grid of height samples spaced cell_size apart on the XZ plane, Y up), finds
// the grid cells overlapping the sphere's XZ footprint, rebuilds each candidate
// cell's two triangles, runs the same closest-point query and manifold
// construction the CPU twin runs (`narrowphase/heightfield.rs`), and keeps the
// deepest penetrating contact. It writes one contact slot: a unit normal
// pointing from the terrain toward the sphere (the push-out direction), the
// penetration depth, the world contact point, and a validity flag. A pair that
// touches no cell triangle writes a zeroed slot with valid = 0, so the output
// index stays aligned with the input pair index.
//
// Geometry: the sample at grid index (r, c) is the world point
// origin + (c * cell, height[r][c], r * cell). Cell (r, c) is bounded by the
// samples (r,c), (r,c+1), (r+1,c), (r+1,c+1), split along the (r,c)-to-(r+1,c+1)
// diagonal into [v00, v10, v11] and [v00, v11, v01], wound so a flat field faces
// +Y. For each candidate triangle, find the closest point q on the solid
// triangle to the sphere centre p with Ericson's Voronoi-region cascade, then
// inspect diff = p - q: off the triangle (dot(diff,diff) > COINCIDENT_EPS2) a
// strict dist < r contacts with normal diff/dist and depth r - dist at q; on the
// triangle the normal falls back to the geometric face normal with the full
// radius depth. The reduction keeps the contact with the greatest depth, broken
// by iteration order (row outer, column inner, first triangle first), replacing
// the running best only on a strictly greater depth so the first deepest wins.
//
// The arithmetic is bit-for-bit with the twin apart from the square root, the
// floor, and the barycentric reciprocals, whose WGSL rounding differs in the low
// bits; the parity test matches the validity flag exactly and the normal, depth,
// and point within a tight tolerance.
//
// Provenance: closest-point-on-triangle is the Voronoi-region method from
// Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
// heightfield cell triangulation is textbook. No Unreal Engine source or derived
// code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which the offset from the nearest triangle
// point to the sphere centre is treated as zero (the centre lies on the
// triangle and the face-normal fallback runs); kept identical to
// `COINCIDENT_EPS2` in `narrowphase/sphere_triangle.rs`.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// One heightfield's metadata. `dims` is (rows, cols, heights_offset, pad);
// `geom` is (cell_size, origin.x, origin.y, origin.z). The field's samples live
// in the shared `heights` array starting at `heights_offset`, row-major.
struct Field {
    dims: vec4<u32>,
    geom: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Spheres in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// Heightfield metadata, one Field each.
@group(0) @binding(2) var<storage, read> fields: array<Field>;
// Concatenated row-major height samples for every field.
@group(0) @binding(3) var<storage, read> heights: array<f32>;
// Candidate pairs, one (sphere, field) index couple each.
@group(0) @binding(4) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(5) var<storage, read_write> contacts: array<Contact>;

// World-space position of a field's sample at grid index (r, c), matching
// `Heightfield::vertex`.
fn field_vertex(fi: u32, r: u32, c: u32) -> vec3<f32> {
    let f = fields[fi];
    let cols = f.dims.y;
    let off = f.dims.z;
    let h = heights[off + r * cols + c];
    let cell = f.geom.x;
    return vec3<f32>(
        f.geom.y + f32(c) * cell,
        f.geom.z + h,
        f.geom.w + f32(r) * cell,
    );
}

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// `closest_point_on_triangle`, including the fixed comparison order and the
// barycentric reciprocals.
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

// Floors `value` to a cell index and clamps it to [0, last], matching the CPU
// twin's `clamp_index`.
fn clamp_index(value: f32, last: u32) -> u32 {
    let floored = floor(value);
    if (floored <= 0.0) {
        return 0u;
    }
    if (floored >= f32(last)) {
        return last;
    }
    return u32(floored);
}

@compute @workgroup_size(64)
fn narrowphase_sphere_heightfield(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let sphere = spheres[pair.x];
    let p = sphere.xyz;
    let r = sphere.w;

    let fi = pair.y;
    let f = fields[fi];
    let rows = f.dims.x;
    let cols = f.dims.y;
    let cell = f.geom.x;
    let ox = f.geom.y;
    let oz = f.geom.w;

    var out: Contact;
    out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // No complete cell, or the footprint lies wholly off the grid: no contact.
    if (rows < 2u || cols < 2u) {
        contacts[i] = out;
        return;
    }
    let span_x = f32(cols - 1u) * cell;
    let span_z = f32(rows - 1u) * cell;
    let min_x = p.x - r;
    let max_x = p.x + r;
    let min_z = p.z - r;
    let max_z = p.z + r;
    if (max_x < ox || min_x > ox + span_x || max_z < oz || min_z > oz + span_z) {
        contacts[i] = out;
        return;
    }

    let last_col = cols - 2u;
    let last_row = rows - 2u;
    let inv = 1.0 / cell;
    let min_col = clamp_index((min_x - ox) * inv, last_col);
    let max_col = clamp_index((max_x - ox) * inv, last_col);
    let min_row = clamp_index((min_z - oz) * inv, last_row);
    let max_row = clamp_index((max_z - oz) * inv, last_row);

    var best_depth = -1.0;
    var found = false;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var best_point = vec3<f32>(0.0, 0.0, 0.0);

    for (var row = min_row; row <= max_row; row = row + 1u) {
        for (var col = min_col; col <= max_col; col = col + 1u) {
            let v00 = field_vertex(fi, row, col);
            let v01 = field_vertex(fi, row, col + 1u);
            let v10 = field_vertex(fi, row + 1u, col);
            let v11 = field_vertex(fi, row + 1u, col + 1u);

            // The cell's two triangles share the (r,c)-to-(r+1,c+1) diagonal.
            for (var t = 0u; t < 2u; t = t + 1u) {
                var a = v00;
                var b = v10;
                var c = v11;
                if (t == 1u) {
                    a = v00;
                    b = v11;
                    c = v01;
                }

                let q = closest_point_on_triangle(p, a, b, c);
                let diff = p - q;
                let d2 = dot(diff, diff);

                var hit = false;
                var normal = vec3<f32>(0.0, 0.0, 0.0);
                var depth = 0.0;

                if (d2 > COINCIDENT_EPS2) {
                    let dist = sqrt(d2);
                    if (dist < r) {
                        normal = diff / dist;
                        depth = r - dist;
                        hit = true;
                    }
                } else {
                    // Centre on the triangle: fall back to the face normal.
                    let raw = cross(b - a, c - a);
                    let len2 = dot(raw, raw);
                    if (len2 > COINCIDENT_EPS2) {
                        normal = raw / sqrt(len2);
                    } else {
                        normal = vec3<f32>(1.0, 0.0, 0.0);
                    }
                    depth = r;
                    hit = true;
                }

                if (hit && (!found || depth > best_depth)) {
                    best_depth = depth;
                    best_normal = normal;
                    best_point = q;
                    found = true;
                }
            }
        }
    }

    if (found) {
        out.normal_depth = vec4<f32>(best_normal, best_depth);
        out.point_valid = vec4<f32>(best_point, 1.0);
    }
    contacts[i] = out;
}
