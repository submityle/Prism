// OBB-versus-halfspace narrow-phase contact kernel.
//
// One invocation per (box, plane) candidate couple. Each invocation reads an
// oriented bounding box (centre, three orthonormal local axes, and per-axis
// half extents) and a plane (unit outward normal and offset), runs the same
// support-function overlap test and manifold construction the CPU twin runs
// (`narrowphase/obb_halfspace.rs`), and writes one contact slot: the plane's
// outward normal (the direction that pushes the box out), the penetration
// depth, the deepest vertex projected onto the surface, and a validity flag. A
// couple whose box floats clear or only grazes the surface writes a zeroed slot
// with valid = 0, so the output index stays aligned with the input couple
// index.
//
// Geometry: a plane is a unit outward normal `n` and an offset `d`; the surface
// is `dot(n, x) = d` and the solid occupies `dot(n, x) <= d`. The box centre's
// signed distance to the surface is `s_c = dot(n, center) - d`; the support
// half-width along the normal is `proj = |dot(n, a0)|*he.x + |dot(n, a1)|*he.y
// + |dot(n, a2)|*he.z`; the deepest vertex's signed distance is
// `s_min = s_c - proj`. A strict `s_min < 0` means penetration by
// `depth = -s_min`. The deepest vertex steps along each axis toward the solid
// (`sign_i = -1` when `dot(n, axis_i) >= 0`, else `+1`), and the reported point
// is that vertex projected back onto the surface.
//
// The arithmetic is bit-for-bit with the twin: three dot products, the centre
// distance, the support projection with `abs`, the strict rejection, the
// per-axis sign choice, the deepest-vertex accumulation, and the surface
// projection, with no square root or reciprocal on this path, so the parity
// test matches the validity flag exactly and the normal, depth, and point to
// within the tightest float tolerance.
//
// Single-point manifold: this slice reports the single deepest vertex per
// couple, matching the one-contact-per-pair architecture the sibling slices
// establish. That is a deliberate, honest simplification (the deepest vertex is
// the worst-case point a solver resolves first), not a stub; a full multi-point
// resting manifold is a separate follow-up slice.
//
// Provenance: textbook OBB-versus-halfspace (support-function) collision
// manifold. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// One oriented bounding box. The half extents ride in the w lanes of the axis
// rows to keep the box to four vec4s; `center.w` is unused padding.
struct Obb {
    center: vec4<f32>,
    axis0: vec4<f32>,
    axis1: vec4<f32>,
    axis2: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Planes: xyz outward unit normal, w offset.
@group(0) @binding(2) var<storage, read> planes: array<vec4<f32>>;
// Candidate couples, one (box, plane) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per couple.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn narrowphase_obb_halfspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let couple = pairs[i];
    let box_ = boxes[couple.x];
    let plane = planes[couple.y];

    let bc = box_.center.xyz;
    let a0 = box_.axis0.xyz;
    let a1 = box_.axis1.xyz;
    let a2 = box_.axis2.xyz;
    let he = vec3<f32>(box_.axis0.w, box_.axis1.w, box_.axis2.w);
    let n = plane.xyz;
    let d = plane.w;

    var out: Contact;

    // Projection of each box axis onto the plane normal.
    let d0 = dot(n, a0);
    let d1 = dot(n, a1);
    let d2 = dot(n, a2);

    // Signed distance of the box centre from the surface.
    let s_c = dot(n, bc) - d;
    // Support half-width of the box along the normal.
    let proj = abs(d0) * he.x + abs(d1) * he.y + abs(d2) * he.z;
    // Signed distance of the deepest vertex from the surface.
    let s_min = s_c - proj;

    // Strict overlap: the deepest vertex must cross into the solid.
    if (s_min >= 0.0) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    // Deepest vertex: step along each axis toward the solid side of the plane.
    let sign0 = select(1.0, -1.0, d0 >= 0.0);
    let sign1 = select(1.0, -1.0, d1 >= 0.0);
    let sign2 = select(1.0, -1.0, d2 >= 0.0);
    let deepest = bc + a0 * (sign0 * he.x) + a1 * (sign1 * he.y) + a2 * (sign2 * he.z);

    let depth = -s_min;
    // Deepest vertex projected onto the surface: the resting contact point.
    let s_deepest = dot(n, deepest) - d;
    let point = deepest - n * s_deepest;

    out.normal_depth = vec4<f32>(n, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
