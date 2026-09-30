// OBB-versus-halfspace incident-face contact-manifold kernel.
//
// One invocation per candidate (box, plane) couple. Each invocation runs the
// same support-function overlap gate the single-point kernel runs
// (narrowphase_obb_halfspace.wgsl), then promotes that one contact to the full
// incident-face manifold exactly as the CPU twin does
// (narrowphase/obb_halfspace_manifold.rs): it finds the box face whose axis is
// most aligned with the plane normal (lower index winning a tie), steps that
// axis to the solid side, sweeps the two remaining axes over their half
// extents, and keeps every one of the face's four corners that has crossed the
// surface.
//
// The output is one fixed-stride manifold record per couple: the shared normal
// (the plane's outward push-out direction) with the live point count in its w
// lane, then four (position.xyz, depth) points. A box that floats clear or only
// grazes the surface (`s_min >= 0`) writes count = 0. A penetrating box always
// yields at least one corner, so a corner-first poke collapses honestly to
// count = 1, an edge-first landing to count = 2, and a flush rest to the full
// count = 4.
//
// The arithmetic is operation-for-operation with the twin: the same three axis
// projections, the same centre distance and support projection with `abs`, the
// same strict `s_min >= 0` rejection, the same incident-axis argmax with a
// lower-index tie-break, the same solid-side sign, and the same nested
// (su outer, sv inner) corner sweep with the identical `s < 0` keep test and
// `position = vertex - n * s` surface projection. No square root or reciprocal
// is on this path, so the parity test matches the count exactly and the normal,
// positions, and depths to within the tightest float tolerance.
//
// Provenance: textbook oriented-bounding-box-versus-halfspace incident-face
// clipping (the standard box-on-plane resting manifold). No Unreal Engine
// source or derived code.

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

// One output manifold: (normal.xyz, count) then four (position.xyz, depth)
// points. Only the first `count` points are live.
struct Manifold {
    normal_count: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Planes: xyz outward unit normal, w offset.
@group(0) @binding(2) var<storage, read> planes: array<vec4<f32>>;
// Candidate couples, one (box, plane) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one fixed-stride record per couple.
@group(0) @binding(4) var<storage, read_write> manifolds: array<Manifold>;

@compute @workgroup_size(64)
fn narrowphase_obb_halfspace_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let couple = pairs[i];
    let box_ = boxes[couple.x];
    let plane = planes[couple.y];

    let bc = box_.center.xyz;
    // Function-scope arrays so the incident axis can be selected by dynamic
    // index while staying addressable.
    var axes = array<vec3<f32>, 3>(box_.axis0.xyz, box_.axis1.xyz, box_.axis2.xyz);
    var he = array<f32, 3>(box_.axis0.w, box_.axis1.w, box_.axis2.w);
    let n = plane.xyz;
    let offset = plane.w;

    // Projection of each box axis onto the plane normal.
    var d = array<f32, 3>(dot(n, axes[0]), dot(n, axes[1]), dot(n, axes[2]));

    // Signed distance of the box centre from the surface and the box's support
    // half-width along the normal: the deepest vertex sits at `s_c - proj`.
    let s_c = dot(n, bc) - offset;
    let proj = abs(d[0]) * he[0] + abs(d[1]) * he[1] + abs(d[2]) * he[2];
    let s_min = s_c - proj;

    var out: Manifold;

    // Strict overlap: the deepest vertex must cross into the solid. A box that
    // only grazes the surface carries no penetration.
    if (s_min >= 0.0) {
        out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[i] = out;
        return;
    }

    // Incident face: the axis most aligned with the plane normal, lower index
    // winning a tie (strict `>` so ties keep the earlier axis).
    let ad0 = abs(d[0]);
    let ad1 = abs(d[1]);
    let ad2 = abs(d[2]);
    var k = 0u;
    var best = ad0;
    if (ad1 > best) {
        best = ad1;
        k = 1u;
    }
    if (ad2 > best) {
        k = 2u;
    }

    // The two axes swept over the face, kept in ascending index order so the
    // device and CPU enumerate the four corners identically.
    var u = 1u;
    var v = 2u;
    if (k == 1u) {
        u = 0u;
        v = 2u;
    } else if (k == 2u) {
        u = 0u;
        v = 1u;
    }

    // Step axis `k` toward the solid side (opposite the normal projection) to
    // reach the incident face; the other two axes stay free to sweep.
    var sign_k = 1.0;
    if (d[k] >= 0.0) {
        sign_k = -1.0;
    }
    let base = bc + axes[k] * (sign_k * he[k]);

    // Sweep the two free axes over `±half_extent` in a fixed order (su outer, sv
    // inner) and keep every corner that has crossed the surface.
    var signs = array<f32, 2>(-1.0, 1.0);
    var count = 0u;
    var pts = array<vec4<f32>, 4>(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
    for (var iu = 0u; iu < 2u; iu = iu + 1u) {
        let su = signs[iu];
        for (var iv = 0u; iv < 2u; iv = iv + 1u) {
            let sv = signs[iv];
            let vertex = base + axes[u] * (su * he[u]) + axes[v] * (sv * he[v]);
            let s = dot(n, vertex) - offset;
            if (s < 0.0) {
                let position = vertex - n * s;
                pts[count] = vec4<f32>(position, -s);
                count = count + 1u;
            }
        }
    }

    // The deepest vertex is one of the four swept corners and penetrates by
    // construction, so `count` is always in `1..=4` here.
    out.normal_count = vec4<f32>(n, f32(count));
    out.p0 = pts[0];
    out.p1 = pts[1];
    out.p2 = pts[2];
    out.p3 = pts[3];
    manifolds[i] = out;
}
