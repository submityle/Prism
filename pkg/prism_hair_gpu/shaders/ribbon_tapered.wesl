// Tapered strand-to-ribbon (Cards LOD proxy) twin: evaluates on-device the same
// view-independent ribbon meshing the CPU golden `build_ribbon_tapered`
// computes, one thread per strand.
//
// This is the tapered sibling of `ribbon.wesl`. The plain ribbon offsets each
// control point by an externally supplied per-point radius; the tapered ribbon
// instead derives that radius from the authored groom so the card narrows from
// root to tip exactly as it was authored. At control point `i` the half-width
// is `StrandAttributes::radius_at(t)` evaluated at the point's normalized
// arc-length position `t = i / (n - 1)` (`0` = root, `1` = tip). `radius_at` is
// a clamped linear ramp: `t` is clamped to `0..=1` (a no-op here since
// `i / (n - 1)` already lands in that range) and the radius is
// `root_radius + (tip_radius - root_radius) * t`. Everything else mirrors
// `ribbon.wesl` bit-for-bit. The CPU golden is `build_ribbon_tapered` (in
// `prism_render_architecture::hair::ribbon`), which derives the per-point radii
// this way and defers to `build_ribbon`.
//
// One thread per strand walks the strand root->tip, first summing segment
// lengths for the arc-length `v` coordinate, then emitting two edge vertices per
// control point offset +/- radius along the rotation-minimizing bitangent, with
// the tangent carried through for anisotropic shading. Strands are independent,
// so the batch is embarrassingly parallel. Every guard is reproduced: a strand
// with fewer than two points writes nothing (an empty mesh), and a zero-length
// strand (all points coincident, total arc length 0) falls back to even
// parameter spacing for `v` rather than dividing by zero.
//
// It uses only the portable core-WGSL subset (sqrt via length / mul / add) - no
// exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12. It is not bit-exact with the scalar CPU reference: a GPU may
// fuse a multiply-add (the radius lerp and edge offset are prime candidates) or
// reassociate the arc-length accumulation, perturbing the low mantissa bits. The
// parity test therefore asserts a per-component tolerance rather than exact
// equality.
//
// Provenance: standard view-independent ribbon/card meshing with an authored
// linear radius taper; no Unreal Engine source or derived code.

struct Params {
    // Number of strands in `strands`.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One strand descriptor. `32`-byte stride matching the `bytemuck` upload struct
// `TaperedRibbonStrand`: the base index of this strand's first control point in
// the flat `attrs` array (stride `9` f32/point), its point count, the base index
// of its first output vertex in the flat `verts` array (stride `8` f32/vertex,
// two vertices per control point), and the authored root / tip radii the
// half-width tapers between.
struct Strand {
    point_offset: u32,
    point_count: u32,
    vertex_offset: u32,
    pad0: u32,
    root_radius: f32,
    tip_radius: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Interleaved per-point attributes, stride 9 f32:
// position xyz, tangent xyz, bitangent xyz.
@group(0) @binding(2) var<storage, read> attrs: array<f32>;
// Interleaved per-vertex output, stride 8 f32: position xyz, tangent xyz, uv.
@group(0) @binding(3) var<storage, read_write> verts: array<f32>;

fn attr_base(k: u32) -> u32 {
    return 9u * k;
}

fn load_point(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b], attrs[b + 1u], attrs[b + 2u]);
}

fn load_tangent(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b + 3u], attrs[b + 4u], attrs[b + 5u]);
}

fn load_bitangent(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b + 6u], attrs[b + 7u], attrs[b + 8u]);
}

// Writes output vertex `vi`: position xyz, tangent xyz, then the (u, v) pair.
fn store_vertex(vi: u32, pos: vec3<f32>, tan: vec3<f32>, u: f32, v: f32) {
    let b = 8u * vi;
    verts[b] = pos.x;
    verts[b + 1u] = pos.y;
    verts[b + 2u] = pos.z;
    verts[b + 3u] = tan.x;
    verts[b + 4u] = tan.y;
    verts[b + 5u] = tan.z;
    verts[b + 6u] = u;
    verts[b + 7u] = v;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let strand = strands[s];
    let offset = strand.point_offset;
    let n = strand.point_count;
    let voff = strand.vertex_offset;
    // Fewer than two points: no ribbon, matching the golden's `n < 2` guard.
    if (n < 2u) {
        return;
    }

    let root_radius = strand.root_radius;
    let tip_radius = strand.tip_radius;

    // Total arc length for the `v` coordinate (same left-to-right accumulation
    // order as the golden so the tip `v` lands exactly on 1).
    var total = 0.0;
    for (var j: u32 = 1u; j < n; j = j + 1u) {
        total = total + length(load_point(offset + j) - load_point(offset + j - 1u));
    }

    var cumulative = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        if (i > 0u) {
            cumulative = cumulative + length(load_point(offset + i) - load_point(offset + i - 1u));
        }
        let p = load_point(offset + i);
        let tan = load_tangent(offset + i);

        // Authored linear radius taper: radius_at(i / (n - 1)).
        let t = clamp(f32(i) / f32(n - 1u), 0.0, 1.0);
        let r = root_radius + (tip_radius - root_radius) * t;
        let off_v = load_bitangent(offset + i) * r;

        var v: f32;
        if (total > 0.0) {
            v = cumulative / total;
        } else {
            // Zero-length strand: even parameter spacing, matching the golden.
            v = f32(i) / f32(n - 1u);
        }

        let vi = voff + 2u * i;
        store_vertex(vi, p - off_v, tan, 0.0, v);
        store_vertex(vi + 1u, p + off_v, tan, 1.0, v);
    }
}
