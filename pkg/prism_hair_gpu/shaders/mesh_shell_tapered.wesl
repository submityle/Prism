// Tapered strand-to-shell (Mesh LOD proxy) twin: evaluates on-device the same
// solid shell-tube meshing the CPU golden `build_shell_tapered` computes, one
// thread per strand.
//
// This is the tapered sibling of `mesh_shell.wesl`. The plain shell sweeps a
// uniform rectangular cross-section along the centerline; the tapered shell
// instead narrows the section from root to tip exactly as the groom was
// authored. At ring `i` the section is a square whose half-width and
// half-thickness are both `StrandAttributes::radius_at(t)` evaluated at the
// ring's normalized arc-length position `t = i / (n - 1)` (`0` = root, `1` =
// tip). `radius_at` is a clamped linear ramp: `t` is clamped to `0..=1` (a
// no-op here since `i / (n - 1)` already lands in that range) and the radius is
// `root_radius + (tip_radius - root_radius) * t`. The CPU golden for that
// meshing is `build_shell_tapered` (in
// `prism_render_architecture::hair::mesh_shell`).
//
// Everything else mirrors `mesh_shell.wesl` bit-for-bit: one thread per strand
// walks the strand root->tip, first summing segment lengths for the arc-length
// `v` coordinate, then emitting four ring vertices per control point. Corner
// `j` sits at `sign_w * bitangent * r + sign_h * normal * r` around the
// centerline (the section is square, so width and thickness share the radius
// `r`), its outward normal is the (normalized) `sign_w * bitangent + sign_h *
// normal` diagonal, and its uv is `(j * 0.25, v)`. Strands are independent, so
// the batch is embarrassingly parallel. Every guard is reproduced: a strand
// with fewer than two points writes nothing (an empty mesh), and a zero-length
// strand (all points coincident, total arc length 0) falls back to even
// parameter spacing for `v` rather than dividing by zero.
//
// It uses only the portable core-WGSL subset (sqrt via length / mul / add) - no
// exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12. It is not bit-exact with the scalar CPU reference: a GPU may
// fuse a multiply-add (the ring radius lerp and corner offset are prime
// candidates) or reassociate the arc-length accumulation, perturbing the low
// mantissa bits. The parity test therefore asserts a per-component tolerance
// rather than exact equality. The triangle-list indices carry no float error
// (a pure function of the point count) and are rebuilt on the host.
//
// Provenance: standard rectangular shell-tube meshing with an authored linear
// radius taper; no Unreal Engine source or derived code.

struct Params {
    // Number of strands in `strands`.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One strand descriptor. `32`-byte stride matching the `bytemuck` upload struct
// `TaperedShellStrand`: the base index of this strand's first control point in
// the flat `attrs` array (stride `9` f32/point), its point count, the base
// index of its first output vertex in the flat `verts` array (stride `8`
// f32/vertex, four vertices per control point), and the authored root / tip
// radii the square cross-section tapers between.
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
// position xyz, bitangent xyz, normal xyz.
@group(0) @binding(2) var<storage, read> attrs: array<f32>;
// Interleaved per-vertex output, stride 8 f32: position xyz, normal xyz, uv.
@group(0) @binding(3) var<storage, read_write> verts: array<f32>;

fn attr_base(k: u32) -> u32 {
    return 9u * k;
}

fn load_point(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b], attrs[b + 1u], attrs[b + 2u]);
}

fn load_bitangent(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b + 3u], attrs[b + 4u], attrs[b + 5u]);
}

fn load_normal(k: u32) -> vec3<f32> {
    let b = attr_base(k);
    return vec3<f32>(attrs[b + 6u], attrs[b + 7u], attrs[b + 8u]);
}

// Clamped linear radius ramp, mirroring the golden `StrandAttributes::radius_at`:
// `t` is clamped to `0..=1`, then the radius is `root + (tip - root) * t`.
fn radius_at(root_radius: f32, tip_radius: f32, t: f32) -> f32 {
    let tc = clamp(t, 0.0, 1.0);
    return root_radius + (tip_radius - root_radius) * tc;
}

// Unit vector, or `fallback` when `v` is (near) zero length. Mirrors the
// golden `Vec3::normalize_or`: the guard compares the squared length against
// `f32::EPSILON` (1.1920929e-7) and divides by `sqrt(len_sq)`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1.1920929e-7) {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

// Writes output vertex `vi`: position xyz, normal xyz, then the (u, v) pair.
fn store_vertex(vi: u32, pos: vec3<f32>, nrm: vec3<f32>, u: f32, v: f32) {
    let b = 8u * vi;
    verts[b] = pos.x;
    verts[b + 1u] = pos.y;
    verts[b + 2u] = pos.z;
    verts[b + 3u] = nrm.x;
    verts[b + 4u] = nrm.y;
    verts[b + 5u] = nrm.z;
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
    let root_radius = strand.root_radius;
    let tip_radius = strand.tip_radius;
    // Fewer than two points: no shell, matching the golden's `n < 2` guard.
    if (n < 2u) {
        return;
    }

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
        let center = load_point(offset + i);
        let bitangent = load_bitangent(offset + i);
        let nrm = load_normal(offset + i);

        // Square section radius at this ring's normalized arc position.
        let t = f32(i) / f32(n - 1u);
        let r = radius_at(root_radius, tip_radius, t);

        var v: f32;
        if (total > 0.0) {
            v = cumulative / total;
        } else {
            // Zero-length strand: even parameter spacing, matching the golden.
            v = f32(i) / f32(n - 1u);
        }

        let ring = voff + 4u * i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            // CORNER_SIGNS = [(1,1), (-1,1), (-1,-1), (1,-1)]: sign_w is +1 for
            // corners 0 and 3, sign_h is +1 for corners 0 and 1.
            let sign_w = select(-1.0, 1.0, (j == 0u) || (j == 3u));
            let sign_h = select(-1.0, 1.0, (j == 0u) || (j == 1u));
            let pos = center
                + bitangent * (sign_w * r)
                + nrm * (sign_h * r);
            let outward = bitangent * sign_w + nrm * sign_h;
            let out_n = normalize_or(outward, nrm);
            store_vertex(ring + j, pos, out_n, f32(j) * 0.25, v);
        }
    }
}
