// Per-strand curly-hair rest-state twin: evaluates on-device the same helical
// rest pose plus anisotropic bending compliance the CPU golden
// `prism_render_architecture::hair::rest_helix::build_rest_helix` computes, one
// thread per strand.
//
// Type-3/type-4 textured hair stores a helical rest shape (not a straight line)
// and bends anisotropically (tightening the curl along the fibre tangent is far
// cheaper than bending out of the coil plane). The CPU golden builds, per
// strand, the coil vertices, their per-segment rest lengths, the per-joint
// discrete `darboux` (curvature binormal) and the two per-segment compliances;
// this kernel mirrors that body value-for-value so a passing real-device parity
// test is direct evidence the ported kernel computes the same rest state, not
// merely that it compiles. Every strand in a dispatch shares one segment count
// (the fixed output stride) but carries its own root, tangent and coil params.
//
// Trig-free: the transverse offset is stepped around the coil cross-section by a
// unit-complex multiply `(x', y') = (x*cos - y*sin, x*sin + y*cos)`, exactly as
// the golden, so no `sin`/`cos`/`pow` is ever called. It uses only the portable
// core-WGSL subset (sqrt, dot, cross, clamp, multiply/add) — no optional device
// feature — so the twin runs unmodified on Metal, Vulkan and DX12. It is not
// bit-exact with the scalar CPU reference: a GPU may fuse a multiply-add,
// perturbing the low mantissa bits, so the parity test asserts a per-component
// tolerance rather than exact equality.
//
// Inputs are assumed finite and legal (radius >= 0, finite pitch, in-range
// ratio, segments pre-clamped on the host); the golden's NaN/infinity sanitation
// is idempotent on that domain, so the GPU twin mirrors the arithmetic path
// without a non-finite guard WGSL core lacks. The host still re-normalizes the
// rotation step and clamps the ratio exactly as the golden does.
//
// Provenance: standard discrete-elastic-rod helical rest construction plus wgpu
// compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of strands, bounding the dispatch.
    strand_count: u32,
    // Coil segments shared by every strand (the fixed output stride); the vertex
    // count is `segments + 1` and the interior-joint count is `segments - 1`.
    segments: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Strand-major inputs, stride 11 f32:
// [root.xyz, tangent.xyz, radius, pitch, rot_cos, rot_sin, ratio].
@group(0) @binding(1) var<storage, read> strands: array<f32>;
// Per-strand coil vertices, stride (segments + 1) * 3 f32 (xyz).
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
// Per-strand rest lengths, stride `segments` f32.
@group(0) @binding(3) var<storage, read_write> rest_lengths: array<f32>;
// Per-strand discrete curvature binormals, stride `segments` * 3 f32; only the
// first `segments - 1` joints are written (the last slot stays 0).
@group(0) @binding(4) var<storage, read_write> darboux: array<f32>;
// Per-strand compliances, stride 2 f32: (tangential, normal).
@group(0) @binding(5) var<storage, read_write> compliance: array<f32>;

// Squared-length degeneracy threshold, matching `rest_helix::NORMALIZE_EPS_SQ`.
fn normalize_eps_sq() -> f32 {
    return 1.0e-24;
}

// Mirrors `Vec3::normalize_or_zero`: unit vector unless the squared length is at
// or below `NORMALIZE_EPS_SQ`, in which case the zero vector is returned.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > normalize_eps_sq()) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Mirrors `rest_helix::build_orthonormal_basis`, returning the (normal,
// binormal) pair spanning the plane perpendicular to `axis`.
fn basis_normal(a_in: vec3<f32>) -> vec3<f32> {
    let fallback = vec3<f32>(0.0, 1.0, 0.0);
    var a = normalize_or_zero(a_in);
    if (dot(a, a) <= normalize_eps_sq()) {
        a = fallback;
    }
    var helper = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(a.x) < 0.9) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    var normal = normalize_or_zero(cross(a, helper));
    if (dot(normal, normal) <= normalize_eps_sq()) {
        normal = normalize_or_zero(cross(a, vec3<f32>(0.0, 0.0, 1.0)));
    }
    return normal;
}

fn sanitized_axis(tangent: vec3<f32>) -> vec3<f32> {
    let fallback = vec3<f32>(0.0, 1.0, 0.0);
    let t = normalize_or_zero(tangent);
    if (dot(t, t) > normalize_eps_sq()) {
        return t;
    }
    return fallback;
}

// Mirrors `rest_helix::curvature_binormal`.
fn curvature_binormal(e_prev: vec3<f32>, e_cur: vec3<f32>) -> vec3<f32> {
    let len_prev = sqrt(dot(e_prev, e_prev));
    let len_cur = sqrt(dot(e_cur, e_cur));
    let denom = len_prev * len_cur + dot(e_prev, e_cur);
    if (abs(denom) > normalize_eps_sq()) {
        return cross(e_prev, e_cur) * (2.0 / denom);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn load_vertex(base: u32, i: u32) -> vec3<f32> {
    let b = base + 3u * i;
    return vec3<f32>(positions[b], positions[b + 1u], positions[b + 2u]);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }

    let in_base = s * 11u;
    let root = vec3<f32>(strands[in_base], strands[in_base + 1u], strands[in_base + 2u]);
    let tangent = vec3<f32>(strands[in_base + 3u], strands[in_base + 4u], strands[in_base + 5u]);
    let radius_raw = strands[in_base + 6u];
    let pitch = strands[in_base + 7u];
    let cos_raw = strands[in_base + 8u];
    let sin_raw = strands[in_base + 9u];
    let ratio_raw = strands[in_base + 10u];

    // radius: finite and > 0 kept, otherwise 0 (degenerate straight hair).
    var radius = 0.0;
    if (radius_raw > 0.0) {
        radius = radius_raw;
    }

    // Re-normalize the unit-complex rotation step exactly as the golden's
    // `sanitize_unit_complex` (identity rotation when degenerate).
    var cos_step = 1.0;
    var sin_step = 0.0;
    let mag_sq = cos_raw * cos_raw + sin_raw * sin_raw;
    if (mag_sq > normalize_eps_sq()) {
        let inv = 1.0 / sqrt(mag_sq);
        cos_step = cos_raw * inv;
        sin_step = sin_raw * inv;
    }

    // ratio clamped into [RATIO_MIN, RATIO_MAX].
    let ratio = clamp(ratio_raw, 0.03125, 32.0);

    let axis = sanitized_axis(tangent);
    let basis_u = basis_normal(axis);
    let basis_v = cross(axis, basis_u);

    let segments = params.segments;
    let vertex_count = segments + 1u;
    let pos_base = s * vertex_count * 3u;

    var offset_x = radius;
    var offset_y = 0.0;
    for (var i: u32 = 0u; i < vertex_count; i = i + 1u) {
        let along = pitch * f32(i);
        let transverse = basis_u * offset_x + basis_v * offset_y;
        let vertex = root + axis * along + transverse;
        let b = pos_base + 3u * i;
        positions[b] = vertex.x;
        positions[b + 1u] = vertex.y;
        positions[b + 2u] = vertex.z;

        let next_x = offset_x * cos_step - offset_y * sin_step;
        let next_y = offset_x * sin_step + offset_y * cos_step;
        offset_x = next_x;
        offset_y = next_y;
    }

    let len_base = s * segments;
    for (var i: u32 = 0u; i < segments; i = i + 1u) {
        let edge = load_vertex(pos_base, i + 1u) - load_vertex(pos_base, i);
        rest_lengths[len_base + i] = sqrt(dot(edge, edge));
    }

    let dar_base = s * segments * 3u;
    if (segments >= 2u) {
        let joint_count = segments - 1u;
        for (var i: u32 = 0u; i < joint_count; i = i + 1u) {
            let e_prev = load_vertex(pos_base, i + 1u) - load_vertex(pos_base, i);
            let e_cur = load_vertex(pos_base, i + 2u) - load_vertex(pos_base, i + 1u);
            let kb = curvature_binormal(e_prev, e_cur);
            let b = dar_base + 3u * i;
            darboux[b] = kb.x;
            darboux[b + 1u] = kb.y;
            darboux[b + 2u] = kb.z;
        }
    }

    // Anisotropic compliance: normal = BASE_BEND_COMPLIANCE (1.0),
    // tangential = BASE_BEND_COMPLIANCE / ratio.
    let comp_base = 2u * s;
    compliance[comp_base] = 1.0 / ratio;
    compliance[comp_base + 1u] = 1.0;
}
