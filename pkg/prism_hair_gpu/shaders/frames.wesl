// Rotation-minimizing strand-frame twin: evaluates on-device the same
// double-reflection frame transport the CPU golden `build_strand_frames`
// computes, one thread per strand.
//
// A render strand is a polyline, but ribbon/card expansion and anisotropic
// shading (Kajiya-Kay / Marschner / Chiang) need a coherent orthonormal frame
// at every control point, not just a tangent. Production hair engines (UE5
// Groom, AMD TressFX) carry one reference normal along the strand with minimal
// twist rather than rebuilding an arbitrary basis per point (which flips and
// shimmers). The CPU golden for that transport is `build_strand_frames` (in
// `prism_render_architecture::hair::frames`), implementing the double-reflection
// rotation-minimizing frame (Wang et al. 2008): seed one orthonormal frame at
// the root, then propagate the reference normal point-to-point with two
// reflections so it never rotates about the tangent more than the curve
// geometry forces.
//
// This kernel mirrors that reference exactly, one thread per strand walking the
// strand root->tip in sequence (strands are independent, so the batch is
// embarrassingly parallel). Every guard is reproduced: an empty strand writes
// nothing, a single point yields one frame around the fallback tangent, and
// coincident points reuse the previous reference direction rather than dividing
// by zero. It uses only the portable core-WGSL subset (sqrt/dot/cross/mul/add)
// - no exp, no pow, no optional feature - so this twin runs unmodified on
// Metal, Vulkan and DX12.
//
// It is not bit-exact with the scalar CPU reference: a GPU may fuse a
// multiply-add or reassociate a dot product the reference leaves separate,
// perturbing the low mantissa bits. The parity test therefore asserts a
// per-component tolerance rather than exact equality, tight enough that a wrong
// port (a swapped reflection, a missing re-orthonormalization, a sign error)
// fails, loose enough that legal fma contraction passes. Test inputs stay well
// away from the degenerate `len_sq <= f32::EPSILON` thresholds so CPU and GPU
// never diverge on a branch.
//
// Provenance: standard double-reflection rotation-minimizing frame; no Unreal
// Engine source or derived code.

// Matches `f32::EPSILON` (2^-23), the zero-length / coincident-point guard the
// CPU golden uses in `Vec3::normalize_or` and the reflection denominators.
const F32_EPSILON: f32 = 1.1920928955078125e-7;

// Fallback tangent for degenerate strands (single point or coincident points),
// matching `frames::FALLBACK_TANGENT`: an arbitrary but fixed unit direction.
const FALLBACK_TANGENT: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

// Fallback used when a reference/bitangent cross product collapses, matching the
// `Vec3::new(0.0, 0.0, 1.0)` fallbacks in the CPU golden.
const FALLBACK_AXIS: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

struct Params {
    // Number of strands in `strands`.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One strand descriptor. `16`-byte stride matching the `bytemuck` upload struct
// `FrameStrand`: the base index of this strand's first point (and first frame)
// in the flat `points` / `frames` arrays, then the point count.
struct Strand {
    point_offset: u32,
    point_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Flat f32 triples: point `k` is `(points[3k], points[3k+1], points[3k+2])`.
@group(0) @binding(2) var<storage, read> points: array<f32>;
// Flat f32 nine-tuples: frame `k` is tangent xyz, normal xyz, bitangent xyz.
@group(0) @binding(3) var<storage, read_write> frames: array<f32>;

// Unit vector along `v`, or `fallback` when `v` is (numerically) zero, mirroring
// `Vec3::normalize_or` term order (`v * (1 / sqrt(len_sq))`) and its
// `len_sq <= f32::EPSILON` guard.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= F32_EPSILON) {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

// Unit reference normal perpendicular to `tangent`, mirroring
// `frames::orthonormal_reference`: pick whichever cardinal axis is least aligned
// with the tangent as a helper, cross, then normalize.
fn orthonormal_reference(tangent: vec3<f32>) -> vec3<f32> {
    let helper = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(tangent.x) < 0.9,
    );
    return normalize_or(cross(tangent, helper), FALLBACK_AXIS);
}

// Reflects `v` in the plane whose normal is `axis` (with `c = axis . axis`),
// mirroring `frames::reflect`: `v - (2 * (axis . v) / c) * axis`.
fn reflect_plane(v: vec3<f32>, axis: vec3<f32>, c: f32) -> vec3<f32> {
    return v - axis * (2.0 * dot(axis, v) / c);
}

// Loads control point `k` (a global point index) from the flat buffer.
fn load_point(k: u32) -> vec3<f32> {
    let base = 3u * k;
    return vec3<f32>(points[base], points[base + 1u], points[base + 2u]);
}

// Forward-difference unit tangent at local index `i` of the strand whose points
// start at `offset` and run for `count`, mirroring `frames::strand_tangents`.
fn strand_tangent(offset: u32, count: u32, i: u32) -> vec3<f32> {
    var raw: vec3<f32>;
    if (count <= 1u) {
        raw = FALLBACK_TANGENT;
    } else if (i + 1u < count) {
        raw = load_point(offset + i + 1u) - load_point(offset + i);
    } else {
        raw = load_point(offset + i) - load_point(offset + i - 1u);
    }
    return normalize_or(raw, FALLBACK_TANGENT);
}

// Writes frame `k` (a global frame index) as tangent/normal/bitangent triples.
fn store_frame(k: u32, tangent: vec3<f32>, normal: vec3<f32>, bitangent: vec3<f32>) {
    let base = 9u * k;
    frames[base] = tangent.x;
    frames[base + 1u] = tangent.y;
    frames[base + 2u] = tangent.z;
    frames[base + 3u] = normal.x;
    frames[base + 4u] = normal.y;
    frames[base + 5u] = normal.z;
    frames[base + 6u] = bitangent.x;
    frames[base + 7u] = bitangent.y;
    frames[base + 8u] = bitangent.z;
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
    if (n == 0u) {
        return;
    }

    // Root frame: seed the reference normal from an arbitrary orthonormal axis.
    let t0 = strand_tangent(offset, n, 0u);
    let r0 = orthonormal_reference(t0);
    let b0 = normalize_or(cross(t0, r0), FALLBACK_AXIS);
    store_frame(offset, t0, r0, b0);

    var prev_t = t0;
    var prev_n = r0;
    for (var j: u32 = 0u; j + 1u < n; j = j + 1u) {
        let t_next = strand_tangent(offset, n, j + 1u);

        // Reflection 1: reflect the previous normal and tangent across the plane
        // bisecting the offset between the two control points.
        let v1 = load_point(offset + j + 1u) - load_point(offset + j);
        let c1 = dot(v1, v1);
        var r_l = prev_n;
        var t_l = prev_t;
        if (c1 > F32_EPSILON) {
            r_l = reflect_plane(prev_n, v1, c1);
            t_l = reflect_plane(prev_t, v1, c1);
        }

        // Reflection 2: reflect again so the transported tangent lands on the
        // next tangent, carrying the normal along with minimal twist.
        let v2 = t_next - t_l;
        let c2 = dot(v2, v2);
        var r_reflected = r_l;
        if (c2 > F32_EPSILON) {
            r_reflected = reflect_plane(r_l, v2, c2);
        }

        // Re-orthonormalize the normal against the next tangent, then rebuild the
        // bitangent, so accumulated float error never skews the basis.
        let r_perp = r_reflected - t_next * dot(t_next, r_reflected);
        let normal = normalize_or(r_perp, orthonormal_reference(t_next));
        let bitangent = normalize_or(cross(t_next, normal), FALLBACK_AXIS);
        store_frame(offset + j + 1u, t_next, normal, bitangent);

        prev_t = t_next;
        prev_n = normal;
    }
}
