//! GPU coherent per-vertex strand frame kernel: rotation-minimizing transport.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::frames::build_strand_frames`. A `GPU` is now
//! available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware; it is the GPU-driven *frame transport* stage (design §10.4)
//! that hands every downstream width/anisotropy consumer — the ribbon
//! ([`hair_ribbon`]) and shell ([`hair_mesh_shell`]) meshers plus anisotropic
//! shading — a stable orthonormal basis at every control point, mirroring the
//! `UE5` Groom / `AMD` `TressFX` minimal-twist reference transport at the
//! algorithm level without reusing their code.
//!
//! One invocation owns one strand and walks it root→tip, because the
//! **double-reflection rotation-minimizing frame** (Wang et al. 2008) is
//! inherently sequential: each frame's reference normal is propagated from the
//! previous one with two reflections so it never rotates about the tangent more
//! than the curve geometry forces. The naive per-point basis (arbitrary helper
//! axis, cross twice) flips wherever the tangent crosses the helper axis, which
//! makes ribbons self-fold and shimmer; this transport avoids that.
//!
//! Each emitted frame is right-handed and re-orthonormalized against its
//! tangent for numerical hygiene, in the identical order and math as the CPU
//! golden so the GPU path stays a faithful twin: `tangent` runs root→tip by
//! forward difference, `normal` is the transported reference direction, and
//! `bitangent = tangent × normal` is the ribbon-widening direction. All the
//! degenerate guards match the CPU reference — an empty / single-point strand
//! and coincident control points fall back to fixed directions rather than
//! dividing by zero.
//!
//! Buffer layout: `points` is the flat pool of all strand control points
//! (`xyz` used); `strands[s]` slices it for strand `s` and names the output
//! offset. Outputs `out_tangents` / `out_normals` / `out_bitangents` run
//! parallel, one `vec4` (`xyz` axis, `w = 0`) per control point. Everything is
//! bounds-guarded: a strand past the scratch bound emits nothing rather than
//! indexing out of range, matching the CPU length handling.

/// Upper bound on control points per strand, bounding the sequential walk so
/// the kernel needs no dynamic allocation. Matches the ribbon / shell /
/// interpolation twins.
const MAX_HAIR_STRAND_POINTS: u32 = 256u;

/// Squared-length threshold below which a direction is treated as degenerate.
/// Mirrors the CPU `Vec3::normalize_or` guard and the reflection `c > EPSILON`
/// checks (`f32::EPSILON`).
const FRAME_EPSILON: f32 = 1.1920929e-7;

/// Per-strand slice descriptor into the flat input pool and output buffers.
struct HairFrameStrand {
    // First control point of this strand in `points`.
    point_offset: u32,
    // Number of control points; one point still emits a fallback-tangent frame.
    point_count: u32,
    // First output slot (`out_tangents` / `out_normals` / `out_bitangents`).
    out_offset: u32,
}

/// Tunable parameters shared by every strand of a groom. Only the strand count
/// that bounds the dispatch is needed; the geometry is fully data-driven.
struct HairFrameParams {
    // Number of strands, bounding the dispatch.
    strand_count: u32,
}

// Flat pool of all strand control points (`xyz` used).
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// Per-strand slice descriptors.
@group(0) @binding(1) var<storage, read> strands: array<HairFrameStrand>;
// Output per-vertex tangents (`xyz`, `w = 0`); one per control point.
@group(0) @binding(2) var<storage, read_write> out_tangents: array<vec4<f32>>;
// Output per-vertex rotation-minimizing normals (`xyz`, `w = 0`).
@group(0) @binding(3) var<storage, read_write> out_normals: array<vec4<f32>>;
// Output per-vertex bitangents (`xyz`, `w = 0`), the ribbon-widening direction.
@group(0) @binding(4) var<storage, read_write> out_bitangents: array<vec4<f32>>;

var<immediate> params: HairFrameParams;

// Fixed fallback tangent for degenerate strands, matching the CPU
// `FALLBACK_TANGENT = (0, 1, 0)`.
fn fallback_tangent() -> vec3<f32> {
    return vec3<f32>(0.0, 1.0, 0.0);
}

/// Unit vector, or `fallback` when `v` is (near) zero length. Bit-mirrors the
/// CPU `Vec3::normalize_or`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq <= FRAME_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

/// Unit reference normal perpendicular to `tangent`; picks the cardinal axis
/// least aligned with the tangent as a helper. Mirrors the CPU
/// `orthonormal_reference`.
fn orthonormal_reference(tangent: vec3<f32>) -> vec3<f32> {
    var helper = vec3<f32>(0.0, 1.0, 0.0);
    if abs(tangent.x) < 0.9 {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    return normalize_or(cross(tangent, helper), vec3<f32>(0.0, 0.0, 1.0));
}

/// Reflects `v` in the plane whose normal is `axis` (with `c = axis·axis`); the
/// caller has already checked `c` is non-degenerate. Mirrors the CPU `reflect`.
fn reflect_plane(v: vec3<f32>, axis: vec3<f32>, c: f32) -> vec3<f32> {
    return v - axis * (2.0 * dot(axis, v) / c);
}

/// Forward-difference unit tangent at control point `k` of a strand of `count`
/// points based at `base`. Mirrors the CPU `strand_tangents` convention.
fn strand_tangent(base: u32, count: u32, k: u32) -> vec3<f32> {
    if count <= 1u {
        return fallback_tangent();
    }
    var raw = vec3<f32>(0.0, 0.0, 0.0);
    if k + 1u < count {
        raw = points[base + k + 1u].xyz - points[base + k].xyz;
    } else {
        raw = points[base + k].xyz - points[base + k - 1u].xyz;
    }
    return normalize_or(raw, fallback_tangent());
}

@compute @workgroup_size(64)
fn hair_frames(@builtin(global_invocation_id) gid: vec3<u32>) {
    let strand = gid.x;
    if strand >= params.strand_count {
        return;
    }

    let desc = strands[strand];
    let n = desc.point_count;
    // Empty strand emits nothing; a point count past the scratch bound is
    // skipped rather than indexing out of range (CPU length handling).
    if n == 0u || n > MAX_HAIR_STRAND_POINTS {
        return;
    }

    let pbase = desc.point_offset;
    let obase = desc.out_offset;

    // Root frame: seed a reference normal from an arbitrary helper axis.
    var cur_tangent = strand_tangent(pbase, n, 0u);
    var cur_normal = orthonormal_reference(cur_tangent);
    var cur_bitangent = normalize_or(cross(cur_tangent, cur_normal), vec3<f32>(0.0, 0.0, 1.0));
    out_tangents[obase] = vec4<f32>(cur_tangent, 0.0);
    out_normals[obase] = vec4<f32>(cur_normal, 0.0);
    out_bitangents[obase] = vec4<f32>(cur_bitangent, 0.0);

    // Propagate every later frame by the double-reflection method.
    for (var i = 0u; i + 1u < n; i = i + 1u) {
        let t_next = strand_tangent(pbase, n, i + 1u);

        // Reflection 1: reflect the previous normal and tangent across the
        // plane bisecting the offset between the two control points.
        let v1 = points[pbase + i + 1u].xyz - points[pbase + i].xyz;
        let c1 = dot(v1, v1);
        var r_l = cur_normal;
        var t_l = cur_tangent;
        if c1 > FRAME_EPSILON {
            r_l = reflect_plane(cur_normal, v1, c1);
            t_l = reflect_plane(cur_tangent, v1, c1);
        }

        // Reflection 2: reflect again so the transported tangent lands exactly
        // on the next tangent, carrying the normal along with minimal twist.
        let v2 = t_next - t_l;
        let c2 = dot(v2, v2);
        var r_reflected = r_l;
        if c2 > FRAME_EPSILON {
            r_reflected = reflect_plane(r_l, v2, c2);
        }

        // Re-orthonormalize the normal against the next tangent, then rebuild
        // the bitangent, so accumulated float error never skews the basis.
        let r_perp = r_reflected - t_next * dot(t_next, r_reflected);
        let normal = normalize_or(r_perp, orthonormal_reference(t_next));
        let bitangent = normalize_or(cross(t_next, normal), vec3<f32>(0.0, 0.0, 1.0));

        let slot = obase + i + 1u;
        out_tangents[slot] = vec4<f32>(t_next, 0.0);
        out_normals[slot] = vec4<f32>(normal, 0.0);
        out_bitangents[slot] = vec4<f32>(bitangent, 0.0);

        cur_tangent = t_next;
        cur_normal = normal;
        cur_bitangent = bitangent;
    }
}
