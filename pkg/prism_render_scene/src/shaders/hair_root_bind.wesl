//! GPU groom-root binding projection for guide strands.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::binding::bind_roots` / `bind_one_root`.
//! This is the *import-time* half of a `UE5` Groom binding asset / the `TressFX`
//! root-skinning setup: a groom is authored in a character's rest pose, and each
//! strand root must be pinned to the scalp triangle it grew from so it can ride
//! the skinned head every frame. This shader performs that one-time projection —
//! for every root it brute-force scans all rest-pose scalp triangles for the one
//! whose closest surface point is nearest, then records a barycentric attachment
//! plus a signed height along the face normal (`HairMeshBinding`). The runtime
//! per-frame reconstruction of the world transform from a deformed triangle is
//! the sibling kernel `hair_root_skinning.wesl` (CPU `resolve_root_frames`).
//!
//! One invocation owns one root; it loops over every triangle. The scan keeps
//! the first strict minimum, so ties resolve to the lowest triangle index —
//! bit-identical to the CPU golden's `dist_sq < best_dist` test. Everything is
//! bit-faithful: the same roots, vertices and topology produce identical
//! bindings. A mesh with no bindable triangle (empty topology, or every triangle
//! carrying an out-of-range vertex index) yields the unbound sentinel
//! (`triangle == HAIR_ROOT_UNBOUND`) so a root is never dropped and indices stay
//! aligned with the caller's strand array. The closest-point test is the
//! standard Ericson Voronoi-region clamp, so a projection that falls outside the
//! triangle lands on the nearest edge or vertex instead of off the face. A
//! `GPU` is now available, so this shader is compile- and type-checked through
//! the same render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware.

/// Sentinel triangle index meaning "this root could not be bound". Mirrors the
/// CPU `binding::UNBOUND` (`u32::MAX`).
const HAIR_ROOT_UNBOUND: u32 = 0xffffffffu;

/// Squared-length threshold below which a face / denominator is treated as
/// degenerate. Mirrors the CPU `binding::EPS_LEN_SQ`.
const HAIR_ROOT_EPS_LEN_SQ: f32 = 1.0e-12;

/// `f32::EPSILON`; the guard `normalize_or` uses to fall back to a fixed axis.
/// Mirrors the CPU `interpolation::Vec3::normalize_or`, which compares the
/// squared length against `f32::EPSILON` (not the coarser `EPS_LEN_SQ`).
const HAIR_ROOT_F32_EPSILON: f32 = 1.1920929e-7;

/// One strand root's attachment to the scalp mesh. Packs the CPU `MeshBinding`
/// into a `vec4` plus the triangle index on a clean 32-byte stride:
/// `bary_height.xyz` = barycentric weights, `bary_height.w` = signed height
/// along the face normal, `triangle` = triangle index (or `HAIR_ROOT_UNBOUND`).
/// This is the exact struct `hair_root_skinning.wesl` consumes as input.
struct HairMeshBinding {
    bary_height: vec4<f32>,
    triangle: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Result of the closest-point-on-triangle test: the surface point and the
/// barycentric weights `(u, v, w)` such that the point is `u*a + v*b + w*c`.
struct HairClosest {
    point: vec3<f32>,
    bary: vec3<f32>,
}

/// Dispatch parameters. `root_count` bounds the dispatch; `triangle_count`
/// bounds the scan and `vertex_count` bounds the index reads so an out-of-range
/// triangle is skipped instead of reading out of bounds.
struct HairRootParams {
    root_count: u32,
    triangle_count: u32,
    vertex_count: u32,
}

// Rest-pose strand root positions to project (`xyz` used).
@group(0) @binding(0) var<storage, read> roots: array<vec4<f32>>;
// Rest-pose scalp vertex positions (`xyz` used).
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
// Flat triangle index list, three `u32` per triangle.
@group(0) @binding(2) var<storage, read> indices: array<u32>;
// Resolved per-root attachments, one per root, in input order.
@group(0) @binding(3) var<storage, read_write> out_bindings: array<HairMeshBinding>;

var<immediate> params: HairRootParams;

// Mirror of CPU `interpolation::Vec3::normalize_or`: unit vector, or `fallback`
// when `self` is (near) zero length. The guard is `f32::EPSILON`, matching the
// CPU golden exactly.
fn hair_root_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq <= HAIR_ROOT_F32_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

/// The unbound attachment. Mirrors CPU `MeshBinding::UNBOUND`.
fn hair_root_unbound() -> HairMeshBinding {
    return HairMeshBinding(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        HAIR_ROOT_UNBOUND,
        0u,
        0u,
        0u,
    );
}

// Point on triangle `abc` closest to `p`, plus its barycentric coordinates.
// Bit-faithful port of the CPU `binding::closest_point_on_triangle`, the
// standard Ericson *Real-Time Collision Detection* Voronoi-region test: it
// decides whether the projection falls in a vertex, edge or face region and
// clamps accordingly, so the result is always on the triangle even for obtuse
// faces or points far outside. A zero-area face collapses onto vertex `a`.
fn hair_closest_point_on_triangle(
    p: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
) -> HairClosest {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return HairClosest(a, vec3<f32>(1.0, 0.0, 0.0));
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    // Vertex region B.
    if d3 >= 0.0 && d4 <= d3 {
        return HairClosest(b, vec3<f32>(0.0, 1.0, 0.0));
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        var v = 0.0;
        if denom > HAIR_ROOT_EPS_LEN_SQ {
            v = d1 / denom;
        }
        let point = a + ab * v;
        return HairClosest(point, vec3<f32>(1.0 - v, v, 0.0));
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    // Vertex region C.
    if d6 >= 0.0 && d5 <= d6 {
        return HairClosest(c, vec3<f32>(0.0, 0.0, 1.0));
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        var w = 0.0;
        if denom > HAIR_ROOT_EPS_LEN_SQ {
            w = d2 / denom;
        }
        let point = a + ac * w;
        return HairClosest(point, vec3<f32>(1.0 - w, 0.0, w));
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    let bc_num = d4 - d3;
    let bc_den = (d4 - d3) + (d5 - d6);
    if va <= 0.0 && bc_num >= 0.0 && (d5 - d6) >= 0.0 {
        var w = 0.0;
        if bc_den > HAIR_ROOT_EPS_LEN_SQ {
            w = bc_num / bc_den;
        }
        let point = b + (c - b) * w;
        return HairClosest(point, vec3<f32>(0.0, 1.0 - w, w));
    }

    // Interior face region: barycentric via the precomputed numerators.
    let denom_sum = va + vb + vc;
    if denom_sum > HAIR_ROOT_EPS_LEN_SQ {
        let inv = 1.0 / denom_sum;
        let v = vb * inv;
        let w = vc * inv;
        let point = a + ab * v + ac * w;
        return HairClosest(point, vec3<f32>(1.0 - v - w, v, w));
    }
    // Zero-area triangle: fall back to vertex A.
    return HairClosest(a, vec3<f32>(1.0, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn bind_roots(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.root_count {
        return;
    }

    let root = roots[index].xyz;

    var has_best = false;
    var best_dist = 0.0;
    var best_triangle = 0u;
    var best_bary = vec3<f32>(0.0, 0.0, 0.0);
    var best_height = 0.0;

    for (var tri = 0u; tri < params.triangle_count; tri = tri + 1u) {
        let base = tri * 3u;
        let i0 = indices[base];
        let i1 = indices[base + 1u];
        let i2 = indices[base + 2u];
        // Out-of-range vertex index → skip (mirrors `triangle_corners` None).
        if i0 >= params.vertex_count || i1 >= params.vertex_count || i2 >= params.vertex_count {
            continue;
        }
        let a = vertices[i0].xyz;
        let b = vertices[i1].xyz;
        let c = vertices[i2].xyz;

        let closest = hair_closest_point_on_triangle(root, a, b, c);
        let offset = root - closest.point;
        let dist_sq = dot(offset, offset);

        // First strict minimum wins → ties resolve to the lowest triangle index.
        var closer = true;
        if has_best {
            closer = dist_sq < best_dist;
        }
        if closer {
            // Signed height keeps the authored float-off distance: the offset
            // projected onto the face normal, or zero for a zero-area face
            // (mirrors `face_normal(...).map_or(0.0, ...)`).
            let n_raw = cross(b - a, c - a);
            var height = 0.0;
            if dot(n_raw, n_raw) > HAIR_ROOT_EPS_LEN_SQ {
                let normal = hair_root_normalize_or(n_raw, vec3<f32>(0.0, 1.0, 0.0));
                height = dot(offset, normal);
            }
            best_dist = dist_sq;
            best_triangle = tri;
            best_bary = closest.bary;
            best_height = height;
            has_best = true;
        }
    }

    if has_best {
        out_bindings[index] = HairMeshBinding(
            vec4<f32>(best_bary, best_height),
            best_triangle,
            0u,
            0u,
            0u,
        );
    } else {
        out_bindings[index] = hair_root_unbound();
    }
}
