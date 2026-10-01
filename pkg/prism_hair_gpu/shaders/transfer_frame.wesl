// Follicle local-frame transfer twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::follicle_bind::transfer_frame`. After a scalp
// triangle deforms, each follicle needs a fresh orthonormal local frame
// (normal, tangent, bitangent) at its root so the strand can be re-oriented. The
// golden interpolates the per-vertex normal and tangent by the binding's
// (sanitised) barycentric weights, then Gram-Schmidt re-orthonormalises them
// into a right-handed basis. This kernel broadcasts one shared deformed triangle
// (per-vertex normals and tangents) to a whole batch of bindings, one thread per
// binding, and emits that binding's orthonormal frame.
//
// It is a distinct sibling of the `follicle_bind` root-transfer twin: that one
// reconstructs the root *position* (barycentric surface point plus a signed
// normal offset); this one reconstructs the root's orthonormal *frame*
// (interpolation + Gram-Schmidt), a different computation and a different output.
// Only the binding's barycentric weights are consumed here — the normal offset
// plays no part in the frame.
//
// Buffer layout: `params` carries the one shared deformed triangle (three
// per-vertex normals then three per-vertex tangents as `vec4`, xyz used) plus
// the binding count; `bindings[i]` is binding `i` (sanitised-or-raw u, v, w in
// xyz, normal offset in w and unused here); `out` holds three `vec4` per binding
// (normal, tangent, bitangent, each xyz used).
//
// Degenerate inputs fall back exactly as the golden does: an all-non-positive
// (or non-finite) weight set sanitises to the centroid; a zero-length
// interpolated normal falls back to +Z; a tangent (near) parallel to the normal
// falls back to a canonical axis chosen by the normal's x magnitude; a
// zero-length bitangent falls back to +Y — so the result is always a finite
// orthonormal-ish basis and never a NaN. An empty batch dispatches nothing.
//
// The interpolation folds, the Gram-Schmidt projection and three normalises are
// multiply-adds and `sqrt`-reciprocals a GPU may fuse, so the twin is matched
// against a tolerance (abs_diff < 1e-4 or rel_diff < 1e-3) rather than
// bit-for-bit. The kernel uses only `dot`, `cross`, compare, `abs`, `sqrt`, a
// reciprocal and multiply/add in the portable core-WGSL subset — no `exp`,
// `pow` or optional device feature — so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard barycentric mesh-attachment frame transfer plus
// Gram-Schmidt orthonormalisation; no Unreal Engine source or derived code.

// Zero-area / non-finite / zero-length guard, mirroring
// `follicle_bind::DEGENERATE_EPS`.
const DEGENERATE_EPS: f32 = 1.0e-12;
// Largest finite `f32`; used with the self-equality test to reject NaN/inf.
const F32_MAX: f32 = 3.4028235e38;

struct Params {
    // Shared deformed-triangle per-vertex normals (xyz used, w padding).
    nrm0: vec4<f32>,
    nrm1: vec4<f32>,
    nrm2: vec4<f32>,
    // Shared deformed-triangle per-vertex tangents (xyz used, w padding).
    tan0: vec4<f32>,
    tan1: vec4<f32>,
    tan2: vec4<f32>,
    // Number of bindings, bounding the dispatch.
    binding_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per binding, (u, v, w) in xyz; the w lane (normal offset) is unused here.
@group(0) @binding(1) var<storage, read> bindings: array<vec4<f32>>;
// Per binding, three vec4: normal, tangent, bitangent (each xyz used).
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

fn is_finite_scalar(x: f32) -> bool {
    return x == x && abs(x) <= F32_MAX;
}

fn normalize_or(a: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(a, a));
    if (len > DEGENERATE_EPS && is_finite_scalar(len)) {
        return a * (1.0 / len);
    }
    return fallback;
}

fn clamp0(x: f32) -> f32 {
    if (is_finite_scalar(x) && x > 0.0) {
        return x;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.binding_count) {
        return;
    }

    // Sanitise the stored weights exactly as `Barycentric::sanitized`.
    let raw = bindings[i].xyz;
    let third = 1.0 / 3.0;
    var su = clamp0(raw.x);
    var sv = clamp0(raw.y);
    var sw = clamp0(raw.z);
    let sum = su + sv + sw;
    if (sum > DEGENERATE_EPS) {
        let inv_sum = 1.0 / sum;
        su = su * inv_sum;
        sv = sv * inv_sum;
        sw = sw * inv_sum;
    } else {
        su = third;
        sv = third;
        sw = third;
    }

    // Interpolated normal, with +Z fallback.
    let raw_normal = params.nrm0.xyz * su + params.nrm1.xyz * sv + params.nrm2.xyz * sw;
    let normal = normalize_or(raw_normal, vec3<f32>(0.0, 0.0, 1.0));

    // Interpolated tangent, Gram-Schmidt against the normal.
    let raw_tangent = params.tan0.xyz * su + params.tan1.xyz * sv + params.tan2.xyz * sw;
    let projected = raw_tangent - normal * dot(raw_tangent, normal);

    // Fallback tangent: any axis not parallel to the normal.
    var fallback = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(normal.x) < 0.9) {
        fallback = normalize_or(cross(normal, vec3<f32>(1.0, 0.0, 0.0)), vec3<f32>(0.0, 1.0, 0.0));
    } else {
        fallback = normalize_or(cross(normal, vec3<f32>(0.0, 1.0, 0.0)), vec3<f32>(1.0, 0.0, 0.0));
    }
    let tangent = normalize_or(projected, fallback);
    let bitangent = normalize_or(cross(normal, tangent), vec3<f32>(0.0, 1.0, 0.0));

    out[3u * i + 0u] = vec4<f32>(normal, 0.0);
    out[3u * i + 1u] = vec4<f32>(tangent, 0.0);
    out[3u * i + 2u] = vec4<f32>(bitangent, 0.0);
}
