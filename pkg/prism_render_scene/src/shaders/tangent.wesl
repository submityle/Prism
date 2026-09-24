//! GPU tangent-frame reconstruction for normal mapping and anisotropic shading.
//!
//! Byte-for-byte numerical twin of the CPU golden reference in
//! `prism_render_shading::tangent`.  The visibility-buffer resolve needs a
//! per-pixel orthonormal tangent basis `(T, B, N)` to (a) rotate tangent-space
//! normal-map samples into world space and (b) orient the anisotropic GGX
//! lobe.  Three sources are tried in order, matching Unreal's derived-tangent
//! policy:
//!
//! 1. **Authored tangents** interpolated from the vertex table (produced by a
//!    MikkTSpace pass at import time), re-orthonormalized against the shaded
//!    normal so the basis stays orthogonal after interpolation.
//! 2. **Analytic tangents** from the triangle position/UV gradients (Lengyel).
//! 3. **A canonical orthonormal basis** (Duff, Cigolle, Jimenez, Wyman & Ku
//!    2017, "Building an Orthonormal Basis, Revisited") when the UVs are also
//!    degenerate; that path is flagged so downstream passes can react.
//!
//! `dot`, `cross`, `abs`, `select` and `inverseSqrt` are WESL builtins.  The
//! arithmetic mirrors the CPU reference term-for-term (left-associative,
//! component-wise) so the golden parity tests stay in lock-step.

// Squared length below which a tangent candidate is treated as degenerate.
const TANGENT_DEGENERATE_EPSILON_SQ: f32 = 1.0e-12;
// Determinant magnitude below which the UV gradient is treated as degenerate.
const TANGENT_UV_DETERMINANT_EPSILON: f32 = 1.0e-8;

// An orthonormal tangent basis paired with the surface normal.  `tangent` and
// `bitangent` are unit length and orthogonal to the normal the basis was built
// against; `bitangent` already carries the handedness sign.  Mirrors the CPU
// `TangentBasis`.
struct TangentBasis {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    // `true` when neither authored nor analytic tangents were usable and the
    // canonical fallback basis was synthesized from the normal alone.
    degenerate: bool,
}

// Raw tangent/bitangent pair with no handedness/degenerate bookkeeping, used
// as the return of the canonical Duff construction.
struct OrthonormalBasis {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

// Analytic result carrying an explicit validity flag (WESL has no `Option`).
struct AnalyticTangent {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    valid: bool,
}

// Normalizes `value`, falling back to `fallback` for degenerate inputs.
// Mirrors `normalize_or` in the CPU `vecmath` module (single `inverseSqrt`,
// matching the established `surface_normalize_or` convention).
fn tangent_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared > TANGENT_DEGENERATE_EPSILON_SQ {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

// Branchless orthonormal basis from a *unit* normal (Duff et al. 2017).  Stays
// exactly orthogonal to `normal` for any unit input and never hits the
// `z == -1` singularity of the naive frame.  Mirrors `orthonormal_basis`.
fn tangent_orthonormal_basis(normal: vec3<f32>) -> OrthonormalBasis {
    // `copysign(1, n.z)` expressed as a select so CPU and WESL agree.
    let s = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (s + normal.z);
    let b = normal.x * normal.y * a;
    var basis: OrthonormalBasis;
    basis.tangent = vec3<f32>(
        1.0 + s * normal.x * normal.x * a,
        s * b,
        -s * normal.x,
    );
    basis.bitangent = vec3<f32>(b, s + normal.y * normal.y * a, -normal.y);
    return basis;
}

// Derives a tangent basis from the triangle edge and UV gradients (Lengyel).
// `valid == false` when the UV parameterization is degenerate (zero-area in UV
// space) or the resulting tangent collapses onto the normal.  Mirrors
// `analytic_tangent_basis`.
fn analytic_tangent_basis(
    normal: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
) -> AnalyticTangent {
    var result: AnalyticTangent;
    result.tangent = vec3<f32>(0.0);
    result.bitangent = vec3<f32>(0.0);
    result.valid = false;

    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let du1 = uv1.x - uv0.x;
    let dv1 = uv1.y - uv0.y;
    let du2 = uv2.x - uv0.x;
    let dv2 = uv2.y - uv0.y;
    let determinant = du1 * dv2 - du2 * dv1;
    if abs(determinant) <= TANGENT_UV_DETERMINANT_EPSILON {
        return result;
    }
    let inverse = 1.0 / determinant;
    let raw_tangent = (edge1 * dv2 - edge2 * dv1) * inverse;
    let raw_bitangent = (edge2 * du1 - edge1 * du2) * inverse;
    let projected = raw_tangent - normal * dot(normal, raw_tangent);
    if dot(projected, projected) <= TANGENT_DEGENERATE_EPSILON_SQ {
        return result;
    }
    let unit = tangent_normalize_or(projected, normal);
    let handedness = select(1.0, -1.0, dot(cross(normal, unit), raw_bitangent) < 0.0);
    result.tangent = unit;
    result.bitangent = cross(normal, unit) * handedness;
    result.valid = true;
    return result;
}

// Builds the per-pixel tangent basis from the best available source.
// `has_authored` selects whether `authored` (xyz = tangent, w = handedness) is
// used; pass `false` when the mesh supplied no tangent attribute.  `p*`/`uv*`
// are the triangle corners for the analytic fallback.  Mirrors
// `resolve_tangent_basis`.
fn resolve_tangent_basis(
    normal: vec3<f32>,
    has_authored: bool,
    authored: vec4<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
) -> TangentBasis {
    var basis: TangentBasis;
    // 1. Authored tangent, re-orthonormalized against the shaded normal.
    if has_authored {
        let raw = authored.xyz;
        let projected = raw - normal * dot(normal, raw);
        if dot(projected, projected) > TANGENT_DEGENERATE_EPSILON_SQ {
            let unit = tangent_normalize_or(projected, normal);
            let handedness = select(1.0, -1.0, authored.w < 0.0);
            basis.tangent = unit;
            basis.bitangent = cross(normal, unit) * handedness;
            basis.degenerate = false;
            return basis;
        }
    }
    // 2. Analytic tangent from the triangle position/UV gradients.
    let analytic = analytic_tangent_basis(normal, p0, p1, p2, uv0, uv1, uv2);
    if analytic.valid {
        basis.tangent = analytic.tangent;
        basis.bitangent = analytic.bitangent;
        basis.degenerate = false;
        return basis;
    }
    // 3. Canonical fallback from the normal alone.
    let ortho = tangent_orthonormal_basis(normal);
    basis.tangent = ortho.tangent;
    basis.bitangent = ortho.bitangent;
    basis.degenerate = true;
    return basis;
}

// Rotates a tangent-space normal-map sample (`x` along T, `y` along B, `z`
// along N) into world space, renormalizing the result.  Mirrors
// `apply_tangent_space_normal`.
fn apply_tangent_space_normal(
    basis: TangentBasis,
    normal: vec3<f32>,
    tangent_space_normal: vec3<f32>,
) -> vec3<f32> {
    let world = basis.tangent * tangent_space_normal.x
        + basis.bitangent * tangent_space_normal.y
        + normal * tangent_space_normal.z;
    return tangent_normalize_or(world, normal);
}
