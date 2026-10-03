//! Specular / glossy GI — GGX visible-normal (VNDF) lobe, the on-device twin of
//! the CPU golden `prism_render_shading::gi::spec_gi::ggx_lobe`.
//!
//! These are the microfacet primitives the glossy-ReSTIR trace/resolve passes
//! build on: the Trowbridge–Reitz NDF, the height-correlated Smith masking–
//! shadowing term, Schlick Fresnel, Heitz-2018 VNDF sampling and its reflection
//! pdf, and the anisotropic GGX BRDF. Everything runs in the local shading
//! frame (`+Z` = geometric normal) and is transcribed op-for-op from the golden
//! so `spec_gi/shader_tests.rs` can assert numeric parity to `1e-6`.
//!
//! WGSL has no `isFinite`; the golden's `is_finite` guards are redundant on the
//! physically-valid input domain because every denominator is floored
//! identically here (`max(_, 1e-20)` / `max(_, 1e-6)`), so no in-range input
//! produces a non-finite result. The module keeps the same floors rather than
//! the guards.
//!
//! References: Trowbridge–Reitz 1975; Heitz 2014 (*Understanding the Masking-
//! Shadowing Function*); Heitz 2018 (*Sampling the GGX Distribution of Visible
//! Normals*, JCGT); Disney/Burley 2012 (`alpha = roughness^2`).

const SPEC_PI: f32 = 3.1415926535897932;
const SPEC_FRAC_1_PI: f32 = 0.3183098861837907;
const SPEC_TAU: f32 = 6.2831853071795865;

/// Minimum GGX `alpha`. A perfect mirror would make `alpha = 0`, turning `D`
/// into a Dirac delta and dividing by zero in the pdf; the floor keeps the lobe
/// a finite, very sharp distribution. Mirrors `MIN_ALPHA`.
const SPEC_MIN_ALPHA: f32 = 1.0e-3;

// `v.normalize_or_zero()`: unit vector for a positive-length input, else zero.
fn spec_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 0.0 {
        return v / sqrt(l2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Disney/Burley perceptual `roughness` -> isotropic GGX width `alpha`
/// (`alpha = roughness^2`), clamped to `[0, 1]` then floored. Mirrors
/// `roughness_to_alpha`.
fn spec_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SPEC_MIN_ALPHA);
}

/// `roughness` + signed `anisotropy in [-1, 1]` -> anisotropic widths
/// `(alpha_x, alpha_y)` along tangent/bitangent. Mirrors
/// `roughness_to_alpha_anisotropic`.
fn spec_roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let alpha = spec_roughness_to_alpha(roughness);
    let aniso = clamp(anisotropy, -1.0, 1.0);
    let aspect = sqrt(max(1.0 - 0.9 * aniso, 1.0e-4));
    let ax = max(alpha / aspect, SPEC_MIN_ALPHA);
    let ay = max(alpha * aspect, SPEC_MIN_ALPHA);
    return vec2<f32>(ax, ay);
}

/// Isotropic GGX NDF `D(n·h)`, normalised so `∫ D(h)(n·h) dω = 1`. Zero for a
/// back-facing half vector. Mirrors `ndf_ggx`.
fn spec_ndf_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    if n_dot_h <= 0.0 {
        return 0.0;
    }
    let a = max(alpha, SPEC_MIN_ALPHA);
    let a2 = a * a;
    let cos2 = n_dot_h * n_dot_h;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    let d = a2 * SPEC_FRAC_1_PI / max(denom * denom, 1.0e-20);
    return max(d, 0.0);
}

/// Anisotropic GGX NDF for a local half vector `h` (`h.z` = cosine with the
/// normal). Zero for a back-facing half vector. Mirrors `ndf_ggx_anisotropic`.
fn spec_ndf_ggx_anisotropic(h: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = max(alpha_x, SPEC_MIN_ALPHA);
    let ay = max(alpha_y, SPEC_MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    let d = 1.0 / (SPEC_PI * ax * ay * max(s * s, 1.0e-20));
    return max(d, 0.0);
}

/// Smith `Lambda` for an isotropic GGX surface at cosine `cos_theta` (= n·w).
/// Mirrors `smith_lambda`.
fn spec_smith_lambda(cos_theta: f32, alpha: f32) -> f32 {
    let c = clamp(abs(cos_theta), 1.0e-6, 1.0);
    let a = max(alpha, SPEC_MIN_ALPHA);
    let cos2 = c * c;
    let tan2 = (1.0 - cos2) / cos2;
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + a * a * tan2, 0.0)));
    return max(lambda, 0.0);
}

/// Smith `Lambda` for an anisotropic GGX surface at local direction `w`.
/// Mirrors `smith_lambda_anisotropic`.
fn spec_smith_lambda_anisotropic(w: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    let cz = clamp(abs(w.z), 1.0e-6, 1.0);
    let ax = max(alpha_x, SPEC_MIN_ALPHA);
    let ay = max(alpha_y, SPEC_MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + tan2, 0.0)));
    return max(lambda, 0.0);
}

/// Smith single-direction masking `G1 = 1 / (1 + Lambda)` (isotropic). Mirrors
/// `smith_g1`.
fn spec_smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + spec_smith_lambda(cos_theta, alpha));
}

/// Height-correlated Smith `G2` (isotropic). Zero if either direction is below
/// the surface. Mirrors `smith_g2`.
fn spec_smith_g2(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    let g = 1.0 / (1.0 + spec_smith_lambda(n_dot_v, alpha) + spec_smith_lambda(n_dot_l, alpha));
    return clamp(g, 0.0, 1.0);
}

/// Height-correlated Smith `G2` (anisotropic) from local view/light directions.
/// Mirrors `smith_g2_anisotropic`.
fn spec_smith_g2_anisotropic(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0
            + spec_smith_lambda_anisotropic(wo, alpha_x, alpha_y)
            + spec_smith_lambda_anisotropic(wi, alpha_x, alpha_y));
    return clamp(g, 0.0, 1.0);
}

/// Schlick Fresnel for RGB `f0`: `F = f0 + (1 - f0)(1 - cosθ)^5`, clamped per
/// channel. Mirrors `fresnel_schlick`.
fn spec_fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let c = max(1.0 - clamp(cos_theta, 0.0, 1.0), 0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (vec3<f32>(1.0) - f0) * c5;
    return clamp(f, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Scalar Schlick Fresnel. Mirrors `fresnel_schlick_scalar`.
fn spec_fresnel_schlick_scalar(f0: f32, cos_theta: f32) -> f32 {
    let c = max(1.0 - clamp(cos_theta, 0.0, 1.0), 0.0);
    let c5 = (c * c) * (c * c) * c;
    return clamp(f0 + (1.0 - f0) * c5, 0.0, 1.0);
}

/// Mirror-reflects `wo` about the local normal `+Z`. Mirrors `reflect_z`.
fn spec_reflect_z(wo: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-wo.x, -wo.y, wo.z);
}

/// Heitz-2018 bounded spherical-cap VNDF sample for local view `wo`. `(u1, u2)`
/// are canonical uniforms; returns a unit half vector in the upper hemisphere
/// distributed as `D_v`. A below-horizon `wo` falls back to `+Z`. Mirrors
/// `sample_ggx_vndf`.
fn spec_sample_ggx_vndf(wo: vec3<f32>, alpha_x: f32, alpha_y: f32, u1: f32, u2: f32) -> vec3<f32> {
    let ax = max(alpha_x, SPEC_MIN_ALPHA);
    let ay = max(alpha_y, SPEC_MIN_ALPHA);
    if wo.z <= 0.0 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    // Stretch the view into the hemisphere configuration (alpha = 1).
    let vh0 = spec_normalize_or_zero(vec3<f32>(ax * wo.x, ay * wo.y, wo.z));
    var vh = vh0;
    if dot(vh0, vh0) <= 0.0 {
        vh = vec3<f32>(0.0, 0.0, 1.0);
    }

    // Orthonormal basis of the stretched view (Heitz's robust construction).
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if lensq > 1.0e-12 {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
    }
    let t2 = cross(vh, t1);

    // Sample a point on the projected (and reparameterised) disk.
    let r = sqrt(clamp(u1, 0.0, 1.0));
    let phi = SPEC_TAU * clamp(u2, 0.0, 1.0);
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;

    // Reproject onto the hemisphere and unstretch.
    let pz = sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    let nh = p1 * t1 + p2 * t2 + pz * vh;
    let h = spec_normalize_or_zero(vec3<f32>(ax * nh.x, ay * nh.y, max(nh.z, 0.0)));
    if dot(h, h) > 0.0 {
        return h;
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

/// Solid-angle pdf of the VNDF half vector `h` for local view `wo`:
/// `D_v(h) = G1(wo) max(0, wo·h) D(h) / wo.z`. Mirrors `vndf_pdf_h`.
fn spec_vndf_pdf_h(wo: vec3<f32>, h: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = max(dot(wo, h), 0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + spec_smith_lambda_anisotropic(wo, alpha_x, alpha_y));
    let d = spec_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let pdf = g1 * v_dot_h * d / max(wo.z, 1.0e-6);
    return max(pdf, 0.0);
}

/// Solid-angle pdf of a VNDF-sampled reflected direction `wi` (reflection
/// Jacobian `1 / (4 wo·h)`, `h = normalize(wo + wi)`). Mirrors
/// `vndf_pdf_reflect`.
fn spec_vndf_pdf_reflect(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = spec_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return 0.0;
    }
    let v_dot_h = max(dot(wo, h), 0.0);
    if v_dot_h <= 1.0e-8 {
        return 0.0;
    }
    let pdf = spec_vndf_pdf_h(wo, h, alpha_x, alpha_y) / (4.0 * v_dot_h);
    return max(pdf, 0.0);
}

/// Anisotropic GGX specular BRDF (RGB `f0`, excludes the `n·l` cosine):
/// `f = D G2 F / (4 (n·wo)(n·wi))`. Mirrors `ggx_brdf`.
fn spec_ggx_brdf(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32, f0: vec3<f32>) -> vec3<f32> {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = spec_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return vec3<f32>(0.0);
    }
    let d = spec_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = spec_smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = spec_fresnel_schlick(f0, max(dot(wo, h), 0.0));
    let denom = max(4.0 * wo.z * wi.z, 1.0e-6);
    let v = f * (d * g / denom);
    return max(v, vec3<f32>(0.0));
}

/// Scalar GGX specular BRDF (scalar `f0`), a directional target / lobe weight.
/// Mirrors `ggx_brdf_scalar`.
fn spec_ggx_brdf_scalar(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32, f0: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = spec_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return 0.0;
    }
    let d = spec_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = spec_smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = spec_fresnel_schlick_scalar(f0, max(dot(wo, h), 0.0));
    let denom = max(4.0 * wo.z * wi.z, 1.0e-6);
    return max(d * g * f / denom, 0.0);
}

// --- compile-coverage harness -------------------------------------------------
// A trivial entry point that folds every public lobe helper into one output so
// the module validates standalone through the render world's `ShaderCache`
// (naga would otherwise prune the functions as dead code). Not scheduled.
struct SpecLobeProbe {
    value: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> spec_lobe_probe: SpecLobeProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_gi_lobe_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    let aa = spec_roughness_to_alpha_anisotropic(0.4, 0.25);
    let wo = spec_normalize_or_zero(vec3<f32>(0.3, 0.1, 0.9));
    let h = spec_sample_ggx_vndf(wo, aa.x, aa.y, 0.37, 0.62);
    let wi = spec_reflect_z(wo);
    let brdf = spec_ggx_brdf(wo, wi, aa.x, aa.y, vec3<f32>(0.04));
    let pdf = spec_vndf_pdf_reflect(wo, wi, aa.x, aa.y);
    let scalar = spec_ggx_brdf_scalar(wo, wi, aa.x, aa.y, 0.04);
    let g2 = spec_smith_g2(wo.z, wi.z, spec_roughness_to_alpha(0.4));
    let g1 = spec_smith_g1(wo.z, spec_roughness_to_alpha(0.4));
    let ndf = spec_ndf_ggx(h.z, spec_roughness_to_alpha(0.4));
    spec_lobe_probe.value = vec4<f32>(brdf) + vec4<f32>(pdf, scalar, g2, g1 + ndf + spec_vndf_pdf_h(wo, h, aa.x, aa.y));
}
