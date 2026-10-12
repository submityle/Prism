//! Prism PBR surface shading — the WGSL twin of the CPU golden
//! `prism_render_shading::lighting::evaluate_principled_direct`.
//!
//! This file is single-sourced into the crate as
//! `shader::SURFACE_SHADER_WGSL` via `include_str!`, so the parity test and any
//! tool share this exact text.
//
// ---------------------------------------------------------------------------
// MIRROR SCOPE (kept in sync with `prism_render_shading_gpu::MIRROR_SCOPE`)
// ---------------------------------------------------------------------------
// Matched, term for term, against the CPU base lobe:
//   * `base_color`, `base_weight` (whole base layer), `metallic`
//   * `diffuse_roughness` -> Oren-Nayar rough diffuse
//   * `perceptual_roughness` -> GGX width `alpha = r^2`
//   * `normal_variance` -> the GGX-width filter (`filter_alpha_sq`, KAPPA_MAX)
//   * `specular_weight`, `specular_color`, `specular_ior` -> dielectric Fresnel
//     through the specification's `modulated_ior`, and the metal `F82`-tint
//   * anisotropic GGX `D`/`V` with `anisotropy` / `anisotropy_rotation`
//   * Kulla-Conty + Turquin multiple-scattering compensation
//   * `emissive`, and the `n.l <= 0` early-out that returns it
//
// NOT mirrored (the OpenPBR lobes that live outside the base layer):
//   * coat / clearcoat, including `coat_*` and the two-pass macro attenuation
//   * fuzz / sheen
//   * transmission, subsurface, and thin-film iridescence
// The caller must leave the corresponding CPU fields at their identity
// defaults (coat = 0, transmission = 0, iridescence = 0) for parity to hold;
// `GpuSurfaceParams` exposes no field for them precisely so a call site cannot
// silently ask the GPU for a lobe this shader does not evaluate.
//
// Also not mirrored: the CPU reference's defensive `is_finite` guards and its
// saturating fallbacks. WGSL (as compiled by naga 30 / wgpu 30) exposes no
// `isNan`/`isInf` builtin, and `x * 0.0 != 0.0` style NaN probes are unsafe
// under Metal's fast-math. The guards only fire on non-finite inputs, which no
// physical surface produces, so the twin is defined for — and parity-verified
// on — finite inputs. Dropping them leaves every finite case numerically
// identical to the CPU path.

const PI: f32 = 3.141592653589793;
const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;
const MIN_ALPHA: f32 = 1.0e-3;
const MIN_ANISOTROPIC_ALPHA: f32 = 1.0e-3;
const MIN_DIELECTRIC_IOR: f32 = 1.0 + 1.0e-3;
// Tokuyoshi-Kaplanyan `kappa_max` (`DEFAULT_KAPPA_MAX`).
const KAPPA_MAX: f32 = 0.18;
// Lower bound keeping `1 - E_avg` strictly positive (`MIN_ENERGY_DEFICIT`).
const MIN_ENERGY_DEFICIT: f32 = 1.0e-4;
// `1 / 21`, the Schlick average-Fresnel denominator reciprocal.
const INV_21: f32 = 1.0 / 21.0;
// The metal `F82`-tint fixed grazing-edge cosine.
const MU_BAR: f32 = 1.0 / 7.0;
const OREN_A_OFFSET: f32 = 0.33;
const OREN_B_OFFSET: f32 = 0.09;
const OREN_A_SCALE: f32 = 0.5;
const OREN_B_SCALE: f32 = 0.45;
const OREN_MIN_T: f32 = 1.0e-6;
// `diffuse_roughness == 1` maps to a fully rough cavity field (`FRAC_PI_2`).
const OREN_MAX_SIGMA: f32 = 1.5707963267948966;

// The uniform batch capacity. Mirrors `GpuSurfaceParams::MAX_BATCH`.
const MAX_BATCH: u32 = 128u;

// The layout mirrors `#[repr(C)] GpuSurfaceParams` field for field; every
// `vec3<f32>` sits at a 16-byte offset, so the Rust `[f32; 3]` members (natural
// align 4) land on the same byte offsets. See `params.rs` for the table.
struct SurfaceParams {
    base_color: vec3<f32>,
    base_weight: f32,
    specular_color: vec3<f32>,
    diffuse_roughness: f32,
    emissive: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    specular_weight: f32,
    specular_ior: f32,
    normal_variance: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    reserved0: f32,
    reserved1: f32,
    normal: vec3<f32>,
    normal_pad: f32,
    view: vec3<f32>,
    view_pad: f32,
    tangent: vec3<f32>,
    tangent_pad: f32,
    bitangent: vec3<f32>,
    bitangent_pad: f32,
    direction: vec3<f32>,
    visibility: f32,
    illuminance: vec3<f32>,
    illuminance_pad: f32,
};

struct TangentFrame {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: array<SurfaceParams, MAX_BATCH>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4<f32>>;

fn vdot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn vadd(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a + b;
}

fn vsub(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a - b;
}

fn vmul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a * b;
}

fn vscale(a: vec3<f32>, s: f32) -> vec3<f32> {
    return a * s;
}

fn vmix3(a: vec3<f32>, b: vec3<f32>, factor: f32) -> vec3<f32> {
    return vadd(vscale(a, 1.0 - factor), vscale(b, factor));
}

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = vdot(value, value);
    // The CPU also requires `length_squared.is_finite()` here; see the header.
    if (length_squared > 1.0e-12) {
        return vscale(value, 1.0 / sqrt(length_squared));
    }
    return fallback;
}

// Branchless orthonormal basis from a unit normal (Duff et al. 2017).
fn orthonormal_basis(normal: vec3<f32>) -> TangentFrame {
    let sign = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = vec3<f32>(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = vec3<f32>(b, sign + normal.y * normal.y * a, -normal.y);
    return TangentFrame(tangent, bitangent);
}

fn anisotropic_axes(
    normal: vec3<f32>,
    basis_tangent: vec3<f32>,
    basis_bitangent: vec3<f32>,
    rotation: f32,
) -> TangentFrame {
    let fallback = orthonormal_basis(normal);
    let projected = vsub(basis_tangent, vscale(normal, vdot(normal, basis_tangent)));
    let tangent0 = normalize_or(projected, fallback.tangent);
    let handedness = select(1.0, -1.0, vdot(cross(normal, tangent0), basis_bitangent) < 0.0);
    let bitangent0 = vscale(cross(normal, tangent0), handedness);
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    let tangent = vadd(vscale(tangent0, cos_r), vscale(bitangent0, sin_r));
    let bitangent = vsub(vscale(bitangent0, cos_r), vscale(tangent0, sin_r));
    return TangentFrame(tangent, bitangent);
}

fn dielectric_f0(s: SurfaceParams) -> vec3<f32> {
    let ior = clamp(s.specular_ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    let f0 = r * r * clamp(s.specular_weight, 0.0, 1.0);
    return vec3<f32>(f0 * s.specular_color.x, f0 * s.specular_color.y, f0 * s.specular_color.z);
}

fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let one_minus = 1.0 - clamp(v_dot_h, 0.0, 1.0);
    let squared = one_minus * one_minus;
    let factor = squared * squared * one_minus;
    return vadd(f0, vscale(vsub(vec3<f32>(1.0), f0), factor));
}

// `OpenPBR` `modulated_ior`: the index ratio whose exact Fresnel reproduces the
// `specular_weight`-scaled reflectance.
fn modulated_ior(eta: f32, specular_weight: f32) -> f32 {
    let w = max(specular_weight, 0.0);
    if (w == 0.0) {
        return 1.0;
    }
    let r = (1.0 - eta) / (1.0 + eta);
    let f_s = r * r;
    let sign = select(-1.0, 1.0, eta >= 1.0);
    let epsilon = clamp(sign * sqrt(w * f_s), -(1.0 - 1.0e-4), 1.0 - 1.0e-4);
    let one_minus = 1.0 - epsilon;
    let eta_prime = (1.0 + epsilon) / one_minus;
    if (eta_prime > 0.0) {
        return eta_prime;
    }
    return MIN_DIELECTRIC_IOR;
}

// Unpolarized dielectric Fresnel reflectance (Walter 2007), total internal
// reflection included.
fn fresnel_dielectric(cos_i_in: f32, ior_in: f32) -> f32 {
    let cos_i = clamp(cos_i_in, 0.0, 1.0);
    let ior = max(ior_in, MIN_DIELECTRIC_IOR);
    let eta = 1.0 / ior;
    let sin2_i = max(1.0 - cos_i * cos_i, 0.0);
    let sin2_t = eta * eta * sin2_i;
    if (sin2_t >= 1.0) {
        return 1.0;
    }
    let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
    let r_parl = (ior * cos_i - cos_t) / (ior * cos_i + cos_t);
    let r_perp = (cos_i - ior * cos_t) / (cos_i + ior * cos_t);
    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

fn base_dielectric_fresnel_rgb(
    cos_i: f32,
    specular_ior: f32,
    specular_weight: f32,
    specular_color: vec3<f32>,
) -> vec3<f32> {
    let eta = modulated_ior(clamp(specular_ior, 1.0, 3.0), clamp(specular_weight, 0.0, 1.0));
    var f = 0.0;
    // `eta' == 1` is a non-interface: no grazing reflection (f32::EPSILON gate).
    if (abs(eta - 1.0) > 1.1920929e-7) {
        f = fresnel_dielectric(cos_i, eta);
    }
    return vec3<f32>(f * specular_color.x, f * specular_color.y, f * specular_color.z);
}

// `OpenPBR` metal `F82`-tint conductor Fresnel (Kutz 2021 / Hoffman 2019).
fn base_metal_fresnel_rgb(
    cos_i: f32,
    base_weight: f32,
    base_color: vec3<f32>,
    specular_weight: f32,
    specular_color: vec3<f32>,
) -> vec3<f32> {
    let bw = clamp(base_weight, 0.0, 1.0);
    let f0 = vec3<f32>(bw * base_color.x, bw * base_color.y, bw * base_color.z);
    let f_bar_curve = fresnel_schlick(f0, MU_BAR);
    let om_bar = 1.0 - MU_BAR;
    let om_bar2 = om_bar * om_bar;
    let denom = MU_BAR * om_bar2 * om_bar2 * om_bar2;
    let mu = clamp(cos_i, 0.0, 1.0);
    let om = 1.0 - mu;
    let om2 = om * om;
    let correction = mu * om2 * om2 * om2 / denom;
    let curve = fresnel_schlick(f0, mu);
    let scale = clamp(specular_weight, 0.0, 1.0);
    let tint = clamp(specular_color, vec3<f32>(0.0), vec3<f32>(1.0));
    let f_edge = tint * f_bar_curve;
    return scale * (curve - correction * (f_bar_curve - f_edge));
}

fn openpbr_anisotropy_to_alpha(alpha: f32, anisotropy: f32) -> vec2<f32> {
    let a = clamp(anisotropy, 0.0, 1.0);
    let one_minus = 1.0 - a;
    let stretch = sqrt(2.0 / (1.0 + one_minus * one_minus));
    let alpha_t = max(alpha * stretch, MIN_ANISOTROPIC_ALPHA);
    let alpha_b = max(one_minus * alpha * stretch, MIN_ANISOTROPIC_ALPHA);
    return vec2<f32>(alpha_t, alpha_b);
}

// Anisotropic GGX/Trowbridge-Reitz distribution (UE `D_GGXaniso`).
fn distribution_ggx_anisotropic(
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let dx = x_dot_h / ax;
    let dy = y_dot_h / ay;
    let d = dx * dx + dy * dy + n_dot_h * n_dot_h;
    return 1.0 / max(PI * ax * ay * d * d, MIN_N_DOT);
}

// Height-correlated anisotropic Smith visibility (UE `Vis_SmithJointAniso`).
fn visibility_smith_ggx_anisotropic(
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let ax_v = ax * x_dot_v;
    let ay_v = ay * y_dot_v;
    let lambda_v = n_dot_l * sqrt(ax_v * ax_v + ay_v * ay_v + n_dot_v * n_dot_v);
    let ax_l = ax * x_dot_l;
    let ay_l = ay * y_dot_l;
    let lambda_l = n_dot_v * sqrt(ax_l * ax_l + ay_l * ay_l + n_dot_l * n_dot_l);
    return 0.5 / max(lambda_v + lambda_l, MIN_N_DOT);
}

// Oren-Nayar (1994) rough-diffuse factor (trig-free Gotanda/Fujii form).
fn oren_nayar_factor(
    n_dot_v: f32,
    n_dot_l: f32,
    v_dot_l: f32,
    diffuse_roughness: f32,
) -> f32 {
    let sigma = clamp(diffuse_roughness, 0.0, 1.0) * OREN_MAX_SIGMA;
    let sigma_sq = sigma * sigma;
    let a = 1.0 - OREN_A_SCALE * sigma_sq / (sigma_sq + OREN_A_OFFSET);
    let b = OREN_B_SCALE * sigma_sq / (sigma_sq + OREN_B_OFFSET);
    let s = v_dot_l - n_dot_v * n_dot_l;
    var t = 1.0;
    if (s > 0.0) {
        t = max(max(n_dot_v, n_dot_l), OREN_MIN_T);
    }
    return max(a + b * s / t, 0.0);
}

// Closed-form Karis environment-BRDF fit `(scale, bias)`.
fn env_brdf_approx(n_dot_v: f32, perceptual_roughness: f32) -> vec2<f32> {
    let ndv = clamp(n_dot_v, 0.0, 1.0);
    let rough = clamp(perceptual_roughness, 0.0, 1.0);
    // `r[i] = roughness * C0[i] + C1[i]` with
    // C0 = [-1.0, -0.0275, -0.572, 0.022] and
    // C1 = [1.0, 0.0425, 1.04, -0.04].
    let r0 = rough * (-1.0) + 1.0;
    let r1 = rough * (-0.0275) + 0.0425;
    let r2 = rough * (-0.572) + 1.04;
    let r3 = rough * 0.022 + (-0.04);
    let a004 = min(r0 * r0, exp2(-9.28 * ndv)) * r0 + r1;
    return vec2<f32>(-1.04 * a004 + r2, 1.04 * a004 + r3);
}

fn directional_albedo_analytic(n_dot: f32, perceptual_roughness: f32) -> f32 {
    let sb = env_brdf_approx(n_dot, perceptual_roughness);
    return clamp(sb.x + sb.y, 0.0, 1.0);
}

// One node of the 8-node Gauss-Legendre rule: `w * E(mu) * mu`.
fn gl_term(node: f32, weight: f32, perceptual_roughness: f32) -> f32 {
    let mu = 0.5 * (node + 1.0);
    return weight * directional_albedo_analytic(mu, perceptual_roughness) * mu;
}

fn average_albedo_analytic(perceptual_roughness: f32) -> f32 {
    var sum = 0.0;
    sum = sum + gl_term(-0.9602898564975363, 0.1012285362903763, perceptual_roughness);
    sum = sum + gl_term(-0.7966664774136267, 0.2223810344533745, perceptual_roughness);
    sum = sum + gl_term(-0.5255324099163290, 0.3137066458778873, perceptual_roughness);
    sum = sum + gl_term(-0.1834346424956498, 0.3626837833783620, perceptual_roughness);
    sum = sum + gl_term(0.1834346424956498, 0.3626837833783620, perceptual_roughness);
    sum = sum + gl_term(0.5255324099163290, 0.3137066458778873, perceptual_roughness);
    sum = sum + gl_term(0.7966664774136267, 0.2223810344533745, perceptual_roughness);
    sum = sum + gl_term(0.9602898564975363, 0.1012285362903763, perceptual_roughness);
    return clamp(sum, 0.0, 1.0 - MIN_ENERGY_DEFICIT);
}

fn f_avg(f0: vec3<f32>) -> vec3<f32> {
    let c = clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0));
    return clamp(c + (vec3<f32>(1.0) - c) * INV_21, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Kulla-Conty achromatic multiple-scattering BRDF kernel.
fn kulla_conty_ms(e_o_in: f32, e_i_in: f32, e_avg_in: f32) -> f32 {
    let eo = clamp(e_o_in, 0.0, 1.0);
    let ei = clamp(e_i_in, 0.0, 1.0);
    let ea = clamp(e_avg_in, 0.0, 1.0);
    let denom = PI * (1.0 - ea);
    if (denom <= 1.0e-6) {
        return 0.0;
    }
    let v = (1.0 - eo) * (1.0 - ei) / denom;
    return max(v, 0.0);
}

// Added multiple-scattering specular BRDF: the achromatic Kulla-Conty kernel
// scaled by Turquin's colored multiple-bounce Fresnel.
fn multiscatter_direct(
    f0: vec3<f32>,
    n_dot_v: f32,
    n_dot_l: f32,
    perceptual_roughness: f32,
) -> vec3<f32> {
    let e_o = directional_albedo_analytic(n_dot_v, perceptual_roughness);
    let e_i = directional_albedo_analytic(n_dot_l, perceptual_roughness);
    let e_avg = average_albedo_analytic(perceptual_roughness);
    let f_ms = kulla_conty_ms(e_o, e_i, e_avg);
    let favg = f_avg(f0);
    let denom = max(
        vec3<f32>(1.0) - favg * (1.0 - e_avg),
        vec3<f32>(MIN_ENERGY_DEFICIT),
    );
    let color = favg * favg * e_avg / denom;
    let out = color * f_ms;
    return max(out, vec3<f32>(0.0));
}

// Specular anti-aliasing width filter (`filter_alpha_sq`), in the `alpha^2`
// domain with the Tokuyoshi-Kaplanyan kernel cap.
fn kernel_roughness_sq(variance: f32, max_kernel: f32) -> f32 {
    let v = max(variance, 0.0);
    let cap = max(max_kernel, 0.0);
    return min(2.0 * v, cap);
}

fn filter_alpha_sq(base_alpha_sq: f32, variance: f32, max_kernel: f32) -> f32 {
    let base = max(base_alpha_sq, 0.0);
    let eff = base + kernel_roughness_sq(variance, max_kernel);
    let floor_value = MIN_ALPHA * MIN_ALPHA;
    return clamp(eff, floor_value, 1.0);
}

// The reusable shading function: the WGSL twin of
// `evaluate_principled_direct` restricted to the base lobe (see MIRROR SCOPE).
fn evaluate_principled_direct(s: SurfaceParams) -> vec3<f32> {
    let n = normalize_or(s.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = normalize_or(s.view, n);
    let l = normalize_or(s.direction, n);
    let h = normalize_or(vadd(v, l), n);
    let n_dot_l = max(vdot(n, l), 0.0);
    let n_dot_v = max(vdot(n, v), MIN_N_DOT);
    let n_dot_h = max(vdot(n, h), MIN_N_DOT);
    let v_dot_h = max(vdot(v, h), 0.0);
    if (n_dot_l <= 0.0) {
        return s.emissive;
    }

    let metallic = clamp(s.metallic, 0.0, 1.0);
    let roughness = clamp(s.perceptual_roughness, MIN_ROUGHNESS, 1.0);
    let base_alpha = roughness * roughness;
    let alpha = sqrt(filter_alpha_sq(base_alpha * base_alpha, s.normal_variance, KAPPA_MAX));
    let f0 = vmix3(dielectric_f0(s), s.base_color, metallic);
    let f_dielectric = base_dielectric_fresnel_rgb(
        v_dot_h,
        s.specular_ior,
        s.specular_weight,
        s.specular_color,
    );
    let f_metal = base_metal_fresnel_rgb(
        v_dot_h,
        s.base_weight,
        s.base_color,
        s.specular_weight,
        s.specular_color,
    );
    let f = vmix3(f_dielectric, f_metal, metallic);

    let axes = anisotropic_axes(n, s.tangent, s.bitangent, s.anisotropy_rotation);
    let widths = openpbr_anisotropy_to_alpha(alpha, s.anisotropy);
    let ax = widths.x;
    let ay = widths.y;
    let d = distribution_ggx_anisotropic(
        vdot(axes.tangent, h),
        vdot(axes.bitangent, h),
        n_dot_h,
        ax,
        ay,
    );
    let g = visibility_smith_ggx_anisotropic(
        vdot(axes.tangent, v),
        vdot(axes.bitangent, v),
        vdot(axes.tangent, l),
        vdot(axes.bitangent, l),
        n_dot_v,
        n_dot_l,
        ax,
        ay,
    );
    let single_scatter = vscale(f, d * g);

    let compensation = multiscatter_direct(f0, n_dot_v, n_dot_l, roughness);
    let specular = vadd(single_scatter, compensation);
    let diffuse_weight = vscale(vsub(vec3<f32>(1.0), f), 1.0 - metallic);
    let diffuse_factor = oren_nayar_factor(n_dot_v, n_dot_l, vdot(v, l), s.diffuse_roughness);
    let diffuse = vmul(
        vscale(s.base_color, (1.0 / PI) * diffuse_factor),
        diffuse_weight,
    );
    // `base_weight` scales the whole base layer; `1.0` is exact.
    let base = vscale(vadd(diffuse, specular), n_dot_l * clamp(s.base_weight, 0.0, 1.0));

    // coat and transmission are identity at 0, so `reflected == base`.
    let radiance = vmul(base, s.illuminance);
    return vadd(vscale(radiance, clamp(s.visibility, 0.0, 1.0)), s.emissive);
}

// The batch driver: one invocation per surface evaluation, results written to
// a storage buffer. A raster surface pass composes the same
// `evaluate_principled_direct` function from its fragment stage.
@compute @workgroup_size(64)
fn evaluate_surfaces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = arrayLength(&results);
    if (gid.x >= count) {
        return;
    }
    results[gid.x] = vec4<f32>(evaluate_principled_direct(params[gid.x]), 1.0);
}
