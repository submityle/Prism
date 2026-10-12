//! GPU cloth (fabric) BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::cloth::evaluate_cloth_direct`.  The closure follows
//! the physically based cloth model popularised by Filament / Unreal:
//!
//! * sheen specular = Estevez-Kulla *Charlie* distribution * Ashikhmin/Neubelt
//!   visibility, tinted by the scalar sheen weight, with **no Fresnel** term,
//! * diffuse = Lambertian base, optionally softened by an energy-conserving
//!   wrap term and tinted toward the base color when subsurface > 0.
//!
//! `pow`, `dot`, `max`, `clamp` and `inverseSqrt` are WESL builtins.

import prism_render_scene::shaders::lighting::{
    PI,
    INV_PI,
};
import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};

// Floors mirroring `cloth.rs` (`MIN_ROUGHNESS`, `MIN_N_DOT`, `MIN_SIN2`).
const CLOTH_MIN_ROUGHNESS: f32 = 0.045;
const CLOTH_MIN_N_DOT: f32 = 1.0e-5;
const CLOTH_MIN_SIN2: f32 = 0.0078125;
const CLOTH_WRAP_WIDTH: f32 = 0.5;

// Estevez-Kulla *Charlie* sheen distribution. Mirrors `distribution_charlie`.
const FUZZ_N_MU: i32 = 17;
const FUZZ_N_R: i32 = 9;
const FUZZ_ALBEDO: array<f32, 153> = array<f32, 153>(
    1.000000, 1.000000, 0.649052, 0.282295, 0.116762, 0.044961, 0.015821, 0.004987, 0.001375, 0.000322, 0.000061, 0.000009, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000,
    1.000000, 0.868676, 0.677266, 0.538730, 0.431285, 0.345282, 0.275282, 0.217819, 0.170482, 0.131497, 0.099501, 0.073411, 0.052347, 0.035572, 0.022467, 0.012500, 0.005209,
    0.928916, 0.716657, 0.595808, 0.504953, 0.431375, 0.369553, 0.316454, 0.270209, 0.229555, 0.193579, 0.161614, 0.133140, 0.107750, 0.085115, 0.064965, 0.047073, 0.031251,
    0.810533, 0.646225, 0.552173, 0.480724, 0.422060, 0.371965, 0.328132, 0.289150, 0.254069, 0.222217, 0.193104, 0.166352, 0.141669, 0.118817, 0.097613, 0.077898, 0.059546,
    0.744316, 0.604668, 0.524836, 0.463977, 0.413764, 0.370588, 0.332490, 0.298281, 0.267155, 0.238550, 0.212042, 0.187323, 0.164145, 0.142310, 0.121656, 0.102055, 0.083397,
    0.701662, 0.577044, 0.506022, 0.451871, 0.407107, 0.368500, 0.334302, 0.303441, 0.275201, 0.249079, 0.224705, 0.201788, 0.180111, 0.159499, 0.139803, 0.120900, 0.102697,
    0.671750, 0.557241, 0.492234, 0.442722, 0.401779, 0.366425, 0.335052, 0.306665, 0.280621, 0.256435, 0.233778, 0.212381, 0.192041, 0.172592, 0.153896, 0.135837, 0.118325,
    0.649554, 0.542347, 0.481705, 0.435588, 0.397465, 0.364541, 0.335312, 0.308831, 0.284496, 0.261864, 0.240608, 0.220482, 0.201294, 0.182884, 0.165127, 0.147906, 0.131134,
    0.632425, 0.530705, 0.473381, 0.429855, 0.393906, 0.362883, 0.335331, 0.310368, 0.287407, 0.266029, 0.245935, 0.226876, 0.208675, 0.191184, 0.174267, 0.157825, 0.141770,
);

// Bilinear `E_fuzz(µ, r)`; arguments clamp to `[0, 1]`.
fn fuzz_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = clamp(n_dot_v, 0.0, 1.0) * f32(FUZZ_N_MU - 1);
    let y = clamp(roughness, 0.0, 1.0) * f32(FUZZ_N_R - 1);
    let i0 = min(i32(x), FUZZ_N_MU - 2);
    let j0 = min(i32(y), FUZZ_N_R - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = FUZZ_ALBEDO[j0 * FUZZ_N_MU + i0];
    let b = FUZZ_ALBEDO[j0 * FUZZ_N_MU + i0 + 1];
    let c = FUZZ_ALBEDO[(j0 + 1) * FUZZ_N_MU + i0];
    let d = FUZZ_ALBEDO[(j0 + 1) * FUZZ_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

fn cloth_distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = max(1.0 - n_dot_h * n_dot_h, CLOTH_MIN_SIN2);
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) * (1.0 / (2.0 * PI));
}

// Ashikhmin/Neubelt visibility term. Mirrors `visibility_ashikhmin`.
fn cloth_visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), CLOTH_MIN_N_DOT);
}

// Energy-conserving diffuse wrap. Mirrors `fd_wrap`.
fn cloth_fd_wrap(x: f32, w: f32) -> f32 {
    return clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0);
}

// Cloth BSDF for a single analytic light. Mirrors `evaluate_cloth_direct`.
fn cloth_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);

    let n_dot_l = max(dot(n, l), 0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }
    let n_dot_v = max(dot(n, v), CLOTH_MIN_N_DOT);
    let n_dot_h = max(dot(n, h), CLOTH_MIN_N_DOT);
    // Fuzz geometry uses the lobe's own `OpenPBR` `fuzz_roughness`; the base is
    // Lambertian and its `perceptual_roughness` no longer leaks into the fuzz.
    let fuzz_roughness = clamp(surface.fuzz_roughness, CLOTH_MIN_ROUGHNESS, 1.0);

    // Sheen specular lobe (no Fresnel), tinted by `OpenPBR` `fuzz_color`.
    let sheen = clamp(surface.sheen, 0.0, 1.0);
    let d = cloth_distribution_charlie(fuzz_roughness, n_dot_h);
    let vis = cloth_visibility_ashikhmin(n_dot_v, n_dot_l);
    let sheen_specular = surface.fuzz_color * (sheen * d * vis);
    // `OpenPBR` `fuzz-layering-approx`: the substrate loses exactly the fuzz
    // layer's albedo (`F·E_fuzz`), so the sheen cannot over-reflect. `E_fuzz` is
    // the baked Charlie lobe albedo. Mirrors `cloth.rs`.
    let e_fuzz = fuzz_albedo_lookup(n_dot_v, fuzz_roughness);
    let base_scale = clamp(1.0 - sheen * e_fuzz, 0.0, 1.0);

    // Diffuse lobe, optionally softened + tinted by the subsurface weight.
    let subsurface = clamp(surface.subsurface, 0.0, 1.0);
    var diffuse = surface.base_color * INV_PI;
    if subsurface > 0.0 {
        diffuse = diffuse * cloth_fd_wrap(n_dot_l, CLOTH_WRAP_WIDTH);
        let tint = clamp(
            surface.base_color * subsurface + vec3<f32>(n_dot_l),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        diffuse = diffuse * tint;
    }

    let radiance = (diffuse * base_scale + sheen_specular) * light.illuminance
        * (n_dot_l * clamp(light.visibility, 0.0, 1.0));
    return radiance + surface.emissive;
}
