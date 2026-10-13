//! The measured (tabulated) BRDF/BTF lobe's per-light response (§17.6) — GPU
//! twin of `prism_render_shading::measured_lobe`.
//!
//! `measured_brdf.wesl` is the kernel (Rusinkiewicz coordinates, grid mapping,
//! trilinear interpolation). This module is the *lobe* half: it fetches the
//! table's eight bracketing texels from the material texture heap and turns the
//! interpolated BRDF into a radiance contribution for one direct light.
//!
//! # The provider
//!
//! A measured table is data, so it ships as an ordinary material texture bound
//! with `SEMANTIC_MEASURED_BRDF`, laid out as an atlas:
//!
//! ```text
//! texel (i, j * phi_d + k)  =  RGB of grid cell (i, j, k)
//! width  = theta_h resolution
//! height = theta_d resolution * phi_d resolution
//! ```
//!
//! so `phi_d` (the one axis a texture's dimensions cannot express) is the lobe
//! field `measured_phi_d`, and `theta_d` follows from the height. Integer
//! `textureLoad`s (never filtered samples) keep the trilinear interpolation the
//! kernel's own — the hardware must not pre-filter a BRDF table for us.

import prism_render_scene::shaders::material_sample::{
    material_texture_dimensions,
    material_texture_load,
};
import prism_render_scene::shaders::brdf::{
    DirectLightSample,
    ShadingFrame,
};
import prism_render_scene::shaders::measured_brdf::{
    mb_directions_to_rusinkiewicz,
    mb_grid_coord_phi_d,
    mb_grid_coord_theta_d,
    mb_grid_coord_theta_h,
    mb_grid_frac,
    mb_grid_lower,
    mb_trilinear,
};

/// `value` normalised, falling back to `fallback` when it is degenerate.
fn measured_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if (length_squared > 1.0e-12) {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

/// Semantic of the measured-BRDF table row (§17.6). Mirrors
/// `MEASURED_BRDF_SEMANTIC` in `prism_render_material`.
const SEMANTIC_MEASURED_BRDF: u32 = 9u;

/// One table texel as RGB, from the atlas at `(i, j, k)`.
fn measured_texel(slot: u32, i: i32, j: i32, k: i32, phi_d: i32) -> vec3<f32> {
    let coord = vec2<i32>(i, j * phi_d + k);
    return material_texture_load(slot, coord).xyz;
}

/// Evaluate the measured lobe for one direct light.
///
/// `frame` supplies the surface basis; the kernel works in a local frame whose
/// `+Z` is the shading normal and `+X` the tangent, so both directions are
/// rotated into it here. Returns zero for an inert weight, a below-horizon
/// light (the table is only defined for the upper hemisphere) or a degenerate
/// frame, so a material that never authors the lobe is an exact identity.
fn measured_direct(
    table_slot: u32,
    phi_d: f32,
    scale: f32,
    weight: f32,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let w = clamp(weight, 0.0, 1.0);
    if (w <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let n = measured_normalize_or(frame.normal, vec3<f32>(0.0, 0.0, 1.0));
    let v = measured_normalize_or(frame.view, n);
    let l = measured_normalize_or(light.direction, n);
    let n_dot_l = dot(n, l);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // The non-bindless specialization returns an identity texel from
    // `material_texture_load`; its measured lobe is therefore inert without
    // ever naming or indexing a binding array.
    let dimensions = vec2<f32>(material_texture_dimensions(table_slot));
    let phi_count = max(floor(phi_d), 2.0);
    let theta_d_count = max(floor(dimensions.y / phi_count), 2.0);

    let wi = vec3<f32>(dot(frame.tangent, l), dot(frame.bitangent, l), dot(n, l));
    let wo = vec3<f32>(dot(frame.tangent, v), dot(frame.bitangent, v), dot(n, v));
    let coords = mb_directions_to_rusinkiewicz(wi, wo);

    let grid_h = mb_grid_coord_theta_h(coords.x, dimensions.x);
    let grid_d = mb_grid_coord_theta_d(coords.y, theta_d_count);
    let grid_p = mb_grid_coord_phi_d(coords.z, phi_count);

    let i0 = i32(mb_grid_lower(grid_h, dimensions.x));
    let j0 = i32(mb_grid_lower(grid_d, theta_d_count));
    let k0 = i32(mb_grid_lower(grid_p, phi_count));
    let i1 = min(i0 + 1, i32(dimensions.x) - 1);
    let j1 = min(j0 + 1, i32(theta_d_count) - 1);
    let k1 = min(k0 + 1, i32(phi_count) - 1);

    let c000 = measured_texel(table_slot, i0, j0, k0, i32(phi_count));
    let c001 = measured_texel(table_slot, i0, j0, k1, i32(phi_count));
    let c010 = measured_texel(table_slot, i0, j1, k0, i32(phi_count));
    let c011 = measured_texel(table_slot, i0, j1, k1, i32(phi_count));
    let c100 = measured_texel(table_slot, i1, j0, k0, i32(phi_count));
    let c101 = measured_texel(table_slot, i1, j0, k1, i32(phi_count));
    let c110 = measured_texel(table_slot, i1, j1, k0, i32(phi_count));
    let c111 = measured_texel(table_slot, i1, j1, k1, i32(phi_count));

    let f = mb_trilinear(
        c000,
        c001,
        c010,
        c011,
        c100,
        c101,
        c110,
        c111,
        mb_grid_frac(grid_h, dimensions.x),
        mb_grid_frac(grid_d, theta_d_count),
        mb_grid_frac(grid_p, phi_count),
    );

    let energy = n_dot_l * w * max(scale, 0.0) * clamp(light.visibility, 0.0, 1.0);
    return max(f, vec3<f32>(0.0, 0.0, 0.0)) * energy * light.illuminance;
}
