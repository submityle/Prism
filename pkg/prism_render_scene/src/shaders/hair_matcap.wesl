//! GPU stylized (NPR) `MatCap` (material-capture / lit-sphere) body shade for
//! anime hair.
//!
//! Numerical twin of the CPU golden
//! `prism_render_shading::hair_matcap` (`hair_matcap_uv` /
//! `evaluate_hair_matcap`). A `MatCap` bakes a whole lighting environment plus
//! material response into one pre-lit sphere texture indexed by the view-space
//! surface normal; it supplies the soft body/ambient shade that the additive
//! `hair_angel_ring.wesl` highlight rings and the `stylized_hair.wesl` cel ramp
//! compose on top of.
//!
//! Two pieces mirror the CPU reference value-for-value:
//!
//! * `hair_matcap_uv` rebuilds a screen basis from the view direction and a
//!   world-space `up_hint` (`right = normalize(cross(up, v))`,
//!   `up = cross(v, right)`), projects the normal onto it and maps the
//!   `[-1, 1]` projection to a clamped `[0, 1]` `MatCap` coordinate. A
//!   degenerate `up_hint` falls back to the frame tangent, then `+X`, so the
//!   basis never collapses.
//! * `hair_matcap_shade` modulates the sampled `MatCap` radiance by the surface
//!   base color, the authored tint and `max(strength, 0)`, then adds emissive
//!   exactly once.
//!
//! The texture fetch itself is a hardware `textureSampleLevel` through a linear
//! clamp-to-edge sampler; the CPU golden mirrors that filter with
//! `MatCapTexture::sample_bilinear`, so the sampled `vec3` is passed into
//! `hair_matcap_shade` here rather than re-deriving the bilinear weights on the
//! ALU. `dot`, `cross`, `clamp`, `max` are WESL builtins; `brdf_normalize_or`
//! (same `1e-12` degeneracy floor as the CPU `normalize_or`) is shared with the
//! principled lobe. Keeping the sampler out of this module avoids pinning a
//! bind-group layout before the hair resolve pass owns one.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    brdf_normalize_or,
};

// Squared-length floor below which a reconstructed basis vector is treated as
// degenerate. Mirrors `BASIS_EPSILON` and the shared `brdf_normalize_or` floor.
const HAIR_MATCAP_BASIS_EPSILON: f32 = 1e-12;

// Authoring parameters for the hair `MatCap` body shade. Field-for-field twin
// of the CPU `HairMatCapParams` (scalar grouped between the two `vec3`s to keep
// std140-friendly packing predictable).
struct HairMatCapParams {
    // Extra linear tint multiplied onto the sampled `MatCap` radiance.
    tint: vec3<f32>,
    // Scales the sampled radiance; 0 disables the body shade.
    strength: f32,
    // World-space up hint used to rebuild the screen basis.
    up_hint: vec3<f32>,
}

// Reconstructs the `MatCap` texture coordinate for one shaded fragment. Mirrors
// `hair_matcap_uv`: builds a screen basis from the view direction and up hint,
// projects the normal onto it, and maps the [-1, 1] projection to a clamped
// [0, 1]. A degenerate up hint falls back to the frame tangent, then +X.
fn hair_matcap_uv(frame: ShadingFrame, up_hint: vec3<f32>) -> vec2<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    var right = cross(up_hint, v);
    if dot(right, right) < HAIR_MATCAP_BASIS_EPSILON {
        right = cross(frame.tangent, v);
    }
    let right_n = brdf_normalize_or(right, vec3<f32>(1.0, 0.0, 0.0));
    let up = brdf_normalize_or(cross(v, right_n), vec3<f32>(0.0, 1.0, 0.0));
    let x = dot(n, right_n);
    let y = dot(n, up);
    return vec2<f32>(
        clamp(x * 0.5 + 0.5, 0.0, 1.0),
        clamp(y * 0.5 + 0.5, 0.0, 1.0),
    );
}

// Modulates the sampled `MatCap` radiance into the final body shade. Mirrors
// `evaluate_hair_matcap`'s folding: sampled * base_color * tint * strength plus
// emissive added exactly once. `matcap_color` is the hardware
// `textureSampleLevel` result (linear clamp-to-edge filter) at the UV from
// `hair_matcap_uv`.
fn hair_matcap_shade(
    surface: SurfaceSample,
    params: HairMatCapParams,
    matcap_color: vec3<f32>,
) -> vec3<f32> {
    let strength = max(params.strength, 0.0);
    let tinted = matcap_color * surface.base_color * params.tint;
    let body = tinted * strength;
    return body + surface.emissive;
}

// Convenience front end mirroring the CPU `evaluate_hair_matcap(surface, frame,
// params, matcap)`: reconstructs the UV, then shades the supplied sampled
// `MatCap` radiance. The caller performs the `textureSampleLevel` with the UV
// returned by `hair_matcap_uv` and feeds the result back in as `matcap_color`.
fn evaluate_hair_matcap(
    surface: SurfaceSample,
    frame: ShadingFrame,
    params: HairMatCapParams,
    matcap_color: vec3<f32>,
) -> vec3<f32> {
    return hair_matcap_shade(surface, params, matcap_color);
}
