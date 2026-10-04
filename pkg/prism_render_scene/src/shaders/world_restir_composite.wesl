//! World-space `ReSTIR` composite: fold the resolved direct illumination over
//! the shaded scene under an energy-conserving substitution.
//!
//! The resolve (`world_restir_resolve.wesl`) writes `rgb` = the pre-BRDF,
//! cosine-weighted `ReSTIR` direct irradiance and `a` = a `[0, 1]` hit
//! confidence into `gi_out`. The shading resolve already folded the clustered
//! punctual diffuse into `scene_color` and, in the same pass, exported that
//! pre-BRDF punctual irradiance at `world_restir_direct` (identical units to
//! `gi_out`: raw `sum max(dot(n, l), 0) * illuminance * visibility`, no `1/pi`,
//! no Fresnel, no albedo). This stage swaps the clustered punctual diffuse for
//! the `ReSTIR` estimate under the hit confidence, an energy-conserving
//! substitution rather than a naive add:
//!
//!     scene = max(base + confidence * albedo * INV_PI * (gi_out - direct), 0)
//!
//! where `albedo = base_color * (1 - metallic)` and `INV_PI` converts the
//! pre-BRDF irradiance to Lambertian diffuse radiance. A fully confident pixel
//! (`a == 1`) replaces the clustered punctual diffuse with the `ReSTIR`
//! estimate; a miss (`a == 0`) leaves the shaded colour untouched. The fold
//! drops the Fresnel `(1 - F)` factor the shading resolve applied to the
//! diffuse term — the accepted GI-substitution approximation shared with SSGI /
//! world-space GI — so the seam is energy-neutral up to that factor. Specular,
//! directional and area-light contributions are untouched.
//!
//! `scene_color` is written as an `rgba16float` storage image, but
//! `rgba16float` is not read-write storage-capable, so the *base* colour is
//! lifted into a scratch `gi_base` texture by a preceding copy pass (a
//! byte-for-byte 1:1 copy of the shading-resolved `scene_color`) and only the
//! write side of the fold touches `scene_color`.

// Reciprocal of pi, converting pre-BRDF irradiance to Lambertian diffuse
// radiance. Matches the `INV_PI` the shading resolve applies to the diffuse
// term, so the subtracted clustered punctual diffuse cancels (up to Fresnel).
const INV_PI: f32 = 0.318309886183790671538;

// Immediate (push-constant) block. Mirrors `GpuWorldRestirCompositeParams`
// (16 bytes: the framebuffer extent then two padding `u32`s).
struct CompositeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

var<immediate> params: CompositeParams;

// --- copy pass: lift the shading-resolved scene_color into the composite base
// scratch. `scene_color` (read) is the composited HDR colour prior passes
// produced; the copy avoids the read/write hazard the fold would otherwise hit
// on a single storage image.
@group(0) @binding(0) var copy_scene_color: texture_2d<f32>;
@group(0) @binding(1) var copy_gi_base: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn wr_copy_base(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(copy_gi_base, coord, textureLoad(copy_scene_color, coord, 0));
}

// --- fold pass: apply the energy-conserving direct substitution into
// scene_color.
// The base copy (a 1:1 lift of the shading-resolved scene_color).
@group(0) @binding(0) var gi_base: texture_2d<f32>;
// World-space ReSTIR resolve output: rgb = pre-BRDF cosine-weighted direct
// irradiance, a = hit confidence.
@group(0) @binding(1) var gi_out: texture_2d<f32>;
// Shaded scene colour, blended in place with the ReSTIR estimate.
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// Surface Lambertian albedo (base_color * (1 - metallic)) the resolve exported.
@group(0) @binding(3) var albedo: texture_2d<f32>;
// Clustered punctual direct irradiance the shading resolve folded into
// scene_color (pre-BRDF, no 1/pi / Fresnel / albedo); the substitution removes
// it under confidence.
@group(0) @binding(4) var world_restir_direct: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn wr_composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    // Base radiance from the scratch copy (a 1:1 lift of the shading-resolved
    // scene_color, so any prior fold is preserved).
    let base = textureLoad(gi_base, coord, 0);
    let gather = textureLoad(gi_out, coord, 0);
    let confidence = clamp(gather.a, 0.0, 1.0);

    // Energy-conserving substitution: the shading resolve folded
    // `albedo * INV_PI * direct` (times the diffuse Fresnel) into scene_color,
    // so replacing it with `albedo * INV_PI * gi_out` under confidence adds
    // exactly `confidence * albedo * INV_PI * (gi_out - direct)`, dropping only
    // the Fresnel factor. A miss (confidence == 0) is a no-op; both irradiances
    // are non-negative and the result is clamped to non-negative energy.
    let surf_albedo = textureLoad(albedo, coord, 0).rgb;
    let direct = textureLoad(world_restir_direct, coord, 0).rgb;
    let delta = confidence * surf_albedo * INV_PI * (gather.rgb - direct);
    let resolved = max(base.rgb + delta, vec3<f32>(0.0));
    textureStore(scene_color, coord, vec4<f32>(resolved, base.a));
}
