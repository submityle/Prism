//! Glossy-specular ReSTIR composite (UE-style option C).
//!
//! Folds the glossy-ReSTIR *reuse* pass's resolved specular back into the shaded
//! HDR `scene_color` under an energy-conserving substitution:
//!
//!     scene = base + (contrib - env_specular) * confidence
//!
//! where `contrib` is the reuse pass's resolved glossy contribution (already
//! BRDF-weighted: `throughput * radiance * W`, so it is directly comparable to
//! the IBL specular radiance and is *not* scaled by the split-sum env-BRDF
//! again), `env_specular` is the IBL specular the shading resolve already folded
//! into `scene_color` at this pixel, and `confidence` is the reservoir's
//! normalised sample count. A hit (`confidence > 0`) replaces the IBL specular
//! with the screen-space glossy reflection; a miss (`confidence == 0`) leaves
//! the resolve's IBL specular untouched — UE's "SSR/Lumen replaces env specular
//! under confidence, falls back to IBL on a miss".
//!
//! This stage owns the specular slot whenever the glossy-ReSTIR subsystem is
//! enabled; the SSR composite early-returns so the IBL specular is substituted
//! exactly once. `scene_color` is an `rgba16float` storage image (not
//! read/write capable), so the *base* colour is lifted from colour-pyramid
//! level 0 — a byte-for-byte copy of `scene_color` the colour-mip build took
//! immediately before this pass — and only the write side touches `scene_color`,
//! keeping the pass free of any read/write aliasing hazard.

struct CompositeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

// Colour-pyramid level 0 — the untouched shaded colour (a copy of `scene_color`).
@group(0) @binding(0) var color_l0: texture_2d<f32>;
// Reuse pass's resolved target: rgb = glossy contribution, a = confidence.
@group(0) @binding(1) var resolved: texture_2d<f32>;
// Shaded scene colour, blended in place with the glossy reflection.
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// IBL specular radiance the shading resolve already folded into scene_color at
// this pixel (subtracted under confidence so no double-count).
@group(0) @binding(3) var env_specular: texture_2d<f32>;

var<immediate> params: CompositeParams;

@compute @workgroup_size(8, 8, 1)
fn spec_gi_composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    // Base radiance from the pyramid copy (a 1:1 lift of `scene_color`).
    let base = textureLoad(color_l0, coord, 0);
    let r = textureLoad(resolved, coord, 0);
    let contrib = max(r.rgb, vec3<f32>(0.0));
    let confidence = clamp(r.a, 0.0, 1.0);

    // Energy-conserving substitution (option C): swap the resolve-added IBL
    // specular for the glossy ReSTIR reflection under confidence. `contrib` is
    // already BRDF-weighted, so it stands in for `env_specular` directly.
    let ibl_specular = textureLoad(env_specular, coord, 0).rgb;
    let delta = (contrib - ibl_specular) * confidence;
    let resolved_rgb = max(base.rgb + delta, vec3<f32>(0.0));
    textureStore(scene_color, coord, vec4<f32>(resolved_rgb, base.a));
}
