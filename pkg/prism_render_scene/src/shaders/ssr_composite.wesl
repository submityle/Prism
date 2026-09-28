//! SSR composite: blend the traced reflection buffer over the shaded scene.
//!
//! The trace (`ssr.wesl`) writes `rgb` = reflected radiance and `a` = a `[0, 1]`
//! blend confidence into its reflection output. This stage folds that reflection
//! back into the shaded HDR `scene_color` under an energy-conserving (split-sum)
//! substitution rather than a naive lerp:
//!
//!     scene = base + (reflection * spec_weight - env_specular) * confidence
//!
//! where `env_specular` is the IBL specular the resolve already folded into
//! `scene_color` at this pixel and `spec_weight = (f0 * dfg.x + dfg.y) *
//! occlusion` is the environment-BRDF factor it used. A fully confident hit
//! (`a == 1`) swaps that IBL specular for the screen-space reflection under the
//! identical Fresnel/DFG weight (no double counting); a miss (`a == 0`) leaves
//! the shaded colour untouched, with the trace's fades ramping between the two.
//!
//! `scene_color` is written as an `rgba16float` storage image, but `rgba16float`
//! is not read-write storage-capable, so the *base* colour is read from colour-
//! pyramid level 0 instead — a byte-for-byte copy of `scene_color` taken by the
//! colour-mip build immediately before this pass. Reading the base from a
//! distinct texture and writing `scene_color` keeps the pass free of any
//! read/write aliasing hazard on a single resource.

// Immediate (push-constant) block. Mirrors `GpuSsrCompositeParams` (16 bytes:
// the framebuffer extent then two padding `u32`s).
struct CompositeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

// Colour-pyramid level 0 — the untouched shaded colour (a copy of `scene_color`).
@group(0) @binding(0) var color_l0: texture_2d<f32>;
// Reflection output: rgb = reflected radiance, a = blend confidence.
@group(0) @binding(1) var ssr_out: texture_2d<f32>;
// Shaded scene colour, blended in place with the reflection.
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// SSR energy-conservation inputs from the shading resolve. env_specular = the
// IBL specular radiance already folded into scene_color at this pixel;
// spec_weight = the split-sum environment-BRDF factor it was scaled by.
@group(0) @binding(3) var env_specular: texture_2d<f32>;
@group(0) @binding(4) var spec_weight: texture_2d<f32>;

var<immediate> params: CompositeParams;

@compute @workgroup_size(8, 8, 1)
fn ssr_composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    // Base radiance from the pyramid copy (a 1:1 lift of `scene_color`).
    let base = textureLoad(color_l0, coord, 0);
    let reflection = textureLoad(ssr_out, coord, 0);
    let confidence = clamp(reflection.a, 0.0, 1.0);

    // Energy-conserving substitution (split-sum): at a confident hit swap the
    // resolve-added IBL specular for the screen-space reflection, both weighted
    // by the identical environment-BRDF factor, so no specular energy is double
    // counted. A miss (confidence == 0) leaves the shaded colour untouched.
    //
    //   scene = base + (reflection * spec_weight - env_specular) * confidence
    //
    // env_specular already carries the resolve occlusion/DFG weighting, and
    // spec_weight = (f0 * dfg.x + dfg.y) * occlusion, so the SSR reflection is
    // Fresnel- and roughness-correct against the surface it lands on.
    let ibl_specular = textureLoad(env_specular, coord, 0).rgb;
    let weight = textureLoad(spec_weight, coord, 0).rgb;
    let delta = (reflection.rgb * weight - ibl_specular) * confidence;
    let resolved = max(base.rgb + delta, vec3<f32>(0.0));
    textureStore(scene_color, coord, vec4<f32>(resolved, base.a));
}
