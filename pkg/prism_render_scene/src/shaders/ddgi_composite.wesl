//! DDGI composite: fold the resolved diffuse-GI irradiance volume over the
//! shaded scene under an energy-conserving substitution.
//!
//! The DDGI sample pass (`ddgi_sample.wesl` `sample_main`) reconstructs each
//! pixel's enclosing probe cell, blends the eight corner probes' octahedral
//! irradiance under combined trilinear x back-face x Chebyshev-visibility
//! weights, and writes `rgb` = the pre-albedo diffuse-GI irradiance (times the
//! artistic gain) and `a` = a `[0, 1]` blend confidence (zero where the probe
//! cage misses or the shading point leaves the volume) into `gi_out`.
//!
//! The shading resolve has already folded the *pure* IBL / SH diffuse ambient
//! into `scene_color` as `albedo * ambient` (with `ambient` the pre-albedo
//! diffuse irradiance the resolve also exports at `ssgi_ambient`). This stage
//! performs the same energy-conserving substitution SSGI and world-space GI do,
//! swapping that flat ambient for the irradiance-volume gather under the pass
//! confidence (UE's DDGI-replaces-sky-diffuse behaviour, falling back to IBL on
//! a miss):
//!
//!     scene = base + confidence * albedo * (gi_out - ambient)
//!
//! A fully confident pixel (`a == 1`) replaces the flat ambient with the probe
//! gather; a miss (`a == 0`) leaves the shaded colour untouched, preserving the
//! IBL ambient; a sky / unshaded pixel has `albedo == 0`, so the substitution
//! is a no-op there regardless of confidence. `gi_out.rgb` is exactly the
//! golden irradiance-volume evaluation (the CPU golden has no composite step of
//! its own), so this fold matches golden semantics and never creates energy.
//!
//! `scene_color` is written as an `rgba16float` storage image, but
//! `rgba16float` is not read-write storage-capable, so the *base* colour is
//! lifted into a scratch `gi_base` texture by a preceding copy pass (a
//! byte-for-byte 1:1 copy of the shading-resolved `scene_color`) and only the
//! write side of the fold touches `scene_color`. Reading the base from a
//! distinct texture and writing `scene_color` keeps the pass free of any
//! storage read/write aliasing hazard on a single resource.

// Immediate (push-constant) block. Mirrors `GpuDdgiCompositeParams`
// (16 bytes: the framebuffer extent then two padding `u32`s).
struct CompositeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

var<immediate> params: CompositeParams;

// --- copy pass: lift the shading-resolved scene_color into the GI base
// scratch. `scene_color` (read) is the composited HDR colour the prior passes
// produced; the copy avoids the read/write hazard the fold would otherwise hit
// on a single storage image.
@group(0) @binding(0) var copy_scene_color: texture_2d<f32>;
@group(0) @binding(1) var copy_gi_base: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn ddgi_copy_base(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(copy_gi_base, coord, textureLoad(copy_scene_color, coord, 0));
}

// --- fold pass: apply the energy-conserving GI substitution into scene_color.
// The base copy (a 1:1 lift of the shading-resolved scene_color).
@group(0) @binding(0) var gi_base: texture_2d<f32>;
// DDGI sample output: rgb = pre-albedo diffuse-GI irradiance, a = confidence.
@group(0) @binding(1) var gi_out: texture_2d<f32>;
// Shaded scene colour, blended in place with the indirect irradiance.
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// Surface Lambertian albedo (base_color * (1 - metallic)) the resolve exported.
@group(0) @binding(3) var albedo: texture_2d<f32>;
// Pre-albedo IBL / SH diffuse irradiance the shading resolve already folded
// into scene_color as albedo * ambient; the substitution removes it under
// confidence.
@group(0) @binding(4) var ambient: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn ddgi_composite(@builtin(global_invocation_id) id: vec3<u32>) {
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
    // `albedo * ambient` into scene_color, so replacing it with
    // `albedo * gi_out` under the gather confidence adds exactly
    // `confidence * albedo * (gi_out - ambient)`. A miss (confidence == 0) is a
    // no-op, preserving the IBL ambient; gi_out is the golden irradiance, so no
    // energy is created.
    let surf_albedo = textureLoad(albedo, coord, 0).rgb;
    let sky = textureLoad(ambient, coord, 0).rgb;
    let delta = confidence * surf_albedo * (gather.rgb - sky);
    let resolved = max(base.rgb + delta, vec3<f32>(0.0));
    textureStore(scene_color, coord, vec4<f32>(resolved, base.a));
}
