//! SSGI composite: fold the traced indirect-diffuse buffer over the shaded
//! scene under energy conservation.
//!
//! The gather (`ssgi.wesl`) writes `rgb` = the pre-albedo mean indirect
//! radiance and `a` = a `[0, 1]` blend confidence into its output. The resolve
//! has already folded the *pure* IBL/SH ambient into `scene_color` as
//! `albedo * ambient` (with `ambient` the pre-albedo diffuse irradiance this
//! pass also exports at `ssgi_ambient`). This stage swaps that ambient for the
//! screen-space gather under the pass confidence, an energy-conserving
//! substitution rather than a naive add:
//!
//!     scene = base + confidence * albedo * (ssgi_out - ambient)
//!
//! A fully confident pixel (`a == 1`) replaces the flat ambient with the
//! screen-space indirect radiance; a miss (`a == 0`) leaves the shaded colour
//! untouched. On a full miss the gather returns `ssgi_out ~= ambient`, so the
//! delta collapses to zero and no energy is created.
//!
//! `scene_color` is written as an `rgba16float` storage image, but
//! `rgba16float` is not read-write storage-capable, so the *base* colour is
//! lifted into a scratch `gi_base` texture by a preceding copy pass (a
//! byte-for-byte 1:1 copy of the SSR-composited `scene_color`) and only the
//! write side of the fold touches `scene_color`. Reading the base from a
//! distinct texture and writing `scene_color` keeps the pass free of any
//! storage read/write aliasing hazard on a single resource.

// Immediate (push-constant) block. Mirrors `GpuSsgiCompositeParams` (16 bytes:
// the framebuffer extent then two padding `u32`s).
struct CompositeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

var<immediate> params: CompositeParams;

// --- copy pass: lift the SSR-composited scene_color into the GI base scratch.
// `scene_color` (read) is the composited HDR colour the SSR fold produced; the
// copy avoids the read/write hazard the fold would otherwise hit on a single
// storage image.
@group(0) @binding(0) var copy_scene_color: texture_2d<f32>;
@group(0) @binding(1) var copy_gi_base: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn ssgi_copy_base(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(copy_gi_base, coord, textureLoad(copy_scene_color, coord, 0));
}

// --- fold pass: apply the energy-conserving GI substitution into scene_color.
// The base copy (a 1:1 lift of the composited scene_color).
@group(0) @binding(0) var gi_base: texture_2d<f32>;
// SSGI gather output: rgb = pre-albedo mean indirect radiance, a = confidence.
@group(0) @binding(1) var ssgi_out: texture_2d<f32>;
// Shaded scene colour, blended in place with the indirect radiance.
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// Surface Lambertian albedo (base_color * (1 - metallic)) the resolve exported.
@group(0) @binding(3) var albedo: texture_2d<f32>;
// Pre-albedo IBL/SH diffuse irradiance the resolve already folded into
// scene_color as albedo * ambient; the substitution removes it under confidence.
@group(0) @binding(4) var ambient: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn ssgi_composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    // Base radiance from the scratch copy (a 1:1 lift of the composited
    // scene_color, so any SSR reflection already folded in is preserved).
    let base = textureLoad(gi_base, coord, 0);
    let gather = textureLoad(ssgi_out, coord, 0);
    let confidence = clamp(gather.a, 0.0, 1.0);

    // Energy-conserving substitution: the resolve folded `albedo * ambient`
    // into scene_color, so replacing it with `albedo * ssgi_out` under the
    // gather confidence adds exactly `confidence * albedo * (ssgi_out -
    // ambient)`. A miss (confidence == 0) is a no-op; a full-miss gather returns
    // `ssgi_out ~= ambient`, so the delta collapses to zero and no energy is
    // created.
    let surf_albedo = textureLoad(albedo, coord, 0).rgb;
    let sky = textureLoad(ambient, coord, 0).rgb;
    let delta = confidence * surf_albedo * (gather.rgb - sky);
    let resolved = max(base.rgb + delta, vec3<f32>(0.0));
    textureStore(scene_color, coord, vec4<f32>(resolved, base.a));
}
