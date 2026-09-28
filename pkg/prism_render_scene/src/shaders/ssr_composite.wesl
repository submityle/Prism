//! SSR composite: blend the traced reflection buffer over the shaded scene.
//!
//! The trace (`ssr.wesl`) writes `rgb` = reflected radiance and `a` = a `[0, 1]`
//! blend confidence into its reflection output. This stage folds that reflection
//! back into the shaded HDR `scene_color` the main pass presents:
//!
//!     scene_color.rgb = mix(scene_color.rgb, reflection.rgb, reflection.a)
//!
//! so a fully confident mirror hit (`a == 1`) replaces the surface radiance with
//! the reflection and a miss (`a == 0`) leaves the shaded colour untouched, with
//! the trace's fades ramping smoothly between the two.
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
    let blended = mix(base.rgb, reflection.rgb, clamp(reflection.a, 0.0, 1.0));
    textureStore(scene_color, coord, vec4<f32>(blended, base.a));
}
