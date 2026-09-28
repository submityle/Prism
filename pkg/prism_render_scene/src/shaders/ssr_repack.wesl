//! SSR material-roughness repack compute entry point.
//!
//! The screen-space trace (`ssr.wesl`) samples a single `normal_roughness`
//! texture: `rgb = view_normal * 0.5 + 0.5` (decoded as `nr.xyz * 2 - 1`) and
//! `a = perceptual roughness`. The geometry prepass (`ssr_prepass.wesl`) already
//! rebuilt the *signed* view-space normal, but it has no material bind group and
//! therefore cannot fetch roughness. This stage bridges the gap: it re-reads the
//! prepass normal, looks the covered pixel's material up through the shared
//! material tables (identical generation/bounds guards to `shading_resolve.wesl`),
//! and packs both into the trace's `normal_roughness` input, one texel per
//! framebuffer pixel.
//!
//! Roughness is taken from the *authored* surface parameters
//! (`material_parameters[header.parameter_offset].perceptual_roughness`). That is
//! exact for every material without a roughness texture; texture-modulated
//! roughness would need the full bindless `material_sample` + interpolated UV
//! path and is deliberately deferred to a later refinement (the trace only uses
//! roughness for its mip selection and confidence fade, so the base parameter is
//! a faithful first-order input).

import prism_render_scene::shaders::material::{
    PrismMaterialHeader,
    PrismSurfaceParameters,
};

// Sentinel packed into the visibility targets for uncovered pixels. Matches
// `VisibilityPixel::is_valid` on the CPU side, `shading_resolve.wesl` and
// `ssr_prepass.wesl`.
const VISIBILITY_BACKGROUND: u32 = 0xffffffffu;

// Immediate (push-constant) block. Mirrors `GpuSsrRepackParams` in
// `shading/ssr/abi.rs` (16 bytes: the framebuffer extent plus two padding
// scalars to satisfy the 16-byte immediate alignment).
struct RepackParams {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

// --- group 0: visibility inputs + prepass normal in, packed output ---
@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
// Signed unit view-space normal written by `ssr_prepass.wesl`.
@group(0) @binding(2) var view_normal_in: texture_2d<f32>;
// Output: rgb = normal * 0.5 + 0.5, a = perceptual roughness.
@group(0) @binding(3) var normal_roughness_out: texture_storage_2d<rgba16float, write>;

// --- group 1: shared material tables (reused MaterialBindGroup layout) ---
// Only the header + surface-parameter tables are read; the layout may carry
// further bindings (material textures, and in bindless mode the texture/sampler
// arrays), which this kernel simply does not declare.
@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<PrismSurfaceParameters>;

var<immediate> params: RepackParams;

// Perceptual roughness for one covered pixel, or a fully-rough fallback (`1.0`)
// whenever any material generation/bounds guard rejects the sample. A fully
// rough surface drives the trace's `roughness_fade` — and thus SSR confidence —
// to zero, so a stale or absent material reflects nothing rather than smearing
// an incorrect mirror sample. Mirrors the resolve stage's material validation
// (`shading_resolve.wesl`): `metadata.x` is the material index, `metadata.y` the
// expected generation.
fn resolve_roughness(metadata: vec4<u32>) -> f32 {
    let material_index = metadata.x;
    if material_index >= arrayLength(&material_headers) {
        return 1.0;
    }
    let header = material_headers[material_index];
    if header.is_active == 0u || header.generation != metadata.y {
        return 1.0;
    }
    let parameter_row = header.parameter_offset;
    if parameter_row >= arrayLength(&material_parameters) {
        return 1.0;
    }
    return clamp(material_parameters[parameter_row].perceptual_roughness, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn ssr_repack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    let ids = textureLoad(visibility_ids, coord, 0);
    let metadata = textureLoad(visibility_metadata, coord, 0);
    if ids.x == VISIBILITY_BACKGROUND || metadata.x == VISIBILITY_BACKGROUND {
        // Background: the prepass stored `0` device depth here, so the trace's
        // reverse-Z early-out drops the pixel before it ever reads this texel;
        // still, keep it well-defined at zero.
        textureStore(normal_roughness_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // The prepass already reconstructed and normalized the signed view-space
    // normal; re-encode it into the trace's [0, 1] convention.
    let view_normal = textureLoad(view_normal_in, coord, 0).xyz;
    let roughness = resolve_roughness(metadata);

    let encoded_normal = view_normal * 0.5 + vec3<f32>(0.5);
    textureStore(normal_roughness_out, coord, vec4<f32>(encoded_normal, roughness));
}
