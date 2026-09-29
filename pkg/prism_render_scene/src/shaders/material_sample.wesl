//! Bindless material texture sampling for the compute shading resolve.
//!
//! This is the GPU twin of the CPU golden `prism_render_shading::texture_sample`.
//! `shading_resolve.wesl` reconstructs the surface and its interpolated UV, then
//! folds every texture bound by the material's `PrismMaterialHeader` into a
//! `SampledMaterial` before running the analytic BSDF.  The textures live in a
//! bindless `binding_array` indexed by the slot the CPU `BindlessTextureHeap`
//! assigned (`PrismMaterialTexture.index`); the samplers live in a parallel
//! `binding_array<sampler>` indexed by `PrismMaterialTexture.sampler_index`.
//!
//! Reserved slots 0/1/2 are the heap's default white / flat-normal / black
//! fallbacks, so a semantic with no authored texture still resolves to an
//! identity modulation without any per-pixel branch.
//!
//! Color-space conventions match glTF 2.0 / `StandardMaterial`: base-color and
//! emissive are sRGB (decoded to linear here); metallic-roughness is linear with
//! roughness in G, metallic in B; occlusion is linear in R; normal maps are
//! linear, `[0, 1]`-remapped tangent-space normals.
//!
//! Sampling uses `textureSampleLevel(..., 0.0)`: the compute resolve has no
//! screen-space derivatives, so an explicit LOD is mandatory.  `pow`, `dot` and
//! `inverseSqrt` are WGSL builtins used directly.

import prism_render_scene::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};

enable wgpu_binding_array;

// Texture semantics, mirroring `TextureSemantic` (`#[repr(u32)]`) in
// `prism_render_material::bevy_bridge` and `SEMANTIC_*` on the CPU golden.
const SEMANTIC_BASE_COLOR: u32 = 0u;
const SEMANTIC_EMISSIVE: u32 = 1u;
const SEMANTIC_METALLIC_ROUGHNESS: u32 = 2u;
const SEMANTIC_NORMAL: u32 = 3u;
const SEMANTIC_OCCLUSION: u32 = 4u;
const SEMANTIC_CLEAR_COAT: u32 = 5u;
const SEMANTIC_CLEAR_COAT_ROUGHNESS: u32 = 6u;
const SEMANTIC_CLEAR_COAT_NORMAL: u32 = 7u;
const SEMANTIC_FACE_SDF: u32 = 8u;

// Squared-length floor for a decoded normal before it collapses to +Z. Mirrors
// `NORMAL_EPSILON` on the CPU golden and `normalize_or` across the reference.
const NORMAL_EPSILON: f32 = 1e-12;

// --- group 1: bindless material texture heap (shared MaterialBindGroup) ---
// Bindings 0/1/2 are the material header / parameter / texture storage tables
// (declared by the resolve entry point); the heap arrays extend the same group.
@group(1) @binding(3) var material_texture_heap: binding_array<texture_2d<f32>>;
@group(1) @binding(4) var material_sampler_heap: binding_array<sampler>;

// Texture-modulated surface inputs handed to the analytic BSDF. Mirrors
// `SampledMaterial` on the CPU golden field for field.
struct SampledMaterial {
    base_color: vec3<f32>,
    base_alpha: f32,
    emissive: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    occlusion: f32,
    normal_tangent: vec3<f32>,
    has_normal_map: bool,
    clearcoat: f32,
    clearcoat_roughness: f32,
    clearcoat_normal_tangent: vec3<f32>,
    has_clearcoat_normal: bool,
}

// Exact IEC 61966-2-1 sRGB electro-optical transfer function (single channel).
// Mirrors `srgb_channel_to_linear` on the CPU golden.
fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

// Channel-wise sRGB -> linear. Mirrors `srgb_to_linear`.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(c.x),
        srgb_channel_to_linear(c.y),
        srgb_channel_to_linear(c.z),
    );
}

// Decodes a `[0, 1]`-remapped tangent-space normal, scaling XY by
// `normal_scale`, renormalizing, and collapsing a degenerate result to +Z.
// Mirrors `decode_tangent_normal`.
fn decode_tangent_normal(texel: vec3<f32>, normal_scale: f32) -> vec3<f32> {
    let xy = (texel.xy * 2.0 - 1.0) * normal_scale;
    let z = texel.z * 2.0 - 1.0;
    let n = vec3<f32>(xy.x, xy.y, z);
    let len_sq = dot(n, n);
    if len_sq > NORMAL_EPSILON {
        return n * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Seeds a `SampledMaterial` from the authored factors alone, before any texture
// is folded in. Mirrors `sampled_material_defaults`.
fn sampled_material_defaults(parameters: PrismSurfaceParameters) -> SampledMaterial {
    var out: SampledMaterial;
    out.base_color = parameters.base_color.xyz;
    out.base_alpha = parameters.base_color.w;
    out.emissive = parameters.emissive.xyz;
    out.metallic = parameters.metallic;
    out.perceptual_roughness = parameters.perceptual_roughness;
    out.occlusion = parameters.ambient_occlusion;
    out.normal_tangent = vec3<f32>(0.0, 0.0, 1.0);
    out.has_normal_map = false;
    out.clearcoat = parameters.clearcoat;
    out.clearcoat_roughness = parameters.clearcoat_roughness;
    out.clearcoat_normal_tangent = vec3<f32>(0.0, 0.0, 1.0);
    out.has_clearcoat_normal = false;
    return out;
}

// Samples one bound texture record from the bindless heap at LOD 0. The slot
// and sampler indices are the ones the CPU `BindlessTextureHeap` assigned; an
// unbound semantic points at a reserved default slot, never out of range.
fn sample_texture_slot(record: PrismMaterialTexture, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(
        material_texture_heap[record.index],
        material_sampler_heap[record.sampler_index],
        uv,
        0.0,
    );
}

// Folds a single sampled texel into `sampled` according to its semantic,
// returning the updated accumulator. Mirrors `fold_material_texel` arm-for-arm.
fn fold_material_texel(
    record: PrismMaterialTexture,
    texel: vec4<f32>,
    normal_scale: f32,
    sampled: SampledMaterial,
) -> SampledMaterial {
    var out = sampled;
    let rgb = texel.xyz;
    switch record.semantic {
        case SEMANTIC_BASE_COLOR: {
            out.base_color = out.base_color * srgb_to_linear(rgb);
            out.base_alpha = out.base_alpha * texel.w;
        }
        case SEMANTIC_EMISSIVE: {
            out.emissive = out.emissive * srgb_to_linear(rgb);
        }
        case SEMANTIC_METALLIC_ROUGHNESS: {
            out.perceptual_roughness = out.perceptual_roughness * texel.y;
            out.metallic = out.metallic * texel.z;
        }
        case SEMANTIC_OCCLUSION: {
            out.occlusion = out.occlusion * texel.x;
        }
        case SEMANTIC_NORMAL: {
            out.normal_tangent = decode_tangent_normal(rgb, normal_scale);
            out.has_normal_map = true;
        }
        case SEMANTIC_CLEAR_COAT: {
            out.clearcoat = out.clearcoat * texel.x;
        }
        case SEMANTIC_CLEAR_COAT_ROUGHNESS: {
            out.clearcoat_roughness = out.clearcoat_roughness * texel.y;
        }
        case SEMANTIC_CLEAR_COAT_NORMAL: {
            out.clearcoat_normal_tangent = decode_tangent_normal(rgb, normal_scale);
            out.has_clearcoat_normal = true;
        }
        default: {}
    }
    return out;
}
