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

@publish import prism_render_material::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};
import prism_render_material::shaders::material_virtual_texture::{
    MaterialVirtualTextureConfig,
    material_vt_atlas_uv,
    material_vt_empty_resolution,
    material_vt_page_request,
    material_vt_resolve,
    material_vt_sample_resolved,
};

@if(MATERIAL_BINDLESS) enable wgpu_binding_array;

// Texture semantics, mirroring `TextureSemantic` (`#[repr(u32)]`) in
// `prism_render_material::bevy_bridge` and `SEMANTIC_*` on the CPU golden.
const SEMANTIC_BASE_COLOR: u32 = 0u;
const SEMANTIC_EMISSIVE: u32 = 1u;
const SEMANTIC_METALLIC_ROUGHNESS: u32 = 2u;
const SEMANTIC_NORMAL: u32 = 3u;
// Explicit linear RGB normal + alpha projected variance σ². Ordinary NORMAL
// alpha is ignored. Mirrors specular_aa::normal_variance on the CPU; not a heap
// slot or a new texture-row member. The producer bakes for the actual normal
// scale; runtime must use that scale and must not also roughness-bake the input.
const SEMANTIC_NORMAL_VARIANCE: u32 = 11u;
const SEMANTIC_OCCLUSION: u32 = 4u;
const SEMANTIC_CLEAR_COAT: u32 = 5u;
const SEMANTIC_CLEAR_COAT_ROUGHNESS: u32 = 6u;
const SEMANTIC_CLEAR_COAT_NORMAL: u32 = 7u;
const SEMANTIC_FACE_SDF: u32 = 8u;

// Squared-length floor for a decoded normal before it collapses to +Z. Mirrors
// `NORMAL_EPSILON` on the CPU golden and `normalize_or` across the reference.
const NORMAL_EPSILON: f32 = 1e-12;

// --- group 1: optional bindless material texture heap ---
// Bindings 0/1/2 are the material header / parameter / texture storage tables
// declared by each consumer. The two heap arrays are emitted only for the
// bindless specialization. This is important: a non-bindless device has no
// binding-array entries in MaterialBindGroup, so merely leaving the arrays
// unused would still make its pipeline layout invalid.
@if(MATERIAL_BINDLESS) @group(1) @binding(3) var material_texture_heap: binding_array<texture_2d<f32>>;
@if(MATERIAL_BINDLESS) @group(1) @binding(4) var material_sampler_heap: binding_array<sampler>;
// Bindless-only slot generation publication. The CPU bind-group owner supplies
// this dense table at binding 9; keeping it behind the same specialization is
// what makes the non-bindless layout free of both arrays and binding 9.
@if(MATERIAL_BINDLESS) @group(1) @binding(9) var<storage, read> material_texture_generation_table: array<u32>;

// Material-VT production ABI. Bindings 5-8 deliberately extend the existing
// material group without moving the bindless arrays: the parent integration
// supplies these resources, including valid dummy resources when disabled.
@group(1) @binding(5) var<storage, read> material_vt_page_table: array<u32>;
@group(1) @binding(6) var<storage, read> material_vt_config: MaterialVirtualTextureConfig;
@group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@group(1) @binding(8) var material_vt_sampler: sampler;
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
    offline_normal_variance: f32,
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
    out.emissive = parameters.emission_color.xyz;
    out.metallic = parameters.base_metalness;
    out.perceptual_roughness = parameters.specular_roughness;
    out.occlusion = parameters.ambient_occlusion;
    out.normal_tangent = vec3<f32>(0.0, 0.0, 1.0);
    out.has_normal_map = false;
    out.offline_normal_variance = 0.0;
    out.clearcoat = parameters.coat_weight;
    out.clearcoat_roughness = parameters.coat_roughness;
    out.clearcoat_normal_tangent = vec3<f32>(0.0, 0.0, 1.0);
    out.has_clearcoat_normal = false;
    return out;
}

// A material row is valid when it carries a published generation. In the
// bindless specialization the index, sampler, and generation are additionally
// checked against the resources published for the current heap epoch before
// either array can be indexed.
fn material_texture_record_is_valid(record: PrismMaterialTexture) -> bool {
    if record.generation == 0u {
        return false;
    }
    @if(MATERIAL_BINDLESS) {
        // The generation table is allocated with the same device-clamped
        // capacity as both bindless arrays, so it is the portable runtime
        // bounds source; naga cannot take `arrayLength` of a binding array.
        if record.index >= arrayLength(&material_texture_generation_table)
            || record.sampler_index >= arrayLength(&material_texture_generation_table)
            || material_texture_generation_table[record.index] != record.generation
            || material_texture_generation_table[record.sampler_index] != record.generation {
            return false;
        }
    }
    return true;
}

fn material_default_texture_texel(semantic: u32) -> vec4<f32> {
    if semantic == SEMANTIC_NORMAL_VARIANCE {
        // Generation mismatch, absent resource and missing VT pages must never
        // turn the legacy flat-normal alpha=1 into a variance payload.
        return vec4<f32>(0.5, 0.5, 1.0, 0.0);
    }
    if semantic == SEMANTIC_NORMAL || semantic == SEMANTIC_CLEAR_COAT_NORMAL {
        return vec4<f32>(0.5, 0.5, 1.0, 1.0);
    }
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}

// Samples one bound texture record from the bindless heap at an explicit level
// of detail. The slot and sampler indices are the ones the CPU
// `BindlessTextureHeap` assigned; an unbound semantic points at a reserved
// default slot, never out of range.
//
// The resolve is a compute dispatch, so there are no `dpdx`/`dpdy` derivatives
// for the hardware to pick a mip with: the caller passes the level it derived
// analytically (`prism_texture_lod` below, §7.1). Callers that look up a
// *data* map rather than a filtered albedo pass `0.0` explicitly.
fn sample_texture_slot_direct(
    record: PrismMaterialTexture,
    uv: vec2<f32>,
    lod: f32,
) -> vec4<f32> {
    if !material_texture_record_is_valid(record) {
        return material_default_texture_texel(record.semantic);
    }
    @if(MATERIAL_BINDLESS) {
        return textureSampleLevel(
            material_texture_heap[record.index],
            material_sampler_heap[record.sampler_index],
            uv,
            lod,
        );
    }
    @if(!MATERIAL_BINDLESS) {
        return material_default_texture_texel(record.semantic);
    }
}

// Measured BRDF tables share the material heap, but the lobe module must also
// compile for the non-bindless pipeline. Keep that consumer's symbol stable
// while making the non-bindless variant a true identity path: it has no
// binding-array declaration and cannot touch an absent array.
fn material_texture_load(slot: u32, coord: vec2<i32>) -> vec4<f32> {
    @if(MATERIAL_BINDLESS) {
        return textureLoad(material_texture_heap[slot], coord, 0);
    }
    @if(!MATERIAL_BINDLESS) {
        return vec4<f32>(0.0);
    }
}

fn material_texture_dimensions(slot: u32) -> vec2<u32> {
    @if(MATERIAL_BINDLESS) {
        return textureDimensions(material_texture_heap[slot]);
    }
    @if(!MATERIAL_BINDLESS) {
        return vec2<u32>(1u, 1u);
    }
}

// Material-VT consumer used by the production resolve. Bindless pipelines use
// their direct image only for non-VT rows and VT misses. A non-bindless
// pipeline has no direct image/sampler bindings by contract, so its only
// authored-texture route is a valid, resident source-backed VT row; every
// non-VT, malformed, unavailable, or non-resident row keeps the semantic
// fallback below.
fn sample_texture_slot(
    record: PrismMaterialTexture,
    uv: vec2<f32>,
    lod: f32,
) -> vec4<f32> {
    if !material_texture_record_is_valid(record) {
        return material_default_texture_texel(record.semantic);
    }
    @if(!MATERIAL_BINDLESS) {
        if record.virtual_texture_id == 0xfffffffeu
            && material_vt_config.enabled != 0u
            && record.index < material_vt_config.atlas_layers * material_vt_config.tiles_per_row * material_vt_config.tiles_per_row
            && record.virtual_texture_page_size == material_vt_config.page_size {
            let source_uv = clamp(
                uv * vec2<f32>(
                    f32(record.virtual_texture_width) / f32(record.virtual_texture_page_size),
                    f32(record.virtual_texture_height) / f32(record.virtual_texture_page_size),
                ),
                vec2<f32>(0.0),
                vec2<f32>(1.0),
            );
            let resolution = material_vt_empty_resolution(true);
            var portable = resolution;
            portable.hit = true;
            portable.slot = record.index;
            portable.atlas_layer = record.index / max(material_vt_config.tiles_per_row * material_vt_config.tiles_per_row, 1u);
            portable.page_local_uv = source_uv;
            portable.atlas_uv = material_vt_atlas_uv(
                source_uv,
                record.index,
                material_vt_config.tile_extent_px,
                material_vt_config.tiles_per_row,
                material_vt_config.page_size,
                material_vt_config.gutter_px,
            );
            return material_vt_sample_resolved(material_vt_atlas, material_vt_sampler, portable);
        }
        return material_default_texture_texel(record.semantic);
    }
    let request = material_vt_page_request(
        record.virtual_texture_id,
        record.virtual_texture_width,
        record.virtual_texture_height,
        record.virtual_texture_page_size,
        uv,
        lod,
    );
    if !request.valid {
        @if(MATERIAL_BINDLESS) {
            return sample_texture_slot_direct(record, uv, lod);
        }
        @if(!MATERIAL_BINDLESS) {
            return material_default_texture_texel(record.semantic);
        }
    }
    // One page table/configuration describes one atlas geometry. A row with a
    // different page size cannot be resolved safely and remains direct-bound.
    if material_vt_config.enabled == 0u
        || record.virtual_texture_page_size != material_vt_config.page_size {
        @if(MATERIAL_BINDLESS) {
            return sample_texture_slot_direct(record, uv, lod);
        }
        @if(!MATERIAL_BINDLESS) {
            return material_default_texture_texel(record.semantic);
        }
    }

    let resolution = material_vt_resolve(
        material_vt_config,
        record.virtual_texture_id,
        0u,
        request.mip,
        request.page_x,
        request.page_y,
        request.local_uv,
    );
    if !resolution.vt_enabled || !resolution.hit {
        @if(MATERIAL_BINDLESS) {
            return sample_texture_slot_direct(record, uv, lod);
        }
        @if(!MATERIAL_BINDLESS) {
            return material_default_texture_texel(record.semantic);
        }
    }
    let atlas_dimensions = textureDimensions(material_vt_atlas);
    let atlas_layers = textureNumLayers(material_vt_atlas);
    let configured_extent = material_vt_config.tile_extent_px
        * material_vt_config.tiles_per_row;
    if atlas_dimensions.x == 0u
        || atlas_dimensions.y == 0u
        || atlas_dimensions.x != configured_extent
        || atlas_dimensions.y != configured_extent
        || atlas_layers < material_vt_config.atlas_layers
        || resolution.atlas_layer >= atlas_layers {
        @if(MATERIAL_BINDLESS) {
            return sample_texture_slot_direct(record, uv, lod);
        }
        @if(!MATERIAL_BINDLESS) {
            return material_default_texture_texel(record.semantic);
        }
    }
    return material_vt_sample_resolved(material_vt_atlas, material_vt_sampler, resolution);
}

// Screen-space triangles at or below this area (px²) are treated as degenerate
// and fall back to the base level. Mirrors `MIN_SCREEN_AREA` in
// `prism_render_shading::texture_lod`.
const PRISM_MIN_SCREEN_AREA: f32 = 1.0e-6;

// Inputs outside ±`PRISM_FINITE_RANGE` are treated as non-finite and keep the
// base level. Expressed as range comparisons rather than a `(x - x) == 0`
// finiteness idiom because Metal's fast-math folds the latter to a constant
// (measured: an infinite UV then sampled at an infinite LOD). Mirrors
// `FINITE_RANGE` in `prism_render_shading::texture_lod`.
const PRISM_FINITE_RANGE: f32 = 1.0e37;

fn prism_lod_in_range(value: f32) -> bool {
    return value > -PRISM_FINITE_RANGE && value < PRISM_FINITE_RANGE;
}

/// Per-pixel UV gradients `[dUV/dx, dUV/dy]` of the reconstructed triangle, in
/// UV units per pixel.
///
/// Barycentric coordinates are affine over a triangle, so their gradients are
/// constant across it and come from the inverse of the 2x2 screen-space edge
/// matrix; differentiating the UV interpolation then gives the exact per-pixel
/// texture footprint without any derivative hardware. A degenerate triangle
/// (|det| <= `PRISM_MIN_SCREEN_AREA`) yields zero gradients, which maps to the
/// base level — the resolve's behaviour before this existed.
///
/// Mirrors `prism_render_shading::texture_lod::uv_gradients` operation for
/// operation, so the CPU golden and the GPU twin stay within 1 ULP.
fn prism_uv_gradients(
    uv: array<vec2<f32>, 3>,
    screen: array<vec2<f32>, 3>,
) -> array<vec2<f32>, 2> {
    // Guard the inputs before any arithmetic can turn a malformed vertex into
    // an infinite level.
    for (var vertex = 0u; vertex < 3u; vertex = vertex + 1u) {
        if !prism_lod_in_range(uv[vertex].x)
            || !prism_lod_in_range(uv[vertex].y)
            || !prism_lod_in_range(screen[vertex].x)
            || !prism_lod_in_range(screen[vertex].y) {
            return array<vec2<f32>, 2>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
        }
    }
    let e1 = screen[1] - screen[0];
    let e2 = screen[2] - screen[0];
    let det = e1.x * e2.y - e1.y * e2.x;
    if !(abs(det) > PRISM_MIN_SCREEN_AREA) {
        return array<vec2<f32>, 2>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
    }
    let inv_det = 1.0 / det;
    let d_lambda_b_dx = e2.y * inv_det;
    let d_lambda_c_dx = -e1.y * inv_det;
    let d_lambda_b_dy = -e2.x * inv_det;
    let d_lambda_c_dy = e1.x * inv_det;
    let d_uv_b = uv[1] - uv[0];
    let d_uv_c = uv[2] - uv[0];
    return array<vec2<f32>, 2>(
        d_uv_b * d_lambda_b_dx + d_uv_c * d_lambda_c_dx,
        d_uv_b * d_lambda_b_dy + d_uv_c * d_lambda_c_dy,
    );
}

/// Isotropic mip level for a `dimensions`-texel map sampled with `gradients`.
///
/// The standard GPU rule — the longer texel-space gradient axis wins — so the
/// level tracks distance continuously instead of snapping. Magnified or exactly
/// 1:1 footprints stay on the base level. Mirrors
/// `prism_render_shading::texture_lod::texture_lod`.
fn prism_texture_lod(gradients: array<vec2<f32>, 2>, dimensions: vec2<f32>) -> f32 {
    if !prism_lod_in_range(dimensions.x)
        || !prism_lod_in_range(dimensions.y)
        || !prism_lod_in_range(gradients[0].x)
        || !prism_lod_in_range(gradients[0].y)
        || !prism_lod_in_range(gradients[1].x)
        || !prism_lod_in_range(gradients[1].y) {
        return 0.0;
    }
    let d_x = gradients[0] * dimensions;
    let d_y = gradients[1] * dimensions;
    let len_x = length(d_x);
    let len_y = length(d_y);
    let extent = max(len_x, len_y);
    // Magnified / exactly 1:1: the base level is already correct, and log2 of a
    // sub-unit extent would be negative.
    if !(extent > 1.0) {
        return 0.0;
    }
    return log2(extent);
}

// Folds a single sampled texel into `sampled` according to its semantic,
// returning the updated accumulator. Mirrors `fold_material_texel` arm-for-arm.
fn material_is_normal_semantic(semantic: u32) -> bool {
    return semantic == SEMANTIC_NORMAL || semantic == SEMANTIC_NORMAL_VARIANCE;
}

fn material_offline_normal_variance(semantic: u32, texel: vec4<f32>) -> f32 {
    if semantic == SEMANTIC_NORMAL_VARIANCE && texel.w >= 0.0 && texel.w < 1.0e37 {
        return texel.w;
    }
    return 0.0;
}

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
        case SEMANTIC_NORMAL, SEMANTIC_NORMAL_VARIANCE: {
            out.normal_tangent = decode_tangent_normal(rgb, normal_scale);
            out.has_normal_map = true;
            // Last normal record wins, including clearing an earlier payload
            // when an ordinary normal record replaces it.
            out.offline_normal_variance = material_offline_normal_variance(record.semantic, texel);
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
