//! SSR material-roughness repack compute entry point.
//!
//! The screen-space trace (`ssr.wesl`) samples a single `normal_roughness`
//! texture: `rgb = view_normal * 0.5 + 0.5` (decoded as `nr.xyz * 2 - 1`) and
//! `a = perceptual roughness`. The geometry prepass (`ssr_prepass.wesl`) already
//! rebuilt the *signed* view-space normal, but it has no material bind group and
//! therefore cannot fetch roughness. This stage bridges the gap: it re-reads the
//! prepass normal, reconstructs the covered pixel's interpolated UV from the
//! visibility buffer (walking the identical scene -> geometry -> primitive ->
//! vertex tables `shading_resolve.wesl` uses), samples the material's
//! metallic-roughness texture through the shared bindless heap, and packs the
//! *texture-modulated* roughness into the trace's `normal_roughness` input, one
//! texel per framebuffer pixel.
//!
//! This is the full `material_sample` + interpolated-UV path: roughness now
//! observes the authored roughness texture exactly as the shading resolve does,
//! so a rough-painted region of an otherwise smooth material widens (and fades)
//! its reflection correctly. The reconstruction stays on the *current* frame in
//! this single pass, so it never lags the trace. Only the metallic-roughness
//! semantic is sampled (roughness lives in the texel's G channel); the other
//! bound textures the resolve folds do not affect roughness and are skipped so
//! the repack stays a lightweight one-fetch pass.
//!
//! It reads three bind groups:
//!
//! * **group 0** - the two visibility textures (ids/metadata) read, the prepass
//!   view-normal read, and the packed `normal_roughness` written. Owned by this
//!   pass.
//! * **group 1** - the shared material tables, reusing [`MaterialBindGroup`]'s
//!   layout descriptor so the header/parameter/texture buffers and the bindless
//!   texture + sampler heaps bind byte-for-byte with the resolve.
//! * **group 2** - the scene-instance and shading-geometry tables the surface
//!   reconstruction walks to recover the interpolated UV. Owned by this pass
//!   (a four-buffer subset of the resolve's group 2, which also carries the
//!   worklist the repack does not need).

import prism_render_scene::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};
import prism_render_scene::shaders::surface::{
    RenderShadingGeometryHeader,
    GpuShadingVertex,
    GpuShadingPrimitive,
    SurfaceSampleGeometry,
    SHADING_GEOMETRY_FLAG_ACTIVE,
    surface_decode_barycentrics,
    interpolate_surface,
};
import prism_render_scene::shaders::gpu_scene::{
    GpuSceneInstance,
};
import prism_render_scene::shaders::material_sample::{
    sampled_material_defaults,
    sample_texture_slot,
    fold_material_texel,
};

// `material_sample` folds bindless textures through a `binding_array`, so this
// merged module must enable the extension. WESL requires every `enable` to
// follow all `import`s.
enable wgpu_binding_array;

// Sentinel packed into the visibility targets for uncovered pixels. Matches
// `VisibilityPixel::is_valid` on the CPU side, `shading_resolve.wesl` and
// `ssr_prepass.wesl`.
const VISIBILITY_BACKGROUND: u32 = 0xffffffffu;

// Metallic-roughness texture semantic (roughness in G, metallic in B), mirroring
// `SEMANTIC_METALLIC_ROUGHNESS` in `material_sample.wesl` / `TextureSemantic` on
// the CPU golden. Only this semantic modulates roughness, so the repack samples
// it alone.
const SEMANTIC_METALLIC_ROUGHNESS: u32 = 2u;

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

// --- group 1: shared material tables + bindless heap (reused MaterialBindGroup) ---
// The header / parameter / texture-record tables plus the bindless texture and
// sampler heaps, all part of the shared `MaterialBindGroup` layout so they bind
// byte-for-byte with the resolve. Bindings 3/4 are declared by
// `material_sample.wesl` (imported above).
@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<PrismSurfaceParameters>;
@group(1) @binding(2) var<storage, read> material_textures: array<PrismMaterialTexture>;

// --- group 2: scene-instance + shading-geometry tables (owned by this pass) ---
// The four read-only tables the surface reconstruction walks to recover the
// interpolated UV: the per-instance scene records, then the geometry headers,
// vertices and primitives. A four-buffer subset of the resolve's group 2.
@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> geometry_headers: array<RenderShadingGeometryHeader>;
@group(2) @binding(2) var<storage, read> geometry_vertices: array<GpuShadingVertex>;
@group(2) @binding(3) var<storage, read> geometry_primitives: array<GpuShadingPrimitive>;

var<immediate> params: RepackParams;

// Reconstructs the interpolated surface UV for a covered pixel from the
// visibility ids/metadata, walking the same scene -> geometry -> primitive ->
// vertex chain as `shading_resolve.wesl` and guarding every table read twice
// (against the owning header's logical count and the physical `arrayLength`).
// Returns `vec3(uv, 1.0)` on success or `vec3(0)` (`.z == 0`) whenever any guard
// rejects the sample, so the caller can fall back to the authored roughness.
fn reconstruct_uv(ids: vec4<u32>, metadata: vec4<u32>) -> vec3<f32> {
    // ids = [scene_index, scene_generation, primitive_index, geometry_lod].
    let scene_index = ids.x;
    if scene_index >= arrayLength(&scene_instances) {
        return vec3<f32>(0.0);
    }
    let instance = scene_instances[scene_index];
    if instance.is_active == 0u || instance.generation != ids.y {
        return vec3<f32>(0.0);
    }
    let geometry_index = instance.geometry_index;
    if geometry_index >= arrayLength(&geometry_headers) {
        return vec3<f32>(0.0);
    }
    let header = geometry_headers[geometry_index];
    if (header.flags & SHADING_GEOMETRY_FLAG_ACTIVE) == 0u {
        return vec3<f32>(0.0);
    }
    if header.generation != instance.geometry_generation {
        return vec3<f32>(0.0);
    }
    let primitive_index = ids.z;
    if primitive_index >= header.primitive_count {
        return vec3<f32>(0.0);
    }
    let primitive_row = header.primitive_offset + primitive_index;
    if primitive_row >= arrayLength(&geometry_primitives) {
        return vec3<f32>(0.0);
    }
    let primitive = geometry_primitives[primitive_row];
    let index_a = primitive.indices[0];
    let index_b = primitive.indices[1];
    let index_c = primitive.indices[2];
    if index_a >= header.vertex_count
        || index_b >= header.vertex_count
        || index_c >= header.vertex_count {
        return vec3<f32>(0.0);
    }
    let row_a = header.vertex_offset + index_a;
    let row_b = header.vertex_offset + index_b;
    let row_c = header.vertex_offset + index_c;
    let vertex_len = arrayLength(&geometry_vertices);
    if row_a >= vertex_len || row_b >= vertex_len || row_c >= vertex_len {
        return vec3<f32>(0.0);
    }
    let vertex_a = geometry_vertices[row_a];
    let vertex_b = geometry_vertices[row_b];
    let vertex_c = geometry_vertices[row_c];
    let weights = surface_decode_barycentrics(metadata.w);
    let geometry: SurfaceSampleGeometry =
        interpolate_surface(vertex_a, vertex_b, vertex_c, primitive.flags, weights);
    return vec3<f32>(geometry.uv, 1.0);
}

// Perceptual roughness for one covered pixel: the authored parameter modulated
// by the material's metallic-roughness texture sampled at the reconstructed UV,
// or a fully-rough fallback (`1.0`) whenever any material generation/bounds guard
// rejects the sample. A fully rough surface drives the trace's `roughness_fade`
// - and thus SSR confidence - to zero, so a stale or absent material reflects
// nothing rather than smearing an incorrect mirror sample. If the material is
// valid but the UV cannot be reconstructed, the authored roughness stands (no
// texture modulation), matching the pre-upgrade base-parameter behaviour.
// `metadata.x` is the material index, `metadata.y` the expected generation.
fn resolve_roughness(ids: vec4<u32>, metadata: vec4<u32>) -> f32 {
    let material_index = metadata.x;
    if material_index >= arrayLength(&material_headers) {
        return 1.0;
    }
    let material_header = material_headers[material_index];
    if material_header.is_active == 0u || material_header.generation != metadata.y {
        return 1.0;
    }
    let parameter_row = material_header.parameter_offset;
    if parameter_row >= arrayLength(&material_parameters) {
        return 1.0;
    }
    let parameters = material_parameters[parameter_row];

    // Seed from the authored factors, then fold the metallic-roughness texture
    // (if any) over the authored roughness exactly like the resolve's texture
    // fold. Skips folding when the UV could not be reconstructed so a degenerate
    // primitive falls back to the authored roughness rather than sampling at a
    // garbage UV.
    var sampled = sampled_material_defaults(parameters);
    let uv_sample = reconstruct_uv(ids, metadata);
    if uv_sample.z > 0.5 {
        let uv = uv_sample.xy;
        let tex_end = material_header.texture_offset + material_header.texture_count;
        let tex_physical = arrayLength(&material_textures);
        for (var i = material_header.texture_offset; i < tex_end && i < tex_physical; i = i + 1u) {
            let record = material_textures[i];
            if record.semantic == SEMANTIC_METALLIC_ROUGHNESS {
                let texel = sample_texture_slot(record, uv);
                sampled = fold_material_texel(record, texel, parameters.normal_scale, sampled);
            }
        }
    }
    return clamp(sampled.perceptual_roughness, 0.0, 1.0);
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
    // normal; re-encode it into the trace's [0, 1] convention. Roughness comes
    // from the material's metallic-roughness texture at the reconstructed UV.
    let view_normal = textureLoad(view_normal_in, coord, 0).xyz;
    let roughness = resolve_roughness(ids, metadata);

    let encoded_normal = view_normal * 0.5 + vec3<f32>(0.5);
    textureStore(normal_roughness_out, coord, vec4<f32>(encoded_normal, roughness));
}
