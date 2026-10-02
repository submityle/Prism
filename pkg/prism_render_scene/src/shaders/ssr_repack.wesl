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
import prism_render_scene::shaders::material_unpack::{
    prism_unpack_surface, prism_packed_word_count, prism_neutral_surface
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
    GpuSceneTransform,
};
import prism_render_scene::shaders::scene_transform::scene_apply_current_transform;
import prism_render_scene::shaders::tangent::{
    TangentBasis,
    apply_tangent_space_normal,
};
import prism_render_scene::shaders::material_sample::{
    SampledMaterial,
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

// Tangent-space normal-map semantic (`decode_tangent_normal` in
// `material_sample.wesl` unpacks the RGB into the tangent-space normal, sets
// `has_normal_map`). Folded alongside the roughness so the trace marches
// against the same normal-mapped surface the resolve shades, not the raw
// geometric normal.
const SEMANTIC_NORMAL: u32 = 3u;

// --- Geometric specular anti-aliasing (Tokuyoshi-Kaplanyan 2019) ---
//
// Mirrors the CPU golden
// `prism_render_shading::gi::specular_aa::normal_variance::`
// `geometric_specular_aa_roughness` bit-for-bit. A curved or bumpy surface packs
// many micro-orientations into one shaded pixel, so a tight GGX lobe flickers
// between frames under motion. We estimate the *screen-space normal variance*
// from the geometric view-normal's finite differences and widen the lobe just
// enough to cover it (additive-variance rule with the Tokuyoshi kappa cap).
//
// The derivatives are taken on the *geometric* `view_normal_in` only, so this
// captures curvature/silhouette aliasing. High-frequency normal-*map* aliasing
// lives in the Toksvig domain and is deliberately NOT folded here; do not
// over-claim it covers texture-detail aliasing.
const SPEC_AA_MIN_ALPHA: f32 = 1.0e-3;
const SPEC_AA_SIGMA2: f32 = 0.15915494;
const SPEC_AA_KAPPA_MAX: f32 = 0.18;

// Perceptual roughness -> GGX alpha = clamp(r,0,1)^2, floored at MIN_ALPHA,
// matching `roughness_to_alpha` in `gi::spec_gi::ggx_lobe`.
fn spec_aa_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SPEC_AA_MIN_ALPHA);
}

// WGSL has no isFinite; `x - x == 0.0` is true only for finite x (NaN and inf
// both yield NaN here), matching the golden's `Vec3::is_finite` sanitisation.
fn spec_aa_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

// Squared length of a normal-derivative vector; any non-finite component makes
// the whole vector contribute zero variance (golden `safe_sq_len`).
fn spec_aa_safe_sq_len(v: vec3<f32>) -> f32 {
    let finite = spec_aa_is_finite(v.x) && spec_aa_is_finite(v.y) && spec_aa_is_finite(v.z);
    if (!finite) {
        return 0.0;
    }
    return max(dot(v, v), 0.0);
}

// Central differences of the geometric view-normal, coords clamped into the
// framebuffer so edge texels reuse the border sample.
fn spec_aa_normal_derivatives(coord: vec2<i32>) -> array<vec3<f32>, 2> {
    let max_x = i32(params.width) - 1;
    let max_y = i32(params.height) - 1;
    let xl = clamp(coord.x - 1, 0, max_x);
    let xr = clamp(coord.x + 1, 0, max_x);
    let yu = clamp(coord.y - 1, 0, max_y);
    let yd = clamp(coord.y + 1, 0, max_y);
    let nxl = textureLoad(view_normal_in, vec2<i32>(xl, coord.y), 0).xyz;
    let nxr = textureLoad(view_normal_in, vec2<i32>(xr, coord.y), 0).xyz;
    let nyu = textureLoad(view_normal_in, vec2<i32>(coord.x, yu), 0).xyz;
    let nyd = textureLoad(view_normal_in, vec2<i32>(coord.x, yd), 0).xyz;
    return array<vec3<f32>, 2>(nxr - nxl, nyd - nyu);
}

// Full filter mirroring `geometric_specular_aa_roughness`:
//   alpha        = roughness_to_alpha(r)            (= r^2, floored MIN_ALPHA)
//   variance     = SIGMA2 * (|ddx|^2 + |ddy|^2)
//   kernel       = min(2 * variance, KAPPA_MAX)
//   eff_alpha_sq = clamp(alpha^2 + kernel, MIN_ALPHA^2, 1)
//   roughness'   = (eff_alpha_sq)^(1/4)             (alpha = r^2, so ^(1/4))
// The injected variance is non-negative, so the output never sharpens the lobe.
fn spec_aa_filter_roughness(base_roughness: f32, ddx: vec3<f32>, ddy: vec3<f32>) -> f32 {
    let alpha = spec_aa_roughness_to_alpha(base_roughness);
    let base_alpha_sq = alpha * alpha;
    let variance = SPEC_AA_SIGMA2 * (spec_aa_safe_sq_len(ddx) + spec_aa_safe_sq_len(ddy));
    let kernel = min(2.0 * variance, SPEC_AA_KAPPA_MAX);
    let floor = SPEC_AA_MIN_ALPHA * SPEC_AA_MIN_ALPHA;
    let eff = clamp(base_alpha_sq + kernel, floor, 1.0);
    return clamp(sqrt(sqrt(eff)), 0.0, 1.0);
}

// Immediate (push-constant) block. Mirrors `GpuSsrRepackParams` in
// `shading/ssr/abi.rs` (80 bytes: the `view_from_world` matrix, the framebuffer
// extent, plus two padding scalars to satisfy the 16-byte immediate alignment).
struct RepackParams {
    // World -> view (rigid, camera-at-origin, looking down `-Z`). A normal-mapped
    // world-space normal is rotated into the view frame with this before it is
    // encoded, matching the frame `ssr_prepass.wesl` writes for the geometric
    // normal.
    view_from_world: mat4x4<f32>,
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
@group(1) @binding(1) var<storage, read> material_parameters: array<u32>;

// See opaque.wesl: gather a material's packed block out of the shared u32 heap
// and expand it, returning the neutral surface if the declared range overruns.
fn load_surface_parameters(word_offset: u32, lobe_mask: u32) -> PrismSurfaceParameters {
    let n = min(prism_packed_word_count(lobe_mask), 40u);
    if word_offset + n > arrayLength(&material_parameters) {
        return prism_neutral_surface();
    }
    var words: array<u32, 40>;
    for (var i = 0u; i < n; i = i + 1u) {
        words[i] = material_parameters[word_offset + i];
    }
    return prism_unpack_surface(words, lobe_mask);
}
@group(1) @binding(2) var<storage, read> material_textures: array<PrismMaterialTexture>;

// --- group 2: scene-instance + shading-geometry tables (owned by this pass) ---
// The four read-only tables the surface reconstruction walks to recover the
// interpolated UV: the per-instance scene records, then the geometry headers,
// vertices and primitives. A four-buffer subset of the resolve's group 2.
@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> geometry_headers: array<RenderShadingGeometryHeader>;
@group(2) @binding(2) var<storage, read> geometry_vertices: array<GpuShadingVertex>;
@group(2) @binding(3) var<storage, read> geometry_primitives: array<GpuShadingPrimitive>;
// Per-instance affine `world_from_local` (indexed by `scene_index`), matching
// `GpuSceneTransform` in `gpu_scene.wesl` and the buffer the prepass/resolve
// bind. The reconstructed vertices are *local* space, so a normal-mapped normal
// must be built against the world-space interpolated basis; the identity path
// leaves an untransformed instance exactly as before.
@group(2) @binding(4) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;

var<immediate> params: RepackParams;

// World-space surface reconstruction for a covered pixel plus a validity flag
// (WESL has no `Option`). `valid == false` leaves `geometry` untouched.
struct ReconstructedSurface {
    geometry: SurfaceSampleGeometry,
    valid: bool,
}

// Reconstructs the interpolated *world-space* surface for a covered pixel from
// the visibility ids/metadata, walking the same scene -> geometry -> primitive
// -> vertex chain as `shading_resolve.wesl` and guarding every table read twice
// (against the owning header's logical count and the physical `arrayLength`).
// The interpolated vertices are local space, so the owning instance's
// `world_from_local` is applied (mirroring the prepass/resolve) before the basis
// is used for normal-map perturbation. `valid == false` whenever any guard
// rejects the sample, so the caller falls back to the authored roughness and
// the prepass geometric normal.
fn reconstruct_geometry(ids: vec4<u32>, metadata: vec4<u32>) -> ReconstructedSurface {
    var out: ReconstructedSurface;
    out.valid = false;

    // ids = [scene_index, scene_generation, primitive_index, geometry_lod].
    let scene_index = ids.x;
    if scene_index >= arrayLength(&scene_instances) {
        return out;
    }
    let instance = scene_instances[scene_index];
    if instance.is_active == 0u || instance.generation != ids.y {
        return out;
    }
    let geometry_index = instance.geometry_index;
    if geometry_index >= arrayLength(&geometry_headers) {
        return out;
    }
    let header = geometry_headers[geometry_index];
    if (header.flags & SHADING_GEOMETRY_FLAG_ACTIVE) == 0u {
        return out;
    }
    if header.generation != instance.geometry_generation {
        return out;
    }
    let primitive_index = ids.z;
    if primitive_index >= header.primitive_count {
        return out;
    }
    let primitive_row = header.primitive_offset + primitive_index;
    if primitive_row >= arrayLength(&geometry_primitives) {
        return out;
    }
    let primitive = geometry_primitives[primitive_row];
    let index_a = primitive.indices[0];
    let index_b = primitive.indices[1];
    let index_c = primitive.indices[2];
    if index_a >= header.vertex_count
        || index_b >= header.vertex_count
        || index_c >= header.vertex_count {
        return out;
    }
    let row_a = header.vertex_offset + index_a;
    let row_b = header.vertex_offset + index_b;
    let row_c = header.vertex_offset + index_c;
    let vertex_len = arrayLength(&geometry_vertices);
    if row_a >= vertex_len || row_b >= vertex_len || row_c >= vertex_len {
        return out;
    }
    let vertex_a = geometry_vertices[row_a];
    let vertex_b = geometry_vertices[row_b];
    let vertex_c = geometry_vertices[row_c];
    let weights = surface_decode_barycentrics(metadata.w);
    let local_geometry =
        interpolate_surface(vertex_a, vertex_b, vertex_c, primitive.flags, weights);

    // Lift the local-space reconstruction into world space with the instance
    // transform (falling back to the untransformed sample for a short/stale
    // buffer), so the tangent basis matches the world-space normal the resolve
    // perturbs.
    if scene_index < arrayLength(&scene_current_transforms) {
        out.geometry = scene_apply_current_transform(
            scene_current_transforms[scene_index], local_geometry);
    } else {
        out.geometry = local_geometry;
    }
    out.valid = true;
    return out;
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
// The trace's `normal_roughness` inputs for one covered pixel: the
// texture-modulated perceptual roughness plus (when a normal map is bound) the
// world-space normal-mapped normal, mirroring `shading_resolve.wesl`.
struct RepackSample {
    // Perceptual roughness to pack, already clamped. `1.0` (fully rough, which
    // drives the trace's `roughness_fade` and thus SSR confidence to zero) is
    // the fallback whenever a material generation/bounds guard rejects the
    // sample, so a stale or absent material reflects nothing.
    roughness: f32,
    // World-space shading normal after normal-map perturbation. Only meaningful
    // when `normal_overridden`.
    world_normal: vec3<f32>,
    // `true` when a normal map was folded against a valid reconstructed basis,
    // so the caller replaces the prepass geometric normal with `world_normal`
    // (rotated into view space). `false` keeps the prepass view normal.
    normal_overridden: bool,
}

// Folds the metallic-roughness *and* normal textures for one covered pixel over
// the authored parameters, exactly like the resolve's texture fold, then (if a
// normal map is bound) rotates its tangent-space sample into world space against
// the reconstructed interpolated basis. The other semantics the resolve folds
// do not affect the trace's `normal_roughness` inputs and are skipped so the
// repack stays a lightweight two-fetch pass. Folding is skipped when the UV
// could not be reconstructed, so a degenerate primitive falls back to the
// authored roughness (and the prepass normal) rather than sampling at a garbage
// UV. `metadata.x` is the material index, `metadata.y` the expected generation.
fn repack_sample(ids: vec4<u32>, metadata: vec4<u32>) -> RepackSample {
    var out: RepackSample;
    out.roughness = 1.0;
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    out.normal_overridden = false;

    let material_index = metadata.x;
    if material_index >= arrayLength(&material_headers) {
        return out;
    }
    let material_header = material_headers[material_index];
    if material_header.is_active == 0u || material_header.generation != metadata.y {
        return out;
    }
    let parameter_words = min(prism_packed_word_count(material_header.lobe_mask), 40u);
    if material_header.parameter_offset + parameter_words > arrayLength(&material_parameters) {
        return out;
    }
    let parameters = load_surface_parameters(material_header.parameter_offset, material_header.lobe_mask);

    var sampled = sampled_material_defaults(parameters);
    let recon = reconstruct_geometry(ids, metadata);
    if recon.valid {
        let uv = recon.geometry.uv;
        let tex_end = material_header.texture_offset + material_header.texture_count;
        let tex_physical = arrayLength(&material_textures);
        for (var i = material_header.texture_offset; i < tex_end && i < tex_physical; i = i + 1u) {
            let record = material_textures[i];
            if record.semantic == SEMANTIC_METALLIC_ROUGHNESS
                || record.semantic == SEMANTIC_NORMAL {
                let texel = sample_texture_slot(record, uv);
                sampled = fold_material_texel(record, texel, parameters.normal_scale, sampled);
            }
        }

        // A bound normal map rotates its tangent-space sample into world space
        // against the interpolated basis, exactly as `shading_resolve.wesl`
        // does; the geometry (and thus the basis) is already world space from
        // `scene_apply_current_transform`.
        if sampled.has_normal_map {
            var basis: TangentBasis;
            basis.tangent = recon.geometry.tangent;
            basis.bitangent = recon.geometry.bitangent;
            basis.degenerate = false;
            out.world_normal = apply_tangent_space_normal(
                basis, recon.geometry.normal, sampled.normal_tangent);
            out.normal_overridden = true;
        }
    }
    out.roughness = clamp(sampled.perceptual_roughness, 0.0, 1.0);
    return out;
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

    let sample = repack_sample(ids, metadata);

    // Default to the prepass view-space geometric normal; when a normal map was
    // folded, rotate the world-space normal-mapped normal into the view frame
    // instead so the trace marches against the same surface the resolve shades.
    // `view_from_world` is rigid, so the rotated normal stays unit.
    var view_normal = textureLoad(view_normal_in, coord, 0).xyz;
    if sample.normal_overridden {
        let view_dir = (params.view_from_world * vec4<f32>(sample.world_normal, 0.0)).xyz;
        view_normal = normalize(view_dir);
    }

    // Geometric specular AA: widen the packed roughness by the screen-space
    // variance of the geometric view-normal so curvature/silhouette aliasing
    // stops the SSR specular lobe from flickering. Uses `view_normal_in`
    // (geometric) derivatives even when a normal map overrode `view_normal`.
    let dn = spec_aa_normal_derivatives(coord);
    let aa_roughness = spec_aa_filter_roughness(sample.roughness, dn[0], dn[1]);

    let encoded_normal = view_normal * 0.5 + vec3<f32>(0.5);
    textureStore(normal_roughness_out, coord, vec4<f32>(encoded_normal, aa_roughness));
}
