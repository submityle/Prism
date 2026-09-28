//! GPU shading *resolve* compute entry point.
//!
//! This is the terminal stage of the visibility -> classify -> shade chain.
//! `material_classification.wesl` has already bucketed every covered pixel into
//! a per-`MaterialShadingClass` worklist (`work[class_offsets[class] + local]`)
//! and written the per-class indirect dispatch arguments.  This shader is
//! dispatched once per class with those indirect args; each invocation drains
//! one work item, reconstructs the surface, evaluates the lighting for its
//! class and stores linear HDR radiance into the `scene_color` storage texture.
//!
//! The heavy lifting lives in cohesive imported libraries so this entry point
//! stays a thin orchestrator that mirrors the CPU golden reference
//! `prism_render_shading::resolve::resolve_pixel` branch-for-branch:
//!
//! * `surface.wesl`  - vertex fetch interpolation + barycentric decode.
//! * `brdf.wesl`     - GGX principled / clearcoat / toon / image-based lobes.
//! * `lighting.wesl` - analytic light ABI + sampling + SH probe helpers.
//! * `material.wesl` - material header + surface-parameter ABI.
//! * `gpu_scene.wesl`- stable per-instance scene record.
//!
//! Every table read is guarded twice: once against the logical count carried by
//! the owning header, and once against the physical `arrayLength` of the bound
//! storage buffer, so a stale or malformed id can never sample a neighbour's
//! rows.  Generations are validated end-to-end (scene -> geometry -> material)
//! exactly like the CPU reference before any shading math runs.

import prism_render_scene::shaders::surface::{
    RenderShadingGeometryHeader,
    GpuShadingVertex,
    GpuShadingPrimitive,
    SurfaceSampleGeometry,
    SHADING_GEOMETRY_FLAG_ACTIVE,
    surface_decode_barycentrics,
    interpolate_surface,
};
import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
    principled_direct,
    stylized_direct,
};
import prism_render_scene::shaders::cloth::{
    cloth_direct,
};
import prism_render_scene::shaders::subsurface::{
    subsurface_direct,
};
import prism_render_scene::shaders::hair::{
    hair_direct,
};
import prism_render_scene::shaders::water::{
    water_direct,
};
import prism_render_scene::shaders::clearcoat::{
    clearcoat_direct,
};
import prism_render_scene::shaders::lighting::{
    DirectionalLight,
    PunctualLight,
    LightEnvironment,
    LightSample,
    INV_PI,
    environment_has_image_based,
    sample_directional_light,
    sample_punctual_light,
    sh_irradiance,
    sh_radiance,
    env_brdf_approx,
};
import prism_render_scene::shaders::shadow::{
    directional_shadow_visibility,
    point_shadow_visibility,
    spot_shadow_visibility,
};
import prism_render_scene::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};
import prism_render_scene::shaders::gpu_scene::{
    GpuSceneInstance,
    GpuSceneTransform,
};
import prism_render_scene::shaders::material_sample::{
    sampled_material_defaults,
    sample_texture_slot,
    fold_material_texel,
};
import prism_render_scene::shaders::tangent::{
    TangentBasis,
    apply_tangent_space_normal,
};
import bevy_render::maths::affine3_to_square;

// `material_sample` folds bindless textures through a `binding_array`, so this
// merged module must enable the extension. WESL requires every `enable` to
// follow all `import`s.
enable wgpu_binding_array;

// Sentinel packed into the visibility targets for uncovered pixels. Matches
// `VisibilityPixel::is_valid` on the CPU side.
const VISIBILITY_BACKGROUND: u32 = 0xffffffffu;

// `MaterialShadingClass` discriminants, mirroring the Rust enum in
// `prism_render_shading`. Each is handled by an explicit `switch` arm below.
const SHADING_CLASS_PRINCIPLED: u32 = 0u;
const SHADING_CLASS_UNLIT: u32 = 1u;
const SHADING_CLASS_SUBSURFACE: u32 = 2u;
const SHADING_CLASS_CLEARCOAT: u32 = 3u;
const SHADING_CLASS_CLOTH: u32 = 4u;
const SHADING_CLASS_HAIR: u32 = 5u;
const SHADING_CLASS_WATER: u32 = 6u;
const SHADING_CLASS_NPR: u32 = 7u;
const SHADING_CLASS_CUSTOM: u32 = 8u;

// Immediate (push-constant) block. Mirrors `GpuShadingResolveParams` in
// `shading/resolve/abi.rs` (32 bytes; the trailing vec4 is 16-byte aligned).
struct ResolveParams {
    shading_class: u32,
    width: u32,
    height: u32,
    // Packed feature bits; see the `RESOLVE_FLAG_*` masks. Bit 0 folds the
    // screen-space GTAO multiply below; bit 1 selects the prefiltered
    // image-based specular path over the low-frequency SH-radiance fallback.
    flags: u32,
    view_position: vec4<f32>,
}

// `ResolveParams.flags` masks, mirroring `RESOLVE_FLAG_*` in `resolve/abi.rs`.
// Bit 0 gates the screen-space GTAO fold; bit 1 signals that the prefiltered
// environment cube plus DFG table are resident so the resolve can sample the
// real specular reflection instead of the SH-radiance approximation.
const RESOLVE_FLAG_GTAO: u32 = 1u;
const RESOLVE_FLAG_IBL_SPECULAR: u32 = 2u;

// One compacted work item. Mirrors `ShadingWorkItem` (16 bytes) and the
// `WorkItem` produced by `material_classification.wesl`.
struct WorkItem {
    pixel_index: u32,
    material_index: u32,
    shading_class: u32,
    flags: u32,
}

// Current + previous frame view-projection matrices for motion-vector output.
// Mirrors `MotionMatrices` in `resolve/motion.rs` (128 bytes: two column-major
// mat4x4<f32>).
struct MotionMatrices {
    clip_from_world: mat4x4<f32>,
    previous_clip_from_world: mat4x4<f32>,
}

// --- group 0: visibility inputs + HDR output ---
@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;
// Screen-space GTAO visibility (1 = unoccluded) in `.x`; a 1x1 white fallback
// is bound when GTAO is disabled so this multiply is a no-op.
@group(0) @binding(3) var gtao_ao: texture_2d<f32>;
// GGX-prefiltered environment radiance cube (one mip per roughness level) plus
// its trilinear clamp sampler. Sampled along the reflection vector at a
// roughness-selected LOD when `RESOLVE_FLAG_IBL_SPECULAR` is set. When no probe
// is resident the global fallback cube is bound and the flag stays clear, so
// these are never read on that path.
@group(0) @binding(4) var prefiltered_env: texture_cube<f32>;
@group(0) @binding(5) var prefiltered_sampler: sampler;
// Split-sum environment BRDF ("DFG") table indexed by `(n_dot_v, roughness)`
// in `.xy` = `(scale, bias)`, plus its linear clamp sampler.
@group(0) @binding(6) var dfg_lut: texture_2d<f32>;
@group(0) @binding(7) var dfg_sampler: sampler;
// SSR energy-conservation exports (always bound; written every covered pixel).
// ssr_env_specular = the IBL specular radiance this resolve folded into
// scene_color at this pixel, so the SSR composite can subtract it before
// substituting the screen-space reflection. ssr_spec_weight = the split-sum
// environment-BRDF weight (f0 * dfg.x + dfg.y) * occlusion used for that
// specular, so the composite weights the SSR reflection by the identical
// Fresnel/DFG factor. Both are full-resolution rgba16float; .a is unused.
@group(0) @binding(8) var ssr_env_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var ssr_spec_weight: texture_storage_2d<rgba16float, write>;
// Per-pixel motion-vector G-buffer (rg16float): this frame's screen-space
// displacement `cur_uv - prev_uv` in [0,1] UV, capturing *both* camera and
// per-object motion (object motion enters through `scene_previous_transforms`).
// Consumed by `ssr_temporal.wesl` (and later TAA) to reproject history along
// true surface motion instead of camera-only reprojection, which kills the
// ghosting dynamic geometry otherwise leaves in the accumulated reflection.
@group(0) @binding(10) var motion_vectors: texture_storage_2d<rg16float, write>;
// Current + previous view-projection used to project `world_cur`/`world_prev`
// into clip space for the motion vector. A 128-byte uniform block (two mat4x4)
// rather than the 32-byte `var<immediate>` push-constant, which has no room.
@group(0) @binding(11) var<uniform> motion_matrices: MotionMatrices;
// Screen-space GI exports written every covered pixel, consumed by the SSGI
// trace + composite. ssgi_ambient = the pre-albedo diffuse *irradiance* the
// hemisphere gather blends misses toward (so SSGI never darkens the ambient).
// ssgi_albedo = the Lambertian albedo (`base_color * (1 - metallic)`) the
// composite multiplies the pre-albedo SSGI radiance by before blending it over
// the IBL diffuse under the pass's confidence.
@group(0) @binding(12) var ssgi_ambient: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var ssgi_albedo: texture_storage_2d<rgba16float, write>;

// --- group 1: material tables (shared MaterialBindGroup layout) ---
@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<PrismSurfaceParameters>;
@group(1) @binding(2) var<storage, read> material_textures: array<PrismMaterialTexture>;

// --- group 2: compacted worklist + scene/geometry tables ---
@group(2) @binding(0) var<storage, read> work: array<WorkItem>;
@group(2) @binding(1) var<storage, read> class_offsets: array<u32>;
@group(2) @binding(2) var<storage, read> class_counts: array<u32>;
@group(2) @binding(3) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(4) var<storage, read> geometry_headers: array<RenderShadingGeometryHeader>;
@group(2) @binding(5) var<storage, read> geometry_vertices: array<GpuShadingVertex>;
@group(2) @binding(6) var<storage, read> geometry_primitives: array<GpuShadingPrimitive>;
// Per-instance affine `world_from_local` (indexed by `scene_index`), matching
// `GpuSceneTransform` in `gpu_scene.wesl` and the transform buffer bound by the
// visibility/opaque passes. The reconstructed local-space surface is lifted
// into world space with it before any lighting, mirroring the CPU golden
// `resolve_pixel`; guarded against a short/stale buffer below.
@group(2) @binding(7) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
// Per-instance affine `world_from_local` from the *previous* frame (same layout
// and `scene_index` indexing as `scene_current_transforms`). Used only to place
// the surface's local position in last frame's world space for the motion
// vector; never touches shading. Guarded against a short/stale buffer below.
@group(2) @binding(8) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;

// --- group 3: analytic lights (shared LightBindGroup layout) ---
@group(3) @binding(0) var<storage, read> directional_lights: array<DirectionalLight>;
@group(3) @binding(1) var<storage, read> punctual_lights: array<PunctualLight>;
@group(3) @binding(2) var<storage, read> light_env: array<LightEnvironment>;

// --- group 5: clustered-light tables (shared ClusterBindGroup layout) ---
//
// Mirrors `GpuClusterGrid` (`lighting/cluster/abi.rs`, 128 bytes) with scalar
// fields so the std430 layout matches the `#[repr(C)]` record byte-for-byte.
// `cluster_offsets[c]` is the `[offset, count]` slice into `cluster_indices`
// for cluster `c`; each index addresses `punctual_lights` in extraction order.
struct ClusterGrid {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    cluster_count: u32,
    tile_x: u32,
    tile_y: u32,
    screen_x: u32,
    screen_y: u32,
    near: f32,
    far: f32,
    depth_scale: f32,
    depth_bias: f32,
    max_lights_per_cluster: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    view_from_world: mat4x4<f32>,
}

@group(5) @binding(0) var<storage, read> cluster_grid: array<ClusterGrid>;
@group(5) @binding(1) var<storage, read> cluster_offsets: array<vec2<u32>>;
@group(5) @binding(2) var<storage, read> cluster_indices: array<u32>;

// The resolved per-fragment punctual-light iteration range. When
// `use_clusters` is 0 the integrators fall back to scanning every punctual
// light (the neutral single-cluster grid, or a non-perspective view); otherwise
// they walk `cluster_indices[offset .. offset + count]`.
struct ClusterSlice {
    use_clusters: u32,
    offset: u32,
    count: u32,
}

// Inverse-transpose of a linear 3x3, carrying normals correctly under a
// non-uniform scale. Copied verbatim from `opaque.wesl`/`visibility_raster.wesl`
// (cofactor columns over the determinant); returns identity for a singular
// matrix so a degenerate transform is a no-op. Mirrors the CPU golden
// `resolve::normal_matrix`.
fn inverse_transpose_3x3(value: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(value[1], value[2]);
    let y = cross(value[2], value[0]);
    let z = cross(value[0], value[1]);
    let determinant = dot(value[2], z);
    if abs(determinant) <= 1.0e-8 {
        return mat3x3<f32>(1.0);
    }
    return mat3x3<f32>(x, y, z) / determinant;
}

// Projects a world-space point through `clip_from_world` and returns the
// reverse-Z, top-left screen UV in `.xy` with the clip-space `w` in `.z`. A
// non-positive `w` (point behind the eye) is reported through `.z` so the
// caller can drop the sample; the UV convention matches the golden
// `project_world_to_screen` in `prism_render_shading::screen_space::motion` and
// `ssr_temporal.wesl`: `(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5)`.
fn project_world_to_uv(clip_from_world: mat4x4<f32>, world: vec3<f32>) -> vec3<f32> {
    let clip = clip_from_world * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return vec3<f32>(0.0, 0.0, clip.w);
    }
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5, clip.w);
}

// Locates the froxel a fragment falls in and returns its light-index slice.
//
// Mirrors the CPU golden addressing exactly: `ClusterGrid::xy_tile` (floor of
// pixel / tile, clamped), `z_slice` (the precomputed `depth_scale/bias` log,
// identical to `GpuClusterGrid::depth_slice`), and `linear_index`
// (`x + dim_x * (y + dim_y * z)`, x fastest).  A grid with no per-cluster
// capacity (the neutral fallback) reports `use_clusters = 0`.
fn resolve_cluster_slice(world_position: vec3<f32>, px: u32, py: u32) -> ClusterSlice {
    var slice: ClusterSlice;
    slice.use_clusters = 0u;
    slice.offset = 0u;
    slice.count = 0u;

    if arrayLength(&cluster_grid) == 0u {
        return slice;
    }
    let grid = cluster_grid[0];
    // The neutral single-cluster fallback carries no capacity; scan all lights.
    if grid.max_lights_per_cluster == 0u || grid.cluster_count <= 1u {
        return slice;
    }
    if grid.dim_x == 0u || grid.dim_y == 0u || grid.dim_z == 0u {
        return slice;
    }

    let tile_x = max(grid.tile_x, 1u);
    let tile_y = max(grid.tile_y, 1u);
    let cx = min(px / tile_x, grid.dim_x - 1u);
    let cy = min(py / tile_y, grid.dim_y - 1u);

    let view_pos = grid.view_from_world * vec4<f32>(world_position, 1.0);
    let depth = max(-view_pos.z, grid.near);
    let raw = grid.depth_scale * log(depth) + grid.depth_bias;
    let cz = min(u32(max(raw, 0.0)), grid.dim_z - 1u);

    let cluster = cx + grid.dim_x * (cy + grid.dim_y * cz);
    if cluster >= arrayLength(&cluster_offsets) || cluster >= grid.cluster_count {
        return slice;
    }

    let oc = cluster_offsets[cluster];
    slice.use_clusters = 1u;
    slice.offset = oc.x;
    slice.count = oc.y;
    return slice;
}

var<immediate> params: ResolveParams;

// Converts a `LightSample` (lighting.wesl) into the reduced `DirectLightSample`
// consumed by the BRDF lobes (brdf.wesl).
fn to_direct_sample(sample: LightSample) -> DirectLightSample {
    var direct: DirectLightSample;
    direct.direction = sample.direction;
    direct.illuminance = sample.illuminance;
    direct.visibility = sample.visibility;
    return direct;
}

// Samples the directional light at `index` and folds its cascaded-shadow
// visibility into the sample's `visibility` term.  `n_dot_l` (against the
// world-space shading normal) drives the slope/normal-offset bias, and the
// view depth is approximated by the camera-to-fragment distance so cascade
// selection needs no view matrix here.  On-device parity of that cascade
// choice still requires hardware verification.
fn sample_directional_shadowed(index: u32, frame: ShadingFrame, world_position: vec3<f32>) -> LightSample {
    var sample = sample_directional_light(directional_lights[index]);
    let n_dot_l = max(dot(frame.normal, sample.direction), 0.0);
    let view_depth = length(world_position - params.view_position.xyz);
    let visibility = directional_shadow_visibility(index, world_position, frame.normal, n_dot_l, view_depth);
    sample.visibility = sample.visibility * visibility;
    return sample;
}

// Samples the punctual light at `index` and folds its shadow into the
// visibility term.  Spot lights (non-zero `spot_scale`) sample their single
// perspective map; point lights sample the distance cube map.  Invalid samples
// (out of range / outside a spot cone) are left untouched so the caller's
// `valid == 0.0` skip still fires.
fn sample_punctual_shadowed(index: u32, frame: ShadingFrame, world_position: vec3<f32>) -> LightSample {
    let light = punctual_lights[index];
    var sample = sample_punctual_light(light, world_position);
    if sample.valid != 0.0 {
        let n_dot_l = max(dot(frame.normal, sample.direction), 0.0);
        var visibility = 1.0;
        if light.spot_scale != 0.0 {
            visibility = spot_shadow_visibility(index, world_position, frame.normal, n_dot_l);
        } else {
            visibility = point_shadow_visibility(index, world_position, n_dot_l);
        }
        sample.visibility = sample.visibility * visibility;
    }
    return sample;
}

// Constant-ambient indirect fallback used when no image-based probe is bound.
// Mirrors `ambient_term` in the CPU reference: Lambertian albedo weighted by
// occlusion and (1 - metallic).
// Split indirect result shared by every ambient/image-based path so the SSR
// composite can conserve energy. total = the full indirect radiance folded
// into scene_color (diffuse + specular). env_specular = the specular half
// alone, which the composite subtracts before substituting the screen-space
// reflection. spec_weight = the environment-BRDF factor (f0 * dfg + bias) *
// occlusion the composite reuses to weight the SSR reflection identically.
struct IndirectResult {
    total: vec3<f32>,
    env_specular: vec3<f32>,
    spec_weight: vec3<f32>,
    // Pre-albedo diffuse *irradiance* term (the diffuse half of `total` divided
    // out by the surface albedo), exported so screen-space GI can gather over
    // the same ambient a missed ray would otherwise receive and the composite
    // can subtract the IBL diffuse before substituting the SSGI bounce. Holds
    // `env.ambient * occlusion` for the constant path and
    // `irradiance * INV_PI * occlusion` for the image-based paths.
    ambient_irradiance: vec3<f32>,
}

// Grazing-angle F0 for the surface, shared by every indirect path.
fn indirect_f0(surface: SurfaceSample) -> vec3<f32> {
    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    return mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);
}

// Constant-ambient indirect fallback used when no image-based probe is bound.
// Mirrors ambient_term in the CPU reference: Lambertian albedo weighted by
// occlusion and (1 - metallic). Carries no environment specular, but still
// exports the analytic environment-BRDF weight so the SSR composite can add a
// Fresnel-correct screen-space reflection even without a probe.
fn ambient_indirect(surface: SurfaceSample, frame: ShadingFrame, env: LightEnvironment) -> IndirectResult {
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let diffuse_weight = 1.0 - clamp(surface.metallic, 0.0, 1.0);
    let diffuse = surface.base_color * env.ambient * (occlusion * diffuse_weight);

    let normal = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let view = brdf_normalize_or(frame.view, normal);
    let n_dot_v = max(dot(normal, view), 1.0e-4);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let spec_weight = (indirect_f0(surface) * dfg.x + dfg.y) * occlusion;

    var out: IndirectResult;
    out.total = diffuse;
    out.env_specular = vec3<f32>(0.0);
    out.spec_weight = spec_weight;
    // Pre-albedo ambient the SSGI gather blends misses toward: `env.ambient`
    // weighted by occlusion (the `diffuse_weight`/base_color that make up the
    // albedo are re-applied at composite, so they are left out here).
    out.ambient_irradiance = env.ambient * occlusion;
    return out;
}

// Low-frequency SH-radiance image-based path (used when the prefiltered cube
// and DFG table are not resident). Mirrors brdf.wesl::image_based, but returns
// the diffuse+specular split so the SSR composite can replace the SH specular
// with the screen-space reflection under the same environment-BRDF weight.
fn image_based_sh(surface: SurfaceSample, frame: ShadingFrame, env: LightEnvironment) -> IndirectResult {
    let normal = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let view = brdf_normalize_or(frame.view, normal);
    let n_dot_v = max(dot(normal, view), 1.0e-4);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let f0 = indirect_f0(surface);

    let irradiance = max(sh_irradiance(env, normal), vec3<f32>(0.0));
    let diffuse_weight = 1.0 - clamp(surface.metallic, 0.0, 1.0);
    let diffuse = surface.base_color * diffuse_weight * irradiance * INV_PI;

    let reflection = reflect(-view, normal);
    let prefiltered = max(sh_radiance(env, reflection), vec3<f32>(0.0));
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let spec_weight = (f0 * dfg.x + dfg.y) * occlusion;
    let env_specular = prefiltered * spec_weight;

    var out: IndirectResult;
    out.total = diffuse * occlusion + env_specular;
    out.env_specular = env_specular;
    out.spec_weight = spec_weight;
    // Pre-albedo ambient: the cosine-convolved SH irradiance over PI, occluded.
    // `diffuse = base_color * diffuse_weight * irradiance * INV_PI`, so dividing
    // the albedo back out leaves `irradiance * INV_PI` (times occlusion).
    out.ambient_irradiance = irradiance * INV_PI * occlusion;
    return out;
}

// Prefiltered image-based indirect term. Mirrors the CPU golden
// `evaluate_image_based_light_specular`: the diffuse half is the SH-irradiance
// Lambertian lobe (identical to `image_based` in `brdf.wesl`), while the
// specular half samples the GGX-prefiltered radiance cube along the reflection
// vector at a roughness-selected mip (`lod = roughness * (mip_count - 1)`) and
// weights it by the integrated DFG table (`F0 * scale + bias`). This replaces
// the low-frequency `sh_radiance` specular approximation so glossy surfaces
// read a sharp mip and rough surfaces read a pre-blurred one. Bound only when
// `RESOLVE_FLAG_IBL_SPECULAR` guarantees both textures are resident; the GPU
// probe carries no independent intensity, so intensity is 1.0 as in the SH
// path, and both halves clamp to non-negative to match the golden guards.
fn image_based_prefiltered(
    surface: SurfaceSample,
    frame: ShadingFrame,
    env: LightEnvironment,
) -> IndirectResult {
    let normal = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let view = brdf_normalize_or(frame.view, normal);
    let n_dot_v = max(dot(normal, view), 1.0e-4);

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);

    // Diffuse: Lambertian albedo lit by cosine-convolved SH irradiance.
    let irradiance = max(sh_irradiance(env, normal), vec3<f32>(0.0));
    let diffuse_weight = 1.0 - metallic;
    let diffuse = surface.base_color * diffuse_weight * irradiance * INV_PI;

    // Specular: sample the prefiltered cube at the roughness LOD, then weight by
    // the integrated split-sum DFG term. `textureNumLevels` returns the mip
    // count baked into the bound cube view so the LOD scale never drifts from
    // `PREFILTERED_ENV_MIP_COUNT`.
    let reflection = reflect(-view, normal);
    let mip_max = f32(max(textureNumLevels(prefiltered_env), 1u) - 1u);
    let lod = roughness * mip_max;
    let prefiltered = max(
        textureSampleLevel(prefiltered_env, prefiltered_sampler, reflection, lod).rgb,
        vec3<f32>(0.0),
    );
    let dfg = textureSampleLevel(dfg_lut, dfg_sampler, vec2<f32>(n_dot_v, roughness), 0.0).xy;
    let spec_weight = (f0 * dfg.x + dfg.y) * occlusion;
    let env_specular = prefiltered * spec_weight;

    var out: IndirectResult;
    out.total = diffuse * occlusion + env_specular;
    out.env_specular = env_specular;
    out.spec_weight = spec_weight;
    // Pre-albedo ambient: identical diffuse lobe to the SH path.
    out.ambient_irradiance = irradiance * INV_PI * occlusion;
    return out;
}

// Physically based integrator shared by every lit closure. Accumulates the
// principled GGX lobe over all analytic lights, then adds the indirect and
// emissive terms exactly once. Mirrors the CPU `_ =>` principled branch.
fn shade_principled(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + principled_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + principled_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

// Non-photoreal integrator: the quantized toon lobe over every analytic light,
// plus indirect and emissive. Mirrors the CPU `MaterialShadingClass::Npr` arm.
fn shade_toon(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);
    let stylized = env.stylized;

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + stylized_direct(lit, frame, to_direct_sample(sample), stylized);
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + stylized_direct(lit, frame, to_direct_sample(sample), stylized);
    }

    return accumulated + indirect + emissive;
}

// Cloth integrator: the fabric lobe over every analytic light, plus indirect
// and emissive. Mirrors the CPU `MaterialShadingClass::Cloth` arm.
fn shade_cloth(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + cloth_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + cloth_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

// Integrates the subsurface (SSS) lobe over every analytic light. Mirrors the
// CPU `shade_subsurface` closure; the lobe accepts back-facing lights so
// radiance can transmit through thin geometry.
fn shade_subsurface(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + subsurface_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + subsurface_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

// Integrates the hair (strand) lobe over every analytic light. Mirrors the CPU
// `shade_hair` closure: the Kajiya-Kay diffuse survives backward grazing light
// so the fibre keeps its translucent rim, while the two shifted Scheuermann
// highlights ride the strand tangent. Emissive is folded back exactly once.
fn shade_hair(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + hair_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + hair_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

// Water integrator: the single-layer water lobe over every analytic light,
// plus indirect and emissive. Mirrors the CPU `MaterialShadingClass::Water`
// arm; the Fresnel split routes energy between the surface glint and the
// Beer-Lambert-absorbed refracted body.
fn shade_water(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + water_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + water_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

fn shade_clearcoat(
    lit: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    indirect: vec3<f32>,
    emissive: vec3<f32>,
    env: LightEnvironment,
    slice: ClusterSlice,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_shadowed(i, frame, world_position);
        accumulated = accumulated + clearcoat_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    let punctual_loop = select(punctual_total, slice.count, slice.use_clusters != 0u);
    for (var s = 0u; s < punctual_loop; s = s + 1u) {
        var punctual_i = s;
        if slice.use_clusters != 0u {
            let idx_pos = slice.offset + s;
            if idx_pos >= arrayLength(&cluster_indices) { break; }
            punctual_i = cluster_indices[idx_pos];
        }
        if punctual_i >= arrayLength(&punctual_lights) { continue; }
        let sample = sample_punctual_shadowed(punctual_i, frame, world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + clearcoat_direct(lit, frame, to_direct_sample(sample));
    }

    return accumulated + indirect + emissive;
}

@compute @workgroup_size(64)
fn shading_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let local = id.x;

    // Drain this dispatch's slice of the compacted worklist. `class_counts`
    // and `class_offsets` were produced by the classification prefix scan.
    if params.shading_class >= arrayLength(&class_counts)
        || params.shading_class >= arrayLength(&class_offsets) {
        return;
    }
    if local >= class_counts[params.shading_class] {
        return;
    }
    let work_index = class_offsets[params.shading_class] + local;
    if work_index >= arrayLength(&work) {
        return;
    }
    let item = work[work_index];
    if item.shading_class != params.shading_class {
        return;
    }

    // Unflatten the pixel index and reject uncovered background samples.
    if params.width == 0u {
        return;
    }
    let coord = vec2<i32>(
        i32(item.pixel_index % params.width),
        i32(item.pixel_index / params.width),
    );
    let ids = textureLoad(visibility_ids, coord, 0);
    let metadata = textureLoad(visibility_metadata, coord, 0);
    if ids.x == VISIBILITY_BACKGROUND || metadata.x == VISIBILITY_BACKGROUND {
        return;
    }

    // ids = [scene_index, scene_generation, primitive_index, geometry_lod].
    let scene_index = ids.x;
    if scene_index >= arrayLength(&scene_instances) {
        return;
    }
    let instance = scene_instances[scene_index];
    if instance.is_active == 0u || instance.generation != ids.y {
        return;
    }

    let geometry_index = instance.geometry_index;
    if geometry_index >= arrayLength(&geometry_headers) {
        return;
    }
    let header = geometry_headers[geometry_index];
    if (header.flags & SHADING_GEOMETRY_FLAG_ACTIVE) == 0u {
        return;
    }
    if header.generation != instance.geometry_generation {
        return;
    }

    // Resolve the primitive row: bounded by the header's logical count and by
    // the physical primitive-buffer length.
    let primitive_index = ids.z;
    if primitive_index >= header.primitive_count {
        return;
    }
    let primitive_row = header.primitive_offset + primitive_index;
    if primitive_row >= arrayLength(&geometry_primitives) {
        return;
    }
    let primitive = geometry_primitives[primitive_row];

    // Resolve the three vertex rows: local indices bounded by the header's
    // vertex count, absolute rows bounded by the physical vertex-buffer length.
    let index_a = primitive.indices[0];
    let index_b = primitive.indices[1];
    let index_c = primitive.indices[2];
    if index_a >= header.vertex_count
        || index_b >= header.vertex_count
        || index_c >= header.vertex_count {
        return;
    }
    let row_a = header.vertex_offset + index_a;
    let row_b = header.vertex_offset + index_b;
    let row_c = header.vertex_offset + index_c;
    if row_a >= arrayLength(&geometry_vertices)
        || row_b >= arrayLength(&geometry_vertices)
        || row_c >= arrayLength(&geometry_vertices) {
        return;
    }
    let vertex_a = geometry_vertices[row_a];
    let vertex_b = geometry_vertices[row_b];
    let vertex_c = geometry_vertices[row_c];

    let weights = surface_decode_barycentrics(metadata.w);
    var geometry: SurfaceSampleGeometry =
        interpolate_surface(vertex_a, vertex_b, vertex_c, primitive.flags, weights);

    // Lift the reconstructed *local-space* surface into world space with the
    // instance's `world_from_local` before any lighting runs, mirroring the CPU
    // golden `resolve_pixel`: the position by the full affine, the normal by its
    // inverse-transpose (correct under a non-uniform scale) and the
    // tangent/bitangent by the linear part. The UV stays local. Guarded against
    // a short/stale transform buffer; the identity transform leaves every field
    // untouched so an untransformed instance shades exactly as before.
    // Keep the reconstructed *local* position: the current transform overwrites
    // `geometry.position` in place below, but the motion vector also needs the
    // same local point placed in the *previous* frame's world space.
    let local_position = geometry.position;
    var world_cur = local_position;
    var world_prev = local_position;
    if scene_index < arrayLength(&scene_current_transforms) {
        let world_from_local =
            affine3_to_square(scene_current_transforms[scene_index].world_from_local);
        let linear = mat3x3<f32>(
            world_from_local[0].xyz,
            world_from_local[1].xyz,
            world_from_local[2].xyz,
        );
        geometry.position = (world_from_local * vec4<f32>(local_position, 1.0)).xyz;
        geometry.normal =
            brdf_normalize_or(inverse_transpose_3x3(linear) * geometry.normal, geometry.normal);
        geometry.tangent = brdf_normalize_or(linear * geometry.tangent, geometry.tangent);
        geometry.bitangent = brdf_normalize_or(linear * geometry.bitangent, geometry.bitangent);
        world_cur = geometry.position;
        // Place the same local point in last frame's world space for the motion
        // vector. Fall back to the current world position (zero object motion)
        // when the previous-transform buffer is short/stale, so a freshly
        // spawned instance never reprojects to garbage.
        if scene_index < arrayLength(&scene_previous_transforms) {
            let prev_from_local =
                affine3_to_square(scene_previous_transforms[scene_index].world_from_local);
            world_prev = (prev_from_local * vec4<f32>(local_position, 1.0)).xyz;
        } else {
            world_prev = world_cur;
        }
    }

    // Resolve and generation-validate the material record.
    let material_index = metadata.x;
    if material_index >= arrayLength(&material_headers) {
        return;
    }
    let material_header = material_headers[material_index];
    if material_header.is_active == 0u || material_header.generation != metadata.y {
        return;
    }
    let parameter_row = material_header.parameter_offset;
    if parameter_row >= arrayLength(&material_parameters) {
        return;
    }
    let parameters = material_parameters[parameter_row];

    // Fold every bound texture over the authored factors *before* assembling
    // the BSDF sample, so the indirect/ambient terms below observe the
    // texture-modulated base color, emissive and occlusion. Guarded twice: by
    // the header's logical `texture_count` and by the physical table length.
    // Mirrors `resolve_pixel`'s texture fold in the CPU golden arm-for-arm.
    var sampled = sampled_material_defaults(parameters);
    let tex_end = material_header.texture_offset + material_header.texture_count;
    let tex_physical = arrayLength(&material_textures);
    for (var i = material_header.texture_offset; i < tex_end && i < tex_physical; i = i + 1u) {
        let record = material_textures[i];
        let texel = sample_texture_slot(record, geometry.uv);
        sampled = fold_material_texel(record, texel, parameters.normal_scale, sampled);
    }

    // Assemble the analytic BSDF sample from the stable surface parameters.
    var surface: SurfaceSample;
    surface.base_color = sampled.base_color;
    surface.metallic = sampled.metallic;
    surface.perceptual_roughness = sampled.perceptual_roughness;
    surface.reflectance = parameters.reflectance;
    // Fold the screen-space GTAO visibility into the material occlusion so the
    // indirect/ambient term is darkened in geometric creases, mirroring
    // `resolve_pixel` in the CPU golden. The fallback texture and the disabled
    // flag both leave this a no-op (visibility 1).
    var screen_space_ao = 1.0;
    if (params.flags & RESOLVE_FLAG_GTAO) != 0u {
        screen_space_ao = textureLoad(gtao_ao, coord, 0).x;
    }
    surface.ambient_occlusion = clamp(sampled.occlusion * screen_space_ao, 0.0, 1.0);
    surface.emissive = sampled.emissive;
    surface.clearcoat = sampled.clearcoat;
    surface.clearcoat_roughness = sampled.clearcoat_roughness;
    surface.sheen = parameters.sheen;
    surface.subsurface = parameters.subsurface;
    surface.thickness = parameters.thickness;
    surface.anisotropy = parameters.anisotropy;
    surface.anisotropy_rotation = parameters.anisotropy_rotation;

    var frame: ShadingFrame;
    // A bound normal map rotates its tangent-space sample into world space
    // against the interpolated basis; otherwise the geometric normal stands.
    if sampled.has_normal_map {
        var basis: TangentBasis;
        basis.tangent = geometry.tangent;
        basis.bitangent = geometry.bitangent;
        basis.degenerate = false;
        frame.normal = apply_tangent_space_normal(basis, geometry.normal, sampled.normal_tangent);
    } else {
        frame.normal = geometry.normal;
    }
    frame.view = brdf_normalize_or(params.view_position.xyz - geometry.position, geometry.normal);
    frame.tangent = geometry.tangent;
    frame.bitangent = geometry.bitangent;

    // Per-light evaluation runs against an emissive-free copy so accumulation
    // over many lights never multiplies self-illumination; emissive is folded
    // back in exactly once by the integrators / Unlit arm.
    let emissive = surface.emissive;
    var lit = surface;
    lit.emissive = vec3<f32>(0.0);

    let env = light_env[0];
    var indirect_result: IndirectResult;
    if environment_has_image_based(env) {
        // A resident prefiltered cube + DFG table takes the real split-sum
        // specular path; otherwise fall back to the low-frequency SH-radiance
        // approximation (image_based_sh, a local twin of brdf.wesl::image_based).
        if (params.flags & RESOLVE_FLAG_IBL_SPECULAR) != 0u {
            indirect_result = image_based_prefiltered(lit, frame, env);
        } else {
            indirect_result = image_based_sh(lit, frame, env);
        }
    } else {
        indirect_result = ambient_indirect(lit, frame, env);
    }
    let indirect = indirect_result.total;

    // Explicit per-class dispatch. Subsurface/clearcoat/cloth/hair/water now
    // run dedicated closures; only custom still shares the principled
    // integrator, and each keeps its own arm so a real closure drops in
    // independently.
    let slice = resolve_cluster_slice(geometry.position, u32(coord.x), u32(coord.y));
    var color: vec3<f32>;
    switch params.shading_class {
        case SHADING_CLASS_PRINCIPLED: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_UNLIT: {
            color = surface.base_color + surface.emissive;
        }
        case SHADING_CLASS_SUBSURFACE: {
            color = shade_subsurface(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_CLEARCOAT: {
            color = shade_clearcoat(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_CLOTH: {
            color = shade_cloth(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_HAIR: {
            color = shade_hair(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_WATER: {
            color = shade_water(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_NPR: {
            color = shade_toon(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        case SHADING_CLASS_CUSTOM: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
        default: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env, slice);
        }
    }

    textureStore(scene_color, coord, vec4<f32>(color, 1.0));
    // SSR energy-conservation exports: the IBL specular this pixel already
    // received and the environment-BRDF weight it was scaled by. The SSR
    // composite subtracts env_specular and re-adds the screen-space
    // reflection under spec_weight, so a confident hit swaps IBL specular for
    // SSR without double-counting energy (a miss leaves scene_color intact).
    textureStore(ssr_env_specular, coord, vec4<f32>(indirect_result.env_specular, 0.0));
    textureStore(ssr_spec_weight, coord, vec4<f32>(indirect_result.spec_weight, 0.0));

    // Motion vector: this pixel's screen-space displacement between frames,
    // `cur_uv - prev_uv`, capturing camera *and* per-object motion. Both ends
    // are projected with their own frame's view-projection; a point that lands
    // behind either eye (w <= 0) yields zero motion so the temporal reprojection
    // simply falls back to the current sample instead of chasing garbage.
    let cur = project_world_to_uv(motion_matrices.clip_from_world, world_cur);
    let prev = project_world_to_uv(motion_matrices.previous_clip_from_world, world_prev);
    var motion = vec2<f32>(0.0, 0.0);
    if cur.z > 0.0 && prev.z > 0.0 {
        motion = cur.xy - prev.xy;
    }
    textureStore(motion_vectors, coord, vec4<f32>(motion, 0.0, 0.0));

    // Screen-space GI exports. The albedo is the diffuse Lambertian reflectance
    // (`base_color * (1 - metallic)`), and `ambient_irradiance` is the diffuse
    // half of the indirect term with that albedo divided out, so their product
    // reconstructs exactly the IBL diffuse folded into `scene_color`. The SSGI
    // trace gathers over `ssgi_ambient`; the composite multiplies its result by
    // `ssgi_albedo` and blends it over `ssgi_albedo * ssgi_ambient` under the
    // per-pixel confidence, leaving covered pixels energy-consistent on a miss.
    let ssgi_diffuse_weight = 1.0 - clamp(lit.metallic, 0.0, 1.0);
    let ssgi_lambert_albedo = lit.base_color * ssgi_diffuse_weight;
    textureStore(ssgi_albedo, coord, vec4<f32>(ssgi_lambert_albedo, 0.0));
    textureStore(ssgi_ambient, coord, vec4<f32>(indirect_result.ambient_irradiance, 0.0));
}
