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
    toon_direct,
    image_based,
};
import prism_render_scene::shaders::cloth::{
    cloth_direct,
};
import prism_render_scene::shaders::subsurface::{
    subsurface_direct,
};
import prism_render_scene::shaders::lighting::{
    DirectionalLight,
    PunctualLight,
    LightEnvironment,
    LightSample,
    environment_has_image_based,
    sample_directional_light,
    sample_punctual_light,
};
import prism_render_scene::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};
import prism_render_scene::shaders::gpu_scene::{
    GpuSceneInstance,
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
    pad: u32,
    view_position: vec4<f32>,
}

// One compacted work item. Mirrors `ShadingWorkItem` (16 bytes) and the
// `WorkItem` produced by `material_classification.wesl`.
struct WorkItem {
    pixel_index: u32,
    material_index: u32,
    shading_class: u32,
    flags: u32,
}

// --- group 0: visibility inputs + HDR output ---
@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
@group(0) @binding(2) var scene_color: texture_storage_2d<rgba16float, write>;

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

// --- group 3: analytic lights (shared LightBindGroup layout) ---
@group(3) @binding(0) var<storage, read> directional_lights: array<DirectionalLight>;
@group(3) @binding(1) var<storage, read> punctual_lights: array<PunctualLight>;
@group(3) @binding(2) var<storage, read> light_env: array<LightEnvironment>;

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

// Constant-ambient indirect fallback used when no image-based probe is bound.
// Mirrors `ambient_term` in the CPU reference: Lambertian albedo weighted by
// occlusion and (1 - metallic).
fn ambient_indirect(surface: SurfaceSample, env: LightEnvironment) -> vec3<f32> {
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let diffuse = 1.0 - clamp(surface.metallic, 0.0, 1.0);
    return surface.base_color * env.ambient * (occlusion * diffuse);
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
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_light(directional_lights[i]);
        accumulated = accumulated + principled_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    for (var i = 0u; i < punctual_total; i = i + 1u) {
        let sample = sample_punctual_light(punctual_lights[i], world_position);
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
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);
    let bands = env.toon_bands;

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_light(directional_lights[i]);
        accumulated = accumulated + toon_direct(lit, frame, to_direct_sample(sample), bands);
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    for (var i = 0u; i < punctual_total; i = i + 1u) {
        let sample = sample_punctual_light(punctual_lights[i], world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + toon_direct(lit, frame, to_direct_sample(sample), bands);
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
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_light(directional_lights[i]);
        accumulated = accumulated + cloth_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    for (var i = 0u; i < punctual_total; i = i + 1u) {
        let sample = sample_punctual_light(punctual_lights[i], world_position);
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
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);

    let directional_total = min(env.directional_count, arrayLength(&directional_lights));
    for (var i = 0u; i < directional_total; i = i + 1u) {
        let sample = sample_directional_light(directional_lights[i]);
        accumulated = accumulated + subsurface_direct(lit, frame, to_direct_sample(sample));
    }

    let punctual_total = min(env.punctual_count, arrayLength(&punctual_lights));
    for (var i = 0u; i < punctual_total; i = i + 1u) {
        let sample = sample_punctual_light(punctual_lights[i], world_position);
        if sample.valid == 0.0 {
            continue;
        }
        accumulated = accumulated + subsurface_direct(lit, frame, to_direct_sample(sample));
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
    let geometry: SurfaceSampleGeometry =
        interpolate_surface(vertex_a, vertex_b, vertex_c, primitive.flags, weights);

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
    surface.ambient_occlusion = sampled.occlusion;
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
    var indirect: vec3<f32>;
    if environment_has_image_based(env) {
        indirect = image_based(lit, frame, env);
    } else {
        indirect = ambient_indirect(lit, env);
    }

    // Explicit per-class dispatch. Specialized subsurface/clearcoat/cloth/hair/
    // water lobes currently share the principled integrator; each keeps its own
    // arm so a real closure can be dropped in without disturbing the others.
    var color: vec3<f32>;
    switch params.shading_class {
        case SHADING_CLASS_PRINCIPLED: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_UNLIT: {
            color = surface.base_color + surface.emissive;
        }
        case SHADING_CLASS_SUBSURFACE: {
            color = shade_subsurface(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_CLEARCOAT: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_CLOTH: {
            color = shade_cloth(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_HAIR: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_WATER: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_NPR: {
            color = shade_toon(lit, frame, geometry.position, indirect, emissive, env);
        }
        case SHADING_CLASS_CUSTOM: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
        default: {
            color = shade_principled(lit, frame, geometry.position, indirect, emissive, env);
        }
    }

    textureStore(scene_color, coord, vec4<f32>(color, 1.0));
}
