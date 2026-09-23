//! GPU validation and compaction stage for unified visibility. CPU reference
//! remains authoritative until readback parity and GPU sort/binning graduate.

struct GpuSceneInstance {
    generation: u32, flags: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    render_layers: u32, is_active: u32,
}
struct GpuSceneTransform { world_from_local: mat3x4<f32> }
struct GpuSceneBounds { center_radius: vec4<f32>, half_extents: vec4<f32> }
struct MaterialHeader {
    generation: u32, revision: u32, shading_model: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
}
struct SurfaceParameters {
    base_color: vec4<f32>, emissive: vec4<f32>,
    metallic: f32, perceptual_roughness: f32, reflectance: f32, ambient_occlusion: f32,
    normal_scale: f32, alpha_cutoff: f32, transmission: f32, thickness: f32,
    clearcoat: f32, clearcoat_roughness: f32, anisotropy: f32, anisotropy_rotation: f32,
    sheen: f32, subsurface: f32, index_of_refraction: f32, dispersion: f32,
}
struct MaterialTexture { index: u32, generation: u32, semantic: u32, sampler_index: u32 }
struct GeometryHeader {
    generation: u32, revision: u32, lod_offset: u32, lod_count: u32,
    primitive_start: u32, primitive_count: u32, flags: u32, padding: u32,
}
struct GeometryLod {
    element_count: u32, first_element: u32, base_vertex: i32, vertex_count: u32,
    vertex_buffer_class: u32, index_buffer_class: u32, flags: u32, level: u32,
    screen_error: f32, padding: vec3<u32>,
}
struct VisibilityView {
    clip_from_world: mat4x4<f32>, previous_clip_from_world: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>, world_position_lod_scale: vec4<f32>,
    viewport: vec4<u32>, handle_index: u32, handle_generation: u32,
    layer_mask: u32, flags: u32, history_epoch_low: u32, history_epoch_high: u32,
    padding: vec2<u32>,
}
struct VisibilityWorkItem {
    scene_index: u32, scene_generation: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    lod_or_cluster: u32, pass_mask: u32,
    sort_key_low: u32, sort_key_high: u32, padding: vec2<u32>,
}
struct VisibilityRange { view_index: u32, view_generation: u32, start: u32, count: u32 }
struct VisibilityCounter { visible_count: atomic<u32>, rejected_count: atomic<u32>, overflow_count: atomic<u32>, padding: u32 }
struct VisibilityIndirect { index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32 }

@group(0) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(0) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(0) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(0) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;
@group(1) @binding(0) var<storage, read> material_headers: array<MaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<SurfaceParameters>;
@group(1) @binding(2) var<storage, read> material_textures: array<MaterialTexture>;
@group(2) @binding(0) var<storage, read> geometry_headers: array<GeometryHeader>;
@group(2) @binding(1) var<storage, read> geometry_lods: array<GeometryLod>;
@group(3) @binding(0) var<storage, read> views: array<VisibilityView>;
@group(3) @binding(1) var<storage, read_write> counters: array<VisibilityCounter>;
@group(3) @binding(2) var<storage, read_write> work_items: array<VisibilityWorkItem>;
@group(3) @binding(3) var<storage, read_write> ranges: array<VisibilityRange>;
@group(3) @binding(4) var<storage, read_write> indirect: array<VisibilityIndirect>;
@group(3) @binding(5) var<storage, read_write> overflow: array<atomic<u32>>;
@group(3) @binding(6) var<storage, read_write> previous_lods: array<u32>;

struct VisibilityDispatch {
    view_index: u32, candidate_count: u32, output_start: u32, output_end: u32,
    indirect_first_instance: u32, padding: vec3<u32>,
}
var<immediate> dispatch: VisibilityDispatch;

fn pass_mask(render_class: u32, flags: u32) -> u32 {
    var mask = select(4u, 1u, render_class <= 1u || render_class == 11u || render_class == 13u);
    mask = select(mask, 2u, render_class == 2u || render_class == 3u);
    if (flags & 1u) == 0u { mask |= 8u; }
    return mask | 16u | 32u | 64u;
}

fn sphere_visible(view: VisibilityView, sphere: vec4<f32>) -> bool {
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = view.frustum_planes[plane_index];
        if dot(plane.xyz, sphere.xyz) + plane.w < -sphere.w { return false; }
    }
    return true;
}

fn projected_size(view: VisibilityView, sphere: vec4<f32>) -> f32 {
    let distance = max(distance(view.world_position_lod_scale.xyz, sphere.xyz), 1.0e-4);
    return sphere.w * view.world_position_lod_scale.w / distance;
}

fn resolve_geometry_lod(header: GeometryHeader, projected: f32, previous: u32) -> u32 {
    var resolved = 0xffffffffu;
    var resolved_error = -1.0;
    var fallback = 0xffffffffu;
    for (var local = 0u; local < header.lod_count; local += 1u) {
        let index = header.lod_offset + local;
        if index >= arrayLength(&geometry_lods) { break; }
        let lod = geometry_lods[index];
        let resident = (lod.flags & 4u) != 0u;
        if resident && (lod.flags & 8u) != 0u { fallback = index; }
        let projected_error = lod.screen_error * projected;
        if resident && projected_error <= 1.0 && projected_error > resolved_error {
            resolved = index;
            resolved_error = projected_error;
        }
    }
    if previous != 0xffffffffu {
        for (var local = 0u; local < header.lod_count; local += 1u) {
            let index = header.lod_offset + local;
            if index >= arrayLength(&geometry_lods) { break; }
            let lod = geometry_lods[index];
            if lod.level == previous && (lod.flags & 4u) != 0u
                && abs(lod.screen_error * projected - 1.0) < 0.15 {
                return index;
            }
        }
    }
    return select(fallback, resolved, resolved != 0xffffffffu);
}

fn first_resident_geometry_lod(header: GeometryHeader) -> u32 {
    for (var local = 0u; local < header.lod_count; local += 1u) {
        let index = header.lod_offset + local;
        if index >= arrayLength(&geometry_lods) { break; }
        if (geometry_lods[index].flags & 4u) != 0u { return index; }
    }
    return 0xffffffffu;
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) id: vec3<u32>) {
    let candidate = id.x;
    if candidate >= dispatch.candidate_count || candidate >= arrayLength(&scene_instances) { return; }
    let instance = scene_instances[candidate];
    let view = views[dispatch.view_index];
    let scene_valid = instance.is_active != 0u && instance.generation != 0u;
    let layer_visible = (instance.render_layers & view.layer_mask) != 0u;
    let sphere = scene_bounds[candidate].center_radius;
    let frustum_visible = sphere_visible(view, sphere);
    let material_in_bounds = instance.material_index < arrayLength(&material_headers);
    let requested_index = select(0u, instance.material_index, material_in_bounds);
    let requested = material_headers[requested_index];
    let requested_valid = material_in_bounds
        && requested.is_active != 0u
        && requested.generation == instance.material_generation;
    let material_index = select(0u, requested_index, requested_valid);
    let material = material_headers[material_index];
    let geometry_in_bounds = instance.geometry_index < arrayLength(&geometry_headers);
    let geometry_index = select(0u, instance.geometry_index, geometry_in_bounds);
    let geometry = geometry_headers[geometry_index];
    let geometry_valid = geometry_in_bounds && (geometry.flags & 1u) != 0u
        && geometry.generation == instance.geometry_generation;
    let history_index = dispatch.output_start + candidate;
    var previous_lod = 0xffffffffu;
    if history_index < dispatch.output_end && history_index < arrayLength(&previous_lods) {
        previous_lod = previous_lods[history_index];
    }
    let resolved_lod = resolve_geometry_lod(geometry, projected_size(view, sphere), previous_lod);
    var lod_index = first_resident_geometry_lod(geometry);
    if resolved_lod != 0xffffffffu { lod_index = resolved_lod; }
    let lod_valid = geometry_valid && lod_index != 0xffffffffu;
    if !scene_valid || !layer_visible || !frustum_visible || !lod_valid {
        atomicAdd(&counters[dispatch.view_index].rejected_count, 1u);
        return;
    }
    let local = atomicAdd(&counters[dispatch.view_index].visible_count, 1u);
    let output = dispatch.output_start + local;
    if output >= dispatch.output_end || output >= arrayLength(&work_items) {
        atomicAdd(&counters[dispatch.view_index].overflow_count, 1u);
        atomicStore(&overflow[dispatch.view_index], 1u);
        return;
    }
    work_items[output] = VisibilityWorkItem(
        candidate, instance.generation,
        instance.geometry_index, instance.geometry_generation,
        material_index, instance.material_generation,
        geometry_lods[lod_index].level, pass_mask(material.render_class, instance.flags),
        instance.geometry_index, material.render_class, vec2<u32>(0u),
    );
    let lod = geometry_lods[lod_index];
    if history_index < dispatch.output_end && history_index < arrayLength(&previous_lods) {
        previous_lods[history_index] = lod.level;
    }
    let indexed = (lod.flags & 2u) != 0u;
    indirect[output] = VisibilityIndirect(
        select(0u, lod.element_count, indexed),
        1u, lod.first_element, lod.base_vertex,
        select(0u, candidate, dispatch.indirect_first_instance != 0u),
    );
}
