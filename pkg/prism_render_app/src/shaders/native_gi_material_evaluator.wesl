import prism_render_material::shaders::material::PrismMaterialHeader;
import prism_render_material::shaders::material::PrismMaterialTexture;
import prism_render_material::shaders::material_unpack::{
    PRISM_LOBE_ALL,
    PRISM_SURFACE_MAX_WORDS,
    prism_packed_word_count,
    prism_unpack_surface,
};
import prism_render_material::shaders::material_virtual_texture::{
    MaterialVirtualTextureConfig,
    material_vt_atlas_uv,
    material_vt_empty_resolution,
    material_vt_geometry_is_valid,
    material_vt_page_request,
    material_vt_resolve,
    material_vt_sample_resolved,
    material_vt_slot_is_valid,
};

const INVALID: u32 = 0xffffffffu;
const HIT_WORDS: u32 = 32u;
const HIT_ATTRIBUTE_WORDS: u32 = 40u;
const MATERIAL_EVALUATOR_ABI_VERSION: u32 = 1u;
const RESPONSE_PROGRAM_KIND_UNLIT: u32 = 2u;
const BUILTIN_UNLIT_PROGRAM_ID: u32 = 2u;
const TEXTURE_SEMANTIC_BASE_COLOR: u32 = 0u;
const TEXTURE_SEMANTIC_EMISSIVE: u32 = 1u;
const NO_VIRTUAL_TEXTURE: u32 = 0xffffffffu;
const PORTABLE_TEXTURE_ATLAS: u32 = 0xfffffffeu;
const RESERVED_TEXTURE_GENERATION: u32 = 1u;

struct NativeGiMaterialEvaluatorConfig {
    abi_version: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    execution_table_version: u32,
    vt_generation_low: u32,
    vt_generation_high: u32,
    material_header_count: u32,
    parameter_word_count: u32,
    texture_row_count: u32,
    execution_plan_count: u32,
    execution_program_count: u32,
    texture_slot_count: u32,
    vt_page_table_word_count: u32,
    flags: u32,
    padding: vec2<u32>,
};

struct ExecutionPlanRow {
    left_program: u32,
    right_program: u32,
    execution_family: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    reserved: u32,
};

struct ExecutionProgramRow {
    response_kind: u32,
    package_index: u32,
    permutation_count_low: u32,
    permutation_count_high: u32,
    program_id: u32,
};

struct NativeGiMaterialEvaluationRecord {
    response_rgb: vec3<f32>,
    valid: u32,
    instance_id: u32,
    geometry_id: u32,
    material_index: u32,
    material_generation: u32,
    material_revision: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    execution_plan_index: u32,
    execution_program_index: u32,
    execution_package_index: u32,
    execution_permutation_low: u32,
    execution_permutation_high: u32,
};

@group(0) @binding(0) var<storage, read> trace_hits: array<u32>;
@group(0) @binding(1) var<storage, read> hit_attributes: array<u32>;
@group(0) @binding(2) var<uniform> evaluator_config: NativeGiMaterialEvaluatorConfig;
@group(0) @binding(3) var<storage, read_write> evaluations: array<NativeGiMaterialEvaluationRecord>;

@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<u32>;
@group(1) @binding(2) var<storage, read> material_texture_rows: array<PrismMaterialTexture>;
@group(1) @binding(3) var<storage, read> execution_plans: array<ExecutionPlanRow>;
@group(1) @binding(4) var<storage, read> execution_programs: array<ExecutionProgramRow>;
// Binding 5 is declared by material_virtual_texture.wesl.
@group(1) @binding(6) var<storage, read> material_vt_config: MaterialVirtualTextureConfig;
@group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@group(1) @binding(8) var material_vt_sampler: sampler;

fn invalid_evaluation() -> NativeGiMaterialEvaluationRecord {
    return NativeGiMaterialEvaluationRecord(
        vec3<f32>(0.0),
        0u,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
    );
}

fn hit_matches_attributes(hit_base: u32, attribute_base: u32) -> bool {
    if (trace_hits[hit_base] != 1u) { return false; }
    // Hit identity is [instance, geometry, material slot, generation, revision,
    // epoch low/high, plan, program, package, permutation low/high].
    for (var word = 0u; word < 12u; word += 1u) {
        if (trace_hits[hit_base + 6u + word] != hit_attributes[attribute_base + word]) {
            return false;
        }
    }
    return true;
}

struct TextureSample {
    valid: bool,
    value: vec4<f32>,
}

fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(value: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(value.x),
        srgb_channel_to_linear(value.y),
        srgb_channel_to_linear(value.z),
    );
}

fn fallback_texture_row_is_valid(row: PrismMaterialTexture) -> bool {
    return row.virtual_texture_id == NO_VIRTUAL_TEXTURE
        && row.index <= 2u
        && row.sampler_index == row.index
        && row.generation == RESERVED_TEXTURE_GENERATION
        && row.virtual_texture_width == 0u
        && row.virtual_texture_height == 0u
        && row.virtual_texture_page_size == 0u;
}

fn sample_vt_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    var out: TextureSample;
    out.valid = false;
    out.value = vec4<f32>(0.0);
    if row.generation == 0u
        || evaluator_config.vt_page_table_word_count == 0u
        || !(evaluator_config.vt_generation_low != 0u
            || evaluator_config.vt_generation_high != 0u)
        || material_vt_config.enabled == 0u
        || !material_vt_geometry_is_valid(material_vt_config)
        || row.virtual_texture_width == 0u
        || row.virtual_texture_height == 0u
        || row.virtual_texture_page_size != material_vt_config.page_size {
        return out;
    }

    var resolution = material_vt_empty_resolution(true);
    if row.virtual_texture_id == PORTABLE_TEXTURE_ATLAS {
        if !material_vt_slot_is_valid(row.index, material_vt_config)
            || row.virtual_texture_width > row.virtual_texture_page_size
            || row.virtual_texture_height > row.virtual_texture_page_size {
            return out;
        }
        let slots_per_layer = material_vt_config.tiles_per_row
            * material_vt_config.tiles_per_row;
        let atlas_dimensions = textureDimensions(material_vt_atlas);
        if atlas_dimensions.x != material_vt_config.tile_extent_px * material_vt_config.tiles_per_row
            || atlas_dimensions.y != material_vt_config.tile_extent_px * material_vt_config.tiles_per_row
            || textureNumLayers(material_vt_atlas) < material_vt_config.atlas_layers {
            return out;
        }
        let source_uv = clamp(
            uv * vec2<f32>(
                f32(row.virtual_texture_width) / f32(row.virtual_texture_page_size),
                f32(row.virtual_texture_height) / f32(row.virtual_texture_page_size),
            ),
            vec2<f32>(0.0),
            vec2<f32>(1.0),
        );
        resolution.hit = true;
        resolution.slot = row.index;
        resolution.atlas_layer = row.index / max(slots_per_layer, 1u);
        resolution.page_local_uv = source_uv;
        resolution.atlas_uv = material_vt_atlas_uv(
            source_uv,
            row.index,
            material_vt_config.tile_extent_px,
            material_vt_config.tiles_per_row,
            material_vt_config.page_size,
            material_vt_config.gutter_px,
        );
    } else if row.virtual_texture_id != NO_VIRTUAL_TEXTURE {
        let request = material_vt_page_request(
            row.virtual_texture_id,
            row.virtual_texture_width,
            row.virtual_texture_height,
            row.virtual_texture_page_size,
            uv,
            0.0,
        );
        if !request.valid {
            return out;
        }
        resolution = material_vt_resolve(
            material_vt_config,
            row.virtual_texture_id,
            0u,
            request.mip,
            request.page_x,
            request.page_y,
            request.local_uv,
        );
    } else {
        return out;
    }
    if !resolution.hit || !material_vt_slot_is_valid(resolution.slot, material_vt_config) {
        return out;
    }
    let atlas_dimensions = textureDimensions(material_vt_atlas);
    if resolution.atlas_layer >= textureNumLayers(material_vt_atlas)
        || atlas_dimensions.x == 0u
        || atlas_dimensions.y == 0u {
        return out;
    }
    out.value = material_vt_sample_resolved(material_vt_atlas, material_vt_sampler, resolution);
    out.valid = finite(out.value.x)
        && finite(out.value.y)
        && finite(out.value.z)
        && finite(out.value.w);
    return out;
}

fn sample_texture_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    if fallback_texture_row_is_valid(row) {
        var fallback: TextureSample;
        fallback.valid = true;
        fallback.value = vec4<f32>(1.0);
        return fallback;
    }
    return sample_vt_row(row, uv);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if (ray_index >= arrayLength(&evaluations)) { return; }
    var result = invalid_evaluation();

    let hit_base = ray_index * HIT_WORDS;
    if (ray_index >= arrayLength(&trace_hits) / HIT_WORDS) {
        evaluations[ray_index] = result;
        return;
    }
    if (trace_hits[hit_base] == 0u) {
        evaluations[ray_index] = result;
        return;
    }
    let triangle_index = trace_hits[hit_base + 5u];
    if (triangle_index >= arrayLength(&hit_attributes) / HIT_ATTRIBUTE_WORDS) {
        evaluations[ray_index] = result;
        return;
    }
    let attribute_base = triangle_index * HIT_ATTRIBUTE_WORDS;
    if (trace_hits[hit_base] != 1u
        || triangle_index == INVALID
        || !hit_matches_attributes(hit_base, attribute_base)) {
        evaluations[ray_index] = result;
        return;
    }

    if (evaluator_config.abi_version != MATERIAL_EVALUATOR_ABI_VERSION
        || evaluator_config.execution_table_version == 0u
        || evaluator_config.material_header_count == 0u
        || evaluator_config.parameter_word_count == 0u
        || evaluator_config.texture_row_count == 0u
        || evaluator_config.execution_plan_count == 0u
        || evaluator_config.execution_program_count == 0u
        || evaluator_config.material_header_count > arrayLength(&material_headers)
        || evaluator_config.parameter_word_count > arrayLength(&material_parameters)
        || evaluator_config.texture_row_count > arrayLength(&material_texture_rows)
        || evaluator_config.execution_plan_count > arrayLength(&execution_plans)
        || evaluator_config.execution_program_count > arrayLength(&execution_programs)
        || evaluator_config.texture_slot_count != 0u
        || (evaluator_config.vt_page_table_word_count != 0u)
            != (evaluator_config.vt_generation_low != 0u
                || evaluator_config.vt_generation_high != 0u)
        || evaluator_config.vt_page_table_word_count > arrayLength(&material_vt_page_table)
        || evaluator_config.flags != 0u
        || any(evaluator_config.padding != vec2<u32>(0u))) {
        evaluations[ray_index] = result;
        return;
    }

    let material_index = hit_attributes[attribute_base + 2u];
    let material_generation = hit_attributes[attribute_base + 3u];
    let material_revision = hit_attributes[attribute_base + 4u];
    let material_epoch_low = hit_attributes[attribute_base + 5u];
    let material_epoch_high = hit_attributes[attribute_base + 6u];
    let plan_index = hit_attributes[attribute_base + 7u];
    let program_index = hit_attributes[attribute_base + 8u];
    let package_index = hit_attributes[attribute_base + 9u];
    let permutation_low = hit_attributes[attribute_base + 10u];
    let permutation_high = hit_attributes[attribute_base + 11u];
    if (hit_attributes[attribute_base] == INVALID
        || hit_attributes[attribute_base + 1u] == INVALID
        || material_index == INVALID
        || material_generation == INVALID
        || material_revision == INVALID
        || (material_epoch_low == INVALID && material_epoch_high == INVALID)
        || plan_index == INVALID
        || program_index == INVALID
        || package_index == INVALID
        || permutation_low == INVALID
        || permutation_high == INVALID
        || material_index >= evaluator_config.material_header_count
        || material_index >= arrayLength(&material_headers)
        || plan_index >= evaluator_config.execution_plan_count
        || plan_index >= arrayLength(&execution_plans)
        || program_index >= evaluator_config.execution_program_count
        || program_index >= arrayLength(&execution_programs)) {
        evaluations[ray_index] = result;
        return;
    }

    let header = material_headers[material_index];
    if (header.is_active == 0u
        || header.generation != material_generation
        || header.revision != material_revision
        || header.material_epoch_low != material_epoch_low
        || header.material_epoch_high != material_epoch_high
        || material_epoch_low != evaluator_config.material_epoch_low
        || material_epoch_high != evaluator_config.material_epoch_high
        || header.execution_plan_index != plan_index
        || header.execution_program_index != program_index
        || header.execution_package_index != package_index
        || header.execution_permutation_low != permutation_low
        || header.execution_permutation_high != permutation_high
        || (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u
        || header.parameter_size == 0u
        || (header.parameter_size & 3u) != 0u
        || header.texture_offset > evaluator_config.texture_row_count
        || header.texture_count > evaluator_config.texture_row_count - header.texture_offset
        || header.texture_offset + header.texture_count > arrayLength(&material_texture_rows)) {
        evaluations[ray_index] = result;
        return;
    }

    let plan = execution_plans[plan_index];
    let program = execution_programs[program_index];
    if (plan.left_program != program_index
        || plan.right_program != INVALID
        || plan.execution_family == 2u
        || plan.blended_channels != 0u
        || plan.mask_kind != 0u
        || plan.mask_index != 0u
        || plan.blend_domain != 0u
        || plan.reserved != 0u
        || program.response_kind != RESPONSE_PROGRAM_KIND_UNLIT
        || program.program_id != BUILTIN_UNLIT_PROGRAM_ID
        || program.package_index != package_index
        || permutation_high > program.permutation_count_high
        || (permutation_high == program.permutation_count_high
            && permutation_low >= program.permutation_count_low)) {
        evaluations[ray_index] = result;
        return;
    }

    let parameter_words = header.parameter_size / 4u;
    if (parameter_words != prism_packed_word_count(header.lobe_mask)
        || parameter_words > PRISM_SURFACE_MAX_WORDS
        || header.parameter_offset > evaluator_config.parameter_word_count
        || parameter_words > evaluator_config.parameter_word_count - header.parameter_offset
        || evaluator_config.parameter_word_count > arrayLength(&material_parameters)) {
        evaluations[ray_index] = result;
        return;
    }
    var packed_words: array<u32, PRISM_SURFACE_MAX_WORDS>;
    for (var word = 0u; word < parameter_words; word += 1u) {
        packed_words[word] = material_parameters[header.parameter_offset + word];
    }
    let surface = prism_unpack_surface(packed_words, header.lobe_mask);
    let uv = vec2<f32>(
        bitcast<f32>(trace_hits[hit_base + 18u]),
        bitcast<f32>(trace_hits[hit_base + 19u]),
    );
    if !finite(uv.x) || !finite(uv.y) {
        evaluations[ray_index] = result;
        return;
    }
    var base_color = surface.base_color.xyz;
    var emission_color = surface.emission_color.xyz;
    for (var row_index = 0u; row_index < header.texture_count; row_index += 1u) {
        let row = material_texture_rows[header.texture_offset + row_index];
        let sampled = sample_texture_row(row, uv);
        if !sampled.valid {
            // Direct authored images and stale texture generations are not
            // silently replaced by a fallback in this portable evaluator.
            evaluations[ray_index] = result;
            return;
        }
        if row.semantic == TEXTURE_SEMANTIC_BASE_COLOR {
            base_color = base_color * srgb_to_linear(sampled.value.xyz);
        } else if row.semantic == TEXTURE_SEMANTIC_EMISSIVE {
            emission_color = emission_color * srgb_to_linear(sampled.value.xyz);
        } else if !fallback_texture_row_is_valid(row) {
            // Normal/roughness/registered texture semantics are not part of
            // this builtin-Unlit response. Rejecting them keeps the result
            // conservative until the matching response evaluator is bound.
            evaluations[ray_index] = result;
            return;
        }
    }
    let response = base_color + emission_color * surface.emission_luminance;
    if (!finite(response.x) || !finite(response.y) || !finite(response.z)
        || any(response < vec3<f32>(0.0))) {
        evaluations[ray_index] = result;
        return;
    }

    result.response_rgb = response;
    result.valid = 1u;
    result.instance_id = hit_attributes[attribute_base];
    result.geometry_id = hit_attributes[attribute_base + 1u];
    result.material_index = material_index;
    result.material_generation = material_generation;
    result.material_revision = material_revision;
    result.material_epoch_low = material_epoch_low;
    result.material_epoch_high = material_epoch_high;
    result.execution_plan_index = plan_index;
    result.execution_program_index = program_index;
    result.execution_package_index = package_index;
    result.execution_permutation_low = permutation_low;
    result.execution_permutation_high = permutation_high;
    evaluations[ray_index] = result;
}
