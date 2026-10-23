// Canonical material-table admission and unpacking for the Native composite.
// Surface identity is currently the material slot. The frame owner pins the
// referenced allocations; producer identity is not yet preserved per pixel.
import prism_render_material::shaders::material::{PrismMaterialHeader, PrismSurfaceParameters};
import prism_render_material::shaders::material_unpack::{PRISM_LOBE_ALL, PRISM_SURFACE_MAX_WORDS, prism_neutral_surface, prism_packed_word_count, prism_unpack_surface};

struct CompositePlanRow {
    left: u32,
    right: u32,
    family: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    reserved: u32,
};

struct CompositeProgramRow {
    kind: u32,
    package_slot: u32,
    permutation_count_low: u32,
    permutation_count_high: u32,
    program_id: u32,
};

struct CompositeMaterialSurface {
    valid: bool,
    program_kind: u32,
    surface: PrismSurfaceParameters,
};

// The current native composite implements one fused builtin response per
// surface. Until the dual-response evaluator and mask SurfaceData are wired,
// admit only the exact Single-plan encoding; never reinterpret one endpoint of
// a MaskBlend/SplitResponse plan as a complete material.
fn composite_single_fused_builtin_plan_is_supported(
    plan: CompositePlanRow,
    material_program_index: u32,
) -> bool {
    return plan.left == material_program_index
        && plan.right == 0xffffffffu
        && plan.family == 0u
        && plan.blended_channels == 0u
        && plan.mask_kind == 0u
        && plan.mask_index == 0u
        && plan.blend_domain == 0u
        && plan.reserved == 0u;
}

@group(2) @binding(0) var<storage, read> composite_material_headers: array<PrismMaterialHeader>;
@group(2) @binding(1) var<storage, read> composite_material_parameters: array<u32>;
@group(2) @binding(2) var<storage, read> composite_execution_plans: array<CompositePlanRow>;
@group(2) @binding(3) var<storage, read> composite_execution_programs: array<CompositeProgramRow>;

fn composite_neutral_surface() -> PrismSurfaceParameters {
    // This value is never consumed when `valid` is false.
    return prism_neutral_surface();
}

fn prism_composite_material_surface(material_slot: u32) -> CompositeMaterialSurface {
    var result: CompositeMaterialSurface;
    result.valid = false;
    result.program_kind = 0xffffffffu;
    result.surface = composite_neutral_surface();

    let header_count = arrayLength(&composite_material_headers);
    if (material_slot >= header_count) { return result; }
    let header = composite_material_headers[material_slot];
    if (header.is_active == 0u ||
        (header.generation == 0u && material_slot != 0u) ||
        header.execution_plan_index == 0xffffffffu ||
        header.execution_program_index == 0xffffffffu ||
        header.execution_package_index == 0xffffffffu ||
        (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u) { return result; }

    let packed_words = prism_packed_word_count(header.lobe_mask);
    if (header.parameter_size != packed_words * 4u) { return result; }
    let parameter_count = arrayLength(&composite_material_parameters);
    if (header.parameter_offset > parameter_count ||
        packed_words > parameter_count - header.parameter_offset) { return result; }

    let plan_count = arrayLength(&composite_execution_plans);
    let program_count = arrayLength(&composite_execution_programs);
    if (header.execution_plan_index >= plan_count ||
        header.execution_program_index >= program_count) { return result; }
    let plan = composite_execution_plans[header.execution_plan_index];
    let program = composite_execution_programs[header.execution_program_index];
    if (!composite_single_fused_builtin_plan_is_supported(plan, header.execution_program_index) ||
        program.package_slot != header.execution_package_index ||
        program.package_slot == 0xffffffffu) { return result; }

    if (program.permutation_count_high == 0u &&
        (program.permutation_count_low == 0u ||
         header.execution_permutation_high != 0u ||
         header.execution_permutation_low >= program.permutation_count_low)) {
        return result;
    }
    if (program.permutation_count_high != 0u &&
        (header.execution_permutation_high > program.permutation_count_high ||
         (header.execution_permutation_high == program.permutation_count_high &&
          header.execution_permutation_low >= program.permutation_count_low))) {
        return result;
    }

    var words: array<u32, PRISM_SURFACE_MAX_WORDS>;
    for (var index = 0u; index < packed_words; index += 1u) {
        words[index] = composite_material_parameters[header.parameter_offset + index];
    }
    result.surface = prism_unpack_surface(words, header.lobe_mask);
    result.program_kind = program.kind;
    result.valid = true;
    return result;
}
