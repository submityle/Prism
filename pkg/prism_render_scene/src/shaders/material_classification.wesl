//! Three entry points share one ABI.  The CPU schedules them in order:
//! classify/count, prefix, then scatter.  No readback is required.

struct ClassificationParams {
    width: u32,
    height: u32,
    pixel_count: u32,
    material_capacity: u32,
    work_capacity: u32,
    class_count: u32,
}

struct MaterialHeader {
    generation: u32, revision: u32, render_class: u32, render_domain: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
    lobe_mask: u32,
    // PCM extension digest (§5.3): used/required, each split low/high.
    extensions_used_low: u32, extensions_used_high: u32,
    extensions_required_low: u32, extensions_required_high: u32,
    // Runtime dynamic channels (§17.1): wetness / snow / damage / dissolve,
    // each an `f32` bitcast into the word (`0.0` = exact identity).
    dynamic_wetness: u32, dynamic_snow: u32,
    dynamic_damage: u32, dynamic_dissolve: u32,
    // ABI v8 execution binding; all five words use 0xffffffff when unbound.
    execution_plan_index: u32, execution_program_index: u32,
    execution_package_index: u32, execution_permutation_low: u32,
    execution_permutation_high: u32,
}

struct PixelClass {
    shading_class: u32,
    work_index: u32,
    material_index: u32,
    response_kind: u32,
    flags: u32,
}

struct WorkItem {
    pixel_index: u32,
    material_index: u32,
    shading_class: u32,
    response_kind: u32,
    flags: u32,
    right_shading_class: u32,
    right_response_kind: u32,
    response_meta: u32,
}

struct ResponseBucket {
    execution_family: u32,
    program_id: u32,
    render_phase: u32,
    reserved: u32,
}

struct ExecutionPlanRow {
    left_program: u32,
    right_program: u32,
    execution_family: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    reserved: u32,
}

struct ExecutionProgramRow {
    response_kind: u32,
    package_index: u32,
    permutation_count_low: u32,
    permutation_count_high: u32,
    program_id: u32,
}

struct BlendWorkData {
    left_shading_class: u32,
    left_response_kind: u32,
    right_shading_class: u32,
    right_response_kind: u32,
    response_meta: u32,
    valid: u32,
}

@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
@group(1) @binding(0) var<storage, read> materials: array<MaterialHeader>;
@group(1) @binding(10) var<storage, read> execution_plans: array<ExecutionPlanRow>;
@group(1) @binding(11) var<storage, read> execution_programs: array<ExecutionProgramRow>;
@group(1) @binding(12) var<storage, read> execution_packages: array<u32>;
@group(2) @binding(0) var<storage, read_write> pixels: array<PixelClass>;
@group(2) @binding(1) var<storage, read_write> work: array<WorkItem>;
@group(2) @binding(2) var<storage, read_write> class_counts: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> class_offsets: array<u32>;
@group(2) @binding(4) var<storage, read_write> class_cursors: array<atomic<u32>>;
@group(2) @binding(5) var<storage, read_write> dispatch_args: array<u32>;
@group(2) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(2) @binding(7) var<storage, read> response_buckets: array<ResponseBucket>;

var<immediate> params: ClassificationParams;

const FEATURE_ILLUMINATION_BLEND: u32 = 1u << 9u;
const FEATURE_REGION_ILLUMINATION: u32 = 1u << 7u;
const INVALID: u32 = 0xffffffffu;

// BEGIN SCHEMA-GENERATED MATERIAL METADATA
// This block is generated from schema/surface.toml.
const PRISM_SCHEMA_SPECIALIZATION_ILLUMINATION_SHIFT: u32 = 0u;
const PRISM_SCHEMA_SPECIALIZATION_ILLUMINATION_BITS: u32 = 8u;
const PRISM_SCHEMA_SPECIALIZATION_LOBE_MASK_SHIFT: u32 = 8u;
const PRISM_SCHEMA_SPECIALIZATION_LOBE_MASK_BITS: u32 = 32u;
const PRISM_SCHEMA_SPECIALIZATION_RENDER_CLASS_SHIFT: u32 = 40u;
const PRISM_SCHEMA_SPECIALIZATION_RENDER_CLASS_BITS: u32 = 8u;
const PRISM_SCHEMA_SPECIALIZATION_RESERVED_SHIFT: u32 = 48u;
const PRISM_SCHEMA_SPECIALIZATION_RESERVED_BITS: u32 = 16u;
// END SCHEMA-GENERATED MATERIAL METADATA

// Program kinds are the generated execution-table ABI, not lobe buckets.
struct ExecutionIdentity {
    family: u32,
    program_id: u32,
    response_kind: u32,
}

fn execution_identity(header: MaterialHeader, region_selector: u32) -> ExecutionIdentity {
    let invalid = ExecutionIdentity(0xffffffffu, 0xffffffffu, 0xffffffffu);
    let sentinel = 0xffffffffu;
    let all_unbound = header.execution_plan_index == sentinel
        && header.execution_program_index == sentinel
        && header.execution_package_index == sentinel
        && header.execution_permutation_low == sentinel
        && header.execution_permutation_high == sentinel;
    let any_unbound = header.execution_plan_index == sentinel
        || header.execution_program_index == sentinel
        || header.execution_package_index == sentinel
        || header.execution_permutation_low == sentinel
        || header.execution_permutation_high == sentinel;
    if any_unbound && !all_unbound { return invalid; }
    var identity = ExecutionIdentity(0u, 0u, 0u);
    if !all_unbound {
        if header.execution_plan_index >= arrayLength(&execution_plans)
            || header.execution_program_index >= arrayLength(&execution_programs)
            || header.execution_package_index >= arrayLength(&execution_packages) {
            return invalid;
        }
        let plan = execution_plans[header.execution_plan_index];
        if plan.left_program != header.execution_program_index
            && plan.right_program != header.execution_program_index {
            return invalid;
        }
        let program = execution_programs[header.execution_program_index];
        if program.package_index != header.execution_package_index
            || program.response_kind > 3u {
            return invalid;
        }
        let after_limit = header.execution_permutation_high > program.permutation_count_high
            || (header.execution_permutation_high == program.permutation_count_high
                && header.execution_permutation_low >= program.permutation_count_low);
        if after_limit { return invalid; }
        identity = ExecutionIdentity(plan.execution_family, program.program_id, program.response_kind);
    }
    // Visibility metadata is a trusted Lit/NPR selector only when the material
    // explicitly declares the binary region-mask feature. Sentinel, malformed,
    // and unrelated metadata preserve the material-wide execution identity.
    if (header.feature_flags & FEATURE_REGION_ILLUMINATION) == 0u {
        region_selector = INVALID;
    }
    // Binary region endpoints only change builtin Lit/Npr; registered ids,
    // including zero, and Unlit retain their verified implementation.
    if identity.response_kind <= 1u && region_selector <= 1u {
        identity.program_id = region_selector;
        identity.response_kind = region_selector;
    }
    identity.family = execution_family_for_program(identity.family, identity.response_kind);
    return identity;
}

fn material_response_kind(header: MaterialHeader, identity: ExecutionIdentity) -> u32 {
    let surface = header.render_class <= 3u
        && (header.render_domain == 0u || header.render_domain == 3u);
    if header.is_active == 0u || !surface { return 0xffffffffu; }
    return identity.response_kind;
}

fn execution_family_for_program(family: u32, response_kind: u32) -> u32 {
    // Composition and split are explicit table families. A fused request can
    // remain fused only for builtin Lit; NPR, Unlit, and custom programs use
    // the split response contract, matching the CPU resolver.
    if family == 2u || family == 1u {
        return family;
    }
    if response_kind == 0u {
        return 0u;
    }
    return 1u;
}

fn response_bucket_for_execution(header: MaterialHeader, identity: ExecutionIdentity) -> u32 {
    if identity.response_kind > 3u { return 0xffffffffu; }
    var phase = 0u;
    if header.render_class >= 4u && header.render_class <= 6u {
        phase = 1u;
    } else if identity.response_kind == 1u {
        phase = 2u;
    } else if identity.response_kind == 3u {
        phase = 3u;
    }
    // The upload is sorted lexicographically. Search is logarithmic even for
    // project registries with thousands of compiled response entries.
    var low = 0u;
    var high = arrayLength(&response_buckets);
    while low < high {
        let mid = low + (high - low) / 2u;
        let row = response_buckets[mid];
        let less = row.execution_family < identity.family
            || (row.execution_family == identity.family && row.program_id < identity.program_id)
            || (row.execution_family == identity.family && row.program_id == identity.program_id
                && row.render_phase < phase);
        if less { low = mid + 1u; } else { high = mid; }
    }
    if low < arrayLength(&response_buckets) {
        let row = response_buckets[low];
        if row.execution_family == identity.family && row.program_id == identity.program_id
            && row.render_phase == phase { return low; }
    }
    return 0xffffffffu;
}

// Continuous MaskBlend keeps the compact 20-byte PixelClass ABI.  Its right
// arm and packed mask are derived again during scatter from the immutable
// execution tables and the visibility metadata.  This helper is shared by
// classify/count (validation + left bucket) and scatter (right bucket + meta),
// so the two stages cannot silently disagree about the plan shape.
fn blend_work_data(header: MaterialHeader, mask_word: u32) -> BlendWorkData {
    let invalid = BlendWorkData(INVALID, INVALID, INVALID, INVALID, INVALID, 0u);
    if (mask_word > 0xffffu
        || header.execution_plan_index == INVALID
        || header.execution_program_index == INVALID
        || header.execution_package_index == INVALID
        || header.execution_permutation_low == INVALID
        || header.execution_permutation_high == INVALID
        || header.execution_plan_index >= arrayLength(&execution_plans)) {
        return invalid;
    }

    // Validate the material's bound tuple before using the plan endpoints.
    let bound = execution_identity(header, INVALID);
    if bound.response_kind == INVALID {
        return invalid;
    }
    let plan = execution_plans[header.execution_plan_index];
    if plan.right_program == INVALID
        || plan.left_program >= arrayLength(&execution_programs)
        || plan.right_program >= arrayLength(&execution_programs)
        || plan.execution_family > 2u
        || plan.blended_channels > 0xffffu {
        return invalid;
    }
    let left = execution_programs[plan.left_program];
    let right = execution_programs[plan.right_program];
    if left.response_kind > 3u || right.response_kind > 3u {
        return invalid;
    }

    let left_identity = ExecutionIdentity(
        plan.execution_family,
        left.program_id,
        left.response_kind,
    );
    let right_identity = ExecutionIdentity(
        plan.execution_family,
        right.program_id,
        right.response_kind,
    );
    let left_class = response_bucket_for_execution(header, left_identity);
    let right_class = response_bucket_for_execution(header, right_identity);
    if left_class == INVALID || right_class == INVALID {
        return invalid;
    }
    return BlendWorkData(
        left_class,
        left.response_kind,
        right_class,
        right.response_kind,
        (mask_word & 0xffffu) | (plan.blended_channels << 16u),
        1u,
    );
}

@compute @workgroup_size(64)
fn classify_count(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x;
    if pixel >= params.pixel_count || pixel >= arrayLength(&pixels) {
        return;
    }
    let coordinate = vec2<i32>(i32(pixel % params.width), i32(pixel / params.width));
    let ids = textureLoad(visibility_ids, coordinate, 0);
    let metadata = textureLoad(visibility_metadata, coordinate, 0);
    // Slot zero counts uncovered/background pixels. It is diagnostic data,
    // not a malformed-visibility error.
    if ids.x == 0xffffffffu || metadata.x == 0xffffffffu {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 1u);
        atomicAdd(&diagnostics[0], 1u);
        return;
    }
    let material_index = metadata.x;
    if material_index >= params.material_capacity || material_index >= arrayLength(&materials) {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 0xffffffffu, 2u);
        atomicAdd(&diagnostics[1], 1u);
        return;
    }
    let header = materials[material_index];
    if header.is_active == 0u || header.generation != metadata.y {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 0xffffffffu, 4u);
        atomicAdd(&diagnostics[1], 1u);
        return;
    }
    // `metadata.z` is the visibility raster's Lit/Stylized selector
    // (`0`/`1`), or `0xffffffff` when the material has no region mask.
    let identity = execution_identity(header, metadata.z);
    var response_kind = material_response_kind(header, identity);
    var shading_class = response_bucket_for_execution(header, identity);
    if (header.feature_flags & FEATURE_ILLUMINATION_BLEND) != 0u {
        let blend = blend_work_data(header, metadata.z);
        if blend.valid == 0u {
            pixels[pixel] = PixelClass(INVALID, INVALID, material_index, INVALID, 8u);
            atomicAdd(&diagnostics[2], 1u);
            return;
        }
        response_kind = blend.left_response_kind;
        shading_class = blend.left_shading_class;
    }
    if shading_class >= params.class_count {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 0xffffffffu, 8u);
        atomicAdd(&diagnostics[2], 1u);
        return;
    }
    if response_kind == 0xffffffffu || shading_class == 0xffffffffu {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 0xffffffffu, 8u);
        atomicAdd(&diagnostics[2], 1u);
        return;
    }
    // The program kind selects a compiled builtin or registered entry. Lobe
    // combinations never change the execution identity or bucket cardinality.
    pixels[pixel] = PixelClass(
        shading_class,
        0xffffffffu,
        material_index,
        response_kind,
        header.feature_flags,
    );
    atomicAdd(&class_counts[shading_class], 1u);
}

@compute @workgroup_size(1)
fn prefix_classes(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u { return; }
    var cursor = 0u;
    for (var shading_class = 0u; shading_class < params.class_count; shading_class += 1u) {
        let count = atomicLoad(&class_counts[shading_class]);
        class_offsets[shading_class] = cursor;
        class_cursors[shading_class] = 0u;
        cursor += count;
        dispatch_args[shading_class * 3u] = (count + 63u) / 64u;
        dispatch_args[shading_class * 3u + 1u] = 1u;
        dispatch_args[shading_class * 3u + 2u] = 1u;
    }
}

@compute @workgroup_size(64)
fn scatter_work(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x;
    if pixel >= params.pixel_count || pixel >= arrayLength(&pixels) { return; }
    let record = pixels[pixel];
    if record.shading_class == 0xffffffffu || record.shading_class >= params.class_count { return; }
    let local = atomicAdd(&class_cursors[record.shading_class], 1u);
    let destination = class_offsets[record.shading_class] + local;
    if destination >= params.work_capacity || destination >= arrayLength(&work) {
        atomicAdd(&diagnostics[3], 1u);
        return;
    }
    var right_shading_class = INVALID;
    var right_response_kind = 0u;
    var response_meta = INVALID;
    if (record.flags & FEATURE_ILLUMINATION_BLEND) != 0u {
        if record.material_index >= arrayLength(&materials) {
            atomicAdd(&diagnostics[2], 1u);
            return;
        }
        let coordinate = vec2<i32>(
            i32(pixel % params.width),
            i32(pixel / params.width),
        );
        let metadata = textureLoad(visibility_metadata, coordinate, 0);
        let blend = blend_work_data(materials[record.material_index], metadata.z);
        if blend.valid == 0u {
            atomicAdd(&diagnostics[2], 1u);
            return;
        }
        right_shading_class = blend.right_shading_class;
        right_response_kind = blend.right_response_kind;
        response_meta = blend.response_meta;
    }
    work[destination] = WorkItem(
        pixel,
        record.material_index,
        record.shading_class,
        record.response_kind,
        record.flags,
        right_shading_class,
        right_response_kind,
        response_meta,
    );
    pixels[pixel].work_index = destination;
}
