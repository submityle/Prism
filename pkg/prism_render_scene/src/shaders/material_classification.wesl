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
    // Binary region endpoints only change builtin Lit/Npr; registered ids,
    // including zero, and Unlit retain their verified implementation.
    if identity.response_kind <= 1u && region_selector <= 1u {
        identity.program_id = region_selector;
        identity.response_kind = region_selector;
    }
    return identity;
}

fn material_response_kind(header: MaterialHeader, identity: ExecutionIdentity) -> u32 {
    let surface = header.render_class <= 3u
        && (header.render_domain == 0u || header.render_domain == 3u);
    if header.is_active == 0u || !surface { return 0xffffffffu; }
    return identity.response_kind;
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
    let response_kind = material_response_kind(header, identity);
    let shading_class = response_bucket_for_execution(header, identity);
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
    work[destination] = WorkItem(
        pixel,
        record.material_index,
        record.shading_class,
        record.response_kind,
        record.flags,
    );
    pixels[pixel].work_index = destination;
}
