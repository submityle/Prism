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
    flags: u32,
}

struct WorkItem {
    pixel_index: u32,
    material_index: u32,
    shading_class: u32,
    flags: u32,
}

struct ResponseBucket {
    execution_family: u32,
    program_id: u32,
    render_phase: u32,
    legacy_class: u32,
}

struct ExecutionPlanRow {
    left_program: u32,
    right_program: u32,
}

struct ExecutionProgramRow {
    response_kind: u32,
    package_index: u32,
    permutation_count_low: u32,
    permutation_count_high: u32,
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

// Closure-kind bit positions (see `ir::ClosureKind`).
const CLOSURE_CLEARCOAT: u32 = 3u;
const CLOSURE_SHEEN: u32 = 4u;
const CLOSURE_SUBSURFACE: u32 = 5u;
const CLOSURE_HAIR: u32 = 8u;

// MaterialShadingClass discriminants (see `classification::MaterialShadingClass`).
const CLASS_PRINCIPLED: u32 = 0u;
const CLASS_UNLIT: u32 = 1u;
const CLASS_SUBSURFACE: u32 = 2u;
const CLASS_CLEARCOAT: u32 = 3u;
const CLASS_CLOTH: u32 = 4u;
const CLASS_HAIR: u32 = 5u;
const CLASS_WATER: u32 = 6u;
const CLASS_NPR: u32 = 7u;
const CLASS_CUSTOM: u32 = 8u;

fn has_closure(mask: u32, bit: u32) -> bool {
    return (mask & (1u << bit)) != 0u;
}

// Project a Lit surface's closure graph onto a coherent bucket, matching the
// CPU `lit_class_from_closures` derivation exactly.
fn lit_class_from_closures(mask: u32) -> u32 {
    if has_closure(mask, CLOSURE_HAIR) { return CLASS_HAIR; }
    if has_closure(mask, CLOSURE_SUBSURFACE) { return CLASS_SUBSURFACE; }
    if has_closure(mask, CLOSURE_CLEARCOAT) { return CLASS_CLEARCOAT; }
    if has_closure(mask, CLOSURE_SHEEN) { return CLASS_CLOTH; }
    return CLASS_PRINCIPLED;
}

fn material_class(header: MaterialHeader, region_illumination: u32) -> u32 {
    // Only the seven blend/raster classes are surface-compatible. Runtime
    // domains are carried by render_domain; water stays Surface and is marked
    // by FEATURE_WATER.
    let surface = (header.render_class == 0u || header.render_class == 1u
        || header.render_class == 2u || header.render_class == 3u)
        && (header.render_domain == 0u || header.render_domain == 3u);
    if header.is_active == 0u || !surface {
        return 0xffffffffu;
    }
    // Dedicated domain/feature paths bind their bucket directly.
    if header.render_domain == 3u { return CLASS_HAIR; }
    if (header.feature_flags & (1u << 8u)) != 0u { return CLASS_WATER; }
    let sentinel = 0xffffffffu;
    let words_all_sentinel = header.execution_plan_index == sentinel
        && header.execution_program_index == sentinel
        && header.execution_package_index == sentinel
        && header.execution_permutation_low == sentinel
        && header.execution_permutation_high == sentinel;
    let words_partial = header.execution_plan_index == sentinel
        || header.execution_program_index == sentinel
        || header.execution_package_index == sentinel
        || header.execution_permutation_low == sentinel
        || header.execution_permutation_high == sentinel;
    if words_partial && !words_all_sentinel { return sentinel; }
    var response_kind = 0u;
    if !words_all_sentinel {
        if header.execution_plan_index >= arrayLength(&execution_plans)
            || header.execution_program_index >= arrayLength(&execution_programs)
            || header.execution_package_index >= arrayLength(&execution_packages) {
            return sentinel;
        }
        let plan = execution_plans[header.execution_plan_index];
        if plan.left_program != header.execution_program_index
            && plan.right_program != header.execution_program_index {
            return sentinel;
        }
        let program = execution_programs[header.execution_program_index];
        if program.package_index != header.execution_package_index {
            return sentinel;
        }
        let perm_after_limit = header.execution_permutation_high > program.permutation_count_high
            || (header.execution_permutation_high == program.permutation_count_high
                && header.execution_permutation_low >= program.permutation_count_low);
        if perm_after_limit { return sentinel; }
        response_kind = program.response_kind;
        if response_kind > 3u { return sentinel; }
    }
    // A region mask only blends the Lit/NPR endpoints. Unlit and custom
    // response programs remain material-level choices.
    if (response_kind == 0u || response_kind == 1u) && region_illumination == 1u {
        return CLASS_NPR;
    }
    if response_kind == 1u { return CLASS_NPR; }
    if response_kind == 2u { return CLASS_UNLIT; }
    if response_kind == 3u { return CLASS_CUSTOM; }
    return lit_class_from_closures(header.closure_mask);
}

fn response_bucket_for_legacy_class(legacy_class: u32) -> u32 {
    for (var bucket = 0u; bucket < arrayLength(&response_buckets); bucket = bucket + 1u) {
        if response_buckets[bucket].legacy_class == legacy_class {
            return bucket;
        }
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
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, 0xffffffffu, 1u);
        atomicAdd(&diagnostics[0], 1u);
        return;
    }
    let material_index = metadata.x;
    if material_index >= params.material_capacity || material_index >= arrayLength(&materials) {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 2u);
        atomicAdd(&diagnostics[1], 1u);
        return;
    }
    let header = materials[material_index];
    if header.is_active == 0u || header.generation != metadata.y {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 4u);
        atomicAdd(&diagnostics[1], 1u);
        return;
    }
    // `metadata.z` is the visibility raster's Lit/Stylized selector
    // (`0`/`1`), or `0xffffffff` when the material has no region mask.
    let legacy_class = material_class(header, metadata.z);
    let shading_class = response_bucket_for_legacy_class(legacy_class);
    if shading_class >= params.class_count {
        pixels[pixel] = PixelClass(0xffffffffu, 0xffffffffu, material_index, 8u);
        atomicAdd(&diagnostics[2], 1u);
        return;
    }
    pixels[pixel] = PixelClass(shading_class, 0xffffffffu, material_index, header.feature_flags);
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
    work[destination] = WorkItem(pixel, record.material_index, record.shading_class, record.flags);
    pixels[pixel].work_index = destination;
}
