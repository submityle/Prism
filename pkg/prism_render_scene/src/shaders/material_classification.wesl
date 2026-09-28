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
    generation: u32, revision: u32, illumination: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
    closure_graph_offset: u32, specialization_low: u32, specialization_high: u32, reserved: u32,
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

@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
@group(1) @binding(0) var<storage, read> materials: array<MaterialHeader>;
@group(2) @binding(0) var<storage, read_write> pixels: array<PixelClass>;
@group(2) @binding(1) var<storage, read_write> work: array<WorkItem>;
@group(2) @binding(2) var<storage, read_write> class_counts: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> class_offsets: array<u32>;
@group(2) @binding(4) var<storage, read_write> class_cursors: array<atomic<u32>>;
@group(2) @binding(5) var<storage, read_write> dispatch_args: array<u32>;
@group(2) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;

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

fn material_class(header: MaterialHeader) -> u32 {
    // Surface-compatible MaterialRenderClass values: Opaque, OpaqueTwoSided,
    // Masked, MaskedTwoSided, Hair, Water. Transparent/additive/transmissive/
    // volume/decal materials stay on forward/specialized paths. The former
    // Npr*/Custom* render classes are gone; style is the illumination axis now.
    let surface = header.render_class == 0u || header.render_class == 1u
        || header.render_class == 2u || header.render_class == 3u
        || header.render_class == 8u || header.render_class == 9u;
    if header.is_active == 0u || !surface {
        return 0xffffffffu;
    }
    // Dedicated subsystem render classes bind their bucket directly.
    if header.render_class == 9u { return CLASS_WATER; }
    if header.render_class == 8u { return CLASS_HAIR; }
    // Illumination axis (Lit=0, Stylized=1, Unlit=2, Custom=3).
    switch header.illumination {
        case 2u: { return CLASS_UNLIT; }
        case 1u: { return CLASS_NPR; }
        case 3u: { return CLASS_CUSTOM; }
        case 0u: { return lit_class_from_closures(header.closure_mask); }
        default: { return 0xffffffffu; }
    }
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
    let shading_class = material_class(header);
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
