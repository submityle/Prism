// Virtual-shadow-map receiver-driven page-request marking.
//
// GPU twin of the CPU golden prism_render_shading::shadow::virtual_sm
// page-request pass (clipmap level selection + request footprint expansion).
// Before any shadow depth is rasterised the virtual shadow map has to learn
// *which* pages the visible surfaces will sample this frame; that set is driven
// by the shadow *receivers*. Each receiver, projected into the light's clipmap
// plane, selects a clip level from its view distance, lands on one absolute
// world page, and the soft-shadow (PCF/PCSS) filter kernel then widens that
// footprint by whole page rings, because a sample near a page edge reads texels
// from the neighbouring pages too.
//
// This compute pass marks exactly those pages in a per-level, camera-snapped
// resident-window request bitmap -- the GPU counterpart of the CPU
// generate_page_requests. De-duplication is by construction: several receivers
// whose footprints overlap set the same window slot, so the marked set is the
// unique page-request set the physical-page allocator must make resident.
//
// Clipmap addressing matches the golden line for line:
//   * absolute page identity  floor(world / page_world_size) -- camera
//     independent, so a static receiver keeps the same page as the camera pans;
//   * camera-snapped resident window  center_page - edge/2 (whole-page snap),
//     exactly ClipmapConfig::build_level, so the addressable request set is the
//     level's resident window and requests outside it are dropped (the same
//     bounded working set the golden allocator budgets).
//
// Self-contained (no imports); group 0 is
//   0 = receivers      (this frame's shadow receivers, storage read)
//   1 = page_requests  (per-window-slot request flags, atomic storage r/w)
// with a VsmPageMarkParams immediate block.

// One shadow receiver, already projected into the light's clipmap plane; the
// byte-for-byte twin of the CPU golden Receiver.
struct VsmReceiver {
    // Receiver position in the light-space clipmap plane (world units).
    light_space_x: f32,
    light_space_y: f32,
    // Positive view-space distance; selects the clip level (finer when nearer).
    view_distance: f32,
    // Soft-shadow filter kernel half-width in shadow texels; widens the page
    // footprint so edge samples still find resident neighbours.
    filter_radius_texels: f32,
}

// Clipmap layout + this frame's driving state, uploaded as the immediate block.
// The clipmap fields mirror ClipmapConfig; camera_* drives the same whole-page
// window snapping as ClipmapConfig::build_level.
struct VsmPageMarkParams {
    // --- clipmap layout (mirrors ClipmapConfig) ---
    levels: u32,
    page_size: u32,
    pages_per_level_edge: u32,
    page_coord_bias: i32,
    level0_texel_world_size: f32,
    level0_max_distance: f32,
    // --- per-frame driving state ---
    light: u32,
    receiver_count: u32,
    camera_x: f32,
    camera_y: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> receivers: array<VsmReceiver>;
// One flag per resident-window page slot across every level:
//   slot = level * edge*edge + local_y * edge + local_x
// 1 means "requested this frame". Set with atomicOr so overlapping receiver
// footprints collapse to a single request (the golden de-duplication).
@group(0) @binding(1) var<storage, read_write> page_requests: array<atomic<u32>>;

var<immediate> params: VsmPageMarkParams;

// Number of clipmap levels, always >= 1 (mirrors level_count).
fn vsm_level_count() -> u32 {
    return max(params.levels, 1u);
}

// Pages along one edge of a level's resident window, always >= 1.
fn vsm_window_edge() -> i32 {
    return i32(max(params.pages_per_level_edge, 1u));
}

// World-space edge length of one texel at level (doubles each level), the twin
// of ClipmapConfig::texel_world_size.
fn vsm_texel_world_size(level: u32) -> f32 {
    let clamped = min(level, vsm_level_count() - 1u);
    return max(params.level0_texel_world_size, 1.0e-6) * exp2(f32(clamped));
}

// World-space edge length of one page at level (texel * page_size).
fn vsm_page_world_size(level: u32) -> f32 {
    return vsm_texel_world_size(level) * f32(max(params.page_size, 1u));
}

// Finest clip level whose texels are still dense enough for a receiver at
// view_distance: level 0 up to level0_max_distance, then one coarser level per
// doubling of distance, clamped to the stack. Twin of select_level.
fn vsm_select_level(view_distance: f32) -> u32 {
    let distance = max(view_distance, 0.0);
    let reference = max(params.level0_max_distance, 1.0e-6);
    let ratio = distance / reference;
    var level_i: i32;
    if (ratio <= 1.0) {
        level_i = 0;
    } else {
        level_i = i32(ceil(log2(ratio)));
    }
    let max_level = i32(vsm_level_count()) - 1;
    return u32(clamp(level_i, 0, max_level));
}

// Absolute world page coordinate of world_xy at level
// (floor(world / page_world_size)), camera independent. Twin of
// world_page_coords.
fn vsm_world_page_coords(level: u32, world_xy: vec2<f32>) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    return vec2<i32>(
        i32(floor(world_xy.x / pws)),
        i32(floor(world_xy.y / pws)),
    );
}

// Bottom-left page of level's resident window, snapped to whole pages around
// the camera. Twin of ClipmapConfig::build_level's origin_page:
// floor(camera / page_world_size) - edge/2.
fn vsm_window_origin(level: u32) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    let center = vec2<i32>(
        i32(floor(params.camera_x / pws)),
        i32(floor(params.camera_y / pws)),
    );
    let half = vsm_window_edge() / 2;
    return center - vec2<i32>(half, half);
}

// Extra page rings a filter kernel of radius_texels adds around a receiver's
// page: ceil(radius / page_size) (0 for a point sample). Twin of
// filter_page_radius; never under-requests.
fn vsm_filter_page_radius(radius_texels: f32) -> i32 {
    let radius = max(radius_texels, 0.0);
    if (radius <= 0.0) {
        return 0;
    }
    let page = f32(max(params.page_size, 1u));
    return i32(ceil(radius / page));
}

// Flat slot index of a window-local page at level, or -1 when the page lies
// outside the level's resident window (an unaddressable request, dropped just
// as the golden clipmap window bounds the working set).
fn vsm_window_slot(level: u32, world_page: vec2<i32>) -> i32 {
    let edge = vsm_window_edge();
    let local = world_page - vsm_window_origin(level);
    if (local.x < 0 || local.y < 0 || local.x >= edge || local.y >= edge) {
        return -1;
    }
    return i32(level) * edge * edge + local.y * edge + local.x;
}

@compute @workgroup_size(64, 1, 1)
fn vsm_mark_pages(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.receiver_count) {
        return;
    }

    let receiver = receivers[index];
    let world_xy = vec2<f32>(receiver.light_space_x, receiver.light_space_y);

    let level = vsm_select_level(receiver.view_distance);
    let center = vsm_world_page_coords(level, world_xy);
    let rings = vsm_filter_page_radius(receiver.filter_radius_texels);

    // Expand the receiver's page into a square footprint sized by its filter
    // radius, marking each addressable cell exactly as generate_page_requests
    // enumerates its footprint. atomicOr de-duplicates overlapping footprints.
    for (var dy = -rings; dy <= rings; dy = dy + 1) {
        for (var dx = -rings; dx <= rings; dx = dx + 1) {
            let page = center + vec2<i32>(dx, dy);
            let slot = vsm_window_slot(level, page);
            if (slot >= 0) {
                atomicOr(&page_requests[slot], 1u);
            }
        }
    }
}
