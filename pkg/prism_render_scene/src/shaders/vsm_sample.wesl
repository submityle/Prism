// Virtual-shadow-map resolve-side sampling: virtual page -> physical page -> PCF.
//
// GPU twin of the CPU golden prism_render_shading::shadow::virtual_sm sampling
// path. Once the page-request pass (vsm_page_mark.wesl) has marked the pages a
// frame touches and the allocator has made them resident in the physical page
// atlas, the resolve pass asks this shader a single question per shadow
// receiver: is this world point in shadow? Answering it walks the exact same
// clipmap addressing the golden uses, looks the virtual page up in the
// page-table buffer, and -- only if the page is resident -- reads the stored
// depth from the physical atlas and runs a PCF box filter.
//
// The chain mirrors the golden line for line:
//   * clip level from view distance (select_level), matching clipmap.rs;
//   * absolute world page floor(world / page_world_size), camera independent;
//   * camera-snapped resident window (window_origin), so the flat slot index
//     matches the request bitmap vsm_page_mark.wesl wrote;
//   * page_table[slot] -> physical page index (or VSM_PAGE_UNMAPPED), the twin
//     of VirtualPageTable::get returning a resident Residency::physical_page.
//
// Physical atlas layout: the bounded set of physical pages is packed as a
// square 2D texture grid physical_pages_per_edge on a side; physical page P
// occupies tile (P % edge_phys, P / edge_phys). Because physical pages are not
// contiguous in virtual space (a page's atlas neighbours belong to unrelated
// virtual pages), the PCF taps are clamped to the sampled page's own texel
// footprint so the filter never bleeds shadow across a page seam. A page that
// is not resident, or a receiver whose page falls outside the resident window,
// returns SHADOW_LIT (fully lit) -- exactly the golden cache-miss behaviour and
// the off-map convention shared with shadow.wesl.
//
// Self-contained (no imports); group 0 is
//   0 = page_table       (per-window-slot physical page index, storage read)
//   1 = physical_atlas    (physical page atlas, .r = stored NDC depth)
//   2 = physical_sampler  (linear sampler; depth compared manually in-shader)
//   3 = vsm_selftest_out  (standalone self-test output, storage r/w)
// with a VsmSampleParams immediate block.

// A texel is lit when reference_depth <= stored; unmapped / off-window points
// stay lit. Matches the shadow.wesl convention.
const SHADOW_LIT: f32 = 1.0;
// Sentinel physical index for a virtual page that is not resident this frame;
// the twin of VirtualPageTable::get returning None (a cache miss).
const VSM_PAGE_UNMAPPED: u32 = 0xffffffffu;

// Clipmap layout + this frame's physical-atlas geometry, uploaded as the
// immediate block. The clipmap fields mirror ClipmapConfig; the physical_*
// fields describe how the resident pages are packed into the atlas texture.
struct VsmSampleParams {
    // --- clipmap layout (mirrors ClipmapConfig) ---
    levels: u32,
    page_size: u32,
    pages_per_level_edge: u32,
    page_coord_bias: i32,
    level0_texel_world_size: f32,
    level0_max_distance: f32,
    // --- physical atlas geometry ---
    physical_pages: u32,
    physical_pages_per_edge: u32,
    // Soft-shadow PCF box half-width in physical texels (0 = single hard tap).
    pcf_radius: i32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One index per resident-window page slot across every level, laid out
// identically to vsm_page_mark.wesl's request bitmap:
//   slot = level * edge*edge + local_y * edge + local_x
// Each entry holds the backing physical page index, or VSM_PAGE_UNMAPPED when
// the virtual page is not resident.
@group(0) @binding(0) var<storage, read> page_table: array<u32>;
// The physical page atlas: a square grid of resident pages whose .r channel
// stores the wgpu NDC depth (near = 0, far = 1), the twin of the golden
// ShadowDepthSampler's stored value.
@group(0) @binding(1) var physical_atlas: texture_2d<f32>;
// Linear sampler; the depth comparison is done manually in-shader to stay a
// byte-for-byte twin of the CPU reference.
@group(0) @binding(2) var physical_sampler: sampler;

var<immediate> params: VsmSampleParams;

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
// view_distance. Twin of select_level.
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

// Absolute world page coordinate of world_xy at level. Twin of
// world_page_coords.
fn vsm_world_page_coords(level: u32, world_xy: vec2<f32>) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    return vec2<i32>(
        i32(floor(world_xy.x / pws)),
        i32(floor(world_xy.y / pws)),
    );
}

// Bottom-left page of level's resident window, snapped to whole pages around
// the camera. Twin of ClipmapConfig::build_level's origin_page.
fn vsm_window_origin(level: u32, camera_xy: vec2<f32>) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    let center = vec2<i32>(
        i32(floor(camera_xy.x / pws)),
        i32(floor(camera_xy.y / pws)),
    );
    let half = vsm_window_edge() / 2;
    return center - vec2<i32>(half, half);
}

// Flat slot index of a window-local page at level, or -1 when the page lies
// outside the level's resident window. Identical addressing to
// vsm_page_mark.wesl so the physical index read here corresponds to the flag
// marked there.
fn vsm_window_slot(level: u32, world_page: vec2<i32>, camera_xy: vec2<f32>) -> i32 {
    let edge = vsm_window_edge();
    let local = world_page - vsm_window_origin(level, camera_xy);
    if (local.x < 0 || local.y < 0 || local.x >= edge || local.y >= edge) {
        return -1;
    }
    return i32(level) * edge * edge + local.y * edge + local.x;
}

// Physical pages along one edge of the square atlas grid, always >= 1.
fn vsm_physical_edge() -> u32 {
    return max(params.physical_pages_per_edge, 1u);
}

// Samples the shadow visibility of a world point from the physical page atlas.
//
//   1. select the clip level from the receiver's view distance;
//   2. find its absolute world page and the flat resident-window slot;
//   3. look the slot up in the page table -> physical page (or a miss);
//   4. map the in-page fraction into the physical atlas tile;
//   5. PCF-filter the stored depth, clamping taps to the page's own texels.
//
// reference_depth is the receiver's light-space NDC depth (bias already folded
// in by the caller). Returns the lit fraction in [0, 1]; a page miss or an
// out-of-window receiver returns SHADOW_LIT.
fn sample_virtual_shadow(
    world_xy: vec2<f32>,
    view_distance: f32,
    camera_xy: vec2<f32>,
    reference_depth: f32,
) -> f32 {
    let level = vsm_select_level(view_distance);
    let pws = vsm_page_world_size(level);
    let world_page = vsm_world_page_coords(level, world_xy);

    let slot = vsm_window_slot(level, world_page, camera_xy);
    if (slot < 0) {
        return SHADOW_LIT;
    }
    let physical = page_table[u32(slot)];
    if (physical == VSM_PAGE_UNMAPPED) {
        return SHADOW_LIT;
    }

    // Fraction of the sampled point within its virtual page, in [0, 1).
    let page_origin_world = vec2<f32>(f32(world_page.x), f32(world_page.y)) * pws;
    let in_page_frac = clamp((world_xy - page_origin_world) / pws, vec2<f32>(0.0), vec2<f32>(1.0));

    // The physical page's tile in the square atlas grid.
    let edge_phys = vsm_physical_edge();
    let tile = vec2<f32>(
        f32(physical % edge_phys),
        f32(physical / edge_phys),
    );
    let inv_edge_phys = 1.0 / f32(edge_phys);
    let page_size = f32(max(params.page_size, 1u));
    // Half a physical texel, in page-local fraction units; used both as the PCF
    // step and to clamp taps to this page's interior so the filter cannot bleed
    // into the unrelated neighbouring pages packed beside it in the atlas.
    let texel_frac = 1.0 / page_size;
    let half_texel = 0.5 * texel_frac;

    let r = max(params.pcf_radius, 0);
    var lit = 0.0;
    var taps = 0.0;
    for (var y: i32 = -r; y <= r; y = y + 1) {
        for (var x: i32 = -r; x <= r; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_frac;
            // Clamp to this page's texel centres: no bleed across the seam.
            let tap_frac = clamp(
                in_page_frac + offset,
                vec2<f32>(half_texel),
                vec2<f32>(1.0 - half_texel),
            );
            let atlas_uv = (tile + tap_frac) * inv_edge_phys;
            let stored = textureSampleLevel(physical_atlas, physical_sampler, atlas_uv, 0.0).r;
            if (reference_depth <= stored) {
                lit = lit + 1.0;
            }
            taps = taps + 1.0;
        }
    }
    return lit / taps;
}

// ---------------------------------------------------------------------------
// Standalone compilation self-test entry point
// ---------------------------------------------------------------------------
// This shader is a shared library with no draw entry point of its own; the
// resolve pass calls sample_virtual_shadow. A trivial compute entry keeps the
// whole call graph reachable so the standalone WESL unit test type-checks the
// clipmap helpers, the page-table lookup and the PCF filter end to end.
@group(0) @binding(3) var<storage, read_write> vsm_selftest_out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn vsm_selftest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world_xy = vec2<f32>(f32(gid.x), f32(gid.y));
    let visibility = sample_virtual_shadow(world_xy, 1.0, vec2<f32>(0.0, 0.0), 0.5);
    let index = gid.y * 8u + gid.x;
    vsm_selftest_out[index] = visibility;
}
