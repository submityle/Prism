// Material virtual-texture page resolution and physical-atlas address math.
//
// This module deliberately declares no resource bindings. The caller supplies
// the resident GpuPageTable as a read-only storage pointer and owns all binding
// layout, upload/publish ordering, feedback, and atlas sampling. The page-table
// entry ABI is the existing GpuPageTable contract:
//   word 0 = texture id
//   word 1 = (mip << 24) | (layer << 8)
//   word 2 = (page_x << 16) | page_y
//   word 3 = physical slot
// Entries are sorted by unsigned lexicographic order over words 0..2.
//
// AtlasGeometry defines slot -> array-layer/tile-origin and tile extent, but
// does not define a texel gutter or the uploaded tile's content footprint.
// Therefore gutter_px and page_size are explicit caller-provided configuration;
// this resolver assumes the atlas uploader places page_size content texels at
// [gutter_px, gutter_px + page_size] in each tile and has populated gutters.

const MATERIAL_VT_ENTRY_WORDS: u32 = 4u;
const MATERIAL_VT_MISS: u32 = 0xffffffffu;
const MATERIAL_VT_MAX_KEY_MIP: u32 = 255u;
const MATERIAL_VT_MAX_KEY_COORD: u32 = 65535u;
const MATERIAL_VT_MAX_KEY_LAYER: u32 = 65535u;
const MATERIAL_VT_NON_VIRTUAL_TEXTURE: u32 = 0xffffffffu;

// All fields are caller-owned configuration, not a Rust/GPU ABI declaration.
// `enabled == 0` disables VT. `coarsest_mip` bounds the fallback walk.
struct MaterialVirtualTextureConfig {
    enabled: u32,
    entry_count: u32,
    page_size: u32,
    tile_extent_px: u32,
    tiles_per_row: u32,
    atlas_layers: u32,
    gutter_px: u32,
    coarsest_mip: u32,
}

// `vt_enabled` means the input represented an enabled VT request;
// `hit` means an exact/coarser page was found and its address is usable.
// Disabled/non-VT and total-miss results both have hit=false, but are distinct
// through `vt_enabled`. `resolved_mip` and `slot` use MATERIAL_VT_MISS on failure.
struct MaterialVirtualTextureResolution {
    vt_enabled: bool,
    hit: bool,
    slot: u32,
    resolved_mip: u32,
    atlas_layer: u32,
    atlas_uv: vec2<f32>,
    page_local_uv: vec2<f32>,
}

fn material_vt_compare_words(
    e0: u32,
    e1: u32,
    e2: u32,
    t0: u32,
    t1: u32,
    t2: u32,
) -> i32 {
    if e0 < t0 { return -1; }
    if e0 > t0 { return 1; }
    if e1 < t1 { return -1; }
    if e1 > t1 { return 1; }
    if e2 < t2 { return -1; }
    if e2 > t2 { return 1; }
    return 0;
}

// Mirrors GpuPageTable::lookup and page_resolve.wgsl. `entry_count` is clipped
// to the actual runtime-array length so stale config cannot index past the
// supplied storage buffer.
fn material_vt_exact_slot(
    table: ptr<storage, array<u32>, read>,
    entry_count: u32,
    t0: u32,
    t1: u32,
    t2: u32,
) -> u32 {
    let available_entries = arrayLength(table) / MATERIAL_VT_ENTRY_WORDS;
    var lo = 0u;
    var hi = min(entry_count, available_entries);
    loop {
        if lo >= hi { break; }
        let mid = lo + (hi - lo) / 2u;
        let base = mid * MATERIAL_VT_ENTRY_WORDS;
        let order = material_vt_compare_words(
            (*table)[base],
            (*table)[base + 1u],
            (*table)[base + 2u],
            t0,
            t1,
            t2,
        );
        if order < 0 {
            lo = mid + 1u;
        } else if order > 0 {
            hi = mid;
        } else {
            return (*table)[base + 3u];
        }
    }
    return MATERIAL_VT_MISS;
}

fn material_vt_shift_page(value: u32, delta: u32) -> u32 {
    if delta >= 16u { return 0u; }
    return value >> delta;
}

// A fine page's sub-page position within the coarser page that covers it.
// Page coordinates are u16 in GpuPageTable; deltas >= 16 collapse the page
// grid coordinate to zero, matching the CPU and page_resolve.wgsl contracts.
fn material_vt_coarse_page_uv(
    page_x: u32,
    page_y: u32,
    local_uv: vec2<f32>,
    mip_delta: u32,
) -> vec2<f32> {
    let safe_local = clamp(local_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    var sub_page = vec2<u32>(page_x, page_y);
    if mip_delta >= 16u {
        // Page coordinates are u16, so every coordinate is within the first
        // 65536-way subdivision of a mip-16-or-coarser parent page.
        sub_page = sub_page & vec2<u32>(MATERIAL_VT_MAX_KEY_COORD);
    } else {
        let mask = (1u << mip_delta) - 1u;
        sub_page = sub_page & vec2<u32>(mask, mask);
    }
    let scale = exp2(-f32(mip_delta));
    return clamp(
        (vec2<f32>(sub_page) + safe_local) * scale,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
}

// Map page-local [0,1] coordinates to the normalized full-atlas UV. Tile
// extent includes gutters; sampler filtering may use those populated gutters
// to cross a virtual-page seam without crossing into an unrelated atlas slot.
fn material_vt_atlas_uv(
    page_local_uv: vec2<f32>,
    slot: u32,
    tile_extent_px: u32,
    tiles_per_row: u32,
    page_size: u32,
    gutter_px: u32,
) -> vec2<f32> {
    let tiles = max(tiles_per_row, 1u);
    let tile = vec2<u32>(slot % tiles, (slot / tiles) % tiles);
    let atlas_extent = f32(tile_extent_px) * f32(tiles);
    let texel = vec2<f32>(tile) * f32(tile_extent_px)
        + vec2<f32>(f32(gutter_px))
        + clamp(page_local_uv, vec2<f32>(0.0), vec2<f32>(1.0)) * f32(page_size);
    return texel / atlas_extent;
}

fn material_vt_empty_resolution(vt_is_enabled: bool) -> MaterialVirtualTextureResolution {
    var out: MaterialVirtualTextureResolution;
    out.vt_enabled = vt_is_enabled;
    out.hit = false;
    out.slot = MATERIAL_VT_MISS;
    out.resolved_mip = MATERIAL_VT_MISS;
    out.atlas_layer = MATERIAL_VT_MISS;
    out.atlas_uv = vec2<f32>(0.0);
    out.page_local_uv = vec2<f32>(0.0);
    return out;
}

// Resolve a material page to an exact resident tile or the nearest available
// coarser covering tile. The caller supplies the page table pointer; no
// production group/binding or feedback protocol is implied by this helper.
// Inputs use the packed page-table key's representable ranges (u8 mip/u16
// layer/u16 page coordinates). Non-VT and disabled requests return inactive;
// invalid config, key, atlas slot, or an exhausted fallback walk returns an
// enabled miss with hit=false.
fn material_vt_resolve(
    table: ptr<storage, array<u32>, read>,
    config: MaterialVirtualTextureConfig,
    texture_id: u32,
    texture_layer: u32,
    requested_mip: u32,
    requested_page_x: u32,
    requested_page_y: u32,
    local_uv: vec2<f32>,
) -> MaterialVirtualTextureResolution {
    if config.enabled == 0u || texture_id == MATERIAL_VT_NON_VIRTUAL_TEXTURE {
        return material_vt_empty_resolution(false);
    }

    var miss = material_vt_empty_resolution(true);
    if requested_mip > MATERIAL_VT_MAX_KEY_MIP
        || texture_layer > MATERIAL_VT_MAX_KEY_LAYER
        || requested_page_x > MATERIAL_VT_MAX_KEY_COORD
        || requested_page_y > MATERIAL_VT_MAX_KEY_COORD
        || config.entry_count == 0u
        || config.page_size == 0u
        || config.tile_extent_px == 0u
        || config.tiles_per_row == 0u
        || config.atlas_layers == 0u
    {
        return miss;
    }
    if config.tiles_per_row > 0xffffffffu / config.tiles_per_row
        || config.tile_extent_px > 0xffffffffu / config.tiles_per_row {
        return miss;
    }
    // Keep subtraction well-defined while validating the caller-owned packing
    // contract: tile_extent = page_size + 2 * gutter (+ optional spare texels).
    if config.gutter_px > (config.tile_extent_px - 1u) / 2u {
        return miss;
    }
    if config.page_size > config.tile_extent_px - 2u * config.gutter_px {
        return miss;
    }

    let per_layer = config.tiles_per_row * config.tiles_per_row;
    let last_mip = min(max(config.coarsest_mip, requested_mip), MATERIAL_VT_MAX_KEY_MIP);
    var mip = requested_mip;
    loop {
        let delta = mip - requested_mip;
        let page_x = material_vt_shift_page(requested_page_x, delta);
        let page_y = material_vt_shift_page(requested_page_y, delta);
        let key1 = (mip << 24u) | (texture_layer << 8u);
        let key2 = (page_x << 16u) | page_y;
        let slot = material_vt_exact_slot(
            table,
            config.entry_count,
            texture_id,
            key1,
            key2,
        );
        if slot != MATERIAL_VT_MISS {
            let atlas_layer = slot / per_layer;
            if atlas_layer >= config.atlas_layers {
                return miss;
            }
            let resolved_local_uv = material_vt_coarse_page_uv(
                requested_page_x,
                requested_page_y,
                local_uv,
                delta,
            );
            var out: MaterialVirtualTextureResolution;
            out.vt_enabled = true;
            out.hit = true;
            out.slot = slot;
            out.resolved_mip = mip;
            out.atlas_layer = atlas_layer;
            out.page_local_uv = resolved_local_uv;
            out.atlas_uv = material_vt_atlas_uv(
                resolved_local_uv,
                slot,
                config.tile_extent_px,
                config.tiles_per_row,
                config.page_size,
                config.gutter_px,
            );
            return out;
        }
        if mip == last_mip { break; }
        mip = mip + 1u;
    }
    return miss;
}
