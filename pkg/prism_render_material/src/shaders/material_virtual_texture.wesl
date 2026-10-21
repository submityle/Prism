// Material virtual-texture page resolution and physical-atlas address math.
//
// The page-table binding is part of the production group(1) ABI and is owned by
// this resolver module. Keeping it here avoids passing storage pointers through
// helper calls, which is rejected by naga/Metal for runtime arrays. The
// page-table entry ABI is the existing GpuPageTable contract:
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

// Stable material-VT config ABI. The production consumer binds this exact
// eight-word record as a 32-byte read-only storage record at group(1), binding(6). Binding(5)
// is the page-table word buffer, (7) the 2D-array atlas, (8) its filtering
// sampler, and (9) the bindless slot-generation table.
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

@group(1) @binding(5) var<storage, read> material_vt_page_table: array<u32>;

// Request derived from a material row and the compute resolve's explicit LOD.
// Coordinates are in the virtual image's mip-space page grid.
struct MaterialVirtualTexturePageRequest {
    valid: bool,
    mip: u32,
    page_x: u32,
    page_y: u32,
    local_uv: vec2<f32>,
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
    entry_count: u32,
    t0: u32,
    t1: u32,
    t2: u32,
) -> u32 {
    let available_entries = arrayLength(&material_vt_page_table) / MATERIAL_VT_ENTRY_WORDS;
    var lo = 0u;
    var hi = min(entry_count, available_entries);
    loop {
        if lo >= hi { break; }
        let mid = lo + (hi - lo) / 2u;
        let base = mid * MATERIAL_VT_ENTRY_WORDS;
        let order = material_vt_compare_words(
            material_vt_page_table[base],
            material_vt_page_table[base + 1u],
            material_vt_page_table[base + 2u],
            t0,
            t1,
            t2,
        );
        if order < 0 {
            lo = mid + 1u;
        } else if order > 0 {
            hi = mid;
        } else {
            return material_vt_page_table[base + 3u];
        }
    }
    return MATERIAL_VT_MISS;
}

fn material_vt_shift_page(value: u32, delta: u32) -> u32 {
    if delta >= 16u { return 0u; }
    return value >> delta;
}

fn material_vt_mip_extent(value: u32, mip: u32) -> u32 {
    if mip >= 32u { return 1u; }
    return max(value >> mip, 1u);
}

// Convert normalized material UV + analytic LOD into the exact page-table key
// requested at that mip. This is deliberately independent of texture/resource
// bindings so it can be shared by a caller with its own production layout.
fn material_vt_page_request(
    texture_id: u32,
    texture_width: u32,
    texture_height: u32,
    page_size: u32,
    uv: vec2<f32>,
    lod: f32,
) -> MaterialVirtualTexturePageRequest {
    var out: MaterialVirtualTexturePageRequest;
    out.valid = false;
    out.mip = 0u;
    out.page_x = 0u;
    out.page_y = 0u;
    out.local_uv = vec2<f32>(0.0);
    if texture_id == MATERIAL_VT_NON_VIRTUAL_TEXTURE
        || texture_width == 0u
        || texture_height == 0u
        || page_size == 0u
        || !(uv.x >= -1.0e30 && uv.x <= 1.0e30)
        || !(uv.y >= -1.0e30 && uv.y <= 1.0e30)
        || !(lod >= -1.0e30 && lod <= 1.0e30) {
        return out;
    }

    // Texture dimensions are u32, so mip 31 is the coarsest representable
    // extent. Mips above that all address the same 1x1 tail page.
    let mip = min(u32(max(floor(max(lod, 0.0)), 0.0)), 31u);
    let mip_width = material_vt_mip_extent(texture_width, mip);
    let mip_height = material_vt_mip_extent(texture_height, mip);
    let pages_x = (mip_width - 1u) / page_size + 1u;
    let pages_y = (mip_height - 1u) / page_size + 1u;
    if pages_x == 0u || pages_y == 0u
        || pages_x - 1u > MATERIAL_VT_MAX_KEY_COORD
        || pages_y - 1u > MATERIAL_VT_MAX_KEY_COORD {
        return out;
    }

    let safe_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = safe_uv * vec2<f32>(f32(mip_width), f32(mip_height));
    let page_position = texel / f32(page_size);
    let page_x = min(u32(floor(page_position.x)), pages_x - 1u);
    let page_y = min(u32(floor(page_position.y)), pages_y - 1u);
    var local_uv = fract(page_position);
    // UV == 1 is the outer image edge, not the start of a phantom page. Keep
    // that edge on the final page's far boundary so populated gutters filter
    // consistently with the direct clamp-to-edge fallback.
    if safe_uv.x >= 1.0 { local_uv.x = 1.0; }
    if safe_uv.y >= 1.0 { local_uv.y = 1.0; }

    out.valid = true;
    out.mip = mip;
    out.page_x = page_x;
    out.page_y = page_y;
    out.local_uv = local_uv;
    return out;
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

// Sample a resolved page from a physical 2D-array atlas. Resource handles are
// parameters rather than module bindings: the production caller must provide
// the matching page table/config/atlas/sampler in its own bind-group layout.
// Each physical tile contains one already-filtered virtual mip page, so the
// physical atlas is sampled at level zero.
fn material_vt_sample_resolved(
    atlas: texture_2d_array<f32>,
    atlas_sampler: sampler,
    resolution: MaterialVirtualTextureResolution,
) -> vec4<f32> {
    return textureSampleLevel(
        atlas,
        atlas_sampler,
        resolution.atlas_uv,
        i32(resolution.atlas_layer),
        0.0,
    );
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

// A page-table hit is not sufficient by itself: w3 is an atlas slot and must
// address a real tile in the configured layered atlas. Keep the multiplication
// checked before comparing the slot so a malformed config cannot turn a slot
// into an out-of-range layer or an overflowing atlas capacity.
fn material_vt_slot_is_valid(slot: u32, config: MaterialVirtualTextureConfig) -> bool {
    if config.tiles_per_row == 0u || config.atlas_layers == 0u {
        return false;
    }
    if config.tiles_per_row > 0xffffffffu / config.tiles_per_row {
        return false;
    }
    let slots_per_layer = config.tiles_per_row * config.tiles_per_row;
    if config.atlas_layers > 0xffffffffu / slots_per_layer {
        return false;
    }
    return slot < slots_per_layer * config.atlas_layers;
}

// Resolve a material page to an exact resident tile or the nearest available
// coarser covering tile. The page table is the production binding above; the
// config and request remain explicit values so the sampling consumer controls
// its fallback policy.
// Inputs use the packed page-table key's representable ranges (u8 mip/u16
// layer/u16 page coordinates). Non-VT and disabled requests return inactive;
// invalid config, key, atlas slot, or an exhausted fallback walk returns an
// enabled miss with hit=false.
fn material_vt_resolve(
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
            config.entry_count,
            texture_id,
            key1,
            key2,
        );
        if slot != MATERIAL_VT_MISS {
            if !material_vt_slot_is_valid(slot, config) {
                return miss;
            }
            let atlas_layer = slot / per_layer;
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
