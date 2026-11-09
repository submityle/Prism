// CLST v3 vertex-pulling opaque surface raster shader.
//
// This is a standalone WGSL module. Its bindings are an explicit integration
// contract, not yet installed by MeshPipelines or the CLST queue producer.
// Draw one cluster as triangle_count * 3 non-indexed vertices. The per-draw
// uniform selects the cluster and scene row; vertex_index supplies the corner
// stream. Invalid/stale input is sent outside clip space and rejected again by
// the fragment stage before it can write any GBuffer attachment.

const CLST_NON_RESIDENT: u32 = 0xffffffffu;
const CLST_MAGIC: u32 = 0x54534c43u;
const CLST_VERSION_V3: u32 = 3u;
const CLST_PAGE_ROW_WORDS: u32 = 8u;
const CLST_CLUSTER_ROW_WORDS: u32 = 8u;
const PRISM_MATERIAL_HEADER_WORDS: u32 = 29u; // GpuMaterialHeader is 116 bytes.
const PRISM_LOBE_ALL: u32 = 458879u;
const PRISM_LOBE_EMISSION: u32 = 1u;
const PRISM_LOBE_CLEARCOAT: u32 = 2u;
const PRISM_LOBE_ANISOTROPY: u32 = 4u;
const PRISM_LOBE_SHEEN: u32 = 8u;
const PRISM_LOBE_SUBSURFACE: u32 = 16u;
const PRISM_LOBE_TRANSMISSION: u32 = 32u;
const PRISM_LOBE_THIN_FILM: u32 = 64u;
const PRISM_LOBE_NPR_FACE: u32 = 65536u;
const PRISM_LOBE_GLINT: u32 = 131072u;
const PRISM_LOBE_MEASURED_BRDF: u32 = 262144u;
const MATERIAL_TEXTURE_SEMANTIC_BASE_COLOR: u32 = 0u;
const MATERIAL_VT_PORTABLE_TEXTURE_ATLAS: u32 = 0xfffffffeu;
const CLST_NORMAL_LOCATION: u32 = 1u;
const CLST_UV0_LOCATION: u32 = 2u;

fn clst_is_nan(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

fn clst_is_inf(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u;
}

fn clst_is_nan2(value: vec2<f32>) -> bool {
    return clst_is_nan(value.x) || clst_is_nan(value.y);
}

fn clst_is_inf2(value: vec2<f32>) -> bool {
    return clst_is_inf(value.x) || clst_is_inf(value.y);
}

fn clst_is_nan3(value: vec3<f32>) -> bool {
    return clst_is_nan(value.x) || clst_is_nan(value.y) || clst_is_nan(value.z);
}

fn clst_is_inf3(value: vec3<f32>) -> bool {
    return clst_is_inf(value.x) || clst_is_inf(value.y) || clst_is_inf(value.z);
}

fn clst_is_nan4(value: vec4<f32>) -> bool {
    return clst_is_nan3(value.xyz) || clst_is_nan(value.w);
}

fn clst_is_inf4(value: vec4<f32>) -> bool {
    return clst_is_inf3(value.xyz) || clst_is_inf(value.w);
}

struct ClstPage {
    generation: vec2<u32>,
    physical_page: u32,
    flags: u32,
    byte_offset: u32,
    byte_len: u32,
    cluster_start: u32,
    cluster_count: u32,
};

struct ClstCluster {
    node_index: u32,
    page_index: u32,
    payload_offset: u32,
    payload_size: u32,
    material_index: u32,
    material_generation: u32,
    vertex_count: u32,
    triangle_count: u32,
};

// 52-byte per-draw record. expected_generation is the low/high publication
// generation used by VirtualGeometryGpuPage. scene_index/generation address
// GpuSceneInstance and GpuSceneTransform stable-index tables.
struct ClstSurfaceParams {
    page_count: u32,
    cluster_count: u32,
    page_storage_words: u32,
    material_count: u32,
    expected_generation: vec2<u32>,
    cluster_index: u32,
    scene_index: u32,
    scene_generation: u32,
    expected_geometry_index: u32,
    expected_geometry_generation: u32,
    _reserved2: u32,
    _reserved3: u32,
};

// Byte-for-byte SurfaceFrame ABI from forward_lit.rs (144 bytes). This stage
// consumes the current view-projection matrix at offset zero and retains the
// rest of the canonical frame record for a direct Rust binding.
struct SurfaceFrame {
    clip_from_world: mat4x4<f32>,
    previous_clip_from_world: mat4x4<f32>,
    exposure_stops: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};

struct SceneInstance {
    generation: u32,
    flags: u32,
    geometry_index: u32,
    geometry_generation: u32,
    material_index: u32,
    material_generation: u32,
    active: u32,
    render_layers: u32,
};

struct SceneTransform {
    rows: array<vec4<f32>, 3>,
};

// GpuMaterialHeader ABI v8, 29 words / 116 bytes. Keep field order identical
// to prism_render_material::GpuMaterialHeader and material.wesl.
struct MaterialHeader {
    generation: u32,
    revision: u32,
    render_class: u32,
    render_domain: u32,
    feature_flags: u32,
    closure_mask: u32,
    parameter_offset: u32,
    parameter_size: u32,
    texture_offset: u32,
    texture_count: u32,
    sampler_offset: u32,
    sampler_count: u32,
    is_active: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    lobe_mask: u32,
    extensions_used_low: u32,
    extensions_used_high: u32,
    extensions_required_low: u32,
    extensions_required_high: u32,
    dynamic_wetness: u32,
    dynamic_snow: u32,
    dynamic_damage: u32,
    dynamic_dissolve: u32,
    execution_plan_index: u32,
    execution_program_index: u32,
    execution_package_index: u32,
    execution_permutation_low: u32,
    execution_permutation_high: u32,
};

// Canonical GpuMaterialTexture row (32-byte storage ABI). Texture rows are
// addressed through header.texture_offset/count; their order is significant
// because the WESL material sampler folds repeated semantics in row order.
struct MaterialTexture {
    index: u32,
    generation: u32,
    semantic: u32,
    sampler_index: u32,
    virtual_texture_id: u32,
    virtual_texture_width: u32,
    virtual_texture_height: u32,
    virtual_texture_page_size: u32,
};

struct MaterialVirtualTextureConfig {
    enabled: u32,
    entry_count: u32,
    page_size: u32,
    tile_extent_px: u32,
    tiles_per_row: u32,
    atlas_layers: u32,
    gutter_px: u32,
    coarsest_mip: u32,
};

struct SurfaceMaterial {
    index: u32,
    generation: u32,
    revision: u32,
    execution_plan_index: u32,
};

struct PulledAttribute {
    value: vec4<f32>,
    format: u32,
    valid: u32,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) material_index: u32,
    @location(4) @interpolate(flat) material_generation: u32,
    @location(5) @interpolate(flat) valid: u32,
};

struct GBufferOut {
    @location(0) position_x: f32,
    @location(1) position_y: f32,
    @location(2) position_z: f32,
    @location(3) normal: vec4<f32>,
    @location(4) albedo: vec4<f32>,
    @location(5) surface_id: u32,
};

@group(0) @binding(0) var<storage, read> clst_pages: array<ClstPage>;
@group(0) @binding(1) var<storage, read> clst_clusters: array<ClstCluster>;
@group(0) @binding(2) var<storage, read> clst_page_storage: array<u32>;
@group(0) @binding(3) var<storage, read> material_headers: array<MaterialHeader>;
@group(0) @binding(4) var<uniform> clst: ClstSurfaceParams;
@group(1) @binding(0) var<uniform> frame: SurfaceFrame;
@group(1) @binding(1) var<storage, read> scene_instances: array<SceneInstance>;
@group(1) @binding(2) var<storage, read> scene_transforms: array<SceneTransform>;
@group(1) @binding(3) var<storage, read> material_parameters: array<u32>;
@group(1) @binding(4) var<uniform> draw_material: SurfaceMaterial;
@group(1) @binding(5) var<storage, read> material_textures: array<MaterialTexture>;
@group(1) @binding(6) var<uniform> material_vt_config: MaterialVirtualTextureConfig;
@group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@group(1) @binding(8) var material_vt_sampler: sampler;

fn fail_attribute() -> PulledAttribute {
    var result: PulledAttribute;
    result.value = vec4<f32>(0.0);
    result.format = 0xffffffffu;
    result.valid = 0u;
    return result;
}

fn read_byte(base_word: u32, byte_offset: u32) -> u32 {
    let word = clst_page_storage[base_word + byte_offset / 4u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}

fn read_u16(base_word: u32, byte_offset: u32) -> u32 {
    return read_byte(base_word, byte_offset) | (read_byte(base_word, byte_offset + 1u) << 8u);
}

fn read_u32(base_word: u32, byte_offset: u32) -> u32 {
    return read_byte(base_word, byte_offset)
        | (read_byte(base_word, byte_offset + 1u) << 8u)
        | (read_byte(base_word, byte_offset + 2u) << 16u)
        | (read_byte(base_word, byte_offset + 3u) << 24u);
}

fn packed_material_word_count(mask: u32) -> u32 {
    var words = 20u;
    if (mask & PRISM_LOBE_EMISSION) != 0u { words += 8u; }
    if (mask & PRISM_LOBE_CLEARCOAT) != 0u { words += 16u; }
    if (mask & PRISM_LOBE_ANISOTROPY) != 0u { words += 4u; }
    if (mask & PRISM_LOBE_SHEEN) != 0u { words += 8u; }
    if (mask & PRISM_LOBE_SUBSURFACE) != 0u { words += 16u; }
    if (mask & PRISM_LOBE_TRANSMISSION) != 0u { words += 16u; }
    if (mask & PRISM_LOBE_THIN_FILM) != 0u { words += 4u; }
    if (mask & PRISM_LOBE_NPR_FACE) != 0u { words += 4u; }
    if (mask & PRISM_LOBE_GLINT) != 0u { words += 4u; }
    if (mask & PRISM_LOBE_MEASURED_BRDF) != 0u { words += 4u; }
    return words;
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_channel_to_linear(c.x), srgb_channel_to_linear(c.y),
        srgb_channel_to_linear(c.z));
}

fn portable_base_color_texel(row: MaterialTexture, uv: vec2<f32>) -> vec4<f32> {
    let config = material_vt_config;
    if row.generation == 0u || row.virtual_texture_id != MATERIAL_VT_PORTABLE_TEXTURE_ATLAS
        || config.enabled == 0u || config.page_size == 0u || config.tiles_per_row == 0u
        || config.tile_extent_px == 0u || config.atlas_layers == 0u
        || row.virtual_texture_page_size != config.page_size
        || row.virtual_texture_width == 0u || row.virtual_texture_height == 0u { discard; }
    if config.tiles_per_row > 65535u { discard; }
    let tiles_per_layer = config.tiles_per_row * config.tiles_per_row;
    if tiles_per_layer == 0u || config.atlas_layers > 0xffffffffu / tiles_per_layer
        || config.tile_extent_px > 0xffffffffu / config.tiles_per_row
        || row.index >= config.atlas_layers * tiles_per_layer { discard; }
    let atlas_extent = config.tile_extent_px * config.tiles_per_row;
    if atlas_extent == 0u || config.gutter_px > config.tile_extent_px / 2u
        || config.page_size > config.tile_extent_px - 2u * config.gutter_px { discard; }
    if !(uv.x >= -1.0e30 && uv.x <= 1.0e30 && uv.y >= -1.0e30 && uv.y <= 1.0e30) { discard; }
    let source_uv = clamp(uv * vec2<f32>(f32(row.virtual_texture_width) / f32(config.page_size),
        f32(row.virtual_texture_height) / f32(config.page_size)), vec2<f32>(0.0), vec2<f32>(1.0));
    let tile_x = row.index % config.tiles_per_row;
    let tile_y = (row.index / config.tiles_per_row) % config.tiles_per_row;
    let layer = row.index / tiles_per_layer;
    let atlas_uv = (vec2<f32>(f32(tile_x * config.tile_extent_px + config.gutter_px),
        f32(tile_y * config.tile_extent_px + config.gutter_px))
        + source_uv * f32(config.page_size)) / f32(atlas_extent);
    let texel = textureSampleLevel(material_vt_atlas, material_vt_sampler, atlas_uv, i32(layer), 0.0);
    if clst_is_nan4(texel) || clst_is_inf4(texel) { discard; }
    return texel;
}

fn sample_canonical_base_color(header: MaterialHeader, uv: vec2<f32>, authored: vec4<f32>) -> vec4<f32> {
    let rows_len = arrayLength(&material_textures);
    if header.texture_offset > rows_len || header.texture_count > rows_len - header.texture_offset { discard; }
    var color = authored;
    for (var i = 0u; i < header.texture_count; i += 1u) {
        let row = material_textures[header.texture_offset + i];
        if row.semantic == MATERIAL_TEXTURE_SEMANTIC_BASE_COLOR {
            let texel = portable_base_color_texel(row, uv);
            color = color * vec4<f32>(srgb_to_linear(texel.rgb), texel.a);
        } else if row.semantic == 1u || row.semantic == 2u || row.semantic == 3u
            || row.semantic == 4u || row.semantic == 5u || row.semantic == 6u
            || row.semantic == 7u || row.semantic == 8u || row.semantic == 11u {
            // This CLST GBuffer path has no canonical emissive/metal-roughness/
            // tangent-space/coat outputs. Refuse rather than silently drop a
            // bound material input that the WESL surface path consumes.
            discard;
        } else {
            // Unknown semantic means the shader cannot prove equivalent output.
            discard;
        }
    }
    return color;
}

fn pull_position(cluster: ClstCluster, page: ClstPage, triangle_index: u32, corner: u32)
    -> vec4<f32> {
    let base_word = page.byte_offset / 4u;
    let payload = cluster.payload_offset;
    let index_start = payload + 34u + cluster.vertex_count * 6u;
    let vertex_index = read_u16(base_word, index_start + triangle_index * 6u + corner * 2u);
    let q_at = payload + 34u + vertex_index * 6u;
    let q = vec3<f32>(f32(read_u16(base_word, q_at)),
        f32(read_u16(base_word, q_at + 2u)), f32(read_u16(base_word, q_at + 4u))) / 65535.0;
    let bounds_min = vec3<f32>(bitcast<f32>(read_u32(base_word, payload + 10u)),
        bitcast<f32>(read_u32(base_word, payload + 14u)), bitcast<f32>(read_u32(base_word, payload + 18u)));
    let bounds_max = vec3<f32>(bitcast<f32>(read_u32(base_word, payload + 22u)),
        bitcast<f32>(read_u32(base_word, payload + 26u)), bitcast<f32>(read_u32(base_word, payload + 30u)));
    if clst_is_nan3(bounds_min) || clst_is_inf3(bounds_min) || clst_is_nan3(bounds_max)
        || clst_is_inf3(bounds_max) || any(bounds_max < bounds_min) { return vec4<f32>(0.0); }
    return vec4<f32>(bounds_min + (bounds_max - bounds_min) * q, 1.0);
}

fn pull_attribute(cluster: ClstCluster, page: ClstPage, vertex_index: u32, location: u32)
    -> PulledAttribute {
    let base_word = page.byte_offset / 4u;
    let payload = cluster.payload_offset;
    let payload_end = payload + cluster.payload_size;
    let geometry_end = payload + 34u + cluster.vertex_count * 6u + cluster.triangle_count * 6u;
    let footer = (geometry_end + 3u) & ~3u;
    if footer > payload_end || payload_end - footer < 8u { return fail_attribute(); }
    let position_location = read_u32(base_word, footer);
    let count = read_u16(base_word, footer + 4u);
    if count >= 32u || read_u16(base_word, footer + 6u) != 0u { return fail_attribute(); }
    if location == position_location { return fail_attribute(); }
    var cursor = footer + 8u;
    for (var entry = 0u; entry < count; entry += 1u) {
        if cursor > payload_end || payload_end - cursor < 12u { return fail_attribute(); }
        let name_len = read_u16(base_word, cursor);
        let format = read_u16(base_word, cursor + 2u);
        let entry_location = read_u32(base_word, cursor + 4u);
        let data_len = read_u32(base_word, cursor + 8u);
        let name_begin = cursor + 12u;
        if name_len == 0u || name_len > 255u || name_begin > payload_end
            || name_len > payload_end - name_begin
            || name_begin + name_len > 0xfffffffcu { return fail_attribute(); }
        let data_begin = (name_begin + name_len + 3u) & ~3u;
        var element_size = 0u;
        switch format {
            case 0u, 6u, 9u, 10u, 11u: { element_size = 4u; }
            case 1u, 4u, 7u: { element_size = 8u; }
            case 2u: { element_size = 12u; }
            case 3u, 5u, 8u: { element_size = 16u; }
            default: { return fail_attribute(); }
        }
        if data_begin < name_begin || data_begin > payload_end
            || cluster.vertex_count > 0xffffffffu / element_size
            || data_len != cluster.vertex_count * element_size
            || data_len > payload_end - data_begin { return fail_attribute(); }
        if entry_location == location {
            let at = data_begin + vertex_index * element_size;
            var value = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            let components = element_size / select(4u, 2u, format == 4u || format == 5u);
            for (var component = 0u; component < min(components, 4u); component += 1u) {
                if format == 4u || format == 5u {
                    let packed = read_u32(base_word, at + (component & ~1u) * 2u);
                    value[component] = unpack2x16float(packed)[component & 1u];
                } else if format == 10u {
                    value[component] = f32(read_byte(base_word, at + component)) / 255.0;
                } else if format == 11u {
                    let signed_byte = bitcast<i32>(read_byte(base_word, at + component) << 24u) >> 24;
                    value[component] = max(f32(signed_byte) / 127.0, -1.0);
                } else if format == 9u {
                    value[component] = f32(bitcast<i32>(read_u32(base_word, at + component * 4u)));
                } else if format == 6u || format == 7u || format == 8u {
                    value[component] = f32(read_u32(base_word, at + component * 4u));
                } else {
                    value[component] = bitcast<f32>(read_u32(base_word, at + component * 4u));
                }
            }
            var result: PulledAttribute;
            result.value = value;
            result.format = format;
            result.valid = 1u;
            return result;
        }
        cursor = data_begin + data_len;
    }
    return fail_attribute();
}

fn load_valid(cluster_index: u32, vertex_index: u32) -> bool {
    if clst.page_count == 0u || clst.cluster_index != cluster_index
        || clst.page_count > arrayLength(&clst_pages)
        || clst.cluster_count > arrayLength(&clst_clusters)
        || clst.material_count > arrayLength(&material_headers)
        || clst.page_storage_words > arrayLength(&clst_page_storage)
        || clst.cluster_index >= clst.cluster_count
        || clst.scene_index >= arrayLength(&scene_instances)
        || clst.scene_index >= arrayLength(&scene_transforms) { return false; }
    let cluster = clst_clusters[clst.cluster_index];
    if cluster.page_index >= clst.page_count || cluster.vertex_count == 0u
        || cluster.vertex_count > 64u || cluster.triangle_count == 0u
        || cluster.triangle_count > 128u || vertex_index >= cluster.triangle_count * 3u
        || cluster.payload_size < 34u { return false; }
    let page = clst_pages[cluster.page_index];
    if (page.flags & 1u) == 0u || page.physical_page == CLST_NON_RESIDENT
        || any(page.generation != clst.expected_generation) || page.byte_len == 0u
        || cluster.payload_offset > page.byte_len || cluster.payload_size > page.byte_len - cluster.payload_offset
        || (page.byte_offset & 3u) != 0u || page.byte_offset / 4u > clst.page_storage_words
        || page.byte_len / 4u + select(0u, 1u, (page.byte_len & 3u) != 0u)
            > clst.page_storage_words - page.byte_offset / 4u { return false; }
    if clst.cluster_index < page.cluster_start
        || clst.cluster_index - page.cluster_start >= page.cluster_count { return false; }
    let base_word = page.byte_offset / 4u;
    let payload = cluster.payload_offset;
    if read_u32(base_word, payload) != CLST_MAGIC { return false; }
    let version_and_vertices = read_u32(base_word, payload + 4u);
    let version = version_and_vertices & 65535u;
    let vertex_count = version_and_vertices >> 16u;
    let triangle_count = read_u16(base_word, payload + 8u);
    if version != CLST_VERSION_V3
        || vertex_count != cluster.vertex_count || triangle_count != cluster.triangle_count { return false; }
    let geometry_size = 34u + vertex_count * 6u + triangle_count * 6u;
    if cluster.payload_size < geometry_size { return false; }
    let bounds_min = vec3<f32>(bitcast<f32>(read_u32(base_word, payload + 10u)),
        bitcast<f32>(read_u32(base_word, payload + 14u)), bitcast<f32>(read_u32(base_word, payload + 18u)));
    let bounds_max = vec3<f32>(bitcast<f32>(read_u32(base_word, payload + 22u)),
        bitcast<f32>(read_u32(base_word, payload + 26u)), bitcast<f32>(read_u32(base_word, payload + 30u)));
    if clst_is_nan3(bounds_min) || clst_is_inf3(bounds_min) || clst_is_nan3(bounds_max)
        || clst_is_inf3(bounds_max) || any(bounds_max < bounds_min) { return false; }
    let index_start = payload + 34u + vertex_count * 6u;
    let geometry_end = index_start + triangle_count * 6u;
    if geometry_end > payload + cluster.payload_size { return false; }
    let local_vertex = read_u16(base_word, index_start + (vertex_index / 3u) * 6u
        + (vertex_index % 3u) * 2u);
    if local_vertex >= vertex_count { return false; }
    // Every corner of a triangle must be valid. Otherwise a malformed corner
    // clipped to the invalid sentinel could still produce partially rasterized
    // fragments whose flat validity depends on the provoking-vertex rule.
    for (var corner = 0u; corner < 3u; corner += 1u) {
        let tri_vertex = read_u16(base_word, index_start + (vertex_index / 3u) * 6u + corner * 2u);
        if tri_vertex >= vertex_count { return false; }
        let normal = pull_attribute(cluster, page, tri_vertex, CLST_NORMAL_LOCATION);
        let uv = pull_attribute(cluster, page, tri_vertex, CLST_UV0_LOCATION);
        if normal.valid == 0u || uv.valid == 0u
            || !(normal.format == 2u || normal.format == 3u || normal.format == 5u || normal.format == 11u)
            || !(uv.format == 1u || uv.format == 4u)
            || clst_is_nan4(normal.value) || clst_is_inf4(normal.value)
            || clst_is_nan4(uv.value) || clst_is_inf4(uv.value) { return false; }
    }
    let instance = scene_instances[clst.scene_index];
    if instance.active == 0u || instance.generation != clst.scene_generation
        || instance.geometry_index != clst.expected_geometry_index
        || instance.geometry_generation != clst.expected_geometry_generation
        || instance.material_index != cluster.material_index
        || instance.material_generation != cluster.material_generation { return false; }
    if cluster.material_index >= clst.material_count { return false; }
    let header = material_headers[cluster.material_index];
    if header.is_active == 0u || header.generation != cluster.material_generation
        || header.generation != draw_material.generation || draw_material.index != cluster.material_index
        || header.revision != draw_material.revision
        || header.execution_plan_index != draw_material.execution_plan_index
        || (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u { return false; }
    let words = packed_material_word_count(header.lobe_mask);
    let parameter_words = arrayLength(&material_parameters);
    if header.parameter_size != words * 4u || header.parameter_offset > parameter_words
        || words > parameter_words - header.parameter_offset { return false; }
    // This file deliberately has no texture/VTM bindings yet. Never emit a
    // wrong texel for a textured material: refuse until the canonical Surface
    // texture sampling group is installed by the Rust pipeline integration.
    if header.texture_count != 0u { return false; }
    let footer = (geometry_end + 3u) & ~3u;
    if footer > payload + cluster.payload_size || payload + cluster.payload_size - footer < 8u
        || read_u16(base_word, footer + 6u) != 0u { return false; }
    let encoded_position_location = read_u32(base_word, footer);
    if encoded_position_location == CLST_NORMAL_LOCATION
        || encoded_position_location == CLST_UV0_LOCATION { return false; }
    return true;
}

fn invalid_vertex() -> VertexOut {
    var output: VertexOut;
    output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.world_position = vec3<f32>(0.0);
    output.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    output.uv0 = vec2<f32>(0.0);
    output.material_index = CLST_NON_RESIDENT;
    output.material_generation = 0u;
    output.valid = 0u;
    return output;
}

@vertex
fn vs_clst(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    if !load_valid(clst.cluster_index, vertex_index) { return invalid_vertex(); }
    let cluster = clst_clusters[clst.cluster_index];
    let page = clst_pages[cluster.page_index];
    let triangle = vertex_index / 3u;
    let corner = vertex_index % 3u;
    let local_position = pull_position(cluster, page, triangle, corner);
    let transform = scene_transforms[clst.scene_index].rows;
    let world = vec4<f32>(dot(transform[0], local_position), dot(transform[1], local_position),
        dot(transform[2], local_position), 1.0);
    var local_normal: vec3<f32>;
    var uv0 = vec2<f32>(0.0);
    let base_word = page.byte_offset / 4u;
    let payload = cluster.payload_offset;
    let index_start = payload + 34u + cluster.vertex_count * 6u;
    let local_vertex = read_u16(base_word, index_start + triangle * 6u + corner * 2u);
    let normal = pull_attribute(cluster, page, local_vertex, CLST_NORMAL_LOCATION);
    let uv = pull_attribute(cluster, page, local_vertex, CLST_UV0_LOCATION);
    local_normal = normal.value.xyz;
    uv0 = uv.value.xy;
    let cofactor0 = cross(transform[1].xyz, transform[2].xyz);
    let cofactor1 = cross(transform[2].xyz, transform[0].xyz);
    let cofactor2 = cross(transform[0].xyz, transform[1].xyz);
    let determinant = dot(transform[0].xyz, cofactor0);
    var world_normal = vec3<f32>(0.0, 1.0, 0.0);
    if abs(determinant) > 1.0e-8 {
        let n = vec3<f32>(dot(cofactor0, local_normal), dot(cofactor1, local_normal),
            dot(cofactor2, local_normal)) / determinant;
        if dot(n, n) > 1.0e-12 { world_normal = normalize(n); }
    } else if dot(local_normal, local_normal) > 1.0e-12 {
        let n = vec3<f32>(dot(transform[0].xyz, local_normal), dot(transform[1].xyz, local_normal),
            dot(transform[2].xyz, local_normal));
        if dot(n, n) > 1.0e-12 { world_normal = normalize(n); }
    }
    var output: VertexOut;
    output.clip_position = frame.clip_from_world * world;
    output.world_position = world.xyz;
    output.world_normal = world_normal;
    output.uv0 = uv0;
    output.material_index = cluster.material_index;
    output.material_generation = cluster.material_generation;
    output.valid = 1u;
    return output;
}

@fragment
fn fs_clst(input: VertexOut) -> GBufferOut {
    if input.valid == 0u || input.material_index >= arrayLength(&material_headers) { discard; }
    let header = material_headers[input.material_index];
    if header.is_active == 0u || header.generation != input.material_generation
        || header.generation != draw_material.generation || header.revision != draw_material.revision
        || header.execution_plan_index != draw_material.execution_plan_index
        || input.material_index != draw_material.index { discard; }
    // parameter_offset is a word offset into the canonical packed heap.
    // load_valid already validated the exact variable block length derived from
    // every lobe footprint; core fields are the first 20 words for every mask.
    let color_at = header.parameter_offset;
    let albedo = vec4<f32>(bitcast<f32>(material_parameters[color_at]),
        bitcast<f32>(material_parameters[color_at + 1u]),
        bitcast<f32>(material_parameters[color_at + 2u]),
        bitcast<f32>(material_parameters[color_at + 3u]));
    if clst_is_nan4(albedo) || clst_is_inf4(albedo) { discard; }
    let opacity = bitcast<f32>(material_parameters[header.parameter_offset + 18u]);
    if clst_is_nan(opacity) || clst_is_inf(opacity) { discard; }
    var output: GBufferOut;
    output.position_x = input.world_position.x;
    output.position_y = input.world_position.y;
    output.position_z = input.world_position.z;
    let normal_length_squared = dot(input.world_normal, input.world_normal);
    let normal = select(vec3<f32>(0.0, 1.0, 0.0),
        input.world_normal * inverseSqrt(normal_length_squared), normal_length_squared > 1.0e-12);
    output.normal = vec4<f32>(normal, 1.0);
    let textured_albedo = sample_canonical_base_color(header, input.uv0, albedo);
    output.albedo = vec4<f32>(textured_albedo.rgb, textured_albedo.a * opacity);
    output.surface_id = input.material_index;
    return output;
}
