// CLST vertex-pulling contract. Rust mirrors are the repr(C) records in
// prism_render_architecture::virtual_geometry::gpu and
// prism_render_material::GpuMaterialHeader.

const CLST_NON_RESIDENT: u32 = 0xffffffffu;
const CLST_MAGIC: u32 = 0x54534c43u;
const CLST_VERSION_V2: u32 = 2u;
const CLST_VERSION_V3: u32 = 3u;
const CLST_PAGE_ROW_WORDS: u32 = 8u;
const CLST_CLUSTER_ROW_WORDS: u32 = 8u;
const CLST_MATERIAL_ROW_WORDS: u32 = 40u;

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

struct ClstPullParams {
    page_count: u32,
    cluster_count: u32,
    page_storage_words: u32,
    material_count: u32,
    expected_generation: vec2<u32>,
    cluster_index: u32,
    triangle_index: u32,
    requested_location: u32,
    pad: u32,
};

struct ClstPulledVertex {
    position: vec4<f32>,
    attribute: vec4<f32>,
    material_index: u32,
    material_generation: u32,
    valid: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> clst_pages: array<ClstPage>;
@group(0) @binding(1) var<storage, read> clst_clusters: array<ClstCluster>;
@group(0) @binding(2) var<storage, read> clst_page_storage: array<u32>;
// The 40-word stride is GpuMaterialHeader ABI8; generation is word 0 and
// is_active is word 13. Keep the full stride even though this path reads only
// those two words.
@group(0) @binding(3) var<storage, read> clst_material_headers: array<u32>;
@group(0) @binding(4) var<uniform> clst_pull_params: ClstPullParams;
@group(0) @binding(5) var<storage, read_write> clst_pull_output: array<ClstPulledVertex>;

fn clst_read_byte(base_word: u32, byte_offset: u32) -> u32 {
    let word = clst_page_storage[base_word + byte_offset / 4u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}

fn clst_read_u16(base_word: u32, byte_offset: u32) -> u32 {
    return clst_read_byte(base_word, byte_offset)
        | (clst_read_byte(base_word, byte_offset + 1u) << 8u);
}

fn clst_read_u32(base_word: u32, byte_offset: u32) -> u32 {
    return clst_read_byte(base_word, byte_offset)
        | (clst_read_byte(base_word, byte_offset + 1u) << 8u)
        | (clst_read_byte(base_word, byte_offset + 2u) << 16u)
        | (clst_read_byte(base_word, byte_offset + 3u) << 24u);
}

fn clst_fail() -> ClstPulledVertex {
    var result: ClstPulledVertex;
    result.position = vec4<f32>(0.0);
    result.attribute = vec4<f32>(0.0);
    result.material_index = CLST_NON_RESIDENT;
    result.material_generation = 0u;
    result.valid = 0u;
    result.pad = 0u;
    return result;
}

fn clst_decode_attribute(base_word: u32, footer: u32, payload_end: u32, vertex_count: u32,
    vertex_index: u32, location: u32) -> vec4<f32> {
    if footer > payload_end || payload_end - footer < 8u { return vec4<f32>(0.0); }
    let position_location = clst_read_u32(base_word, footer);
    let count = clst_read_u16(base_word, footer + 4u);
    if count > 31u { return vec4<f32>(0.0); }
    var cursor = footer + 8u;
    if location == position_location { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
    for (var entry = 0u; entry < count; entry = entry + 1u) {
        if cursor > payload_end || payload_end - cursor < 12u { return vec4<f32>(0.0); }
        let name_len = clst_read_u16(base_word, cursor);
        let format = clst_read_u16(base_word, cursor + 2u);
        let entry_location = clst_read_u32(base_word, cursor + 4u);
        let data_len = clst_read_u32(base_word, cursor + 8u);
        let name_begin = cursor + 12u;
        let data_begin = (name_begin + name_len + 3u) & ~3u;
        let expected_size = select(0u, 4u, format == 0u || format == 6u || format == 9u || format == 10u || format == 11u)
            + select(0u, 8u, format == 1u || format == 4u || format == 7u)
            + select(0u, 12u, format == 2u)
            + select(0u, 16u, format == 3u || format == 5u || format == 8u);
        if expected_size == 0u || data_len != vertex_count * expected_size
            || name_len == 0u || data_begin < name_begin || data_begin > payload_end
            || data_len > payload_end - data_begin {
            return vec4<f32>(0.0);
        }
        if entry_location == location {
            let at = data_begin + vertex_index * expected_size;
            var value = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            let components = expected_size / select(4u, 2u, format == 4u || format == 5u);
            for (var component = 0u; component < min(components, 4u); component = component + 1u) {
                if format == 4u || format == 5u {
                    value[component] = unpack2x16float(clst_read_u32(base_word, at + (component & ~1u) * 2u))[component & 1u];
                } else if format == 10u {
                    value[component] = f32(clst_read_byte(base_word, at + component)) / 255.0;
                } else if format == 11u {
                    value[component] = max(f32(bitcast<i32>(clst_read_byte(base_word, at + component) << 24u)) / 127.0, -1.0);
                } else if format == 9u {
                    value[component] = f32(bitcast<i32>(clst_read_u32(base_word, at + component * 4u)));
                } else if format == 6u || format == 7u || format == 8u {
                    value[component] = f32(clst_read_u32(base_word, at + component * 4u));
                } else {
                    value[component] = bitcast<f32>(clst_read_u32(base_word, at + component * 4u));
                }
            }
            return value;
        }
        cursor = data_begin + data_len;
    }
    return vec4<f32>(0.0);
}

// Pulls one triangle corner. `corner` must be 0..2; invalid residency,
// generation, material, framing, counts, or offsets return a zeroed invalid
// record before any address derived from the untrusted records is read.
fn clst_pull_triangle_vertex(cluster_index: u32, triangle_index: u32, corner: u32,
    requested_location: u32) -> ClstPulledVertex {
    if corner >= 3u || cluster_index >= clst_pull_params.cluster_count
        || clst_pull_params.page_count == 0u { return clst_fail(); }
    let cluster = clst_clusters[cluster_index];
    if cluster.page_index >= clst_pull_params.page_count
        || cluster.vertex_count == 0u || cluster.vertex_count > 64u
        || cluster.triangle_count == 0u || cluster.triangle_count > 128u
        || triangle_index >= cluster.triangle_count || cluster.payload_size < 34u {
        return clst_fail();
    }
    let page = clst_pages[cluster.page_index];
    if (page.flags & 1u) == 0u || page.physical_page == CLST_NON_RESIDENT
        || any(page.generation != clst_pull_params.expected_generation)
        || page.byte_len == 0u || cluster.payload_offset > page.byte_len
        || cluster.payload_size > page.byte_len - cluster.payload_offset
        || page.byte_offset & 3u != 0u || page.byte_len > clst_pull_params.page_storage_words * 4u
        || page.byte_offset / 4u > clst_pull_params.page_storage_words
        || page.byte_len / 4u > clst_pull_params.page_storage_words - page.byte_offset / 4u {
        return clst_fail();
    }
    if cluster.material_index >= clst_pull_params.material_count
        || clst_pull_params.material_count > arrayLength(&clst_material_headers) / CLST_MATERIAL_ROW_WORDS {
        return clst_fail();
    }
    let material_at = cluster.material_index * CLST_MATERIAL_ROW_WORDS;
    if clst_material_headers[material_at] != cluster.material_generation
        || clst_material_headers[material_at + 13u] != 1u {
        return clst_fail();
    }
    let base_word = page.byte_offset / 4u;
    let payload = cluster.payload_offset;
    let payload_end = payload + cluster.payload_size;
    if clst_read_u32(base_word, payload) != CLST_MAGIC { return clst_fail(); }
    let version_and_vertices = clst_read_u32(base_word, payload + 4u);
    let version = version_and_vertices & 65535u;
    let vertex_count = version_and_vertices >> 16u;
    let triangle_count = clst_read_u16(base_word, payload + 8u);
    if (version != CLST_VERSION_V2 && version != CLST_VERSION_V3)
        || vertex_count != cluster.vertex_count || triangle_count != cluster.triangle_count {
        return clst_fail();
    }
    let position_bytes = vertex_count * 6u;
    let index_start = payload + 34u + position_bytes;
    let geometry_end = index_start + triangle_count * 6u;
    if geometry_end > payload_end { return clst_fail(); }
    let index = clst_read_u16(base_word, index_start + triangle_index * 6u + corner * 2u);
    if index >= vertex_count { return clst_fail(); }
    let q_at = payload + 34u + index * 6u;
    let q = vec3<f32>(f32(clst_read_u16(base_word, q_at)),
        f32(clst_read_u16(base_word, q_at + 2u)), f32(clst_read_u16(base_word, q_at + 4u))) / 65535.0;
    let bounds_min = vec3<f32>(bitcast<f32>(clst_read_u32(base_word, payload + 10u)),
        bitcast<f32>(clst_read_u32(base_word, payload + 14u)), bitcast<f32>(clst_read_u32(base_word, payload + 18u)));
    let bounds_max = vec3<f32>(bitcast<f32>(clst_read_u32(base_word, payload + 22u)),
        bitcast<f32>(clst_read_u32(base_word, payload + 26u)), bitcast<f32>(clst_read_u32(base_word, payload + 30u)));
    if any(bounds_max < bounds_min) { return clst_fail(); }
    let position = bounds_min + (bounds_max - bounds_min) * q;
    var attribute = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if requested_location != clst_read_u32(base_word, (geometry_end + 3u) & ~3u) {
        if version != CLST_VERSION_V3 { return clst_fail(); }
        let footer = (geometry_end + 3u) & ~3u;
        if footer + 8u > payload_end { return clst_fail(); }
        attribute = clst_decode_attribute(base_word, footer, payload_end, vertex_count, index, requested_location);
    }
    var result: ClstPulledVertex;
    result.position = vec4<f32>(position, 1.0);
    result.attribute = attribute;
    result.material_index = cluster.material_index;
    result.material_generation = cluster.material_generation;
    result.valid = 1u;
    result.pad = 0u;
    return result;
}

@compute @workgroup_size(1)
fn clst_pull_contract_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= 3u { return; }
    clst_pull_output[id.x] = clst_pull_triangle_vertex(clst_pull_params.cluster_index,
        clst_pull_params.triangle_index, id.x, clst_pull_params.requested_location);
}
