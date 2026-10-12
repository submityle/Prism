struct Vertex {
    position: vec3<f32>,
    _position_padding: f32,
    normal: vec3<f32>,
    _normal_padding: f32,
    tangent: vec4<f32>,
    uv: vec2<f32>,
    flags: u32,
    _padding: u32,
};

struct ShadeJob {
    extent: vec2<u32>,
    atlas_origin: vec2<u32>,
    page_index: u32,
    generation: u32,
    epoch: u32,
    _proof_padding: u32,
    vertex_indices: vec3<u32>,
    _vertex_padding: u32,
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    light_direction: vec3<f32>,
    _light_direction_padding: f32,
    light_radiance: vec3<f32>,
    _light_radiance_padding: f32,
};

struct PageEntry {
    atlas_x: u32,
    atlas_y: u32,
    generation: u32,
    epoch: u32,
};

@group(0) @binding(0) var atlas: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<storage, read> jobs: array<ShadeJob>;
@group(0) @binding(2) var<storage, read> page_table: array<PageEntry>;
@group(0) @binding(3) var<storage, read> vertices: array<Vertex>;

fn barycentrics_for_uv(uv: vec2<f32>, uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>) -> vec3<f32> {
    let e1 = uv1 - uv0;
    let e2 = uv2 - uv0;
    let p = uv - uv0;
    let determinant = e1.x * e2.y - e1.y * e2.x;
    let b1 = (p.x * e2.y - p.y * e2.x) / determinant;
    let b2 = (e1.x * p.y - e1.y * p.x) / determinant;
    return vec3<f32>(1.0 - b1 - b2, b1, b2);
}

@compute @workgroup_size(8, 8, 1)
fn shade_pages(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    // The z dispatch dimension selects a job. x/y are 8x8 tiles, so charts
    // larger than one workgroup remain fully covered without baking a small
    // maximum extent into the CPU chart contract.
    let job = jobs[group.z];
    let texel = group.xy * vec2<u32>(8u, 8u) + local.xy;
    if (texel.x >= job.extent.x || texel.y >= job.extent.y) {
        return;
    }
    let entry = page_table[job.page_index];
    if (entry.atlas_x != job.atlas_origin.x || entry.atlas_y != job.atlas_origin.y ||
        entry.generation != job.generation || entry.epoch != job.epoch) {
        return;
    }

    let atlas_size = textureDimensions(atlas);
    if (job.atlas_origin.x >= atlas_size.x || job.atlas_origin.y >= atlas_size.y ||
        texel.x >= atlas_size.x - job.atlas_origin.x ||
        texel.y >= atlas_size.y - job.atlas_origin.y) {
        return;
    }

    let v0 = vertices[job.vertex_indices.x];
    let v1 = vertices[job.vertex_indices.y];
    let v2 = vertices[job.vertex_indices.z];
    let extent = vec2<f32>(job.extent);
    let uv = job.uv_min + (vec2<f32>(texel) + vec2<f32>(0.5)) / extent * (job.uv_max - job.uv_min);
    let bary = barycentrics_for_uv(uv, v0.uv, v1.uv, v2.uv);
    if (any(bary < vec3<f32>(0.0)) || any(bary > vec3<f32>(1.0))) {
        return;
    }

    let interpolated_normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    let normal_length_squared = dot(interpolated_normal, interpolated_normal);
    let light_length_squared = dot(job.light_direction, job.light_direction);
    if (normal_length_squared <= 1.0e-12 || light_length_squared <= 1.0e-12) {
        return;
    }
    let normal = normalize(interpolated_normal);
    let light_direction = normalize(job.light_direction);
    let cosine = max(dot(normal, light_direction), 0.0);
    let incoming = job.light_radiance * cosine;
    let pixel = vec2<i32>(i32(job.atlas_origin.x + texel.x), i32(job.atlas_origin.y + texel.y));
    textureStore(atlas, pixel, vec4<f32>(incoming, 1.0));
}
