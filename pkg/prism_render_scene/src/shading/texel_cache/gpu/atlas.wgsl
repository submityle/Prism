struct Job {
    width: u32,
    height: u32,
    atlas_x: u32,
    atlas_y: u32,
    page_index: u32,
    generation: u32,
    epoch: u32,
    _padding: u32,
};

struct PageEntry {
    atlas_x: u32,
    atlas_y: u32,
    generation: u32,
    epoch: u32,
};

@group(0) @binding(0) var atlas: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read> page_table: array<PageEntry>;

@compute @workgroup_size(4, 4, 1)
fn shade_pages(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let job_index = group.x;
    let job = jobs[job_index];
    if (local.x >= job.width || local.y >= job.height) {
        return;
    }

    let entry = page_table[job.page_index];
    if (entry.atlas_x != job.atlas_x || entry.atlas_y != job.atlas_y ||
        entry.generation != job.generation || entry.epoch != job.epoch) {
        return;
    }

    let atlas_size = textureDimensions(atlas);
    if (job.atlas_x >= atlas_size.x || job.atlas_y >= atlas_size.y ||
        local.x >= atlas_size.x - job.atlas_x || local.y >= atlas_size.y - job.atlas_y) {
        return;
    }
    let pixel = vec2<i32>(i32(job.atlas_x + local.x), i32(job.atlas_y + local.y));
    // This is the GPU page-write ABI contract, not a material-shading result:
    // the first channel identifies the source job, the second carries the
    // input-local x coordinate, and the last two channels preserve the page
    // table's generation/epoch proof for readback validation.
    textureStore(atlas, pixel, vec4<u32>(job.page_index, local.x, job.generation, job.epoch));
}
