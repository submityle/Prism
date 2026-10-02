// Hardware ray_query traversal kernel: the first real rayQuery WESL shader in
// Prism ray_scene, the hardware-ray-tracing (HWRT) tier of the ray-tracing
// backend, aligned with UE5 Lumen's HWRT path (DXR / Vulkan ray tracing).
//
// Where software_bvh_trace.wesl walks a packed BVH on compute (the SWRT
// fallback), this kernel delegates traversal to the device's acceleration
// structure through the ray_query intrinsics
// (rayQueryInitialize / rayQueryProceed / rayQueryGetCommittedIntersection),
// so the TLAS/BLAS the driver built does the descent.
//
// It shares the ray_scene::gpu_trace_io record ABI with the software kernel:
//   trace_rays   input ray records,  8 u32 words each (gpu_trace_io::TRACE_RAY_WORDS)
//                layout: origin.xyz (0..3), t_min (3), dir.xyz (4..7), t_max (7)
//   trace_hits   output hit records, 8 u32 words each (gpu_trace_io::TRACE_HIT_WORDS)
//                layout: t (0), bary.x (1), bary.y (2), primitive (3),
//                        hit flag (4), padding (5..8)
// so a host can swap SWRT and HWRT kernels behind one buffer contract.
//
// Note on barycentrics: a committed hardware intersection reports barycentrics
// following the DXR / Vulkan ray-tracing convention; the host decodes words 1..3
// as those committed barycentrics (not the software Moller-Trumbore u/v), which
// is why the two kernels share the record *layout* but not a bit-exact result.
enable wgpu_ray_query;

const RAY_WORDS: u32 = 8u;
const HIT_WORDS: u32 = 8u;
const RAY_FLAG_NONE: u32 = 0u;
const RAY_CULL_MASK_ALL: u32 = 255u;
const HIT_FLAG_MISS: u32 = 0u;
const HIT_FLAG_HIT: u32 = 1u;

@group(0) @binding(0) var scene_tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read> trace_rays: array<u32>;
@group(0) @binding(2) var<storage, read_write> trace_hits: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn trace_closest_hit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_index = gid.x;
    let ray_total = arrayLength(&trace_rays) / RAY_WORDS;
    if (ray_index >= ray_total) {
        return;
    }

    let rbase = ray_index * RAY_WORDS;
    let origin = vec3<f32>(
        bitcast<f32>(trace_rays[rbase + 0u]),
        bitcast<f32>(trace_rays[rbase + 1u]),
        bitcast<f32>(trace_rays[rbase + 2u]),
    );
    let t_min = bitcast<f32>(trace_rays[rbase + 3u]);
    let dir = vec3<f32>(
        bitcast<f32>(trace_rays[rbase + 4u]),
        bitcast<f32>(trace_rays[rbase + 5u]),
        bitcast<f32>(trace_rays[rbase + 6u]),
    );
    let t_max = bitcast<f32>(trace_rays[rbase + 7u]);

    var rq: ray_query;
    let desc = RayDesc(RAY_FLAG_NONE, RAY_CULL_MASK_ALL, t_min, t_max, origin, dir);
    rayQueryInitialize(&rq, scene_tlas, desc);
    loop {
        if (!rayQueryProceed(&rq)) { break; }
    }
    let it = rayQueryGetCommittedIntersection(&rq);

    let hbase = ray_index * HIT_WORDS;
    if (it.kind != RAY_QUERY_INTERSECTION_NONE) {
        trace_hits[hbase + 0u] = bitcast<u32>(it.t);
        trace_hits[hbase + 1u] = bitcast<u32>(it.barycentrics.x);
        trace_hits[hbase + 2u] = bitcast<u32>(it.barycentrics.y);
        trace_hits[hbase + 3u] = it.primitive_index;
        trace_hits[hbase + 4u] = HIT_FLAG_HIT;
    } else {
        trace_hits[hbase + 0u] = 0u;
        trace_hits[hbase + 1u] = 0u;
        trace_hits[hbase + 2u] = 0u;
        trace_hits[hbase + 3u] = 0u;
        trace_hits[hbase + 4u] = HIT_FLAG_MISS;
    }
    trace_hits[hbase + 5u] = 0u;
    trace_hits[hbase + 6u] = 0u;
    trace_hits[hbase + 7u] = 0u;
}
