// Software stackless BVH-on-compute traversal kernel (GPU twin of the CPU
// golden). This is the authoritative software fallback traversal path for the
// Prism ray_scene backend: when hardware ray tracing (ray_query /
// acceleration_structure, DXR/Vulkan RT, Lumen-style HWRT) is unavailable, the
// renderer walks the same packed BVH on compute that the CPU reference walks.
//
// It mirrors ray_scene::traversal_stackless_gpu_layout::GpuStacklessBvh
// closest-hit walk word-for-word: a single ray cursor over the escape-threaded
// (stackless) BVH, descending to the left child (node + 1) on a bounds hit and
// jumping to the escape index on a miss or after processing a leaf. The slab
// test and the single-precision Moller-Trumbore triangle test reproduce the f32
// arithmetic in ray_scene::traversal.
//
// Buffer ABI:
//   bvh_nodes      packed BVH nodes,  12 u32 words each (gpu_layout::NODE_WORDS)
//   bvh_triangles  packed triangles,  12 u32 words each (gpu_layout::TRIANGLE_WORDS)
//   bvh_escape     one escape index per node (ESCAPE_SENTINEL = u32::MAX)
//   trace_rays     input ray records,  8 u32 words each (gpu_trace_io::TRACE_RAY_WORDS)
//   trace_hits     output hit records, 8 u32 words each (gpu_trace_io::TRACE_HIT_WORDS)

const ESCAPE_SENTINEL: u32 = 4294967295u;
const NODE_WORDS: u32 = 12u;
const TRIANGLE_WORDS: u32 = 12u;
const RAY_WORDS: u32 = 8u;
const HIT_WORDS: u32 = 8u;
const TRI_EPS: f32 = 1e-8;
const HIT_FLAG_MISS: u32 = 0u;
const HIT_FLAG_HIT: u32 = 1u;

@group(0) @binding(0) var<storage, read> bvh_nodes: array<u32>;
@group(0) @binding(1) var<storage, read> bvh_triangles: array<u32>;
@group(0) @binding(2) var<storage, read> bvh_escape: array<u32>;
@group(0) @binding(3) var<storage, read> trace_rays: array<u32>;
@group(0) @binding(4) var<storage, read_write> trace_hits: array<u32>;

// Reads three consecutive bitcast<f32> words from the node buffer.
fn node_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(bvh_nodes[base + 0u]),
        bitcast<f32>(bvh_nodes[base + 1u]),
        bitcast<f32>(bvh_nodes[base + 2u]),
    );
}

// Reads three consecutive bitcast<f32> words from the triangle buffer.
fn triangle_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(bvh_triangles[base + 0u]),
        bitcast<f32>(bvh_triangles[base + 1u]),
        bitcast<f32>(bvh_triangles[base + 2u]),
    );
}

// Slab test clipped to [t_lo, t_hi]; mirrors ray_scene::traversal::slab_interval
// (only "does the ray enter the box within the interval" is needed by the walk).
// The branch picks (near, far) with the same `t0 <= t1` comparison as the CPU
// reference so NaN handling matches bit-for-bit.
fn slab_hit(
    bmin: vec3<f32>,
    bmax: vec3<f32>,
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    t_lo: f32,
    t_hi: f32,
) -> bool {
    var tmin = t_lo;
    var tmax = t_hi;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let t0 = (bmin[axis] - origin[axis]) * inv_dir[axis];
        let t1 = (bmax[axis] - origin[axis]) * inv_dir[axis];
        var near = t1;
        var far = t0;
        if (t0 <= t1) {
            near = t0;
            far = t1;
        }
        if (near > tmin) {
            tmin = near;
        }
        if (far < tmax) {
            tmax = far;
        }
        if (tmin > tmax) {
            return false;
        }
    }
    return true;
}

struct TriHit {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
}

// Single-precision Moller-Trumbore test; mirrors
// ray_scene::traversal::intersect_triangle (both faces tested, EPS = 1e-8).
fn intersect_triangle(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
) -> TriHit {
    var result = TriHit(false, 0.0, 0.0, 0.0);
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if (abs(det) < TRI_EPS) {
        return result;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - v0;
    let u = dot(tvec, p) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return result;
    }
    let q = cross(tvec, e1);
    let v = dot(dir, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }
    let t = dot(e2, q) * inv_det;
    if (t < t_min || t > t_max) {
        return result;
    }
    return TriHit(true, t, u, v);
}

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
    // t_max shrinks as closer hits commit, exactly as Ray::new rebuilds the
    // interval on the CPU reference.
    var t_max = bitcast<f32>(trace_rays[rbase + 7u]);
    let inv_dir = vec3<f32>(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

    var best_hit = false;
    var best_t = 0.0;
    var best_u = 0.0;
    var best_v = 0.0;
    var best_primitive = 0u;

    let node_total = arrayLength(&bvh_nodes) / NODE_WORDS;
    if (node_total != 0u) {
        var node_index = 0u;
        loop {
            if (node_index == ESCAPE_SENTINEL) {
                break;
            }
            let nbase = node_index * NODE_WORDS;
            let bmin = node_vec3(nbase);
            let bmax = node_vec3(nbase + 3u);
            if (slab_hit(bmin, bmax, origin, inv_dir, t_min, t_max)) {
                let primitive_count = bvh_nodes[nbase + 8u];
                if (primitive_count > 0u) {
                    let start = bvh_nodes[nbase + 6u];
                    let end = start + primitive_count;
                    for (var pi = start; pi < end; pi = pi + 1u) {
                        let tbase = pi * TRIANGLE_WORDS;
                        let v0 = triangle_vec3(tbase);
                        let v1 = triangle_vec3(tbase + 3u);
                        let v2 = triangle_vec3(tbase + 6u);
                        let th = intersect_triangle(origin, dir, t_min, t_max, v0, v1, v2);
                        if (th.hit) {
                            t_max = th.t;
                            best_hit = true;
                            best_t = th.t;
                            best_u = th.u;
                            best_v = th.v;
                            best_primitive = bvh_triangles[tbase + 9u];
                        }
                    }
                    node_index = bvh_escape[node_index];
                } else {
                    node_index = node_index + 1u;
                }
            } else {
                node_index = bvh_escape[node_index];
            }
        }
    }

    let hbase = ray_index * HIT_WORDS;
    if (best_hit) {
        trace_hits[hbase + 0u] = bitcast<u32>(best_t);
        trace_hits[hbase + 1u] = bitcast<u32>(best_u);
        trace_hits[hbase + 2u] = bitcast<u32>(best_v);
        trace_hits[hbase + 3u] = best_primitive;
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
