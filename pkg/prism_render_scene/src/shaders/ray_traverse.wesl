//! Bottom-level `BVH` traversal kernel: the real-device twin of the `CPU`
//! golden walk in `prism_render_architecture::ray_scene::gpu_layout`
//! (`GpuBvhBuffers::closest_hit` / `GpuBvhBuffers::any_hit`).
//!
//! The kernel binds the exact packed buffers that `GpuBvhBuffers::from_bvh`
//! emits — a depth-first `nodes` array (`NODE_WORDS = 12` `u32` per node) and a
//! leaf-contiguous `triangles` array (`TRIANGLE_WORDS = 12` `u32` per triangle),
//! every `f32` field stored as its `to_bits` pattern — and reproduces the walk
//! bit-for-bit: the same reciprocal-slab rejection
//! (`prism_render_architecture::ray_scene::traversal::slab_interval`), the same
//! near/far child ordering by split-axis sign, the same running `t_max` shrink,
//! and the same double-sided single-precision Möller–Trumbore test
//! (`intersect_triangle`, `EPS = 1e-8`). A green
//! `raytrace::shader_tests` proves this source parses and type-checks exactly as
//! it will on device; `raytrace::gpu_tests` diffs the dispatched hits against
//! the `CPU` golden `closest_hit`.
//!
//! Buffer contract (kept byte-identical to `abi.rs`, which pins these strides to
//! the golden `NODE_WORDS` / `TRIANGLE_WORDS` constants):
//!
//! * `nodes`     : `NODE_WORDS` (12) `u32` per node — `min.xyz` (0..3),
//!                 `max.xyz` (3..6), `first_primitive` (6), `second_child` (7),
//!                 `primitive_count` (8), `axis` (9), pad (10..12).
//! * `triangles` : `TRIANGLE_WORDS` (12) `u32` per triangle — `v0.xyz` (0..3),
//!                 `v1.xyz` (3..6), `v2.xyz` (6..9), `primitive` (9), pad.
//! * `rays`      : `RAY_WORDS` (8) `u32` per ray — `origin.xyz` (0..3),
//!                 `t_min` (3), `dir.xyz` (4..7), `t_max` (7). The host packs
//!                 already-normalized rays (`Ray::new` clamp applied: `t_min >=
//!                 0`, `t_max >= t_min`, non-normalized `dir` allowed), so the
//!                 kernel reads them directly and derives `inv_dir = 1.0 / dir`.
//! * `hits`      : `HIT_WORDS` (4) `u32` per ray — `t` (0), `u` (1), `v` (2),
//!                 `primitive` (3). A miss writes `primitive = 0xFFFFFFFF` and
//!                 `t = +inf` so a consumer branches on the sentinel primitive.

const NODE_WORDS: u32 = 12u;
const TRIANGLE_WORDS: u32 = 12u;
const RAY_WORDS: u32 = 8u;
const HIT_WORDS: u32 = 4u;

/// Traversal stack depth; matches the fixed `[0u32; 64]` short stack the `CPU`
/// golden walk uses, so a pathological deep `BVH` prunes identically (a far
/// child is dropped once the stack is full in both walks).
const STACK_SIZE: u32 = 64u;

/// Determinant epsilon of the Möller–Trumbore test; mirrors `EPS = 1e-8` in the
/// golden `intersect_triangle`.
const TRI_EPS: f32 = 1e-8;

/// Sentinel primitive id written on a miss; `0xFFFFFFFF`.
const MISS_PRIMITIVE: u32 = 4294967295u;

/// `f32` `+inf` bit pattern, written to the `t` slot of a miss.
const POSITIVE_INF_BITS: u32 = 2139095040u;

/// `mode == 0` runs the nearest-hit walk (`closest_hit`); `mode == 1` runs the
/// early-out occlusion walk (`any_hit`).
struct RayTraverseParams {
    ray_count: u32,
    mode: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read> rays: array<u32>;
@group(0) @binding(3) var<storage, read_write> hits: array<u32>;
@group(0) @binding(4) var<uniform> params: RayTraverseParams;

/// A ray with precomputed reciprocal direction and its valid `t` interval.
struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
}

/// Result of a single triangle test: `hit` gates the barycentric payload.
struct TriHit {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
}

/// Decodes packed ray `index`, deriving `inv_dir = 1.0 / dir` exactly as
/// `Ray::new`'s `reciprocal` (a zero component yields `+/-inf`, which the slab
/// test's `min`/`max` ordering absorbs).
fn load_ray(index: u32) -> Ray {
    let base = index * RAY_WORDS;
    let origin = vec3<f32>(
        bitcast<f32>(rays[base + 0u]),
        bitcast<f32>(rays[base + 1u]),
        bitcast<f32>(rays[base + 2u]),
    );
    let t_min = bitcast<f32>(rays[base + 3u]);
    let dir = vec3<f32>(
        bitcast<f32>(rays[base + 4u]),
        bitcast<f32>(rays[base + 5u]),
        bitcast<f32>(rays[base + 6u]),
    );
    let t_max = bitcast<f32>(rays[base + 7u]);
    let inv_dir = vec3<f32>(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    return Ray(origin, dir, inv_dir, t_min, t_max);
}

/// Decoded `min.xyz` of packed node `ni`.
fn node_min(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(nodes[base + 0u]),
        bitcast<f32>(nodes[base + 1u]),
        bitcast<f32>(nodes[base + 2u]),
    );
}

/// Decoded `max.xyz` of packed node `ni`.
fn node_max(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(nodes[base + 3u]),
        bitcast<f32>(nodes[base + 4u]),
        bitcast<f32>(nodes[base + 5u]),
    );
}

/// Split-axis direction component; unrolled to avoid dynamic vector indexing.
fn dir_axis(dir: vec3<f32>, axis: u32) -> f32 {
    if axis == 0u {
        return dir.x;
    }
    if axis == 1u {
        return dir.y;
    }
    return dir.z;
}

/// Reciprocal-slab test clipped to `[t_lo, t_hi]`; mirrors `slab_interval`,
/// returning only the boolean the golden walk consumes (`aabb_interval(..).is_some()`).
fn slab_hit(
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    bmin: vec3<f32>,
    bmax: vec3<f32>,
    t_lo: f32,
    t_hi: f32,
) -> bool {
    var tmin = t_lo;
    var tmax = t_hi;

    let tx0 = (bmin.x - origin.x) * inv_dir.x;
    let tx1 = (bmax.x - origin.x) * inv_dir.x;
    let nx = min(tx0, tx1);
    let fx = max(tx0, tx1);
    if nx > tmin { tmin = nx; }
    if fx < tmax { tmax = fx; }
    if tmin > tmax { return false; }

    let ty0 = (bmin.y - origin.y) * inv_dir.y;
    let ty1 = (bmax.y - origin.y) * inv_dir.y;
    let ny = min(ty0, ty1);
    let fy = max(ty0, ty1);
    if ny > tmin { tmin = ny; }
    if fy < tmax { tmax = fy; }
    if tmin > tmax { return false; }

    let tz0 = (bmin.z - origin.z) * inv_dir.z;
    let tz1 = (bmax.z - origin.z) * inv_dir.z;
    let nz = min(tz0, tz1);
    let fz = max(tz0, tz1);
    if nz > tmin { tmin = nz; }
    if fz < tmax { tmax = fz; }
    if tmin > tmax { return false; }

    return true;
}

/// Double-sided single-precision Möller–Trumbore test; mirrors the golden
/// `intersect_triangle` (no back-face cull, `EPS = 1e-8`, `u`/`v` are the
/// barycentric weights of `v1`/`v2`).
fn intersect_triangle(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> TriHit {
    var out = TriHit(false, 0.0, 0.0, 0.0);
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(ray.dir, e2);
    let det = dot(e1, p);
    if abs(det) < TRI_EPS {
        return out;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - v0;
    let u = dot(tvec, p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return out;
    }
    let q = cross(tvec, e1);
    let v = dot(ray.dir, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return out;
    }
    let t = dot(e2, q) * inv_det;
    if t < ray.t_min || t > ray.t_max {
        return out;
    }
    out.hit = true;
    out.t = t;
    out.u = u;
    out.v = v;
    return out;
}

/// Decodes leaf triangle `pi` into its three vertices.
fn triangle_v0(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(triangles[base + 0u]),
        bitcast<f32>(triangles[base + 1u]),
        bitcast<f32>(triangles[base + 2u]),
    );
}

fn triangle_v1(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(triangles[base + 3u]),
        bitcast<f32>(triangles[base + 4u]),
        bitcast<f32>(triangles[base + 5u]),
    );
}

fn triangle_v2(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(triangles[base + 6u]),
        bitcast<f32>(triangles[base + 7u]),
        bitcast<f32>(triangles[base + 8u]),
    );
}

/// Writes a hit record for ray `ri`.
fn write_hit(ri: u32, t: f32, u: f32, v: f32, primitive: u32) {
    let base = ri * HIT_WORDS;
    hits[base + 0u] = bitcast<u32>(t);
    hits[base + 1u] = bitcast<u32>(u);
    hits[base + 2u] = bitcast<u32>(v);
    hits[base + 3u] = primitive;
}

/// Writes the miss sentinel for ray `ri` (`primitive = 0xFFFFFFFF`, `t = +inf`).
fn write_miss(ri: u32) {
    let base = ri * HIT_WORDS;
    hits[base + 0u] = POSITIVE_INF_BITS;
    hits[base + 1u] = 0u;
    hits[base + 2u] = 0u;
    hits[base + 3u] = MISS_PRIMITIVE;
}

@compute @workgroup_size(64)
fn ray_traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if ri >= params.ray_count {
        return;
    }

    // Empty `BVH`: no nodes to walk, every ray misses (mirrors the
    // `self.nodes.is_empty()` guard in both golden walks).
    if arrayLength(&nodes) < NODE_WORDS {
        write_miss(ri);
        return;
    }

    var ray = load_ray(ri);
    let any_mode = params.mode == 1u;

    var stack: array<u32, 64>;
    var sp = 0u;
    var node_index = 0u;

    var best_hit = false;
    var best_t = 0.0;
    var best_u = 0.0;
    var best_v = 0.0;
    var best_prim = 0u;

    loop {
        let base = node_index * NODE_WORDS;
        let bmin = node_min(node_index);
        let bmax = node_max(node_index);
        if slab_hit(ray.origin, ray.inv_dir, bmin, bmax, ray.t_min, ray.t_max) {
            let primitive_count = nodes[base + 8u];
            if primitive_count > 0u {
                let start = nodes[base + 6u];
                let end = start + primitive_count;
                var pi = start;
                loop {
                    if pi >= end {
                        break;
                    }
                    let th = intersect_triangle(
                        ray,
                        triangle_v0(pi),
                        triangle_v1(pi),
                        triangle_v2(pi),
                    );
                    if th.hit {
                        let prim = triangles[pi * TRIANGLE_WORDS + 9u];
                        if any_mode {
                            // Occlusion query: first hit blocks the ray.
                            write_hit(ri, th.t, th.u, th.v, prim);
                            return;
                        }
                        // Nearest query: shrink the interval and record the hit
                        // (mirrors `ray = Ray::new(.., t)` + `best = Some(..)`).
                        ray.t_max = th.t;
                        best_hit = true;
                        best_t = th.t;
                        best_u = th.u;
                        best_v = th.v;
                        best_prim = prim;
                    }
                    pi = pi + 1u;
                }
                // Leaf processed: pop.
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                node_index = stack[sp];
            } else {
                let first_child = node_index + 1u;
                let second_child = nodes[base + 7u];
                if any_mode {
                    // `any_hit` always pushes `second_child`, descends `first_child`.
                    if sp < STACK_SIZE {
                        stack[sp] = second_child;
                        sp = sp + 1u;
                    }
                    node_index = first_child;
                } else {
                    // `closest_hit` visits the near-side child first by split-axis sign.
                    let axis = nodes[base + 9u];
                    let neg = dir_axis(ray.dir, axis) < 0.0;
                    var near = first_child;
                    var far = second_child;
                    if neg {
                        near = second_child;
                        far = first_child;
                    }
                    if sp < STACK_SIZE {
                        stack[sp] = far;
                        sp = sp + 1u;
                    }
                    node_index = near;
                }
            }
        } else {
            // Slab miss: pop.
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            node_index = stack[sp];
        }
    }

    if best_hit {
        write_hit(ri, best_t, best_u, best_v, best_prim);
    } else {
        write_miss(ri);
    }
}
