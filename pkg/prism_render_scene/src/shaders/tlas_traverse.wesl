//! Top-level `TLAS` traversal kernel: the real-device twin of the `CPU` golden
//! walk in `prism_render_architecture::ray_scene::gpu_layout`
//! (`GpuTlasBuffers::closest_hit` / `GpuTlasBuffers::any_hit` over the shared
//! `GpuBlasPool`).
//!
//! The kernel binds the exact packed buffers `GpuTlasBuffers::from_tlas` and
//! `GpuBlasPool::from_blases` emit and reproduces the two-level walk bit-for-bit:
//! a top-level `nodes` walk selects candidate instances; each leaf instance
//! pulls its packed `world_to_object` affine (three linear columns + a
//! translation), transforms the world-space ray into the instance's object
//! space *without renormalizing the direction* (so the ray parameter `t` is
//! preserved), then descends the referenced `BLAS` slice of the shared pool with
//! the same reciprocal-slab rejection, near/far split-axis ordering, running
//! `t_max` shrink and double-sided Möller–Trumbore test the bottom-level kernel
//! uses. The nearest walk shrinks `best_t` across every instance exactly like
//! the in-memory `TLAS`; the any-hit walk returns on the first confirmed
//! occluder and never shrinks its interval.
//!
//! A green `raytrace::shader_tests` proves this source parses and type-checks
//! exactly as it will on device; `raytrace::gpu_tests` diffs the dispatched hits
//! against the `CPU` golden `GpuTlasBuffers::closest_hit` / `any_hit`.
//!
//! Buffer contract (kept byte-identical to `abi.rs`, which pins these strides to
//! the golden `NODE_WORDS` / `TRIANGLE_WORDS` / `INSTANCE_WORDS` /
//! `BLAS_OFFSET_WORDS` constants):
//!
//! * `tlas_nodes`   : `NODE_WORDS` (12) `u32` per node — `min.xyz` (0..3),
//!                    `max.xyz` (3..6), `first_primitive` (6, first instance),
//!                    `second_child` (7), `primitive_count` (8, instance count),
//!                    `axis` (9), pad (10..12).
//! * `instances`    : `INSTANCE_WORDS` (16) `u32` per instance —
//!                    `world_to_object` linear columns `c0.xyz` (0..3),
//!                    `c1.xyz` (3..6), `c2.xyz` (6..9), translation (9..12),
//!                    `blas_index` (12), `instance_id` (13), pad (14..16).
//! * `pool_nodes`   : `NODE_WORDS` (12) `u32` per `BLAS` node — child indices are
//!                    `BLAS`-local (rebased by `node_base` at read time), leaf
//!                    `first_primitive` (6) is a global pool triangle index.
//! * `pool_triangles`: `TRIANGLE_WORDS` (12) `u32` per triangle — `v0.xyz`
//!                    (0..3), `v1.xyz` (3..6), `v2.xyz` (6..9), `primitive` (9).
//! * `pool_offsets` : `BLAS_OFFSET_WORDS` (4) `u32` per `BLAS` — `node_base` (0),
//!                    `node_count` (1), `triangle_base` (2), `triangle_count`
//!                    (3).
//! * `rays`         : `RAY_WORDS` (8) `u32` per ray — `origin.xyz` (0..3),
//!                    `t_min` (3), `dir.xyz` (4..7), `t_max` (7).
//! * `hits`         : `TLAS_HIT_WORDS` (8) `u32` per ray — `t` (0), `u` (1),
//!                    `v` (2), `primitive` (3), `instance_id` (4),
//!                    `instance_index` (5), pad (6..8). A miss writes
//!                    `primitive = 0xFFFFFFFF` and `t = +inf`.

const NODE_WORDS: u32 = 12u;
const TRIANGLE_WORDS: u32 = 12u;
const INSTANCE_WORDS: u32 = 16u;
const BLAS_OFFSET_WORDS: u32 = 4u;
const RAY_WORDS: u32 = 8u;
const TLAS_HIT_WORDS: u32 = 8u;

/// Traversal stack depth; matches the fixed `[0u32; 64]` short stacks the `CPU`
/// golden top-level and bottom-level walks use, so a pathological deep tree
/// prunes identically in both.
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

@group(0) @binding(0) var<storage, read> tlas_nodes: array<u32>;
@group(0) @binding(1) var<storage, read> instances: array<u32>;
@group(0) @binding(2) var<storage, read> pool_nodes: array<u32>;
@group(0) @binding(3) var<storage, read> pool_triangles: array<u32>;
@group(0) @binding(4) var<storage, read> pool_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> rays: array<u32>;
@group(0) @binding(6) var<storage, read_write> hits: array<u32>;
@group(0) @binding(7) var<uniform> params: RayTraverseParams;

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

/// Nearest bottom-level hit against one pooled `BLAS`: `hit` gates the payload,
/// `primitive` is the stable triangle id.
struct BlasHit {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
    primitive: u32,
}

/// Decodes packed world-space ray `index`; `inv_dir = 1.0 / dir` exactly as
/// `Ray::new`'s `reciprocal` (a zero component yields `+/-inf`, which the slab
/// test's `min`/`max` ordering absorbs). The host stores non-normalized dirs.
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

/// Builds a ray in an instance's object space, mirroring `Ray::new(obj_origin,
/// obj_dir, t_min, t_max)`: the direction is carried through the affine linear
/// part *without renormalizing* so `t` is preserved, and `inv_dir` is recomputed
/// from the transformed direction.
fn make_ray(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> Ray {
    let inv_dir = vec3<f32>(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    return Ray(origin, dir, inv_dir, t_min, t_max);
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

/// Decoded `min.xyz` of packed top-level node `ni`.
fn tlas_node_min(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(tlas_nodes[base + 0u]),
        bitcast<f32>(tlas_nodes[base + 1u]),
        bitcast<f32>(tlas_nodes[base + 2u]),
    );
}

/// Decoded `max.xyz` of packed top-level node `ni`.
fn tlas_node_max(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(tlas_nodes[base + 3u]),
        bitcast<f32>(tlas_nodes[base + 4u]),
        bitcast<f32>(tlas_nodes[base + 5u]),
    );
}

/// Decoded `min.xyz` of packed pool `BLAS` node `ni` (absolute pool index).
fn pool_node_min(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(pool_nodes[base + 0u]),
        bitcast<f32>(pool_nodes[base + 1u]),
        bitcast<f32>(pool_nodes[base + 2u]),
    );
}

/// Decoded `max.xyz` of packed pool `BLAS` node `ni` (absolute pool index).
fn pool_node_max(ni: u32) -> vec3<f32> {
    let base = ni * NODE_WORDS;
    return vec3<f32>(
        bitcast<f32>(pool_nodes[base + 3u]),
        bitcast<f32>(pool_nodes[base + 4u]),
        bitcast<f32>(pool_nodes[base + 5u]),
    );
}

/// Decodes pooled triangle `pi` (global pool index) into its three vertices.
fn pool_triangle_v0(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(pool_triangles[base + 0u]),
        bitcast<f32>(pool_triangles[base + 1u]),
        bitcast<f32>(pool_triangles[base + 2u]),
    );
}

fn pool_triangle_v1(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(pool_triangles[base + 3u]),
        bitcast<f32>(pool_triangles[base + 4u]),
        bitcast<f32>(pool_triangles[base + 5u]),
    );
}

fn pool_triangle_v2(pi: u32) -> vec3<f32> {
    let base = pi * TRIANGLE_WORDS;
    return vec3<f32>(
        bitcast<f32>(pool_triangles[base + 6u]),
        bitcast<f32>(pool_triangles[base + 7u]),
        bitcast<f32>(pool_triangles[base + 8u]),
    );
}

/// Applies an instance's packed `world_to_object` linear part to a
/// direction/vector (ignores translation), mirroring `Affine3::transform_vector`
/// column-by-column: `c0 * v.x + c1 * v.y + c2 * v.z`.
fn transform_vector(ib: u32, v: vec3<f32>) -> vec3<f32> {
    let c0 = vec3<f32>(
        bitcast<f32>(instances[ib + 0u]),
        bitcast<f32>(instances[ib + 1u]),
        bitcast<f32>(instances[ib + 2u]),
    );
    let c1 = vec3<f32>(
        bitcast<f32>(instances[ib + 3u]),
        bitcast<f32>(instances[ib + 4u]),
        bitcast<f32>(instances[ib + 5u]),
    );
    let c2 = vec3<f32>(
        bitcast<f32>(instances[ib + 6u]),
        bitcast<f32>(instances[ib + 7u]),
        bitcast<f32>(instances[ib + 8u]),
    );
    return vec3<f32>(
        c0.x * v.x + c1.x * v.y + c2.x * v.z,
        c0.y * v.x + c1.y * v.y + c2.y * v.z,
        c0.z * v.x + c1.z * v.y + c2.z * v.z,
    );
}

/// Applies an instance's full `world_to_object` affine to a point, mirroring
/// `Affine3::transform_point` (linear part then translation).
fn transform_point(ib: u32, p: vec3<f32>) -> vec3<f32> {
    let linear = transform_vector(ib, p);
    let translation = vec3<f32>(
        bitcast<f32>(instances[ib + 9u]),
        bitcast<f32>(instances[ib + 10u]),
        bitcast<f32>(instances[ib + 11u]),
    );
    return linear + translation;
}

/// Nearest bottom-level intersection of the object-space `ray` against pooled
/// `BLAS` `blas`, walking the shared pool buffers directly by per-`BLAS` offset
/// — the real-device twin of `GpuBlasPool::closest_hit`. Node child indices are
/// `BLAS`-local (rebased by `node_base` at read); leaf `first_primitive` is a
/// global pool triangle index.
fn blas_closest_hit(blas: u32, ray: Ray) -> BlasHit {
    var out = BlasHit(false, 0.0, 0.0, 0.0, 0u);
    let o = blas * BLAS_OFFSET_WORDS;
    let node_base = pool_offsets[o + 0u];
    let node_count = pool_offsets[o + 1u];
    if node_count == 0u {
        return out;
    }

    var r = ray;
    var stack: array<u32, 64>;
    var sp = 0u;
    var node_index = 0u;

    loop {
        let base = (node_base + node_index) * NODE_WORDS;
        let bmin = pool_node_min(node_base + node_index);
        let bmax = pool_node_max(node_base + node_index);
        if slab_hit(r.origin, r.inv_dir, bmin, bmax, r.t_min, r.t_max) {
            let primitive_count = pool_nodes[base + 8u];
            if primitive_count > 0u {
                let start = pool_nodes[base + 6u];
                let end = start + primitive_count;
                var pi = start;
                loop {
                    if pi >= end {
                        break;
                    }
                    let th = intersect_triangle(
                        r,
                        pool_triangle_v0(pi),
                        pool_triangle_v1(pi),
                        pool_triangle_v2(pi),
                    );
                    if th.hit {
                        r.t_max = th.t;
                        out.hit = true;
                        out.t = th.t;
                        out.u = th.u;
                        out.v = th.v;
                        out.primitive = pool_triangles[pi * TRIANGLE_WORDS + 9u];
                    }
                    pi = pi + 1u;
                }
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                node_index = stack[sp];
            } else {
                let first_child = node_index + 1u;
                let second_child = pool_nodes[base + 7u];
                let axis = pool_nodes[base + 9u];
                let neg = dir_axis(r.dir, axis) < 0.0;
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
        } else {
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            node_index = stack[sp];
        }
    }

    return out;
}

/// True when *any* triangle of pooled `BLAS` `blas` intersects the object-space
/// `ray` inside its interval — the real-device twin of `GpuBlasPool::any_hit`.
fn blas_any_hit(blas: u32, ray: Ray) -> bool {
    let o = blas * BLAS_OFFSET_WORDS;
    let node_base = pool_offsets[o + 0u];
    let node_count = pool_offsets[o + 1u];
    if node_count == 0u {
        return false;
    }

    var stack: array<u32, 64>;
    var sp = 0u;
    var node_index = 0u;

    loop {
        let base = (node_base + node_index) * NODE_WORDS;
        let bmin = pool_node_min(node_base + node_index);
        let bmax = pool_node_max(node_base + node_index);
        if slab_hit(ray.origin, ray.inv_dir, bmin, bmax, ray.t_min, ray.t_max) {
            let primitive_count = pool_nodes[base + 8u];
            if primitive_count > 0u {
                let start = pool_nodes[base + 6u];
                let end = start + primitive_count;
                var pi = start;
                loop {
                    if pi >= end {
                        break;
                    }
                    let th = intersect_triangle(
                        ray,
                        pool_triangle_v0(pi),
                        pool_triangle_v1(pi),
                        pool_triangle_v2(pi),
                    );
                    if th.hit {
                        return true;
                    }
                    pi = pi + 1u;
                }
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                node_index = stack[sp];
            } else {
                // `any_hit` always pushes `second_child`, descends `first_child`.
                let first_child = node_index + 1u;
                let second_child = pool_nodes[base + 7u];
                if sp < STACK_SIZE {
                    stack[sp] = second_child;
                    sp = sp + 1u;
                }
                node_index = first_child;
            }
        } else {
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            node_index = stack[sp];
        }
    }

    return false;
}

/// Writes a nearest `TLAS` hit record for ray `ri`.
fn write_hit(
    ri: u32,
    t: f32,
    u: f32,
    v: f32,
    primitive: u32,
    instance_id: u32,
    instance_index: u32,
) {
    let base = ri * TLAS_HIT_WORDS;
    hits[base + 0u] = bitcast<u32>(t);
    hits[base + 1u] = bitcast<u32>(u);
    hits[base + 2u] = bitcast<u32>(v);
    hits[base + 3u] = primitive;
    hits[base + 4u] = instance_id;
    hits[base + 5u] = instance_index;
    hits[base + 6u] = 0u;
    hits[base + 7u] = 0u;
}

/// Writes the miss sentinel for ray `ri` (`primitive = 0xFFFFFFFF`, `t = +inf`).
fn write_miss(ri: u32) {
    let base = ri * TLAS_HIT_WORDS;
    hits[base + 0u] = POSITIVE_INF_BITS;
    hits[base + 1u] = 0u;
    hits[base + 2u] = 0u;
    hits[base + 3u] = MISS_PRIMITIVE;
    hits[base + 4u] = 0u;
    hits[base + 5u] = 0u;
    hits[base + 6u] = 0u;
    hits[base + 7u] = 0u;
}

/// Occlusion (any-hit) record: `t = 0`, `primitive` doubles as the blocked flag
/// (`0` blocked, `0xFFFFFFFF` clear), mirroring the boolean the golden
/// `GpuTlasBuffers::any_hit` returns.
fn write_blocked(ri: u32, blocked: bool) {
    let base = ri * TLAS_HIT_WORDS;
    hits[base + 0u] = 0u;
    hits[base + 1u] = 0u;
    hits[base + 2u] = 0u;
    if blocked {
        hits[base + 3u] = 0u;
    } else {
        hits[base + 3u] = MISS_PRIMITIVE;
    }
    hits[base + 4u] = 0u;
    hits[base + 5u] = 0u;
    hits[base + 6u] = 0u;
    hits[base + 7u] = 0u;
}

/// Nearest-hit top-level walk; mirrors `GpuTlasBuffers::closest_hit` over the
/// packed node buffer and shared pool, shrinking `best_t` across instances.
fn tlas_closest_hit(ri: u32, ray: Ray) {
    var best_hit = false;
    var best_t = ray.t_max;
    var best_u = 0.0;
    var best_v = 0.0;
    var best_prim = 0u;
    var best_instance_id = 0u;
    var best_instance_index = 0u;
    let t_min = ray.t_min;

    var stack: array<u32, 64>;
    var sp = 0u;
    var node_index = 0u;

    loop {
        let base = node_index * NODE_WORDS;
        let bmin = tlas_node_min(node_index);
        let bmax = tlas_node_max(node_index);
        if slab_hit(ray.origin, ray.inv_dir, bmin, bmax, t_min, best_t) {
            let primitive_count = tlas_nodes[base + 8u];
            if primitive_count > 0u {
                let start = tlas_nodes[base + 6u];
                let end = start + primitive_count;
                var inst_idx = start;
                loop {
                    if inst_idx >= end {
                        break;
                    }
                    let ib = inst_idx * INSTANCE_WORDS;
                    let obj_origin = transform_point(ib, ray.origin);
                    let obj_dir = transform_vector(ib, ray.dir);
                    let obj_ray = make_ray(obj_origin, obj_dir, t_min, best_t);
                    let blas = instances[ib + 12u];
                    let instance_id = instances[ib + 13u];
                    let bh = blas_closest_hit(blas, obj_ray);
                    if bh.hit && bh.t < best_t {
                        best_t = bh.t;
                        best_hit = true;
                        best_u = bh.u;
                        best_v = bh.v;
                        best_prim = bh.primitive;
                        best_instance_id = instance_id;
                        best_instance_index = inst_idx;
                    }
                    inst_idx = inst_idx + 1u;
                }
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                node_index = stack[sp];
            } else {
                let first_child = node_index + 1u;
                let second_child = tlas_nodes[base + 7u];
                let axis = tlas_nodes[base + 9u];
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
        } else {
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            node_index = stack[sp];
        }
    }

    if best_hit {
        write_hit(ri, best_t, best_u, best_v, best_prim, best_instance_id, best_instance_index);
    } else {
        write_miss(ri);
    }
}

/// Early-out occlusion top-level walk; mirrors `GpuTlasBuffers::any_hit`. Never
/// shrinks the interval and always pushes `second_child` before descending
/// `first_child`, returning on the first confirmed occluder.
fn tlas_any_hit(ri: u32, ray: Ray) {
    let t_min = ray.t_min;
    let t_max = ray.t_max;

    var stack: array<u32, 64>;
    var sp = 0u;
    var node_index = 0u;

    loop {
        let base = node_index * NODE_WORDS;
        let bmin = tlas_node_min(node_index);
        let bmax = tlas_node_max(node_index);
        if slab_hit(ray.origin, ray.inv_dir, bmin, bmax, t_min, t_max) {
            let primitive_count = tlas_nodes[base + 8u];
            if primitive_count > 0u {
                let start = tlas_nodes[base + 6u];
                let end = start + primitive_count;
                var inst_idx = start;
                loop {
                    if inst_idx >= end {
                        break;
                    }
                    let ib = inst_idx * INSTANCE_WORDS;
                    let obj_origin = transform_point(ib, ray.origin);
                    let obj_dir = transform_vector(ib, ray.dir);
                    let obj_ray = make_ray(obj_origin, obj_dir, t_min, t_max);
                    let blas = instances[ib + 12u];
                    if blas_any_hit(blas, obj_ray) {
                        write_blocked(ri, true);
                        return;
                    }
                    inst_idx = inst_idx + 1u;
                }
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                node_index = stack[sp];
            } else {
                let first_child = node_index + 1u;
                let second_child = tlas_nodes[base + 7u];
                if sp < STACK_SIZE {
                    stack[sp] = second_child;
                    sp = sp + 1u;
                }
                node_index = first_child;
            }
        } else {
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            node_index = stack[sp];
        }
    }

    write_blocked(ri, false);
}

@compute @workgroup_size(64)
fn tlas_traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if ri >= params.ray_count {
        return;
    }

    // Empty `TLAS`: no nodes to walk, every ray misses (mirrors the
    // `self.nodes.is_empty()` guard in both golden walks).
    if arrayLength(&tlas_nodes) < NODE_WORDS {
        if params.mode == 1u {
            write_blocked(ri, false);
        } else {
            write_miss(ri);
        }
        return;
    }

    let ray = load_ray(ri);
    if params.mode == 1u {
        tlas_any_hit(ri, ray);
    } else {
        tlas_closest_hit(ri, ray);
    }
}
