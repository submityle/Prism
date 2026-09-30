//! Watertight bottom-level `BVH` traversal kernel: the real-device twin of the
//! `CPU` golden watertight walk in
//! `prism_render_architecture::ray_scene::traversal`
//! (`Bvh::closest_hit_watertight` / `Bvh::any_hit_watertight`, built on
//! `intersect_triangle_watertight`).
//!
//! This shares the exact packed buffer contract of `ray_traverse.wesl` (the
//! `GpuBvhBuffers::from_bvh` layout: `NODE_WORDS = 12`, `TRIANGLE_WORDS = 12`,
//! `RAY_WORDS = 8`, `HIT_WORDS = 4`, every `f32` stored as its `to_bits`
//! pattern) and reproduces the identical acceleration walk — the same
//! reciprocal-slab rejection, near/far child ordering by split-axis sign, and
//! running `t_max` shrink. The only difference from `ray_traverse.wesl` is the
//! leaf primitive test: instead of the double-sided Möller–Trumbore test this
//! kernel runs the Woop 2013 watertight test, so a primary/reflection ray that
//! strikes a seam shared by two triangles is never lost between them.
//!
//! # Precision parity with the `CPU` golden
//! The golden `intersect_triangle_watertight` computes the three scaled edge
//! areas in `f32` and, whenever any is exactly `0.0`, recomputes all three in
//! `f64` — that exact refinement is what removes the single-precision rounding
//! that would otherwise let a shared edge fall between two triangles. `WGSL`
//! has no `f64`, so the boundary-zero refinement here uses Kahan's
//! fused-multiply-add compensated `a*b - c*d` (`diff_prod`), which recovers the
//! sign of a near-zero determinant to ~1.5 ULP — the `GPU`-portable stand-in
//! for the `f64` recompute. The common (non-boundary) path is plain `f32` and
//! matches the golden bit-for-bit; a future `raytrace::watertight_gpu_tests`
//! device diff must therefore treat only the boundary-zero fallback path with a
//! tolerance while asserting exact equality elsewhere.
//!
//! A green `raytrace::shader_tests` proves this source parses and type-checks
//! exactly as it will on device.

const NODE_WORDS: u32 = 12u;
const TRIANGLE_WORDS: u32 = 12u;
const RAY_WORDS: u32 = 8u;
const HIT_WORDS: u32 = 4u;

/// Traversal stack depth; matches the fixed `[0u32; 64]` short stack the `CPU`
/// golden walk uses, so a pathological deep `BVH` prunes identically.
const STACK_SIZE: u32 = 64u;

/// Sentinel primitive id written on a miss; `0xFFFFFFFF`.
const MISS_PRIMITIVE: u32 = 4294967295u;

/// `f32` `+inf` bit pattern, written to the `t` slot of a miss.
const POSITIVE_INF_BITS: u32 = 2139095040u;

/// `mode == 0` runs the nearest-hit walk (`closest_hit_watertight`); `mode == 1`
/// runs the early-out occlusion walk (`any_hit_watertight`).
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
/// `Ray::new`'s `reciprocal`.
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

/// Component `axis` of `v`; unrolled to avoid dynamic vector indexing (`WGSL`
/// forbids indexing a vector by a runtime value).
fn comp(v: vec3<f32>, axis: u32) -> f32 {
    if axis == 0u {
        return v.x;
    }
    if axis == 1u {
        return v.y;
    }
    return v.z;
}

/// Reciprocal-slab test clipped to `[t_lo, t_hi]`; mirrors `slab_interval`,
/// returning only the boolean the golden walk consumes.
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

/// Kahan's fused-multiply-add compensated `a*b - c*d`, accurate to ~1.5 ULP.
/// This is the `GPU`-portable stand-in for the golden's `f64` boundary-zero
/// recompute: it recovers the sign of a near-zero scaled edge area that a plain
/// `f32` `a*b - c*d` would round to exactly `0.0`.
fn diff_prod(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let cd = c * d;
    let err = fma(-c, d, cd);
    let dop = fma(a, b, -cd);
    return dop + err;
}

/// Double-sided Woop 2013 watertight test; mirrors the golden
/// `intersect_triangle_watertight` (`u`/`v` are the barycentric weights of
/// `v1`/`v2`). Picks the largest `|dir|` component as the projection axis,
/// shears the ray onto `+kz`, and forms the three scaled 2D edge areas; a shared
/// edge yields identical-but-opposite areas on the two triangles, so a boundary
/// zero is accepted by both and never leaks.
fn intersect_triangle_watertight(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> TriHit {
    var out = TriHit(false, 0.0, 0.0, 0.0);
    let dir = ray.dir;

    // Projection axis `kz` = largest-magnitude direction component.
    var kz = 0u;
    var max_abs = abs(dir.x);
    if abs(dir.y) > max_abs {
        kz = 1u;
        max_abs = abs(dir.y);
    }
    if abs(dir.z) > max_abs {
        kz = 2u;
    }
    var kx = kz + 1u;
    if kx == 3u {
        kx = 0u;
    }
    var ky = kx + 1u;
    if ky == 3u {
        ky = 0u;
    }
    // Swap `kx`/`ky` when the ray points down `-kz` so winding is preserved.
    let dkz = dir_axis(dir, kz);
    if dkz < 0.0 {
        let tmp = kx;
        kx = ky;
        ky = tmp;
    }

    // Shear/scale constants aligning the ray with `+kz`.
    let sx = dir_axis(dir, kx) / dkz;
    let sy = dir_axis(dir, ky) / dkz;
    let sz = 1.0 / dkz;

    // Triangle vertices relative to the ray origin.
    let a = v0 - ray.origin;
    let b = v1 - ray.origin;
    let c = v2 - ray.origin;

    let akz = comp(a, kz);
    let bkz = comp(b, kz);
    let ckz = comp(c, kz);

    // Sheared 2D coordinates in the (kx, ky) plane.
    let ax = comp(a, kx) - sx * akz;
    let ay = comp(a, ky) - sy * akz;
    let bx = comp(b, kx) - sx * bkz;
    let by = comp(b, ky) - sy * bkz;
    let cx = comp(c, kx) - sx * ckz;
    let cy = comp(c, ky) - sy * ckz;

    // Scaled barycentric coordinates: `wa`/`wb`/`wc` weight `v0`/`v1`/`v2`.
    var wa = cx * by - cy * bx;
    var wb = ax * cy - ay * cx;
    var wc = bx * ay - by * ax;

    // Boundary-zero refinement (see file header): compensated recompute when
    // any `f32` scaled area rounds to exactly zero.
    if wa == 0.0 || wb == 0.0 || wc == 0.0 {
        wa = diff_prod(cx, by, cy, bx);
        wb = diff_prod(ax, cy, ay, cx);
        wc = diff_prod(bx, ay, by, ax);
    }

    // Outside the triangle when the edge signs disagree (a boundary zero is
    // accepted by both branches, so edges never leak).
    if (wa < 0.0 || wb < 0.0 || wc < 0.0) && (wa > 0.0 || wb > 0.0 || wc > 0.0) {
        return out;
    }

    // Determinant; a zero determinant means the ray grazes the triangle plane.
    let det = wa + wb + wc;
    if det == 0.0 {
        return out;
    }

    // Scaled hit distance, then the interval test carrying `det`'s sign so we
    // reject before dividing.
    let az = sz * akz;
    let bz = sz * bkz;
    let cz = sz * ckz;
    let t_scaled = wa * az + wb * bz + wc * cz;
    if det > 0.0 {
        if t_scaled < ray.t_min * det || t_scaled > ray.t_max * det {
            return out;
        }
    } else {
        if t_scaled > ray.t_min * det || t_scaled < ray.t_max * det {
            return out;
        }
    }

    let inv_det = 1.0 / det;
    out.hit = true;
    out.t = t_scaled * inv_det;
    out.u = wb * inv_det;
    out.v = wc * inv_det;
    return out;
}

/// Decodes leaf triangle `pi` into its first vertex.
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
fn ray_watertight(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if ri >= params.ray_count {
        return;
    }

    // Empty `BVH`: no nodes to walk, every ray misses.
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
                    let th = intersect_triangle_watertight(
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
                        // Nearest query: shrink the interval and record the hit.
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
