//! `GPU`-ready flat buffer layout for the software `BVH`/`TLAS`, plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! The `CPU` [`Bvh`]/[`Tlas`] own `Rust` structs convenient for building and
//! testing, but a compute kernel binds plain storage buffers. This module pins
//! the exact word layout the `WESL` traversal kernel consumes and proves — with
//! the same [`intersect_triangle`] and [`Ray::aabb_interval`] arithmetic the
//! in-memory walk uses — that a walk over the flattened buffers returns the
//! identical hit. That makes this the authoritative `ABI` contract: the shader
//! mirrors these strides and field offsets, and the `GPU`↔`CPU` parity test
//! compares against [`Bvh::closest_hit`].
//!
//! Encoding is dependency-free: every buffer is a `Vec<u32>`, with `f32` fields
//! stored as their `to_bits` pattern (little-endian words on upload), matching a
//! `WESL` `array<u32>` or a scalar-field `struct` in `std430`. Strides are a
//! multiple of four words (16 bytes) so each record is 16-byte aligned.

use super::bvh::{Aabb, Bvh, Triangle};
use super::tlas::{Affine3, Tlas};
use super::traversal::{intersect_triangle, Hit, Ray};

/// `u32` words per packed `BVH`/`TLAS` node (48 bytes, 16-byte aligned).
///
/// Layout: `min.xyz` (0..3), `max.xyz` (3..6), `first_primitive` (6),
/// `second_child` (7), `primitive_count` (8), `axis` (9), padding (10..12).
pub const NODE_WORDS: usize = 12;

/// `u32` words per packed triangle (48 bytes, 16-byte aligned).
///
/// Layout: `v0.xyz` (0..3), `v1.xyz` (3..6), `v2.xyz` (6..9), `primitive` (9),
/// padding (10..12).
pub const TRIANGLE_WORDS: usize = 12;

/// `u32` words per packed `TLAS` instance (64 bytes, 16-byte aligned).
///
/// Layout: `world_to_object` linear columns `c0.xyz` (0..3), `c1.xyz` (3..6),
/// `c2.xyz` (6..9), translation `t.xyz` (9..12), `blas_index` (12),
/// `instance_id` (13), padding (14..16).
pub const INSTANCE_WORDS: usize = 16;

/// `u32` words per packed `BLAS` offset record (16 bytes, 16-byte aligned).
///
/// Layout: `node_base` (0), `node_count` (1), `triangle_base` (2),
/// `triangle_count` (3). Indexes the shared pool arrays so one `BLAS` is stored
/// once and referenced by many instances.
pub const BLAS_OFFSET_WORDS: usize = 4;

/// Flattened, `GPU`-uploadable buffers for a single bottom-level `BVH`.
///
/// `nodes` and `triangles` are packed with [`NODE_WORDS`]/[`TRIANGLE_WORDS`]
/// strides; a kernel binds them as read-only storage and walks them exactly as
/// [`GpuBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// Packed triangle records, [`TRIANGLE_WORDS`] words each, in reordered
    /// (leaf-contiguous) order matching the node primitive ranges.
    pub triangles: Vec<u32>,
}

/// Writes an `[f32; 3]` as three `to_bits` words at `out[base..base + 3]`.
fn write_vec3(out: &mut [u32], base: usize, v: [f32; 3]) {
    out[base] = v[0].to_bits();
    out[base + 1] = v[1].to_bits();
    out[base + 2] = v[2].to_bits();
}

/// Reads three `from_bits` words at `words[base..base + 3]` back into `[f32; 3]`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

impl GpuBvhBuffers {
    /// Serializes a built [`Bvh`] into flat node and triangle buffers.
    ///
    /// Node and triangle order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed triangle
    /// array the same way the in-memory leaf ranges index [`Bvh::primitives`].
    #[must_use]
    pub fn from_bvh(bvh: &Bvh) -> Self {
        let mut nodes = vec![0u32; bvh.nodes().len() * NODE_WORDS];
        for (i, node) in bvh.nodes().iter().enumerate() {
            let b = i * NODE_WORDS;
            write_vec3(&mut nodes, b, node.bounds.min);
            write_vec3(&mut nodes, b + 3, node.bounds.max);
            nodes[b + 6] = node.first_primitive;
            nodes[b + 7] = node.second_child;
            nodes[b + 8] = u32::from(node.primitive_count);
            nodes[b + 9] = u32::from(node.axis);
        }
        let mut triangles = vec![0u32; bvh.primitives().len() * TRIANGLE_WORDS];
        for (i, tri) in bvh.primitives().iter().enumerate() {
            let b = i * TRIANGLE_WORDS;
            write_vec3(&mut triangles, b, tri.v0);
            write_vec3(&mut triangles, b + 3, tri.v1);
            write_vec3(&mut triangles, b + 6, tri.v2);
            triangles[b + 9] = tri.primitive;
        }
        Self { nodes, triangles }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / TRIANGLE_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Decoded triangle `i`.
    fn triangle(&self, i: usize) -> Triangle {
        let b = i * TRIANGLE_WORDS;
        Triangle::new(
            read_vec3(&self.triangles, b),
            read_vec3(&self.triangles, b + 3),
            read_vec3(&self.triangles, b + 6),
            self.triangles[b + 9],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`Bvh::closest_hit`] exactly — same slab rejection, same near/far
    /// child ordering by split-axis sign, same running `t_max` shrink and the
    /// same [`intersect_triangle`] test — so the result is bit-for-bit identical
    /// and the `GPU` kernel that binds these buffers can be diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<Hit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        let tri = self.triangle(pi);
                        if let Some((t, u, v)) = intersect_triangle(&ray, &tri) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), t);
                            best = Some(Hit {
                                t,
                                u,
                                v,
                                primitive: tri.primitive,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = self.nodes[base + 7];
                    let axis = self.nodes[base + 9] as usize;
                    let neg = ray.direction()[axis] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* triangle intersects `ray`; mirrors [`Bvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        let tri = self.triangle(pi);
                        if intersect_triangle(ray, &tri).is_some() {
                            return true;
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = self.nodes[base + 7];
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// A shared pool of bottom-level buffers plus per-`BLAS` offsets into two
/// concatenated arrays, so a `TLAS` kernel binds one node buffer and one
/// triangle buffer for the whole scene and each instance selects its slice.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuBlasPool {
    /// All `BLAS` nodes concatenated, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// All `BLAS` triangles concatenated, [`TRIANGLE_WORDS`] words each.
    pub triangles: Vec<u32>,
    /// Per-`BLAS` offset records, [`BLAS_OFFSET_WORDS`] words each.
    pub offsets: Vec<u32>,
}

impl GpuBlasPool {
    /// Concatenates a `BLAS` pool into shared buffers with an offset table.
    #[must_use]
    pub fn from_blases(blases: &[Bvh]) -> Self {
        let mut nodes = Vec::new();
        let mut triangles = Vec::new();
        let mut offsets = vec![0u32; blases.len() * BLAS_OFFSET_WORDS];
        for (i, bvh) in blases.iter().enumerate() {
            let node_base = nodes.len() / NODE_WORDS;
            let tri_base = triangles.len() / TRIANGLE_WORDS;
            let packed = GpuBvhBuffers::from_bvh(bvh);
            let node_count = packed.node_count();
            let triangle_count = packed.triangle_count();
            // Rebase each leaf's `first_primitive` (word 6) from BLAS-local into
            // the shared pool's global triangle index, so a kernel binding the
            // single pooled triangle buffer indexes it directly; `blas_view`
            // reverses this to reconstruct a standalone per-BLAS buffer.
            let mut blas_nodes = packed.nodes;
            for n in 0..node_count {
                let b = n * NODE_WORDS;
                if blas_nodes[b + 8] > 0 {
                    blas_nodes[b + 6] += tri_base as u32;
                }
            }
            nodes.extend_from_slice(&blas_nodes);
            triangles.extend_from_slice(&packed.triangles);
            let o = i * BLAS_OFFSET_WORDS;
            offsets[o] = node_base as u32;
            offsets[o + 1] = node_count as u32;
            offsets[o + 2] = tri_base as u32;
            offsets[o + 3] = triangle_count as u32;
        }
        Self {
            nodes,
            triangles,
            offsets,
        }
    }

    /// Number of `BLAS` entries in the pool.
    #[must_use]
    pub fn blas_count(&self) -> usize {
        self.offsets.len() / BLAS_OFFSET_WORDS
    }

    /// A [`GpuBvhBuffers`] view of one `BLAS` in the pool (copies its slice).
    ///
    /// The copy re-bases the leaf `first_primitive` fields to zero so the
    /// returned standalone buffer traverses correctly on its own; the shared
    /// pool keeps the original global bases for kernel use.
    #[must_use]
    fn blas_view(&self, blas: usize) -> GpuBvhBuffers {
        let o = blas * BLAS_OFFSET_WORDS;
        let node_base = self.offsets[o] as usize;
        let node_count = self.offsets[o + 1] as usize;
        let tri_base = self.offsets[o + 2] as usize;
        let tri_count = self.offsets[o + 3] as usize;
        let nodes_slice =
            &self.nodes[node_base * NODE_WORDS..(node_base + node_count) * NODE_WORDS];
        let mut nodes = nodes_slice.to_vec();
        // Leaf `first_primitive` (word 6) is a global pool index; re-base to the
        // per-BLAS triangle array this view owns.
        for i in 0..node_count {
            let b = i * NODE_WORDS;
            if nodes[b + 8] > 0 {
                nodes[b + 6] -= tri_base as u32;
            }
        }
        let triangles =
            self.triangles[tri_base * TRIANGLE_WORDS..(tri_base + tri_count) * TRIANGLE_WORDS]
                .to_vec();
        GpuBvhBuffers { nodes, triangles }
    }
}

/// Flattened `GPU` buffers for a top-level acceleration structure.
///
/// Pairs a packed top-level node array with a packed instance array; traversal
/// pulls each instance's `world_to_object` from the buffer, transforms the ray,
/// and queries the referenced `BLAS` slice of `pool`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuTlasBuffers {
    /// Packed top-level node records, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed instance records, [`INSTANCE_WORDS`] words each, in the `TLAS`
    /// reordered instance order.
    pub instances: Vec<u32>,
}

/// Writes an [`Affine3`] `world_to_object` into `out[base..base + 12]` as three
/// linear columns followed by the translation.
fn write_affine(out: &mut [u32], base: usize, m: &Affine3) {
    let c = m.columns();
    write_vec3(out, base, c[0]);
    write_vec3(out, base + 3, c[1]);
    write_vec3(out, base + 6, c[2]);
    write_vec3(out, base + 9, m.translation());
}

/// Reads an [`Affine3`] back from `words[base..base + 12]`.
fn read_affine(words: &[u32], base: usize) -> Affine3 {
    Affine3::from_cols(
        [
            read_vec3(words, base),
            read_vec3(words, base + 3),
            read_vec3(words, base + 6),
        ],
        read_vec3(words, base + 9),
    )
}

impl GpuTlasBuffers {
    /// Serializes a built [`Tlas`] into a packed node buffer and instance buffer.
    #[must_use]
    pub fn from_tlas(tlas: &Tlas) -> Self {
        let mut nodes = vec![0u32; tlas.nodes().len() * NODE_WORDS];
        for (i, node) in tlas.nodes().iter().enumerate() {
            let b = i * NODE_WORDS;
            write_vec3(&mut nodes, b, node.bounds.min);
            write_vec3(&mut nodes, b + 3, node.bounds.max);
            nodes[b + 6] = node.first_primitive;
            nodes[b + 7] = node.second_child;
            nodes[b + 8] = u32::from(node.primitive_count);
            nodes[b + 9] = u32::from(node.axis);
        }
        let mut instances = vec![0u32; tlas.instances().len() * INSTANCE_WORDS];
        for (i, inst) in tlas.instances().iter().enumerate() {
            let b = i * INSTANCE_WORDS;
            write_affine(&mut instances, b, &inst.world_to_object());
            instances[b + 12] = inst.blas() as u32;
            instances[b + 13] = inst.instance_id();
        }
        Self { nodes, instances }
    }

    /// Number of packed top-level nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed instances.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len() / INSTANCE_WORDS
    }

    /// True when the `TLAS` holds no instances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decoded bounds of packed top-level node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Nearest intersection along the world-space `ray`, mirroring
    /// [`Tlas::closest_hit`] over the packed buffers and shared `pool`.
    ///
    /// For each leaf instance the packed `world_to_object` transforms the ray
    /// into object space (direction carried through the linear part without
    /// renormalizing, so `t` is preserved), then the referenced `BLAS` slice is
    /// queried with the same [`GpuBvhBuffers::closest_hit`] walk; the running
    /// `t_max` shrinks across instances exactly like the in-memory `TLAS`.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray, pool: &GpuBlasPool) -> Option<TlasPackedHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<TlasPackedHit> = None;
        let mut best_t = ray.t_max();
        let t_min = ray.t_min();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, t_min, best_t).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for inst_idx in start..end {
                        let ib = inst_idx * INSTANCE_WORDS;
                        let world_to_object = read_affine(&self.instances, ib);
                        let blas = self.instances[ib + 12] as usize;
                        let instance_id = self.instances[ib + 13];
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        let view = pool.blas_view(blas);
                        if let Some(hit) = view.closest_hit(&obj_ray)
                            && hit.t < best_t
                        {
                            best_t = hit.t;
                            best = Some(TlasPackedHit {
                                t: hit.t,
                                u: hit.u,
                                v: hit.v,
                                primitive: hit.primitive,
                                instance_id,
                                instance_index: inst_idx as u32,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = self.nodes[base + 7];
                    let axis = self.nodes[base + 9] as usize;
                    let neg = ray.direction()[axis] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* instance intersects the world-space `ray` inside its
    /// interval, mirroring [`Tlas::any_hit`] over the packed buffers and shared
    /// `pool`. Returns on the first confirmed hit and never shrinks `t_max`
    /// (the cheap shadow / ambient-occlusion occlusion query), making it the
    /// packed GPU-ABI twin of the in-memory any-hit walk.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray, pool: &GpuBlasPool) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let t_min = ray.t_min();
        let t_max = ray.t_max();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, t_min, t_max).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for inst_idx in start..end {
                        let ib = inst_idx * INSTANCE_WORDS;
                        let world_to_object = read_affine(&self.instances, ib);
                        let blas = self.instances[ib + 12] as usize;
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        if pool.blas_view(blas).any_hit(&obj_ray) {
                            return true;
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = self.nodes[base + 7];
                    if sp < stack.len() {
                        stack[sp] = second_child;
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// A `TLAS` intersection reported by the packed walk; mirrors
/// [`TlasHit`](super::tlas::TlasHit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasPackedHit {
    /// Ray parameter at the hit (identical in world and object space).
    pub t: f32,
    /// Barycentric `u`.
    pub u: f32,
    /// Barycentric `v`.
    pub v: f32,
    /// Stable primitive id from the hit `BLAS` triangle.
    pub primitive: u32,
    /// Stable id of the hit instance.
    pub instance_id: u32,
    /// Index of the hit instance in the packed instance buffer.
    pub instance_index: u32,
}

/// Pops the top of the short traversal stack, or `None` when empty.
#[inline]
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        return None;
    }
    *sp -= 1;
    Some(stack[*sp])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::{Bvh, Triangle};
    use crate::ray_scene::tlas::{Affine3, Instance, Tlas};

    /// Deterministic xorshift generator, matching the shape used across the
    /// `ray_scene` module tests so scenes are reproducible without `libm`.
    struct Rng(u64);

    impl Rng {
        /// Seeds the generator, forcing a non-zero state.
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        /// Advances the state and returns the high 32 bits.
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }

        /// A value in `[0, 1]`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }

        /// A value in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// A cloud of `n` small random triangles spread over a cube, each carrying a
    /// stable primitive id, so the built `BVH` has interior nodes and multiple
    /// leaves to traverse.
    fn random_triangles(n: u32, rng: &mut Rng) -> Vec<Triangle> {
        (0..n)
            .map(|id| {
                let c = [
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                ];
                let p = |r: &mut Rng| {
                    [
                        c[0] + r.range(-0.6, 0.6),
                        c[1] + r.range(-0.6, 0.6),
                        c[2] + r.range(-0.6, 0.6),
                    ]
                };
                Triangle::new(p(rng), p(rng), p(rng), id)
            })
            .collect()
    }

    /// A random invertible object→world transform (scale, then rotation, then
    /// translation), with scales kept away from zero so the inverse exists.
    fn random_affine(rng: &mut Rng) -> Affine3 {
        let quat = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let rot = Affine3::from_quaternion(quat);
        let scale = Affine3::from_scale([
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
        ]);
        let trans = Affine3::from_translation([
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ]);
        trans.compose(&rot.compose(&scale))
    }

    #[test]
    fn packed_blas_closest_hit_matches_in_memory_bit_for_bit() {
        let mut rng = Rng::new(0x5EED_0F17);
        let tris = random_triangles(600, &mut rng);
        let bvh = Bvh::build(&tris);
        let packed = GpuBvhBuffers::from_bvh(&bvh);

        assert_eq!(packed.node_count(), bvh.nodes().len());
        assert_eq!(packed.triangle_count(), bvh.primitives().len());

        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::infinite(origin, dir);
            let reference = bvh.closest_hit(&ray);
            let via_packed = packed.closest_hit(&ray);
            // Bit-for-bit equality on `Hit` (t/u/v/primitive) — the packed walk
            // must be indistinguishable from the in-memory walk.
            assert_eq!(reference, via_packed, "closest-hit divergence");
            // any_hit must agree with closest_hit existence on the same buffers.
            assert_eq!(reference.is_some(), packed.any_hit(&ray), "any-hit divergence");
            if reference.is_some() {
                hits += 1;
            }
        }
        assert!(hits > 100, "scene too sparse to be a meaningful test: {hits}");
    }

    #[test]
    fn packed_blas_bounded_interval_matches_in_memory() {
        // Exercise the running t_max shrink with a finite interval so early-out
        // pruning is stressed as well as the open-ended case above.
        let mut rng = Rng::new(0x1357_9BDF);
        let tris = random_triangles(400, &mut rng);
        let bvh = Bvh::build(&tris);
        let packed = GpuBvhBuffers::from_bvh(&bvh);
        for _ in 0..3000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::new(origin, dir, rng.range(0.0, 1.0), rng.range(2.0, 20.0));
            assert_eq!(bvh.closest_hit(&ray), packed.closest_hit(&ray));
        }
    }

    #[test]
    fn packed_empty_blas_never_hits() {
        let bvh = Bvh::build(&[]);
        let packed = GpuBvhBuffers::from_bvh(&bvh);
        assert!(packed.is_empty());
        assert_eq!(packed.node_count(), 0);
        assert_eq!(packed.triangle_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(packed.closest_hit(&ray).is_none());
        assert!(!packed.any_hit(&ray));
    }

    #[test]
    fn packed_tlas_closest_hit_matches_in_memory_bit_for_bit() {
        let mut rng = Rng::new(0x0DDB_A11E);
        // A shared pool of a few distinct BLASes so the offset table and the
        // per-instance BLAS selection are both exercised.
        let blases: Vec<Bvh> = (0..3)
            .map(|_| Bvh::build(&random_triangles(120, &mut rng)))
            .collect();
        let pool = GpuBlasPool::from_blases(&blases);
        assert_eq!(pool.blas_count(), blases.len());

        for _ in 0..30 {
            let count = 1 + (rng.next_u32() % 10) as usize;
            let mut instances = Vec::with_capacity(count);
            for id in 0..count {
                let blas = (rng.next_u32() as usize) % blases.len();
                instances.push(Instance::new(random_affine(&mut rng), blas, id as u32).unwrap());
            }
            let tlas = Tlas::build(&instances, &blases);
            let packed = GpuTlasBuffers::from_tlas(&tlas);
            assert_eq!(packed.instance_count(), tlas.instances().len());
            assert_eq!(packed.node_count(), tlas.nodes().len());

            for _ in 0..300 {
                let origin = [
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                ];
                let dir = [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ];
                let ray = Ray::infinite(origin, dir);
                let reference = tlas.closest_hit(&ray, &blases);
                let via_packed = packed.closest_hit(&ray, &pool);
                match (reference, via_packed) {
                    (None, None) => {}
                    (Some(r), Some(v)) => {
                        // Every field must match bit-for-bit, including the
                        // reordered instance_index the two walks assign.
                        assert_eq!(r.t, v.t, "t divergence");
                        assert_eq!(r.u, v.u, "u divergence");
                        assert_eq!(r.v, v.v, "v divergence");
                        assert_eq!(r.primitive, v.primitive, "primitive divergence");
                        assert_eq!(r.instance_id, v.instance_id, "instance_id divergence");
                        assert_eq!(
                            r.instance_index, v.instance_index,
                            "instance_index divergence"
                        );
                    }
                    (r, v) => panic!("existence mismatch: {r:?} vs {v:?}"),
                }
            }
        }
    }

    #[test]
    fn packed_tlas_any_hit_matches_in_memory() {
        let mut rng = Rng::new(0x7A5C_0DE9);
        // A shared pool of a few distinct BLASes so instance BLAS selection and
        // the pool offset table are both exercised, mirroring the closest-hit
        // parity test's multi-instance / multi-BLAS setup.
        let blases: Vec<Bvh> = (0..3)
            .map(|_| Bvh::build(&random_triangles(120, &mut rng)))
            .collect();
        let pool = GpuBlasPool::from_blases(&blases);

        let mut occluded = 0u32;
        for _ in 0..30 {
            let count = 1 + (rng.next_u32() % 10) as usize;
            let mut instances = Vec::with_capacity(count);
            for id in 0..count {
                let blas = (rng.next_u32() as usize) % blases.len();
                instances.push(Instance::new(random_affine(&mut rng), blas, id as u32).unwrap());
            }
            let tlas = Tlas::build(&instances, &blases);
            let packed = GpuTlasBuffers::from_tlas(&tlas);

            for _ in 0..300 {
                let origin = [
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                ];
                let dir = [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ];
                // Mix open-ended and bounded intervals so the fixed-`t_max`
                // any-hit early-out is stressed both ways.
                let ray = if rng.unit() < 0.5 {
                    Ray::infinite(origin, dir)
                } else {
                    Ray::new(origin, dir, rng.range(0.0, 1.0), rng.range(2.0, 20.0))
                };
                let reference = tlas.any_hit(&ray, &blases);
                let via_packed = packed.any_hit(&ray, &pool);
                assert_eq!(reference, via_packed, "tlas any-hit divergence");
                // any_hit must also agree with closest_hit existence over the
                // same bounded interval on the packed buffers.
                assert_eq!(
                    via_packed,
                    packed.closest_hit(&ray, &pool).is_some(),
                    "any-hit vs closest-hit existence divergence"
                );
                if via_packed {
                    occluded += 1;
                }
            }
        }
        assert!(occluded > 100, "scene too sparse to be a meaningful test: {occluded}");
    }

    #[test]
    fn packed_empty_tlas_never_hits() {
        let blases = vec![Bvh::build(&[Triangle::new(
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
        )])];
        let pool = GpuBlasPool::from_blases(&blases);
        let tlas = Tlas::build(&[], &blases);
        let packed = GpuTlasBuffers::from_tlas(&tlas);
        assert!(packed.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(packed.closest_hit(&ray, &pool).is_none());
        assert!(!packed.any_hit(&ray, &pool));
    }
}
