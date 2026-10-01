//! `GPU`-ready flat buffer layout for the [`BilinearPatchBvh`], plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the bilinear-patch counterpart of [`super::capsule_gpu_layout`] and
//! [`super::round_cone_gpu_layout`]: the `CPU` [`BilinearPatchBvh`] is
//! convenient for building and testing, but a compute kernel binds plain
//! storage buffers. This module pins the exact word layout the
//! procedural-primitive intersection kernel consumes and proves — with the same
//! [`BilinearPatch::intersect`] and [`Ray::aabb_interval`] arithmetic the
//! in-memory walk uses — that a walk over the flattened buffers returns the
//! identical hit.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across every hierarchy; only the leaf payload differs.
//! Encoding is dependency-free: every buffer is a `Vec<u32>` with `f32` fields
//! stored as their `to_bits` pattern, and the patch stride is a multiple of four
//! words (16-byte aligned). Because [`BilinearPatch::new`] stores all four
//! corners verbatim, a packed patch decodes back through the constructor
//! bit-for-bit with no re-normalization step.

use super::bilinear_patch::{BilinearPatch, BilinearPatchBvh, BilinearPatchHit};
use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::traversal::Ray;

/// `u32` words per packed bilinear patch (64 bytes, 16-byte aligned).
///
/// Layout: `p00.xyz` (0..3), `p10.xyz` (3..6), `p11.xyz` (6..9),
/// `p01.xyz` (9..12), `primitive` (12), padding (13..16).
pub const BILINEAR_PATCH_WORDS: usize = 16;

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

/// Pops the top node index off the traversal stack, or `None` when empty.
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

/// Flat, `GPU`-uploadable form of a [`BilinearPatchBvh`].
///
/// `nodes` holds the shared [`NODE_WORDS`] `BVH` records and `patches` holds the
/// leaf payloads packed with the [`BILINEAR_PATCH_WORDS`] stride in reordered
/// (leaf-contiguous) primitive order, so a leaf's
/// `[first_primitive, first_primitive + primitive_count)` range indexes
/// `patches` exactly as it indexes [`BilinearPatchBvh::patches`] in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBilinearPatchBvhBuffers {
    /// Packed `BVH` nodes, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed patch records, [`BILINEAR_PATCH_WORDS`] words each,
    /// leaf-contiguous.
    pub patches: Vec<u32>,
}

impl GpuBilinearPatchBvhBuffers {
    /// Serializes a built [`BilinearPatchBvh`] into flat node and patch buffers.
    ///
    /// Node and patch order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed patch array
    /// the same way the in-memory leaf ranges index
    /// [`BilinearPatchBvh::patches`].
    #[must_use]
    pub fn from_bvh(bvh: &BilinearPatchBvh) -> Self {
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
        let mut patches = vec![0u32; bvh.patches().len() * BILINEAR_PATCH_WORDS];
        for (i, patch) in bvh.patches().iter().enumerate() {
            let b = i * BILINEAR_PATCH_WORDS;
            write_vec3(&mut patches, b, patch.p00());
            write_vec3(&mut patches, b + 3, patch.p10());
            write_vec3(&mut patches, b + 6, patch.p11());
            write_vec3(&mut patches, b + 9, patch.p01());
            patches[b + 12] = patch.primitive();
        }
        Self { nodes, patches }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed patches.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len() / BILINEAR_PATCH_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decodes the `BVH` bounds stored at packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let base = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, base), read_vec3(&self.nodes, base + 3))
    }

    /// Decoded patch `i`.
    fn patch(&self, i: usize) -> BilinearPatch {
        let b = i * BILINEAR_PATCH_WORDS;
        // `BilinearPatch::new` stores all four corners verbatim, so decoding
        // through it round-trips the packed words bit-for-bit.
        BilinearPatch::new(
            read_vec3(&self.patches, b),
            read_vec3(&self.patches, b + 3),
            read_vec3(&self.patches, b + 6),
            read_vec3(&self.patches, b + 9),
            self.patches[b + 12],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`BilinearPatchBvh::closest_hit`] exactly — same slab rejection,
    /// same near/far child ordering by split-axis sign, same running `t_max`
    /// shrink and the same [`BilinearPatch::intersect`] test — so the result is
    /// bit-for-bit identical and a `GPU` kernel binding these buffers can be
    /// diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<BilinearPatchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<BilinearPatchHit> = None;

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
                        let patch = self.patch(pi);
                        if let Some(hit) = patch.intersect(&ray) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
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

    /// True when *any* patch intersects `ray`; mirrors
    /// [`BilinearPatchBvh::any_hit`].
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
                        if self.patch(pi).intersect(ray).is_some() {
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
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn random_patch(rng: &mut Rng, primitive: u32) -> BilinearPatch {
        let p = |rng: &mut Rng| {
            [
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
            ]
        };
        BilinearPatch::new(p(rng), p(rng), p(rng), p(rng), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<BilinearPatch> {
        (0..count).map(|i| random_patch(rng, i)).collect()
    }

    #[test]
    fn patch_stride_is_sixteen_byte_aligned() {
        assert_eq!(BILINEAR_PATCH_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let scene = random_scene(&mut rng, 40);
        let bvh = BilinearPatchBvh::build(&scene);
        let gpu = GpuBilinearPatchBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.patch_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.patches.len(), gpu.patch_count() * BILINEAR_PATCH_WORDS);
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = BilinearPatchBvh::build(&[]);
        let gpu = GpuBilinearPatchBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.patch_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn decoded_patch_round_trips_fields() {
        let patch = BilinearPatch::new(
            [0.1, 0.2, 0.3],
            [1.4, -0.5, 1.6],
            [1.7, 1.8, -0.9],
            [-0.2, 1.1, 0.5],
            9,
        );
        let bvh = BilinearPatchBvh::build(&[patch]);
        let gpu = GpuBilinearPatchBvhBuffers::from_bvh(&bvh);
        let decoded = gpu.patch(0);
        assert_eq!(decoded, patch);
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        let mut rng = Rng::new(0x5EED_F00D);
        let scene = random_scene(&mut rng, 64);
        let bvh = BilinearPatchBvh::build(&scene);
        let gpu = GpuBilinearPatchBvhBuffers::from_bvh(&bvh);

        for _ in 0..3_000 {
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
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let cpu = bvh.closest_hit(&ray);
            let packed = gpu.closest_hit(&ray);
            match (cpu, packed) {
                (None, None) => {}
                (Some(c), Some(p)) => {
                    assert_eq!(c.primitive, p.primitive);
                    assert_eq!(c.t.to_bits(), p.t.to_bits(), "t bits differ");
                    assert_eq!(c.front_face, p.front_face);
                    assert_eq!(c.u.to_bits(), p.u.to_bits(), "u bits differ");
                    assert_eq!(c.v.to_bits(), p.v.to_bits(), "v bits differ");
                    for k in 0..3 {
                        assert_eq!(
                            c.normal[k].to_bits(),
                            p.normal[k].to_bits(),
                            "normal[{k}] bits differ"
                        );
                    }
                }
                (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), gpu.any_hit(&ray));
        }
    }
}
