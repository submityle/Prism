//! `GPU`-ready flat buffer layout for the [`ShadedBilinearPatchBvh`], plus a
//! packed traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the smooth-shaded counterpart of [`super::bilinear_patch_gpu_layout`]
//! and [`super::shaded_triangle_gpu_layout`]: the `CPU`
//! [`ShadedBilinearPatchBvh`] is convenient for building and testing, but a
//! compute kernel binds plain storage buffers. This module pins the exact word
//! layout the procedural-primitive intersection kernel consumes and proves —
//! with the same [`ShadedBilinearPatch::intersect`] and [`Ray::aabb_interval`]
//! arithmetic the in-memory walk uses — that a walk over the flattened buffers
//! returns the identical hit, including the interpolated shading normal and
//! `UV`.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across every hierarchy; only the leaf payload differs.
//! Encoding is dependency-free: every buffer is a `Vec<u32>` with `f32` fields
//! stored as their `to_bits` pattern, and the patch stride is a multiple of four
//! words (16-byte aligned). Because [`ShadedBilinearPatch::new`] stores every
//! corner attribute verbatim, a packed patch decodes back through the
//! constructor bit-for-bit with no re-normalization step.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::shaded_bilinear_patch::{
    ShadedBilinearPatch, ShadedBilinearPatchBvh, ShadedBilinearPatchHit,
};
use super::traversal::Ray;

/// `u32` words per packed shaded bilinear patch (144 bytes, 16-byte aligned).
///
/// Layout: four corner positions `p00/p10/p11/p01.xyz` (0..12), four corner
/// shading normals `n00/n10/n11/n01.xyz` (12..24), four corner texture
/// coordinates `uv00/uv10/uv11/uv01.xy` (24..32), `primitive` (32), padding
/// (33..36).
pub const SHADED_BILINEAR_PATCH_WORDS: usize = 36;

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

/// Writes an `[f32; 2]` as two `to_bits` words at `out[base..base + 2]`.
fn write_vec2(out: &mut [u32], base: usize, v: [f32; 2]) {
    out[base] = v[0].to_bits();
    out[base + 1] = v[1].to_bits();
}

/// Reads two `from_bits` words at `words[base..base + 2]` back into `[f32; 2]`.
fn read_vec2(words: &[u32], base: usize) -> [f32; 2] {
    [f32::from_bits(words[base]), f32::from_bits(words[base + 1])]
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

/// Flat, `GPU`-uploadable form of a [`ShadedBilinearPatchBvh`].
///
/// `nodes` holds the shared [`NODE_WORDS`] `BVH` records and `patches` holds the
/// leaf payloads packed with the [`SHADED_BILINEAR_PATCH_WORDS`] stride in
/// reordered (leaf-contiguous) primitive order, so a leaf's
/// `[first_primitive, first_primitive + primitive_count)` range indexes
/// `patches` exactly as it indexes [`ShadedBilinearPatchBvh::patches`] in
/// memory.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuShadedBilinearPatchBvhBuffers {
    /// Packed `BVH` nodes, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed patch records, [`SHADED_BILINEAR_PATCH_WORDS`] words each,
    /// leaf-contiguous.
    pub patches: Vec<u32>,
}

impl GpuShadedBilinearPatchBvhBuffers {
    /// Serializes a built [`ShadedBilinearPatchBvh`] into flat node and patch
    /// buffers.
    ///
    /// Node and patch order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed patch array
    /// the same way the in-memory leaf ranges index
    /// [`ShadedBilinearPatchBvh::patches`].
    #[must_use]
    pub fn from_bvh(bvh: &ShadedBilinearPatchBvh) -> Self {
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
        let mut patches = vec![0u32; bvh.patches().len() * SHADED_BILINEAR_PATCH_WORDS];
        for (i, patch) in bvh.patches().iter().enumerate() {
            let b = i * SHADED_BILINEAR_PATCH_WORDS;
            let [p00, p10, p11, p01] = patch.positions();
            write_vec3(&mut patches, b, p00);
            write_vec3(&mut patches, b + 3, p10);
            write_vec3(&mut patches, b + 6, p11);
            write_vec3(&mut patches, b + 9, p01);
            let [n00, n10, n11, n01] = patch.normals();
            write_vec3(&mut patches, b + 12, n00);
            write_vec3(&mut patches, b + 15, n10);
            write_vec3(&mut patches, b + 18, n11);
            write_vec3(&mut patches, b + 21, n01);
            let [uv00, uv10, uv11, uv01] = patch.uvs();
            write_vec2(&mut patches, b + 24, uv00);
            write_vec2(&mut patches, b + 26, uv10);
            write_vec2(&mut patches, b + 28, uv11);
            write_vec2(&mut patches, b + 30, uv01);
            patches[b + 32] = patch.primitive();
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
        self.patches.len() / SHADED_BILINEAR_PATCH_WORDS
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
    fn patch(&self, i: usize) -> ShadedBilinearPatch {
        let b = i * SHADED_BILINEAR_PATCH_WORDS;
        // `ShadedBilinearPatch::new` stores every attribute verbatim, so
        // decoding through it round-trips the packed words bit-for-bit.
        ShadedBilinearPatch::new(
            [
                read_vec3(&self.patches, b),
                read_vec3(&self.patches, b + 3),
                read_vec3(&self.patches, b + 6),
                read_vec3(&self.patches, b + 9),
            ],
            [
                read_vec3(&self.patches, b + 12),
                read_vec3(&self.patches, b + 15),
                read_vec3(&self.patches, b + 18),
                read_vec3(&self.patches, b + 21),
            ],
            [
                read_vec2(&self.patches, b + 24),
                read_vec2(&self.patches, b + 26),
                read_vec2(&self.patches, b + 28),
                read_vec2(&self.patches, b + 30),
            ],
            self.patches[b + 32],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`ShadedBilinearPatchBvh::closest_hit`] exactly — same slab
    /// rejection, same near/far child ordering by split-axis sign, same running
    /// `t_max` shrink and the same [`ShadedBilinearPatch::intersect`] test — so
    /// the result is bit-for-bit identical and a `GPU` kernel binding these
    /// buffers can be diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ShadedBilinearPatchHit> = None;

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
    /// [`ShadedBilinearPatchBvh::any_hit`].
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

    fn random_patch(rng: &mut Rng, primitive: u32) -> ShadedBilinearPatch {
        let p = |rng: &mut Rng| {
            [
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
            ]
        };
        // Normals biased toward +z so corner blends never fully cancel.
        let n = |rng: &mut Rng| [rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), 1.0];
        let uv = |rng: &mut Rng| [rng.range(0.0, 1.0), rng.range(0.0, 1.0)];
        ShadedBilinearPatch::new(
            [p(rng), p(rng), p(rng), p(rng)],
            [n(rng), n(rng), n(rng), n(rng)],
            [uv(rng), uv(rng), uv(rng), uv(rng)],
            primitive,
        )
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<ShadedBilinearPatch> {
        (0..count).map(|i| random_patch(rng, i)).collect()
    }

    #[test]
    fn patch_stride_is_sixteen_byte_aligned() {
        assert_eq!(SHADED_BILINEAR_PATCH_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let scene = random_scene(&mut rng, 40);
        let bvh = ShadedBilinearPatchBvh::build(&scene);
        let gpu = GpuShadedBilinearPatchBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.patch_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(
            gpu.patches.len(),
            gpu.patch_count() * SHADED_BILINEAR_PATCH_WORDS
        );
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = ShadedBilinearPatchBvh::build(&[]);
        let gpu = GpuShadedBilinearPatchBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.patch_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn decoded_patch_round_trips_fields() {
        let patch = ShadedBilinearPatch::new(
            [
                [0.1, 0.2, 0.3],
                [1.4, -0.5, 1.6],
                [1.7, 1.8, -0.9],
                [-0.2, 1.1, 0.5],
            ],
            [
                [0.0, 0.1, 1.0],
                [0.2, -0.1, 0.9],
                [-0.3, 0.2, 1.1],
                [0.1, 0.3, 0.8],
            ],
            [[0.0, 0.0], [1.0, 0.25], [0.75, 1.0], [0.1, 0.9]],
            9,
        );
        let bvh = ShadedBilinearPatchBvh::build(&[patch]);
        let gpu = GpuShadedBilinearPatchBvhBuffers::from_bvh(&bvh);
        let decoded = gpu.patch(0);
        assert_eq!(decoded, patch);
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        for seed in [0x5EED_F00D, 0x1234_ABCDu64] {
            let mut rng = Rng::new(seed);
            let scene = random_scene(&mut rng, 64);
            let bvh = ShadedBilinearPatchBvh::build(&scene);
            let gpu = GpuShadedBilinearPatchBvhBuffers::from_bvh(&bvh);

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
                        for (cg, pg) in c.geometric_normal.iter().zip(p.geometric_normal.iter()) {
                            assert_eq!(cg.to_bits(), pg.to_bits(), "geometric normal bits differ");
                        }
                        for (cs, ps) in c.shading_normal.iter().zip(p.shading_normal.iter()) {
                            assert_eq!(cs.to_bits(), ps.to_bits(), "shading normal bits differ");
                        }
                        for (cu, pu) in c.uv.iter().zip(p.uv.iter()) {
                            assert_eq!(cu.to_bits(), pu.to_bits(), "uv bits differ");
                        }
                    }
                    (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
                }
                assert_eq!(bvh.any_hit(&ray), gpu.any_hit(&ray));
            }
        }
    }
}
