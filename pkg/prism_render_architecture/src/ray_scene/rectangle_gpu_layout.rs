//! `GPU`-ready flat buffer layout for the [`RectangleBvh`], plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the rectangle counterpart of [`super::disk_gpu_layout`] and
//! [`super::cylinder_gpu_layout`]: the `CPU` [`RectangleBvh`] is convenient for
//! building and testing, but a compute kernel binds plain storage buffers. This
//! module pins the exact word layout the procedural-primitive intersection
//! kernel consumes and proves — with the same [`Rectangle::intersect`] and
//! [`Ray::aabb_interval`] arithmetic the in-memory walk uses — that a walk over
//! the flattened buffers returns the identical hit.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across triangle, sphere, box, curve, cylinder, disk, and
//! rectangle hierarchies; only the leaf payload differs. Encoding is
//! dependency-free: every buffer is a `Vec<u32>` with `f32` fields stored as
//! their `to_bits` pattern, and the rectangle stride is a multiple of four
//! words (16-byte aligned). Because [`Rectangle::new`] stores its fields
//! verbatim, a packed rectangle decodes back through the constructor bit-for-bit
//! with no re-normalization step.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::rectangle::{Rectangle, RectangleBvh, RectangleHit};
use super::traversal::Ray;

/// `u32` words per packed rectangle (48 bytes, 16-byte aligned).
///
/// Layout: `center.xyz` (0..3), `axis_u.xyz` (3..6), `axis_v.xyz` (6..9),
/// `primitive` (9), padding (10..12).
pub const RECTANGLE_WORDS: usize = 12;

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

/// Flat, `GPU`-uploadable form of a [`RectangleBvh`].
///
/// `nodes` holds the shared [`NODE_WORDS`] `BVH` records and `rectangles` holds
/// the leaf payloads packed with the [`RECTANGLE_WORDS`] stride in reordered
/// (leaf-contiguous) primitive order, so a leaf's
/// `[first_primitive, first_primitive + primitive_count)` range indexes
/// `rectangles` exactly as it indexes [`RectangleBvh::rectangles`] in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuRectangleBvhBuffers {
    /// Packed `BVH` nodes, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed rectangle records, [`RECTANGLE_WORDS`] words each, leaf-contiguous.
    pub rectangles: Vec<u32>,
}

impl GpuRectangleBvhBuffers {
    /// Serializes a built [`RectangleBvh`] into flat node and rectangle buffers.
    ///
    /// Node and rectangle order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed rectangle
    /// array the same way the in-memory leaf ranges index
    /// [`RectangleBvh::rectangles`].
    #[must_use]
    pub fn from_bvh(bvh: &RectangleBvh) -> Self {
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
        let mut rectangles = vec![0u32; bvh.rectangles().len() * RECTANGLE_WORDS];
        for (i, rectangle) in bvh.rectangles().iter().enumerate() {
            let b = i * RECTANGLE_WORDS;
            write_vec3(&mut rectangles, b, rectangle.center());
            write_vec3(&mut rectangles, b + 3, rectangle.axis_u());
            write_vec3(&mut rectangles, b + 6, rectangle.axis_v());
            rectangles[b + 9] = rectangle.primitive();
        }
        Self { nodes, rectangles }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed rectangles.
    #[must_use]
    pub fn rectangle_count(&self) -> usize {
        self.rectangles.len() / RECTANGLE_WORDS
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

    /// Decoded rectangle `i`.
    fn rectangle(&self, i: usize) -> Rectangle {
        let b = i * RECTANGLE_WORDS;
        // `Rectangle::new` stores its fields verbatim, so decoding through it
        // round-trips the packed words bit-for-bit.
        Rectangle::new(
            read_vec3(&self.rectangles, b),
            read_vec3(&self.rectangles, b + 3),
            read_vec3(&self.rectangles, b + 6),
            self.rectangles[b + 9],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`RectangleBvh::closest_hit`] exactly — same slab rejection, same
    /// near/far child ordering by split-axis sign, same running `t_max` shrink
    /// and the same [`Rectangle::intersect`] test — so the result is bit-for-bit
    /// identical and a `GPU` kernel binding these buffers can be diffed against
    /// it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<RectangleHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<RectangleHit> = None;

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
                        let rectangle = self.rectangle(pi);
                        if let Some(hit) = rectangle.intersect(&ray) {
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

    /// True when *any* rectangle intersects `ray`; mirrors
    /// [`RectangleBvh::any_hit`].
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
                        if self.rectangle(pi).intersect(ray).is_some() {
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

    fn random_rectangle(rng: &mut Rng, primitive: u32) -> Rectangle {
        let center = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let axis_u = [
            rng.range(0.3, 1.5),
            rng.range(-1.5, 1.5),
            rng.range(-1.5, 1.5),
        ];
        let axis_v = [
            rng.range(-1.5, 1.5),
            rng.range(0.3, 1.5),
            rng.range(-1.5, 1.5),
        ];
        Rectangle::new(center, axis_u, axis_v, primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Rectangle> {
        (0..count).map(|i| random_rectangle(rng, i)).collect()
    }

    #[test]
    fn rectangle_stride_is_sixteen_byte_aligned() {
        assert_eq!(RECTANGLE_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let rectangles = random_scene(&mut rng, 40);
        let bvh = RectangleBvh::build(&rectangles);
        let gpu = GpuRectangleBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.rectangle_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.rectangles.len(), gpu.rectangle_count() * RECTANGLE_WORDS);
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = RectangleBvh::build(&[]);
        let gpu = GpuRectangleBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.rectangle_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn decoded_rectangle_round_trips_fields() {
        let rect = Rectangle::new([0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, 0.9], 9);
        let bvh = RectangleBvh::build(&[rect]);
        let gpu = GpuRectangleBvhBuffers::from_bvh(&bvh);
        let decoded = gpu.rectangle(0);
        assert_eq!(decoded, rect);
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        let mut rng = Rng::new(0x5EED_F00D);
        let rectangles = random_scene(&mut rng, 64);
        let bvh = RectangleBvh::build(&rectangles);
        let gpu = GpuRectangleBvhBuffers::from_bvh(&bvh);

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
