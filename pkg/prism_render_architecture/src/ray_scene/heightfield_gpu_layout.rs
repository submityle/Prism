//! `GPU`-ready flat buffer layout for the [`HeightfieldBvh`], plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! A compute kernel does not bind the `CPU` [`Heightfield`]; it binds plain
//! storage buffers. This module pins the exact word layout a terrain-`BLAS`
//! traversal kernel consumes and proves — with the very same
//! [`Heightfield::intersect_cell_triangle`] and [`Ray::aabb_interval`]
//! arithmetic the in-memory walk uses — that a walk over the flattened buffers
//! returns the identical [`super::heightfield::HeightfieldHit`], including the
//! cell / triangle, geometric normal, and interpolated domain `UV`.
//!
//! Three buffers plus a small header mirror the heightfield's compact form:
//! `nodes` reuse the shared [`NODE_WORDS`] record; `order` maps each `BVH`
//! primitive slot back to its original cell index (exactly as
//! [`HeightfieldBvh::order`] does); and `heights` packs the row-major samples as
//! their `to_bits` pattern. The [`HEIGHTFIELD_HEADER_WORDS`] header stores the
//! grid dimensions and planar domain (`origin` + `extent`), so decoding routes
//! back through [`Heightfield::new`] and rebuilds a bit-identical primitive.
//! Encoding is dependency-free: every buffer is a `Vec<u32>` and the header is a
//! fixed `[u32; 8]`, a multiple of four words so it stays 16-byte aligned.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::heightfield::{Heightfield, HeightfieldBvh, HeightfieldHit};
use super::traversal::Ray;

/// `u32` words in the packed heightfield header (16-byte aligned).
///
/// Layout: `width` (0), `height` (1), `origin` bits (2..5), `extent` bits
/// (5..7), padding (7).
pub const HEIGHTFIELD_HEADER_WORDS: usize = 8;

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

/// Flattened, `GPU`-uploadable buffers for a single [`HeightfieldBvh`].
///
/// `nodes` are packed with the shared [`NODE_WORDS`] stride; `order` is the
/// `BVH`-slot → original-cell-index map that leaf primitive ranges index into;
/// `heights` packs the row-major samples as `to_bits`; and `header` carries the
/// grid dimensions and planar domain ([`HEIGHTFIELD_HEADER_WORDS`] words). A
/// kernel binds them as read-only storage and walks them exactly as
/// [`GpuHeightfieldBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuHeightfieldBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// `BVH`-slot → original-cell-index map; leaf ranges index this.
    pub order: Vec<u32>,
    /// Row-major height samples as `to_bits` words, `width * height` of them.
    pub heights: Vec<u32>,
    /// Grid dimensions + planar domain; see [`HEIGHTFIELD_HEADER_WORDS`].
    pub header: [u32; HEIGHTFIELD_HEADER_WORDS],
}

impl GpuHeightfieldBvhBuffers {
    /// Serializes a built [`HeightfieldBvh`] into flat buffers.
    ///
    /// Node and `order` entries are preserved exactly and the height samples are
    /// copied verbatim as `to_bits`, so the decoded heightfield is bit-identical
    /// and the packed walk reproduces every hit.
    #[must_use]
    pub fn from_bvh(bvh: &HeightfieldBvh) -> Self {
        let field = bvh.field();
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

        let heights: Vec<u32> = field.heights().iter().map(|h| h.to_bits()).collect();

        let origin = field.origin();
        let extent = field.extent();
        let header = [
            field.width() as u32,
            field.height() as u32,
            origin[0].to_bits(),
            origin[1].to_bits(),
            origin[2].to_bits(),
            extent[0].to_bits(),
            extent[1].to_bits(),
            0,
        ];

        Self {
            nodes,
            order: bvh.order().to_vec(),
            heights,
            header,
        }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Grid columns stored in the header.
    #[must_use]
    pub fn width(&self) -> usize {
        self.header[0] as usize
    }

    /// Grid rows stored in the header.
    #[must_use]
    pub fn height(&self) -> usize {
        self.header[1] as usize
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

    /// Rebuilds the [`Heightfield`] from the packed header and height samples.
    ///
    /// Decoding routes through [`Heightfield::new`] with the exact stored words,
    /// so the reconstructed field is bit-identical to the one the `CPU`
    /// [`HeightfieldBvh`] holds and `intersect_cell_triangle` reproduces every
    /// hit.
    #[must_use]
    pub fn decode(&self) -> Heightfield {
        let width = self.width();
        let height = self.height();
        let heights: Vec<f32> = self.heights.iter().map(|&w| f32::from_bits(w)).collect();
        let origin = read_vec3(&self.header, 2);
        let extent = [f32::from_bits(self.header[5]), f32::from_bits(self.header[6])];
        Heightfield::new(width, height, heights, origin, extent)
            .expect("packed buffers encode a valid heightfield")
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Decodes the field, then mirrors [`HeightfieldBvh::closest_hit`] exactly —
    /// same slab rejection, same near/far child ordering by split-axis sign,
    /// same running `t_max` shrink, and the same `order`-indexed per-cell
    /// two-triangle test — so the result is bit-for-bit identical and a `GPU`
    /// kernel binding these buffers can be diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<HeightfieldHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let field = self.decode();
        let mut ray = *ray;
        let mut best: Option<HeightfieldHit> = None;

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
                    for &slot in &self.order[start..end] {
                        for tri in 0..2u8 {
                            if let Some(hit) =
                                field.intersect_cell_triangle(slot as usize, tri, &ray)
                            {
                                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                                best = Some(hit);
                            }
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

    /// True when *any* cell triangle intersects `ray`; mirrors
    /// [`HeightfieldBvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let field = self.decode();
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
                    for &slot in &self.order[start..end] {
                        for tri in 0..2u8 {
                            if field.intersect_cell_triangle(slot as usize, tri, ray).is_some() {
                                return true;
                            }
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
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    fn norm(v: [f32; 3]) -> [f32; 3] {
        let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    /// Builds a random `width * height` heightfield with bounded heights and a
    /// random planar domain.
    fn random_field(rng: &mut Rng, width: usize, height: usize) -> Heightfield {
        let heights: Vec<f32> = (0..width * height).map(|_| rng.range(-1.5, 1.5)).collect();
        let origin = rng.point(-5.0, 5.0);
        let extent = [rng.range(2.0, 6.0), rng.range(2.0, 6.0)];
        Heightfield::new(width, height, heights, origin, extent).expect("valid grid")
    }

    #[test]
    fn header_is_sixteen_byte_aligned() {
        assert_eq!(HEIGHTFIELD_HEADER_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_reports_consistent_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let field = random_field(&mut rng, 6, 5);
        let bvh = HeightfieldBvh::build(field.clone());
        let gpu = GpuHeightfieldBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.order, bvh.order());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.heights.len(), field.width() * field.height());
        assert_eq!(gpu.width(), field.width());
        assert_eq!(gpu.height(), field.height());
    }

    #[test]
    fn decoded_field_round_trips() {
        let mut rng = Rng::new(0x5EED_F00D);
        for _ in 0..50 {
            let width = 2 + (rng.next_u32() % 7) as usize;
            let height = 2 + (rng.next_u32() % 7) as usize;
            let field = random_field(&mut rng, width, height);
            let bvh = HeightfieldBvh::build(field.clone());
            let gpu = GpuHeightfieldBvhBuffers::from_bvh(&bvh);
            // The decoded field must equal the original bit-for-bit (PartialEq
            // over the verbatim samples / domain).
            assert_eq!(gpu.decode(), field);
        }
    }

    #[test]
    fn empty_buffers_never_hit() {
        let gpu = GpuHeightfieldBvhBuffers::default();
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        for seed in [0xABCD_1234u64, 0x5EED_F00D] {
            let mut rng = Rng::new(seed);
            let mut shared = 0usize;
            for _ in 0..60 {
                let width = 2 + (rng.next_u32() % 7) as usize;
                let height = 2 + (rng.next_u32() % 7) as usize;
                let field = random_field(&mut rng, width, height);
                let bounds = field.aabb();
                let center = [
                    0.5 * (bounds.min[0] + bounds.max[0]),
                    0.5 * (bounds.min[1] + bounds.max[1]),
                    0.5 * (bounds.min[2] + bounds.max[2]),
                ];
                let bvh = HeightfieldBvh::build(field);
                let gpu = GpuHeightfieldBvhBuffers::from_bvh(&bvh);
                for _ in 0..120 {
                    // Aim most rays at a jittered point over the domain so a
                    // healthy fraction actually strike the thin terrain sheet,
                    // then shoot from a random origin toward it.
                    let target = [
                        center[0] + rng.range(-3.0, 3.0),
                        center[1] + rng.range(-2.0, 2.0),
                        center[2] + rng.range(-3.0, 3.0),
                    ];
                    let origin = rng.point(-8.0, 8.0);
                    let dir = [
                        target[0] - origin[0],
                        target[1] - origin[1],
                        target[2] - origin[2],
                    ];
                    if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                        continue;
                    }
                    let ray = Ray::infinite(origin, norm(dir));
                    let cpu = bvh.closest_hit(&ray);
                    let packed = gpu.closest_hit(&ray);
                    match (cpu, packed) {
                        (None, None) => {}
                        (Some(c), Some(p)) => {
                            assert_eq!(c.t.to_bits(), p.t.to_bits(), "t bits differ");
                            assert_eq!(c.u.to_bits(), p.u.to_bits(), "u bits differ");
                            assert_eq!(c.v.to_bits(), p.v.to_bits(), "v bits differ");
                            assert_eq!(c.cell, p.cell);
                            assert_eq!(c.triangle, p.triangle);
                            assert_eq!(c.front_face, p.front_face);
                            for k in 0..3 {
                                assert_eq!(
                                    c.position[k].to_bits(),
                                    p.position[k].to_bits(),
                                    "position[{k}] bits differ"
                                );
                                assert_eq!(
                                    c.normal[k].to_bits(),
                                    p.normal[k].to_bits(),
                                    "normal[{k}] bits differ"
                                );
                            }
                            assert_eq!(c.uv[0].to_bits(), p.uv[0].to_bits(), "uv.x bits differ");
                            assert_eq!(c.uv[1].to_bits(), p.uv[1].to_bits(), "uv.y bits differ");
                            shared += 1;
                        }
                        (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
                    }
                    assert_eq!(gpu.any_hit(&ray), bvh.any_hit(&ray));
                }
            }
            assert!(shared > 100, "too few shared hits for seed {seed:#x}: {shared}");
        }
    }
}
