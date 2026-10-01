//! `GPU`-ready flat buffer layout for the [`SdfBrickBvh`], plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the `SDF`-brick counterpart of [`super::ellipsoid_gpu_layout`]: the
//! `CPU` [`SdfBrickBvh`] is convenient for building and testing, but a compute
//! kernel binds plain storage buffers. This module pins the exact word layout
//! the procedural-primitive sphere-trace kernel consumes and proves — with the
//! same [`SdfBrick::intersect`] and [`Ray::aabb_interval`] arithmetic the
//! in-memory walk uses — that a walk over the flattened buffers returns the
//! identical hit.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across every primitive hierarchy; only the leaf payload
//! differs. Because bricks vary in resolution, each brick's distance samples
//! live in a single shared pool ([`GpuSdfBrickBvhBuffers::data`]) and the
//! fixed-stride header ([`SDF_BRICK_WORDS`]) carries the brick's dimensions,
//! box, stable id, and the `[offset, len)` slice into that pool. Encoding is
//! dependency-free: every buffer is a `Vec<u32>` with `f32` fields stored as
//! their `to_bits` pattern.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::sdf_brick::{SdfBrick, SdfBrickBvh, SdfBrickHit};
use super::traversal::Ray;

/// `u32` words per packed brick header (48 bytes, 16-byte aligned).
///
/// Layout: `dims.xyz` (0..3), `origin.xyz` bits (3..6), `spacing.xyz` bits
/// (6..9), `primitive` (9), `data_offset` (10), `data_len` (11).
pub const SDF_BRICK_WORDS: usize = 12;

/// Header word index of the row-major distance-pool start offset.
const OFFSET_WORD: usize = 10;
/// Header word index of the row-major distance-pool length.
const LEN_WORD: usize = 11;

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

/// Flattened, `GPU`-uploadable buffers for a single [`SdfBrickBvh`].
///
/// `nodes` are packed with the shared [`NODE_WORDS`] stride; `headers` are
/// packed with the [`SDF_BRICK_WORDS`] stride in reordered (leaf-contiguous)
/// order matching the node primitive ranges; and `data` is the shared,
/// row-major distance pool each header slices with its `[offset, len)`. A
/// kernel binds all three as read-only storage and walks them exactly as
/// [`GpuSdfBrickBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuSdfBrickBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// Packed brick headers, [`SDF_BRICK_WORDS`] words each, leaf-contiguous.
    pub headers: Vec<u32>,
    /// Shared distance pool: all bricks' row-major samples as `to_bits`.
    pub data: Vec<u32>,
}

impl GpuSdfBrickBvhBuffers {
    /// Serializes a built [`SdfBrickBvh`] into flat node, header, and data
    /// buffers.
    ///
    /// Node and brick order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed header array
    /// the same way the in-memory leaf ranges index [`SdfBrickBvh::bricks`].
    /// Each brick's samples are appended to the shared pool in order and the
    /// header records the resulting `[offset, len)` slice.
    #[must_use]
    pub fn from_bvh(bvh: &SdfBrickBvh) -> Self {
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

        let mut headers = vec![0u32; bvh.bricks().len() * SDF_BRICK_WORDS];
        let mut data: Vec<u32> = Vec::new();
        for (i, brick) in bvh.bricks().iter().enumerate() {
            let b = i * SDF_BRICK_WORDS;
            let dims = brick.dims();
            headers[b] = dims[0];
            headers[b + 1] = dims[1];
            headers[b + 2] = dims[2];
            write_vec3(&mut headers, b + 3, brick.origin());
            write_vec3(&mut headers, b + 6, brick.spacing());
            headers[b + 9] = brick.primitive();
            headers[b + OFFSET_WORD] = data.len() as u32;
            headers[b + LEN_WORD] = brick.data().len() as u32;
            data.extend(brick.data().iter().map(|d| d.to_bits()));
        }

        Self {
            nodes,
            headers,
            data,
        }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed brick headers.
    #[must_use]
    pub fn brick_count(&self) -> usize {
        self.headers.len() / SDF_BRICK_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decodes packed brick `i` back into an owned [`SdfBrick`], slicing its
    /// samples out of the shared distance pool.
    #[must_use]
    pub fn brick(&self, i: usize) -> SdfBrick {
        let b = i * SDF_BRICK_WORDS;
        let dims = [self.headers[b], self.headers[b + 1], self.headers[b + 2]];
        let origin = read_vec3(&self.headers, b + 3);
        let spacing = read_vec3(&self.headers, b + 6);
        let primitive = self.headers[b + 9];
        let offset = self.headers[b + OFFSET_WORD] as usize;
        let len = self.headers[b + LEN_WORD] as usize;
        let samples: Vec<f32> = self.data[offset..offset + len]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect();
        SdfBrick::new(dims, samples, origin, spacing, primitive).expect("valid packed brick")
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`SdfBrickBvh::closest_hit`] exactly — same slab rejection, same
    /// near/far child ordering by split-axis sign, same running `t_max` shrink
    /// and the same decoded [`SdfBrick::intersect`] test — so the result is
    /// bit-for-bit identical and a `GPU` kernel binding these buffers can be
    /// diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<SdfBrickHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<SdfBrickHit> = None;

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
                        let brick = self.brick(pi);
                        if let Some(hit) = brick.intersect(&ray) {
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

    /// True when *any* brick intersects `ray`; mirrors
    /// [`SdfBrickBvh::any_hit`].
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
                        if self.brick(pi).intersect(ray).is_some() {
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

    /// Samples a sphere `SDF` onto a `dims` grid spanning
    /// `[origin, origin + (dims - 1) * spacing]`.
    fn sphere_brick(
        dims: [u32; 3],
        origin: [f32; 3],
        spacing: [f32; 3],
        center: [f32; 3],
        radius: f32,
        primitive: u32,
    ) -> SdfBrick {
        let mut data = Vec::with_capacity(dims[0] as usize * dims[1] as usize * dims[2] as usize);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let p = [
                        origin[0] + x as f32 * spacing[0],
                        origin[1] + y as f32 * spacing[1],
                        origin[2] + z as f32 * spacing[2],
                    ];
                    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - radius;
                    data.push(dist);
                }
            }
        }
        SdfBrick::new(dims, data, origin, spacing, primitive).expect("valid sphere brick")
    }

    /// Builds a scene of randomly placed sphere bricks of varying resolution so
    /// the shared distance pool holds bricks of differing lengths.
    fn random_scene(rng: &mut Rng, count: u32) -> Vec<SdfBrick> {
        (0..count)
            .map(|i| {
                let center = [
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                ];
                let radius = rng.range(0.4, 1.1);
                let half = radius + 0.3;
                let n = 9 + (i % 4) * 2; // 9, 11, 13, 15 — differing pool slices.
                let dims = [n, n, n];
                let spacing = [
                    2.0 * half / (dims[0] - 1) as f32,
                    2.0 * half / (dims[1] - 1) as f32,
                    2.0 * half / (dims[2] - 1) as f32,
                ];
                let origin = [center[0] - half, center[1] - half, center[2] - half];
                sphere_brick(dims, origin, spacing, center, radius, i)
            })
            .collect()
    }

    #[test]
    fn brick_header_stride_is_sixteen_byte_aligned() {
        assert_eq!(SDF_BRICK_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_reports_consistent_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let bricks = random_scene(&mut rng, 20);
        let bvh = SdfBrickBvh::build(&bricks);
        let gpu = GpuSdfBrickBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.brick_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.headers.len(), gpu.brick_count() * SDF_BRICK_WORDS);
        let pool: usize = bvh.bricks().iter().map(|b| b.data().len()).sum();
        assert_eq!(gpu.data.len(), pool);
    }

    #[test]
    fn decoded_brick_round_trips_fields() {
        let mut rng = Rng::new(0xABCD_1234);
        let bricks = random_scene(&mut rng, 16);
        let bvh = SdfBrickBvh::build(&bricks);
        let gpu = GpuSdfBrickBvhBuffers::from_bvh(&bvh);
        for (i, expected) in bvh.bricks().iter().enumerate() {
            assert_eq!(gpu.brick(i), *expected);
        }
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = SdfBrickBvh::build(&[]);
        let gpu = GpuSdfBrickBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.brick_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    fn packed_walk_matches_in_memory(seed: u64) {
        let mut rng = Rng::new(seed);
        let bricks = random_scene(&mut rng, 40);
        let bvh = SdfBrickBvh::build(&bricks);
        let gpu = GpuSdfBrickBvhBuffers::from_bvh(&bvh);

        for _ in 0..3_000 {
            let origin = [
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
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
                }
                (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
            }
            assert_eq!(gpu.any_hit(&ray), bvh.any_hit(&ray));
        }
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        packed_walk_matches_in_memory(0x5EED_F00D);
        packed_walk_matches_in_memory(0x1234_ABCD);
    }
}
