//! `GPU`-ready flat buffer layout for the analytic [`ObbBvh`], plus a packed
//! traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the oriented-box counterpart of [`super::sphere_gpu_layout`] and
//! [`super::ellipsoid_gpu_layout`]: the `CPU` [`ObbBvh`] is convenient for
//! building and testing, but a compute kernel binds plain storage buffers. This
//! module pins the exact word layout the procedural-primitive intersection
//! kernel consumes and proves — with the same [`Obb::intersect`] and
//! [`Ray::aabb_interval`] arithmetic the in-memory walk uses — that a walk over
//! the flattened buffers returns the identical hit.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across triangle, sphere, ellipsoid, and oriented-box
//! hierarchies; only the leaf payload differs. Encoding is dependency-free:
//! every buffer is a `Vec<u32>` with `f32` fields stored as their `to_bits`
//! pattern, and the oriented-box stride is a multiple of four words (16 bytes)
//! so each record stays 16-byte aligned.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::obb::{Obb, ObbBvh, ObbHit};
use super::traversal::Ray;

/// `u32` words per packed oriented box (64 bytes, 16-byte aligned).
///
/// Layout: `center.xyz` (0..3), `half.xyz` (3..6), `axes[u].xyz` (6..9),
/// `axes[v].xyz` (9..12), `axes[w].xyz` (12..15), `primitive` (15). The 16-word
/// stride is already 16-byte aligned, so no padding word is required.
pub const OBB_WORDS: usize = 16;

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

/// Flattened, `GPU`-uploadable buffers for a single [`ObbBvh`].
///
/// `nodes` are packed with the shared [`NODE_WORDS`] stride; `boxes` are packed
/// with the [`OBB_WORDS`] stride in reordered (leaf-contiguous) order matching
/// the node primitive ranges. A kernel binds them as read-only storage and
/// walks them exactly as [`GpuObbBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuObbBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// Packed oriented-box records, [`OBB_WORDS`] words each, leaf-contiguous.
    pub boxes: Vec<u32>,
}

impl GpuObbBvhBuffers {
    /// Serializes a built [`ObbBvh`] into flat node and box buffers.
    ///
    /// Node and box order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed box array
    /// the same way the in-memory leaf ranges index [`ObbBvh::boxes`].
    #[must_use]
    pub fn from_bvh(bvh: &ObbBvh) -> Self {
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
        let mut boxes = vec![0u32; bvh.boxes().len() * OBB_WORDS];
        for (i, obb) in bvh.boxes().iter().enumerate() {
            let b = i * OBB_WORDS;
            let axes = obb.axes();
            write_vec3(&mut boxes, b, obb.center());
            write_vec3(&mut boxes, b + 3, obb.half());
            write_vec3(&mut boxes, b + 6, axes[0]);
            write_vec3(&mut boxes, b + 9, axes[1]);
            write_vec3(&mut boxes, b + 12, axes[2]);
            boxes[b + 15] = obb.primitive();
        }
        Self { nodes, boxes }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed oriented boxes.
    #[must_use]
    pub fn box_count(&self) -> usize {
        self.boxes.len() / OBB_WORDS
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

    /// Decoded oriented box `i`.
    ///
    /// Decoding routes through [`Obb::new`] with the exact stored words (axes
    /// verbatim, half folded to magnitude), so the reconstructed box is
    /// bit-identical to the one the `CPU` [`ObbBvh`] holds.
    fn obb(&self, i: usize) -> Obb {
        let b = i * OBB_WORDS;
        Obb::new(
            read_vec3(&self.boxes, b),
            [
                read_vec3(&self.boxes, b + 6),
                read_vec3(&self.boxes, b + 9),
                read_vec3(&self.boxes, b + 12),
            ],
            read_vec3(&self.boxes, b + 3),
            self.boxes[b + 15],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`ObbBvh::closest_hit`] exactly — same slab rejection, same
    /// near/far child ordering by split-axis sign, same running `t_max` shrink
    /// and the same [`Obb::intersect`] test — so the result is bit-for-bit
    /// identical and a `GPU` kernel binding these buffers can be diffed against
    /// it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ObbHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ObbHit> = None;

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
                        let obb = self.obb(pi);
                        if let Some(hit) = obb.intersect(&ray) {
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

    /// True when *any* oriented box intersects `ray`; mirrors
    /// [`ObbBvh::any_hit`].
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
                        if self.obb(pi).intersect(ray).is_some() {
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

    /// Builds an orthonormal frame from a tangent half-angle parameter, keeping
    /// clippy's transcendental ban satisfied (no trig): `unit_pair` maps a free
    /// scalar to a point on the unit circle through the rational
    /// `((1 - m²)/(1 + m²), 2m/(1 + m²))` identity.
    fn unit_pair(m: f32) -> (f32, f32) {
        let d = 1.0 + m * m;
        ((1.0 - m * m) / d, 2.0 * m / d)
    }

    /// Rotation about `x` by the angle encoded in `(c, s)` applied to the
    /// canonical basis, yielding an orthonormal `[u, v, w]` frame.
    fn frame_x(c: f32, s: f32) -> [[f32; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
    }

    /// Composes two axis rotations into one orthonormal frame so test boxes are
    /// genuinely oriented in 3-space rather than axis-aligned.
    fn tilted_frame(m0: f32, m1: f32) -> [[f32; 3]; 3] {
        let (cx, sx) = unit_pair(m0);
        let (cz, sz) = unit_pair(m1);
        let fx = frame_x(cx, sx);
        // Rotate the x-tilted frame about world z.
        let rz = [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]];
        let mut out = [[0.0f32; 3]; 3];
        for (i, row) in fx.iter().enumerate() {
            for k in 0..3 {
                out[i][k] = row[0] * rz[0][k] + row[1] * rz[1][k] + row[2] * rz[2][k];
            }
        }
        out
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Obb> {
        (0..count)
            .map(|i| {
                let center = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let half = [
                    rng.range(0.2, 1.2),
                    rng.range(0.2, 1.2),
                    rng.range(0.2, 1.2),
                ];
                let axes = tilted_frame(rng.range(-1.5, 1.5), rng.range(-1.5, 1.5));
                Obb::new(center, axes, half, i)
            })
            .collect()
    }

    #[test]
    fn obb_stride_is_sixteen_byte_aligned() {
        assert_eq!(OBB_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let boxes = random_scene(&mut rng, 40);
        let bvh = ObbBvh::build(&boxes);
        let gpu = GpuObbBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.box_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.boxes.len(), gpu.box_count() * OBB_WORDS);
    }

    #[test]
    fn decoded_obb_round_trips_fields() {
        let mut rng = Rng::new(0xABCD_1234);
        let boxes = random_scene(&mut rng, 24);
        let bvh = ObbBvh::build(&boxes);
        let gpu = GpuObbBvhBuffers::from_bvh(&bvh);
        for (i, expected) in bvh.boxes().iter().enumerate() {
            let decoded = gpu.obb(i);
            assert_eq!(decoded, *expected);
        }
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = ObbBvh::build(&[]);
        let gpu = GpuObbBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.box_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        let mut rng = Rng::new(0x5EED_F00D);
        let boxes = random_scene(&mut rng, 64);
        let bvh = ObbBvh::build(&boxes);
        let gpu = GpuObbBvhBuffers::from_bvh(&bvh);

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
                    // Bit-for-bit: identical arithmetic must yield identical bits.
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
            assert_eq!(gpu.any_hit(&ray), bvh.any_hit(&ray));
        }
    }
}
