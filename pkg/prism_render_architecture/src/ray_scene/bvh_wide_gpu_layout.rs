//! Flat, `GPU`-uploadable layout for the compressed wide (`BVH8`) tree.
//!
//! [`super::bvh_wide::WideBvh`] is the in-memory golden reference; this module
//! is its wire format: the exact `array<u32>` blob a `WESL` traversal kernel
//! binds, plus a packed walk that decodes those words and reproduces the
//! in-memory wide walk (and therefore [`super::bvh::Bvh::closest_hit`])
//! bit-for-bit. It is the `CPU`↔`GPU` parity anchor for the wide traversal
//! kernel, mirroring the role [`super::gpu_layout`] plays for the binary tree.
//!
//! ## Node record ([`WIDE_NODE_WORDS`] words)
//! - `0..3` — quantization `origin.xyz` (`f32` bits).
//! - `3` — packed `exp_x | exp_y<<8 | exp_z<<16 | child_count<<24`, where each
//!   `exp` is the biased IEEE-754 exponent of the per-axis power-of-two scale
//!   (the scale's mantissa is always zero, so the byte round-trips exactly).
//! - then [`super::bvh_wide::WIDE_BRANCHING`] child records of `4` words each:
//!   - `+0` — `qlo.x | qlo.y<<8 | qlo.z<<16 | tag<<24` ([`CHILD_EMPTY`] /
//!     [`CHILD_INTERIOR`] / [`CHILD_LEAF`]).
//!   - `+1` — `qhi.x | qhi.y<<8 | qhi.z<<16` (high byte reserved, zero).
//!   - `+2` — interior child node index, or leaf first-primitive index.
//!   - `+3` — leaf primitive count (zero for interior/empty).
//!
//! ## Triangle record ([`WIDE_TRIANGLE_WORDS`] words)
//! `v0.xyz` (`0..3`), `v1.xyz` (`3..6`), `v2.xyz` (`6..9`), `primitive` (`9`),
//! padding (`10..12`) — the same packing [`super::gpu_layout`] uses.

use super::bvh::{Aabb, Triangle};
use super::bvh_wide::{WideBvh, WideChild, WIDE_BRANCHING};
use super::traversal::{intersect_triangle, intersect_triangle_watertight, Hit, Ray};

/// Words per packed wide node: 4 header words (`origin.xyz` + packed
/// exponents/child-count) plus 4 words for each of [`WIDE_BRANCHING`] children.
pub const WIDE_NODE_WORDS: usize = 4 + WIDE_BRANCHING * 4;

/// Words per packed triangle (matching [`super::gpu_layout::TRIANGLE_WORDS`]).
pub const WIDE_TRIANGLE_WORDS: usize = 12;

/// Child-slot tag: unused slot (never traversed).
pub const CHILD_EMPTY: u32 = 0;
/// Child-slot tag: interior child (word `+2` is the child node index).
pub const CHILD_INTERIOR: u32 = 1;
/// Child-slot tag: leaf child (word `+2`/`+3` are first-primitive/count).
pub const CHILD_LEAF: u32 = 2;

/// Writes a 3-component vector as consecutive `f32`-bit words at `base`.
fn write_vec3(out: &mut [u32], base: usize, v: [f32; 3]) {
    out[base] = v[0].to_bits();
    out[base + 1] = v[1].to_bits();
    out[base + 2] = v[2].to_bits();
}

/// Reads a 3-component vector from consecutive `f32`-bit words at `base`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

/// Reconstructs a power-of-two scale from its biased IEEE-754 exponent byte.
fn scale_from_exp(exp: u32) -> f32 {
    f32::from_bits((exp & 0xff) << 23)
}

/// Flat, `GPU`-uploadable form of a [`WideBvh`].
///
/// `nodes` and `triangles` are the exact word blobs a traversal kernel binds;
/// [`WIDE_NODE_WORDS`]/[`WIDE_TRIANGLE_WORDS`] give their strides.
#[derive(Clone, Debug)]
pub struct GpuWideBvh {
    /// Packed wide nodes, [`WIDE_NODE_WORDS`] words each, root first.
    pub nodes: Vec<u32>,
    /// Packed triangles, [`WIDE_TRIANGLE_WORDS`] words each, in the reordered
    /// (leaf-contiguous) order the node leaf ranges index.
    pub triangles: Vec<u32>,
}

impl GpuWideBvh {
    /// Packs a [`WideBvh`] into its flat `GPU` layout.
    #[must_use]
    pub fn from_wide(wide: &WideBvh) -> Self {
        let mut nodes = vec![0u32; wide.node_count() * WIDE_NODE_WORDS];
        for (i, node) in wide.nodes().iter().enumerate() {
            let b = i * WIDE_NODE_WORDS;
            write_vec3(&mut nodes, b, node.origin());
            let scale = node.scale();
            let ex = (scale[0].to_bits() >> 23) & 0xff;
            let ey = (scale[1].to_bits() >> 23) & 0xff;
            let ez = (scale[2].to_bits() >> 23) & 0xff;
            nodes[b + 3] = ex | (ey << 8) | (ez << 16) | (u32::from(node.child_count()) << 24);
            for c in 0..node.child_count() as usize {
                let cb = b + 4 + c * 4;
                let qlo = node.child_qlo(c);
                let qhi = node.child_qhi(c);
                let (tag, payload_a, payload_b) = match node.child_kind(c) {
                    WideChild::Empty => (CHILD_EMPTY, 0, 0),
                    WideChild::Interior(idx) => (CHILD_INTERIOR, idx, 0),
                    WideChild::Leaf { first, count } => (CHILD_LEAF, first, u32::from(count)),
                };
                nodes[cb] = u32::from(qlo[0])
                    | (u32::from(qlo[1]) << 8)
                    | (u32::from(qlo[2]) << 16)
                    | (tag << 24);
                nodes[cb + 1] =
                    u32::from(qhi[0]) | (u32::from(qhi[1]) << 8) | (u32::from(qhi[2]) << 16);
                nodes[cb + 2] = payload_a;
                nodes[cb + 3] = payload_b;
            }
        }

        let mut triangles = vec![0u32; wide.primitive_count() * WIDE_TRIANGLE_WORDS];
        for (i, tri) in wide.primitives().iter().enumerate() {
            let b = i * WIDE_TRIANGLE_WORDS;
            write_vec3(&mut triangles, b, tri.v0);
            write_vec3(&mut triangles, b + 3, tri.v1);
            write_vec3(&mut triangles, b + 6, tri.v2);
            triangles[b + 9] = tri.primitive;
        }
        Self { nodes, triangles }
    }

    /// Number of packed wide nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / WIDE_NODE_WORDS
    }

    /// Number of packed triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / WIDE_TRIANGLE_WORDS
    }

    /// True when there are no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Quantization origin of packed node `ni`.
    #[cfg(test)]
    fn node_origin(&self, ni: usize) -> [f32; 3] {
        read_vec3(&self.nodes, ni * WIDE_NODE_WORDS)
    }

    /// Per-axis dequantization scale of packed node `ni`.
    fn node_scale(&self, ni: usize) -> [f32; 3] {
        let packed = self.nodes[ni * WIDE_NODE_WORDS + 3];
        [
            scale_from_exp(packed),
            scale_from_exp(packed >> 8),
            scale_from_exp(packed >> 16),
        ]
    }

    /// Populated child count of packed node `ni`.
    fn node_child_count(&self, ni: usize) -> usize {
        (self.nodes[ni * WIDE_NODE_WORDS + 3] >> 24) as usize
    }

    /// Dequantized (conservative) bounds of child `c` of packed node `ni`.
    fn child_bounds(&self, ni: usize, c: usize) -> Aabb {
        let b = ni * WIDE_NODE_WORDS;
        let origin = read_vec3(&self.nodes, b);
        let scale = self.node_scale(ni);
        let cb = b + 4 + c * 4;
        let lo = self.nodes[cb];
        let hi = self.nodes[cb + 1];
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for a in 0..3 {
            let qlo = (lo >> (a * 8)) & 0xff;
            let qhi = (hi >> (a * 8)) & 0xff;
            min[a] = origin[a] + (qlo as f32) * scale[a];
            max[a] = origin[a] + (qhi as f32) * scale[a];
        }
        Aabb::new(min, max)
    }

    /// Tag of child `c` of packed node `ni` ([`CHILD_EMPTY`]/[`CHILD_INTERIOR`]/
    /// [`CHILD_LEAF`]).
    fn child_tag(&self, ni: usize, c: usize) -> u32 {
        self.nodes[ni * WIDE_NODE_WORDS + 4 + c * 4] >> 24
    }

    /// Word `+2` (interior index or leaf first-primitive) of child `c`.
    fn child_payload_a(&self, ni: usize, c: usize) -> u32 {
        self.nodes[ni * WIDE_NODE_WORDS + 4 + c * 4 + 2]
    }

    /// Word `+3` (leaf primitive count) of child `c`.
    fn child_payload_b(&self, ni: usize, c: usize) -> u32 {
        self.nodes[ni * WIDE_NODE_WORDS + 4 + c * 4 + 3]
    }

    /// Decoded floored lower-corner quantization bytes of child `c` of packed
    /// node `ni` (one byte per axis, low byte first).
    #[cfg(test)]
    fn child_qlo_bytes(&self, ni: usize, c: usize) -> [u8; 3] {
        let w = self.nodes[ni * WIDE_NODE_WORDS + 4 + c * 4];
        [
            (w & 0xff) as u8,
            ((w >> 8) & 0xff) as u8,
            ((w >> 16) & 0xff) as u8,
        ]
    }

    /// Decoded ceiled upper-corner quantization bytes of child `c` of packed
    /// node `ni` (one byte per axis, low byte first).
    #[cfg(test)]
    fn child_qhi_bytes(&self, ni: usize, c: usize) -> [u8; 3] {
        let w = self.nodes[ni * WIDE_NODE_WORDS + 4 + c * 4 + 1];
        [
            (w & 0xff) as u8,
            ((w >> 8) & 0xff) as u8,
            ((w >> 16) & 0xff) as u8,
        ]
    }

    /// Decodes packed triangle `i`.
    fn triangle(&self, i: usize) -> Triangle {
        let b = i * WIDE_TRIANGLE_WORDS;
        Triangle::new(
            read_vec3(&self.triangles, b),
            read_vec3(&self.triangles, b + 3),
            read_vec3(&self.triangles, b + 6),
            self.triangles[b + 9],
        )
    }

    /// Nearest [`Hit`] over the packed layout, reproducing
    /// [`WideBvh::closest_hit`] bit-for-bit.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        self.walk_closest(ray, intersect_triangle)
    }

    /// Watertight nearest [`Hit`], reproducing
    /// [`WideBvh::closest_hit_watertight`] bit-for-bit.
    #[must_use]
    pub fn closest_hit_watertight(&self, ray: &Ray) -> Option<Hit> {
        self.walk_closest(ray, intersect_triangle_watertight)
    }

    /// True when any primitive intersects `ray` (Möller–Trumbore).
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        self.walk_any(ray, intersect_triangle)
    }

    /// True when any primitive intersects `ray` (watertight).
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray) -> bool {
        self.walk_any(ray, intersect_triangle_watertight)
    }

    /// Packed nearest-hit walk mirroring [`WideBvh`] exactly: near-first child
    /// ordering and `t_max` shrink so the result matches bit-for-bit.
    fn walk_closest(
        &self,
        ray: &Ray,
        test: fn(&Ray, &Triangle) -> Option<(f32, f32, f32)>,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<Hit> = None;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_index) = stack.pop() {
            let ni = node_index as usize;
            let count = self.node_child_count(ni);
            let mut order: [(f32, usize); WIDE_BRANCHING] = [(0.0, 0); WIDE_BRANCHING];
            let mut hit_count = 0usize;
            for c in 0..count {
                let bounds = self.child_bounds(ni, c);
                if let Some((t_enter, _)) = ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()) {
                    order[hit_count] = (t_enter, c);
                    hit_count += 1;
                }
            }
            order[..hit_count].sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
            for &(_, c) in &order[..hit_count] {
                match self.child_tag(ni, c) {
                    CHILD_INTERIOR => stack.push(self.child_payload_a(ni, c)),
                    CHILD_LEAF => {
                        let start = self.child_payload_a(ni, c) as usize;
                        let end = start + self.child_payload_b(ni, c) as usize;
                        for i in start..end {
                            let tri = self.triangle(i);
                            if let Some((t, u, v)) = test(&ray, &tri) {
                                best = Some(Hit {
                                    t,
                                    u,
                                    v,
                                    primitive: tri.primitive,
                                });
                                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), t);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        best
    }

    /// Packed any-hit walk mirroring [`WideBvh`].
    fn walk_any(
        &self,
        ray: &Ray,
        test: fn(&Ray, &Triangle) -> Option<(f32, f32, f32)>,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_index) = stack.pop() {
            let ni = node_index as usize;
            let count = self.node_child_count(ni);
            for c in 0..count {
                let bounds = self.child_bounds(ni, c);
                if ray
                    .aabb_interval(&bounds, ray.t_min(), ray.t_max())
                    .is_none()
                {
                    continue;
                }
                match self.child_tag(ni, c) {
                    CHILD_INTERIOR => stack.push(self.child_payload_a(ni, c)),
                    CHILD_LEAF => {
                        let start = self.child_payload_a(ni, c) as usize;
                        let end = start + self.child_payload_b(ni, c) as usize;
                        for i in start..end {
                            let tri = self.triangle(i);
                            if test(ray, &tri).is_some() {
                                return true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::{Bvh, Triangle};

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites.
    struct Rng(u64);
    impl Rng {
        /// Seeds the generator (forcing a non-zero state).
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
        /// Uniform value in `[0, 1]`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        /// Uniform value in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Random general-position triangle scene (mirrors the wide-`BVH` suite).
    fn random_scene(n: u32, seed: u64) -> Vec<Triangle> {
        let mut rng = Rng::new(seed);
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
                Triangle::new(p(&mut rng), p(&mut rng), p(&mut rng), id)
            })
            .collect()
    }

    /// The packed geometry (origin, per-axis scale, child corners, tags, and
    /// payloads) must decode to exactly the in-memory [`WideBvh`] node data.
    #[test]
    fn packed_geometry_matches_in_memory() {
        let tris = random_scene(500, 0x51de_51de_1234_5678);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let gpu = GpuWideBvh::from_wide(&wide);
        assert_eq!(gpu.node_count(), wide.node_count());
        assert_eq!(gpu.triangle_count(), wide.primitive_count());

        for (ni, node) in wide.nodes().iter().enumerate() {
            for (packed, orig) in gpu.node_origin(ni).into_iter().zip(node.origin()) {
                assert_eq!(packed.to_bits(), orig.to_bits());
            }
            // Scale round-trips exactly (power-of-two mantissa is zero).
            for (packed, orig) in gpu.node_scale(ni).into_iter().zip(node.scale()) {
                assert_eq!(packed.to_bits(), orig.to_bits());
            }
            assert_eq!(gpu.node_child_count(ni), node.child_count() as usize);
            for c in 0..node.child_count() as usize {
                assert_eq!(gpu.child_qlo_bytes(ni, c), node.child_qlo(c));
                assert_eq!(gpu.child_qhi_bytes(ni, c), node.child_qhi(c));
                match node.child_kind(c) {
                    WideChild::Empty => panic!("populated slot decoded as empty"),
                    WideChild::Interior(idx) => {
                        assert_eq!(gpu.child_tag(ni, c), CHILD_INTERIOR);
                        assert_eq!(gpu.child_payload_a(ni, c), idx);
                    }
                    WideChild::Leaf { first, count } => {
                        assert_eq!(gpu.child_tag(ni, c), CHILD_LEAF);
                        assert_eq!(gpu.child_payload_a(ni, c), first);
                        assert_eq!(gpu.child_payload_b(ni, c), u32::from(count));
                    }
                }
            }
        }
    }

    /// The packed walk must reproduce the in-memory wide walk — and therefore
    /// the binary `BVH` nearest hit — bit-for-bit.
    #[test]
    fn packed_closest_hit_matches_cpu_walks_bit_for_bit() {
        let tris = random_scene(600, 0xabcd_1234_5678_9f0e);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let gpu = GpuWideBvh::from_wide(&wide);
        let mut rng = Rng::new(0x0f0e_0d0c_0b0a_0908);
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
            let want_binary = bvh.closest_hit(&ray);
            let want_wide = wide.closest_hit(&ray);
            let got = gpu.closest_hit(&ray);
            assert_eq!(
                want_binary.map(|h| h.t.to_bits()),
                want_wide.map(|h| h.t.to_bits())
            );
            match (want_wide, got) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                    assert_eq!(a.u.to_bits(), b.u.to_bits());
                    assert_eq!(a.v.to_bits(), b.v.to_bits());
                    assert_eq!(a.primitive, b.primitive);
                    hits += 1;
                }
                (a, b) => panic!("packed disagreement: {a:?} vs {b:?}"),
            }
        }
        assert!(hits > 200, "expected many hits, got {hits}");
    }

    /// Watertight + any-hit packed walks must agree with the in-memory wide
    /// walk on every ray.
    #[test]
    fn packed_watertight_and_any_hit_match_cpu() {
        let tris = random_scene(400, 0x7777_3333_dddd_9999);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let gpu = GpuWideBvh::from_wide(&wide);
        let mut rng = Rng::new(0x1122_3344_5566_7788);
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
            let ray = Ray::infinite(origin, dir);
            match (wide.closest_hit_watertight(&ray), gpu.closest_hit_watertight(&ray)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                    assert_eq!(a.primitive, b.primitive);
                }
                (a, b) => panic!("watertight disagreement: {a:?} vs {b:?}"),
            }
            assert_eq!(wide.any_hit(&ray), gpu.any_hit(&ray));
            assert_eq!(wide.any_hit_watertight(&ray), gpu.any_hit_watertight(&ray));
        }
    }

    /// An empty wide tree packs to empty buffers that never hit.
    #[test]
    fn empty_wide_bvh_packs_empty() {
        let bvh = Bvh::build(&[]);
        let wide = WideBvh::from_bvh(&bvh);
        let gpu = GpuWideBvh::from_wide(&wide);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }
}
