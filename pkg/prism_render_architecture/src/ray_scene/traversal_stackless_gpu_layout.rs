//! Flat, `GPU`-uploadable stackless (threaded / escape-index) `BVH`.
//!
//! This is the `GPU` companion of [`super::traversal_stackless`]: it pairs the
//! packed node/triangle buffers from [`super::gpu_layout::GpuBvhBuffers`] with a
//! parallel **escape-index** array (one `u32` per node, from
//! [`super::traversal_stackless::BvhEscapeTable`]) and walks them with a single
//! ray cursor and no per-thread stack. A real kernel binds three flat buffers —
//! packed nodes (`array<u32>`), packed triangles (`array<u32>`), and the escape
//! array (`array<u32>`) — and needs neither a traversal stack nor near/far child
//! sorting: it descends to the left child (`node + 1`) on a bounds hit and jumps
//! to the escape index on a miss or after a leaf.
//!
//! Because the visit order is a fixed left-before-right descent and the nearest
//! hit is the global minimum `t`, the packed walks here reproduce
//! [`super::traversal_stackless::BvhEscapeTable::closest_hit`] — and therefore
//! [`super::bvh::Bvh::closest_hit`] — bit-for-bit, so the kernel that binds these
//! buffers can be diffed against deterministic `CPU` arithmetic.

use super::bvh::{Aabb, Bvh};
use super::gpu_layout::{GpuBvhBuffers, NODE_WORDS, TRIANGLE_WORDS};
use super::traversal::{intersect_triangle, intersect_triangle_watertight, Hit, Ray};
use super::traversal_stackless::{BvhEscapeTable, ESCAPE_SENTINEL};

/// Word offset of `first_primitive` inside a packed node record.
const NODE_FIRST_PRIMITIVE: usize = 6;
/// Word offset of `primitive_count` inside a packed node record.
const NODE_PRIMITIVE_COUNT: usize = 8;
/// Word offset of the `primitive` id inside a packed triangle record.
const TRIANGLE_PRIMITIVE: usize = 9;

/// Reads three consecutive `from_bits` words at `words[base..base + 3]`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

/// Packed, `GPU`-uploadable stackless `BVH`: packed geometry plus the escape
/// (skip) index array that replaces the traversal stack.
///
/// The `buffers` hold the same [`NODE_WORDS`]/[`TRIANGLE_WORDS`] records the
/// stack-based [`GpuBvhBuffers`] kernel consumes; `escape` is parallel to the
/// packed nodes and is uploaded as a single `array<u32>`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuStacklessBvh {
    /// Packed node/triangle buffers, shared with the stack-based layout.
    buffers: GpuBvhBuffers,
    /// One escape index per packed node, parallel to `buffers.nodes`.
    escape: Vec<u32>,
}

impl GpuStacklessBvh {
    /// Packs `bvh` into stackless `GPU` buffers.
    ///
    /// Reuses [`GpuBvhBuffers::from_bvh`] for the node/triangle words and
    /// [`BvhEscapeTable::build`] for the escape array, so the packed geometry is
    /// byte-identical to the stack-based upload and only the escape array is
    /// added.
    #[must_use]
    pub fn from_bvh(bvh: &Bvh) -> Self {
        Self {
            buffers: GpuBvhBuffers::from_bvh(bvh),
            escape: BvhEscapeTable::build(bvh).escape_indices().to_vec(),
        }
    }

    /// The packed node/triangle buffers (shared with the stack-based layout).
    #[must_use]
    pub fn buffers(&self) -> &GpuBvhBuffers {
        &self.buffers
    }

    /// The escape indices, parallel to the packed nodes.
    #[must_use]
    pub fn escape_indices(&self) -> &[u32] {
        &self.escape
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.buffers.nodes.len() / NODE_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.nodes.is_empty()
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(
            read_vec3(&self.buffers.nodes, b),
            read_vec3(&self.buffers.nodes, b + 3),
        )
    }

    /// Nearest intersection along `ray`, walked stacklessly over the packed
    /// buffers; mirrors [`BvhEscapeTable::closest_hit`] bit-for-bit.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        self.closest_hit_with(ray, intersect_triangle)
    }

    /// Nearest intersection using the watertight leaf test; mirrors
    /// [`BvhEscapeTable::closest_hit_watertight`].
    #[must_use]
    pub fn closest_hit_watertight(&self, ray: &Ray) -> Option<Hit> {
        self.closest_hit_with(ray, intersect_triangle_watertight)
    }

    /// Shared packed stackless nearest-hit walk parameterized by the leaf test.
    fn closest_hit_with(
        &self,
        ray: &Ray,
        test: fn(&Ray, &super::bvh::Triangle) -> Option<(f32, f32, f32)>,
    ) -> Option<Hit> {
        if self.buffers.nodes.is_empty() {
            return None;
        }
        let nodes = &self.buffers.nodes;
        let triangles = &self.buffers.triangles;
        let mut ray = *ray;
        let mut best: Option<Hit> = None;

        let mut node_index = 0u32;
        while node_index != ESCAPE_SENTINEL {
            let ni = node_index as usize;
            let base = ni * NODE_WORDS;
            if ray
                .aabb_interval(&self.node_bounds(ni), ray.t_min(), ray.t_max())
                .is_some()
            {
                let primitive_count = nodes[base + NODE_PRIMITIVE_COUNT];
                if primitive_count > 0 {
                    let start = nodes[base + NODE_FIRST_PRIMITIVE] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        let tb = pi * TRIANGLE_WORDS;
                        let tri = super::bvh::Triangle::new(
                            read_vec3(triangles, tb),
                            read_vec3(triangles, tb + 3),
                            read_vec3(triangles, tb + 6),
                            triangles[tb + TRIANGLE_PRIMITIVE],
                        );
                        if let Some((t, u, v)) = test(&ray, &tri) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), t);
                            best = Some(Hit {
                                t,
                                u,
                                v,
                                primitive: tri.primitive,
                            });
                        }
                    }
                    node_index = self.escape[ni];
                } else {
                    node_index += 1;
                }
            } else {
                node_index = self.escape[ni];
            }
        }
        best
    }

    /// True when *any* triangle intersects `ray`; mirrors
    /// [`BvhEscapeTable::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        self.any_hit_with(ray, intersect_triangle)
    }

    /// Occlusion query using the watertight leaf test; mirrors
    /// [`BvhEscapeTable::any_hit_watertight`].
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray) -> bool {
        self.any_hit_with(ray, intersect_triangle_watertight)
    }

    /// Shared packed stackless occlusion walk parameterized by the leaf test.
    fn any_hit_with(
        &self,
        ray: &Ray,
        test: fn(&Ray, &super::bvh::Triangle) -> Option<(f32, f32, f32)>,
    ) -> bool {
        if self.buffers.nodes.is_empty() {
            return false;
        }
        let nodes = &self.buffers.nodes;
        let triangles = &self.buffers.triangles;

        let mut node_index = 0u32;
        while node_index != ESCAPE_SENTINEL {
            let ni = node_index as usize;
            let base = ni * NODE_WORDS;
            if ray
                .aabb_interval(&self.node_bounds(ni), ray.t_min(), ray.t_max())
                .is_some()
            {
                let primitive_count = nodes[base + NODE_PRIMITIVE_COUNT];
                if primitive_count > 0 {
                    let start = nodes[base + NODE_FIRST_PRIMITIVE] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        let tb = pi * TRIANGLE_WORDS;
                        let tri = super::bvh::Triangle::new(
                            read_vec3(triangles, tb),
                            read_vec3(triangles, tb + 3),
                            read_vec3(triangles, tb + 6),
                            triangles[tb + TRIANGLE_PRIMITIVE],
                        );
                        if test(ray, &tri).is_some() {
                            return true;
                        }
                    }
                    node_index = self.escape[ni];
                } else {
                    node_index += 1;
                }
            } else {
                node_index = self.escape[ni];
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::Triangle;

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

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Triangle> {
        (0..count)
            .map(|i| {
                let base = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let edge = |rng: &mut Rng| {
                    [
                        base[0] + rng.range(-1.5, 1.5),
                        base[1] + rng.range(-1.5, 1.5),
                        base[2] + rng.range(-1.5, 1.5),
                    ]
                };
                Triangle::new(base, edge(rng), edge(rng), i)
            })
            .collect()
    }

    #[test]
    fn empty_bvh_yields_empty_packed_table() {
        let bvh = Bvh::build(&[]);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert!(gpu.escape_indices().is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn escape_array_matches_cpu_table_and_geometry_matches_stack_layout() {
        let mut rng = Rng::new(0x5151_2727);
        let tris = random_scene(&mut rng, 64);
        let bvh = Bvh::build(&tris);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);

        // Escape array is byte-identical to the CPU escape table.
        let cpu = BvhEscapeTable::build(&bvh);
        assert_eq!(gpu.escape_indices(), cpu.escape_indices());
        assert_eq!(gpu.node_count(), bvh.node_count());

        // Packed geometry is byte-identical to the stack-based upload.
        let stack_layout = GpuBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.buffers().nodes, stack_layout.nodes);
        assert_eq!(gpu.buffers().triangles, stack_layout.triangles);
    }

    #[test]
    fn packed_stackless_matches_cpu_walks_bit_for_bit() {
        let mut rng = Rng::new(0x0FF1_CE99);
        let tris = random_scene(&mut rng, 96);
        let bvh = Bvh::build(&tris);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);
        let cpu = BvhEscapeTable::build(&bvh);

        for _ in 0..4_000 {
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

            // Packed stackless == CPU stackless == CPU stack walk, bit-for-bit.
            assert_eq!(gpu.closest_hit(&ray), cpu.closest_hit(&bvh, &ray));
            assert_eq!(gpu.closest_hit(&ray), bvh.closest_hit(&ray));
            assert_eq!(
                gpu.closest_hit_watertight(&ray),
                bvh.closest_hit_watertight(&ray)
            );
            assert_eq!(gpu.any_hit(&ray), bvh.any_hit(&ray));
            assert_eq!(gpu.any_hit_watertight(&ray), bvh.any_hit_watertight(&ray));
        }
    }
}
