//! Stackless (threaded / "escape-index") `BVH` traversal.
//!
//! The stack-based walk in [`super::traversal`] keeps a per-ray traversal stack,
//! which on a `GPU` costs scarce per-thread registers or shared memory and caps
//! the depth a warp can descend. Production hardware ray tracers instead prefer
//! a *stackless* walk over a threaded `BVH`: each node stores a single **escape
//! index** — the node to jump to when this node's subtree is skipped (either
//! because its bounds were missed or because a leaf was just processed). A ray
//! then needs only one cursor and no stack at all.
//!
//! [`BvhEscapeTable`] precomputes one escape index per [`super::bvh::Bvh`] node
//! from the depth-first flattened layout, and the walks here reproduce the exact
//! nearest-hit result of [`Bvh::closest_hit`]/[`Bvh::any_hit`] (and their
//! watertight variants). Because the nearest hit is the global minimum `t`, the
//! fixed left-before-right visit order of the stackless walk yields a
//! bit-for-bit identical [`Hit`] even though it visits nodes in a different
//! order than the ray-ordered stack walk. The escape table is itself a flat
//! `Vec<u32>` the `GPU` kernel binds alongside the packed nodes.

use super::bvh::Bvh;
use super::traversal::{intersect_triangle, intersect_triangle_watertight, Hit, Ray};

/// Sentinel escape index meaning "traversal complete".
///
/// Chosen as `u32::MAX` so it can never collide with a real node index (a `BVH`
/// with `u32::MAX` nodes is not representable in the buffers this mirrors).
pub const ESCAPE_SENTINEL: u32 = u32::MAX;

/// Per-node escape (skip) indices for stackless traversal of a [`Bvh`].
///
/// Index `i` holds the node to jump to when the subtree rooted at node `i` is
/// skipped; the root's escape is [`ESCAPE_SENTINEL`]. The table is parallel to
/// [`Bvh::nodes`] and is uploadable as a single `array<u32>`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BvhEscapeTable {
    /// One escape index per node, parallel to [`Bvh::nodes`].
    escape: Vec<u32>,
}

impl BvhEscapeTable {
    /// Builds the escape table for `bvh` from its depth-first node layout.
    ///
    /// For an interior node with left child `i + 1` and right child
    /// `second_child`, the left child's escape is the right child index and the
    /// right child's escape is the parent's escape, so skipping any subtree
    /// jumps directly past it. Empty input yields an empty table.
    #[must_use]
    pub fn build(bvh: &Bvh) -> Self {
        let nodes = bvh.nodes();
        let mut escape = vec![ESCAPE_SENTINEL; nodes.len()];
        if nodes.is_empty() {
            return Self { escape };
        }
        // Iterative depth-first assignment: (node_index, escape_for_node).
        let mut stack = [(0u32, ESCAPE_SENTINEL); 64];
        let mut sp = 0usize;
        stack[sp] = (0, ESCAPE_SENTINEL);
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let (i, esc) = stack[sp];
            escape[i as usize] = esc;
            let node = &nodes[i as usize];
            if !node.is_leaf() {
                let left = i + 1;
                let right = node.second_child;
                // Right child escapes to this node's escape; left child escapes
                // to the right child. Push right first so left is popped first
                // (matches the depth-first left-before-right order).
                if sp < stack.len() {
                    stack[sp] = (right, esc);
                    sp += 1;
                }
                if sp < stack.len() {
                    stack[sp] = (left, right);
                    sp += 1;
                }
            }
        }
        Self { escape }
    }

    /// The raw escape indices, parallel to [`Bvh::nodes`].
    #[must_use]
    pub fn escape_indices(&self) -> &[u32] {
        &self.escape
    }

    /// Number of escape entries (equals the node count of the source `BVH`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.escape.len()
    }

    /// True when the table (and thus the source `BVH`) is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.escape.is_empty()
    }

    /// Nearest intersection along `ray`, walked stacklessly over `bvh`.
    ///
    /// Returns the same [`Hit`] as [`Bvh::closest_hit`] using the same
    /// [`intersect_triangle`] test; only the node visit order differs.
    #[must_use]
    pub fn closest_hit(&self, bvh: &Bvh, ray: &Ray) -> Option<Hit> {
        self.closest_hit_with(bvh, ray, intersect_triangle)
    }

    /// Nearest intersection using the watertight leaf test; mirrors
    /// [`Bvh::closest_hit_watertight`].
    #[must_use]
    pub fn closest_hit_watertight(&self, bvh: &Bvh, ray: &Ray) -> Option<Hit> {
        self.closest_hit_with(bvh, ray, intersect_triangle_watertight)
    }

    /// Shared stackless nearest-hit walk parameterized by the leaf test.
    fn closest_hit_with(
        &self,
        bvh: &Bvh,
        ray: &Ray,
        test: fn(&Ray, &super::bvh::Triangle) -> Option<(f32, f32, f32)>,
    ) -> Option<Hit> {
        let nodes = bvh.nodes();
        if nodes.is_empty() {
            return None;
        }
        let primitives = bvh.primitives();
        let mut ray = *ray;
        let mut best: Option<Hit> = None;

        let mut node_index = 0u32;
        while node_index != ESCAPE_SENTINEL {
            let ni = node_index as usize;
            let node = &nodes[ni];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for tri in &primitives[start..end] {
                        if let Some((t, u, v)) = test(&ray, tri) {
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
                    // Descend to the left child, which is the next node in the
                    // depth-first layout.
                    node_index += 1;
                }
            } else {
                node_index = self.escape[ni];
            }
        }
        best
    }

    /// True when *any* triangle intersects `ray`; mirrors [`Bvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, bvh: &Bvh, ray: &Ray) -> bool {
        self.any_hit_with(bvh, ray, intersect_triangle)
    }

    /// Occlusion query using the watertight leaf test; mirrors
    /// [`Bvh::any_hit_watertight`].
    #[must_use]
    pub fn any_hit_watertight(&self, bvh: &Bvh, ray: &Ray) -> bool {
        self.any_hit_with(bvh, ray, intersect_triangle_watertight)
    }

    /// Shared stackless occlusion walk parameterized by the leaf test.
    fn any_hit_with(
        &self,
        bvh: &Bvh,
        ray: &Ray,
        test: fn(&Ray, &super::bvh::Triangle) -> Option<(f32, f32, f32)>,
    ) -> bool {
        let nodes = bvh.nodes();
        if nodes.is_empty() {
            return false;
        }
        let primitives = bvh.primitives();

        let mut node_index = 0u32;
        while node_index != ESCAPE_SENTINEL {
            let ni = node_index as usize;
            let node = &nodes[ni];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for tri in &primitives[start..end] {
                        if test(ray, tri).is_some() {
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
    fn empty_bvh_yields_empty_table_that_never_hits() {
        let bvh = Bvh::build(&[]);
        let ropes = BvhEscapeTable::build(&bvh);
        assert!(ropes.is_empty());
        assert_eq!(ropes.len(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(ropes.closest_hit(&bvh, &ray).is_none());
        assert!(!ropes.any_hit(&bvh, &ray));
    }

    #[test]
    fn root_escape_is_the_sentinel() {
        let mut rng = Rng::new(0x1111_2222);
        let tris = random_scene(&mut rng, 32);
        let bvh = Bvh::build(&tris);
        let ropes = BvhEscapeTable::build(&bvh);
        assert_eq!(ropes.len(), bvh.node_count());
        assert_eq!(ropes.escape_indices()[0], ESCAPE_SENTINEL);
    }

    #[test]
    fn escape_indices_are_valid_nodes_or_sentinel() {
        let mut rng = Rng::new(0x3333_4444);
        let tris = random_scene(&mut rng, 50);
        let bvh = Bvh::build(&tris);
        let ropes = BvhEscapeTable::build(&bvh);
        let n = bvh.node_count() as u32;
        for &e in ropes.escape_indices() {
            assert!(e == ESCAPE_SENTINEL || e < n, "escape {e} out of range");
        }
    }

    #[test]
    fn stackless_matches_stack_walk_bit_for_bit() {
        let mut rng = Rng::new(0xDEAD_C0DE);
        let tris = random_scene(&mut rng, 96);
        let bvh = Bvh::build(&tris);
        let ropes = BvhEscapeTable::build(&bvh);

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

            let expected = bvh.closest_hit(&ray);
            let actual = ropes.closest_hit(&bvh, &ray);
            assert_eq!(expected, actual, "stackless closest_hit disagrees");

            let expected_w = bvh.closest_hit_watertight(&ray);
            let actual_w = ropes.closest_hit_watertight(&bvh, &ray);
            assert_eq!(expected_w, actual_w, "watertight stackless disagrees");

            assert_eq!(ropes.any_hit(&bvh, &ray), bvh.any_hit(&ray));
            assert_eq!(
                ropes.any_hit_watertight(&bvh, &ray),
                bvh.any_hit_watertight(&ray)
            );
        }
    }
}
