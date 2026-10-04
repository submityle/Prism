//! Correctness tests for the parallel linear `BVH`.
//!
//! The linear builder must produce a hierarchy that is (a) structurally valid
//! for the shared depth-first traversal and (b) returns *identical* ray hits to
//! the quality `SAH` builder, since both index the same exact triangle set.

use alloc::vec::Vec;

use crate::ray_scene::bvh::{Bvh, LinearBvhNode, Triangle};
use crate::ray_scene::traversal::Ray;

use super::LinearBvh;
use super::morton::{MORTON_GRID_MAX, MortonQuantizer, morton3d};
use super::radix_sort::{MortonEntry, radix_sort};

/// Tiny deterministic xorshift generator so tests need no external crate.
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

    /// Uniform `f32` in `[-span, span]`.
    fn signed(&mut self, span: f32) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        (unit * 2.0 - 1.0) * span
    }
}

fn random_triangles(rng: &mut Rng, count: u32) -> Vec<Triangle> {
    (0..count)
        .map(|primitive| {
            let base = [rng.signed(50.0), rng.signed(50.0), rng.signed(50.0)];
            let edge = |r: &mut Rng| {
                [
                    base[0] + r.signed(2.0),
                    base[1] + r.signed(2.0),
                    base[2] + r.signed(2.0),
                ]
            };
            Triangle::new(edge(rng), edge(rng), edge(rng), primitive)
        })
        .collect()
}

/// Depth-first tree height of the flattened node array, counting edges + 1.
fn tree_height(nodes: &[LinearBvhNode]) -> usize {
    fn walk(nodes: &[LinearBvhNode], index: usize) -> usize {
        let node = nodes[index];
        if node.is_leaf() {
            return 1;
        }
        let left = walk(nodes, index + 1);
        let right = walk(nodes, node.second_child as usize);
        1 + left.max(right)
    }
    if nodes.is_empty() {
        0
    } else {
        walk(nodes, 0)
    }
}

#[test]
fn empty_input_yields_empty_hierarchy() {
    let bvh = LinearBvh::build(&[]);
    assert!(bvh.is_empty());
    assert_eq!(bvh.node_count(), 0);
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    assert!(bvh.closest_hit(&ray).is_none());
}

#[test]
fn single_triangle_is_one_leaf() {
    let tri = Triangle::new([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], 7);
    let bvh = LinearBvh::build(&[tri]);
    assert_eq!(bvh.node_count(), 1);
    assert_eq!(bvh.primitive_count(), 1);
    assert!(bvh.nodes()[0].is_leaf());
    let ray = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let hit = bvh.closest_hit(&ray).expect("ray hits the triangle");
    assert_eq!(hit.primitive, 7);
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn node_and_primitive_counts_are_canonical() {
    let mut rng = Rng::new(0x1234_5678);
    for &count in &[2_u32, 3, 8, 37, 256] {
        let tris = random_triangles(&mut rng, count);
        let bvh = LinearBvh::build(&tris);
        // A full binary tree over `count` leaves has `2*count - 1` nodes.
        assert_eq!(bvh.node_count(), (2 * count - 1) as usize);
        assert_eq!(bvh.primitive_count(), count as usize);
        // Every leaf owns exactly one primitive and all primitives appear once.
        let mut seen = alloc::vec![false; count as usize];
        let mut leaves = 0_u32;
        for node in bvh.nodes() {
            if node.is_leaf() {
                assert_eq!(node.primitive_count, 1);
                let pos = node.first_primitive as usize;
                assert!(!seen[pos], "primitive slot {pos} referenced twice");
                seen[pos] = true;
                leaves += 1;
            }
        }
        assert_eq!(leaves, count);
        assert!(seen.into_iter().all(|s| s));
    }
}

#[test]
fn interior_bounds_enclose_children() {
    let mut rng = Rng::new(0x9e37_79b9);
    let tris = random_triangles(&mut rng, 200);
    let bvh = LinearBvh::build(&tris);
    let nodes = bvh.nodes();
    for (index, node) in nodes.iter().enumerate() {
        if node.is_leaf() {
            continue;
        }
        for child in [index + 1, node.second_child as usize] {
            let c = nodes[child].bounds;
            for axis in 0..3 {
                assert!(node.bounds.min[axis] <= c.min[axis] + 1e-4);
                assert!(node.bounds.max[axis] >= c.max[axis] - 1e-4);
            }
        }
    }
}

#[test]
fn height_fits_traversal_stack() {
    // The stack-based traversal uses a fixed 64-entry stack; a well-formed
    // Morton tree stays far below that for any realistic primitive count.
    let mut rng = Rng::new(0xdead_beef);
    let tris = random_triangles(&mut rng, 4096);
    let bvh = LinearBvh::build(&tris);
    assert!(tree_height(bvh.nodes()) < 64);
}

#[test]
fn duplicate_morton_keys_stay_well_formed() {
    // All triangles share a centroid, so every Morton key is identical; the
    // index tie-break must still yield a valid, balanced single-prim-leaf tree.
    let tris: Vec<Triangle> = (0..64)
        .map(|primitive| {
            Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], primitive)
        })
        .collect();
    let bvh = LinearBvh::build(&tris);
    assert_eq!(bvh.node_count(), 2 * 64 - 1);
    assert!(tree_height(bvh.nodes()) < 32);
}

#[test]
fn closest_hits_match_sah_builder() {
    let mut rng = Rng::new(0x00c0_ffee);
    let tris = random_triangles(&mut rng, 300);
    let linear = LinearBvh::build(&tris);
    let sah = Bvh::build(&tris);

    let mut ray_rng = Rng::new(0x0bad_f00d);
    let mut tested = 0;
    for _ in 0..2000 {
        let origin = [
            ray_rng.signed(60.0),
            ray_rng.signed(60.0),
            ray_rng.signed(60.0),
        ];
        let dir = [
            ray_rng.signed(1.0),
            ray_rng.signed(1.0),
            ray_rng.signed(1.0),
        ];
        if dir[0].abs() + dir[1].abs() + dir[2].abs() < 1e-3 {
            continue;
        }
        let ray = Ray::new(origin, dir, 0.0, f32::INFINITY);
        let a = linear.closest_hit(&ray);
        let b = sah.closest_hit(&ray);
        match (a, b) {
            (None, None) => {}
            (Some(ha), Some(hb)) => {
                assert_eq!(ha.primitive, hb.primitive, "different primitive hit");
                assert!((ha.t - hb.t).abs() < 1e-3, "t mismatch {} vs {}", ha.t, hb.t);
            }
            _ => panic!("hit/miss disagreement between linear and SAH builders"),
        }
        tested += 1;
    }
    assert!(tested > 100, "expected a meaningful number of valid rays");
}

#[test]
fn any_hits_match_sah_builder() {
    let mut rng = Rng::new(0x5a5a_1234);
    let tris = random_triangles(&mut rng, 180);
    let linear = LinearBvh::build(&tris);
    let sah = Bvh::build(&tris);
    let mut ray_rng = Rng::new(0x1357_9bdf);
    for _ in 0..2000 {
        let origin = [
            ray_rng.signed(60.0),
            ray_rng.signed(60.0),
            ray_rng.signed(60.0),
        ];
        let dir = [
            ray_rng.signed(1.0),
            ray_rng.signed(1.0),
            ray_rng.signed(1.0),
        ];
        if dir[0].abs() + dir[1].abs() + dir[2].abs() < 1e-3 {
            continue;
        }
        let ray = Ray::new(origin, dir, 0.0, f32::INFINITY);
        assert_eq!(linear.any_hit(&ray), sah.any_hit(&ray));
    }
}

#[test]
fn morton_keys_round_trip_through_sort() {
    // A hand-built key set must come out of the radix sort fully ordered and
    // carrying its payloads, since the tree builder depends on both.
    let mut entries = alloc::vec![
        MortonEntry { key: morton3d(5, 1, 9), primitive: 0 },
        MortonEntry { key: morton3d(0, 0, 0), primitive: 1 },
        MortonEntry { key: morton3d(MORTON_GRID_MAX, MORTON_GRID_MAX, MORTON_GRID_MAX), primitive: 2 },
        MortonEntry { key: morton3d(3, 7, 2), primitive: 3 },
    ];
    radix_sort(&mut entries);
    assert!(entries.windows(2).all(|w| w[0].key <= w[1].key));
    // The all-zero key must sort first and the all-max key last.
    assert_eq!(entries[0].primitive, 1);
    assert_eq!(entries[entries.len() - 1].primitive, 2);
}

#[test]
fn quantizer_pins_degenerate_axis_to_zero() {
    // A flat (zero-extent) Z axis must not produce a non-finite coordinate.
    let q = MortonQuantizer::new([0.0, 0.0, 5.0], [10.0, 10.0, 5.0]);
    let key_lo = q.key([0.0, 0.0, 5.0]);
    let key_hi = q.key([10.0, 10.0, 5.0]);
    assert!(key_lo <= key_hi);
    // Points outside the bounds clamp instead of overflowing the grid.
    let _clamped = q.key([1e9, -1e9, 5.0]);
}
