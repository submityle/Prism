//! Core dynamic bounding-volume hierarchy: a dynamic axis-aligned bounding box
//! tree with incremental insert / remove / update, refit, and rotation-based
//! rebalancing.
//!
//! The implementation follows publicly documented dynamic axis-aligned
//! bounding box tree techniques (fat boxes, surface-area-heuristic guided
//! sibling selection, and height-balanced rotations). It contains **no Unreal
//! Engine source or derived code**.

use alloc::vec;
use alloc::vec::Vec;

use crate::bounding::Aabb;
use crate::proxy::ProxyId;

use super::node::{Node, NULL};

/// A dynamic bounding-volume hierarchy used as a broad-phase acceleration
/// structure.
///
/// Leaves store a user-supplied `u64` payload and a *fat* bounding box (the
/// tight box grown by a small margin). Fattening lets a proxy move within its
/// margin without triggering a re-insertion, which keeps updates cheap for
/// slowly moving objects. Internal nodes enclose their two children.
///
/// Nodes live in a single pooled [`Vec`]; freed slots are recycled through a
/// free list, and each slot carries a generation counter so a stale
/// [`ProxyId`] cannot alias a recycled slot.
#[derive(Clone, Debug)]
pub struct DynamicBvh {
    /// Pooled node storage (live nodes and recycled free slots).
    pub(in crate::bvh) nodes: Vec<Node>,
    /// Index of the root node, or [`NULL`] when the tree is empty.
    pub(in crate::bvh) root: u32,
    /// Head of the free-slot list, or [`NULL`] when there are none.
    free_list: u32,
    /// Number of live leaves.
    leaf_count: usize,
}

impl Default for DynamicBvh {
    #[inline]
    fn default() -> Self {
        DynamicBvh::new()
    }
}

impl DynamicBvh {
    /// Margin (in world units) added around each tight box to form its fat box.
    const MARGIN: f32 = 0.1;

    /// Creates an empty tree.
    #[inline]
    pub fn new() -> Self {
        DynamicBvh {
            nodes: Vec::new(),
            root: NULL,
            free_list: NULL,
            leaf_count: 0,
        }
    }

    /// Creates an empty tree with pre-allocated node storage for `capacity`
    /// nodes.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        DynamicBvh {
            nodes: Vec::with_capacity(capacity),
            root: NULL,
            free_list: NULL,
            leaf_count: 0,
        }
    }

    /// Number of live leaves in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.leaf_count
    }

    /// Returns `true` if the tree contains no leaves.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Height of the tree (root height); `0` for an empty or single-leaf tree.
    #[inline]
    pub fn height(&self) -> u32 {
        if self.root == NULL {
            0
        } else {
            self.nodes[self.root as usize].height.max(0) as u32
        }
    }

    /// Removes all leaves and internal nodes, resetting the tree to empty.
    ///
    /// Any previously issued [`ProxyId`] becomes invalid.
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = NULL;
        self.free_list = NULL;
        self.leaf_count = 0;
    }

    /// Inserts a proxy with tight bounds `aabb` and payload `data`.
    ///
    /// The stored box is `aabb` grown by [`DynamicBvh::MARGIN`]. Returns a
    /// handle that stays valid until the proxy is removed.
    pub fn insert(&mut self, aabb: Aabb, data: u64) -> ProxyId {
        let fat = aabb.expanded_by(Self::MARGIN);
        let leaf = self.allocate_node();
        {
            let node = &mut self.nodes[leaf as usize];
            node.aabb = fat;
            node.data = data;
            node.child1 = NULL;
            node.child2 = NULL;
            node.height = 0;
        }
        self.insert_leaf(leaf);
        self.leaf_count += 1;
        let generation = self.nodes[leaf as usize].generation;
        ProxyId::new(leaf, generation)
    }

    /// Removes the proxy referenced by `proxy`.
    ///
    /// Returns `true` if the proxy was live and removed, or `false` if the
    /// handle was stale or never valid.
    pub fn remove(&mut self, proxy: ProxyId) -> bool {
        let Some(leaf) = self.resolve_leaf(proxy) else {
            return false;
        };
        self.remove_leaf(leaf);
        self.free_node(leaf);
        self.leaf_count -= 1;
        true
    }

    /// Updates the proxy's bounds to `new_tight`.
    ///
    /// If the new tight box still fits inside the stored fat box, nothing
    /// changes and this returns `false`. Otherwise the proxy is re-inserted
    /// with a freshly fattened box (the handle stays valid) and this returns
    /// `true`.
    pub fn update(&mut self, proxy: ProxyId, new_tight: Aabb) -> bool {
        let Some(leaf) = self.resolve_leaf(proxy) else {
            return false;
        };
        if self.nodes[leaf as usize].aabb.contains_aabb(&new_tight) {
            return false;
        }
        let data = self.nodes[leaf as usize].data;
        self.remove_leaf(leaf);
        let fat = new_tight.expanded_by(Self::MARGIN);
        {
            let node = &mut self.nodes[leaf as usize];
            node.aabb = fat;
            node.data = data;
            node.child1 = NULL;
            node.child2 = NULL;
            node.height = 0;
        }
        self.insert_leaf(leaf);
        true
    }

    /// Returns the payload of the proxy, or [`None`] if the handle is stale.
    #[inline]
    pub fn get_data(&self, proxy: ProxyId) -> Option<u64> {
        self.resolve_leaf(proxy)
            .map(|leaf| self.nodes[leaf as usize].data)
    }

    /// Returns the stored fat box of the proxy, or [`None`] if the handle is
    /// stale.
    #[inline]
    pub fn get_aabb(&self, proxy: ProxyId) -> Option<Aabb> {
        self.resolve_leaf(proxy)
            .map(|leaf| self.nodes[leaf as usize].aabb)
    }

    /// Validates internal invariants: parent/child links, per-node heights, and
    /// that each internal node encloses its children. Intended for tests.
    pub fn validate(&self) -> bool {
        if self.root == NULL {
            return self.leaf_count == 0;
        }
        if self.nodes[self.root as usize].parent_or_next != NULL {
            return false;
        }
        let mut leaves = 0usize;
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if node.height < 0 {
                return false;
            }
            if node.is_leaf() {
                if node.child2 != NULL || node.height != 0 {
                    return false;
                }
                leaves += 1;
                continue;
            }
            let c1 = node.child1;
            let c2 = node.child2;
            if c1 == NULL || c2 == NULL {
                return false;
            }
            if c1 as usize >= self.nodes.len() || c2 as usize >= self.nodes.len() {
                return false;
            }
            if self.nodes[c1 as usize].parent_or_next != index
                || self.nodes[c2 as usize].parent_or_next != index
            {
                return false;
            }
            let expected = 1 + self.nodes[c1 as usize]
                .height
                .max(self.nodes[c2 as usize].height);
            if node.height != expected {
                return false;
            }
            let merged = self.nodes[c1 as usize]
                .aabb
                .merged(&self.nodes[c2 as usize].aabb);
            if !node.aabb.contains_aabb(&merged) {
                return false;
            }
            stack.push(c1);
            stack.push(c2);
        }
        leaves == self.leaf_count
    }

    /// Collects `(payload, fat box)` for every live leaf. Crate-internal helper
    /// used by the broad-phase.
    pub(crate) fn collect_leaves(&self) -> Vec<(u64, Aabb)> {
        let mut out = Vec::with_capacity(self.leaf_count);
        for node in &self.nodes {
            if node.height >= 0 && node.is_leaf() {
                out.push((node.data, node.aabb));
            }
        }
        out
    }

    /// Resolves a handle to a live leaf slot index, validating the generation.
    fn resolve_leaf(&self, proxy: ProxyId) -> Option<u32> {
        let idx = proxy.index();
        let node = self.nodes.get(idx as usize)?;
        if node.height < 0 || node.generation != proxy.generation() || !node.is_leaf() {
            return None;
        }
        Some(idx)
    }

    /// Allocates a node slot, reusing a free slot when available.
    fn allocate_node(&mut self) -> u32 {
        if self.free_list == NULL {
            let idx = self.nodes.len() as u32;
            self.nodes.push(Node {
                aabb: Aabb::EMPTY,
                parent_or_next: NULL,
                child1: NULL,
                child2: NULL,
                data: 0,
                height: 0,
                generation: 0,
            });
            idx
        } else {
            let idx = self.free_list;
            self.free_list = self.nodes[idx as usize].parent_or_next;
            let node = &mut self.nodes[idx as usize];
            node.parent_or_next = NULL;
            node.child1 = NULL;
            node.child2 = NULL;
            node.height = 0;
            idx
        }
    }

    /// Returns a slot to the free list, bumping its generation.
    fn free_node(&mut self, idx: u32) {
        let head = self.free_list;
        let node = &mut self.nodes[idx as usize];
        node.parent_or_next = head;
        node.height = -1;
        node.generation = node.generation.wrapping_add(1);
        self.free_list = idx;
    }

    /// Cost of descending into `child` when inserting a leaf with `leaf_aabb`.
    fn descend_cost(&self, child: u32, leaf_aabb: &Aabb, inheritance_cost: f32) -> f32 {
        let child_node = self.nodes[child as usize];
        let combined = leaf_aabb.merged(&child_node.aabb);
        if child_node.is_leaf() {
            combined.surface_area() + inheritance_cost
        } else {
            (combined.surface_area() - child_node.aabb.surface_area()) + inheritance_cost
        }
    }

    /// Inserts an already-initialized leaf node into the tree, choosing a
    /// sibling via the surface-area heuristic and rebalancing on the way up.
    fn insert_leaf(&mut self, leaf: u32) {
        if self.root == NULL {
            self.root = leaf;
            self.nodes[leaf as usize].parent_or_next = NULL;
            return;
        }
        let leaf_aabb = self.nodes[leaf as usize].aabb;

        // Descend to the best sibling.
        let mut index = self.root;
        while !self.nodes[index as usize].is_leaf() {
            let child1 = self.nodes[index as usize].child1;
            let child2 = self.nodes[index as usize].child2;

            let area = self.nodes[index as usize].aabb.surface_area();
            let combined = self.nodes[index as usize].aabb.merged(&leaf_aabb);
            let combined_area = combined.surface_area();

            let cost = 2.0 * combined_area;
            let inheritance_cost = 2.0 * (combined_area - area);

            let cost1 = self.descend_cost(child1, &leaf_aabb, inheritance_cost);
            let cost2 = self.descend_cost(child2, &leaf_aabb, inheritance_cost);

            if cost < cost1 && cost < cost2 {
                break;
            }
            index = if cost1 < cost2 { child1 } else { child2 };
        }
        let sibling = index;

        // Create a new parent for `sibling` and `leaf`.
        let old_parent = self.nodes[sibling as usize].parent_or_next;
        let new_parent = self.allocate_node();
        let sib_aabb = self.nodes[sibling as usize].aabb;
        {
            let node = &mut self.nodes[new_parent as usize];
            node.parent_or_next = old_parent;
            node.data = 0;
            node.aabb = leaf_aabb.merged(&sib_aabb);
            node.child1 = sibling;
            node.child2 = leaf;
        }
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;
        self.nodes[sibling as usize].parent_or_next = new_parent;
        self.nodes[leaf as usize].parent_or_next = new_parent;

        if old_parent != NULL {
            if self.nodes[old_parent as usize].child1 == sibling {
                self.nodes[old_parent as usize].child1 = new_parent;
            } else {
                self.nodes[old_parent as usize].child2 = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        // Walk back up: rebalance, then refit heights and boxes.
        let mut index = self.nodes[leaf as usize].parent_or_next;
        while index != NULL {
            index = self.balance(index);
            let child1 = self.nodes[index as usize].child1;
            let child2 = self.nodes[index as usize].child2;
            let a1 = self.nodes[child1 as usize].aabb;
            let a2 = self.nodes[child2 as usize].aabb;
            let h1 = self.nodes[child1 as usize].height;
            let h2 = self.nodes[child2 as usize].height;
            self.nodes[index as usize].height = 1 + h1.max(h2);
            self.nodes[index as usize].aabb = a1.merged(&a2);
            index = self.nodes[index as usize].parent_or_next;
        }
    }

    /// Detaches a leaf from the tree (without freeing its slot), rebalancing
    /// and refitting the affected ancestors.
    fn remove_leaf(&mut self, leaf: u32) {
        if leaf == self.root {
            self.root = NULL;
            return;
        }
        let parent = self.nodes[leaf as usize].parent_or_next;
        let grand_parent = self.nodes[parent as usize].parent_or_next;
        let sibling = if self.nodes[parent as usize].child1 == leaf {
            self.nodes[parent as usize].child2
        } else {
            self.nodes[parent as usize].child1
        };

        if grand_parent != NULL {
            if self.nodes[grand_parent as usize].child1 == parent {
                self.nodes[grand_parent as usize].child1 = sibling;
            } else {
                self.nodes[grand_parent as usize].child2 = sibling;
            }
            self.nodes[sibling as usize].parent_or_next = grand_parent;
            self.free_node(parent);

            let mut index = grand_parent;
            while index != NULL {
                index = self.balance(index);
                let child1 = self.nodes[index as usize].child1;
                let child2 = self.nodes[index as usize].child2;
                let a1 = self.nodes[child1 as usize].aabb;
                let a2 = self.nodes[child2 as usize].aabb;
                let h1 = self.nodes[child1 as usize].height;
                let h2 = self.nodes[child2 as usize].height;
                self.nodes[index as usize].aabb = a1.merged(&a2);
                self.nodes[index as usize].height = 1 + h1.max(h2);
                index = self.nodes[index as usize].parent_or_next;
            }
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].parent_or_next = NULL;
            self.free_node(parent);
        }
    }

    /// Performs a single height-balancing rotation at `ia` if its subtree is
    /// unbalanced, returning the (possibly new) subtree root index.
    fn balance(&mut self, ia: u32) -> u32 {
        if self.nodes[ia as usize].is_leaf() || self.nodes[ia as usize].height < 2 {
            return ia;
        }
        let ib = self.nodes[ia as usize].child1;
        let ic = self.nodes[ia as usize].child2;
        let balance = self.nodes[ic as usize].height - self.nodes[ib as usize].height;

        // Rotate C (child2) up.
        if balance > 1 {
            let if_ = self.nodes[ic as usize].child1;
            let ig = self.nodes[ic as usize].child2;
            let a_parent = self.nodes[ia as usize].parent_or_next;

            self.nodes[ic as usize].child1 = ia;
            self.nodes[ic as usize].parent_or_next = a_parent;
            self.nodes[ia as usize].parent_or_next = ic;

            if a_parent != NULL {
                if self.nodes[a_parent as usize].child1 == ia {
                    self.nodes[a_parent as usize].child1 = ic;
                } else {
                    self.nodes[a_parent as usize].child2 = ic;
                }
            } else {
                self.root = ic;
            }

            let b_aabb = self.nodes[ib as usize].aabb;
            let f_aabb = self.nodes[if_ as usize].aabb;
            let g_aabb = self.nodes[ig as usize].aabb;
            let bh = self.nodes[ib as usize].height;
            let fh = self.nodes[if_ as usize].height;
            let gh = self.nodes[ig as usize].height;

            if fh > gh {
                self.nodes[ic as usize].child2 = if_;
                self.nodes[ia as usize].child2 = ig;
                self.nodes[ig as usize].parent_or_next = ia;
                let a_new = b_aabb.merged(&g_aabb);
                self.nodes[ia as usize].aabb = a_new;
                self.nodes[ic as usize].aabb = a_new.merged(&f_aabb);
                self.nodes[ia as usize].height = 1 + bh.max(gh);
                self.nodes[ic as usize].height = 1 + self.nodes[ia as usize].height.max(fh);
            } else {
                self.nodes[ic as usize].child2 = ig;
                self.nodes[ia as usize].child2 = if_;
                self.nodes[if_ as usize].parent_or_next = ia;
                let a_new = b_aabb.merged(&f_aabb);
                self.nodes[ia as usize].aabb = a_new;
                self.nodes[ic as usize].aabb = a_new.merged(&g_aabb);
                self.nodes[ia as usize].height = 1 + bh.max(fh);
                self.nodes[ic as usize].height = 1 + self.nodes[ia as usize].height.max(gh);
            }
            return ic;
        }

        // Rotate B (child1) up.
        if balance < -1 {
            let id = self.nodes[ib as usize].child1;
            let ie = self.nodes[ib as usize].child2;
            let a_parent = self.nodes[ia as usize].parent_or_next;

            self.nodes[ib as usize].child1 = ia;
            self.nodes[ib as usize].parent_or_next = a_parent;
            self.nodes[ia as usize].parent_or_next = ib;

            if a_parent != NULL {
                if self.nodes[a_parent as usize].child1 == ia {
                    self.nodes[a_parent as usize].child1 = ib;
                } else {
                    self.nodes[a_parent as usize].child2 = ib;
                }
            } else {
                self.root = ib;
            }

            let c_aabb = self.nodes[ic as usize].aabb;
            let d_aabb = self.nodes[id as usize].aabb;
            let e_aabb = self.nodes[ie as usize].aabb;
            let ch = self.nodes[ic as usize].height;
            let dh = self.nodes[id as usize].height;
            let eh = self.nodes[ie as usize].height;

            if dh > eh {
                self.nodes[ib as usize].child2 = id;
                self.nodes[ia as usize].child1 = ie;
                self.nodes[ie as usize].parent_or_next = ia;
                let a_new = c_aabb.merged(&e_aabb);
                self.nodes[ia as usize].aabb = a_new;
                self.nodes[ib as usize].aabb = a_new.merged(&d_aabb);
                self.nodes[ia as usize].height = 1 + ch.max(eh);
                self.nodes[ib as usize].height = 1 + self.nodes[ia as usize].height.max(dh);
            } else {
                self.nodes[ib as usize].child2 = ie;
                self.nodes[ia as usize].child1 = id;
                self.nodes[id as usize].parent_or_next = ia;
                let a_new = c_aabb.merged(&d_aabb);
                self.nodes[ia as usize].aabb = a_new;
                self.nodes[ib as usize].aabb = a_new.merged(&e_aabb);
                self.nodes[ia as usize].height = 1 + ch.max(dh);
                self.nodes[ib as usize].height = 1 + self.nodes[ia as usize].height.max(eh);
            }
            return ib;
        }

        ia
    }
}

#[cfg(test)]
mod tests {
    use super::DynamicBvh;
    use crate::bounding::{Aabb, Ray};
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;
    use glam::Vec3;

    /// A tiny deterministic xorshift generator so tests are reproducible
    /// without external crates.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }
    }

    fn random_box(rng: &mut Rng) -> Aabb {
        let c = Vec3::new(
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
        );
        let h = Vec3::new(
            rng.range(0.5, 4.0),
            rng.range(0.5, 4.0),
            rng.range(0.5, 4.0),
        );
        Aabb::from_center_half_extents(c, h)
    }

    #[test]
    fn empty_tree_invariants() {
        let tree = DynamicBvh::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.height(), 0);
        assert!(tree.validate());
    }

    #[test]
    fn query_matches_brute_force() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut tree = DynamicBvh::with_capacity(256);
        let mut fats: Vec<(u64, Aabb)> = Vec::new();

        for i in 0..200u64 {
            let tight = random_box(&mut rng);
            let id = tree.insert(tight, i);
            fats.push((i, tree.get_aabb(id).unwrap()));
        }
        assert!(tree.validate());
        assert_eq!(tree.len(), 200);

        for _ in 0..50 {
            let q = random_box(&mut rng);
            let mut got: BTreeSet<u64> = tree.query_aabb_collect(q).into_iter().collect();
            let expected: BTreeSet<u64> = fats
                .iter()
                .filter(|(_, a)| a.intersects(&q))
                .map(|(id, _)| *id)
                .collect();
            assert_eq!(got, expected);
            got.clear();
        }
    }

    #[test]
    fn remove_updates_membership_and_validity() {
        let mut rng = Rng(0xdead_beef_0000_0001);
        let mut tree = DynamicBvh::new();
        let mut ids = Vec::new();
        for i in 0..120u64 {
            ids.push(tree.insert(random_box(&mut rng), i));
        }
        assert!(tree.validate());

        // Remove every third proxy.
        let mut removed_count = 0usize;
        for (k, id) in ids.iter().enumerate() {
            if k % 3 == 0 {
                assert!(tree.remove(*id));
                removed_count += 1;
            }
        }
        // Removing again must fail (stale handle), and lookups must miss.
        for (k, id) in ids.iter().enumerate() {
            if k % 3 == 0 {
                assert!(!tree.remove(*id));
                assert!(tree.get_data(*id).is_none());
                assert!(tree.get_aabb(*id).is_none());
            } else {
                assert!(tree.get_data(*id).is_some());
            }
        }
        assert_eq!(tree.len(), ids.len() - removed_count);
        assert!(tree.validate());
    }

    #[test]
    fn update_noop_vs_reinsert() {
        let mut tree = DynamicBvh::new();
        let tight = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let id = tree.insert(tight, 42);

        // Tiny move stays inside the fat box -> no reinsert.
        let small = Aabb::from_center_half_extents(Vec3::splat(0.05), Vec3::splat(1.0));
        assert!(!tree.update(id, small));

        // Large move escapes the fat box -> reinsert, handle stays valid.
        let far = Aabb::from_center_half_extents(Vec3::splat(20.0), Vec3::splat(1.0));
        assert!(tree.update(id, far));
        assert_eq!(tree.get_data(id), Some(42));
        assert!(tree.get_aabb(id).unwrap().contains_aabb(&far));
        assert!(tree.validate());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn many_inserts_and_removes_stay_valid() {
        let mut rng = Rng(0x00c0_ffee_1234_5678);
        let mut tree = DynamicBvh::new();
        let mut live: Vec<crate::proxy::ProxyId> = Vec::new();
        for i in 0..400u64 {
            live.push(tree.insert(random_box(&mut rng), i));
            if live.len() > 3 && rng.next_f32() < 0.4 {
                let victim = (rng.next_u64() as usize) % live.len();
                let id = live.swap_remove(victim);
                assert!(tree.remove(id));
            }
            assert!(tree.validate());
        }
        assert_eq!(tree.len(), live.len());
    }

    #[test]
    fn ray_cast_hits_expected_leaves() {
        let mut tree = DynamicBvh::new();
        // Boxes along +x at x = 0,10,20 near the x-axis.
        let a = tree.insert(
            Aabb::from_center_half_extents(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(1.0)),
            0,
        );
        let b = tree.insert(
            Aabb::from_center_half_extents(Vec3::new(10.0, 0.0, 0.0), Vec3::splat(1.0)),
            1,
        );
        let _c = tree.insert(
            Aabb::from_center_half_extents(Vec3::new(20.0, 50.0, 0.0), Vec3::splat(1.0)),
            2,
        );

        let ray = Ray::with_tmax(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 100.0);
        let mut hit: BTreeSet<u64> = BTreeSet::new();
        tree.ray_cast(&ray, &mut |data, _aabb| {
            hit.insert(data);
        });
        assert!(hit.contains(&0));
        assert!(hit.contains(&1));
        assert!(!hit.contains(&2));
        let _ = (a, b);
    }
}
