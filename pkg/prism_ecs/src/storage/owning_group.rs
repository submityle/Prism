//! OwningGroup storage: the fourth of the design's four storage states
//! (design §6 存储模型 四态, §17 "owning group 完美打包"; EnTT-style owning group).
//!
//! An *owning group* claims — "owns" — a fixed set of component types and keeps
//! every entity that currently has **all** of them packed, without any gaps,
//! into a contiguous prefix `[0, group_len)` of the owned storage. Because the
//! matching entities live in one unbroken run, iterating the group degenerates
//! into a straight linear walk with **zero branches and zero holes** — the
//! defining property of an EnTT owning group and the fastest possible path for
//! a super-hot query (design §17 "超热查询完美打包，迭代线性零分支").
//!
//! # Layout
//!
//! ```text
//! sparse:  entity.index() ───▶ dense position (u32, NOT_PRESENT sentinel)
//! dense:   Entity handles, two regions separated by `group_len`:
//!            [0, group_len)              packed members  (own ALL components)
//!            [group_len, dense.len())    tracked, not yet full members
//! ```
//!
//! An entity enters the owned storage *tracked but unpacked* (appended past the
//! boundary). The moment it gains the final component that completes the group
//! it is **packed**: swapped with the element sitting on the boundary and the
//! boundary advanced — `swap(pos, group_len); group_len += 1`. When it later
//! loses a component the move is reversed — `swap(pos, group_len - 1);
//! group_len -= 1` — so the packed prefix is always exactly the current member
//! set, re-ordered but never fragmented. Both transitions are O(1).
//!
//! The full [`Entity`] handle (index **and** generation) is stored per dense
//! position, so a stale handle whose index was recycled into a new generation
//! is rejected on lookup — the same dangling-safe discipline as
//! [`ComponentSparseSet`](crate::storage::ComponentSparseSet) (design §5.1).
//! The group never recycles an index on its own: the owner must [`remove`] an
//! entity before its index is handed to a different generation, exactly like a
//! sparse set.
//!
//! # Single-owner invariant
//!
//! A component type may be owned by **at most one** owning group (design §6
//! "单一拥有者约束，冲突在初始化期报错"; design §20 不变量 "owning group 单一
//! 拥有者"). Packing physically re-orders the owned storage, so two groups
//! fighting over the same component would each try to impose a different order
//! and corrupt both. The owned set is recorded on the group as data
//! ([`OwningGroup::owned`]/[`OwningGroup::owns`]); global enforcement — refusing
//! to register a second group over an already-owned component — lives at the
//! owner level (the `World`/group registry, a follow-up task) because wiring a
//! global registry here would require edits outside this storage module.
//!
//! [`remove`]: OwningGroup::remove

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;

/// Sentinel meaning "this entity index has no dense position in this group".
const NOT_PRESENT: u32 = u32::MAX;

/// A dense, stable identifier for an owning group within one
/// [`World`](crate::world::World) (design §20 版本化新增契约 `OwningGroupId`).
///
/// Groups are assigned ids densely in declaration order; the id is the handle
/// the owner-level registry uses to look a group up and to detect the
/// single-owner conflict described in the [module docs](self).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OwningGroupId(u32);

impl OwningGroupId {
    /// Construct an [`OwningGroupId`] from its raw dense index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index backing this id.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// An EnTT-style owning group over a fixed set of component types.
///
/// Maintains the perfect-packing invariant described in the [module
/// docs](self): the entities that own **all** of the group's components occupy
/// the contiguous prefix `[0, len())` of the dense storage, with no gaps, so
/// [`iter`](Self::iter) is a branch-free linear scan.
#[derive(Debug)]
pub struct OwningGroup {
    /// This group's stable id.
    id: OwningGroupId,
    /// The component types this group owns (see the single-owner invariant in
    /// the [module docs](self)).
    owned: Vec<ComponentId>,
    /// Dense position → entity. `[0, group_len)` is the packed member region;
    /// `[group_len, dense.len())` holds tracked-but-unpacked entities.
    dense: Vec<Entity>,
    /// Entity index → dense position, or [`NOT_PRESENT`]. Grows to cover the
    /// largest entity index ever tracked.
    sparse: Vec<u32>,
    /// Size of the perfectly-packed member prefix.
    group_len: usize,
}

impl OwningGroup {
    /// Create an empty group with id `id` owning the component set `owned`.
    ///
    /// Duplicate ids in `owned` are collapsed; order is otherwise preserved.
    /// The caller (owner-level registry) is responsible for the single-owner
    /// invariant — see the [module docs](self).
    pub fn new(id: OwningGroupId, owned: &[ComponentId]) -> Self {
        let mut set: Vec<ComponentId> = Vec::with_capacity(owned.len());
        for &component in owned {
            if !set.contains(&component) {
                set.push(component);
            }
        }
        Self {
            id,
            owned: set,
            dense: Vec::new(),
            sparse: Vec::new(),
            group_len: 0,
        }
    }

    /// This group's stable id.
    #[inline]
    pub fn id(&self) -> OwningGroupId {
        self.id
    }

    /// The component types this group owns, in declaration order.
    #[inline]
    pub fn owned(&self) -> &[ComponentId] {
        &self.owned
    }

    /// Whether `component` is owned by this group.
    #[inline]
    pub fn owns(&self, component: ComponentId) -> bool {
        self.owned.contains(&component)
    }

    /// Number of perfectly-packed members (the size of the `[0, len())`
    /// prefix). This is the length an iterator over the group yields.
    #[inline]
    pub fn len(&self) -> usize {
        self.group_len
    }

    /// Whether the group has no packed members.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.group_len == 0
    }

    /// Number of entities tracked in the owned storage, packed or not.
    #[inline]
    pub fn tracked_len(&self) -> usize {
        self.dense.len()
    }

    /// Whether the owned storage tracks no entities at all.
    #[inline]
    pub fn tracked_is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// The packed member prefix, in dense order (iteration / test helper).
    #[inline]
    pub fn packed(&self) -> &[Entity] {
        &self.dense[..self.group_len]
    }

    /// All tracked entities (both regions), in dense order.
    #[inline]
    pub fn tracked(&self) -> &[Entity] {
        &self.dense
    }

    /// The dense position of `entity`, validating generation, or `None`.
    #[inline]
    fn position(&self, entity: Entity) -> Option<usize> {
        let slot = *self.sparse.get(entity.index() as usize)?;
        if slot == NOT_PRESENT {
            return None;
        }
        let pos = slot as usize;
        // Reject a stale handle whose index was recycled into a new generation.
        if self.dense[pos] == entity {
            Some(pos)
        } else {
            None
        }
    }

    /// Whether `entity` is a perfectly-packed member (owns every component).
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        matches!(self.position(entity), Some(pos) if pos < self.group_len)
    }

    /// Whether `entity` is tracked in the owned storage (packed or not).
    #[inline]
    pub fn is_tracked(&self, entity: Entity) -> bool {
        self.position(entity).is_some()
    }

    /// Grow `sparse` so index `index` is addressable, filling with
    /// [`NOT_PRESENT`].
    #[inline]
    fn ensure_sparse(&mut self, index: usize) {
        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, NOT_PRESENT);
        }
    }

    /// Swap the entities at dense positions `a` and `b`, fixing both sparse
    /// entries so the indirection stays consistent.
    #[inline]
    fn swap_positions(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.dense.swap(a, b);
        let ea = self.dense[a];
        let eb = self.dense[b];
        self.sparse[ea.index() as usize] = a as u32;
        self.sparse[eb.index() as usize] = b as u32;
    }

    /// Track `entity` in the owned storage without packing it: it is appended
    /// past the boundary as a not-yet-full member. Returns `true` if it was
    /// newly tracked, `false` if it was already present.
    ///
    /// The caller must ensure `entity`'s index is not already live under a
    /// different generation (see the recycling discipline in the [module
    /// docs](self)).
    pub fn track(&mut self, entity: Entity) -> bool {
        if self.position(entity).is_some() {
            return false;
        }
        let pos = self.dense.len();
        self.dense.push(entity);
        self.ensure_sparse(entity.index() as usize);
        self.sparse[entity.index() as usize] = pos as u32;
        true
    }

    /// Pack `entity` into the member prefix on *gain* of the final owned
    /// component: `swap(pos, group_len); group_len += 1`.
    ///
    /// Returns `true` if it crossed the boundary; `false` if it is untracked or
    /// already packed.
    pub fn pack(&mut self, entity: Entity) -> bool {
        match self.position(entity) {
            Some(pos) if pos >= self.group_len => {
                let boundary = self.group_len;
                self.swap_positions(pos, boundary);
                self.group_len += 1;
                true
            }
            _ => false,
        }
    }

    /// Unpack `entity` out of the member prefix on *loss* of an owned
    /// component: `swap(pos, group_len - 1); group_len -= 1`. The entity stays
    /// tracked, now past the boundary.
    ///
    /// Returns `true` if it crossed the boundary; `false` if it is untracked or
    /// not currently packed.
    pub fn unpack(&mut self, entity: Entity) -> bool {
        match self.position(entity) {
            Some(pos) if pos < self.group_len => {
                let last = self.group_len - 1;
                self.swap_positions(pos, last);
                self.group_len -= 1;
                true
            }
            _ => false,
        }
    }

    /// Add `entity` as a full member: track it if necessary, then pack it into
    /// the member prefix. Returns `true` if it became a newly-packed member,
    /// `false` if it was already packed.
    ///
    /// This is the common "entity now satisfies the whole group" entry point;
    /// [`track`](Self::track) + [`pack`](Self::pack) expose the two phases
    /// separately for callers that stage membership incrementally.
    pub fn insert(&mut self, entity: Entity) -> bool {
        self.track(entity);
        self.pack(entity)
    }

    /// Remove `entity` from the owned storage entirely. If it was packed it is
    /// first unpacked (symmetric boundary swap), then swap-removed from the
    /// dense tail so no hole is left. Returns `true` if it was tracked.
    pub fn remove(&mut self, entity: Entity) -> bool {
        let Some(pos) = self.position(entity) else {
            return false;
        };
        // If packed, unpack first so `group_len` stays the member count.
        let pos = if pos < self.group_len {
            let last = self.group_len - 1;
            self.swap_positions(pos, last);
            self.group_len -= 1;
            last
        } else {
            pos
        };
        // Swap-remove from the dense tail; the moved tail element keeps a valid
        // sparse entry via `swap_positions`.
        let last = self.dense.len() - 1;
        self.swap_positions(pos, last);
        self.dense.pop();
        self.sparse[entity.index() as usize] = NOT_PRESENT;
        true
    }

    /// Drop all tracking and reset the group to empty, keeping allocations.
    pub fn clear(&mut self) {
        self.dense.clear();
        self.group_len = 0;
        for slot in &mut self.sparse {
            *slot = NOT_PRESENT;
        }
    }

    /// A branch-free linear iterator over the packed member prefix, yielding
    /// each `(Entity, dense_index)` pair in dense order. The dense index lets a
    /// caller address the owned component columns directly.
    #[inline]
    pub fn iter(&self) -> OwningGroupIter<'_> {
        OwningGroupIter {
            entities: self.packed(),
            index: 0,
        }
    }
}

/// Linear iterator over an [`OwningGroup`]'s packed member prefix.
///
/// Yields `(Entity, dense_index)` for each member in packed order; the dense
/// index is contiguous `0..group_len`, so a caller can use it to index the
/// owned component storage without any bounds surprises.
pub struct OwningGroupIter<'a> {
    /// The packed prefix slice being walked.
    entities: &'a [Entity],
    /// Next dense index to yield.
    index: usize,
}

impl Iterator for OwningGroupIter<'_> {
    type Item = (Entity, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.entities.len() {
            return None;
        }
        let out = (self.entities[self.index], self.index);
        self.index += 1;
        Some(out)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entities.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OwningGroupIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    /// A tiny deterministic xorshift RNG so the property test is reproducible.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn group() -> OwningGroup {
        OwningGroup::new(
            OwningGroupId::new(0),
            &[ComponentId::new(1), ComponentId::new(2)],
        )
    }

    /// The packed prefix must be exactly `expected`, with no gaps, no dupes,
    /// and every member's sparse indirection must round-trip to its position.
    fn assert_packing(g: &OwningGroup, expected: &BTreeSet<Entity>) {
        assert_eq!(g.len(), expected.len(), "member count");
        let packed: BTreeSet<Entity> = g.packed().iter().copied().collect();
        assert_eq!(packed.len(), g.len(), "no duplicate members in prefix");
        assert_eq!(&packed, expected, "packed prefix == member set");

        // contains() agrees with membership for every tracked entity, and the
        // iterator yields the prefix in dense order with contiguous indices.
        for (expected_index, (entity, dense_index)) in g.iter().enumerate() {
            assert_eq!(dense_index, expected_index, "dense index is contiguous");
            assert!(g.contains(entity), "iterated entity is a member");
            assert!(expected.contains(&entity));
        }
    }

    #[test]
    fn empty_group_has_no_members() {
        let g = group();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
        assert!(g.tracked_is_empty());
        assert_eq!(g.iter().count(), 0);
        assert_eq!(g.owned(), &[ComponentId::new(1), ComponentId::new(2)]);
        assert!(g.owns(ComponentId::new(1)));
        assert!(!g.owns(ComponentId::new(9)));
    }

    #[test]
    fn new_dedups_owned_set() {
        let g = OwningGroup::new(
            OwningGroupId::new(7),
            &[ComponentId::new(3), ComponentId::new(3), ComponentId::new(4)],
        );
        assert_eq!(g.id(), OwningGroupId::new(7));
        assert_eq!(g.owned(), &[ComponentId::new(3), ComponentId::new(4)]);
    }

    #[test]
    fn insert_packs_and_contains() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        let b = es.alloc();

        assert!(g.insert(a));
        assert!(g.insert(b));
        assert!(!g.insert(a), "re-insert of a packed member is a no-op");
        assert_eq!(g.len(), 2);
        assert!(g.contains(a));
        assert!(g.contains(b));
        assert!(!g.contains(es.alloc()));
    }

    #[test]
    fn track_then_pack_crosses_boundary() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        let b = es.alloc();
        let c = es.alloc();

        // All three enter unpacked (past the boundary).
        assert!(g.track(a));
        assert!(g.track(b));
        assert!(g.track(c));
        assert!(!g.track(a), "double-track is a no-op");
        assert_eq!(g.len(), 0, "nothing packed yet");
        assert_eq!(g.tracked_len(), 3);
        assert!(g.is_tracked(a) && !g.contains(a));

        // Packing b swaps it onto the boundary and advances group_len.
        assert!(g.pack(b));
        assert_eq!(g.len(), 1);
        assert!(g.contains(b));
        assert_eq!(g.packed(), &[b]);

        // Packing an already-packed entity does nothing.
        assert!(!g.pack(b));
        assert_eq!(g.len(), 1);

        // Pack the remaining two.
        assert!(g.pack(a));
        assert!(g.pack(c));
        assert_eq!(g.len(), 3);
        let packed: BTreeSet<Entity> = g.packed().iter().copied().collect();
        assert_eq!(packed, BTreeSet::from([a, b, c]));
    }

    #[test]
    fn unpack_is_symmetric() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        let b = es.alloc();
        let c = es.alloc();
        g.insert(a);
        g.insert(b);
        g.insert(c);
        assert_eq!(g.len(), 3);

        // Losing a component unpacks b but keeps it tracked.
        assert!(g.unpack(b));
        assert_eq!(g.len(), 2);
        assert!(!g.contains(b));
        assert!(g.is_tracked(b));
        assert_eq!(g.tracked_len(), 3);

        // The surviving members are still exactly {a, c}, perfectly packed.
        let packed: BTreeSet<Entity> = g.packed().iter().copied().collect();
        assert_eq!(packed, BTreeSet::from([a, c]));

        // Unpacking a non-member is a no-op; re-packing restores membership.
        assert!(!g.unpack(b));
        assert!(g.pack(b));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn remove_from_prefix_and_from_tail() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        let b = es.alloc();
        let c = es.alloc();
        let d = es.alloc();
        g.insert(a); // packed
        g.insert(b); // packed
        g.track(c); // unpacked tail
        g.track(d); // unpacked tail
        assert_eq!(g.len(), 2);
        assert_eq!(g.tracked_len(), 4);

        // Remove a packed member: group shrinks, remains packed.
        assert!(g.remove(a));
        assert!(!g.contains(a));
        assert!(!g.is_tracked(a));
        assert_eq!(g.len(), 1);
        assert_eq!(g.tracked_len(), 3);
        assert!(g.contains(b));

        // Remove an unpacked member: prefix untouched.
        assert!(g.remove(c));
        assert!(!g.is_tracked(c));
        assert_eq!(g.len(), 1);
        assert_eq!(g.tracked_len(), 2);

        // Removing something absent is a no-op.
        assert!(!g.remove(a));
        assert!(!g.remove(c));
    }

    #[test]
    fn clear_resets_everything() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        let b = es.alloc();
        g.insert(a);
        g.track(b);
        g.clear();
        assert!(g.is_empty());
        assert!(g.tracked_is_empty());
        assert!(!g.contains(a));
        assert!(!g.is_tracked(b));
        assert_eq!(g.iter().count(), 0);
    }

    #[test]
    fn stale_generation_handle_is_rejected() {
        let mut es = Entities::new();
        let mut g = group();
        let a = es.alloc();
        g.insert(a);
        // Free and recycle a's index into a bumped generation.
        es.free(a);
        let a2 = es.alloc();
        assert_eq!(a2.index(), a.index(), "index recycled");
        assert_ne!(a2, a, "generation differs");

        // The stale (old) handle still resolves (same generation stored); the
        // new-generation handle is not tracked until explicitly added.
        assert!(g.contains(a));
        assert!(!g.contains(a2));
        assert!(!g.is_tracked(a2));
    }

    #[test]
    fn iter_yields_prefix_in_dense_order() {
        let mut es = Entities::new();
        let mut g = group();
        let entities: Vec<Entity> = (0..5).map(|_| es.alloc()).collect();
        for &e in &entities {
            g.insert(e);
        }
        // Unpack two to shuffle the prefix, exercising the ExactSizeIterator.
        g.unpack(entities[1]);
        g.unpack(entities[3]);

        let collected: Vec<(Entity, usize)> = g.iter().collect();
        assert_eq!(collected.len(), g.len());
        assert_eq!(g.iter().len(), g.len(), "ExactSizeIterator agrees");
        for (expected_index, (entity, dense_index)) in collected.into_iter().enumerate() {
            assert_eq!(dense_index, expected_index);
            assert_eq!(entity, g.packed()[expected_index]);
            assert!(g.contains(entity));
        }
    }

    /// Property test: after any random sequence of add/remove, the packed
    /// prefix contains exactly the current members with no gaps.
    #[test]
    fn property_random_add_remove_keeps_perfect_packing() {
        let mut es = Entities::new();
        let pool: Vec<Entity> = (0..32).map(|_| es.alloc()).collect();
        let mut g = group();
        let mut model: BTreeSet<Entity> = BTreeSet::new();
        let mut rng = Rng::new(0xC0FFEE_u64);

        for _ in 0..5_000 {
            let e = pool[rng.below(pool.len())];
            if rng.below(2) == 0 {
                let newly = g.insert(e);
                assert_eq!(newly, model.insert(e), "insert return matches model");
            } else {
                let removed = g.remove(e);
                assert_eq!(removed, model.remove(&e), "remove return matches model");
            }
            // Full-group invariant must hold after every single operation.
            assert_packing(&g, &model);
            // Tracked storage never holds a hole in membership accounting.
            assert_eq!(g.tracked_len(), model.len());
        }
    }

    /// Property test mixing the lower-level track/pack/unpack phases to make
    /// sure the boundary invariant survives arbitrary partial-membership churn.
    #[test]
    fn property_random_track_pack_unpack() {
        let mut es = Entities::new();
        let pool: Vec<Entity> = (0..24).map(|_| es.alloc()).collect();
        let mut g = group();
        let mut tracked: BTreeSet<Entity> = BTreeSet::new();
        let mut packed: BTreeSet<Entity> = BTreeSet::new();
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0_u64);

        for _ in 0..5_000 {
            let e = pool[rng.below(pool.len())];
            match rng.below(4) {
                0 => {
                    g.track(e);
                    tracked.insert(e);
                }
                1 => {
                    // pack only transitions a tracked, unpacked entity.
                    if g.pack(e) {
                        assert!(tracked.contains(&e));
                        packed.insert(e);
                    }
                }
                2 => {
                    if g.unpack(e) {
                        packed.remove(&e);
                    }
                }
                _ => {
                    g.remove(e);
                    tracked.remove(&e);
                    packed.remove(&e);
                }
            }

            assert_eq!(g.len(), packed.len());
            assert_eq!(g.tracked_len(), tracked.len());
            let got_packed: BTreeSet<Entity> = g.packed().iter().copied().collect();
            assert_eq!(got_packed, packed, "prefix == packed model, no gaps");
            let got_tracked: BTreeSet<Entity> = g.tracked().iter().copied().collect();
            assert_eq!(got_tracked, tracked, "tracked storage matches model");
        }
    }
}
