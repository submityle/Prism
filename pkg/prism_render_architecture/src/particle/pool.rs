//! Per-emitter particle pool: layout, free-list allocation, and compaction.
//!
//! Every emitter owns one fixed-capacity pool (design §5.2, §11). On the `GPU`
//! the attribute data is Structure-of-Arrays and the allocator runs in compute
//! against atomic counters; this module is the deterministic `CPU` reference for
//! the *allocation and bookkeeping logic* only. It models:
//!
//! * a **free list** — a stack of dead slot indices that [`ParticlePool::spawn`]
//!   pops and [`ParticlePool::kill`] pushes, so the common case never scans the
//!   whole array;
//! * **atomic counters** — `alive` / `spawn-this-frame` / `dead`, matching the
//!   GPU counters that drive indirect dispatch and draw;
//! * a **capacity policy** — when the pool is full a spawn either drops
//!   (discard the newest) or recycles the oldest live particle, an authored
//!   choice (design §11);
//! * **compaction** — an exclusive prefix-sum stream compaction that rebuilds a
//!   contiguous alive list when fragmentation rises (design §9 step 5), the
//!   same scan a GPU compaction pass performs.
//!
//! Every operation is deterministic and total: an out-of-range slot index is
//! ignored rather than panicking, and killing an already-dead slot is a no-op,
//! so the free list can never contain a duplicate (which would hand the same
//! slot to two spawns). The attribute *buffers* themselves and the GPU
//! indirect-args writes are pending the GPU backend; this layer only decides
//! which slots are live and where they compact to.

use alloc::vec::Vec;

/// What a [`ParticlePool`] does when a spawn is requested but every slot is
/// live (design §11 capacity strategy).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CapacityPolicy {
    /// Drop the new particle: [`ParticlePool::spawn`] returns [`None`].
    DiscardNewest,
    /// Recycle the oldest live particle, freeing its slot for the newcomer.
    RecycleOldest,
}

/// Result of an exclusive prefix-sum compaction of a pool's live set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Compaction {
    /// Live slot indices in ascending slot order — the contiguous alive list a
    /// render pass would consume as instances.
    pub alive_list: Vec<u32>,
    /// Per-slot compacted position: `scatter[slot]` is the index this slot
    /// occupies in `alive_list`, or [`None`] when the slot is dead. This is the
    /// scatter address a GPU compaction kernel writes with.
    pub scatter: Vec<Option<u32>>,
}

impl Compaction {
    /// Number of live slots gathered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.alive_list.len()
    }

    /// Returns `true` when no slot was live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.alive_list.is_empty()
    }
}

/// A fixed-capacity particle pool tracking slot liveness and a free list.
///
/// Slot indices are stable for a particle's lifetime. `birth[slot]` records a
/// monotonically increasing spawn sequence so [`CapacityPolicy::RecycleOldest`]
/// can identify the oldest live particle deterministically (ties break to the
/// lowest slot index).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParticlePool {
    capacity: u32,
    alive: Vec<bool>,
    birth: Vec<u64>,
    free: Vec<u32>,
    alive_count: u32,
    spawn_count: u32,
    next_seq: u64,
}

impl ParticlePool {
    /// Builds an empty pool with room for `capacity` particles.
    ///
    /// The free list is seeded so the first spawns hand out ascending slot
    /// indices (`0, 1, 2, …`), which keeps allocation order reproducible.
    #[must_use]
    pub fn with_capacity(capacity: u32) -> Self {
        let cap = capacity as usize;
        // Push slots high-to-low so the LIFO stack pops 0 first.
        let mut free = Vec::with_capacity(cap);
        for slot in (0..capacity).rev() {
            free.push(slot);
        }
        Self {
            capacity,
            alive: alloc::vec![false; cap],
            birth: alloc::vec![0u64; cap],
            free,
            alive_count: 0,
            spawn_count: 0,
            next_seq: 0,
        }
    }

    /// Maximum number of simultaneously live particles.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Live particles right now.
    #[must_use]
    pub fn alive_count(&self) -> u32 {
        self.alive_count
    }

    /// Particles spawned since the last [`ParticlePool::begin_frame`].
    #[must_use]
    pub fn spawn_count(&self) -> u32 {
        self.spawn_count
    }

    /// Free (dead) slots available for spawning.
    #[must_use]
    pub fn dead_count(&self) -> u32 {
        self.capacity - self.alive_count
    }

    /// Returns `true` when no slot is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// Returns `true` when every slot is live.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.alive_count == self.capacity
    }

    /// Reports whether `slot` is currently live. Out-of-range slots are dead.
    #[must_use]
    pub fn is_alive(&self, slot: u32) -> bool {
        self.alive.get(slot as usize).copied().unwrap_or(false)
    }

    /// Resets the per-frame spawn counter at the top of a simulation frame.
    ///
    /// The alive/free state persists across frames; only `spawn_count` is a
    /// per-frame quantity (it feeds the spawn dispatch size).
    pub fn begin_frame(&mut self) {
        self.spawn_count = 0;
    }

    /// Allocates one slot for a new particle, honoring `policy` when full.
    ///
    /// Returns the slot index, or [`None`] when the pool is full and `policy`
    /// is [`CapacityPolicy::DiscardNewest`]. A zero-capacity pool always drops.
    pub fn spawn(&mut self, policy: CapacityPolicy) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(self.activate(slot));
        }
        // Full pool: either drop or evict the oldest live particle.
        match policy {
            CapacityPolicy::DiscardNewest => None,
            CapacityPolicy::RecycleOldest => {
                let oldest = self.oldest_live_slot()?;
                // Free then re-activate the same slot so the counter stays
                // balanced and the free list never holds a duplicate.
                self.kill(oldest);
                let slot = self.free.pop()?;
                Some(self.activate(slot))
            }
        }
    }

    /// Marks `slot` live, stamps its birth sequence, and bumps the counters.
    fn activate(&mut self, slot: u32) -> u32 {
        let idx = slot as usize;
        self.alive[idx] = true;
        self.birth[idx] = self.next_seq;
        self.next_seq += 1;
        self.alive_count += 1;
        self.spawn_count += 1;
        slot
    }

    /// Kills the particle in `slot`, returning it to the free list.
    ///
    /// Out-of-range slots and already-dead slots are ignored, so the free list
    /// can never gain a duplicate entry (double-free safe).
    pub fn kill(&mut self, slot: u32) {
        let idx = slot as usize;
        match self.alive.get(idx) {
            Some(true) => {}
            _ => return,
        }
        self.alive[idx] = false;
        self.free.push(slot);
        self.alive_count -= 1;
    }

    /// The live slot with the smallest birth sequence, ties broken by lowest
    /// slot index; [`None`] when nothing is live.
    fn oldest_live_slot(&self) -> Option<u32> {
        let mut best: Option<(u64, u32)> = None;
        for (slot, &alive) in self.alive.iter().enumerate() {
            if !alive {
                continue;
            }
            let seq = self.birth[slot];
            let slot = slot as u32;
            match best {
                Some((best_seq, _)) if best_seq <= seq => {}
                _ => best = Some((seq, slot)),
            }
        }
        best.map(|(_, slot)| slot)
    }

    /// Fragmentation of the live set in `0..=1`: the fraction of the occupied
    /// index span (up to the highest live slot) that is actually dead.
    ///
    /// A compacted pool reports `0`; a pool whose survivors are scattered across
    /// a wide index range reports higher, which is the trigger the scheduler
    /// uses to decide whether a [`ParticlePool::compact`] pass is worth it.
    #[must_use]
    pub fn fragmentation(&self) -> f32 {
        let mut highest_live: Option<u32> = None;
        for (slot, &alive) in self.alive.iter().enumerate() {
            if alive {
                highest_live = Some(slot as u32);
            }
        }
        let Some(highest) = highest_live else {
            return 0.0;
        };
        let span = highest + 1;
        let holes = span - self.alive_count;
        holes as f32 / span as f32
    }

    /// Rebuilds a contiguous alive list via an exclusive prefix-sum scan.
    ///
    /// The exclusive scan of the per-slot alive flags gives each live slot its
    /// compacted position, exactly as a `GPU` stream-compaction pass would; the
    /// result is deterministic and independent of allocation history.
    #[must_use]
    pub fn compact(&self) -> Compaction {
        let mut alive_list = Vec::with_capacity(self.alive_count as usize);
        let mut scatter = alloc::vec![None; self.alive.len()];
        let mut running = 0u32;
        for (slot, &alive) in self.alive.iter().enumerate() {
            if alive {
                scatter[slot] = Some(running);
                alive_list.push(slot as u32);
                running += 1;
            }
        }
        Compaction {
            alive_list,
            scatter,
        }
    }

    /// Number of compute workgroups needed to cover the live particles for an
    /// indirect dispatch (design §5.2 `indirect_dispatch`).
    ///
    /// A `workgroup_size` of zero yields `0` rather than dividing by zero.
    #[must_use]
    pub fn indirect_dispatch_groups(&self, workgroup_size: u32) -> u32 {
        if workgroup_size == 0 {
            return 0;
        }
        self.alive_count.div_ceil(workgroup_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_hands_out_ascending_slots() {
        let mut pool = ParticlePool::with_capacity(4);
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), Some(0));
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), Some(1));
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), Some(2));
        assert_eq!(pool.alive_count(), 3);
        assert_eq!(pool.spawn_count(), 3);
        assert_eq!(pool.dead_count(), 1);
    }

    #[test]
    fn kill_returns_slot_to_free_list_for_reuse() {
        let mut pool = ParticlePool::with_capacity(4);
        let a = pool.spawn(CapacityPolicy::DiscardNewest).unwrap();
        let b = pool.spawn(CapacityPolicy::DiscardNewest).unwrap();
        pool.kill(a);
        assert!(!pool.is_alive(a));
        assert_eq!(pool.alive_count(), 1);
        // The just-freed slot is reused before untouched slots.
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), Some(a));
        assert!(pool.is_alive(b));
    }

    #[test]
    fn double_kill_is_a_no_op_and_leaves_no_duplicate() {
        let mut pool = ParticlePool::with_capacity(2);
        let a = pool.spawn(CapacityPolicy::DiscardNewest).unwrap();
        pool.kill(a);
        pool.kill(a); // ignored
                      // Only one real free slot came back, so exactly one spawn reuses `a`
                      // and the pool never hands the same slot out twice.
        let x = pool.spawn(CapacityPolicy::DiscardNewest).unwrap();
        let y = pool.spawn(CapacityPolicy::DiscardNewest).unwrap();
        assert_ne!(x, y);
        assert!(pool.is_full());
    }

    #[test]
    fn out_of_range_kill_does_not_panic() {
        let mut pool = ParticlePool::with_capacity(2);
        pool.spawn(CapacityPolicy::DiscardNewest);
        pool.kill(99);
        assert_eq!(pool.alive_count(), 1);
    }

    #[test]
    fn discard_policy_drops_when_full() {
        let mut pool = ParticlePool::with_capacity(2);
        pool.spawn(CapacityPolicy::DiscardNewest);
        pool.spawn(CapacityPolicy::DiscardNewest);
        assert!(pool.is_full());
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), None);
        assert_eq!(pool.alive_count(), 2);
    }

    #[test]
    fn recycle_oldest_evicts_the_first_spawned() {
        let mut pool = ParticlePool::with_capacity(3);
        let a = pool.spawn(CapacityPolicy::RecycleOldest).unwrap();
        let _b = pool.spawn(CapacityPolicy::RecycleOldest).unwrap();
        let _c = pool.spawn(CapacityPolicy::RecycleOldest).unwrap();
        assert!(pool.is_full());
        // The pool is full; the newcomer reuses the oldest slot `a`.
        let reused = pool.spawn(CapacityPolicy::RecycleOldest).unwrap();
        assert_eq!(reused, a);
        assert!(pool.is_full());
        assert_eq!(pool.alive_count(), 3);
    }

    #[test]
    fn zero_capacity_pool_always_drops() {
        let mut pool = ParticlePool::with_capacity(0);
        assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), None);
        assert_eq!(pool.spawn(CapacityPolicy::RecycleOldest), None);
        assert!(pool.is_empty());
        assert!(pool.is_full());
    }

    #[test]
    fn begin_frame_resets_only_spawn_count() {
        let mut pool = ParticlePool::with_capacity(4);
        pool.spawn(CapacityPolicy::DiscardNewest);
        pool.spawn(CapacityPolicy::DiscardNewest);
        assert_eq!(pool.spawn_count(), 2);
        pool.begin_frame();
        assert_eq!(pool.spawn_count(), 0);
        assert_eq!(pool.alive_count(), 2);
    }

    #[test]
    fn compaction_is_prefix_sum_and_deterministic() {
        let mut pool = ParticlePool::with_capacity(5);
        for _ in 0..5 {
            pool.spawn(CapacityPolicy::DiscardNewest);
        }
        // Kill slots 1 and 3, leaving 0, 2, 4 live.
        pool.kill(1);
        pool.kill(3);
        let c = pool.compact();
        assert_eq!(c.alive_list, [0, 2, 4]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.scatter[0], Some(0));
        assert_eq!(c.scatter[1], None);
        assert_eq!(c.scatter[2], Some(1));
        assert_eq!(c.scatter[3], None);
        assert_eq!(c.scatter[4], Some(2));
        // Recomputing yields the identical result.
        assert_eq!(pool.compact(), c);
    }

    #[test]
    fn empty_pool_compacts_to_empty() {
        let pool = ParticlePool::with_capacity(3);
        let c = pool.compact();
        assert!(c.is_empty());
        assert_eq!(c.alive_list.len(), 0);
    }

    #[test]
    fn fragmentation_zero_when_compact_and_positive_when_holed() {
        let mut pool = ParticlePool::with_capacity(4);
        for _ in 0..4 {
            pool.spawn(CapacityPolicy::DiscardNewest);
        }
        assert!(pool.fragmentation().abs() < 1e-6);
        // Kill slot 1: live set {0,2,3}, span 4, one hole -> 0.25.
        pool.kill(1);
        assert!((pool.fragmentation() - 0.25).abs() < 1e-6);
        // Kill the tail slot 3: live set {0,2}, span 3, one hole -> 1/3.
        pool.kill(3);
        assert!((pool.fragmentation() - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn indirect_dispatch_group_count_rounds_up() {
        let mut pool = ParticlePool::with_capacity(200);
        for _ in 0..130 {
            pool.spawn(CapacityPolicy::DiscardNewest);
        }
        assert_eq!(pool.indirect_dispatch_groups(64), 3);
        assert_eq!(pool.indirect_dispatch_groups(0), 0);
    }
}
