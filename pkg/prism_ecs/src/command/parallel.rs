//! Parallel command buffers with deterministic sort-replay (design §9, M2).
//!
//! [`CommandQueue`](super::CommandQueue) is a single shared buffer: fine for one
//! producer, but a contention point when many systems/workers record at once.
//! [`ParallelCommandBuffers`] gives every producer its own independent buffer so
//! the record hot path takes **no shared lock**. The cost of avoiding shared
//! state is reintroduced only at the sync point, where all buffers are merged
//! and replayed.
//!
//! # Determinism invariant
//!
//! Every recorded command is stamped with a [`CommandKey`] of
//! `(producer, seq)`:
//!
//! * `producer` — the buffer's **explicit, caller-assigned index** (not a
//!   thread id). Two runs that assign the same work to the same producer index
//!   get the same `producer` regardless of which OS thread executes it.
//! * `seq` — the command's 0-based ordinal *within its own buffer*, assigned at
//!   record time. Within one producer, `seq` is strictly increasing in record
//!   order, so a producer's own insertion order is always preserved.
//!
//! At the sync point [`ParallelCommandBuffers::apply`] gathers every command
//! into one list and performs a **stable sort by `CommandKey`**. Because
//! `CommandKey` orders by `producer` first and `seq` second, and because each
//! producer's `seq` values are already in record order, the merged replay order
//! is the total order "all of producer 0 in order, then all of producer 1 in
//! order, …". This order depends only on the explicit producer indices and the
//! per-producer record order — **never on thread scheduling or on the order in
//! which the buffers happened to be touched**. Same recorded inputs ⇒ identical
//! replay order and identical resulting world state, every time.
//!
//! # No-std and parallelism
//!
//! The core type is `no_std`: producers are addressed by explicit index, so no
//! thread-id machinery is required. [`ParallelCommandBuffers::split_with`] hands
//! out one disjoint, [`Send`] [`ParallelCommands`] handle per buffer, which is
//! exactly what lets each worker thread record into its own buffer without
//! locking. A `std`-only [`ParallelCommandBuffers::index_for_current_thread`]
//! convenience is provided behind the `std` feature for callers that want to
//! map the running thread onto a producer slot.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::Command;
use crate::bundle::Bundle;
use crate::entity::{Entities, Entity};
use crate::world::World;

/// The deterministic sort key stamped onto every parallel command.
///
/// Ordered by [`producer`](CommandKey::producer) first and
/// [`seq`](CommandKey::seq) second; this ordering is exactly the replay order
/// produced by [`ParallelCommandBuffers::apply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommandKey {
    /// The caller-assigned index of the buffer this command was recorded into.
    pub producer: u32,
    /// The 0-based ordinal of this command within its own buffer, in record
    /// order.
    pub seq: u32,
}

/// One producer's independent command buffer.
///
/// Commands are stored in record order, so a command's index in `commands` is
/// its [`CommandKey::seq`].
struct ProducerBuffer {
    /// This buffer's caller-assigned producer index.
    producer: u32,
    /// Recorded commands, in record order (`index == seq`).
    commands: Vec<Command>,
}

impl ProducerBuffer {
    #[inline]
    fn new(producer: u32) -> Self {
        Self {
            producer,
            commands: Vec::new(),
        }
    }
}

/// A set of per-producer command buffers that merge-and-replay deterministically.
///
/// Create one with [`ParallelCommandBuffers::with_producers`], record into each
/// producer (sequentially via [`producer`](ParallelCommandBuffers::producer) or
/// in parallel via [`split_with`](ParallelCommandBuffers::split_with)), then
/// drain everything into a [`World`] with
/// [`apply`](ParallelCommandBuffers::apply).
#[derive(Default)]
pub struct ParallelCommandBuffers {
    buffers: Vec<ProducerBuffer>,
}

impl ParallelCommandBuffers {
    /// Create `producers` independent buffers, indexed `0..producers`.
    pub fn with_producers(producers: usize) -> Self {
        assert!(
            producers <= u32::MAX as usize,
            "producer count exceeds the u32 index space"
        );
        let mut buffers = Vec::with_capacity(producers);
        for producer in 0..producers as u32 {
            buffers.push(ProducerBuffer::new(producer));
        }
        Self { buffers }
    }

    /// Number of producer buffers.
    #[inline]
    pub fn producer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Total number of recorded-but-unapplied commands across all buffers.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffers.iter().map(|b| b.commands.len()).sum()
    }

    /// Whether every buffer is currently empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffers.iter().all(|b| b.commands.is_empty())
    }

    /// Borrow a [`ParallelCommands`] builder recording into producer `index`,
    /// reserving new entity handles from `entities`.
    ///
    /// Use this for sequential recording (one producer at a time). For
    /// lock-free concurrent recording across producers, use
    /// [`split_with`](ParallelCommandBuffers::split_with).
    ///
    /// # Panics
    /// Panics if `index >= producer_count()`.
    #[inline]
    pub fn producer<'w, 's>(
        &'s mut self,
        index: usize,
        entities: &'w Entities,
    ) -> ParallelCommands<'w, 's> {
        let buffer = &mut self.buffers[index];
        ParallelCommands { buffer, entities }
    }

    /// Hand out one disjoint [`ParallelCommands`] handle per producer buffer,
    /// each reserving from `entities`.
    ///
    /// Each handle borrows a different buffer mutably, so the handles can be
    /// moved onto separate worker threads (they are [`Send`]) and recorded into
    /// concurrently with no shared locking. The returned `Vec` is in producer
    /// index order.
    pub fn split_with<'w, 's>(
        &'s mut self,
        entities: &'w Entities,
    ) -> Vec<ParallelCommands<'w, 's>> {
        self.buffers
            .iter_mut()
            .map(|buffer| ParallelCommands { buffer, entities })
            .collect()
    }

    /// The deterministic replay plan: the [`CommandKey`] of every recorded
    /// command, in the exact order [`apply`](ParallelCommandBuffers::apply)
    /// would replay them.
    ///
    /// This is a stable sort of all keys by `(producer, seq)` and is handy for
    /// asserting determinism without mutating a world.
    pub fn plan(&self) -> Vec<CommandKey> {
        let mut keys = Vec::with_capacity(self.len());
        for buffer in &self.buffers {
            for seq in 0..buffer.commands.len() as u32 {
                keys.push(CommandKey {
                    producer: buffer.producer,
                    seq,
                });
            }
        }
        keys.sort_by_key(|k| *k);
        keys
    }

    /// Merge every buffer and replay all commands into `world` in deterministic
    /// [`CommandKey`] order, emptying the buffers.
    ///
    /// Reserved entity handles are materialised first (via
    /// [`World::flush_reserved`]) so deferred spawns land on live, unplaced
    /// slots — the same reserved-entity discipline as
    /// [`CommandQueue::apply`](super::CommandQueue::apply).
    pub fn apply(&mut self, world: &mut World) {
        world.flush_reserved();

        // Gather every command tagged with its deterministic key, then stable
        // sort by that key. Stable sort keeps a producer's own record order on
        // the (unique) key ties and yields a replay order independent of which
        // worker recorded when (design §9).
        let mut keyed: Vec<(CommandKey, Command)> = Vec::with_capacity(self.len());
        for buffer in &mut self.buffers {
            let producer = buffer.producer;
            for (seq, command) in buffer.commands.drain(..).enumerate() {
                keyed.push((
                    CommandKey {
                        producer,
                        seq: seq as u32,
                    },
                    command,
                ));
            }
        }
        keyed.sort_by_key(|(key, _)| *key);
        for (_, command) in keyed {
            command(world);
        }
    }

    /// Map the currently running thread onto a producer slot in `0..producer_count()`.
    ///
    /// A `std`-only convenience for callers that want a quick "one buffer per
    /// thread" split without threading an explicit index through their own
    /// code. The mapping is stable for the lifetime of a thread but is **not**
    /// part of the determinism contract: deterministic replay relies on the
    /// explicit producer indices recorded into, not on which thread recorded.
    ///
    /// # Panics
    /// Panics if there are no producers.
    #[cfg(feature = "std")]
    pub fn index_for_current_thread(&self) -> usize {
        use core::hash::{Hash, Hasher};

        assert!(
            !self.buffers.is_empty(),
            "cannot map a thread onto zero producers"
        );
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        (hasher.finish() % self.buffers.len() as u64) as usize
    }
}

/// An ergonomic builder recording deferred structural changes into a single
/// producer buffer of a [`ParallelCommandBuffers`], handing back entity handles
/// immediately.
///
/// Mirrors [`Commands`](super::Commands); the only difference is that commands
/// land in this producer's buffer and are stamped with a deterministic
/// [`CommandKey`] at the sync point.
pub struct ParallelCommands<'w, 's> {
    buffer: &'s mut ProducerBuffer,
    entities: &'w Entities,
}

impl<'w, 's> ParallelCommands<'w, 's> {
    /// The index of the producer buffer this handle records into.
    #[inline]
    pub fn producer(&self) -> u32 {
        self.buffer.producer
    }

    /// Reserve a fresh entity and queue a spawn of `bundle` onto it.
    ///
    /// The returned [`Entity`] is valid immediately, but its components do not
    /// exist until the buffers are [applied](ParallelCommandBuffers::apply).
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.reserve_entity();
        self.buffer.commands.push(Box::new(move |world: &mut World| {
            world.spawn_at(entity, bundle);
        }));
        entity
    }

    /// Reserve a fresh entity with no components (an empty bundle spawn).
    #[inline]
    pub fn spawn_empty(&mut self) -> Entity {
        self.spawn(())
    }

    /// Borrow a per-entity command builder for queuing edits to `entity`.
    #[inline]
    pub fn entity(&mut self, entity: Entity) -> ParallelEntityCommands<'_, 'w, 's> {
        ParallelEntityCommands {
            entity,
            commands: self,
        }
    }
}

/// A per-entity command builder returned by [`ParallelCommands::entity`].
///
/// Chains deferred edits (`insert` / `remove` / `despawn`) targeting one
/// [`Entity`] into this producer's buffer.
pub struct ParallelEntityCommands<'a, 'w, 's> {
    entity: Entity,
    commands: &'a mut ParallelCommands<'w, 's>,
}

impl ParallelEntityCommands<'_, '_, '_> {
    /// The entity these commands target.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Queue inserting `bundle` onto the entity (overwriting existing
    /// components last-wins; see [`World::insert`]).
    pub fn insert<B: Bundle>(&mut self, bundle: B) -> &mut Self {
        let entity = self.entity;
        self.commands
            .buffer
            .commands
            .push(Box::new(move |world: &mut World| {
                world.insert(entity, bundle);
            }));
        self
    }

    /// Queue removing the components named by bundle type `B` from the entity
    /// (see [`World::remove`]).
    pub fn remove<B: Bundle>(&mut self) -> &mut Self {
        let entity = self.entity;
        self.commands
            .buffer
            .commands
            .push(Box::new(move |world: &mut World| {
                world.remove::<B>(entity);
            }));
        self
    }

    /// Queue despawning the entity (see [`World::despawn`]).
    pub fn despawn(&mut self) {
        let entity = self.entity;
        self.commands
            .buffer
            .commands
            .push(Box::new(move |world: &mut World| {
                world.despawn(entity);
            }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Position(i32, i32);
    impl Component for Position {}
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Velocity(i32, i32);
    impl Component for Velocity {}
    // Encodes which producer/seq recorded the spawn, so world state can be
    // compared across different recording interleavings.
    #[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
    struct Tag(u32, u32);
    impl Component for Tag {}

    #[test]
    fn parallel_deferred_spawn_insert_remove_despawn() {
        let mut world = World::new();
        let mut buffers = ParallelCommandBuffers::with_producers(2);

        let (a, b) = {
            let mut split = buffers.split_with(world.entities());
            // Producer 0 spawns `a` and edits it; producer 1 spawns `b`.
            let a = split[0].spawn((Position(1, 2), Velocity(3, 4)));
            let b = split[1].spawn(Position(5, 6));
            split[0].entity(a).insert(Velocity(100, 100));
            split[1].entity(b).insert(Velocity(7, 8));
            (a, b)
        };

        // Nothing applied yet.
        assert_eq!(world.entity_count(), 0);
        assert!(!world.contains(a));
        assert_eq!(buffers.len(), 4);

        buffers.apply(&mut world);

        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.get::<Position>(a), Some(&Position(1, 2)));
        assert_eq!(world.get::<Velocity>(a), Some(&Velocity(100, 100)));
        assert_eq!(world.get::<Position>(b), Some(&Position(5, 6)));
        assert_eq!(world.get::<Velocity>(b), Some(&Velocity(7, 8)));
        assert!(buffers.is_empty());

        // A second batch across producers: remove and despawn.
        {
            let mut split = buffers.split_with(world.entities());
            split[0].entity(a).remove::<Velocity>();
            split[1].entity(b).despawn();
        }
        buffers.apply(&mut world);

        assert_eq!(world.get::<Velocity>(a), None);
        assert_eq!(world.get::<Position>(a), Some(&Position(1, 2)));
        assert!(!world.contains(b));
        assert_eq!(world.entity_count(), 1);
    }

    // Record the fixed scenario below into `buffers` following `touch_order`
    // (a list of producer indices, visited in turn). Each visit records the
    // next spawn for that producer. Returns nothing; the buffers hold the
    // result. The per-producer command sequences are identical regardless of
    // `touch_order`.
    fn record_scenario(
        buffers: &mut ParallelCommandBuffers,
        entities: &Entities,
        touch_order: &[usize],
    ) {
        // Per-producer spawn counts: producer 0 -> 2, producer 1 -> 1,
        // producer 2 -> 3.
        let totals = [2u32, 1, 3];
        let mut next = [0u32; 3];
        for &p in touch_order {
            let seq = next[p];
            assert!(seq < totals[p], "touch_order over-records producer {p}");
            next[p] += 1;
            buffers
                .producer(p, entities)
                .spawn(Tag(p as u32, seq));
        }
        for p in 0..3 {
            assert_eq!(next[p], totals[p], "touch_order under-records producer {p}");
        }
    }

    fn collect_tags(world: &mut World) -> Vec<Tag> {
        let state = world.query::<&Tag>();
        let mut tags: Vec<Tag> = state.iter(world).copied().collect();
        tags.sort();
        tags
    }

    #[test]
    fn parallel_replay_is_deterministic_across_interleavings() {
        // Two different recording interleavings of the same scenario.
        let grouped = [0usize, 0, 1, 2, 2, 2]; // producer-by-producer
        let round_robin = [0usize, 1, 2, 0, 2, 2]; // interleaved

        // Interleaving A.
        let mut world_a = World::new();
        let mut buffers_a = ParallelCommandBuffers::with_producers(3);
        record_scenario(&mut buffers_a, world_a.entities(), &grouped);
        let plan_a = buffers_a.plan();
        buffers_a.apply(&mut world_a);
        let tags_a = collect_tags(&mut world_a);

        // Interleaving B.
        let mut world_b = World::new();
        let mut buffers_b = ParallelCommandBuffers::with_producers(3);
        record_scenario(&mut buffers_b, world_b.entities(), &round_robin);
        let plan_b = buffers_b.plan();
        buffers_b.apply(&mut world_b);
        let tags_b = collect_tags(&mut world_b);

        // The deterministic plan is the full (producer, seq) total order and is
        // identical regardless of recording interleaving.
        let expected_plan = [
            CommandKey { producer: 0, seq: 0 },
            CommandKey { producer: 0, seq: 1 },
            CommandKey { producer: 1, seq: 0 },
            CommandKey { producer: 2, seq: 0 },
            CommandKey { producer: 2, seq: 1 },
            CommandKey { producer: 2, seq: 2 },
        ];
        assert_eq!(plan_a, expected_plan);
        assert_eq!(plan_a, plan_b);

        // The resulting world state is identical too.
        let expected_tags = [
            Tag(0, 0),
            Tag(0, 1),
            Tag(1, 0),
            Tag(2, 0),
            Tag(2, 1),
            Tag(2, 2),
        ];
        assert_eq!(tags_a, expected_tags);
        assert_eq!(tags_a, tags_b);
        assert_eq!(world_a.entity_count(), 6);
        assert_eq!(world_b.entity_count(), 6);
    }

    #[test]
    fn empty_buffers_apply_is_noop() {
        let mut world = World::new();
        let mut buffers = ParallelCommandBuffers::with_producers(4);
        assert!(buffers.is_empty());
        assert_eq!(buffers.producer_count(), 4);
        assert_eq!(buffers.plan(), []);
        buffers.apply(&mut world);
        assert_eq!(world.entity_count(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn concurrent_recording_lock_free_hot_path() {
        // Prove the hot path records with no shared lock: hand each thread its
        // own `ParallelCommands` handle via `split_with` and record in true
        // parallel, then merge deterministically at the sync point.
        let mut world = World::new();
        let mut buffers = ParallelCommandBuffers::with_producers(4);

        let per_producer = 25u32;
        {
            let split = buffers.split_with(world.entities());
            std::thread::scope(|scope| {
                for mut handle in split {
                    scope.spawn(move || {
                        let p = handle.producer();
                        for seq in 0..per_producer {
                            handle.spawn(Tag(p, seq));
                        }
                    });
                }
            });
        }

        assert_eq!(buffers.len() as u32, per_producer * 4);
        buffers.apply(&mut world);
        assert_eq!(world.entity_count(), per_producer * 4);

        // Every (producer, seq) tag materialised exactly once.
        let tags = collect_tags(&mut world);
        let mut expected: Vec<Tag> = (0..4u32)
            .flat_map(|p| (0..per_producer).map(move |s| Tag(p, s)))
            .collect();
        expected.sort();
        assert_eq!(tags, expected);
    }
}
