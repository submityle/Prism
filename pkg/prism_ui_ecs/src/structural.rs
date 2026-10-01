//! Signal-driven **structural** bindings: `Show` (conditional mount) and `For`
//! (keyed list), reconciled against a Bevy [`World`] and batched to frame end.
//!
//! Where [`FieldBinding`](crate::FieldBinding) projects a *scalar field* of an
//! existing component, structural bindings decide *which entities exist at all*
//! based on reactive state:
//!
//! * [`ShowBinding`] spawns a child entity while its `Signal<bool>` is `true`
//!   and despawns it when the signal turns `false`.
//! * [`ForBinding`] mirrors a `Signal<Vec<T>>` into one entity per item, keyed
//!   by a user key function, reusing existing entities across reorders via the
//!   minimal-move [`diff_keyed`](prism_ui_tree::diff_keyed) reconciler.
//!
//! # Two-phase batching
//!
//! To avoid interleaving `spawn`/`despawn` with other work mid-frame (which
//! would churn ECS archetypes), reconciliation is split in two:
//!
//! * **reconcile** reads the signal and computes a *plan* (for `For`, a keyed
//!   [`Diff`]); it never mutates the world.
//! * **flush** applies the plan, performing all spawns and despawns at once and
//!   updating the binding's bookkeeping.
//!
//! [`StructuralScope`] groups many bindings so an entire UI subtree can be
//! reconciled and then flushed together, returning aggregate
//! [`StructuralStats`].
//!
//! # Equality guards
//!
//! A `Show` whose boolean is unchanged relative to its mounted state does
//! nothing; a `For` whose key order is identical performs no spawn/despawn
//! (optionally re-running its `update` closure for in-place refresh). Cost is
//! therefore proportional to the structural change.
//!
//! # Example
//!
//! ```
//! use bevy_ecs::prelude::World;
//! use prism_ui_ecs::{ShowBinding, StructuralBinding};
//! use prism_ui_reactive::Runtime;
//!
//! let mut world = World::new();
//! let rt = Runtime::new();
//! let visible = rt.signal(false);
//!
//! let mut show = ShowBinding::new(visible.clone(), |w| w.spawn(()).id());
//!
//! // Hidden: reconcile + flush spawn nothing.
//! show.reconcile(&world);
//! show.flush(&mut world);
//! assert!(show.current_entity().is_none());
//!
//! // Turn it on: one entity is spawned on flush.
//! visible.set(true);
//! show.reconcile(&world);
//! let stats = show.flush(&mut world);
//! assert_eq!(stats.spawned, 1);
//! assert!(show.current_entity().is_some());
//!
//! // Turn it off: the entity is despawned.
//! visible.set(false);
//! show.reconcile(&world);
//! let stats = show.flush(&mut world);
//! assert_eq!(stats.despawned, 1);
//! assert!(show.current_entity().is_none());
//! ```

use bevy_ecs::prelude::{Entity, World};
use prism_ui_reactive::Signal;
use prism_ui_tree::{diff_keyed, Diff, DiffOp};

/// Count of structural mutations produced by a reconcile/flush cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StructuralStats {
    /// Number of entities spawned during the flush.
    pub spawned: usize,
    /// Number of entities despawned during the flush.
    pub despawned: usize,
}

impl StructuralStats {
    /// Returns the component-wise sum of two stat records.
    pub fn merged(self, other: Self) -> Self {
        Self {
            spawned: self.spawned + other.spawned,
            despawned: self.despawned + other.despawned,
        }
    }

    /// Total number of structural mutations (`spawned + despawned`).
    pub fn total(self) -> usize {
        self.spawned + self.despawned
    }
}

/// A structural binding that can be reconciled, then flushed, against a world.
///
/// `reconcile` is pure with respect to the world (it only reads the signal and
/// records a plan); `flush` performs the deferred spawns and despawns. This
/// split lets a [`StructuralScope`] compute every plan before touching the
/// world so all mutations land together at frame end.
pub trait StructuralBinding {
    /// Reads the driving signal and records what should change on flush.
    fn reconcile(&mut self, world: &World);

    /// Applies the recorded plan, returning how many entities were spawned and
    /// despawned.
    fn flush(&mut self, world: &mut World) -> StructuralStats;
}

/// The deferred action a [`ShowBinding`] will take on its next flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShowPlan {
    /// Nothing to do.
    Idle,
    /// Spawn the child entity (signal became `true` while unmounted).
    Spawn,
    /// Despawn the given entity (signal became `false` while mounted).
    Despawn(Entity),
}

/// Conditionally mounts a single child entity based on a `Signal<bool>`.
///
/// While the signal is `true` exactly one entity (created by the `spawn`
/// closure) is kept alive; when it turns `false` that entity is despawned. A
/// signal value matching the current mounted state is a no-op.
pub struct ShowBinding {
    signal: Signal<bool>,
    spawn: Box<dyn Fn(&mut World) -> Entity>,
    current: Option<Entity>,
    plan: ShowPlan,
}

impl ShowBinding {
    /// Creates a `Show` binding driven by `signal`, spawning its child with
    /// `spawn` when the signal is `true`.
    pub fn new(signal: Signal<bool>, spawn: impl Fn(&mut World) -> Entity + 'static) -> Self {
        Self {
            signal,
            spawn: Box::new(spawn),
            current: None,
            plan: ShowPlan::Idle,
        }
    }

    /// Returns the currently mounted entity, if any.
    pub fn current_entity(&self) -> Option<Entity> {
        self.current
    }

    /// Returns `true` while the child entity is mounted.
    pub fn is_mounted(&self) -> bool {
        self.current.is_some()
    }
}

impl StructuralBinding for ShowBinding {
    fn reconcile(&mut self, _world: &World) {
        let want = self.signal.get_untracked();
        self.plan = match (want, self.current) {
            (true, None) => ShowPlan::Spawn,
            (false, Some(entity)) => ShowPlan::Despawn(entity),
            // Already in the desired state: nothing to do.
            (true, Some(_)) | (false, None) => ShowPlan::Idle,
        };
    }

    fn flush(&mut self, world: &mut World) -> StructuralStats {
        let mut stats = StructuralStats::default();
        match core::mem::replace(&mut self.plan, ShowPlan::Idle) {
            ShowPlan::Idle => {}
            ShowPlan::Spawn => {
                let entity = (self.spawn)(world);
                self.current = Some(entity);
                stats.spawned += 1;
            }
            ShowPlan::Despawn(entity) => {
                if world.despawn(entity) {
                    stats.despawned += 1;
                }
                self.current = None;
            }
        }
        stats
    }
}

/// The deferred plan a [`ForBinding`] will apply on its next flush.
struct ForPlan<T, K> {
    diff: Diff,
    new_items: Vec<T>,
    new_keys: Vec<K>,
}

/// Mirrors a `Signal<Vec<T>>` into one entity per item, keyed for reuse.
///
/// On each reconcile the current items' keys are diffed against the previous
/// order via [`diff_keyed`]: unchanged keys reuse their existing entity (so an
/// item's entity id is stable across reorders), keys new to the list spawn
/// fresh entities on flush, and dropped keys despawn theirs. An optional
/// `update` closure runs for every reused entity so per-item state can be
/// refreshed in place.
pub struct ForBinding<T: 'static, K> {
    signal: Signal<Vec<T>>,
    key_of: Box<dyn Fn(&T) -> K>,
    spawn: Box<dyn Fn(&mut World, &T) -> Entity>,
    update: Option<Box<dyn Fn(&mut World, Entity, &T)>>,
    current: Vec<(K, Entity)>,
    plan: Option<ForPlan<T, K>>,
}

impl<T, K> ForBinding<T, K>
where
    T: Clone + 'static,
    K: Ord + Clone + 'static,
{
    /// Creates a `For` binding driven by `signal`, keying items with `key_of`
    /// and creating entities with `spawn`.
    pub fn new(
        signal: Signal<Vec<T>>,
        key_of: impl Fn(&T) -> K + 'static,
        spawn: impl Fn(&mut World, &T) -> Entity + 'static,
    ) -> Self {
        Self {
            signal,
            key_of: Box::new(key_of),
            spawn: Box::new(spawn),
            update: None,
            current: Vec::new(),
            plan: None,
        }
    }

    /// Like [`new`](Self::new) but also runs `update` on every reused entity
    /// during flush, letting items refresh in place without respawning.
    pub fn with_update(
        signal: Signal<Vec<T>>,
        key_of: impl Fn(&T) -> K + 'static,
        spawn: impl Fn(&mut World, &T) -> Entity + 'static,
        update: impl Fn(&mut World, Entity, &T) + 'static,
    ) -> Self {
        let mut this = Self::new(signal, key_of, spawn);
        this.update = Some(Box::new(update));
        this
    }

    /// Returns the mounted entities in their current (new) order.
    pub fn entities(&self) -> Vec<Entity> {
        self.current.iter().map(|(_, entity)| *entity).collect()
    }

    /// Returns the number of mounted entities.
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Returns `true` when no entities are mounted.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Returns the entity currently mounted for `key`, if present.
    pub fn entity_for(&self, key: &K) -> Option<Entity> {
        self.current
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, entity)| *entity)
    }
}

impl<T, K> StructuralBinding for ForBinding<T, K>
where
    T: Clone + 'static,
    K: Ord + Clone + 'static,
{
    fn reconcile(&mut self, _world: &World) {
        let new_items = self.signal.get_untracked();
        let new_keys: Vec<K> = new_items.iter().map(|item| (self.key_of)(item)).collect();
        let old_keys: Vec<K> = self.current.iter().map(|(key, _)| key.clone()).collect();
        let diff = diff_keyed(&old_keys, &new_keys);
        self.plan = Some(ForPlan {
            diff,
            new_items,
            new_keys,
        });
    }

    fn flush(&mut self, world: &mut World) -> StructuralStats {
        let Some(plan) = self.plan.take() else {
            return StructuralStats::default();
        };
        let mut stats = StructuralStats::default();
        let mut next: Vec<(K, Entity)> = Vec::with_capacity(plan.new_keys.len());

        // `ops` are in new order, one per new slot, so the enumeration index is
        // the slot (and the new-item) index.
        for (slot, op) in plan.diff.ops.iter().enumerate() {
            match op {
                DiffOp::Keep { old_index } | DiffOp::Move { old_index } => {
                    let entity = self.current[*old_index].1;
                    if let Some(update) = &self.update {
                        update(world, entity, &plan.new_items[slot]);
                    }
                    next.push((plan.new_keys[slot].clone(), entity));
                }
                DiffOp::Create { new_index } => {
                    let entity = (self.spawn)(world, &plan.new_items[*new_index]);
                    stats.spawned += 1;
                    next.push((plan.new_keys[*new_index].clone(), entity));
                }
            }
        }

        for &old_index in &plan.diff.removals {
            if world.despawn(self.current[old_index].1) {
                stats.despawned += 1;
            }
        }

        self.current = next;
        stats
    }
}

/// A collection of structural bindings reconciled and flushed as a unit.
///
/// Call [`reconcile_all`](Self::reconcile_all) to compute every binding's plan
/// without touching the world, then [`flush`](Self::flush) to apply all spawns
/// and despawns together; [`run`](Self::run) does both in sequence.
#[derive(Default)]
pub struct StructuralScope {
    bindings: Vec<Box<dyn StructuralBinding>>,
}

impl StructuralScope {
    /// Creates an empty scope.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Registers a binding, returning `&mut self` for chaining.
    pub fn add(&mut self, binding: impl StructuralBinding + 'static) -> &mut Self {
        self.bindings.push(Box::new(binding));
        self
    }

    /// Number of registered bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns `true` when no bindings are registered.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Reconciles every binding (computes plans; does not mutate the world).
    pub fn reconcile_all(&mut self, world: &World) {
        for binding in &mut self.bindings {
            binding.reconcile(world);
        }
    }

    /// Flushes every binding, returning the aggregate stats.
    pub fn flush(&mut self, world: &mut World) -> StructuralStats {
        let mut stats = StructuralStats::default();
        for binding in &mut self.bindings {
            stats = stats.merged(binding.flush(world));
        }
        stats
    }

    /// Reconciles all bindings and then flushes them in one call.
    pub fn run(&mut self, world: &mut World) -> StructuralStats {
        self.reconcile_all(world);
        self.flush(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::Component;
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Item {
        value: i32,
    }

    // `World::new()` seeds an internal entity for the default query
    // filters, so user-spawned entities are counted relative to that
    // baseline.
    fn count_entities(world: &mut World) -> usize {
        let baseline = World::new().iter_entities().count();
        world.iter_entities().count() - baseline
    }

    #[test]
    fn show_spawns_when_true() {
        let mut world = World::new();
        let rt = Runtime::new();
        let visible = rt.signal(true);
        let mut show = ShowBinding::new(visible, |w| w.spawn(()).id());
        show.reconcile(&world);
        let stats = show.flush(&mut world);
        assert_eq!(stats.spawned, 1);
        assert!(show.is_mounted());
        assert_eq!(count_entities(&mut world), 1);
    }

    #[test]
    fn show_starts_hidden_is_noop() {
        let mut world = World::new();
        let rt = Runtime::new();
        let visible = rt.signal(false);
        let mut show = ShowBinding::new(visible, |w| w.spawn(()).id());
        show.reconcile(&world);
        let stats = show.flush(&mut world);
        assert_eq!(stats, StructuralStats::default());
        assert!(!show.is_mounted());
    }

    #[test]
    fn show_true_false_true_cycle() {
        let mut world = World::new();
        let rt = Runtime::new();
        let visible = rt.signal(false);
        let mut show = ShowBinding::new(visible.clone(), |w| w.spawn(()).id());

        visible.set(true);
        show.reconcile(&world);
        assert_eq!(show.flush(&mut world).spawned, 1);
        let first = show.current_entity().unwrap();

        visible.set(false);
        show.reconcile(&world);
        assert_eq!(show.flush(&mut world).despawned, 1);
        assert!(!show.is_mounted());

        visible.set(true);
        show.reconcile(&world);
        assert_eq!(show.flush(&mut world).spawned, 1);
        let second = show.current_entity().unwrap();
        // A fresh entity is created on the second mount.
        assert_ne!(first, second);
    }

    #[test]
    fn show_unchanged_true_is_noop() {
        let mut world = World::new();
        let rt = Runtime::new();
        let visible = rt.signal(true);
        let mut show = ShowBinding::new(visible, |w| w.spawn(()).id());
        show.reconcile(&world);
        show.flush(&mut world);
        // Reconcile again without changing the signal.
        show.reconcile(&world);
        let stats = show.flush(&mut world);
        assert_eq!(stats, StructuralStats::default());
        assert_eq!(count_entities(&mut world), 1);
    }

    #[test]
    fn for_creates_entities_for_initial_items() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3]);
        let mut binding = ForBinding::new(
            items,
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats.spawned, 3);
        assert_eq!(stats.despawned, 0);
        assert_eq!(binding.len(), 3);
    }

    #[test]
    fn for_appends_new_items_reusing_existing() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2]);
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let before = binding.entities();

        items.set(vec![1, 2, 3]);
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats.spawned, 1);
        assert_eq!(stats.despawned, 0);
        // First two entities are reused unchanged.
        assert_eq!(&binding.entities()[0..2], &before[..]);
    }

    #[test]
    fn for_removes_dropped_items() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3]);
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let keep_entity = binding.entity_for(&2).unwrap();

        items.set(vec![2]);
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats.spawned, 0);
        assert_eq!(stats.despawned, 2);
        assert_eq!(binding.len(), 1);
        // The surviving item keeps its entity id.
        assert_eq!(binding.entity_for(&2), Some(keep_entity));
    }

    #[test]
    fn for_reorder_reuses_all_entities() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3]);
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let e1 = binding.entity_for(&1).unwrap();
        let e2 = binding.entity_for(&2).unwrap();
        let e3 = binding.entity_for(&3).unwrap();

        // Reverse order: pure reorder, nothing created or destroyed.
        items.set(vec![3, 2, 1]);
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats, StructuralStats::default());
        // Entities are the same ids, now in new order.
        assert_eq!(binding.entities(), vec![e3, e2, e1]);
    }

    #[test]
    fn for_shuffle_minimal_create_remove() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3, 4]);
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let e2 = binding.entity_for(&2).unwrap();
        let e4 = binding.entity_for(&4).unwrap();

        // Drop 1 and 3, add 5; reuse 2 and 4.
        items.set(vec![4, 2, 5]);
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats.spawned, 1); // only key 5
        assert_eq!(stats.despawned, 2); // keys 1 and 3
        assert_eq!(binding.entity_for(&2), Some(e2));
        assert_eq!(binding.entity_for(&4), Some(e4));
        assert_eq!(binding.len(), 3);
    }

    #[test]
    fn for_identical_keys_is_noop() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3]);
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let before = binding.entities();

        // Same list again.
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats, StructuralStats::default());
        assert_eq!(binding.entities(), before);
    }

    #[test]
    fn for_update_runs_on_reused_entities() {
        let mut world = World::new();
        let rt = Runtime::new();
        // Key by id (first field); value (second field) may change in place.
        let items = rt.signal(vec![(1i32, 10i32), (2, 20)]);
        let mut binding = ForBinding::with_update(
            items.clone(),
            |pair: &(i32, i32)| pair.0,
            |w, pair: &(i32, i32)| w.spawn(Item { value: pair.1 }).id(),
            |w, entity, pair: &(i32, i32)| {
                if let Some(mut item) = w.get_mut::<Item>(entity) {
                    item.value = pair.1;
                }
            },
        );
        binding.reconcile(&world);
        binding.flush(&mut world);
        let e1 = binding.entity_for(&1).unwrap();

        // Same keys, changed values: no spawn/despawn, update rewrites fields.
        items.set(vec![(1, 11), (2, 22)]);
        binding.reconcile(&world);
        let stats = binding.flush(&mut world);
        assert_eq!(stats, StructuralStats::default());
        assert_eq!(world.get::<Item>(e1).unwrap().value, 11);
    }

    #[test]
    fn for_empty_to_populated_and_back() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items: Signal<Vec<i32>> = rt.signal(Vec::new());
        let mut binding = ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        );
        binding.reconcile(&world);
        assert_eq!(binding.flush(&mut world), StructuralStats::default());
        assert!(binding.is_empty());

        items.set(vec![7, 8]);
        binding.reconcile(&world);
        assert_eq!(binding.flush(&mut world).spawned, 2);

        items.set(Vec::new());
        binding.reconcile(&world);
        assert_eq!(binding.flush(&mut world).despawned, 2);
        assert!(binding.is_empty());
    }

    #[test]
    fn scope_aggregates_show_and_for() {
        let mut world = World::new();
        let rt = Runtime::new();
        let visible = rt.signal(true);
        let items = rt.signal(vec![1i32, 2]);

        let mut scope = StructuralScope::new();
        scope.add(ShowBinding::new(visible.clone(), |w| w.spawn(()).id()));
        scope.add(ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        ));
        assert_eq!(scope.len(), 2);

        let stats = scope.run(&mut world);
        assert_eq!(stats.spawned, 3); // 1 from show + 2 from for
        assert_eq!(stats.despawned, 0);
    }

    #[test]
    fn scope_batches_mutations_until_flush() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2, 3]);
        let mut scope = StructuralScope::new();
        scope.add(ForBinding::new(
            items,
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        ));

        // Reconcile alone must not spawn anything.
        scope.reconcile_all(&world);
        assert_eq!(count_entities(&mut world), 0);

        let stats = scope.flush(&mut world);
        assert_eq!(stats.spawned, 3);
        assert_eq!(count_entities(&mut world), 3);
    }

    #[test]
    fn scope_run_over_multiple_frames() {
        let mut world = World::new();
        let rt = Runtime::new();
        let items = rt.signal(vec![1i32, 2]);
        let mut scope = StructuralScope::new();
        scope.add(ForBinding::new(
            items.clone(),
            |value: &i32| *value,
            |w, value: &i32| w.spawn(Item { value: *value }).id(),
        ));

        assert_eq!(scope.run(&mut world).spawned, 2);
        items.set(vec![1, 2, 3, 4]);
        let stats = scope.run(&mut world);
        assert_eq!(stats.spawned, 2);
        assert_eq!(stats.despawned, 0);
        assert_eq!(count_entities(&mut world), 4);
    }

    #[test]
    fn stats_merged_and_total() {
        let a = StructuralStats {
            spawned: 2,
            despawned: 1,
        };
        let b = StructuralStats {
            spawned: 3,
            despawned: 4,
        };
        let merged = a.merged(b);
        assert_eq!(
            merged,
            StructuralStats {
                spawned: 5,
                despawned: 5
            }
        );
        assert_eq!(merged.total(), 10);
    }
}
