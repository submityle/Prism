//! The [`World`]: the owning container for all entities, components, and their
//! archetype-grouped storage, plus the public API for spawning, mutating, and
//! despawning entities.
//!
//! A [`World`] bundles three registries:
//! - [`Entities`] — the generational-index allocator and location table.
//! - [`Components`] — the component-type metadata registry.
//! - [`Archetypes`] — the archetype graph and its columnar [`Table`]s.
//!
//! Structural changes (spawn / insert / remove / despawn) move component data
//! between archetype tables while keeping every entity's recorded location in
//! sync. All of these operations are **immediate**; the deferred
//! [`Commands`](crate::command::Commands) buffer records the same operations to
//! be applied later at a synchronization point.

use alloc::vec::Vec;

use crate::archetype::Archetypes;
use crate::bundle::Bundle;
use crate::change::Tick;
use crate::component::{Component, ComponentId, ComponentSet, Components};
use crate::entity::{Entities, Entity, EntityLocation};
use crate::query::{QueryData, QueryFilter, QueryState, ReadOnlyQueryData};
use crate::resource::{Resource, Resources};

/// The authoritative container of all ECS state.
pub struct World {
    entities: Entities,
    components: Components,
    archetypes: Archetypes,
    resources: Resources,
    /// Monotonically increasing change counter (design §10). Stamped onto
    /// component writes and compared against each system's `last_run` to drive
    /// `Added`/`Changed` detection.
    change_tick: Tick,
    /// The tick a one-shot read via [`World::query`] / [`World::get`] treats as
    /// its `last_run` baseline. Advanced by the schedule executor per run.
    last_change_tick: Tick,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Create an empty world (with just the empty archetype allocated).
    ///
    /// The change counter starts at `1` so that `0` can mean "never run": a
    /// freshly spawned value stamped at tick `1` reads as added/changed to a
    /// system whose `last_run` defaults to [`Tick::ZERO`].
    pub fn new() -> Self {
        Self {
            entities: Entities::new(),
            components: Components::new(),
            archetypes: Archetypes::new(),
            resources: Resources::new(),
            change_tick: Tick::new(1),
            last_change_tick: Tick::ZERO,
        }
    }

    /// The world's current change tick (the tick new writes are stamped with).
    #[inline]
    pub fn change_tick(&self) -> Tick {
        self.change_tick
    }

    /// Advance the change tick by one and return the new value.
    ///
    /// The schedule executor calls this around each system run so writes made
    /// by different systems carry distinct ticks.
    #[inline]
    pub fn increment_change_tick(&mut self) -> Tick {
        let next = Tick::new(self.change_tick.get().wrapping_add(1));
        self.change_tick = next;
        next
    }

    /// The baseline `last_run` tick used by one-shot reads.
    #[inline]
    pub fn last_change_tick(&self) -> Tick {
        self.last_change_tick
    }

    /// Overwrite the baseline `last_run` tick (used by the executor).
    #[inline]
    pub fn set_last_change_tick(&mut self, tick: Tick) {
        self.last_change_tick = tick;
    }

    /// Clamp every stored component tick (and the world baseline) against the
    /// current change tick so no tick can wrap past [`Tick::MAX_CHANGE_AGE`]
    /// and alias a recent one. Cheap to run periodically (design §10).
    pub fn check_change_ticks(&mut self) {
        let this_run = self.change_tick;
        for arch in self.archetypes.iter_mut() {
            arch.table_mut().check_change_ticks(this_run);
        }
        self.last_change_tick.check_tick(this_run);
    }

    /// The entity allocator / location table.
    #[inline]
    pub fn entities(&self) -> &Entities {
        &self.entities
    }

    /// The component-type registry.
    #[inline]
    pub fn components(&self) -> &Components {
        &self.components
    }

    /// The archetype graph.
    #[inline]
    pub fn archetypes(&self) -> &Archetypes {
        &self.archetypes
    }

    /// The global resource registry.
    #[inline]
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Mutable access to the component registry (crate-internal: used by the
    /// system-param layer to resolve [`SystemParam`](crate::system::SystemParam)
    /// state against the world).
    #[inline]
    pub(crate) fn components_mut(&mut self) -> &mut Components {
        &mut self.components
    }

    /// Mutable access to the resource store (crate-internal: used by the
    /// system-param layer).
    #[inline]
    pub(crate) fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    /// Insert `value` as the world-global resource `R`, returning the previous
    /// value if one was present.
    #[inline]
    pub fn insert_resource<R: Resource>(&mut self, value: R) -> Option<R> {
        self.resources.insert(value)
    }

    /// Ensure a value for `R` exists, inserting `R::default()` if absent.
    #[inline]
    pub fn init_resource<R: Resource + Default>(&mut self) {
        self.resources.init::<R>();
    }

    /// Borrow resource `R`, or `None` if it has not been inserted.
    #[inline]
    pub fn get_resource<R: Resource>(&self) -> Option<&R> {
        self.resources.get::<R>()
    }

    /// Mutably borrow resource `R`, or `None` if it has not been inserted.
    #[inline]
    pub fn get_resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.resources.get_mut::<R>()
    }

    /// Borrow resource `R`, panicking if it is absent.
    ///
    /// # Panics
    /// Panics if resource `R` has not been inserted.
    #[inline]
    pub fn resource<R: Resource>(&self) -> &R {
        self.resources
            .get::<R>()
            .expect("requested resource does not exist in the world")
    }

    /// Mutably borrow resource `R`, panicking if it is absent.
    ///
    /// # Panics
    /// Panics if resource `R` has not been inserted.
    #[inline]
    pub fn resource_mut<R: Resource>(&mut self) -> &mut R {
        self.resources
            .get_mut::<R>()
            .expect("requested resource does not exist in the world")
    }

    /// Whether a value for resource `R` is currently present.
    #[inline]
    pub fn contains_resource<R: Resource>(&self) -> bool {
        self.resources.contains::<R>()
    }

    /// Remove and return resource `R`, if present.
    #[inline]
    pub fn remove_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// Number of currently-live entities.
    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.entities.len()
    }

    /// Register component type `T`, returning its id (idempotent).
    #[inline]
    pub fn register_component<T: Component>(&mut self) -> ComponentId {
        self.components.register::<T>()
    }

    /// Whether `entity` is live.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entities.contains(entity)
    }

    /// Spawn a new entity carrying the components of `bundle`.
    ///
    /// # Panics
    /// Panics if `bundle` contains the same component type more than once.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.alloc();
        self.place_new_entity(entity, bundle);
        entity
    }

    /// Place `bundle`'s components onto an entity that is live but not yet
    /// resident in any archetype table (an [`EntityLocation::EMPTY`] slot).
    ///
    /// This is the write half of a deferred spawn: the entity handle was handed
    /// out earlier via [`Entities::reserve_entity`] and materialised by
    /// [`World::flush_reserved`], and this call moves it into the archetype that
    /// matches `bundle`.
    ///
    /// # Panics
    /// Panics if `bundle` contains the same component type more than once, or
    /// (in debug builds) if `entity` is not a live, unplaced slot.
    pub fn spawn_at<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        debug_assert!(
            self.entities
                .location(entity)
                .is_some_and(|loc| loc.is_empty()),
            "spawn_at requires a live entity with no existing archetype placement"
        );
        self.place_new_entity(entity, bundle);
    }

    /// Shared spawn core: compute `bundle`'s archetype, allocate a row, move its
    /// values into the columns, and record `entity`'s location. `entity` must be
    /// live and currently unplaced.
    fn place_new_entity<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let change_tick = self.change_tick;
        let mut ids = Vec::new();
        B::component_ids(&mut self.components, &mut ids);
        let set = ComponentSet::from_ids(ids.iter().copied());
        assert_eq!(
            set.len(),
            ids.len(),
            "a bundle may not contain the same component type twice"
        );
        let archetype_id = self.archetypes.get_or_create(&set, &self.components);

        let row = {
            let arch = self
                .archetypes
                .get_mut(archetype_id)
                .expect("archetype just created");
            let table = arch.table_mut();
            let row = table.allocate(entity);
            let mut i = 0usize;
            // SAFETY: `get_components` yields one pointer per id in `ids` order,
            // and each pointer is a valid, owned component value moved into its
            // matching column exactly once, restoring the table invariant.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    i += 1;
                    table.column_for_fill(id).push(ptr, change_tick);
                });
            }
            debug_assert_eq!(i, ids.len());
            row
        };

        self.entities.set_location(
            entity,
            EntityLocation {
                archetype_id,
                row: row as u32,
            },
        );
    }

    /// Insert the components of `bundle` onto an existing `entity`.
    ///
    /// Components the entity already has are overwritten (last-wins); genuinely
    /// new components trigger a move to the appropriate archetype. Returns
    /// `false` if `entity` is not live.
    ///
    /// # Panics
    /// Panics if `bundle` contains the same component type more than once.
    pub fn insert<B: Bundle>(&mut self, entity: Entity, bundle: B) -> bool {
        let change_tick = self.change_tick;
        let mut ids = Vec::new();
        B::component_ids(&mut self.components, &mut ids);
        {
            let set = ComponentSet::from_ids(ids.iter().copied());
            assert_eq!(
                set.len(),
                ids.len(),
                "a bundle may not contain the same component type twice"
            );
        }
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        let src_id = loc.archetype_id;
        let current = self
            .archetypes
            .get(src_id)
            .expect("live entity archetype")
            .components()
            .clone();

        let add_ids: Vec<ComponentId> = ids
            .iter()
            .copied()
            .filter(|id| !current.contains(*id))
            .collect();

        if add_ids.is_empty() {
            // Pure overwrite — no structural move needed.
            let table = self
                .archetypes
                .get_mut(src_id)
                .expect("live entity archetype")
                .table_mut();
            let row = loc.row as usize;
            let mut i = 0usize;
            // SAFETY: every id is already a column of this table (add_ids empty),
            // `row` is in-bounds, and each pointer is a valid owned value that
            // `replace` moves in while dropping the previous value exactly once.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    i += 1;
                    table
                        .column_mut(id)
                        .expect("overwrite column exists")
                        .replace(row, ptr, change_tick);
                });
            }
            return true;
        }

        // Structural move into the archetype that is `current ∪ add_ids`.
        let mut new_set = current.clone();
        for &id in &add_ids {
            new_set = new_set.with(id);
        }
        let dst_id = self.archetypes.get_or_create(&new_set, &self.components);

        let dst_row = self
            .archetypes
            .get_mut(dst_id)
            .expect("dst archetype")
            .table_mut()
            .allocate(entity);

        {
            let (src_arch, dst_arch) = self.archetypes.get_pair_mut(src_id, dst_id);
            // SAFETY: both tables derive from the same registry (identical shared
            // layouts) and `loc.row` is in-bounds in the source table.
            unsafe {
                dst_arch
                    .table_mut()
                    .move_shared_columns_from(src_arch.table_mut(), loc.row as usize);
            }
        }

        {
            let table = self
                .archetypes
                .get_mut(dst_id)
                .expect("dst archetype")
                .table_mut();
            let mut i = 0usize;
            // SAFETY: ids present in `current` were just relocated to `dst_row`
            // and are overwritten in place; genuinely new ids fill their (so far
            // empty) column at `dst_row`. Each value is owned and consumed once.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    i += 1;
                    if current.contains(id) {
                        table
                            .column_mut(id)
                            .expect("moved column exists")
                            .replace(dst_row, ptr, change_tick);
                    } else {
                        table.column_for_fill(id).push(ptr, change_tick);
                    }
                });
            }
        }

        self.finish_move(entity, loc, src_id, current.ids(), dst_id, dst_row);
        true
    }

    /// Remove the components named by bundle type `B` from `entity`.
    ///
    /// Returns `true` if the entity was live and at least one of the named
    /// components was present (and thus removed).
    pub fn remove<B: Bundle>(&mut self, entity: Entity) -> bool {
        let mut ids = Vec::new();
        B::component_ids(&mut self.components, &mut ids);
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        let src_id = loc.archetype_id;
        let current = self
            .archetypes
            .get(src_id)
            .expect("live entity archetype")
            .components()
            .clone();

        let to_remove: Vec<ComponentId> = ids
            .iter()
            .copied()
            .filter(|id| current.contains(*id))
            .collect();
        if to_remove.is_empty() {
            return false;
        }

        let mut new_set = current.clone();
        for &id in &to_remove {
            new_set = new_set.without(id);
        }
        let dst_id = self.archetypes.get_or_create(&new_set, &self.components);

        let dst_row = self
            .archetypes
            .get_mut(dst_id)
            .expect("dst archetype")
            .table_mut()
            .allocate(entity);

        {
            let (src_arch, dst_arch) = self.archetypes.get_pair_mut(src_id, dst_id);
            // SAFETY: shared layouts (same registry) and in-bounds source row;
            // only columns present in `new_set` (⊂ current) are relocated.
            unsafe {
                dst_arch
                    .table_mut()
                    .move_shared_columns_from(src_arch.table_mut(), loc.row as usize);
            }
        }

        // Columns kept (new_set) were moved out of src; removed columns remain
        // and are dropped by `swap_remove_row`.
        self.finish_move(entity, loc, src_id, new_set.ids(), dst_id, dst_row);
        true
    }

    /// Despawn `entity`, dropping all of its components. Returns `false` if the
    /// entity was already dead.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        let Some(loc) = self.entities.free(entity) else {
            return false;
        };
        if loc.is_empty() {
            return true;
        }
        let moved = {
            let table = self
                .archetypes
                .get_mut(loc.archetype_id)
                .expect("live entity archetype")
                .table_mut();
            // SAFETY: `loc.row` is in-bounds; nothing was pre-moved so every
            // column value at that row is dropped exactly once.
            unsafe { table.swap_remove_row(loc.row as usize, &[]) }
        };
        if let Some(moved_entity) = moved {
            self.entities.set_location(moved_entity, loc);
        }
        true
    }

    /// Shared helper: after a destination row has been fully populated, remove
    /// the source row (forgetting `moved_ids` which were relocated) and patch
    /// both the relocated and the swapped entity's locations.
    fn finish_move(
        &mut self,
        entity: Entity,
        src_loc: EntityLocation,
        src_id: crate::archetype::ArchetypeId,
        moved_ids: &[ComponentId],
        dst_id: crate::archetype::ArchetypeId,
        dst_row: usize,
    ) {
        let moved = {
            let table = self
                .archetypes
                .get_mut(src_id)
                .expect("src archetype")
                .table_mut();
            // SAFETY: `src_loc.row` is in-bounds; `moved_ids` columns were
            // relocated (so they are forgotten, not double-dropped) and the rest
            // are dropped.
            unsafe { table.swap_remove_row(src_loc.row as usize, moved_ids) }
        };
        if let Some(moved_entity) = moved {
            self.entities.set_location(moved_entity, src_loc);
        }
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype_id: dst_id,
                row: dst_row as u32,
            },
        );
    }

    /// Borrow component `T` of `entity`, or `None` if the entity is dead or
    /// lacks the component.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let id = self.components.id_of::<T>()?;
        let loc = self.entities.location(entity)?;
        let arch = self.archetypes.get(loc.archetype_id)?;
        let col = arch.table().column(id)?;
        let row = loc.row as usize;
        if row >= col.len() {
            return None;
        }
        // SAFETY: `row < col.len()` and `T` is exactly the type registered for
        // `id` (we looked `id` up from `T`), so the column stores `T`.
        Some(unsafe { col.get::<T>(row) })
    }

    /// Mutably borrow component `T` of `entity`, or `None` if the entity is
    /// dead or lacks the component.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let change_tick = self.change_tick;
        let id = self.components.id_of::<T>()?;
        let loc = self.entities.location(entity)?;
        let arch = self.archetypes.get_mut(loc.archetype_id)?;
        let col = arch.table_mut().column_mut(id)?;
        let row = loc.row as usize;
        if row >= col.len() {
            return None;
        }
        // Handing out `&mut T` is an unconditional write for change-detection
        // purposes, so stamp the changed tick (mirrors `Mut<T>` deref).
        col.set_changed_tick(row, change_tick);
        // SAFETY: `row < col.len()`, `T` matches `id`, and `&mut self` gives us
        // exclusive access, so forming a unique `&mut T` cannot alias.
        Some(unsafe { col.get_mut::<T>(row) })
    }

    /// The change-detection [`ComponentTicks`](crate::change::ComponentTicks)
    /// recorded for `entity`'s component `T`, or `None` if the entity is dead
    /// or lacks the component.
    pub fn get_ticks<T: Component>(&self, entity: Entity) -> Option<crate::change::ComponentTicks> {
        let id = self.components.id_of::<T>()?;
        let loc = self.entities.location(entity)?;
        let arch = self.archetypes.get(loc.archetype_id)?;
        let col = arch.table().column(id)?;
        let row = loc.row as usize;
        if row >= col.len() {
            return None;
        }
        Some(col.component_ticks(row))
    }

    /// Whether `entity` currently has component `T`.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        let Some(id) = self.components.id_of::<T>() else {
            return false;
        };
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        self.archetypes
            .get(loc.archetype_id)
            .is_some_and(|a| a.contains(id))
    }

    /// Materialise every entity handed out by
    /// [`Entities::reserve_entity`](crate::entity::Entities::reserve_entity)
    /// since the last flush into a live, unplaced slot.
    ///
    /// Run at a synchronization point before reading reserved entities; this is
    /// what [`CommandQueue::apply`](crate::command::CommandQueue::apply) calls
    /// before draining deferred spawns.
    #[inline]
    pub fn flush_reserved(&mut self) {
        self.entities.flush();
    }

    /// Build a reusable [`QueryState`] over data terms `D` with no filter.
    ///
    /// Registers every component `D` names and computes its access set. The
    /// returned state can be iterated with [`QueryState::iter`] /
    /// [`QueryState::iter_mut`] and reused across frames.
    #[inline]
    pub fn query<D: QueryData>(&mut self) -> QueryState<D> {
        QueryState::new(&mut self.components)
    }

    /// Build a reusable [`QueryState`] over data terms `D` narrowed by filter
    /// `F` (e.g. [`With`](crate::query::With) / [`Without`](crate::query::Without)).
    #[inline]
    pub fn query_filtered<D: QueryData, F: QueryFilter>(&mut self) -> QueryState<D, F> {
        QueryState::new(&mut self.components)
    }

    /// Convenience: build a read-only query state and immediately collect its
    /// items. Prefer caching the [`QueryState`] via [`World::query`] in hot
    /// loops; this helper is for one-shot reads and tests.
    pub fn for_each<D: ReadOnlyQueryData>(&mut self, mut f: impl FnMut(D::Item<'_>)) {
        let state = self.query::<D>();
        for item in state.iter(self) {
            f(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position(f32, f32);
    impl Component for Position {}
    #[derive(Debug, PartialEq)]
    struct Velocity(f32, f32);
    impl Component for Velocity {}
    #[derive(Debug, PartialEq)]
    struct Name(alloc::string::String);
    impl Component for Name {}

    #[test]
    fn spawn_get_and_mutate() {
        let mut w = World::new();
        let e = w.spawn((Position(1.0, 2.0), Velocity(3.0, 4.0)));
        assert_eq!(w.entity_count(), 1);
        assert_eq!(w.get::<Position>(e), Some(&Position(1.0, 2.0)));
        assert_eq!(w.get::<Velocity>(e), Some(&Velocity(3.0, 4.0)));
        w.get_mut::<Position>(e).unwrap().0 = 10.0;
        assert_eq!(w.get::<Position>(e), Some(&Position(10.0, 2.0)));
    }

    #[test]
    fn spawn_empty_and_despawn() {
        let mut w = World::new();
        let e = w.spawn(());
        assert!(w.contains(e));
        assert_eq!(w.get::<Position>(e), None);
        assert!(w.despawn(e));
        assert!(!w.contains(e));
        assert!(!w.despawn(e));
    }

    #[test]
    fn insert_adds_and_overwrites() {
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 1.0));
        // Add a new component -> structural move.
        assert!(w.insert(e, Velocity(5.0, 5.0)));
        assert_eq!(w.get::<Position>(e), Some(&Position(1.0, 1.0)));
        assert_eq!(w.get::<Velocity>(e), Some(&Velocity(5.0, 5.0)));
        // Overwrite existing -> in place.
        assert!(w.insert(e, Position(2.0, 2.0)));
        assert_eq!(w.get::<Position>(e), Some(&Position(2.0, 2.0)));
    }

    #[test]
    fn remove_component() {
        let mut w = World::new();
        let e = w.spawn((Position(1.0, 1.0), Velocity(2.0, 2.0)));
        assert!(w.has::<Velocity>(e));
        assert!(w.remove::<Velocity>(e));
        assert!(!w.has::<Velocity>(e));
        assert_eq!(w.get::<Velocity>(e), None);
        assert_eq!(w.get::<Position>(e), Some(&Position(1.0, 1.0)));
        // Removing again is a no-op.
        assert!(!w.remove::<Velocity>(e));
    }

    #[test]
    fn despawn_swaps_and_fixes_locations() {
        let mut w = World::new();
        let a = w.spawn(Position(1.0, 0.0));
        let b = w.spawn(Position(2.0, 0.0));
        let c = w.spawn(Position(3.0, 0.0));
        // Despawn the middle entity; `c` should swap into its slot but still be
        // readable at the correct value.
        assert!(w.despawn(b));
        assert_eq!(w.get::<Position>(a), Some(&Position(1.0, 0.0)));
        assert_eq!(w.get::<Position>(c), Some(&Position(3.0, 0.0)));
        assert!(!w.contains(b));
        assert_eq!(w.entity_count(), 2);
    }

    #[test]
    fn drop_runs_on_despawn() {
        let mut w = World::new();
        let e = w.spawn(Name(alloc::string::String::from("hello")));
        assert_eq!(w.get::<Name>(e).map(|n| n.0.as_str()), Some("hello"));
        // Despawn must drop the String without leaking (miri/asan would catch).
        assert!(w.despawn(e));
    }

    #[derive(Debug, PartialEq, Default)]
    struct FrameCount(u32);
    impl Resource for FrameCount {}

    #[test]
    fn spawn_stamps_added_and_changed_ticks() {
        use crate::change::Tick;
        let mut w = World::new();
        // Fresh world starts at change tick 1.
        assert_eq!(w.change_tick(), Tick::new(1));
        let e = w.spawn(Position(1.0, 2.0));
        let ticks = w.get_ticks::<Position>(e).unwrap();
        assert_eq!(ticks.added, Tick::new(1));
        assert_eq!(ticks.changed, Tick::new(1));
        // Visible as added/changed to a system whose last_run is ZERO.
        assert!(ticks.is_added(Tick::ZERO, w.change_tick()));
        assert!(ticks.is_changed(Tick::ZERO, w.change_tick()));
    }

    #[test]
    fn get_mut_bumps_changed_but_not_added() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 2.0)); // added=changed=1
        w.increment_change_tick(); // -> 2
        w.increment_change_tick(); // -> 3
        w.get_mut::<Position>(e).unwrap().0 = 9.0;
        let ticks = w.get_ticks::<Position>(e).unwrap();
        assert_eq!(ticks.added, Tick::new(1), "added tick preserved");
        assert_eq!(ticks.changed, Tick::new(3), "changed bumped to current");
        // A system that last ran at tick 2 sees the change but not an add.
        assert!(!ticks.is_added(Tick::new(2), w.change_tick()));
        assert!(ticks.is_changed(Tick::new(2), w.change_tick()));
    }

    #[test]
    fn insert_new_component_stamps_current_tick_structural() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 1.0)); // Position added=changed=1
        w.increment_change_tick(); // -> 2
        // Structural move: Velocity is new, Position is relocated unchanged.
        assert!(w.insert(e, Velocity(5.0, 5.0)));
        let pos = w.get_ticks::<Position>(e).unwrap();
        assert_eq!(pos.added, Tick::new(1), "relocation preserves Position ticks");
        assert_eq!(pos.changed, Tick::new(1));
        let vel = w.get_ticks::<Velocity>(e).unwrap();
        assert_eq!(vel.added, Tick::new(2), "new component added at current tick");
        assert_eq!(vel.changed, Tick::new(2));
    }

    #[test]
    fn insert_overwrite_bumps_changed_preserves_added() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 1.0)); // added=changed=1
        w.increment_change_tick(); // -> 2
        assert!(w.insert(e, Position(2.0, 2.0))); // in-place overwrite
        let p = w.get_ticks::<Position>(e).unwrap();
        assert_eq!(p.added, Tick::new(1));
        assert_eq!(p.changed, Tick::new(2));
    }

    #[test]
    fn structural_move_on_remove_preserves_kept_ticks() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn((Position(1.0, 1.0), Velocity(2.0, 2.0))); // both at 1
        w.increment_change_tick(); // -> 2
        assert!(w.remove::<Velocity>(e));
        let p = w.get_ticks::<Position>(e).unwrap();
        assert_eq!(p.added, Tick::new(1), "kept component ticks survive the move");
        assert_eq!(p.changed, Tick::new(1));
    }

    #[test]
    fn check_change_ticks_clamps_stale_component_ticks() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 1.0)); // added=changed=1
        // Fast-forward the world tick far past MAX_CHANGE_AGE.
        w.set_last_change_tick(Tick::new(1));
        w.change_tick = Tick::new(Tick::MAX_CHANGE_AGE.wrapping_add(100));
        w.check_change_ticks();
        let ticks = w.get_ticks::<Position>(e).unwrap();
        // The stale added/changed ticks are clamped to exactly MAX_CHANGE_AGE old.
        assert_eq!(ticks.added.age_since(w.change_tick()), Tick::MAX_CHANGE_AGE);
        assert_eq!(ticks.changed.age_since(w.change_tick()), Tick::MAX_CHANGE_AGE);
    }

    #[test]
    fn world_resource_lifecycle() {
        let mut w = World::new();
        assert!(!w.contains_resource::<FrameCount>());
        assert_eq!(w.get_resource::<FrameCount>(), None);

        // init_resource constructs via Default and is idempotent.
        w.init_resource::<FrameCount>();
        assert!(w.contains_resource::<FrameCount>());
        assert_eq!(w.resource::<FrameCount>(), &FrameCount(0));
        w.init_resource::<FrameCount>();
        assert_eq!(w.resource::<FrameCount>(), &FrameCount(0));

        // Mutation through the world flows to the stored value.
        w.resource_mut::<FrameCount>().0 = 7;
        assert_eq!(w.resource::<FrameCount>(), &FrameCount(7));

        // insert overwrites and returns the previous value.
        assert_eq!(w.insert_resource(FrameCount(100)), Some(FrameCount(7)));
        assert_eq!(w.get_resource::<FrameCount>(), Some(&FrameCount(100)));

        // remove returns the value and leaves the type registered.
        assert_eq!(w.remove_resource::<FrameCount>(), Some(FrameCount(100)));
        assert!(!w.contains_resource::<FrameCount>());
        assert_eq!(w.remove_resource::<FrameCount>(), None);
    }
}
