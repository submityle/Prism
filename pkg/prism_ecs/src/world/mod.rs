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
use crate::component::{Component, ComponentId, ComponentSet, Components, StorageType};
use crate::component_hooks::{ComponentHook, HookContext};
use crate::entity::{Entities, Entity, EntityLocation};
use crate::query::{QueryData, QueryFilter, QueryState, ReadOnlyQueryData};
use crate::storage::SparseSets;
use crate::resource::{Resource, Resources};

/// The authoritative container of all ECS state.
pub struct World {
    entities: Entities,
    components: Components,
    archetypes: Archetypes,
    /// Out-of-band columns for sparse-declared components (design §6); keyed
    /// by [`Entity`], never fragmenting the archetype graph.
    sparse_sets: SparseSets,
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
            sparse_sets: SparseSets::new(),
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
        self.sparse_sets.check_change_ticks(this_run);
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

    /// The out-of-band sparse-component registry (design §6).
    ///
    /// Sparse components live here keyed by [`Entity`], not inside archetype
    /// tables, so the query layer resolves their per-entity membership from
    /// this registry during iteration rather than from the archetype component
    /// set.
    #[inline]
    pub(crate) fn sparse_sets(&self) -> &SparseSets {
        &self.sparse_sets
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

    /// Register component type `T` (idempotent) and attach its lifecycle
    /// [`ComponentHooks`](crate::component_hooks::ComponentHooks) (design §12),
    /// replacing any previously registered hooks. Returns `T`'s [`ComponentId`].
    #[inline]
    pub fn register_component_hooks<T: Component>(
        &mut self,
        hooks: crate::component_hooks::ComponentHooks,
    ) -> ComponentId {
        let id = self.components.register::<T>();
        let ok = self.components.set_hooks(id, hooks);
        debug_assert!(ok, "component just registered must accept hooks");
        id
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

        // Uniqueness is checked over the whole bundle (table + sparse ids).
        let set_all = ComponentSet::from_ids(ids.iter().copied());
        assert_eq!(
            set_all.len(),
            ids.len(),
            "a bundle may not contain the same component type twice"
        );

        // Classify each id by storage; pre-create any sparse sets so the fill
        // closure only has to look them up.
        let storages = self.classify_storages(&ids);
        self.ensure_sparse_sets(&ids, &storages);

        // Only table components fragment the archetype; sparse ones are routed
        // out of band (design §6).
        let table_set = ComponentSet::from_ids(
            ids.iter()
                .zip(&storages)
                .filter(|(_, s)| **s == StorageType::Table)
                .map(|(id, _)| *id),
        );
        let archetype_id = self.archetypes.get_or_create(&table_set, &self.components);

        let row = {
            let World {
                archetypes,
                sparse_sets,
                ..
            } = &mut *self;
            let table = archetypes
                .get_mut(archetype_id)
                .expect("archetype just created")
                .table_mut();
            let row = table.allocate(entity);
            let mut i = 0usize;
            // SAFETY: `get_components` yields one pointer per id in `ids` order,
            // each a valid owned component value routed exactly once into its
            // matching table column or sparse set, restoring both invariants.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    let storage = storages[i];
                    i += 1;
                    match storage {
                        StorageType::Table => {
                            table.column_for_fill(id).push(ptr, change_tick);
                        }
                        StorageType::SparseSet => {
                            sparse_sets
                                .get_mut(id)
                                .expect("sparse set pre-created")
                                .insert(entity, ptr, change_tick);
                        }
                    }
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

        // Lifecycle hooks (design §12): a fresh spawn newly adds every
        // component, so fire on_add then on_insert for all of them with the
        // world fully consistent. Gated so hook-free spawns pay nothing.
        if self.any_hooks(&ids) {
            let add = self.collect_hooks(&ids, crate::component_hooks::ComponentHooks::on_add);
            let insert =
                self.collect_hooks(&ids, crate::component_hooks::ComponentHooks::on_insert);
            self.run_hooks(entity, &add);
            self.run_hooks(entity, &insert);
        }
    }

    /// Gather the `(id, hook)` pairs for every id in `ids` whose component has
    /// the lifecycle hook selected by `select` registered (design §12). The
    /// hook fn pointers are copied out so the returned batch borrows nothing
    /// from `self`, letting the caller fire them with `&mut self`.
    fn collect_hooks(
        &self,
        ids: &[ComponentId],
        select: fn(&crate::component_hooks::ComponentHooks) -> Option<ComponentHook>,
    ) -> Vec<(ComponentId, ComponentHook)> {
        ids.iter()
            .filter_map(|&id| {
                let info = self.components.info(id)?;
                select(info.hooks()).map(|hook| (id, hook))
            })
            .collect()
    }

    /// Fire a previously-collected batch of lifecycle hooks for `entity`, in
    /// order. Each hook sees a fully consistent world; see
    /// [`crate::component_hooks`] for the re-entrancy contract.
    fn run_hooks(&mut self, entity: Entity, hooks: &[(ComponentId, ComponentHook)]) {
        for &(component, hook) in hooks {
            hook(HookContext {
                world: self,
                entity,
                component,
            });
        }
    }

    /// Whether any component in `ids` has *any* lifecycle hook registered —
    /// a cheap gate that lets the structural paths skip all hook bookkeeping
    /// for the overwhelmingly common hook-free case.
    fn any_hooks(&self, ids: &[ComponentId]) -> bool {
        // Global short-circuit first: if the world has never registered a hook,
        // skip the per-id scan entirely.
        self.components.has_hooks()
            && ids.iter().any(|&id| {
                self.components
                    .info(id)
                    .is_some_and(|info| !info.hooks().is_empty())
            })
    }

    /// Every [`ComponentId`] currently resident on `entity`: the table columns
    /// of its archetype plus any out-of-band sparse components, in no
    /// particular order. Used by the despawn path to drive on_replace/on_remove
    /// hooks (design §12).
    fn entity_component_ids(&self, entity: Entity, loc: EntityLocation) -> Vec<ComponentId> {
        let mut out = Vec::new();
        if !loc.is_empty()
            && let Some(arch) = self.archetypes.get(loc.archetype_id)
        {
            out.extend(arch.components().ids().iter().copied());
        }
        out.extend(self.sparse_sets.ids_for(entity));
        out
    }

    /// The subset of `ids` (paired with their `storages`) that `entity`
    /// currently holds — table ids present in its archetype, sparse ids present
    /// in their set. Used by the insert path to decide on_replace vs on_add
    /// (design §12).
    fn present_ids(
        &self,
        entity: Entity,
        ids: &[ComponentId],
        storages: &[StorageType],
    ) -> Vec<ComponentId> {
        let loc = self.entities.location(entity);
        ids.iter()
            .zip(storages)
            .filter_map(|(&id, &storage)| {
                let present = match storage {
                    StorageType::Table => loc.is_some_and(|l| {
                        !l.is_empty()
                            && self
                                .archetypes
                                .get(l.archetype_id)
                                .is_some_and(|a| a.contains(id))
                    }),
                    StorageType::SparseSet => self.sparse_sets.contains(id, entity),
                };
                present.then_some(id)
            })
            .collect()
    }

    /// The declared [`StorageType`] of each id in `ids`, in order.
    fn classify_storages(&self, ids: &[ComponentId]) -> Vec<StorageType> {
        ids.iter()
            .map(|&id| {
                self.components
                    .info(id)
                    .expect("registered component")
                    .storage()
            })
            .collect()
    }

    /// Lazily create the backing sparse set for every sparse id in
    /// `ids` (paired with its storage classification), so later writes can
    /// assume the set exists.
    fn ensure_sparse_sets(&mut self, ids: &[ComponentId], storages: &[StorageType]) {
        for (&id, &storage) in ids.iter().zip(storages) {
            if storage == StorageType::SparseSet {
                let info = self.components.info(id).expect("registered component");
                let (layout, drop) = (info.layout(), info.drop_fn());
                self.sparse_sets.get_or_init(id, layout, drop);
            }
        }
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
        // Liveness is resolved before any sparse set is touched.
        if self.entities.location(entity).is_none() {
            return false;
        }

        let storages = self.classify_storages(&ids);
        self.ensure_sparse_sets(&ids, &storages);

        // Lifecycle hooks (design §12): fire on_replace for every currently
        // present component *before* its value is overwritten, then capture the
        // post-replace present set so on_add fires only for genuinely new
        // components. Gated so hook-free inserts pay nothing.
        let hooks_active = self.any_hooks(&ids);
        let had: Vec<ComponentId> = if hooks_active {
            let present = self.present_ids(entity, &ids, &storages);
            let replace =
                self.collect_hooks(&present, crate::component_hooks::ComponentHooks::on_replace);
            self.run_hooks(entity, &replace);
            // A hook may have despawned the entity; abort without writing.
            if !self.entities.contains(entity) {
                return false;
            }
            self.present_ids(entity, &ids, &storages)
        } else {
            Vec::new()
        };

        // Resolve the (possibly hook-mutated) location fresh before planning.
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

        // Only *table* components the entity lacks force an archetype move;
        // sparse components are routed out of band and never fragment (§6).
        let add_ids: Vec<ComponentId> = ids
            .iter()
            .copied()
            .zip(&storages)
            .filter(|(id, s)| **s == StorageType::Table && !current.contains(*id))
            .map(|(id, _)| id)
            .collect();

        if add_ids.is_empty() {
            // No new table column: overwrite existing table columns in place
            // and insert/overwrite sparse components out of band — no move.
            let World {
                archetypes,
                sparse_sets,
                ..
            } = &mut *self;
            let table = archetypes
                .get_mut(src_id)
                .expect("live entity archetype")
                .table_mut();
            let row = loc.row as usize;
            let mut i = 0usize;
            // SAFETY: `get_components` yields one owned pointer per id in `ids`
            // order; table ids are existing columns overwritten in place at the
            // in-bounds `row`, sparse ids are moved into their pre-created set.
            // Each value is consumed exactly once.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    let storage = storages[i];
                    i += 1;
                    match storage {
                        StorageType::Table => {
                            table
                                .column_mut(id)
                                .expect("overwrite column exists")
                                .replace(row, ptr, change_tick);
                        }
                        StorageType::SparseSet => {
                            sparse_sets
                                .get_mut(id)
                                .expect("sparse set pre-created")
                                .insert(entity, ptr, change_tick);
                        }
                    }
                });
            }
        } else {
            // At least one new table component: move into `current ∪ table add_ids`.
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
            let World {
                archetypes,
                sparse_sets,
                ..
            } = &mut *self;
            let table = archetypes
                .get_mut(dst_id)
                .expect("dst archetype")
                .table_mut();
            let mut i = 0usize;
            // SAFETY: sparse ids move into their pre-created set; table ids in
            // `current` were relocated to `dst_row` and are overwritten in place;
            // genuinely new table ids fill their (empty) column at `dst_row`.
            // Each owned value is consumed exactly once.
            unsafe {
                bundle.get_components(&mut |ptr| {
                    let id = ids[i];
                    let storage = storages[i];
                    i += 1;
                    match storage {
                        StorageType::SparseSet => {
                            sparse_sets
                                .get_mut(id)
                                .expect("sparse set pre-created")
                                .insert(entity, ptr, change_tick);
                        }
                        StorageType::Table if current.contains(id) => {
                            table
                                .column_mut(id)
                                .expect("moved column exists")
                                .replace(dst_row, ptr, change_tick);
                        }
                        StorageType::Table => {
                            table.column_for_fill(id).push(ptr, change_tick);
                        }
                    }
                });
            }
        }

            self.finish_move(entity, loc, src_id, current.ids(), dst_id, dst_row);
        }

        // Post-write lifecycle hooks (design §12): on_add fires for components
        // newly added to the entity on this call, then on_insert for every
        // written component. Both observe a fully consistent world.
        if hooks_active {
            let added: Vec<ComponentId> =
                ids.iter().copied().filter(|id| !had.contains(id)).collect();
            let add = self.collect_hooks(&added, crate::component_hooks::ComponentHooks::on_add);
            let insert =
                self.collect_hooks(&ids, crate::component_hooks::ComponentHooks::on_insert);
            self.run_hooks(entity, &add);
            self.run_hooks(entity, &insert);
        }
        true
    }

    /// Remove the components named by bundle type `B` from `entity`.
    ///
    /// Returns `true` if the entity was live and at least one of the named
    /// components was present (and thus removed).
    pub fn remove<B: Bundle>(&mut self, entity: Entity) -> bool {
        let mut ids = Vec::new();
        B::component_ids(&mut self.components, &mut ids);
        if self.entities.location(entity).is_none() {
            return false;
        }

        let storages = self.classify_storages(&ids);

        // Lifecycle hooks (design §12): for every named component the entity
        // actually holds, fire on_replace then on_remove *before* the value is
        // dropped, so a hook can still read the outgoing value. Gated so
        // hook-free removes pay nothing.
        if self.any_hooks(&ids) {
            let present = self.present_ids(entity, &ids, &storages);
            let replace =
                self.collect_hooks(&present, crate::component_hooks::ComponentHooks::on_replace);
            let remove =
                self.collect_hooks(&present, crate::component_hooks::ComponentHooks::on_remove);
            self.run_hooks(entity, &replace);
            self.run_hooks(entity, &remove);
            // A hook may have despawned the entity; nothing remains to remove.
            if !self.entities.contains(entity) {
                return false;
            }
        }

        // Resolve the (possibly hook-mutated) location fresh before removal.
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };

        // Sparse components are removed out of band with no archetype move.
        let mut removed_any = false;
        for (&id, &storage) in ids.iter().zip(&storages) {
            if storage == StorageType::SparseSet && self.sparse_sets.remove(id, entity) {
                removed_any = true;
            }
        }

        let src_id = loc.archetype_id;
        let current = self
            .archetypes
            .get(src_id)
            .expect("live entity archetype")
            .components()
            .clone();

        // Only *table* components present in the current archetype move it.
        let to_remove: Vec<ComponentId> = ids
            .iter()
            .copied()
            .zip(&storages)
            .filter(|(id, s)| **s == StorageType::Table && current.contains(*id))
            .map(|(id, _)| id)
            .collect();
        if to_remove.is_empty() {
            // No structural change; success iff a sparse component was removed.
            return removed_any;
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
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };

        // Lifecycle hooks (design §12): fire on_replace then on_remove for every
        // component the entity holds while it is still fully live, so hooks can
        // read outgoing values. The global `has_hooks` gate keeps hook-free
        // despawns allocation-free.
        if self.components.has_hooks() {
            let ids = self.entity_component_ids(entity, loc);
            if self.any_hooks(&ids) {
                let replace = self
                    .collect_hooks(&ids, crate::component_hooks::ComponentHooks::on_replace);
                let remove =
                    self.collect_hooks(&ids, crate::component_hooks::ComponentHooks::on_remove);
                self.run_hooks(entity, &replace);
                self.run_hooks(entity, &remove);
                // A hook may already have despawned the entity; it is gone.
                if !self.entities.contains(entity) {
                    return true;
                }
            }
        }

        // Re-fetch and free the (possibly hook-mutated) location.
        let Some(loc) = self.entities.free(entity) else {
            return true;
        };
        // Drop out-of-band sparse components regardless of archetype shape; even
        // an empty-archetype entity may still hold sparse components (§6).
        self.sparse_sets.remove_entity_from_all(entity);
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
        if self.components.info(id)?.storage() == StorageType::SparseSet {
            // SAFETY: `T` is exactly the type registered for `id`.
            return unsafe { self.sparse_sets.get(id)?.get::<T>(entity) };
        }
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
        if self.components.info(id)?.storage() == StorageType::SparseSet {
            let set = self.sparse_sets.get_mut(id)?;
            // Handing out `&mut T` is an unconditional write for change-detection
            // purposes; stamp the changed tick (no-op if the entity is absent).
            set.set_changed_tick(entity, change_tick);
            // SAFETY: `T` matches `id` and `&mut self` gives exclusive access,
            // so the formed `&mut T` cannot alias.
            return unsafe { set.get_mut::<T>(entity) };
        }
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
        if self.components.info(id)?.storage() == StorageType::SparseSet {
            return self.sparse_sets.get(id)?.component_ticks(entity);
        }
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
        if self
            .components
            .info(id)
            .is_some_and(|i| i.storage() == StorageType::SparseSet)
        {
            return self.sparse_sets.contains(id, entity);
        }
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

    // --- SparseSet storage integration (design §6) ---

    #[derive(Debug, PartialEq)]
    struct Selected;
    impl Component for Selected {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    #[derive(Debug, PartialEq)]
    struct Charge(u32);
    impl Component for Charge {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    /// The archetype id the entity currently lives in.
    fn arch_of(w: &World, e: Entity) -> crate::archetype::ArchetypeId {
        w.entities.location(e).unwrap().archetype_id
    }

    #[test]
    fn spawn_with_sparse_excludes_it_from_archetype() {
        let mut w = World::new();
        let e = w.spawn((Position(1.0, 2.0), Selected, Charge(7)));
        // Both table and sparse components are readable.
        assert_eq!(w.get::<Position>(e), Some(&Position(1.0, 2.0)));
        assert!(w.has::<Selected>(e));
        assert_eq!(w.get::<Charge>(e), Some(&Charge(7)));
        // The archetype only tracks the TABLE component; sparse ids are routed
        // out of band and never appear in the archetype's component set.
        let sel_id = w.components.id_of::<Selected>().unwrap();
        let charge_id = w.components.id_of::<Charge>().unwrap();
        let pos_id = w.components.id_of::<Position>().unwrap();
        let arch = w.archetypes.get(arch_of(&w, e)).unwrap();
        assert!(arch.contains(pos_id));
        assert!(!arch.contains(sel_id));
        assert!(!arch.contains(charge_id));
    }

    #[test]
    fn toggling_sparse_does_not_change_archetype() {
        let mut w = World::new();
        let e = w.spawn(Position(1.0, 1.0));
        let a0 = arch_of(&w, e);
        // Insert a sparse component: no structural move.
        assert!(w.insert(e, Selected));
        assert_eq!(arch_of(&w, e), a0, "sparse insert must not fragment");
        assert!(w.has::<Selected>(e));
        // Insert another sparse with data: still no move.
        assert!(w.insert(e, Charge(3)));
        assert_eq!(arch_of(&w, e), a0, "second sparse insert must not fragment");
        // Remove them: still the same archetype.
        assert!(w.remove::<Selected>(e));
        assert_eq!(arch_of(&w, e), a0, "sparse remove must not fragment");
        assert!(!w.has::<Selected>(e));
        assert!(w.remove::<Charge>(e));
        assert_eq!(arch_of(&w, e), a0);
        assert!(!w.has::<Charge>(e));
        // Table data is untouched throughout.
        assert_eq!(w.get::<Position>(e), Some(&Position(1.0, 1.0)));
    }

    #[test]
    fn sparse_entities_with_different_toggles_share_one_archetype() {
        let mut w = World::new();
        let a = w.spawn(Position(1.0, 0.0));
        let b = w.spawn(Position(2.0, 0.0));
        w.insert(a, Selected);
        // `a` has a sparse tag, `b` does not, yet both share the Position-only
        // archetype — the hallmark of non-fragmenting sparse storage.
        assert_eq!(arch_of(&w, a), arch_of(&w, b));
        assert!(w.has::<Selected>(a));
        assert!(!w.has::<Selected>(b));
    }

    #[test]
    fn sparse_get_mut_mutates_and_stamps_changed() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Charge(1));
        w.increment_change_tick(); // -> 2
        w.increment_change_tick(); // -> 3
        w.get_mut::<Charge>(e).unwrap().0 = 42;
        assert_eq!(w.get::<Charge>(e), Some(&Charge(42)));
        let ticks = w.get_ticks::<Charge>(e).unwrap();
        assert_eq!(ticks.added, Tick::new(1), "added preserved");
        assert_eq!(ticks.changed, Tick::new(3), "changed bumped to current");
    }

    #[test]
    fn sparse_insert_overwrite_preserves_added_tick() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Charge(1)); // added=changed=1
        w.increment_change_tick(); // -> 2
        assert!(w.insert(e, Charge(9))); // sparse overwrite
        assert_eq!(w.get::<Charge>(e), Some(&Charge(9)));
        let t = w.get_ticks::<Charge>(e).unwrap();
        assert_eq!(t.added, Tick::new(1), "overwrite preserves the added tick");
        assert_eq!(t.changed, Tick::new(2), "overwrite bumps the changed tick");
    }

    #[test]
    fn despawn_clears_sparse_components() {
        let mut w = World::new();
        let e = w.spawn((Position(1.0, 1.0), Charge(5)));
        let charge_id = w.components.id_of::<Charge>().unwrap();
        assert!(w.sparse_sets.contains(charge_id, e));
        assert!(w.despawn(e));
        // The sparse column no longer holds the despawned entity.
        assert!(!w.sparse_sets.contains(charge_id, e));
        assert!(!w.contains(e));
    }

    #[test]
    fn despawn_empty_archetype_entity_clears_sparse() {
        let mut w = World::new();
        // Entity with ONLY a sparse component lives in the empty archetype.
        let e = w.spawn(Selected);
        let sel_id = w.components.id_of::<Selected>().unwrap();
        assert!(w.sparse_sets.contains(sel_id, e));
        assert!(w.despawn(e));
        assert!(!w.sparse_sets.contains(sel_id, e));
    }

    #[test]
    fn remove_returns_true_only_when_sparse_present() {
        let mut w = World::new();
        let e = w.spawn(Position(0.0, 0.0));
        // Nothing sparse yet: removing a sparse component is a no-op.
        assert!(!w.remove::<Selected>(e));
        w.insert(e, Selected);
        assert!(w.remove::<Selected>(e));
        assert!(!w.remove::<Selected>(e));
    }

    #[test]
    fn sparse_check_change_ticks_clamps_stale() {
        use crate::change::Tick;
        let mut w = World::new();
        let e = w.spawn(Charge(1)); // added=changed=1
        w.set_last_change_tick(Tick::new(1));
        w.change_tick = Tick::new(Tick::MAX_CHANGE_AGE.wrapping_add(100));
        w.check_change_ticks();
        let ticks = w.get_ticks::<Charge>(e).unwrap();
        assert_eq!(ticks.added.age_since(w.change_tick()), Tick::MAX_CHANGE_AGE);
        assert_eq!(ticks.changed.age_since(w.change_tick()), Tick::MAX_CHANGE_AGE);
    }
}
