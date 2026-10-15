//! Transactional live schema replacement for runtime components (design §16.2).
//!
//! Conversion only borrows the old values. All conversions and allocations
//! finish before any old value is moved. Commit preserves entity locations,
//! archetype identities, shared bindings, sparse ordering and owning groups.
//! Like snapshot restore, schema replacement does not emit lifecycle events.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::archetype::ArchetypeId;
use crate::change::{ComponentTicks, Tick};
use crate::component::{CloneFn, ComponentId, DropFn, RequiredCtor, SnapshotHashFn, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::entity::Entity;
use crate::observer::LifecycleEvent;
use crate::storage::{ComponentSparseSet, Table};

use super::World;

/// The replacement ABI and glue for one runtime component.
///
/// Storage strategy, component id and name remain unchanged. Clone and hash
/// glue are replaced wholesale: `None` clears old glue, which must never be
/// applied to values with the new ABI.
#[derive(Clone, Copy)]
pub struct DynamicComponentLayout {
    layout: Layout,
    drop: Option<DropFn>,
    clone: Option<CloneFn>,
    hash: Option<SnapshotHashFn>,
}

impl DynamicComponentLayout {
    /// Describe a runtime value's layout and operations.
    ///
    /// # Safety
    /// Every value initialized for this descriptor must be a valid, movable,
    /// `Send + Sync + 'static` value of one consistent runtime type. Glue must
    /// operate on exactly that type and layout, following the contracts of
    /// [`DropFn`], [`CloneFn`] and [`SnapshotHashFn`]. Its code must remain
    /// loaded while any live value or snapshot can use it. Drop glue must not
    /// unwind during another panic (the usual Rust destructor requirement).
    pub unsafe fn new(
        layout: Layout,
        drop: Option<DropFn>,
        clone: Option<CloneFn>,
        hash: Option<SnapshotHashFn>,
    ) -> Self {
        Self {
            layout,
            drop,
            clone,
            hash,
        }
    }

    /// Layout of one initialized value, including tail padding.
    pub fn layout(self) -> Layout {
        self.layout
    }

    /// Replacement destructor, or `None` for trivially dropped values.
    pub fn drop_fn(self) -> Option<DropFn> {
        self.drop
    }

    /// Replacement snapshot clone operation.
    pub fn clone_fn(self) -> Option<CloneFn> {
        self.clone
    }

    /// Replacement deterministic snapshot hash operation.
    pub fn snapshot_hash_fn(self) -> Option<SnapshotHashFn> {
        self.hash
    }

    fn validate(self) -> Result<(), DynamicComponentError> {
        if self.layout.size() % self.layout.align() != 0 {
            return Err(DynamicComponentError::UnpaddedLayout);
        }
        Ok(())
    }
}

/// A runtime value borrowed from a particular schema of the world.
///
/// The borrow prevents safe reload or structural mutation while it is live.
/// Raw pointers do not extend the borrow: retaining one across world mutation
/// requires the caller to establish its validity again.
#[derive(Clone, Copy)]
pub struct DynamicValueRef<'w> {
    ptr: NonNull<u8>,
    layout: Layout,
    _borrow: PhantomData<&'w [u8]>,
}

impl DynamicValueRef<'_> {
    /// The current schema's physical layout.
    pub fn layout(self) -> Layout {
        self.layout
    }

    /// A read-only pointer, valid only for the duration of this borrow.
    /// Padding bytes may be uninitialized; this is not a byte slice.
    pub fn as_ptr(self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    /// Interpret the value using a runtime schema known to the caller.
    ///
    /// # Safety
    /// `T` must be the exact current runtime type, not merely a type with the
    /// same layout. It must not be used to mutate the source during migration,
    /// including through interior mutability or pointers held by the value.
    pub unsafe fn as_ref<T>(&self) -> &T {
        assert_eq!(
            self.layout,
            Layout::new::<T>(),
            "runtime value layout mismatch"
        );
        // SAFETY: exact-type contract above, lifetime tied to this borrow.
        unsafe { self.ptr.cast::<T>().as_ref() }
    }
}

/// Exclusive runtime-component borrow. Merely borrowing or reading it does not
/// mark the value changed; requesting a mutable pointer/reference stamps the
/// row and (for Table storage) its dirty chunk with the current world tick.
///
/// The borrow prevents safe structural changes or reload while it is live.
pub struct DynamicValueMut<'w> {
    ptr: NonNull<u8>,
    layout: Layout,
    added: Tick,
    changed: &'w mut Tick,
    chunk_changed: Option<NonNull<crate::storage::ChunkChange>>,
    this_run: Tick,
    _borrow: PhantomData<&'w mut [u8]>,
}

impl DynamicValueMut<'_> {
    /// The current runtime layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The current added/changed ticks, including writes through this borrow.
    pub fn ticks(&self) -> ComponentTicks {
        ComponentTicks {
            added: self.added,
            changed: *self.changed,
        }
    }

    /// Read-only pointer valid for this borrow. Padding may be uninitialized.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    /// Mark changed and obtain a pointer for in-place mutation. A raw writer
    /// must leave a valid value of the exact current runtime type, even when it
    /// unwinds. To replace owned values, prefer `World::insert_dynamic_component`.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        *self.changed = self.this_run;
        if let Some(chunk) = self.chunk_changed {
            // SAFETY: unique World borrow; the chunk cell is distinct from the
            // value and row-tick cells and remains live for this borrow.
            unsafe { chunk.as_ref().record(self.this_run) };
        }
        self.ptr.as_ptr()
    }

    /// Interpret a shared borrow using the exact current runtime type.
    ///
    /// # Safety
    /// `T` must be the exact runtime type, not merely layout-compatible.
    pub unsafe fn as_ref<T>(&self) -> &T {
        assert_eq!(
            self.layout,
            Layout::new::<T>(),
            "runtime value layout mismatch"
        );
        // SAFETY: forwarded exact-type contract; lifetime tied to this borrow.
        unsafe { self.ptr.cast::<T>().as_ref() }
    }

    /// Mark changed and interpret an exclusive borrow of the runtime value.
    ///
    /// # Safety
    /// `T` must be the exact runtime type, including its registered glue.
    pub unsafe fn as_mut<T>(&mut self) -> &mut T {
        assert_eq!(
            self.layout,
            Layout::new::<T>(),
            "runtime value layout mismatch"
        );
        // SAFETY: forwarded exact-type contract and unique World borrow.
        unsafe { &mut *self.as_mut_ptr().cast::<T>() }
    }
}

/// One aligned, owned output slot supplied to a migration callback.
///
/// Writing marks ownership immediately. Returning an error or panicking after
/// a write drops that output exactly once; returning success without a write
/// rejects the entire migration. The slot can be initialized only once.
pub struct DynamicValueSlot {
    ptr: NonNull<u8>,
    descriptor: DynamicComponentLayout,
    initialized: bool,
}

impl DynamicValueSlot {
    fn new(descriptor: DynamicComponentLayout) -> Self {
        let layout = descriptor.layout;
        let ptr = if layout.size() == 0 {
            // Aligned, non-null address for ZSTs; no allocation or bytes used.
            NonNull::new(core::ptr::without_provenance_mut(layout.align())).unwrap()
        } else {
            // SAFETY: layout has a nonzero size and valid alignment.
            NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout))
        };
        Self {
            ptr,
            descriptor,
            initialized: false,
        }
    }

    /// The required output layout.
    pub fn layout(&self) -> Layout {
        self.descriptor.layout
    }

    /// Whether an owned output has already been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Initialize the slot with a Rust value representing the runtime schema.
    ///
    /// # Safety
    /// `T` must be the exact output runtime type described by the descriptor,
    /// including its drop/clone/hash glue. The value must independently own
    /// its resources; it must not borrow or steal ownership from the source.
    pub unsafe fn write<T: Send + Sync + 'static>(&mut self, value: T) {
        assert!(!self.initialized, "runtime output already initialized");
        assert_eq!(
            self.layout(),
            Layout::new::<T>(),
            "runtime output layout mismatch"
        );
        // SAFETY: allocation has T's layout, the slot is uninitialized, and
        // the caller guarantees T matches all the registered glue.
        unsafe { self.ptr.as_ptr().cast::<T>().write(value) };
        self.initialized = true;
    }

    /// Uninitialized aligned output storage for a foreign runtime.
    ///
    /// Raw writers must call [`Self::assume_initialized`] immediately after a
    /// complete write, before executing code that can return an error or panic.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        assert!(!self.initialized, "runtime output already initialized");
        self.ptr.as_ptr()
    }

    /// Record ownership after initializing through [`Self::as_mut_ptr`].
    ///
    /// # Safety
    /// The slot must contain one fully initialized value of its descriptor's
    /// runtime type. Its ownership must be independent of the old live value.
    pub unsafe fn assume_initialized(&mut self) {
        assert!(!self.initialized, "runtime output already initialized");
        self.initialized = true;
    }
}

impl Drop for DynamicValueSlot {
    fn drop(&mut self) {
        struct Allocation(NonNull<u8>, Layout);
        impl Drop for Allocation {
            fn drop(&mut self) {
                if self.1.size() != 0 {
                    // SAFETY: this guard uniquely owns this exact allocation.
                    unsafe { dealloc(self.0.as_ptr(), self.1) };
                }
            }
        }
        let _allocation = Allocation(self.ptr, self.descriptor.layout);
        if core::mem::take(&mut self.initialized) {
            if let Some(drop) = self.descriptor.drop {
                // SAFETY: the flag proves initialization and unique ownership;
                // it was cleared before calling user code, even if it unwinds.
                unsafe { drop(self.ptr.as_ptr()) };
            }
        }
    }
}

/// A schema that cannot be hot-reloaded by this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicComponentError {
    /// The component id is not registered in this world.
    UnknownComponent(ComponentId),
    /// Rust-typed components cannot change their ABI behind typed queries.
    RustType(ComponentId),
    /// Shared values have external Arc and interning/equality contracts.
    /// Register a new shared component id and explicitly rebind its holders.
    SharedStorage,
    /// Array elements require size to be a multiple of alignment.
    UnpaddedLayout,
    /// Lifecycle hooks may still interpret the outgoing schema.
    HasHooks(ComponentId),
    /// A required-component constructor still produces the outgoing schema.
    RequiredConstructor(ComponentId),
    /// A lifecycle callback replaced the world or its runtime schema. The
    /// surrounding operation stops rather than using its stale descriptor.
    ContextChanged,
}

/// Why a transaction was rejected before changing any live value or metadata.
#[derive(Debug, PartialEq, Eq)]
pub enum DynamicReloadError<E> {
    /// Invalid schema or incompatible registration.
    Component(DynamicComponentError),
    /// Conversion failed for this entity; all staged outputs are discarded.
    Migration {
        /// Entity whose conversion or initialization failed.
        entity: Entity,
        /// Error returned by the caller's conversion or initializer.
        error: E,
    },
    /// The callback returned success without initializing its output.
    Uninitialized {
        /// Entity for which no owned output was produced.
        entity: Entity,
    },
}

impl<E> From<DynamicComponentError> for DynamicReloadError<E> {
    fn from(error: DynamicComponentError) -> Self {
        Self::Component(error)
    }
}

/// The completed live migration's scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicReloadReport {
    /// Number of converted live values.
    pub values: usize,
    /// Number of rebuilt tables, including empty tables with this schema.
    pub tables: usize,
}

struct PreparedTable {
    id: ArchetypeId,
    columns: Vec<ComponentId>,
    table: Table,
}

impl World {
    /// Register one runtime schema and all its snapshot glue. Each call returns
    /// a fresh, world-local id. Table and SparseSet support owned live values;
    /// Shared requires boxing/equality/interning contracts not described by a
    /// layout, so is rejected before creating a registration.
    pub fn register_dynamic_component(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        storage: StorageType,
        descriptor: DynamicComponentLayout,
    ) -> Result<ComponentId, DynamicComponentError> {
        descriptor.validate()?;
        if storage == StorageType::Shared {
            return Err(DynamicComponentError::SharedStorage);
        }
        let id =
            self.components
                .register_dynamic(name, descriptor.layout, storage, descriptor.drop);
        self.components.replace_dynamic_layout(
            id,
            descriptor.layout,
            descriptor.drop,
            descriptor.clone,
            descriptor.hash,
            self.components.schema_generation(),
        );
        Ok(id)
    }

    /// Attach lifecycle hooks to a runtime component. As with typed hooks,
    /// callbacks execute against a consistent World. Hooks must be removed
    /// before changing a schema they may interpret.
    pub fn register_dynamic_component_hooks(
        &mut self,
        id: ComponentId,
        hooks: ComponentHooks,
    ) -> Result<(), DynamicComponentError> {
        self.dynamic_descriptor(id)?;
        self.components.set_hooks(id, hooks);
        Ok(())
    }

    /// Register an auto-constructed requirement of a runtime component.
    /// Returns false for a self-requirement. Missing Shared requirements are
    /// unsupported by the layout-only API and rejected here.
    ///
    /// # Safety
    /// `ctor` must transfer exactly one initialized, independently owned value
    /// of `required`'s exact registered type/layout to its callback, and never
    /// use or drop that value afterwards (including during unwind). Its code
    /// and schema-specific glue must remain loaded while registered.
    pub unsafe fn register_dynamic_required_component(
        &mut self,
        id: ComponentId,
        required: ComponentId,
        ctor: RequiredCtor,
    ) -> Result<bool, DynamicComponentError> {
        self.dynamic_descriptor(id)?;
        let info = self
            .components
            .info(required)
            .ok_or(DynamicComponentError::UnknownComponent(required))?;
        if info.storage() == StorageType::Shared {
            return Err(DynamicComponentError::SharedStorage);
        }
        Ok(self.components.register_required(id, required, ctor))
    }

    fn dynamic_descriptor(
        &self,
        id: ComponentId,
    ) -> Result<DynamicComponentLayout, DynamicComponentError> {
        let info = self
            .components
            .info(id)
            .ok_or(DynamicComponentError::UnknownComponent(id))?;
        if info.type_id().is_some() {
            return Err(DynamicComponentError::RustType(id));
        }
        if info.storage() == StorageType::Shared {
            return Err(DynamicComponentError::SharedStorage);
        }
        Ok(DynamicComponentLayout {
            layout: info.layout(),
            drop: info.drop_fn(),
            clone: info.clone_fn(),
            hash: info.snapshot_hash_fn(),
        })
    }

    fn dynamic_lifecycle(
        &mut self,
        entity: Entity,
        ids: &[ComponentId],
        event: LifecycleEvent,
        context: (super::WorldId, u64),
    ) -> Result<(), DynamicComponentError> {
        let select = match event {
            LifecycleEvent::Add => ComponentHooks::on_add,
            LifecycleEvent::Insert => ComponentHooks::on_insert,
            LifecycleEvent::Replace => ComponentHooks::on_replace,
            LifecycleEvent::Remove => ComponentHooks::on_remove,
        };
        let hooks = self.collect_hooks(ids, select);
        for (component, hook) in hooks {
            if !self.contains(entity) {
                return Ok(());
            }
            hook(HookContext {
                world: self,
                entity,
                component,
            });
            self.check_dynamic_context(context)?;
        }
        if self.contains(entity) {
            self.fire_lifecycle_observers(event, entity, ids);
            self.check_dynamic_context(context)?;
        }
        Ok(())
    }

    fn check_dynamic_context(
        &self,
        context: (super::WorldId, u64),
    ) -> Result<(), DynamicComponentError> {
        if context != (self.id(), self.components.schema_generation()) {
            return Err(DynamicComponentError::ContextChanged);
        }
        Ok(())
    }

    /// Initialize and insert/replace an owned runtime value by id. Returns
    /// false for an entity without live storage and does not invoke `initialize`.
    /// Initializer errors/panics/uninitialized outputs leave live values and
    /// ticks unchanged and invoke no lifecycle callbacks. Missing transitive
    /// requirements are constructed once, without overwriting resident values.
    ///
    /// Successful initialization precedes pre-write hooks. Hook side effects
    /// are ordinary World mutations and are not rolled back on a later panic
    /// or error. Replacement fires Replace before commit, Add for new values,
    /// then Insert for all written values. Old destructors run after storage,
    /// locations and owning groups have been published; a destructor or post-
    /// write hook panic leaves that publication intact.
    pub fn insert_dynamic_component<E>(
        &mut self,
        entity: Entity,
        id: ComponentId,
        initialize: impl FnOnce(&mut DynamicValueSlot) -> Result<(), E>,
    ) -> Result<bool, DynamicReloadError<E>> {
        let descriptor = self.dynamic_descriptor(id)?;
        if self
            .entities
            .location(entity)
            .is_none_or(|loc| loc.is_empty())
        {
            return Ok(false);
        }
        let mut value = DynamicValueSlot::new(descriptor);
        initialize(&mut value).map_err(|error| DynamicReloadError::Migration { entity, error })?;
        if !value.initialized {
            return Err(DynamicReloadError::Uninitialized { entity });
        }
        let context = (self.id(), self.components.schema_generation());
        let storage = self.components.info(id).unwrap().storage();
        if !self.present_ids(entity, &[id], &[storage]).is_empty() {
            self.dynamic_lifecycle(entity, &[id], LifecycleEvent::Replace, context)?;
        }
        let Some(loc) = self.entities.location(entity).filter(|loc| !loc.is_empty()) else {
            return Ok(false);
        };
        let had = !self.present_ids(entity, &[id], &[storage]).is_empty();
        let mut staged = alloc::vec![(id, storage, value)];
        for (rid, ctor) in self.collect_required(&[id]) {
            let info = self.components.info(rid).unwrap();
            let rstorage = info.storage();
            if !self.present_ids(entity, &[rid], &[rstorage]).is_empty() {
                continue;
            }
            if rstorage == StorageType::Shared {
                return Err(DynamicComponentError::SharedStorage.into());
            }
            let mut value = DynamicValueSlot::new(DynamicComponentLayout {
                layout: info.layout(),
                drop: info.drop_fn(),
                clone: info.clone_fn(),
                hash: info.snapshot_hash_fn(),
            });
            ctor(&mut |ptr| {
                assert!(
                    !value.initialized,
                    "required constructor yielded more than one value"
                );
                // SAFETY: registered constructor transfers exactly one value
                // of this descriptor, including ownership if it later panics.
                unsafe {
                    core::ptr::copy_nonoverlapping(ptr, value.ptr.as_ptr(), value.layout().size())
                };
                value.initialized = true;
            });
            if !value.initialized {
                return Err(DynamicReloadError::Uninitialized { entity });
            }
            staged.push((rid, rstorage, value));
        }
        let ids: Vec<_> = staged.iter().map(|(id, _, _)| *id).collect();
        let storages: Vec<_> = staged.iter().map(|(_, storage, _)| *storage).collect();
        self.ensure_sparse_sets(&ids, &storages);
        let source = self.archetypes.get(loc.archetype_id).unwrap();
        let current = source.components().clone();
        let bindings = source.shared_bindings().to_vec();
        let mut destination = current.clone();
        for &(cid, storage, _) in &staged {
            if storage == StorageType::Table {
                destination = destination.with(cid);
            }
        }
        let pairs = source.relation_pairs().to_vec();
        let dst_id = self.archetypes.get_or_create_with_pairs(
            &destination,
            &bindings,
            &pairs,
            &self.components,
        );
        let mut retired = DynamicValueSlot::new(descriptor);
        let now = self.change_tick;
        // Detach the previous explicit value before overwriting, but do not
        // invoke its destructor until all storage and group metadata agree.
        if had {
            let ptr = self.get_dynamic_component(entity, id)?.unwrap().as_ptr();
            // SAFETY: initialized old value; ownership moves out at commit,
            // with no callback between this copy and its storage overwrite.
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, retired.ptr.as_ptr(), descriptor.layout.size())
            };
        }
        if dst_id != loc.archetype_id {
            let dst = self.archetypes.get_mut(dst_id).unwrap().table_mut();
            dst.reserve(1);
            let dst_row = dst.allocate(entity);
            {
                let (src, dst) = self.archetypes.get_pair_mut(loc.archetype_id, dst_id);
                for &cid in destination.ids() {
                    let old = src.table().column(cid);
                    let added = old.map_or(now, |col| col.added_tick(loc.row as usize));
                    if let Some((_, _, output)) = staged.iter_mut().find(|(id, _, _)| *id == cid) {
                        // SAFETY: staged initialized value has this column's
                        // exact ABI and transfers ownership to the reserved row.
                        unsafe {
                            dst.table_mut().column_for_fill(cid).push_with_ticks(
                                output.ptr.as_ptr(),
                                added,
                                now,
                            )
                        };
                        output.initialized = false;
                    } else {
                        let old = old.unwrap();
                        // SAFETY: unchanged column relocates with exact ticks.
                        unsafe {
                            dst.table_mut().column_for_fill(cid).push_with_ticks(
                                old.get_ptr(loc.row as usize),
                                added,
                                old.changed_tick(loc.row as usize),
                            )
                        };
                    }
                }
            }
            // Every source column was moved either to destination or retired.
            self.finish_move(
                entity,
                loc,
                loc.archetype_id,
                current.ids(),
                dst_id,
                dst_row,
            );
        } else {
            for (cid, storage, output) in &mut staged {
                if *storage == StorageType::Table {
                    let column = self
                        .archetypes
                        .get_mut(dst_id)
                        .unwrap()
                        .table_mut()
                        .column_mut(*cid)
                        .unwrap();
                    // SAFETY: existing live row; old explicit value was retired
                    // above. Requirements cannot overwrite resident columns.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            output.ptr.as_ptr(),
                            column.get_ptr(loc.row as usize),
                            output.layout().size(),
                        )
                    };
                    output.initialized = false;
                    column.set_changed_tick(loc.row as usize, now);
                }
            }
        }
        for (cid, storage, output) in &mut staged {
            if *storage != StorageType::SparseSet {
                continue;
            }
            let set = self.sparse_sets.get_mut(*cid).unwrap();
            if let Some(ptr) = unsafe { set.get_ptr(entity) } {
                // SAFETY: only the explicit old value can be present, and was
                // detached above; replacement preserves its original added tick.
                unsafe {
                    core::ptr::copy_nonoverlapping(output.ptr.as_ptr(), ptr, output.layout().size())
                };
                set.set_changed_tick(entity, now);
            } else {
                // SAFETY: initialized staged value, absent entity, exact ABI.
                unsafe { set.insert(entity, output.ptr.as_ptr(), now) };
            }
            output.initialized = false;
        }
        retired.initialized = had;
        self.update_owning_groups(entity, &ids);
        drop(retired);
        let added: Vec<_> = ids
            .iter()
            .copied()
            .filter(|cid| *cid != id || !had)
            .collect();
        self.dynamic_lifecycle(entity, &added, LifecycleEvent::Add, context)?;
        self.dynamic_lifecycle(entity, &ids, LifecycleEvent::Insert, context)?;
        Ok(true)
    }

    /// Remove an owned runtime value by id, returning false when absent.
    /// Replace and Remove callbacks run while the outgoing value is readable.
    /// Required components remain resident, matching typed removal: requirements
    /// supply insertion defaults, not a permanent membership constraint.
    ///
    /// Callback mutations are not rolled back. Storage, entity locations and
    /// owning groups are repaired before old drop glue runs, so a destructor
    /// panic leaves removal committed and the world usable.
    pub fn remove_dynamic_component(
        &mut self,
        entity: Entity,
        id: ComponentId,
    ) -> Result<bool, DynamicComponentError> {
        let descriptor = self.dynamic_descriptor(id)?;
        let storage = self.components.info(id).unwrap().storage();
        if self.present_ids(entity, &[id], &[storage]).is_empty() {
            return Ok(false);
        }
        let context = (self.id(), self.components.schema_generation());
        self.dynamic_lifecycle(entity, &[id], LifecycleEvent::Replace, context)?;
        if self.present_ids(entity, &[id], &[storage]).is_empty() {
            return Ok(false);
        }
        self.dynamic_lifecycle(entity, &[id], LifecycleEvent::Remove, context)?;
        let Some(loc) = self.entities.location(entity).filter(|loc| !loc.is_empty()) else {
            return Ok(false);
        };
        if self.present_ids(entity, &[id], &[storage]).is_empty() {
            return Ok(false);
        }
        let mut retired = DynamicValueSlot::new(descriptor);
        if storage == StorageType::SparseSet {
            // SAFETY: independent aligned empty slot with the exact old ABI.
            retired.initialized = unsafe {
                self.sparse_sets
                    .get_mut(id)
                    .unwrap()
                    .remove_and_copy_out(entity, retired.ptr.as_ptr())
            };
        } else {
            let source = self.archetypes.get(loc.archetype_id).unwrap();
            let current = source.components().clone();
            let bindings = source.shared_bindings().to_vec();
            let destination = current.without(id);
            let pairs = source.relation_pairs().to_vec();
            let dst_id = self.archetypes.get_or_create_with_pairs(
                &destination,
                &bindings,
                &pairs,
                &self.components,
            );
            let dst = self.archetypes.get_mut(dst_id).unwrap().table_mut();
            dst.reserve(1);
            let dst_row = dst.allocate(entity);
            {
                let (src, dst) = self.archetypes.get_pair_mut(loc.archetype_id, dst_id);
                // SAFETY: no user code runs until all ownership and metadata
                // have moved. Retired owns the target; dst owns all kept columns.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src.table().column(id).unwrap().get_ptr(loc.row as usize),
                        retired.ptr.as_ptr(),
                        descriptor.layout.size(),
                    );
                    dst.table_mut()
                        .move_shared_columns_from(src.table_mut(), loc.row as usize);
                }
            }
            self.finish_move(
                entity,
                loc,
                loc.archetype_id,
                current.ids(),
                dst_id,
                dst_row,
            );
            retired.initialized = true;
        }
        self.update_owning_groups(entity, &[id]);
        drop(retired);
        Ok(true)
    }

    /// Borrow a runtime value exclusively, with lazy change tracking. Neither
    /// hooks nor Insert observers fire for in-place mutation, as with `get_mut`.
    pub fn get_dynamic_component_mut(
        &mut self,
        entity: Entity,
        id: ComponentId,
    ) -> Result<Option<DynamicValueMut<'_>>, DynamicComponentError> {
        let descriptor = self.dynamic_descriptor(id)?;
        let Some(loc) = self.entities.location(entity).filter(|loc| !loc.is_empty()) else {
            return Ok(None);
        };
        let (ptr, added, changed, chunk_changed) = match self.components.info(id).unwrap().storage()
        {
            StorageType::Table => {
                let Some(column) = self
                    .archetypes
                    .get(loc.archetype_id)
                    .unwrap()
                    .table()
                    .column(id)
                else {
                    return Ok(None);
                };
                // SAFETY: live row, exclusive World borrow. Bytes, row tick
                // and chunk tick are distinct cells and cannot be relocated.
                unsafe {
                    (
                        column.get_ptr(loc.row as usize),
                        column.added_tick(loc.row as usize),
                        column.changed_tick_ptr(loc.row as usize),
                        NonNull::new(column.chunk_changed_ptr(loc.row as usize)),
                    )
                }
            }
            StorageType::SparseSet => {
                let Some(set) = self.sparse_sets.get(id) else {
                    return Ok(None);
                };
                // SAFETY: exclusive World borrow; entity presence is checked.
                let Some(ptr) = (unsafe { set.get_ptr(entity) }) else {
                    return Ok(None);
                };
                (
                    ptr,
                    set.added_tick(entity).unwrap(),
                    unsafe { set.changed_tick_ptr(entity).unwrap() },
                    None,
                )
            }
            StorageType::Shared => unreachable!(),
        };
        Ok(Some(DynamicValueMut {
            ptr: NonNull::new(ptr).unwrap(),
            layout: descriptor.layout,
            added,
            changed: unsafe { &mut *changed },
            chunk_changed,
            this_run: self.change_tick,
            _borrow: PhantomData,
        }))
    }

    /// Borrow an erased component without extending its lifetime past the world.
    /// No typed reference or initialized padding-byte slice is fabricated.
    pub fn get_dynamic_component(
        &self,
        entity: Entity,
        id: ComponentId,
    ) -> Result<Option<DynamicValueRef<'_>>, DynamicComponentError> {
        let info = self
            .components
            .info(id)
            .ok_or(DynamicComponentError::UnknownComponent(id))?;
        if info.type_id().is_some() {
            return Err(DynamicComponentError::RustType(id));
        }
        if info.storage() == StorageType::Shared {
            return Err(DynamicComponentError::SharedStorage);
        }
        let Some(location) = self.entities.location(entity).filter(|loc| !loc.is_empty()) else {
            return Ok(None);
        };
        let ptr = match info.storage() {
            StorageType::Table => {
                let table = self.archetypes.get(location.archetype_id).unwrap().table();
                let Some(column) = table.column(id) else {
                    return Ok(None);
                };
                // SAFETY: the live entity location identifies an initialized row.
                Some(unsafe { column.get_ptr(location.row as usize) })
            }
            StorageType::SparseSet => {
                // SAFETY: only existence is resolved here, without interpreting
                // the pointer or granting mutable access.
                self.sparse_sets
                    .get(id)
                    .and_then(|set| unsafe { set.get_ptr(entity) })
            }
            StorageType::Shared => unreachable!(),
        };
        Ok(ptr.map(|ptr| DynamicValueRef {
            ptr: NonNull::new(ptr).unwrap(),
            layout: info.layout(),
            _borrow: PhantomData,
        }))
    }

    /// Replace every live value of a runtime component and publish its new ABI.
    ///
    /// Conversion sees the old schema and must initialize an independently
    /// owned output. Errors, missing outputs and callback panics leave the live
    /// world, ticks and generations unchanged. Raw accesses must obey the
    /// source/slot contracts, including not mutating the source's referents.
    /// Callbacks run in archetype/row order (Table) or dense order (SparseSet).
    ///
    /// Success preserves added ticks and stamps changed ticks with the current
    /// world tick. It invalidates query match caches and registry-bound
    /// snapshots, including same-layout value-only migrations. Existing
    /// snapshots retain their own old glue and remain safe to inspect/drop;
    /// they must be rejected by restore's schema-generation check.
    ///
    /// Storage strategy and membership do not change. Shared and Rust-typed
    /// components are rejected, as are hooks and required constructors that
    /// could still use the old schema. No lifecycle callbacks are invoked.
    ///
    /// Old destructors run only after publication. If one panics, the new world
    /// is already committed and remains valid; this is not conversion failure.
    /// As with ordinary Rust cleanup, destructors must not panic during unwind.
    pub fn reload_dynamic_component<E>(
        &mut self,
        id: ComponentId,
        descriptor: DynamicComponentLayout,
        mut migrate: impl FnMut(Entity, DynamicValueRef<'_>, &mut DynamicValueSlot) -> Result<(), E>,
    ) -> Result<DynamicReloadReport, DynamicReloadError<E>> {
        descriptor.validate()?;
        let info = self
            .components
            .info(id)
            .ok_or(DynamicComponentError::UnknownComponent(id))?;
        if info.type_id().is_some() {
            return Err(DynamicComponentError::RustType(id).into());
        }
        if info.storage() == StorageType::Shared {
            return Err(DynamicComponentError::SharedStorage.into());
        }
        if !info.hooks().is_empty() {
            return Err(DynamicComponentError::HasHooks(id).into());
        }
        if (0..self.components.len()).any(|i| {
            self.components
                .info(ComponentId::new(i as u32))
                .unwrap()
                .required()
                .iter()
                .any(|required| required.id() == id)
        }) {
            return Err(DynamicComponentError::RequiredConstructor(id).into());
        }
        // Reserve the revision before any ownership changes. Exhaustion must
        // never panic after replacement tables have already been published.
        let next_schema_generation = self
            .components
            .schema_generation()
            .checked_add(1)
            .expect("runtime component schema generation exhausted");
        let old_layout = info.layout();
        let old_drop = info.drop_fn();
        let storage = info.storage();
        let now = self.change_tick;
        let convert = |entity,
                       ptr,
                       migrate: &mut dyn FnMut(
            Entity,
            DynamicValueRef<'_>,
            &mut DynamicValueSlot,
        ) -> Result<(), E>| {
            let mut output = DynamicValueSlot::new(descriptor);
            let input = DynamicValueRef {
                ptr: NonNull::new(ptr).unwrap(),
                layout: old_layout,
                _borrow: PhantomData,
            };
            migrate(entity, input, &mut output)
                .map_err(|error| DynamicReloadError::Migration { entity, error })?;
            if !output.initialized {
                return Err(DynamicReloadError::Uninitialized { entity });
            }
            Ok(output)
        };

        // Stage every affected table, including empty cached archetypes. Only
        // the replacement column is filled here; other columns remain empty
        // and therefore cannot drop unowned, copied source values on failure.
        let mut tables = Vec::new();
        let mut sparse = None;
        // Keep outputs in individually owned slots throughout conversion.
        // Vec's unwind guard cleans every remaining slot if one staged drop
        // panics while handling a returned error (no storage destructor leaks).
        let mut outputs = Vec::new();
        let mut values = 0;
        if storage == StorageType::Table {
            for arch in self.archetypes.iter().filter(|arch| arch.contains(id)) {
                let columns = arch.components().ids().to_vec();
                let mut table = Table::new(columns.iter().map(|&cid| {
                    let info = self.components.info(cid).unwrap();
                    if cid == id {
                        (cid, descriptor.layout, descriptor.drop)
                    } else {
                        (cid, info.layout(), info.drop_fn())
                    }
                }));
                table.reserve(arch.len());
                let source = arch.table().column(id).unwrap();
                for (row, &entity) in arch.table().entities().iter().enumerate() {
                    // SAFETY: row belongs to this source table/column.
                    let output = convert(entity, unsafe { source.get_ptr(row) }, &mut migrate)?;
                    table.allocate(entity);
                    outputs.push((entity, output, source.added_tick(row)));
                    values += 1;
                }
                tables.push(PreparedTable {
                    id: arch.id(),
                    columns,
                    table,
                });
            }
        } else if let Some(source) = self.sparse_sets.get(id) {
            let replacement = ComponentSparseSet::new(descriptor.layout, descriptor.drop);
            for &entity in source.entities() {
                // SAFETY: this entity is present in the source set.
                let output = convert(
                    entity,
                    unsafe { source.get_ptr(entity) }.unwrap(),
                    &mut migrate,
                )?;
                outputs.push((entity, output, source.added_tick(entity).unwrap()));
                values += 1;
            }
            sparse = Some(replacement);
        }

        // Allocate retirement storage before taking ownership out of the old
        // tables. No user code or allocation is needed during table commit.
        // Individually owned slots ensure that a panicking old destructor does
        // not prevent Rust's Vec cleanup from dropping the other old values.
        let mut retired: Vec<_> = (0..values)
            .map(|_| {
                DynamicValueSlot::new(DynamicComponentLayout {
                    layout: old_layout,
                    drop: old_drop,
                    clone: None,
                    hash: None,
                })
            })
            .collect();
        let mut pending = outputs.iter_mut();
        for prepared in &mut tables {
            for _ in 0..prepared.table.len() {
                let (_, output, added) = pending.next().unwrap();
                // SAFETY: initialized exact-ABI slot, reserved destination.
                unsafe {
                    prepared.table.column_for_fill(id).push_with_ticks(
                        output.ptr.as_ptr(),
                        *added,
                        now,
                    )
                };
                output.initialized = false;
            }
        }
        if let Some(replacement) = &mut sparse {
            for (entity, output, added) in &mut outputs {
                // SAFETY: source entities are unique and each slot transfers
                // one initialized replacement value with the new ABI.
                unsafe { replacement.insert_with_ticks(*entity, output.ptr.as_ptr(), *added, now) };
                output.initialized = false;
            }
        }
        let mut retired_index = 0;
        let table_count = tables.len();
        for prepared in &mut tables {
            let source = self.archetypes.get_mut(prepared.id).unwrap().table_mut();
            for row in 0..source.len() {
                for &cid in &prepared.columns {
                    let column = source.column(cid).unwrap();
                    // SAFETY: source row is live. Target values move to their
                    // old-ABI retirement store; other values relocate to the
                    // reserved replacement columns with their exact ticks.
                    unsafe {
                        let ptr = column.get_ptr(row);
                        if cid == id {
                            let slot = &mut retired[retired_index];
                            core::ptr::copy_nonoverlapping(
                                ptr,
                                slot.ptr.as_ptr(),
                                old_layout.size(),
                            );
                            slot.initialized = true;
                            retired_index += 1;
                        } else {
                            prepared.table.column_for_fill(cid).push_with_ticks(
                                ptr,
                                column.added_tick(row),
                                column.changed_tick(row),
                            );
                        }
                    }
                }
            }
            while !source.is_empty() {
                // SAFETY: every source column was moved exactly once above;
                // removing backwards keeps all original row indices stable.
                unsafe { source.swap_remove_row(source.len() - 1, &prepared.columns) };
            }
            // The old table is empty. Entity order and archetype id are
            // unchanged, so entity locations/groups/shared bindings stay valid.
            core::mem::swap(source, &mut prepared.table);
        }
        if let Some(replacement) = sparse {
            let source = self.sparse_sets.get_mut(id).unwrap();
            // Drain backwards so all indices/ticks are repaired without any
            // user destructor. Individually retired slots own the old values.
            for slot in &mut retired {
                let entity = *source.entities().last().unwrap();
                // SAFETY: exact old ABI and independent uninitialized slot.
                slot.initialized = unsafe { source.remove_and_copy_out(entity, slot.ptr.as_ptr()) };
            }
            *source = replacement;
        }
        self.components.replace_dynamic_layout(
            id,
            descriptor.layout,
            descriptor.drop,
            descriptor.clone,
            descriptor.hash,
            next_schema_generation,
        );
        self.bump_archetype_generation();
        // Only after new metadata/storage are published may old user drop glue
        // execute. These detached stores cannot leave dangling world rows if
        // their destructors unwind.
        drop(retired);
        Ok(DynamicReloadReport {
            values,
            tables: table_count,
        })
    }
}

#[cfg(test)]
#[path = "dynamic_reload_tests.rs"]
mod tests;
