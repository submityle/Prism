//! Reflection bridge (design §16.5 / §24.3): dynamic component access by
//! `prism_reflect` type id.
//!
//! An editor inspector, a scene loader, a save migrator and a replication layer
//! all need to touch components **without naming the Rust type at the call
//! site**. [`World::register_reflect_component`] records, per registered type, a
//! handful of monomorphised function pointers plus the type's
//! [`StableTypeId`] — the path-derived id that survives rebuilds and is safe to
//! persist in save files and network streams (design §24.3). After that, the
//! caller works entirely through `&dyn Reflect`:
//!
//! ```ignore
//! let id = world.register_reflect_component::<Health>();
//! let value: &dyn Reflect = world.get_reflect(entity, id)?;   // O(1)
//! world.insert_reflect(entity, patched.as_reflect())?;        // overwrite
//! world.remove_reflect(entity, id)?;
//! ```
//!
//! # Why function pointers, not a dynamic store
//!
//! Components live in typed, archetype-local columns; a `Box<dyn Reflect>` has
//! no column to live in. Registration therefore records the *operations*
//! ([`ReflectComponentOps`]) rather than the values, and each operation
//! downcasts back to `T` at the boundary. Lookups go through a
//! `StableTypeId → ops` map, so the hot path is one hash probe and one indirect
//! call — no `TypeId` set scan, no per-value boxing of the live component.
//!
//! # Change detection
//!
//! `get_reflect` is a plain shared read. `get_reflect_mut` goes through the
//! ordinary change-detection handle ([`Mut`](crate::change::Mut)) and therefore
//! **marks the component changed**, exactly like a direct `world.get_mut::<T>`:
//! a modification made through reflection must be visible to `Changed<T>`,
//! dirty chunks and the replication layer.

use crate::collections::HashMap;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;
use prism_reflect::{FromReflect, Reflect, StableTypeId};

/// The stable, path-derived type id the bridge keys on (re-exported so callers
/// need not depend on `prism_reflect` directly).
pub use prism_reflect::StableTypeId as ReflectTypeId;

/// The type-erased operations recorded for one reflected component type.
///
/// Each entry is a monomorphised `fn` pointer, so the bridge costs one indirect
/// call per operation and never boxes the live component value.
#[derive(Clone, Copy)]
struct ReflectComponentOps {
    /// The component's dense id in this world's registry.
    component: ComponentId,
    /// `Some(&T)` for a live entity that carries `T`.
    get: fn(&World, Entity) -> Option<&dyn Reflect>,
    /// `Some(&mut T)` (through the change-detection handle, so it marks changed).
    get_mut: fn(&mut World, Entity) -> Option<&mut dyn Reflect>,
    /// Overwrite (or insert) `T` from a reflected value of this type. Returns
    /// `false` when the value's concrete type does not match `T`.
    apply: fn(&mut World, Entity, &dyn Reflect) -> bool,
    /// Remove `T` from `entity`; `false` when the entity did not carry it.
    remove: fn(&mut World, Entity) -> bool,
}

impl ReflectComponentOps {
    /// Build the operations for `T` by instantiating one monomorphised function
    /// per entry point.
    fn of<T: Component + Reflect + FromReflect + Clone>(component: ComponentId) -> Self {
        fn get<T: Component + Reflect>(world: &World, entity: Entity) -> Option<&dyn Reflect> {
            world.get::<T>(entity).map(|value| value as &dyn Reflect)
        }
        fn get_mut<T: Component + Reflect>(
            world: &mut World,
            entity: Entity,
        ) -> Option<&mut dyn Reflect> {
            // `Mut::into_inner` hands back the `'w` borrow (recording the write,
            // as any `Mut` access does), which is what this fn pointer promises.
            world.get_mut::<T>(entity).map(|value| value.into_inner() as &mut dyn Reflect)
        }
        fn apply<T: Component + Reflect + FromReflect + Clone>(
            world: &mut World,
            entity: Entity,
            value: &dyn Reflect,
        ) -> bool {
            // Fast path: a concrete value of exactly `T`.
            if let Some(typed) = value.downcast_ref::<T>() {
                world.insert(entity, typed.clone());
                return true;
            }
            // Slow path: a `Dynamic*` tree — e.g. one decoded from a save file
            // or produced by `reflect_clone` — reconstructs through
            // `FromReflect`, so both forms are accepted at the bridge.
            match T::from_reflect(value) {
                Some(typed) => {
                    world.insert(entity, typed);
                    true
                }
                None => false,
            }
        }
        fn remove<T: Component>(world: &mut World, entity: Entity) -> bool {
            world.remove::<T>(entity)
        }

        Self {
            component,
            get: get::<T>,
            get_mut: get_mut::<T>,
            apply: apply::<T>,
            remove: remove::<T>,
        }
    }
}

/// The world's reflected-component registry.
#[derive(Default)]
pub(crate) struct ReflectRegistry {
    /// `StableTypeId` → operations, the hot lookup for dynamic access.
    ops: HashMap<StableTypeId, ReflectComponentOps>,
    /// `ComponentId` → `StableTypeId`, so a layer that discovered a component
    /// through world introspection (columns, dirty chunks, snapshots) can name it
    /// in the stable, persistable id space.
    by_component: HashMap<ComponentId, StableTypeId>,
}

impl ReflectRegistry {
    /// The stable id registered for component `component`, if any.
    #[inline]
    pub(crate) fn stable_of(&self, component: ComponentId) -> Option<StableTypeId> {
        self.by_component.get(&component).copied()
    }
}

impl World {
    /// Register `T` for dynamic (reflected) access and return its
    /// [`StableTypeId`] (design §16.5 / §24.3). Idempotent.
    ///
    /// `T` must be [`Reflect`] (so its fields can be walked generically),
    /// [`FromReflect`] (so a `Dynamic*` tree — e.g. one decoded from a save file
    /// — can be reconstructed) and [`Clone`] (so a concrete value handed in
    /// dynamically can be materialised into the typed column). `prism_reflect`
    /// is the type basis; the ECS stores only the monomorphised operations, never
    /// a boxed value.
    pub fn register_reflect_component<T: Component + Reflect + FromReflect + Clone>(
        &mut self,
    ) -> StableTypeId {
        let component = self.components_mut().register::<T>();
        let stable = StableTypeId::of_type::<T>();
        self.reflect
            .ops
            .insert(stable, ReflectComponentOps::of::<T>(component));
        self.reflect.by_component.insert(component, stable);
        stable
    }

    /// The [`StableTypeId`] registered for `component`, if it was registered
    /// through [`register_reflect_component`](World::register_reflect_component).
    #[inline]
    pub fn reflect_stable_id(&self, component: ComponentId) -> Option<StableTypeId> {
        self.reflect.stable_of(component)
    }

    /// Number of component types registered for reflection.
    #[inline]
    pub fn reflected_component_count(&self) -> usize {
        self.reflect.ops.len()
    }

    /// Read `entity`'s component of reflected type `type_id` (design §24.3).
    ///
    /// `O(1)`: one `StableTypeId` hash probe plus the ordinary component lookup.
    /// Returns `None` for an unregistered type, a dead entity, or an entity that
    /// does not carry the component.
    #[inline]
    pub fn get_reflect(&self, entity: Entity, type_id: StableTypeId) -> Option<&dyn Reflect> {
        let ops = self.reflect.ops.get(&type_id)?;
        (ops.get)(self, entity)
    }

    /// Mutably access `entity`'s component of reflected type `type_id`.
    ///
    /// Goes through the change-detection handle, so — exactly like
    /// `World::get_mut::<T>` — the component is marked **changed** (visible to
    /// `Changed<T>`, dirty chunks and replication), even if the caller only
    /// reads through the returned reference.
    #[inline]
    pub fn get_reflect_mut(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
    ) -> Option<&mut dyn Reflect> {
        let ops = *self.reflect.ops.get(&type_id)?;
        (ops.get_mut)(self, entity)
    }

    /// Overwrite (or insert) `entity`'s component from a reflected value.
    ///
    /// The value's concrete type selects the component; `false` means the value
    /// is not of a type registered for reflection (or its `StableTypeId` was
    /// registered by a different world). A successful overwrite marks the
    /// component changed.
    pub fn insert_reflect(&mut self, entity: Entity, value: &dyn Reflect) -> bool {
        // `register_reflect_component::<T>` keyed the ops by
        // `StableTypeId::of_type::<T>()`, i.e. `of_path(type_name::<T>())`; a
        // concrete reflected value reports that same path, so this is one hash
        // probe — no scan over the registry.
        let type_id = StableTypeId::of_path(value.type_name());
        let Some(ops) = self.reflect.ops.get(&type_id).copied() else {
            return false;
        };
        (ops.apply)(self, entity, value)
    }

    /// The dense [`ComponentId`] registered for reflected type `type_id`, if any.
    ///
    /// Lets an introspecting layer (editor, snapshot diff, replication) go from
    /// the stable, persistable id space back to the world-local column id.
    #[inline]
    pub fn reflect_component_id(&self, type_id: StableTypeId) -> Option<ComponentId> {
        self.reflect.ops.get(&type_id).map(|ops| ops.component)
    }

    /// Diff a live component against an authoritative reflected value (design
    /// §24.3 / §24.4): the result names only the fields that differ, so a
    /// replication layer ships `patch` instead of the whole component.
    ///
    /// Returns `None` for an unregistered type or an entity without the
    /// component.
    pub fn diff_reflect(
        &self,
        entity: Entity,
        type_id: StableTypeId,
        authoritative: &dyn Reflect,
    ) -> Option<prism_reflect::Patch> {
        let live = self.get_reflect(entity, type_id)?;
        Some(prism_reflect::diff(live, authoritative))
    }

    /// Apply a field-level [`Patch`](prism_reflect::Patch) to a live component
    /// (design §24.3 / §24.5).
    ///
    /// Only the fields the patch names are written. `Patch::Unchanged` is a
    /// true no-op: it neither writes nor marks the component changed, so a
    /// replication receiver that is already up to date does not spuriously
    /// re-trigger `Changed<T>` / dirty chunks. Any real patch goes through the
    /// change-detection handle and therefore marks the component changed.
    /// Returns `false` for an unregistered type, a missing component, or a patch
    /// whose kind does not match the live value.
    pub fn apply_reflect_patch(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
        patch: &prism_reflect::Patch,
    ) -> bool {
        if matches!(patch, prism_reflect::Patch::Unchanged) {
            return self.get_reflect(entity, type_id).is_some();
        }
        let Some(value) = self.get_reflect_mut(entity, type_id) else {
            return false;
        };
        patch.apply(value).is_ok()
    }

    /// Remove `entity`'s component of reflected type `type_id`.
    ///
    /// Returns `false` for an unregistered type or an entity that does not carry
    /// the component.
    pub fn remove_reflect(&mut self, entity: Entity, type_id: StableTypeId) -> bool {
        let Some(ops) = self.reflect.ops.get(&type_id).copied() else {
            return false;
        };
        (ops.remove)(self, entity)
    }
}
