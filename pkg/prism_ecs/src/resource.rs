//! Resources: globally-unique, type-keyed singletons owned by a
//! [`World`](crate::world::World) (design §8.1, §18 — `Res`/`ResMut`).
//!
//! Where a [`Component`](crate::component::Component) is stored *per entity*, a
//! [`Resource`] is stored *once per world*: a renderer handle, the frame clock,
//! an asset server, the active input map, and so on. Each distinct resource
//! type is assigned a dense [`ResourceId`] the first time it is registered, and
//! the value is held type-erased in a slot indexed by that id.
//!
//! # Interior-mutability access model
//!
//! Like the columnar component [`Column`](crate::storage::Column), the resource
//! store hands out a raw `*mut T` from a shared `&self`
//! ([`Resources::get_ptr`]). This is the same discipline the rest of the kernel
//! uses: every reader forms only a shared `&World`, and all *mutation* flows
//! through raw pointers whose non-aliasing is guaranteed by the scheduler's
//! read/write access analysis (design §8.2). It is what lets a system take
//! `Res<A>` and `ResMut<B>` at the same time without ever materialising a
//! `&mut World`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::{type_name, Any, TypeId};

use crate::collections::HashMap;

/// A globally-unique, type-keyed value stored once per
/// [`World`](crate::world::World).
///
/// This is a marker trait implemented via `#[derive(Resource)]` (companion
/// `prism_ecs_macros` crate) or by hand for a plain data type. The
/// `Send + Sync + 'static` bounds make resources safe to store type-erased and
/// to reference from systems running across threads.
pub trait Resource: Send + Sync + 'static {}

/// A dense, stable identifier for a registered resource type within one
/// [`World`](crate::world::World).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ResourceId(u32);

impl ResourceId {
    /// Construct a [`ResourceId`] from its raw index. Primarily internal / for
    /// tests; ids are normally minted by [`Resources::register`].
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

/// Type-erased clone glue for a snapshot-registered [`Resource`] (design
/// §14/§16.5): clones the concrete value behind a `&dyn Any` into a fresh box.
pub type ResourceCloneFn = fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>;

/// Type-erased deterministic value-hash glue for a snapshot-registered
/// [`Resource`] (design §14 逐帧状态哈希): folds the concrete value into a
/// `&mut dyn Hasher`.
pub type ResourceHashFn = fn(&(dyn Any + Send + Sync), &mut dyn core::hash::Hasher);

/// Build type-erased clone glue for a resource type `T`.
fn resource_clone_fn_of<T: Resource + Clone>() -> ResourceCloneFn {
    fn clone_boxed<T: Resource + Clone>(
        value: &(dyn Any + Send + Sync),
    ) -> Box<dyn Any + Send + Sync> {
        let value = value
            .downcast_ref::<T>()
            .expect("resource clone glue invoked on the wrong type");
        Box::new(value.clone())
    }
    clone_boxed::<T>
}

/// Build type-erased deterministic hash glue for a resource type `T`.
fn resource_hash_fn_of<T: Resource + core::hash::Hash>() -> ResourceHashFn {
    fn hash_boxed<T: Resource + core::hash::Hash>(
        value: &(dyn Any + Send + Sync),
        hasher: &mut dyn core::hash::Hasher,
    ) {
        let value = value
            .downcast_ref::<T>()
            .expect("resource hash glue invoked on the wrong type");
        core::hash::Hash::hash(value, &mut ResourceHasherShim(hasher));
    }
    hash_boxed::<T>
}

/// Adapts a `&mut dyn Hasher` so a generic `Hash::hash<H: Hasher>` can drive it;
/// forwards every write through the trait object.
struct ResourceHasherShim<'a>(&'a mut dyn core::hash::Hasher);

impl core::hash::Hasher for ResourceHasherShim<'_> {
    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

/// Runtime metadata for one registered resource type.
struct ResourceInfo {
    name: &'static str,
    type_id: TypeId,
    /// The type-erased value, or `None` if the type is registered (has an id)
    /// but no value is currently inserted.
    value: Option<Box<dyn Any + Send + Sync>>,
    /// Opt-in snapshot clone glue (design §14/§16.5). `Some` iff the resource
    /// was registered via [`Resources::register_snapshot`]; only such resources
    /// participate in world snapshot/restore.
    clone: Option<ResourceCloneFn>,
    /// Opt-in deterministic value-hash glue (design §14). `Some` iff registered
    /// via [`Resources::register_snapshot_hashable`].
    hash: Option<ResourceHashFn>,
}

/// The per-world registry and store of resources.
///
/// Registration (assigning a [`ResourceId`]) is separate from insertion
/// (storing a value): a system can declare it reads `Res<T>` — minting the id —
/// before any value exists, and the executor can detect the missing value at
/// run time rather than at wiring time.
#[derive(Default)]
pub struct Resources {
    infos: Vec<ResourceInfo>,
    by_type: HashMap<TypeId, ResourceId>,
}

impl Resources {
    /// Create an empty resource store.
    #[inline]
    pub fn new() -> Self {
        Self {
            infos: Vec::new(),
            by_type: HashMap::default(),
        }
    }

    /// Number of distinct resource types that have been registered (whether or
    /// not they currently hold a value).
    #[inline]
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// Whether no resource types have been registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Assign (or look up) the dense [`ResourceId`] for type `T`. Idempotent.
    pub fn register<T: Resource>(&mut self) -> ResourceId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.by_type.get(&type_id) {
            return id;
        }
        let id = ResourceId(self.infos.len() as u32);
        self.infos.push(ResourceInfo {
            name: type_name::<T>(),
            type_id,
            value: None,
            clone: None,
            hash: None,
        });
        self.by_type.insert(type_id, id);
        id
    }

    /// Look up the id previously assigned to `T`, without registering it.
    #[inline]
    pub fn id_of<T: Resource>(&self) -> Option<ResourceId> {
        self.by_type.get(&TypeId::of::<T>()).copied()
    }

    /// Human-readable name for a registered resource id, if any.
    #[inline]
    pub fn name(&self, id: ResourceId) -> Option<&'static str> {
        self.infos.get(id.0 as usize).map(|i| i.name)
    }

    /// Insert (or overwrite) the value of resource `T`, returning the previous
    /// value if one was present.
    pub fn insert<T: Resource>(&mut self, value: T) -> Option<T> {
        let id = self.register::<T>();
        let info = &mut self.infos[id.0 as usize];
        let previous = info.value.replace(Box::new(value));
        previous.map(|boxed| {
            *boxed
                .downcast::<T>()
                .expect("resource slot held the wrong type")
        })
    }

    /// Ensure resource `T` holds a value, constructing it with
    /// [`Default`] if it is currently absent. Idempotent: an existing value is
    /// left untouched.
    pub fn init<T: Resource + Default>(&mut self) {
        if !self.contains::<T>() {
            self.insert(T::default());
        }
    }

    /// Whether resource `T` currently holds a value.
    #[inline]
    pub fn contains<T: Resource>(&self) -> bool {
        self.id_of::<T>()
            .and_then(|id| self.infos.get(id.0 as usize))
            .is_some_and(|i| i.value.is_some())
    }

    /// Borrow resource `T` immutably, or `None` if absent.
    pub fn get<T: Resource>(&self) -> Option<&T> {
        let id = self.id_of::<T>()?;
        let info = self.infos.get(id.0 as usize)?;
        info.value.as_ref()?.downcast_ref::<T>()
    }

    /// Borrow resource `T` mutably, or `None` if absent.
    pub fn get_mut<T: Resource>(&mut self) -> Option<&mut T> {
        let id = self.id_of::<T>()?;
        let info = self.infos.get_mut(id.0 as usize)?;
        info.value.as_mut()?.downcast_mut::<T>()
    }

    /// Remove and return resource `T`, if present. The id stays registered.
    pub fn remove<T: Resource>(&mut self) -> Option<T> {
        let id = self.id_of::<T>()?;
        let info = self.infos.get_mut(id.0 as usize)?;
        let boxed = info.value.take()?;
        Some(
            *boxed
                .downcast::<T>()
                .expect("resource slot held the wrong type"),
        )
    }

    /// Register `T` (idempotently) and attach clone glue so its value is
    /// captured by [`World::snapshot`](crate::world::World::snapshot) and
    /// restored by [`restore`](crate::world::World::restore) (design §14/§16.5).
    ///
    /// Resource capture is strictly **opt-in**: unlike a component (whose
    /// resident-but-unregistered value *blocks* a capture), a resource that is
    /// never registered here is simply left untouched by snapshot/restore. This
    /// is deliberate — most resources are infrastructure singletons (task
    /// pools, device handles, asset servers) that are not meaningfully `Clone`
    /// and must not participate in rollback; only explicitly registered
    /// gameplay resources (score, timers, RNG cursor) join a snapshot.
    pub fn register_snapshot<T: Resource + Clone>(&mut self) -> ResourceId {
        let id = self.register::<T>();
        self.infos[id.0 as usize].clone = Some(resource_clone_fn_of::<T>());
        id
    }

    /// Register `T` (idempotently) with clone glue *and* deterministic value
    /// hash glue, so it contributes its value to a world
    /// [`state_hash`](crate::world::snapshot::WorldSnapshot::state_hash)
    /// (design §14). Returns the resource id.
    pub fn register_snapshot_hashable<T: Resource + Clone + core::hash::Hash>(
        &mut self,
    ) -> ResourceId {
        let id = self.register_snapshot::<T>();
        self.infos[id.0 as usize].hash = Some(resource_hash_fn_of::<T>());
        id
    }

    /// Every snapshot-registered resource id (clone glue present), ascending by
    /// id. Includes ids whose value slot is currently empty: capture skips
    /// those, while restore uses them to clear a stale live value.
    pub(crate) fn snapshot_registered_ids(&self) -> Vec<ResourceId> {
        self.infos
            .iter()
            .enumerate()
            .filter(|(_, i)| i.clone.is_some())
            .map(|(idx, _)| ResourceId(idx as u32))
            .collect()
    }

    /// The [`TypeId`] recorded for resource `id`, if registered.
    #[inline]
    pub(crate) fn type_id_of(&self, id: ResourceId) -> Option<TypeId> {
        self.infos.get(id.0 as usize).map(|i| i.type_id)
    }

    /// The clone glue registered for resource `id`, if any.
    #[inline]
    pub(crate) fn clone_fn_of(&self, id: ResourceId) -> Option<ResourceCloneFn> {
        self.infos.get(id.0 as usize).and_then(|i| i.clone)
    }

    /// Clone the current value of snapshot-registered resource `id` into a fresh
    /// box via its clone glue, or `None` if `id` lacks glue or holds no value.
    pub(crate) fn clone_value_boxed(&self, id: ResourceId) -> Option<Box<dyn Any + Send + Sync>> {
        let info = self.infos.get(id.0 as usize)?;
        let clone = info.clone?;
        let value = info.value.as_ref()?;
        Some(clone(&**value))
    }

    /// Fold the value of snapshot-registered resource `id` into `hasher` via its
    /// hash glue. Returns `true` iff a value was hashed (hash glue present and a
    /// value resident), `false` otherwise.
    pub(crate) fn hash_value_into(
        &self,
        id: ResourceId,
        hasher: &mut dyn core::hash::Hasher,
    ) -> bool {
        let Some(info) = self.infos.get(id.0 as usize) else {
            return false;
        };
        match (info.hash, info.value.as_ref()) {
            (Some(hash), Some(value)) => {
                hash(&**value, hasher);
                true
            }
            _ => false,
        }
    }

    /// Overwrite the value slot of `id` with an already-boxed value (restore
    /// path). Debug-asserts the box's dynamic type matches the registered one.
    pub(crate) fn insert_boxed(&mut self, id: ResourceId, value: Box<dyn Any + Send + Sync>) {
        if let Some(info) = self.infos.get_mut(id.0 as usize) {
            debug_assert_eq!(
                (*value).type_id(),
                info.type_id,
                "insert_boxed dynamic-type mismatch for resource {}",
                info.name
            );
            info.value = Some(value);
        }
    }

    /// Clear the value slot of `id`, keeping its registration. Restore uses this
    /// to drop a resource that held no value at capture time.
    pub(crate) fn clear_value(&mut self, id: ResourceId) {
        if let Some(info) = self.infos.get_mut(id.0 as usize) {
            info.value = None;
        }
    }

    /// Obtain a raw `*mut T` to the stored value of `T` from a shared borrow,
    /// or `None` if the type is absent.
    ///
    /// This is the system-param access primitive: it mirrors
    /// [`Column::get_ptr`](crate::storage::Column::get_ptr) by projecting a raw
    /// pointer out of a shared `&self` so that `Res`/`ResMut` params can form
    /// their references without the store ever handing out a `&mut Resources`.
    ///
    /// # Safety
    /// The caller must honour the kernel's access discipline: a `*mut T`
    /// obtained here may be turned into a `&mut T` only if no other live borrow
    /// aliases the same resource. The scheduler's read/write conflict analysis
    /// (design §8.2) upholds this across systems; within one system the
    /// per-system access set rejects conflicting params.
    pub unsafe fn get_ptr<T: Resource>(&self, id: ResourceId) -> Option<*mut T> {
        let info = self.infos.get(id.0 as usize)?;
        let boxed = info.value.as_ref()?;
        debug_assert_eq!(info.type_id, TypeId::of::<T>(), "resource id/type mismatch");
        let value: &T = boxed.downcast_ref::<T>()?;
        Some((value as *const T).cast_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Score(u32);
    impl Resource for Score {}

    #[derive(Debug, PartialEq)]
    struct Label(alloc::string::String);
    impl Resource for Label {}

    #[test]
    fn register_is_idempotent_and_dense() {
        let mut r = Resources::new();
        let s0 = r.register::<Score>();
        let s1 = r.register::<Score>();
        let l = r.register::<Label>();
        assert_eq!(s0, s1);
        assert_ne!(s0, l);
        assert_eq!(s0.index(), 0);
        assert_eq!(l.index(), 1);
        assert_eq!(r.len(), 2);
        assert_eq!(r.id_of::<Score>(), Some(s0));
    }

    #[test]
    fn insert_get_mutate_remove() {
        let mut r = Resources::new();
        assert!(!r.contains::<Score>());
        assert_eq!(r.insert(Score(10)), None);
        assert!(r.contains::<Score>());
        assert_eq!(r.get::<Score>(), Some(&Score(10)));
        r.get_mut::<Score>().unwrap().0 = 42;
        assert_eq!(r.get::<Score>(), Some(&Score(42)));
        // Overwrite returns the previous value.
        assert_eq!(r.insert(Score(7)), Some(Score(42)));
        assert_eq!(r.remove::<Score>(), Some(Score(7)));
        assert!(!r.contains::<Score>());
        // Id stays registered after removal.
        assert_eq!(r.id_of::<Score>().map(|i| i.index()), Some(0));
    }

    #[test]
    fn drop_runs_on_remove_and_overwrite() {
        let mut r = Resources::new();
        r.insert(Label(alloc::string::String::from("first")));
        // Overwrite must drop "first" without leaking.
        r.insert(Label(alloc::string::String::from("second")));
        assert_eq!(r.get::<Label>().map(|l| l.0.as_str()), Some("second"));
        assert!(r.remove::<Label>().is_some());
    }

    #[test]
    fn get_ptr_reads_and_writes_from_shared_borrow() {
        let mut r = Resources::new();
        let id = r.register::<Score>();
        r.insert(Score(1));
        // SAFETY: single-threaded test; the pointer is used for one exclusive
        // write with no other live borrow of the same resource.
        unsafe {
            let p = r.get_ptr::<Score>(id).unwrap();
            (*p).0 = 99;
        }
        assert_eq!(r.get::<Score>(), Some(&Score(99)));
    }

    #[test]
    fn get_ptr_absent_is_none() {
        let mut r = Resources::new();
        let id = r.register::<Score>();
        // Registered but no value inserted yet.
        // SAFETY: call is sound regardless; it returns None for an empty slot.
        assert!(unsafe { r.get_ptr::<Score>(id) }.is_none());
    }
}
