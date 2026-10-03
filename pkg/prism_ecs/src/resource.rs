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
use core::any::{Any, TypeId, type_name};

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

/// Runtime metadata for one registered resource type.
struct ResourceInfo {
    name: &'static str,
    type_id: TypeId,
    /// The type-erased value, or `None` if the type is registered (has an id)
    /// but no value is currently inserted.
    value: Option<Box<dyn Any + Send + Sync>>,
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
