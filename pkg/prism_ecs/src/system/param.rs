//! [`SystemParam`]: the trait that lets a plain function become a system.
//!
//! A system's arguments are its *parameters*. Each parameter type implements
//! [`SystemParam`], which teaches the scheduler three things:
//!
//! 1. **what access it needs** — replayed into one [`Access`] during
//!    [`update_access`](SystemParam::update_access) (so the conflict graph is
//!    exact and intra-system aliasing is rejected deterministically),
//! 2. **how to build the live value** for one run from an
//!    [`UnsafeWorldCell`] ([`get_param`](SystemParam::get_param)),
//! 3. **how to flush any deferred work** afterwards
//!    ([`apply`](SystemParam::apply)), e.g. draining a [`Commands`] buffer.
//!
//! # Soundness model
//!
//! Every param reaches the world only through an [`UnsafeWorldCell`]: it forms
//! a shared `&World` (via [`UnsafeWorldCell::world`]) and then reaches *through*
//! that borrow via interior-mutability pointers ([`Resources::get_ptr`], the
//! query driver's raw iterator). No ordinary param forms a `&mut World`.
//! Non-aliasing between the `&mut` views handed out by different params of one
//! system is guaranteed by the per-system [`Access`] accumulated here (the
//! panicking [`add_resource_read`](Access::add_resource_read) /
//! [`add_resource_write`](Access::add_resource_write) reject genuine overlaps);
//! across systems it is guaranteed by the scheduler's conflict analysis. This
//! is the same discipline the query layer already uses.
//!
//! [`Resources::get_ptr`]: crate::resource::Resources::get_ptr
//! [`Commands`]: crate::command::Commands

use crate::command::{CommandQueue, Commands};
use crate::query::Access;
use crate::resource::{Resource, ResourceId};
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::World;

use core::ops::{Deref, DerefMut};

/// A type usable as a parameter of a system function.
///
/// # Safety
/// This trait is `unsafe` to implement because
/// [`get_param`](SystemParam::get_param) hands out references synthesised from a
/// raw world handle. An implementation must:
///
/// * declare into the [`Access`] passed to
///   [`update_access`](SystemParam::update_access) *every* component and
///   resource it will access (so the conflict analysis that upholds
///   non-aliasing is complete), and
/// * only materialise references consistent with that declared access.
///
/// Implementing it incorrectly can create aliasing `&mut` references and is
/// undefined behaviour.
pub unsafe trait SystemParam {
    /// Per-system cached state (query state, resolved resource id, command
    /// buffer, …). Built once in [`init_state`](SystemParam::init_state) and
    /// reused every run.
    type State: Send + Sync + 'static;

    /// The live value handed to the system body for a single run. The `'w`
    /// lifetime borrows the world; `'s` borrows the cached `State`.
    type Item<'w, 's>;

    /// Resolve cached state against `world` (e.g. register a resource id or
    /// build a [`QueryState`](crate::query::QueryState)).
    fn init_state(world: &mut World) -> Self::State;

    /// Replay this param's resolved access into `access`.
    ///
    /// Called once after [`init_state`](SystemParam::init_state) with the single
    /// per-system access set. Using the panicking adders makes any genuine
    /// `&mut`/`&mut` or `&mut`/`&` overlap with an *earlier* param in the same
    /// system a deterministic panic before the system ever runs.
    fn update_access(state: &Self::State, access: &mut Access);

    /// Build the live value for one run.
    ///
    /// # Safety
    /// `world` must be a live [`UnsafeWorldCell`] valid for `'w`, and the caller
    /// must guarantee that the access this param declared in
    /// [`update_access`](SystemParam::update_access) does not alias any other
    /// live borrow (upheld by the per-system access set and the scheduler).
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        world: UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's>;

    /// Flush any deferred work produced during the run (default: nothing).
    /// Called with exclusive `&mut World` access after the system body returns.
    #[inline]
    fn apply(_state: &mut Self::State, _world: &mut World) {}
}

/// Convenience alias for the item a [`SystemParam`] yields.
pub type SystemParamItem<'w, 's, P> = <P as SystemParam>::Item<'w, 's>;

// ---------------------------------------------------------------------------
// Res / ResMut
// ---------------------------------------------------------------------------

/// Shared access to the world-global resource `T`.
pub struct Res<'w, T: Resource> {
    value: &'w T,
}

impl<T: Resource> Deref for Res<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'w, T: Resource> Res<'w, T> {
    /// Reborrow the underlying shared reference.
    #[inline]
    pub fn into_inner(self) -> &'w T {
        self.value
    }
}

// SAFETY: `update_access` declares a resource *read* of the id minted in
// `init_state`; `get_param` only ever forms `&T` from the resource's heap box
// via `get_ptr`, so no `&mut` aliasing can arise.
unsafe impl<T: Resource> SystemParam for Res<'_, T> {
    type State = ResourceId;
    type Item<'w, 's> = Res<'w, T>;

    #[inline]
    fn init_state(world: &mut World) -> ResourceId {
        world.resources_mut().register::<T>()
    }

    #[inline]
    fn update_access(state: &ResourceId, access: &mut Access) {
        access.add_resource_read(*state);
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut ResourceId,
        world: UnsafeWorldCell<'w>,
    ) -> Res<'w, T> {
        // SAFETY: the caller guarantees nothing aliases this read; we form only
        // a shared `&World` from the cell.
        let world: &'w World = unsafe { world.world() };
        // SAFETY: `state` is the id minted for `T` in `init_state`; the declared
        // read guarantees no `ResMut<T>` aliases it.
        let ptr = unsafe { world.resources().get_ptr::<T>(*state) }
            .expect("Res<T>: resource not present in world");
        // SAFETY: `ptr` points at the live `T` inside the resource store, which
        // outlives `'w`; our declared read means no `&mut T` is live.
        let value = unsafe { &*ptr };
        Res { value }
    }
}

/// Exclusive access to the world-global resource `T`.
pub struct ResMut<'w, T: Resource> {
    value: &'w mut T,
}

impl<T: Resource> Deref for ResMut<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Resource> DerefMut for ResMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<'w, T: Resource> ResMut<'w, T> {
    /// Reborrow the underlying exclusive reference.
    #[inline]
    pub fn into_inner(self) -> &'w mut T {
        self.value
    }
}

// SAFETY: `update_access` declares an exclusive resource *write* of the id; the
// per-system access set (and cross-system conflict analysis) guarantee no other
// borrow of this resource is live while the `&mut T` exists.
unsafe impl<T: Resource> SystemParam for ResMut<'_, T> {
    type State = ResourceId;
    type Item<'w, 's> = ResMut<'w, T>;

    #[inline]
    fn init_state(world: &mut World) -> ResourceId {
        world.resources_mut().register::<T>()
    }

    #[inline]
    fn update_access(state: &ResourceId, access: &mut Access) {
        access.add_resource_write(*state);
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut ResourceId,
        world: UnsafeWorldCell<'w>,
    ) -> ResMut<'w, T> {
        // SAFETY: the caller guarantees nothing aliases this exclusive write; we
        // form only a shared `&World` from the cell and then reach through the
        // interior-mutability `get_ptr` to the resource's own heap box.
        let world: &'w World = unsafe { world.world() };
        // SAFETY: `state` is the id minted for `T`; our declared exclusive write
        // means no other borrow of this resource is live.
        let ptr = unsafe { world.resources().get_ptr::<T>(*state) }
            .expect("ResMut<T>: resource not present in world");
        // SAFETY: `ptr` is the sole live route to this resource's `T` (exclusive
        // write declared), so promoting it to `&mut T` for `'w` is sound.
        let value = unsafe { &mut *ptr };
        ResMut { value }
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

// SAFETY: `Commands` touches no components or resources directly (it only
// reserves entity ids, which is lock-free and conflict-free), so it declares no
// access; its deferred `apply` runs with exclusive `&mut World`.
unsafe impl SystemParam for Commands<'_, '_> {
    type State = CommandQueue;
    type Item<'w, 's> = Commands<'w, 's>;

    #[inline]
    fn init_state(_world: &mut World) -> CommandQueue {
        CommandQueue::new()
    }

    #[inline]
    fn update_access(_state: &CommandQueue, _access: &mut Access) {}

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut CommandQueue,
        world: UnsafeWorldCell<'w>,
    ) -> Commands<'w, 's> {
        // SAFETY: the caller guarantees `world` is live for `'w`; we only form a
        // shared `&World` and borrow its entity allocator for id reservation.
        let world: &'w World = unsafe { world.world() };
        state.commands(world.entities())
    }

    #[inline]
    fn apply(state: &mut CommandQueue, world: &mut World) {
        state.apply(world);
    }
}

// ---------------------------------------------------------------------------
// Local
// ---------------------------------------------------------------------------

/// Per-system state that persists across runs of the *same* system instance.
///
/// Unlike a [`Resource`], a `Local<T>` is private to one system: two systems
/// taking `Local<Counter>` each get their own independent `Counter`. The value
/// is created once (via [`Default`]) when the system is
/// [`initialize`](crate::system::System::initialize)d and then reused — and
/// mutated — on every run. It declares no world access, so it never conflicts
/// with any other param or system.
pub struct Local<'s, T: Default + Send + Sync + 'static> {
    value: &'s mut T,
}

impl<T: Default + Send + Sync + 'static> Deref for Local<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Default + Send + Sync + 'static> DerefMut for Local<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<'s, T: Default + Send + Sync + 'static> Local<'s, T> {
    /// Reborrow the underlying exclusive reference to the persistent state.
    #[inline]
    pub fn into_inner(self) -> &'s mut T {
        self.value
    }
}

// SAFETY: `Local` touches neither components nor resources — its state lives in
// the per-system `State` slot, not in the world — so it declares no access and
// forms no world reference at all. `get_param` only borrows the system's own
// state, which the caller hands out as a unique `&mut` per run.
unsafe impl<T: Default + Send + Sync + 'static> SystemParam for Local<'_, T> {
    type State = T;
    type Item<'w, 's> = Local<'s, T>;

    #[inline]
    fn init_state(_world: &mut World) -> T {
        T::default()
    }

    #[inline]
    fn update_access(_state: &T, _access: &mut Access) {}

    #[inline]
    unsafe fn get_param<'w, 's>(state: &'s mut T, _world: UnsafeWorldCell<'w>) -> Local<'s, T> {
        Local { value: state }
    }
}

// ---------------------------------------------------------------------------
// Tuples
// ---------------------------------------------------------------------------

macro_rules! impl_system_param_tuple {
    ($($param:ident),*) => {
        // SAFETY: each element declares its own access into the shared `Access`
        // and builds its own disjoint item; borrowing disjoint fields of the
        // state tuple hands each element a non-aliasing `&mut` to its own state.
        // Any genuine overlap between two elements is rejected by the panicking
        // adders in `update_access`.
        #[allow(non_snake_case, unused_variables, clippy::unused_unit)]
        unsafe impl<$($param: SystemParam),*> SystemParam for ($($param,)*) {
            type State = ($($param::State,)*);
            type Item<'w, 's> = ($($param::Item<'w, 's>,)*);

            #[inline]
            fn init_state(world: &mut World) -> Self::State {
                ($($param::init_state(world),)*)
            }

            #[inline]
            fn update_access(state: &Self::State, access: &mut Access) {
                let ($($param,)*) = state;
                $($param::update_access($param, access);)*
            }

            #[inline]
            unsafe fn get_param<'w, 's>(
                state: &'s mut Self::State,
                world: UnsafeWorldCell<'w>,
            ) -> Self::Item<'w, 's> {
                let ($($param,)*) = state;
                // SAFETY: forwarded from the caller's guarantee; each element's
                // declared access is disjoint by the per-system conflict checks.
                ($(unsafe { $param::get_param($param, world) },)*)
            }

            #[inline]
            fn apply(state: &mut Self::State, world: &mut World) {
                let ($($param,)*) = state;
                $($param::apply($param, world);)*
            }
        }
    };
}

impl_system_param_tuple!();
impl_system_param_tuple!(P0);
impl_system_param_tuple!(P0, P1);
impl_system_param_tuple!(P0, P1, P2);
impl_system_param_tuple!(P0, P1, P2, P3);
impl_system_param_tuple!(P0, P1, P2, P3, P4);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
impl_system_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
