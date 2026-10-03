//! [`UnsafeWorldCell`]: the single `unsafe` chokepoint through which system
//! params fetch disjoint borrows out of one `&mut World`.
//!
//! The scheduler owns exactly one `&mut World` for the duration of a run. To
//! let several [`SystemParam`](crate::system::SystemParam)s each form their own
//! `&T` / `&mut T` views of *disjoint* parts of that world — without ever
//! materialising two aliasing `&mut World` — it hands each param a copy of this
//! cell. A cell is just a `*mut World` tagged with the lifetime `'w` it is
//! valid for; it is [`Copy`] so it can be threaded through tuple params.
//!
//! # Soundness model
//!
//! The cell itself performs no checks: it is a capability token. Soundness is
//! upheld by two layers that sit *above* it:
//!
//! * **Intra-system:** the per-system [`Access`](crate::query::Access) set,
//!   accumulated by [`SystemParam::update_access`](crate::system::SystemParam::update_access)
//!   with the panicking adders, rejects a single system whose own params would
//!   alias mutably.
//! * **Inter-system:** the scheduler's conflict graph never runs two systems
//!   with incompatible access concurrently, and runs exclusive systems alone.
//!
//! Given those guarantees, forming the shared/exclusive borrows below is sound.
//! Callers of the `unsafe` methods must respect the documented contracts.

use core::marker::PhantomData;

use crate::change::Tick;
use crate::world::World;

/// A `Copy` capability token wrapping a `*mut World` valid for `'w`.
///
/// Params reach the world *only* through this type. Most params form a shared
/// [`world`](UnsafeWorldCell::world) and then reach through interior-mutability
/// pointers (e.g. [`Resources::get_ptr`](crate::resource::Resources::get_ptr),
/// the query driver's raw iterator); only exclusive systems form a
/// [`world_mut`](UnsafeWorldCell::world_mut).
#[derive(Clone, Copy)]
pub struct UnsafeWorldCell<'w> {
    ptr: *mut World,
    /// Start of the running system's change-detection window (the tick it last
    /// completed), exclusive. Threaded into every [`Query`](crate::system::Query)
    /// so `Added`/`Changed` observe exactly the writes since this system last
    /// ran.
    last_run: Tick,
    /// The current world tick, i.e. the upper bound of the change-detection
    /// window.
    this_run: Tick,
    _marker: PhantomData<&'w mut World>,
}

impl<'w> UnsafeWorldCell<'w> {
    /// Wrap an exclusive `&mut World`. This is the entry point used by the
    /// sequential schedule and the parallel executor, both of which own the one
    /// live `&mut World`.
    #[inline]
    pub(crate) fn new_mutable(world: &'w mut World) -> Self {
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        Self {
            ptr: world as *mut World,
            last_run,
            this_run,
            _marker: PhantomData,
        }
    }


    /// Start of the running system's change-detection window (exclusive).
    #[inline]
    pub fn last_run(self) -> Tick {
        self.last_run
    }

    /// The current world tick (upper bound of the change-detection window).
    #[inline]
    pub fn this_run(self) -> Tick {
        self.this_run
    }

    /// The raw world pointer, for params (such as the query driver) that need to
    /// pass a `*mut World` down to a raw iterator.
    #[inline]
    pub fn as_ptr(self) -> *mut World {
        self.ptr
    }

    /// Form a shared `&World` for `'w`.
    ///
    /// # Safety
    /// The caller must guarantee that for the whole of `'w` no `&mut` borrow of
    /// any part of the world that this reference transitively exposes is live
    /// elsewhere. In the system layer this is upheld by the per-system
    /// [`Access`](crate::query::Access) set and the scheduler's conflict graph.
    #[inline]
    pub unsafe fn world(self) -> &'w World {
        // SAFETY: the caller guarantees the pointer is live for `'w` and that no
        // conflicting `&mut` view is active, so a shared `&World` is sound.
        unsafe { &*self.ptr }
    }

    /// Form an exclusive `&mut World` for `'w`.
    ///
    /// # Safety
    /// The caller must guarantee that this is the *only* live borrow of the
    /// world for the whole of `'w`. Only exclusive systems (which the scheduler
    /// runs with no other system in flight) may call this, and only on a cell
    /// created via [`new_mutable`](UnsafeWorldCell::new_mutable).
    #[inline]
    pub unsafe fn world_mut(self) -> &'w mut World {
        // SAFETY: the caller guarantees this is the unique live borrow of the
        // world for `'w` and that the cell wraps a genuine `&mut World`, so
        // promoting the pointer to `&mut World` is sound.
        unsafe { &mut *self.ptr }
    }
}
