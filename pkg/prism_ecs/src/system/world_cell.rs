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
    /// Wrap an exclusive `&mut World`, reading the world-global change ticks.
    ///
    /// Used by the parallel executor to build a per-wave base cell (later
    /// retargeted per system via [`with_ticks`](Self::with_ticks)) and by unit
    /// tests; the per-system executors thread explicit windows through
    /// [`new_mutable_with_ticks`](Self::new_mutable_with_ticks) instead.
    #[cfg(any(test, feature = "multi_thread"))]
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

    /// Wrap an exclusive `&mut World` with an explicit change-detection window.
    ///
    /// Used by both executors to thread a *per-system* `last_run` (the tick the
    /// running system last completed) and `this_run` (a fresh world tick for
    /// this run) rather than the world-global ticks `new_mutable` reads. This
    /// is what makes `Added`/`Changed` observe exactly the writes a given system
    /// has not yet seen (design §10).
    #[inline]
    pub(crate) fn new_mutable_with_ticks(
        world: &'w mut World,
        last_run: Tick,
        this_run: Tick,
    ) -> Self {
        Self {
            ptr: world as *mut World,
            last_run,
            this_run,
            _marker: PhantomData,
        }
    }

    /// Return a copy of this cell retargeted to a different change-detection
    /// window, keeping the same world pointer and lifetime.
    ///
    /// The parallel executor builds one base cell per wave and then hands each
    /// concurrent system its own `(last_run, this_run)` view via this method, so
    /// every system still sees a per-system change window even though the bodies
    /// share one `&mut World` borrow.
    #[cfg(feature = "multi_thread")]
    #[inline]
    pub(crate) fn with_ticks(self, last_run: Tick, this_run: Tick) -> Self {
        Self {
            ptr: self.ptr,
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
    /// created via `new_mutable` / `new_mutable_with_ticks`.
    #[inline]
    pub unsafe fn world_mut(self) -> &'w mut World {
        // SAFETY: the caller guarantees this is the unique live borrow of the
        // world for `'w` and that the cell wraps a genuine `&mut World`, so
        // promoting the pointer to `&mut World` is sound.
        unsafe { &mut *self.ptr }
    }
}
