//! The [`System`] abstraction and the machinery that turns a plain Rust `fn`
//! into one (design §8.1).
//!
//! A [`System`] is anything the scheduler can initialise, inspect for access,
//! and run against a [`World`]. The common case is an ordinary function whose
//! arguments are [`SystemParam`]s; [`IntoSystem`] wraps such a function in a
//! [`FunctionSystem`] that resolves its params once and fetches them each run.

use alloc::boxed::Box;
use core::any::type_name;
use core::marker::PhantomData;

use crate::change::Tick;
use crate::query::Access;
use crate::system::param::{SystemParam, SystemParamItem};
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::{World, WorldId};

/// Something the scheduler can run against a [`World`].
///
/// Implementors expose their [`access`](System::access) (resolved during
/// [`initialize`](System::initialize)) so the executor can build a conflict
/// graph, a [`run_unsafe`](System::run_unsafe) that fetches params and runs the
/// body, and [`apply_deferred`](System::apply_deferred) to flush buffered
/// commands at a sync point.
pub trait System: Send + Sync + 'static {
    /// The value produced by one run (`()` for ordinary systems, `bool` for run
    /// conditions).
    type Out;

    /// A human-readable name (used in diagnostics and ambiguity reports).
    fn name(&self) -> &str;

    /// Resolve all param state against `world`. Must be called once before
    /// [`run`](System::run) / [`run_unsafe`](System::run_unsafe).
    fn initialize(&mut self, world: &mut World);

    /// The resource/component access this system requires. Only meaningful
    /// after [`initialize`](System::initialize).
    fn access(&self) -> &Access;

    /// Whether this system needs exclusive `&mut World` access and must run
    /// alone (not in parallel with any other system).
    #[inline]
    fn is_exclusive(&self) -> bool {
        false
    }

    /// The tick this system last completed a run at (the exclusive lower bound
    /// of its change-detection window). Defaults to [`Tick::ZERO`] for systems
    /// that do not track change detection; [`FunctionSystem`] overrides it.
    #[inline]
    fn get_last_run(&self) -> Tick {
        Tick::ZERO
    }

    /// Record the tick this system just ran at, so its next run observes only
    /// writes made since. Called by the executors after a run completes.
    #[inline]
    fn set_last_run(&mut self, _last_run: Tick) {}

    /// Assert this system may run against `world` — i.e. it was initialized for
    /// that same world. Called by [`run`](System::run) (and by the executors)
    /// *before* any mutation, so a cross-world misuse is rejected without side
    /// effects. Defaults to accepting any world for systems whose state holds no
    /// world-scoped ids.
    #[inline]
    fn validate_world(&self, _world: &World) {}

    /// Run the system body, fetching its params from `world`.
    ///
    /// # Safety
    /// The caller must guarantee that no borrow aliasing this system's declared
    /// [`access`](System::access) is live through `world` for the duration of
    /// the run. The parallel executor upholds this via the conflict graph; the
    /// sequential path upholds it trivially by running one system at a time.
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) -> Self::Out;

    /// Apply any deferred effects (flush [`Commands`](crate::command::Commands))
    /// into `world`.
    fn apply_deferred(&mut self, world: &mut World);

    /// Run the system exclusively against `world` and immediately apply its
    /// deferred effects. This is the simple, always-sound entry point used by
    /// the sequential schedule.
    fn run(&mut self, world: &mut World) -> Self::Out {
        // Reject a cross-world run before touching the world (in particular
        // before advancing its change tick), so a misuse leaves no side effects.
        self.validate_world(world);
        // Per-system change window: `last_run` is where this system left off,
        // `this_run` is a fresh world tick so writes made during this run (and
        // by systems that ran since) land inside `(last_run, this_run]`.
        let last_run = self.get_last_run();
        let this_run = world.increment_change_tick();
        let out = {
            let cell = UnsafeWorldCell::new_mutable_with_ticks(world, last_run, this_run);
            // SAFETY: `cell` is the only handle to `world` for this scope and no
            // other system runs concurrently here, so nothing aliases the
            // system's access.
            unsafe { self.run_unsafe(cell) }
        };
        self.set_last_run(this_run);
        self.apply_deferred(world);
        out
    }
}

/// A plain function usable as a [`System`]: its arguments are [`SystemParam`]s.
///
/// Implemented for every `FnMut(P0, P1, …) -> Out` whose arguments implement
/// [`SystemParam`] (arities 0–12), via the two-`FnMut`-bound inference bridge
/// that lets a closure be called with the params' resolved item types.
pub trait SystemParamFunction<Marker>: Send + Sync + 'static {
    /// The tuple of this function's parameters.
    type Param: SystemParam;
    /// The function's return type.
    type Out;

    /// Invoke the function with the fetched params.
    fn run<'a>(&'a mut self, params: SystemParamItem<'a, 'a, Self::Param>) -> Self::Out;
}

macro_rules! impl_system_param_function {
    ($($param:ident),*) => {
        #[allow(non_snake_case, unused_variables, clippy::too_many_arguments)]
        impl<Out, Func, $($param: SystemParam),*> SystemParamFunction<fn($($param,)*) -> Out>
            for Func
        where
            Func: Send + Sync + 'static,
            for<'a> &'a mut Func:
                FnMut($($param,)*) -> Out
                + FnMut($(<$param as SystemParam>::Item<'_, '_>,)*) -> Out,
            Out: 'static,
        {
            type Param = ($($param,)*);
            type Out = Out;

            #[inline]
            fn run<'a>(&'a mut self, params: SystemParamItem<'a, 'a, Self::Param>) -> Out {
                fn call_inner<Out, $($param,)*>(
                    mut f: impl FnMut($($param,)*) -> Out,
                    $($param: $param,)*
                ) -> Out {
                    f($($param,)*)
                }
                let ($($param,)*) = params;
                call_inner(self, $($param,)*)
            }
        }
    };
}

impl_system_param_function!();
impl_system_param_function!(P0);
impl_system_param_function!(P0, P1);
impl_system_param_function!(P0, P1, P2);
impl_system_param_function!(P0, P1, P2, P3);
impl_system_param_function!(P0, P1, P2, P3, P4);
impl_system_param_function!(P0, P1, P2, P3, P4, P5);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);

/// A [`System`] built from a [`SystemParamFunction`].
///
/// Holds the function, its resolved param state (lazily created by
/// [`initialize`](System::initialize)), and the accumulated [`Access`].
pub struct FunctionSystem<Marker, F>
where
    F: SystemParamFunction<Marker>,
{
    func: F,
    param_state: Option<<F::Param as SystemParam>::State>,
    access: Access,
    name: &'static str,
    /// Tick this system last completed at; threaded into its change window.
    last_run: Tick,
    /// Identity of the world whose components/resources `param_state` resolved
    /// against. Captured by [`initialize`](System::initialize); every run then
    /// verifies the target world carries the same id, so a system can never
    /// replay world A's resolved ids against world B's storage.
    world_id: Option<WorldId>,
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker, F> FunctionSystem<Marker, F>
where
    F: SystemParamFunction<Marker>,
{
    /// Wrap `func`; param state is resolved later by
    /// [`initialize`](System::initialize).
    #[inline]
    fn new(func: F) -> Self {
        Self {
            func,
            param_state: None,
            access: Access::new(),
            name: type_name::<F>(),
            last_run: Tick::ZERO,
            world_id: None,
            _marker: PhantomData,
        }
    }

    /// Assert `world_id` is the world this system resolved its params against.
    ///
    /// The param states hold dense component/resource ids that are only
    /// meaningful within the initialising world; running against another world
    /// would reinterpret its storage (undefined behaviour). A safe API must
    /// therefore reject the mismatch rather than proceed.
    #[inline]
    fn assert_world_id(&self, world_id: WorldId) {
        assert_eq!(
            self.world_id,
            Some(world_id),
            "system was initialized for one World but run against another; \
             re-initialize it (or build a fresh system) for the target world"
        );
    }
}

impl<Marker, F> System for FunctionSystem<Marker, F>
where
    Marker: Send + Sync + 'static,
    F: SystemParamFunction<Marker>,
{
    type Out = F::Out;

    #[inline]
    fn name(&self) -> &str {
        self.name
    }

    #[inline]
    fn initialize(&mut self, world: &mut World) {
        // Re-initializing against a *different* world invalidates the previous
        // change-detection window: the old `last_run` tick has no meaning in the
        // new world's tick space, so restart from the origin (which also keeps
        // `Changed`/`Added` from silently missing the new world's existing
        // writes). Re-initializing against the *same* world preserves it.
        if self.world_id != Some(world.id()) {
            self.last_run = Tick::ZERO;
        }
        let state = <F::Param as SystemParam>::init_state(world);
        let mut access = Access::new();
        <F::Param as SystemParam>::update_access(&state, &mut access);
        self.param_state = Some(state);
        self.access = access;
        self.world_id = Some(world.id());
    }

    #[inline]
    fn access(&self) -> &Access {
        &self.access
    }

    #[inline]
    fn get_last_run(&self) -> Tick {
        self.last_run
    }

    #[inline]
    fn set_last_run(&mut self, last_run: Tick) {
        self.last_run = last_run;
    }

    #[inline]
    fn validate_world(&self, world: &World) {
        self.assert_world_id(world.id());
    }

    #[inline]
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) -> Self::Out {
        // SAFETY: the caller guarantees `world` references a live world, so
        // reading its id is sound; the comparison guards the resolved param ids.
        self.assert_world_id(unsafe { world.world() }.id());
        let state = self
            .param_state
            .as_mut()
            .expect("system was run before initialize()");
        // SAFETY: the caller guarantees nothing aliases this system's declared
        // access for the duration of the fetched params / run.
        let params = unsafe { <F::Param as SystemParam>::get_param(state, world) };
        self.func.run(params)
    }

    #[inline]
    fn apply_deferred(&mut self, world: &mut World) {
        self.assert_world_id(world.id());
        if let Some(state) = self.param_state.as_mut() {
            <F::Param as SystemParam>::apply(state, world);
        }
    }
}

/// Conversion into a [`System`]. Implemented for any [`SystemParamFunction`] and
/// (identity) for any [`System`] already.
pub trait IntoSystem<Out, Marker>: Sized {
    /// The concrete system type produced.
    type System: System<Out = Out>;

    /// Perform the conversion.
    fn into_system(self) -> Self::System;
}

/// Marker distinguishing the function `IntoSystem` impl from the identity impl.
pub struct IsFunctionSystem;

impl<Marker, F> IntoSystem<F::Out, (IsFunctionSystem, Marker)> for F
where
    Marker: Send + Sync + 'static,
    F: SystemParamFunction<Marker>,
{
    type System = FunctionSystem<Marker, F>;

    #[inline]
    fn into_system(self) -> Self::System {
        FunctionSystem::new(self)
    }
}

/// A type-erased, owned [`System`] with the given output type.
pub type BoxedSystem<Out = ()> = Box<dyn System<Out = Out>>;
