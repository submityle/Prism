//! Run conditions: boxed predicates gating whether a configured system (or a
//! group of them) executes on a given schedule run.
//!
//! A run condition is any `FnMut(&World) -> bool` that is `Send + Sync`. It is
//! evaluated immediately before the gated work; returning `false` skips that
//! work for the current run. Conditions are stored type-erased as
//! [`BoxedCondition`] so heterogeneous predicates can live side by side in a
//! [`SystemConfigs`](crate::schedule::config::SystemConfigs) node.
//!
//! The helpers below return concrete `impl FnMut(&World) -> bool` closures so
//! they can be used directly with
//! [`run_if`](crate::schedule::config::IntoSystemConfigs::run_if). They are the
//! real, data-driven building blocks (resource presence/equality, one-shot
//! latching); richer run-condition combinators (and run-conditions-as-systems)
//! are deferred to a later milestone and are honestly absent rather than
//! stubbed.

use crate::resource::Resource;
use crate::world::World;
use alloc::boxed::Box;

/// A type-erased run condition stored inside a schedule configuration node.
///
/// It is evaluated against a shared `&World` right before the gated system or
/// group runs. `Send + Sync` keeps a [`Schedule`](crate::schedule::Schedule)
/// movable across threads for the future parallel executor.
pub type BoxedCondition = Box<dyn FnMut(&World) -> bool + Send + Sync>;

/// A run condition that passes while a resource of type `R` exists in the
/// world.
///
/// ```
/// use prism_ecs::prelude::*;
/// use prism_ecs::schedule::resource_exists;
///
/// #[derive(Default)]
/// struct Config;
/// impl Resource for Config {}
///
/// let mut cond = resource_exists::<Config>();
/// let mut world = World::new();
/// assert!(!cond(&world));
/// world.insert_resource(Config);
/// assert!(cond(&world));
/// ```
pub fn resource_exists<R: Resource>() -> impl FnMut(&World) -> bool + Send + Sync {
    |world: &World| world.contains_resource::<R>()
}

/// A run condition that passes while a resource of type `R` exists and compares
/// equal to `value`.
///
/// The comparison value is captured by the closure, so `R` must be `Send + Sync`
/// (guaranteed by the [`Resource`] bound) to keep the condition thread-movable.
pub fn resource_equals<R>(value: R) -> impl FnMut(&World) -> bool + Send + Sync
where
    R: Resource + PartialEq,
{
    move |world: &World| world.get_resource::<R>() == Some(&value)
}

/// A run condition that passes exactly once: it returns `true` on its first
/// evaluation and `false` on every subsequent one.
///
/// Useful for latching one-shot work onto a repeatedly-run schedule without a
/// dedicated resource flag.
pub fn run_once() -> impl FnMut(&World) -> bool + Send + Sync {
    let mut has_run = false;
    move |_world: &World| {
        if has_run {
            false
        } else {
            has_run = true;
            true
        }
    }
}
