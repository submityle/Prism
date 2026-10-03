//! Systems: turning functions into schedulable units of work over a [`World`].
//!
//! This is the M1 system layer of the design doc (§8). It provides:
//!
//! * [`SystemParam`] — the trait that makes a type usable as a function
//!   argument, declaring its component/resource access and building its live
//!   value from an [`UnsafeWorldCell`] under the kernel's non-aliasing
//!   discipline. Implemented for [`Res`], [`ResMut`],
//!   [`Commands`](crate::command::Commands), [`Query`], and tuples of up to
//!   twelve params.
//! * [`System`] — a runnable unit, implemented by [`FunctionSystem`] (ordinary
//!   `fn(P0, …)`) and [`ExclusiveFunctionSystem`] (`fn(&mut World)`).
//! * [`IntoSystem`] — the conversion `fn → System` used by the scheduler.
//! * [`UnsafeWorldCell`] — the single `unsafe` chokepoint through which params
//!   fetch disjoint borrows from one `&mut World`.
//!
//! # Not yet here (honestly deferred)
//!
//! The parallel, conflict-graph executor and fiber job graph (§8.2–§8.3) are
//! deferred until `prism_tasks` lands; the schedule layer currently runs
//! single-threaded. `SystemSet`s and run-conditions-as-systems are also future
//! work. Nothing here is stubbed — the deferred pieces are simply absent. The
//! per-system [`Access`](crate::query::Access) is already recorded so that
//! executor can be dropped in without touching the param layer.
//!
//! [`Res`]: crate::system::param::Res
//! [`ResMut`]: crate::system::param::ResMut
//! [`Query`]: crate::system::query_param::Query
//! [`FunctionSystem`]: crate::system::function::FunctionSystem
//! [`ExclusiveFunctionSystem`]: crate::system::exclusive::ExclusiveFunctionSystem

pub mod exclusive;
pub mod function;
pub mod param;
pub mod query_param;
pub mod world_cell;

pub use exclusive::{ExclusiveFunctionSystem, IsExclusiveSystem};
pub use function::{
    BoxedSystem, FunctionSystem, IntoSystem, IsFunctionSystem, System, SystemParamFunction,
};
pub use param::{Local, Res, ResMut, SystemParam, SystemParamItem};
pub use query_param::Query;
pub use world_cell::UnsafeWorldCell;

#[cfg(test)]
mod tests;
