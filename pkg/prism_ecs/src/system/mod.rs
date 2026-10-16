//! Systems: turning functions into schedulable units of work over a [`World`].
//!
//! This is the M1 system layer of the design doc (§8). It provides:
//!
//! * [`SystemParam`] — the trait that makes a type usable as a function
//!   argument, declaring its component/resource access and building its live
//!   value from an [`UnsafeWorldCell`] under the kernel's non-aliasing
//!   discipline. Implemented for [`Res`], [`ResMut`],
//!   [`Commands`](crate::command::Commands), [`Query`], [`ParamSet`], and tuples of up to
//!   twelve params, plus the renderer's per-view lookup: [`CurrentView`] and
//!   the `Option`-wrapped [`ViewQuery`].
//! * [`System`] — a runnable unit, implemented by [`FunctionSystem`] (ordinary
//!   `fn(P0, …)`) and [`ExclusiveFunctionSystem`] (`fn(&mut World)`).
//! * [`IntoSystem`] — the conversion `fn → System` used by the scheduler.
//! * [`UnsafeWorldCell`] — the single `unsafe` chokepoint through which params
//!   fetch disjoint borrows from one `&mut World`.
//!
//! # Execution boundary
//!
//! The system parameter layer records the complete access set and is consumed by
//! both the single-threaded and `multi_thread` schedule executors. The
//! conflict-graph executor, `SystemSet`s, and run conditions live in
//! [`crate::schedule`]. System-internal row/slice fan-out and dependency DAGs
//! are available through [`JobGraph`] when `multi_thread` is enabled. The
//! remaining limitation is integration and workload validation: the kernel does
//! not claim that every system is automatically parallel-safe, nor that every
//! workload has near-linear scaling.
//!
//! [`Res`]: crate::system::param::Res
//! [`ResMut`]: crate::system::param::ResMut
//! [`Query`]: crate::system::query_param::Query
//! [`FunctionSystem`]: crate::system::function::FunctionSystem
//! [`ExclusiveFunctionSystem`]: crate::system::exclusive::ExclusiveFunctionSystem
//! [`JobGraph`]: crate::system::JobGraph

pub mod exclusive;
pub mod function;
#[cfg(feature = "multi_thread")]
pub mod job_dag;
#[cfg(feature = "multi_thread")]
pub mod job_graph;
pub mod param;
pub mod param_set;
pub mod query_param;
pub mod view_query;
pub mod world_cell;

pub use exclusive::{ExclusiveFunctionSystem, IsExclusiveSystem};
pub use function::{
    BoxedSystem, FunctionSystem, IntoSystem, IsFunctionSystem, System, SystemParamFunction,
};
pub use param::{Local, Res, ResMut, SystemParam, SystemParamItem};
pub use param_set::ParamSet;
pub use query_param::Query;
pub use view_query::{CurrentView, ViewQuery, ViewQueryState};
pub use world_cell::UnsafeWorldCell;

#[cfg(feature = "multi_thread")]
pub use job_dag::{JobDag, JobId};
#[cfg(feature = "multi_thread")]
pub use job_graph::{ComputeTaskPool, JobGraph};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod change_detection_tests;

#[cfg(all(test, feature = "multi_thread"))]
mod job_dag_tests;

#[cfg(all(test, feature = "multi_thread"))]
mod job_graph_tests;
