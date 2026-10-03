//! Minimal `Future` executor that shares the job-graph thread pool.
//!
//! M3 adds async support without a second runtime: futures are polled as jobs
//! on the same [`TaskPool`](crate::TaskPool), so IO-bound `async` work and
//! CPU-bound fork-join work share one work-stealing load balancer (design
//! §10). The pieces are:
//!
//! - [`waker`]: a hand-rolled [`RawWaker`](std::task::RawWaker) vtable over an
//!   `Arc<W>`.
//! - [`executor`]: the [`TaskPool::spawn_async`](crate::TaskPool::spawn_async)
//!   / [`TaskPool::block_on`](crate::TaskPool::block_on) entry points and the
//!   re-schedulable task harness.
//! - [`task`]: the [`Task`] join handle and its shared result cell.
//! - [`counter_bridge`]: [`Counter::wait_async`](crate::Counter::wait_async),
//!   wiring counter completion to future wakeups.

mod counter_bridge;
mod executor;
mod task;
mod waker;

pub(crate) use executor::RunnableTask;

pub use counter_bridge::CounterFuture;
pub use task::Task;
