//! The unit of work scheduled on the pool.

/// A boxed closure executed by a worker. Jobs are `Send` because they migrate
/// between threads via work-stealing.
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;
