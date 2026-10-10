//! Automatic instrumentation for tasks/jobs that migrate across threads.
//!
//! A job is typically *enqueued* on one thread and *executed* on another worker
//! thread. [`JobFlow`] captures the enqueue site: constructing one allocates a
//! [`FlowId`] and emits a flow *start* event on the enqueuing thread. The id is
//! carried with the job payload to the worker; when the worker begins the body
//! it calls [`JobFlow::execute`], which emits the matching flow *finish* event
//! (drawing the cross-thread arrow) and opens a [`JobScope`] timing guard for
//! the body itself.
//!
//! ```
//! use prism_diagnostic::instrument::JobFlow;
//!
//! // On the producing thread:
//! let flow = JobFlow::start("load_asset");
//! // ... hand `flow` to a worker (here, inline for the doctest) ...
//!
//! // On the worker thread:
//! {
//!     let _job = flow.execute();
//!     // ... job body is timed here ...
//! }
//! ```

extern crate alloc;

use alloc::string::String;

use crate::instrument::flow::{flow_finish, flow_start, next_flow_id, FlowId, FLOW_CATEGORY};
use crate::span::Scope;

/// Trace category/track that instrumented jobs record their bodies under.
pub const JOB_CATEGORY: &str = "job";

/// A hand-off token created where a job is enqueued.
///
/// Construction emits the flow *start* event; [`execute`](JobFlow::execute) on
/// the worker thread emits the matching *finish* and begins timing the body.
#[derive(Clone, Debug)]
pub struct JobFlow {
    id: FlowId,
    name: String,
    category: String,
}

impl JobFlow {
    /// Begin a job flow named `name` under the default [`FLOW_CATEGORY`],
    /// emitting the flow start event on the calling (enqueuing) thread.
    pub fn start(name: impl Into<String>) -> Self {
        Self::start_in_category(name, FLOW_CATEGORY)
    }

    /// Begin a job flow named `name` under an explicit flow `category`,
    /// emitting the flow start event on the calling (enqueuing) thread.
    pub fn start_in_category(name: impl Into<String>, category: impl Into<String>) -> Self {
        let id = next_flow_id();
        let name = name.into();
        let category = category.into();
        flow_start(id, name.clone(), category.clone());
        Self { id, name, category }
    }

    /// The correlation id tying this flow's start and finish together.
    pub fn id(&self) -> FlowId {
        self.id
    }

    /// The job's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Begin executing the job on the calling (worker) thread.
    ///
    /// Emits the flow *finish* event (connecting the arrow back to the enqueue
    /// site) and returns a [`JobScope`] that times the job body under
    /// [`JOB_CATEGORY`] until it is dropped.
    pub fn execute(self) -> JobScope {
        flow_finish(self.id, self.name.clone(), self.category);
        JobScope {
            scope: Scope::new(self.name).with_category(JOB_CATEGORY),
        }
    }
}

/// An `RAII` guard timing one job body on the worker thread.
///
/// Created by [`JobFlow::execute`]; on drop it records the body's completed
/// span into the worker thread's ring buffer under [`JOB_CATEGORY`].
#[derive(Debug)]
pub struct JobScope {
    scope: Scope,
}

impl JobScope {
    /// The nesting depth of the job body scope at entry (0 is top level).
    pub fn depth(&self) -> u32 {
        self.scope.depth()
    }

    /// The Prism-assigned thread id executing the job body.
    pub fn thread_id(&self) -> u64 {
        self.scope.thread_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::flow::{flow_records, FlowPhase};
    use crate::trace::ring;

    #[test]
    fn start_then_execute_emits_connected_start_and_finish() {
        let flow = JobFlow::start("load_asset");
        let id = flow.id();
        {
            let job = flow.execute();
            assert_eq!(job.thread_id(), ring::current_thread_id());
        }
        let mine: Vec<_> = flow_records()
            .into_iter()
            .filter(|r| r.id == id.get())
            .collect();
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].phase, FlowPhase::Start);
        assert_eq!(mine[1].phase, FlowPhase::Finish);
        assert_eq!(mine[0].name, "load_asset");
    }

    #[test]
    fn execute_records_body_span_under_job_category() {
        ring::clear_current_thread();
        let flow = JobFlow::start_in_category("compute", "jobs_hi");
        {
            let _job = flow.execute();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "compute");
        assert_eq!(spans[0].category.as_deref(), Some(JOB_CATEGORY));
        assert!(spans[0].duration_nanos > 0);
    }

    #[test]
    fn flow_connects_across_threads() {
        let flow = JobFlow::start("cross_thread_job");
        let id = flow.id();
        let producer_tid = ring::current_thread_id();
        let worker_tid = std::thread::spawn(move || {
            let job = flow.execute();
            let tid = job.thread_id();
            std::thread::sleep(std::time::Duration::from_millis(1));
            tid
        })
        .join()
        .unwrap();
        assert_ne!(producer_tid, worker_tid);

        let mine: Vec<_> = flow_records()
            .into_iter()
            .filter(|r| r.id == id.get())
            .collect();
        assert_eq!(mine.len(), 2);
        // Start recorded on the producer thread, finish on the worker thread.
        assert_eq!(mine[0].phase, FlowPhase::Start);
        assert_eq!(mine[0].thread_id, producer_tid);
        assert_eq!(mine[1].phase, FlowPhase::Finish);
        assert_eq!(mine[1].thread_id, worker_tid);
    }
}
