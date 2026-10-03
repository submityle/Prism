//! ECS/tasks integration: automatic system/job instrumentation and cross-thread
//! flow connection (M3).
//!
//! This module is fully self-contained inside `prism_diagnostic`: it defines the
//! opt-in integration surface ([`Instrumentable`]) and the scoped guards
//! ([`system::SystemScope`], [`job::JobScope`]) that other crates such as
//! `prism_ecs` and `prism_tasks` can adopt without this crate depending on them.
//!
//! - [`system`] — [`SystemScope`](system::SystemScope) auto-instruments a
//!   system's per-frame work, recording a span on the `system` track that feeds
//!   the [`LoadProfile`](crate::metrics::LoadProfile) aggregation.
//! - [`job`] — [`JobFlow`](job::JobFlow)/[`JobScope`](job::JobScope) instrument a
//!   task that is enqueued on one thread and executed on another, connecting the
//!   two sites with a Chrome flow arrow.
//! - [`flow`] — [`FlowId`](flow::FlowId) allocation plus the low-level
//!   flow/async emission helpers the job instrumentation is built on.

extern crate alloc;

use alloc::string::ToString as _;

pub mod flow;
pub mod job;
pub mod system;

pub use flow::{
    async_begin, async_end, flow_finish, flow_start, flow_step, next_flow_id, FlowId, FLOW_CATEGORY,
};
pub use job::{JobFlow, JobScope, JOB_CATEGORY};
pub use system::{instrument_system, SystemScope, SYSTEM_CATEGORY};

/// Opt-in integration hook implemented by a scheduler's system/job descriptor
/// so `prism_diagnostic` can instrument it without depending on `prism_ecs` or
/// `prism_tasks`.
///
/// Other crates implement this trait on their own types; this crate then opens
/// a [`SystemScope`] for any such descriptor via [`instrument`].
pub trait Instrumentable {
    /// Human-readable label used as the trace scope name.
    fn label(&self) -> &str;

    /// Trace category/track for the descriptor; defaults to
    /// [`SYSTEM_CATEGORY`].
    fn category(&self) -> &str {
        SYSTEM_CATEGORY
    }
}

/// Open a [`SystemScope`] for any [`Instrumentable`] descriptor, timing the work
/// that follows until the returned guard drops.
pub fn instrument(target: &impl Instrumentable) -> SystemScope {
    SystemScope::in_category(target.label().to_string(), target.category().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::ring;

    struct FakeSystem {
        name: &'static str,
    }

    impl Instrumentable for FakeSystem {
        fn label(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn instrument_opens_scope_from_descriptor() {
        ring::clear_current_thread();
        let sys = FakeSystem { name: "spawn_waves" };
        {
            let scope = instrument(&sys);
            assert_eq!(scope.depth(), 0);
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "spawn_waves");
        assert_eq!(spans[0].category.as_deref(), Some(SYSTEM_CATEGORY));
    }
}
