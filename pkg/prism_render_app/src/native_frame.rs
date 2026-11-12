//! Observable lifecycle receipt for the native RenderGraph host.
//!
//! This type is deliberately a receipt, not an executor. The graph and the
//! runtime remain the only authorities for compilation, recording, submission
//! and completion; the receipt only records the boundaries they exposed.

use alloc::string::String;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use prism_ecs::prelude::Resource;
use prism_render_architecture::render_fabric::{RenderDomain, RenderFabricFrame};
use prism_render_graph::{ExecutionPlan, RenderGraph};
use prism_render_runtime::{RenderRuntime, SubmissionError};

use prism_render_driver::{FrameSync, Instance};

/// The furthest lifecycle boundary reached by the latest native frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFramePhase {
    /// No frame has entered the native host yet.
    Idle,
    /// Fabric frame admission rejected the frame before graph construction.
    AdmissionFailed,
    /// At least one graph was compiled, but preparation did not finish.
    Compiled,
    /// At least one graph was prepared and is ready for recording.
    Prepared,
    /// A prepared graph was recorded and submitted, or submission failed
    /// after recording began.
    Recorded,
    /// At least one graph reached the queue.
    Submitted,
    /// The latest queue-ordered submission completion callback was observed.
    Completed,
    /// The frame was rejected or partially failed after entering execution.
    Failed,
    /// The frame had no non-empty output to execute.
    Skipped,
}
/// Native full-frame lifecycle evidence published by `prism_render_app`.
///
/// The counters aggregate all output graphs created by one app update. A
/// multi-window update can therefore have more than one graph and submission;
/// `last_submission` is the queue-ordered fence for the aggregate completion
/// check because all output submissions use the same native queue.
#[derive(Clone, Debug, Resource)]
pub struct NativeFrameExecution {
    frame_index: u64,
    phase: NativeFramePhase,
    graph_count: u32,
    declared_passes: u64,
    compiled_passes: u64,
    prepared_passes: u64,
    recorded_passes: u64,
    imported_resources: u64,
    submissions: u32,
    last_submission: Option<u64>,
    completed_value: u64,
    fabric_items: u64,
    fabric_domain_items: [u64; 5],
    failure: Option<String>,
    completion_signal: Arc<AtomicU64>,
}

impl Default for NativeFrameExecution {
    fn default() -> Self {
        Self {
            frame_index: 0,
            phase: NativeFramePhase::Idle,
            graph_count: 0,
            declared_passes: 0,
            compiled_passes: 0,
            prepared_passes: 0,
            recorded_passes: 0,
            imported_resources: 0,
            submissions: 0,
            last_submission: None,
            completed_value: 0,
            fabric_items: 0,
            fabric_domain_items: [0; 5],
            failure: None,
            completion_signal: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl NativeFrameExecution {
    /// Frame number represented by this receipt.
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Furthest lifecycle phase observed for this frame.
    #[must_use]
    pub fn phase(&self) -> NativeFramePhase {
        if self.phase == NativeFramePhase::Submitted
            && self
                .last_submission
                .is_some_and(|submission| self.completed_value() >= submission)
        {
            NativeFramePhase::Completed
        } else {
            self.phase
        }
    }

    /// Number of output graphs submitted by this frame.
    #[must_use]
    pub const fn graph_count(&self) -> u32 {
        self.graph_count
    }

    /// Number of passes declared before graph culling.
    #[must_use]
    pub const fn declared_passes(&self) -> u64 {
        self.declared_passes
    }

    /// Number of passes in compiled plans before recording.
    #[must_use]
    pub const fn compiled_passes(&self) -> u64 {
        self.compiled_passes
    }

    /// Number of surviving passes in prepared plans.
    #[must_use]
    pub const fn prepared_passes(&self) -> u64 {
        self.prepared_passes
    }

    /// Number of passes recorded into submitted command buffers.
    #[must_use]
    pub const fn recorded_passes(&self) -> u64 {
        self.recorded_passes
    }

    /// Number of imported texture and buffer declarations across output graphs.
    #[must_use]
    pub const fn imported_resources(&self) -> u64 {
        self.imported_resources
    }

    /// Number of successful queue submissions in this frame.
    #[must_use]
    pub const fn submissions(&self) -> u32 {
        self.submissions
    }

    /// Last queue submission value, if any graph reached the queue.
    #[must_use]
    pub const fn last_submission(&self) -> Option<u64> {
        self.last_submission
    }

    /// Greatest completion value observed by the runtime host.
    #[must_use]
    pub fn completed_value(&self) -> u64 {
        let signaled = self.completion_signal.load(Ordering::Acquire);
        if signaled > self.completed_value {
            signaled
        } else {
            self.completed_value
        }
    }

    /// The failure that closed this frame, if execution did not reach a
    /// successful output submission.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Number of validated semantic Fabric items admitted with this frame.
    #[must_use]
    pub const fn fabric_items(&self) -> u64 {
        self.fabric_items
    }

    /// Number of admitted Fabric items by domain in stable domain order:
    /// Surface, Particle, Volume, Hair, Decal.
    #[must_use]
    pub const fn fabric_domain_items(&self) -> [u64; 5] {
        self.fabric_domain_items
    }

    pub(crate) fn begin(&mut self, frame_index: u64) {
        self.frame_index = frame_index;
        self.phase = NativeFramePhase::Idle;
        self.graph_count = 0;
        self.declared_passes = 0;
        self.compiled_passes = 0;
        self.prepared_passes = 0;
        self.recorded_passes = 0;
        self.imported_resources = 0;
        self.submissions = 0;
        self.last_submission = None;
        self.completed_value = self.completion_signal.load(Ordering::Acquire);
        self.fabric_items = 0;
        self.fabric_domain_items = [0; 5];
        self.failure = None;
    }

    pub(crate) fn admitted(&mut self, frame: &RenderFabricFrame) {
        self.fabric_items = frame.len() as u64;
        for item in frame.items() {
            self.fabric_domain_items[domain_index(item.domain)] += 1;
        }
        self.phase = NativeFramePhase::Idle;
    }

    pub(crate) fn admission_failed(&mut self, frame_index: u64) {
        self.frame_index = frame_index;
        self.phase = NativeFramePhase::AdmissionFailed;
        self.graph_count = 0;
        self.declared_passes = 0;
        self.compiled_passes = 0;
        self.prepared_passes = 0;
        self.recorded_passes = 0;
        self.imported_resources = 0;
        self.submissions = 0;
        self.last_submission = None;
        self.completed_value = self.completion_signal.load(Ordering::Acquire);
        self.fabric_items = 0;
        self.fabric_domain_items = [0; 5];
        self.failure = None;
    }

    pub(crate) fn admission_failed_with(&mut self, reason: impl Into<String>) {
        let frame_index = self.frame_index;
        self.admission_failed(frame_index);
        self.failure = Some(reason.into());
    }

    pub(crate) fn declared(&mut self, declared_passes: usize, imported_resources: usize) {
        self.graph_count = self.graph_count.saturating_add(1);
        self.declared_passes = self.declared_passes.saturating_add(declared_passes as u64);
        self.imported_resources = self
            .imported_resources
            .saturating_add(imported_resources as u64);
    }

    pub(crate) fn compiled(&mut self, surviving_passes: usize) {
        self.compiled_passes = self.compiled_passes.saturating_add(surviving_passes as u64);
        self.advance(NativeFramePhase::Compiled);
    }

    pub(crate) fn prepared(&mut self, plan: &ExecutionPlan) {
        self.prepared_passes = self
            .prepared_passes
            .saturating_add(plan.pass_count() as u64);
        self.advance(NativeFramePhase::Prepared);
    }

    pub(crate) fn recorded(&mut self, pass_count: usize) {
        self.recorded_passes = self.recorded_passes.saturating_add(pass_count as u64);
        self.advance(NativeFramePhase::Recorded);
    }

    pub(crate) fn submitted(&mut self, submission: u64) {
        self.submissions = self.submissions.saturating_add(1);
        self.last_submission = Some(submission);
        // Completion is derived from the latest submission. A multi-output
        // frame may submit again after an earlier output completed.
        if self.phase != NativeFramePhase::Failed {
            self.phase = NativeFramePhase::Submitted;
        }
    }

    pub(crate) fn failed(&mut self) {
        self.phase = NativeFramePhase::Failed;
    }

    pub(crate) fn failed_with(&mut self, reason: impl Into<String>) {
        self.failure = Some(reason.into());
        self.failed();
    }

    pub(crate) fn skipped(&mut self) {
        if self.phase == NativeFramePhase::Idle {
            self.phase = NativeFramePhase::Skipped;
        }
    }

    pub(crate) fn observe_completion(&mut self, completed_value: u64) {
        self.completed_value = self.completed_value.max(completed_value);
        self.completion_signal
            .fetch_max(completed_value, Ordering::Release);
        if self.phase == NativeFramePhase::Submitted
            && self
                .last_submission
                .is_some_and(|submission| self.completed_value() >= submission)
        {
            self.phase = NativeFramePhase::Completed;
        }
    }

    pub(crate) fn completion_signal(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.completion_signal)
    }

    fn advance(&mut self, phase: NativeFramePhase) {
        if Self::phase_rank(phase) > Self::phase_rank(self.phase) {
            self.phase = phase;
        }
    }

    const fn phase_rank(phase: NativeFramePhase) -> u8 {
        match phase {
            NativeFramePhase::Idle => 0,
            NativeFramePhase::AdmissionFailed => 1,
            NativeFramePhase::Compiled => 2,
            NativeFramePhase::Prepared => 3,
            NativeFramePhase::Recorded => 4,
            NativeFramePhase::Submitted => 5,
            NativeFramePhase::Completed => 6,
            NativeFramePhase::Failed => 7,
            NativeFramePhase::Skipped => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_graph_admission_failure_does_not_become_post_recording_failure() {
        let mut receipt = NativeFrameExecution::default();
        receipt.begin(7);
        receipt.admission_failed_with("target completion was not confirmed");

        assert_eq!(receipt.phase(), NativeFramePhase::AdmissionFailed);
        assert_eq!(receipt.graph_count(), 0);
        assert_eq!(receipt.submissions(), 0);
        assert_eq!(
            receipt.failure(),
            Some("target completion was not confirmed")
        );
    }
}
/// Execute one production graph while exposing each native lifecycle boundary.
///
/// `RenderRuntime::execute_tracked` intentionally hides compile and prepare.
/// The native app needs those boundaries for admission diagnostics and for
/// fail-closed acceptance evidence, so this helper freezes the plan once and
/// passes that exact prepared graph to the runtime.
pub(crate) fn execute_native_graph<I>(
    execution: &mut NativeFrameExecution,
    runtime: &RenderRuntime<I>,
    graph: RenderGraph,
) -> Result<u64, SubmissionError>
where
    I: Instance + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: FrameSync,
{
    let declared_passes = graph.pass_count();
    let imported_resources = graph.imported_resource_count();
    execution.declared(declared_passes, imported_resources);
    let prepared = match graph.prepare_with(|plan| execution.compiled(plan.pass_count())) {
        Ok(prepared) => prepared,
        Err(error) => {
            execution.failed_with(format!("native graph compile failed: {error:?}"));
            return Err(SubmissionError::Compile(error));
        }
    };
    let prepared_passes = prepared.plan().pass_count();
    execution.prepared(prepared.plan());
    let submission = match runtime.execute_prepared_owned(prepared, &[]) {
        Ok(submission) => submission,
        Err(error) => {
            execution.failed_with(format!("native graph submission failed: {error:?}"));
            return Err(error);
        }
    };
    execution.recorded(prepared_passes);
    let value = submission.completion_value();
    execution.submitted(value);
    let signal = execution.completion_signal();
    runtime.queue().on_submitted_work_done(Box::new(move || {
        signal.fetch_max(value, Ordering::Release);
    }));
    execution.observe_completion(runtime.completed_value());
    Ok(value)
}

const fn domain_index(domain: RenderDomain) -> usize {
    match domain {
        RenderDomain::Surface => 0,
        RenderDomain::Particle => 1,
        RenderDomain::Volume => 2,
        RenderDomain::Hair => 3,
        RenderDomain::Decal => 4,
    }
}
