//! Shared task urgency without adding ownership to copyable cell metadata.

use alloc::sync::Arc;
use core::task::{Context, Poll};
use prism_tasks::PriorityCell;

use crate::{CellIo, CellReadRequest, IoCompletion, RequestId, SubmitError};

/// A cell backend that retains a live task urgency cell for each request.
///
/// Implementations must retain the same cell, not copy its current value.
/// Boosts influence future queued admission; they neither wake the backend nor
/// preempt active work. Backends without this capability do not implement it.
pub trait PriorityCellIo: CellIo {
    /// Admits a request sharing the given urgency cell.
    fn submit_with_priority(
        &mut self,
        request: CellReadRequest,
        priority: Arc<PriorityCell>,
    ) -> Result<RequestId, SubmitError>;
}

/// Applies one shared task urgency cell to ordinary frame-coordinator submissions.
///
/// Clone [`Self::priority`] into `TaskPool::spawn_future_with_priority` to give
/// the driving task and its queued reads the same monotonically boostable lane.
/// Static per-cell distance/stream scores still break ties within a lane.
pub struct PrioritizedCellIo<I: PriorityCellIo> {
    io: I,
    priority: Arc<PriorityCell>,
}

impl<I: PriorityCellIo> PrioritizedCellIo<I> {
    /// Wraps a backend without creating threads, tasks, or requests.
    pub fn new(io: I, priority: Arc<PriorityCell>) -> Self {
        Self { io, priority }
    }

    /// Returns the cell shared by subsequent ordinary submissions.
    pub fn priority(&self) -> &Arc<PriorityCell> {
        &self.priority
    }

    /// Returns the backend, preserving its accepted requests and ownership.
    pub fn into_inner(self) -> I {
        self.io
    }
}

impl<I: PriorityCellIo> CellIo for PrioritizedCellIo<I> {
    type Error = I::Error;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        self.io
            .submit_with_priority(request, Arc::clone(&self.priority))
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        self.io.cancel(id)
    }

    fn in_flight(&self) -> usize {
        self.io.in_flight()
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        self.io.poll_completion(cx)
    }
}

impl<I: PriorityCellIo> PriorityCellIo for PrioritizedCellIo<I> {
    fn submit_with_priority(
        &mut self,
        request: CellReadRequest,
        priority: Arc<PriorityCell>,
    ) -> Result<RequestId, SubmitError> {
        self.io.submit_with_priority(request, priority)
    }
}
