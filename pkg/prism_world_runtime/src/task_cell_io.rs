//! Structured task cancellation for frame-driven cell IO.

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use core::future::Future;
use core::ops::Bound::{Excluded, Unbounded};
use core::pin::Pin;
use core::task::{Context, Poll};

use prism_tasks::CancelToken;

use crate::{CellIo, CellReadRequest, IoCompletion, RequestId, SubmitError};

const CANCELS_PER_POLL: usize = 16;

/// A cell backend owned by a child of a task cancellation scope.
///
/// The scope's cancellation notification wakes a pending IO poll. Subsequent
/// polls cancel at most sixteen accepted requests per turn and suppress any
/// late successful payloads. Quotas remain held by the underlying backend until
/// each terminal event is consumed. Drive this backend until `in_flight() == 0`
/// to observe all cancellation terminals; dropping it requests cancellation of
/// every remaining request. The standard threaded backend detaches its workers
/// instead of joining file IO; custom backends retain their own drop behavior.
///
/// The wrapped backend must be dedicated to this scope and must initially have
/// no accepted requests. Filesystem work remains on its file workers; a
/// `prism_tasks::TaskPool` can drive the completion future without blocking its
/// worker on a filesystem read.
pub struct TaskCellIo<I: CellIo> {
    io: I,
    token: CancelToken,
    notification: Pin<Box<dyn Future<Output = ()> + Send>>,
    outstanding: BTreeSet<RequestId>,
    cancelled: bool,
    cancel_cursor: Option<RequestId>,
}

impl<I: CellIo> TaskCellIo<I> {
    /// Creates a child scope around an idle backend.
    ///
    /// A non-idle backend is returned unchanged because its accepted requests
    /// cannot be reconstructed through the generic [`CellIo`] contract.
    pub fn new(io: I, parent: &CancelToken) -> Result<Self, I> {
        if io.in_flight() != 0 {
            return Err(io);
        }
        let token = parent.child();
        let notified = token.clone();
        Ok(Self {
            io,
            token,
            notification: Box::pin(async move { notified.cancelled().await }),
            outstanding: BTreeSet::new(),
            cancelled: false,
            cancel_cursor: None,
        })
    }

    /// Returns the child token; cancelling it leaves sibling scopes running.
    #[must_use]
    pub fn cancel_token(&self) -> &CancelToken {
        &self.token
    }

    fn cancel_batch(&mut self) -> bool {
        let mut batch = [None; CANCELS_PER_POLL];
        let mut count = 0;
        let range = match self.cancel_cursor {
            Some(cursor) => self.outstanding.range((Excluded(cursor), Unbounded)),
            None => self.outstanding.range(..),
        };
        for id in range.take(CANCELS_PER_POLL) {
            batch[count] = Some(*id);
            count += 1;
        }
        for id in batch.into_iter().flatten() {
            self.io.cancel(id);
            self.cancel_cursor = Some(id);
        }
        count == CANCELS_PER_POLL
            && self
                .outstanding
                .range((
                    Excluded(self.cancel_cursor.expect("nonempty batch")),
                    Unbounded,
                ))
                .next()
                .is_some()
    }
}

impl<I: CellIo> CellIo for TaskCellIo<I> {
    type Error = I::Error;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        if self.token.is_cancelled() {
            return Err(SubmitError::Cancelled);
        }
        let id = self.io.submit(request)?;
        self.outstanding.insert(id);
        Ok(id)
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        self.outstanding.contains(&id) && self.io.cancel(id)
    }

    fn in_flight(&self) -> usize {
        self.outstanding.len()
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        if !self.cancelled && self.notification.as_mut().poll(cx).is_ready() {
            self.cancelled = true;
        }
        if self.cancelled && self.cancel_batch() {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        self.io.poll_completion(cx).map(|result| {
            result.map(|completion| {
                let id = completion.request_id();
                self.outstanding.remove(&id);
                // A cancellation racing the provider's terminal poll wins
                // until this wrapper hands the payload to its caller.
                if self.token.is_cancelled() {
                    IoCompletion::Cancelled {
                        request_id: id,
                        request: completion.request(),
                    }
                } else {
                    completion
                }
            })
        })
    }
}

impl<I: CellIo> Drop for TaskCellIo<I> {
    fn drop(&mut self) {
        use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

        // A user waker or backend callback may panic. Still attempt every
        // accepted cancellation, then preserve the first panic unless we are
        // already unwinding. Dispose other payloads outside the callbacks;
        // even panic payload destructors may contain arbitrary user code.
        let mut first = catch_unwind(AssertUnwindSafe(|| self.token.cancel())).err();
        for &id in &self.outstanding {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.io.cancel(id))) {
                if first.is_none() {
                    first = Some(payload);
                } else {
                    discard_panic(payload);
                }
            }
        }
        if let Some(payload) = first {
            if std::thread::panicking() {
                discard_panic(payload);
            } else {
                resume_unwind(payload);
            }
        }
    }
}

fn discard_panic(payload: Box<dyn core::any::Any + Send>) {
    if let Err(nested) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload))) {
        // A second panicking destructor cannot safely be executed here.
        core::mem::forget(nested);
    }
}
