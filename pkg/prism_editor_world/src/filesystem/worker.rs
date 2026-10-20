use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard,
    },
    task::{Context, Poll, Waker},
    thread,
    time::Instant,
};

use super::{io_error, lock, FileCellIoError, Shared};

#[derive(Default)]
pub(super) struct Cancellation {
    cancelled: AtomicBool,
    commit: Mutex<()>,
    deadline: Option<Instant>,
}

impl Cancellation {
    pub(super) fn check(&self) -> Result<(), FileCellIoError> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(FileCellIoError::Cancelled)
        } else if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            Err(FileCellIoError::TimedOut)
        } else {
            Ok(())
        }
    }

    pub(super) fn commit_guard(&self) -> Result<MutexGuard<'_, ()>, FileCellIoError> {
        let guard = lock(&self.commit);
        self.check()?;
        Ok(guard)
    }

    fn cancel(&self) {
        let _guard = lock(&self.commit);
        self.cancelled.store(true, Ordering::Release);
    }
}

struct ResultState<T> {
    result: Option<Result<T, FileCellIoError>>,
    waker: Option<Waker>,
}

struct Job<T> {
    shared: Arc<Shared>,
    cancel: Cancellation,
    state: Mutex<ResultState<T>>,
    // Drop the stored result before releasing its byte reservation.
    _retained: Option<Arc<dyn Send + Sync>>,
}

impl<T> Drop for Job<T> {
    fn drop(&mut self) {
        self.shared.operations.fetch_sub(1, Ordering::AcqRel);
    }
}

struct WorkerFuture<T> {
    job: Arc<Job<T>>,
}

impl<T> Future for WorkerFuture<T> {
    type Output = Result<T, FileCellIoError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = lock(&self.job.state);
        if let Some(result) = state.result.take() {
            return Poll::Ready(result);
        }
        if state
            .waker
            .as_ref()
            .is_none_or(|waker| !waker.will_wake(cx.waker()))
        {
            state.waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl<T> Drop for WorkerFuture<T> {
    fn drop(&mut self) {
        self.job.cancel.cancel();
    }
}

pub(super) fn start<T: Send + 'static>(
    shared: Arc<Shared>,
    operation: impl FnOnce(&Cancellation) -> Result<T, FileCellIoError> + Send + 'static,
) -> Result<Pin<Box<dyn Future<Output = Result<T, FileCellIoError>> + Send>>, FileCellIoError> {
    start_until(shared, None, operation)
}

pub(super) fn start_until<T: Send + 'static>(
    shared: Arc<Shared>,
    deadline: Option<Instant>,
    operation: impl FnOnce(&Cancellation) -> Result<T, FileCellIoError> + Send + 'static,
) -> Result<Pin<Box<dyn Future<Output = Result<T, FileCellIoError>> + Send>>, FileCellIoError> {
    start_retained(shared, deadline, None, operation)
}

pub(super) fn start_retained<T: Send + 'static>(
    shared: Arc<Shared>,
    deadline: Option<Instant>,
    retained: Option<Arc<dyn Send + Sync>>,
    operation: impl FnOnce(&Cancellation) -> Result<T, FileCellIoError> + Send + 'static,
) -> Result<Pin<Box<dyn Future<Output = Result<T, FileCellIoError>> + Send>>, FileCellIoError> {
    shared
        .operations
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            (count < shared.limits.max_operations).then(|| count + 1)
        })
        .map_err(|_| FileCellIoError::Capacity)?;
    let job = Arc::new(Job {
        _retained: retained,
        shared,
        cancel: Cancellation {
            deadline,
            ..Cancellation::default()
        },
        state: Mutex::new(ResultState {
            result: None,
            waker: None,
        }),
    });
    let worker = job.clone();
    thread::Builder::new()
        .name("prism-cell-io".into())
        .spawn(move || {
            // Turn unexpected worker panics into terminal failures, not a forever
            // Pending future. Temporary/lock RAII still runs during unwinding.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker.cancel.check()?;
                operation(&worker.cancel)
            }))
            .unwrap_or_else(|_| {
                Err(io_error(
                    "cell worker",
                    std::io::Error::other("cell worker panicked"),
                ))
            });
            let waker = {
                let mut state = lock(&worker.state);
                state.result = Some(result);
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        })
        .map_err(|e| io_error("spawn cell worker", e))?;
    Ok(Box::pin(WorkerFuture { job }))
}
