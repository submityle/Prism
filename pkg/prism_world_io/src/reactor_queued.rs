//! Bounded admission separate from worker-owned native streams.

use super::*;

/// Native reads with bounded asynchronous capacity and declared-byte waiting.
///
/// All wrappers of a provider share `capacity` waiting/admission records and
/// `capacity` physical slots. Unpolled starts and unconsumed grants count too.
/// At most `2 * capacity` unfinished operations can exist, with at most
/// `capacity * max_chunk_size` provider-owned payload bytes (excluding returned
/// chunks and allocator overhead). Waiting requests own neither files nor data.
///
/// Poll registration is FIFO, including byte-quota waits: an oversized head
/// waits rather than being bypassed. A request larger than either byte limit is
/// rejected immediately. Grants reserve bytes and slots atomically before waking
/// and remain charged until polled or canceled. No self-wakes or timers occur.
/// Direct clones share physical limits but may bypass ungranted queued requests.
/// An exclusive scheduler whose in-flight limit is at most `capacity` can replace
/// canceled streams while their native cleanup is still pending. Multiple
/// schedulers must budget their aggregate admissions; excess starts are rejected.
#[derive(Clone)]
pub struct QueuedReactorFileProvider<'services> {
    pub(super) provider: ReactorFileProvider<'services>,
}

/// A bounded waiting admission, followed by an acknowledgement-driven stream.
/// Drop releases an inert grant immediately or requests worker cleanup for an
/// active stream, without waiting for native completion or service teardown.
pub struct QueuedReactorFileReadOperation<'services> {
    provider: ReactorFileProvider<'services>,
    control: Arc<Control>,
    request_id: RequestId,
    admission: Option<usize>,
    active: Option<ReactorFileReadOperation<'services>>,
}

impl QueuedReactorFileProvider<'_> {
    /// Configured path; no filesystem operation is performed.
    pub fn path(&self) -> &Path {
        self.provider.path()
    }

    /// Immutable limits shared by every clone and wrapper.
    pub fn config(&self) -> ReactorFileConfig {
        self.provider.config()
    }

    /// Physical slots, including granted admissions and cancellation cleanup.
    pub fn in_flight(&self) -> usize {
        self.provider.in_flight()
    }

    /// Declared bytes held by physical slots, including inert grants.
    pub fn reserved_bytes(&self) -> u64 {
        self.provider.reserved_bytes()
    }

    /// Waiting records, including unpolled starts and unconsumed grants.
    pub fn pending_admissions(&self) -> usize {
        lock(&self.provider.shared.state)
            .admissions
            .iter()
            .flatten()
            .count()
    }
}

impl<'services> ReadProvider for QueuedReactorFileProvider<'services> {
    type Operation = QueuedReactorFileReadOperation<'services>;

    fn start(
        &self,
        request: &ReadRequest,
        request_id: RequestId,
    ) -> Result<Self::Operation, IoError> {
        self.provider.validate_request(request)?;
        let control = Arc::new(Control(AtomicBool::new(false)));
        let mut state = lock(&self.provider.shared.state);
        let index =
            state
                .admissions
                .iter()
                .position(Option::is_none)
                .ok_or(IoError::Backpressure {
                    capacity: self.config().capacity,
                })?;
        state.admissions[index] = Some(Admission {
            control: Arc::clone(&control),
            work: Work {
                request: request.clone(),
                request_id,
                chunk_size: request.chunk_size().min(self.config().max_chunk_size),
                delivered: 0,
                file: None,
            },
            slot: None,
            waker: None,
        });
        drop(state);
        Ok(QueuedReactorFileReadOperation {
            provider: self.provider.clone(),
            control,
            request_id,
            admission: Some(index),
            active: None,
        })
    }
}

impl ChunkedRead for QueuedReactorFileReadOperation<'_> {
    fn poll_chunk(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if let Some(active) = &mut self.active {
            return active.poll_chunk(context);
        }
        if self.control.0.load(Ordering::Acquire) {
            return Poll::Ready(Err(IoError::Cancelled {
                request_id: self.request_id,
            }));
        }
        let new_waker = context.waker().clone();
        let mut state = lock(&self.provider.shared.state);
        let index = self.admission.expect("live admission");
        let entry = state.admissions[index]
            .as_mut()
            .expect("registered admission");
        let old_waker = entry.waker.replace(new_waker);
        if entry.slot.is_none() && old_waker.is_none() {
            state.waiting.push_back(index);
        }
        let available = state.grant_waiters(self.provider.config(), Some(index));
        let slot = state.admissions[index].as_ref().unwrap().slot;
        let Some(slot) = slot else {
            drop(state);
            discard(old_waker);
            notify_all(available);
            return Poll::Pending;
        };
        let mut admission = state.admissions[index].take().unwrap();
        self.admission = None;
        self.active = Some(ReactorFileReadOperation {
            provider: self.provider.clone(),
            control: Arc::clone(&self.control),
            request_id: self.request_id,
            slot,
            finished: false,
        });
        drop(state);
        // Enqueue before foreign callbacks. A grant consumed in this very poll
        // does not self-wake; progress comes from worker completion instead.
        self.provider
            .dispatch(slot, Some(admission.work), Phase::Busy);
        discard(old_waker);
        discard(admission.waker.take());
        notify_all(available);
        self.active.as_mut().unwrap().poll_chunk(context)
    }

    fn cancel(&mut self) {
        if let Some(active) = &mut self.active {
            active.cancel();
            return;
        }
        let Some(index) = self.admission.take() else {
            return;
        };
        self.control.0.store(true, Ordering::Release);
        let mut state = lock(&self.provider.shared.state);
        let mut admission = state.admissions[index]
            .take()
            .expect("registered admission");
        state.waiting.retain(|waiting| *waiting != index);
        // No dispatched job, file, future or payload exists for this grant.
        let available = if let Some(slot) = admission.slot {
            state.release(slot, self.provider.config())
        } else {
            // Removing a byte-quota head may let multiple smaller waiters run.
            state.grant_waiters(self.provider.config(), None)
        };
        drop(state);
        notify(admission.waker.take());
        notify_all(available);
    }
}

impl Drop for QueuedReactorFileReadOperation<'_> {
    fn drop(&mut self) {
        self.cancel();
    }
}
