//! Deterministic residency-budget decisions and synchronous/asynchronous range
//! I/O for runtime asset streaming.
//!
//! The controller remains the state and budget authority. [`StreamIoExecutor`]
//! is the deliberately small synchronous adapter that performs a real
//! [`crate::AssetSources::read_range`] and reports its result back to the
//! controller. [`AsyncStreamIoExecutor`] schedules the same operation on an
//! existing [`prism_tasks::TaskPool`] without holding the controller lock
//! during the blocking read. Neither adapter creates GPU resources or claims
//! DirectStorage support; GPU and staging values are budget-ledger entries
//! only.

use crate::AssetPath;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use prism_tasks::Priority;

mod async_io;
mod io;

pub use async_io::{AsyncStreamIoExecutor, AsyncStreamTask};
pub use io::{StreamIoError, StreamIoExecutor, SubmitAndExecuteError};

/// The three independently bounded residency pools tracked by the decision
/// layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BudgetKind {
    Cpu,
    Gpu,
    Staging,
}

/// A checked three-dimensional byte quantity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    pub cpu: u64,
    pub gpu: u64,
    pub staging: u64,
}

impl Residency {
    #[must_use]
    pub const fn new(cpu: u64, gpu: u64, staging: u64) -> Self {
        Self { cpu, gpu, staging }
    }

    fn checked_add(self, rhs: Self) -> Result<Self, StreamError> {
        Ok(Self {
            cpu: self
                .cpu
                .checked_add(rhs.cpu)
                .ok_or(StreamError::BudgetOverflow)?,
            gpu: self
                .gpu
                .checked_add(rhs.gpu)
                .ok_or(StreamError::BudgetOverflow)?,
            staging: self
                .staging
                .checked_add(rhs.staging)
                .ok_or(StreamError::BudgetOverflow)?,
        })
    }

    fn checked_sub(self, rhs: Self) -> Self {
        Self {
            cpu: self.cpu.saturating_sub(rhs.cpu),
            gpu: self.gpu.saturating_sub(rhs.gpu),
            staging: self.staging.saturating_sub(rhs.staging),
        }
    }

    fn exceeds(self, limit: Self) -> Option<BudgetKind> {
        if self.cpu > limit.cpu {
            Some(BudgetKind::Cpu)
        } else if self.gpu > limit.gpu {
            Some(BudgetKind::Gpu)
        } else if self.staging > limit.staging {
            Some(BudgetKind::Staging)
        } else {
            None
        }
    }
}

/// Hard CPU, GPU, and staging byte ceilings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyLimits {
    pub cpu: u64,
    pub gpu: u64,
    pub staging: u64,
}

impl ResidencyLimits {
    #[must_use]
    pub const fn new(cpu: u64, gpu: u64, staging: u64) -> Self {
        Self { cpu, gpu, staging }
    }
    fn as_residency(self) -> Residency {
        Residency::new(self.cpu, self.gpu, self.staging)
    }
}

/// A request that maps to the existing `AssetReader::read_range` contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamRequest {
    pub path: AssetPath,
    pub offset: u64,
    pub len: u64,
    pub residency: Residency,
    pub priority: Priority,
    pub importance: u32,
}

impl StreamRequest {
    pub fn read_range(&self) -> Result<(u64, u64), StreamError> {
        self.offset
            .checked_add(self.len)
            .ok_or(StreamError::RangeOverflow)
            .map(|_| (self.offset, self.len))
    }
}

/// Stable identity assigned on submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamRequestId(u64);

impl StreamRequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Explicit request lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamState {
    Pending,
    Loading,
    Resident,
    Failed,
    Cancelled,
    Evicting,
    Evicted,
}

/// Terminal failure reported by the external reader/decoder/upload path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamFailure {
    Read,
    Decode,
    External(u32),
}

/// Errors raised by request admission or state transitions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamError {
    RangeOverflow,
    BudgetOverflow,
    BudgetExceeded {
        kind: BudgetKind,
        requested: u64,
        limit: u64,
    },
    UnknownRequest(StreamRequestId),
    InvalidTransition {
        id: StreamRequestId,
        state: StreamState,
    },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RangeOverflow => f.write_str("stream request range overflows u64"),
            Self::BudgetOverflow => f.write_str("stream residency budget overflows u64"),
            Self::BudgetExceeded {
                kind,
                requested,
                limit,
            } => write!(
                f,
                "{kind:?} budget exceeded: requested {requested}, limit {limit}"
            ),
            Self::UnknownRequest(id) => write!(f, "unknown stream request {}", id.get()),
            Self::InvalidTransition { id, state } => write!(
                f,
                "invalid transition for request {} in {state:?}",
                id.get()
            ),
        }
    }
}

impl std::error::Error for StreamError {}

/// Errors raised while submitting a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    Invalid(StreamError),
    IdExhausted,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => error.fmt(f),
            Self::IdExhausted => f.write_str("stream request id space exhausted"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Priority hysteresis policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hysteresis {
    pub promote_after: u32,
    pub demote_after: u32,
}

impl Hysteresis {
    #[must_use]
    pub const fn new(promote_after: u32, demote_after: u32) -> Self {
        Self {
            promote_after: if promote_after == 0 { 1 } else { promote_after },
            demote_after: if demote_after == 0 { 1 } else { demote_after },
        }
    }
}

impl Default for Hysteresis {
    fn default() -> Self {
        Self::new(2, 2)
    }
}

#[derive(Debug)]
struct Record {
    request: StreamRequest,
    state: StreamState,
    sequence: u64,
    failure: Option<StreamFailure>,
    effective_priority: Priority,
    promotion_streak: u32,
    demotion_streak: u32,
}

/// Pure, deterministic streaming admission and residency decision state.
///
/// The controller owns no bytes or GPU resources. A caller performs the
/// external operation after `begin_load`, then reports its result through the
/// explicit transition methods.
pub struct StreamingController {
    limits: ResidencyLimits,
    used: Residency,
    hysteresis: Hysteresis,
    next_id: u64,
    next_sequence: u64,
    records: BTreeMap<StreamRequestId, Record>,
}

impl StreamingController {
    #[must_use]
    pub fn new(limits: ResidencyLimits) -> Self {
        Self::with_hysteresis(limits, Hysteresis::default())
    }

    #[must_use]
    pub fn with_hysteresis(limits: ResidencyLimits, hysteresis: Hysteresis) -> Self {
        Self {
            limits,
            used: Residency::default(),
            hysteresis,
            next_id: 0,
            next_sequence: 0,
            records: BTreeMap::new(),
        }
    }

    #[must_use]
    pub const fn limits(&self) -> ResidencyLimits {
        self.limits
    }

    #[must_use]
    pub const fn used(&self) -> Residency {
        self.used
    }

    pub fn submit(&mut self, request: StreamRequest) -> Result<StreamRequestId, SubmitError> {
        request.read_range().map_err(SubmitError::Invalid)?;
        let id = StreamRequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(SubmitError::IdExhausted)?;
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SubmitError::IdExhausted)?;
        self.records.insert(
            id,
            Record {
                effective_priority: request.priority,
                request,
                state: StreamState::Pending,
                sequence,
                failure: None,
                promotion_streak: 0,
                demotion_streak: 0,
            },
        );
        Ok(id)
    }

    #[must_use]
    pub fn next_pending(&self) -> Option<(StreamRequestId, &StreamRequest)> {
        self.records
            .iter()
            .filter(|(_, record)| record.state == StreamState::Pending)
            .max_by_key(|(id, record)| {
                (
                    record.effective_priority,
                    record.request.importance,
                    core::cmp::Reverse(record.sequence),
                    core::cmp::Reverse(id.0),
                )
            })
            .map(|(id, record)| (*id, &record.request))
    }

    /// Returns resident requests in deterministic eviction order.
    ///
    /// Lower effective priority and importance are evicted first. Ties use
    /// submission order, then the request id. The controller does not track
    /// wall-clock recency; callers that need an LRU policy must feed that
    /// policy through their own request priorities.
    #[must_use]
    pub fn eviction_candidates(&self) -> Vec<StreamRequestId> {
        let mut candidates = self
            .records
            .iter()
            .filter(|(_, record)| record.state == StreamState::Resident)
            .map(|(id, record)| {
                (
                    record.effective_priority,
                    record.request.importance,
                    record.sequence,
                    id.0,
                    *id,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| (candidate.0, candidate.1, candidate.2, candidate.3));
        candidates
            .into_iter()
            .map(|candidate| candidate.4)
            .collect()
    }

    pub fn begin_load(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        if record.state != StreamState::Pending {
            return Err(StreamError::InvalidTransition {
                id,
                state: record.state,
            });
        }
        let requested = self.used.checked_add(record.request.residency)?;
        if let Some(kind) = requested.exceeds(self.limits.as_residency()) {
            let (requested, limit) = match kind {
                BudgetKind::Cpu => (requested.cpu, self.limits.cpu),
                BudgetKind::Gpu => (requested.gpu, self.limits.gpu),
                BudgetKind::Staging => (requested.staging, self.limits.staging),
            };
            return Err(StreamError::BudgetExceeded {
                kind,
                requested,
                limit,
            });
        }
        self.used = requested;
        self.record_mut(id)?.state = StreamState::Loading;
        Ok(())
    }

    pub fn complete_load(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        if record.state != StreamState::Loading {
            return Err(StreamError::InvalidTransition {
                id,
                state: record.state,
            });
        }
        self.record_mut(id)?.state = StreamState::Resident;
        Ok(())
    }

    pub fn fail(&mut self, id: StreamRequestId, failure: StreamFailure) -> Result<(), StreamError> {
        let record = self.record(id)?;
        match record.state {
            StreamState::Pending => {}
            StreamState::Loading => self.used = self.used.checked_sub(record.request.residency),
            state => return Err(StreamError::InvalidTransition { id, state }),
        }
        let record = self.record_mut(id)?;
        record.state = StreamState::Failed;
        record.failure = Some(failure);
        Ok(())
    }

    pub fn cancel(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        match record.state {
            StreamState::Pending => {}
            StreamState::Loading => self.used = self.used.checked_sub(record.request.residency),
            state => return Err(StreamError::InvalidTransition { id, state }),
        }
        self.record_mut(id)?.state = StreamState::Cancelled;
        Ok(())
    }

    pub fn begin_evict(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        if record.state != StreamState::Resident {
            return Err(StreamError::InvalidTransition {
                id,
                state: record.state,
            });
        }
        self.record_mut(id)?.state = StreamState::Evicting;
        Ok(())
    }

    pub fn finish_evict(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        if record.state != StreamState::Evicting {
            return Err(StreamError::InvalidTransition {
                id,
                state: record.state,
            });
        }
        self.used = self.used.checked_sub(record.request.residency);
        self.record_mut(id)?.state = StreamState::Evicted;
        Ok(())
    }

    pub fn cancel_evict(&mut self, id: StreamRequestId) -> Result<(), StreamError> {
        let record = self.record(id)?;
        if record.state != StreamState::Evicting {
            return Err(StreamError::InvalidTransition {
                id,
                state: record.state,
            });
        }
        self.record_mut(id)?.state = StreamState::Resident;
        Ok(())
    }

    pub fn observe_priority(
        &mut self,
        id: StreamRequestId,
        observed: Priority,
    ) -> Result<Priority, StreamError> {
        let hysteresis = self.hysteresis;
        let record = self.record_mut(id)?;
        if observed > record.effective_priority {
            record.promotion_streak = record.promotion_streak.saturating_add(1);
            record.demotion_streak = 0;
            if record.promotion_streak >= hysteresis.promote_after {
                record.effective_priority = observed;
                record.promotion_streak = 0;
            }
        } else if observed < record.effective_priority {
            record.demotion_streak = record.demotion_streak.saturating_add(1);
            record.promotion_streak = 0;
            if record.demotion_streak >= hysteresis.demote_after {
                record.effective_priority = observed;
                record.demotion_streak = 0;
            }
        } else {
            record.promotion_streak = 0;
            record.demotion_streak = 0;
        }
        Ok(record.effective_priority)
    }

    #[must_use]
    pub fn state(&self, id: StreamRequestId) -> Option<StreamState> {
        self.records.get(&id).map(|record| record.state)
    }

    #[must_use]
    pub fn failure(&self, id: StreamRequestId) -> Option<&StreamFailure> {
        self.records
            .get(&id)
            .and_then(|record| record.failure.as_ref())
    }

    #[must_use]
    pub fn request(&self, id: StreamRequestId) -> Option<&StreamRequest> {
        self.records.get(&id).map(|record| &record.request)
    }

    fn record(&self, id: StreamRequestId) -> Result<&Record, StreamError> {
        self.records.get(&id).ok_or(StreamError::UnknownRequest(id))
    }
    fn record_mut(&mut self, id: StreamRequestId) -> Result<&mut Record, StreamError> {
        self.records
            .get_mut(&id)
            .ok_or(StreamError::UnknownRequest(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(name: &str, priority: Priority, importance: u32, bytes: u64) -> StreamRequest {
        StreamRequest {
            path: AssetPath::from(name),
            offset: 0,
            len: bytes,
            residency: Residency::new(bytes, bytes, bytes),
            priority,
            importance,
        }
    }

    #[test]
    fn priority_order_is_stable_and_deterministic() {
        let mut controller = StreamingController::new(ResidencyLimits::new(100, 100, 100));
        let low = controller
            .submit(request("low.bin", Priority::Low, 99, 1))
            .unwrap();
        let first = controller
            .submit(request("first.bin", Priority::Normal, 1, 1))
            .unwrap();
        let second = controller
            .submit(request("second.bin", Priority::Normal, 1, 1))
            .unwrap();
        assert_eq!(controller.next_pending().unwrap().0, first);
        controller.begin_load(first).unwrap();
        assert_eq!(controller.next_pending().unwrap().0, second);
        controller.begin_load(second).unwrap();
        assert_eq!(controller.next_pending().unwrap().0, low);
    }

    #[test]
    fn hysteresis_requires_consecutive_observations() {
        let mut controller = StreamingController::with_hysteresis(
            ResidencyLimits::new(10, 10, 10),
            Hysteresis::new(2, 2),
        );
        let id = controller
            .submit(request("hero.bin", Priority::Low, 0, 1))
            .unwrap();
        assert_eq!(
            controller.observe_priority(id, Priority::High).unwrap(),
            Priority::Low
        );
        assert_eq!(
            controller.observe_priority(id, Priority::Low).unwrap(),
            Priority::Low
        );
        assert_eq!(
            controller.observe_priority(id, Priority::High).unwrap(),
            Priority::Low
        );
        assert_eq!(
            controller.observe_priority(id, Priority::High).unwrap(),
            Priority::High
        );
    }

    #[test]
    fn budget_overflow_and_pool_exhaustion_are_reported() {
        let mut controller = StreamingController::new(ResidencyLimits::new(u64::MAX, 10, 10));
        let overflow = controller
            .submit(StreamRequest {
                path: AssetPath::from("overflow.bin"),
                offset: 0,
                len: 1,
                residency: Residency::new(0, 10, 0),
                priority: Priority::Normal,
                importance: 0,
            })
            .unwrap();
        controller.begin_load(overflow).unwrap();
        let too_much = controller
            .submit(request("gpu.bin", Priority::Normal, 0, 11))
            .unwrap();
        assert_eq!(
            controller.begin_load(too_much),
            Err(StreamError::BudgetExceeded {
                kind: BudgetKind::Gpu,
                requested: 21,
                limit: 10
            })
        );
        let mut overflow_controller =
            StreamingController::new(ResidencyLimits::new(u64::MAX, 10, 10));
        let max_cpu = overflow_controller
            .submit(StreamRequest {
                path: AssetPath::from("max-cpu.bin"),
                offset: 0,
                len: 1,
                residency: Residency::new(u64::MAX, 0, 0),
                priority: Priority::Normal,
                importance: 0,
            })
            .unwrap();
        overflow_controller.begin_load(max_cpu).unwrap();
        let arithmetic_overflow = overflow_controller
            .submit(request("arithmetic.bin", Priority::Normal, 0, 1))
            .unwrap();
        assert_eq!(
            overflow_controller.begin_load(arithmetic_overflow),
            Err(StreamError::BudgetOverflow)
        );
    }

    #[test]
    fn failure_cancellation_and_eviction_release_residency() {
        let mut controller = StreamingController::new(ResidencyLimits::new(10, 10, 10));
        let failed = controller
            .submit(request("failed.bin", Priority::Normal, 0, 3))
            .unwrap();
        controller.begin_load(failed).unwrap();
        controller.fail(failed, StreamFailure::Read).unwrap();
        assert_eq!(controller.used(), Residency::default());
        assert_eq!(controller.failure(failed), Some(&StreamFailure::Read));
        let cancelled = controller
            .submit(request("cancelled.bin", Priority::Normal, 0, 2))
            .unwrap();
        controller.begin_load(cancelled).unwrap();
        controller.cancel(cancelled).unwrap();
        assert_eq!(controller.used(), Residency::default());
        let resident = controller
            .submit(request("resident.bin", Priority::Normal, 0, 4))
            .unwrap();
        controller.begin_load(resident).unwrap();
        controller.complete_load(resident).unwrap();
        controller.begin_evict(resident).unwrap();
        assert_eq!(controller.used(), Residency::new(4, 4, 4));
        controller.finish_evict(resident).unwrap();
        assert_eq!(controller.used(), Residency::default());
        assert_eq!(controller.state(resident), Some(StreamState::Evicted));
    }

    #[test]
    fn eviction_candidates_are_stable_and_ignore_non_resident_requests() {
        let mut controller = StreamingController::new(ResidencyLimits::new(20, 20, 20));
        let important = controller
            .submit(request("important.bin", Priority::High, 10, 1))
            .unwrap();
        let old_low = controller
            .submit(request("old-low.bin", Priority::Low, 1, 1))
            .unwrap();
        let new_low = controller
            .submit(request("new-low.bin", Priority::Low, 1, 1))
            .unwrap();
        let pending = controller
            .submit(request("pending.bin", Priority::Low, 0, 1))
            .unwrap();
        for id in [important, old_low, new_low] {
            controller.begin_load(id).unwrap();
            controller.complete_load(id).unwrap();
        }
        assert_eq!(
            controller.eviction_candidates(),
            vec![old_low, new_low, important]
        );
        assert!(!controller.eviction_candidates().contains(&pending));
    }

    #[test]
    fn range_overflow_is_rejected_before_reader_call() {
        let mut controller = StreamingController::new(ResidencyLimits::new(1, 1, 1));
        assert_eq!(
            controller.submit(StreamRequest {
                path: AssetPath::from("bad.bin"),
                offset: u64::MAX,
                len: 1,
                residency: Residency::default(),
                priority: Priority::Normal,
                importance: 0
            }),
            Err(SubmitError::Invalid(StreamError::RangeOverflow))
        );
    }
}
