//! Real monotonic CPU capture with bounded live frames and nested spans.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use crate::{
    validate_limit, validate_text, Frame, FrameId, ProfilerError, ProfilerLimits, Span, SpanId,
};

static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);

/// Opaque, process-unique capture generation; never reused, even across producers.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CaptureHandle(u64);

impl CaptureHandle {
    pub(crate) fn next() -> Result<Self, CaptureError> {
        NEXT_CAPTURE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(Self)
            .map_err(|_| CaptureError::Overflow)
    }
}

/// Opaque span identity fenced to its frame generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpanHandle {
    pub(crate) capture: CaptureHandle,
    pub(crate) index: usize,
}

/// Capture lifecycle, capacity, backend, or normalized model error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureError {
    /// A handle was cancelled, consumed, or belongs to another producer.
    StaleHandle,
    /// Frame/span operations must obey recording and nested LIFO order.
    InvalidState,
    /// A bounded recording/readback slot pool is full.
    AtCapacity,
    /// A producer configuration cannot be represented by the backend.
    InvalidConfig,
    /// The device did not enable the required timestamp capabilities.
    UnsupportedTimestamps,
    /// Hardware timestamps were reversed, invalid, or not representable.
    InvalidTimestamp,
    /// GPU polling failed. The caller may cancel and recreate the producer.
    GpuPoll,
    /// GPU mapping failed; the corresponding slot has been released.
    GpuMap,
    /// A generation, duration, or allocation-size calculation overflowed.
    Overflow,
    /// An existing profiler model contract was violated.
    Model(ProfilerError),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "profiler capture: {self:?}")
    }
}
impl std::error::Error for CaptureError {}
impl From<ProfilerError> for CaptureError {
    fn from(value: ProfilerError) -> Self {
        Self::Model(value)
    }
}

struct CpuSpan {
    name: String,
    lane: String,
    parent: Option<SpanId>,
    start_ns: u64,
    duration_ns: Option<u64>,
}
struct CpuFrame {
    id: FrameId,
    sequence: u64,
    started: Instant,
    timestamp_ns: u64,
    spans: Vec<CpuSpan>,
    stack: Vec<usize>,
}

/// CPU recorder using `Instant` rather than caller-provided timing values.
///
/// `max_active_frames` bounds all unfinished frames. Completed frames transfer
/// to the caller, which can insert them into an existing bounded `FrameStore`.
/// One frame has one LIFO span stack; use separate frames/producers for threads.
pub struct CpuProducer {
    origin: Instant,
    limits: ProfilerLimits,
    max_active_frames: usize,
    active: BTreeMap<CaptureHandle, CpuFrame>,
}

impl CpuProducer {
    /// Creates a CPU producer; a zero live-frame capacity is rejected.
    pub fn new(limits: ProfilerLimits, max_active_frames: usize) -> Result<Self, CaptureError> {
        if max_active_frames == 0 {
            return Err(CaptureError::InvalidConfig);
        }
        Ok(Self {
            origin: Instant::now(),
            limits,
            max_active_frames,
            active: BTreeMap::new(),
        })
    }

    /// Begins a real CPU frame with a timestamp relative to this producer's origin.
    pub fn begin_frame(
        &mut self,
        id: FrameId,
        sequence: u64,
    ) -> Result<CaptureHandle, CaptureError> {
        if self.active.len() >= self.max_active_frames {
            return Err(CaptureError::AtCapacity);
        }
        if self.active.values().any(|f| f.id == id) {
            return Err(ProfilerError::DuplicateId {
                resource: "active CPU frame",
                id: id.raw(),
            }
            .into());
        }
        let handle = CaptureHandle::next()?;
        let started = Instant::now();
        let timestamp_ns = nanos(started.duration_since(self.origin))?;
        self.active.insert(
            handle,
            CpuFrame {
                id,
                sequence,
                started,
                timestamp_ns,
                spans: Vec::new(),
                stack: Vec::new(),
            },
        );
        Ok(handle)
    }

    /// Starts a nested span; checks count, depth and text before retaining input.
    pub fn begin_span(
        &mut self,
        frame: CaptureHandle,
        name: &str,
        lane: &str,
    ) -> Result<SpanHandle, CaptureError> {
        let active = self
            .active
            .get_mut(&frame)
            .ok_or(CaptureError::StaleHandle)?;
        validate_limit(
            active.spans.len() + 1,
            self.limits.max_spans_per_frame,
            "CPU spans",
        )?;
        validate_limit(
            active.stack.len() + 1,
            self.limits.max_span_depth,
            "CPU span depth",
        )?;
        validate_text(name, "span name", &self.limits)?;
        validate_text(lane, "span lane", &self.limits)?;
        let index = active.spans.len();
        active.spans.push(CpuSpan {
            name: name.into(),
            lane: lane.into(),
            parent: active.stack.last().map(|i| SpanId::from_raw(*i as u64)),
            start_ns: nanos(active.started.elapsed())?,
            duration_ns: None,
        });
        active.stack.push(index);
        Ok(SpanHandle {
            capture: frame,
            index,
        })
    }

    /// Ends the top span using a fresh `Instant` sample.
    pub fn end_span(&mut self, span: SpanHandle) -> Result<(), CaptureError> {
        let active = self
            .active
            .get_mut(&span.capture)
            .ok_or(CaptureError::StaleHandle)?;
        if active.stack.last() != Some(&span.index) {
            return Err(CaptureError::InvalidState);
        }
        let end = nanos(active.started.elapsed())?;
        let data = &mut active.spans[span.index];
        data.duration_ns = Some(end - data.start_ns);
        active.stack.pop();
        Ok(())
    }

    /// Finishes only after all spans end. Returns a validated existing `Frame`.
    pub fn finish_frame(&mut self, handle: CaptureHandle) -> Result<Frame, CaptureError> {
        let active = self.active.get(&handle).ok_or(CaptureError::StaleHandle)?;
        if !active.stack.is_empty() {
            return Err(CaptureError::InvalidState);
        }
        let duration = nanos(active.started.elapsed())?;
        let spans = active
            .spans
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Span::new(
                    SpanId::from_raw(i as u64),
                    s.parent,
                    s.start_ns,
                    s.duration_ns.expect("closed span stack"),
                    &s.name,
                    &s.lane,
                )
            })
            .collect();
        let frame = Frame::try_new(
            active.id,
            active.sequence,
            active.timestamp_ns,
            duration,
            spans,
            &self.limits,
        )?;
        self.active.remove(&handle);
        Ok(frame)
    }

    /// Cancels a frame and all its spans; subsequent handle use is rejected.
    pub fn cancel(&mut self, handle: CaptureHandle) -> Result<(), CaptureError> {
        self.active
            .remove(&handle)
            .map(|_| ())
            .ok_or(CaptureError::StaleHandle)
    }

    /// Cancels every live CPU frame, fencing all issued span/frame handles.
    pub fn cancel_all(&mut self) {
        self.active.clear();
    }

    /// Returns the number of bounded live frame slots currently occupied.
    pub fn active_frames(&self) -> usize {
        self.active.len()
    }
}

fn nanos(duration: std::time::Duration) -> Result<u64, CaptureError> {
    duration
        .as_nanos()
        .try_into()
        .map_err(|_| CaptureError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameStore;

    #[test]
    fn instant_nested_capture_reaches_existing_store() {
        let limits = ProfilerLimits {
            max_frames: 1,
            ..ProfilerLimits::default()
        };
        let mut cpu = CpuProducer::new(limits, 1).unwrap();
        let handle = cpu.begin_frame(FrameId::from_raw(1), 7).unwrap();
        let root = cpu.begin_span(handle, "render", "cpu/main").unwrap();
        let child = cpu.begin_span(handle, "prepare", "cpu/main").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert_eq!(cpu.end_span(root), Err(CaptureError::InvalidState));
        assert_eq!(cpu.finish_frame(handle), Err(CaptureError::InvalidState));
        cpu.end_span(child).unwrap();
        cpu.end_span(root).unwrap();
        let frame = cpu.finish_frame(handle).unwrap();
        assert!(frame.spans().iter().all(|s| s.duration_ns() >= 2_000_000));
        assert_eq!(frame.span_tree().roots()[0].children().len(), 1);
        let mut store = FrameStore::new(limits);
        store.insert(frame.clone()).unwrap();
        assert!(store.insert(frame).is_err());
        assert_eq!(cpu.finish_frame(handle), Err(CaptureError::StaleHandle));
    }

    #[test]
    fn bounded_inputs_and_cancellation_fence_generations() {
        let limits = ProfilerLimits {
            max_spans_per_frame: 2,
            max_span_depth: 1,
            max_text_bytes: 4,
            ..ProfilerLimits::default()
        };
        let mut cpu = CpuProducer::new(limits, 1).unwrap();
        let old = cpu.begin_frame(FrameId::from_raw(0), 0).unwrap();
        assert_eq!(
            cpu.begin_frame(FrameId::from_raw(1), 1),
            Err(CaptureError::AtCapacity)
        );
        assert!(cpu.begin_span(old, "oversized", "cpu").is_err());
        let span = cpu.begin_span(old, "one", "cpu").unwrap();
        assert!(cpu.begin_span(old, "deep", "cpu").is_err());
        cpu.end_span(span).unwrap();
        let span = cpu.begin_span(old, "two", "cpu").unwrap();
        cpu.end_span(span).unwrap();
        assert!(cpu.begin_span(old, "full", "cpu").is_err());
        cpu.cancel(old).unwrap();
        let new = cpu.begin_frame(FrameId::from_raw(0), 0).unwrap();
        assert_ne!(new, old);
        assert_eq!(cpu.end_span(span), Err(CaptureError::StaleHandle));
        assert_eq!(cpu.cancel(old), Err(CaptureError::StaleHandle));
        let mut other = CpuProducer::new(limits, 1).unwrap();
        assert_eq!(other.cancel(new), Err(CaptureError::StaleHandle));
        cpu.cancel_all();
        assert_eq!(cpu.active_frames(), 0);
        assert_eq!(cpu.finish_frame(new), Err(CaptureError::StaleHandle));
    }
}
