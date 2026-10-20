//! Real wgpu timestamp capture with a fixed query/readback slot pool.
//!
//! The producer owns encoders until submission. Cancelled submitted slots stay
//! quarantined until their one mapping callback completes.

use crate::capture::{CaptureError, CaptureHandle};
use crate::{validate_limit, validate_text, Frame, FrameId, ProfilerLimits, Span, SpanId};
use std::sync::mpsc::{self, Receiver, TryRecvError};

/// Timestamp features the application must enable when requesting its device.
pub const REQUIRED_TIMESTAMP_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

/// Bounds on GPU recording and asynchronous readback resources.
#[derive(Clone, Copy, Debug)]
pub struct GpuCaptureConfig {
    /// Total recording and pending slots, including cancelled readbacks draining.
    pub max_in_flight_frames: usize,
    /// Instrumented render/compute passes per frame (two queries each).
    pub max_spans_per_frame: usize,
}
impl Default for GpuCaptureConfig {
    fn default() -> Self {
        Self {
            max_in_flight_frames: 3,
            max_spans_per_frame: 128,
        }
    }
}
struct GpuSpan {
    name: String,
}
struct Recording {
    handle: CaptureHandle,
    id: FrameId,
    sequence: u64,
    spans: Vec<GpuSpan>,
}
enum State {
    Idle,
    Recording {
        encoder: wgpu::CommandEncoder,
        frame: Recording,
    },
    Pending {
        frame: Recording,
        receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
        cancelled: bool,
    },
}
struct Slot {
    query: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: State,
}
/// A completed readback associated with its capture generation.
pub struct GpuCaptureResult {
    /// Submitted generation; cancelled generations never appear here.
    pub handle: CaptureHandle,
    /// Validated frame in the GPU clock domain, or explicit readback/data error.
    pub frame: Result<Frame, CaptureError>,
}
/// Bounded GPU producer for a device and its matching application queue.
///
/// Each slot owns its query set, resolve/staging buffers and encoder. Timing
/// values come exclusively from GPU queries. GPU time is not `Instant` time;
/// keep CPU and GPU frames in separate stores unless externally calibrated.
pub struct GpuProducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: ProfilerLimits,
    config: GpuCaptureConfig,
    period_ns: f64,
    slots: Vec<Slot>,
}
impl GpuProducer {
    /// Allocates the slot pool, rejecting disabled timestamp capabilities.
    ///
    /// `queue` must belong to `device`. GPU span count must fit both the model
    /// limit and wgpu's query-set maximum (2048 passes).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: ProfilerLimits,
        config: GpuCaptureConfig,
    ) -> Result<Self, CaptureError> {
        if !device.features().contains(REQUIRED_TIMESTAMP_FEATURES) {
            return Err(CaptureError::UnsupportedTimestamps);
        }
        let period_ns = f64::from(queue.get_timestamp_period());
        if !period_ns.is_finite() || period_ns <= 0.0 {
            return Err(CaptureError::UnsupportedTimestamps);
        }
        let query_count = config
            .max_spans_per_frame
            .checked_mul(2)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(CaptureError::InvalidConfig)?;
        if config.max_in_flight_frames == 0
            || config.max_spans_per_frame == 0
            || config.max_spans_per_frame > limits.max_spans_per_frame
            || query_count > wgpu::QUERY_SET_MAX_QUERIES
        {
            return Err(CaptureError::InvalidConfig);
        }
        let size = u64::from(query_count) * 8;
        if size > device.limits().max_buffer_size {
            return Err(CaptureError::InvalidConfig);
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.max_in_flight_frames)
            .map_err(|_| CaptureError::InvalidConfig)?;
        for _ in 0..config.max_in_flight_frames {
            slots.push(Slot {
                query: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("editor profiler timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: query_count,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("editor profiler resolve"),
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("editor profiler readback"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: State::Idle,
            });
        }
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            limits,
            config,
            period_ns,
            slots,
        })
    }

    /// Begins a frame; timestamps will be written by its instrumented passes.
    pub fn begin_frame(
        &mut self,
        id: FrameId,
        sequence: u64,
    ) -> Result<CaptureHandle, CaptureError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| matches!(s.state, State::Idle))
            .ok_or(CaptureError::AtCapacity)?;
        let handle = CaptureHandle::next()?;
        // Some Metal passes with no draw/dispatch omit timestamp writes, and
        // Metal's reset_queries is a no-op. A new generation must therefore
        // get fresh query storage rather than exposing previous samples.
        slot.query = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("editor profiler generation timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: slot.query.count(),
        });
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("editor profiler frame"),
            });
        slot.state = State::Recording {
            encoder,
            frame: Recording {
                handle,
                id,
                sequence,
                spans: Vec::new(),
            },
        };
        Ok(handle)
    }
    /// Borrows the owned encoder for copies and other uninstrumented work.
    ///
    /// End all borrowed passes before other producer calls; do not use pass
    /// `forget_lifetime`, following wgpu's encoder exclusivity contract.
    pub fn encoder(
        &mut self,
        handle: CaptureHandle,
    ) -> Result<&mut wgpu::CommandEncoder, CaptureError> {
        match &mut self.slot_mut(handle)?.state {
            State::Recording { encoder, .. } => Ok(encoder),
            _ => Err(CaptureError::InvalidState),
        }
    }
    /// Begins an actual render pass with hardware start/end timestamp writes.
    ///
    /// Drop the returned pass to end it. Existing descriptor timestamp writes
    /// are rejected rather than overwritten. GPU spans are flat pass intervals;
    /// the frame duration is the envelope of measured passes, not queue latency.
    pub fn begin_render_pass(
        &mut self,
        handle: CaptureHandle,
        name: &str,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> Result<wgpu::RenderPass<'_>, CaptureError> {
        if desc.timestamp_writes.is_some() {
            return Err(CaptureError::InvalidState);
        }
        let (slot, index) = self.reserve_pass(handle, name)?;
        let State::Recording { encoder, .. } = &mut slot.state else {
            unreachable!();
        };
        let desc = wgpu::RenderPassDescriptor {
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: &slot.query,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: Some(index + 1),
            }),
            ..desc.clone()
        };
        Ok(encoder.begin_render_pass(&desc))
    }
    /// Begins an actual compute pass with hardware start/end timestamp writes.
    ///
    /// The device must support `TIMESTAMP_QUERY`. No inside-encoder timestamp
    /// feature or artificial marker workload is required. Empty passes may be
    /// omitted by hardware; unresolved all-zero query pairs return an error.
    pub fn begin_compute_pass(
        &mut self,
        handle: CaptureHandle,
        name: &str,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> Result<wgpu::ComputePass<'_>, CaptureError> {
        if desc.timestamp_writes.is_some() {
            return Err(CaptureError::InvalidState);
        }
        let (slot, index) = self.reserve_pass(handle, name)?;
        let State::Recording { encoder, .. } = &mut slot.state else {
            unreachable!();
        };
        Ok(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: desc.label,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &slot.query,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: Some(index + 1),
            }),
        }))
    }
    fn reserve_pass(
        &mut self,
        handle: CaptureHandle,
        name: &str,
    ) -> Result<(&mut Slot, u32), CaptureError> {
        let config = self.config;
        let limits = self.limits;
        let slot = self.slot_mut(handle)?;
        let State::Recording { frame, .. } = &mut slot.state else {
            return Err(CaptureError::InvalidState);
        };
        validate_limit(
            frame.spans.len() + 1,
            config.max_spans_per_frame,
            "GPU passes",
        )?;
        validate_limit(1, limits.max_span_depth, "GPU span depth")?;
        validate_text(name, "span name", &limits)?;
        validate_text("gpu", "span lane", &limits)?;
        let index = 2 * frame.spans.len() as u32;
        frame.spans.push(GpuSpan { name: name.into() });
        Ok((slot, index))
    }
    /// Resolves/copies real pass queries, submits and starts readback mapping.
    ///
    /// Returns the application's real wgpu submission handle. Call `poll`
    /// regularly, including after cancellation, to release readback slots.
    pub fn submit_frame(
        &mut self,
        handle: CaptureHandle,
    ) -> Result<wgpu::SubmissionIndex, CaptureError> {
        let queue = self.queue.clone();
        let slot = self.slot_mut(handle)?;
        match &slot.state {
            State::Recording { frame, .. } if !frame.spans.is_empty() => {}
            _ => return Err(CaptureError::InvalidState),
        }
        let State::Recording { mut encoder, frame } =
            std::mem::replace(&mut slot.state, State::Idle)
        else {
            unreachable!();
        };
        let count = 2 * frame.spans.len() as u32;
        encoder.resolve_query_set(&slot.query, 0..count, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, u64::from(count) * 8);
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        slot.readback
            .slice(..u64::from(count) * 8)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.try_send(result);
            });
        slot.state = State::Pending {
            frame,
            receiver,
            cancelled: false,
        };
        Ok(submission)
    }
    /// Nonblocking device poll and bounded collection of completed readbacks.
    ///
    /// Cancelled slots drain without output. Mapping errors release the slot
    /// and are returned per capture. Poll errors never imply frame completion.
    pub fn poll(&mut self) -> Result<Vec<GpuCaptureResult>, CaptureError> {
        // Native wgpu can panic on fatal device polling errors. Keep that
        // failure explicit at the producer boundary as well as PollError.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.poll(wgpu::PollType::Poll)
        }))
        .map_err(|_| CaptureError::GpuPoll)?
        .map_err(|_| CaptureError::GpuPoll)?;
        let mut completed = Vec::new();
        for slot in &mut self.slots {
            let State::Pending { receiver, .. } = &slot.state else {
                continue;
            };
            let result = match receiver.try_recv() {
                Ok(result) => result.map_err(|_| CaptureError::GpuMap),
                Err(TryRecvError::Disconnected) => Err(CaptureError::GpuMap),
                Err(TryRecvError::Empty) => continue,
            };
            let State::Pending {
                frame, cancelled, ..
            } = std::mem::replace(&mut slot.state, State::Idle)
            else {
                unreachable!();
            };
            if !cancelled {
                let frame_result =
                    result.and_then(|()| read_frame(slot, &frame, self.period_ns, &self.limits));
                completed.push(GpuCaptureResult {
                    handle: frame.handle,
                    frame: frame_result,
                });
            }
            slot.readback.unmap();
        }
        Ok(completed)
    }
    /// Cancels recording immediately or quarantines a submitted readback.
    ///
    /// Submitted slots remain occupied until `poll` observes their callbacks.
    /// Repeated cancellation and stale/cross-producer handles fail.
    pub fn cancel(&mut self, handle: CaptureHandle) -> Result<(), CaptureError> {
        let slot = self.slot_mut(handle)?;
        match &mut slot.state {
            State::Recording { .. } => slot.state = State::Idle,
            State::Pending { cancelled, .. } => *cancelled = true,
            State::Idle => return Err(CaptureError::StaleHandle),
        }
        Ok(())
    }
    /// Cancels every live generation; submitted slots still require draining.
    pub fn cancel_all(&mut self) {
        for slot in &mut self.slots {
            match &mut slot.state {
                State::Recording { .. } => slot.state = State::Idle,
                State::Pending { cancelled, .. } => *cancelled = true,
                State::Idle => {}
            }
        }
    }
    /// Number of occupied slots, including cancelled readbacks still draining.
    pub fn occupied_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s.state, State::Idle))
            .count()
    }
    fn slot_mut(&mut self, handle: CaptureHandle) -> Result<&mut Slot, CaptureError> {
        self.slots
            .iter_mut()
            .find(|s| match &s.state {
                State::Recording { frame, .. } => frame.handle == handle,
                State::Pending {
                    frame, cancelled, ..
                } => frame.handle == handle && !cancelled,
                State::Idle => false,
            })
            .ok_or(CaptureError::StaleHandle)
    }
}

impl Drop for GpuProducer {
    fn drop(&mut self) {
        // Late callbacks only own a bounded sender, no producer/GPU state.
        for slot in &self.slots {
            if matches!(slot.state, State::Pending { .. }) {
                slot.readback.unmap();
            }
        }
    }
}
fn ticks_ns(ticks: u64, period: f64) -> Result<u64, CaptureError> {
    let value = ticks as f64 * period;
    if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
        return Err(CaptureError::InvalidTimestamp);
    }
    Ok(value.round() as u64)
}
fn read_frame(
    slot: &Slot,
    frame: &Recording,
    period: f64,
    limits: &ProfilerLimits,
) -> Result<Frame, CaptureError> {
    let count = 2 * frame.spans.len();
    let view = slot
        .readback
        .slice(..count as u64 * 8)
        .get_mapped_range()
        .map_err(|_| CaptureError::GpuMap)?;
    let ticks: Vec<u64> = view
        .chunks_exact(8)
        .map(|bytes| u64::from_ne_bytes(bytes.try_into().expect("eight-byte chunk")))
        .collect();
    let base = ticks
        .iter()
        .step_by(2)
        .copied()
        .min()
        .ok_or(CaptureError::InvalidTimestamp)?;
    let last = ticks
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .max()
        .ok_or(CaptureError::InvalidTimestamp)?;
    let duration_ticks = last
        .checked_sub(base)
        .ok_or(CaptureError::InvalidTimestamp)?;
    let duration = ticks_ns(duration_ticks, period)?;
    let mut spans = Vec::with_capacity(frame.spans.len());
    for (index, span) in frame.spans.iter().enumerate() {
        if ticks[index * 2] == 0 && ticks[index * 2 + 1] == 0 {
            return Err(CaptureError::InvalidTimestamp);
        }
        let start = ticks[index * 2]
            .checked_sub(base)
            .ok_or(CaptureError::InvalidTimestamp)?;
        let end = ticks[index * 2 + 1]
            .checked_sub(base)
            .ok_or(CaptureError::InvalidTimestamp)?;
        if end < start || end > duration_ticks {
            return Err(CaptureError::InvalidTimestamp);
        }
        // Convert endpoints before subtracting, preserving nested containment
        // even when the device period is a fractional nanosecond.
        let start_ns = ticks_ns(start, period)?;
        let end_ns = ticks_ns(end, period)?;
        spans.push(Span::new(
            SpanId::from_raw(index as u64),
            None,
            start_ns,
            end_ns - start_ns,
            &span.name,
            "gpu",
        ));
    }
    Ok(Frame::try_new(
        frame.id,
        frame.sequence,
        ticks_ns(base, period)?,
        duration,
        spans,
        limits,
    )?)
}
