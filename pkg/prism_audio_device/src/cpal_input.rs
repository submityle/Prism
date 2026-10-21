//! Live CPAL input capture and pollable input-device catalog.
//!
//! CPAL does not expose portable OS push notifications for device-route
//! changes. [`CpalInputCatalog::poll_changes`] therefore re-enumerates input
//! endpoints when requested and reports added/removed endpoints plus a default
//! route change relative to the previous successful poll. Stream callbacks only
//! set a per-generation fault latch and append bounded diagnostic records to a
//! lock-free event log; reopen recovery is explicit
//! and bounded on the control thread through [`CpalInput::recover`].

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use crossbeam_queue::ArrayQueue;

use crate::backend::DeviceState;
use crate::capture::CaptureSink;
use crate::error::DeviceError;

const DEFAULT_EVENT_LOG_CAPACITY: usize = 64;

/// Metadata for one enumerated CPAL input endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDeviceInfo {
    /// Stable CPAL host/device identifier suitable for catalog comparison.
    pub id: String,
    /// Human-readable endpoint name, or `"unknown"` if unavailable.
    pub name: String,
    /// Whether this endpoint was the host's default input at enumeration time.
    pub is_default: bool,
}

/// The default input route changed between two successful catalog polls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultInputChange {
    /// Previous default endpoint ID, or `None` if no input was available.
    pub previous: Option<String>,
    /// Current default endpoint ID, or `None` if no input is available.
    pub current: Option<String>,
}

/// A complete input catalog delta observed by one successful poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputCatalogChange {
    /// Endpoints present in the new snapshot but not the previous snapshot.
    pub added: Vec<InputDeviceInfo>,
    /// Endpoints present in the previous snapshot but not the new snapshot.
    pub removed: Vec<InputDeviceInfo>,
    /// Default route change, if the default endpoint changed.
    pub default: Option<DefaultInputChange>,
}

/// Snapshot and poller for CPAL input endpoints.
///
/// Polling is deterministic: each returned device list is sorted by CPAL ID,
/// and the first successful poll establishes a baseline without emitting a
/// route-change event.
pub struct CpalInputCatalog {
    host: cpal::Host,
    previous_default: Option<Option<String>>,
    previous_devices: Option<Vec<InputDeviceInfo>>,
}

impl Default for CpalInputCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl CpalInputCatalog {
    /// Creates a catalog for CPAL's platform default host.
    #[must_use]
    pub fn new() -> Self {
        Self {
            host: cpal::default_host(),
            previous_default: None,
            previous_devices: None,
        }
    }

    /// Enumerates input-capable endpoints in stable ID order.
    pub fn devices(&self) -> Result<Vec<InputDeviceInfo>, DeviceError> {
        self.enumerate().map(|(devices, _)| devices)
    }

    fn enumerate(&self) -> Result<(Vec<InputDeviceInfo>, Option<String>), DeviceError> {
        let default_id = self
            .host
            .default_input_device()
            .map(|device| device.id().map(|id| id.to_string()))
            .transpose()
            .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
        let devices = self
            .host
            .input_devices()
            .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
        let mut result = Vec::new();
        for device in devices {
            let id = device
                .id()
                .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?
                .to_string();
            let name = device
                .description()
                .map(|description| description.name().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            result.push(InputDeviceInfo {
                is_default: default_id.as_deref() == Some(id.as_str()),
                id,
                name,
            });
        }
        result.sort_by(|left, right| left.id.cmp(&right.id));
        Ok((result, default_id))
    }

    /// Re-enumerates endpoints and reports a default-route change, if any.
    ///
    /// A failed enumeration leaves the prior baseline intact. CPAL route
    /// changes are observed only when the caller polls; no OS notification is
    /// implied. This consumes the full catalog delta; use `poll_changes`
    /// instead when added/removed endpoints also matter.
    pub fn poll(&mut self) -> Result<Option<DefaultInputChange>, DeviceError> {
        Ok(self.poll_changes()?.and_then(|change| change.default))
    }

    /// Re-enumerates endpoints and reports all device and default-route deltas.
    ///
    /// The first successful poll establishes a baseline and returns `None`.
    /// A failed enumeration leaves both baselines intact.
    pub fn poll_changes(&mut self) -> Result<Option<InputCatalogChange>, DeviceError> {
        let (current_devices, current_default) = self.enumerate()?;
        let default = observe_default(&mut self.previous_default, current_default);
        let previous_devices = self.previous_devices.replace(current_devices.clone());
        let Some(previous_devices) = previous_devices else {
            return Ok(None);
        };
        let (added, removed) = diff_devices(&previous_devices, &current_devices);
        if added.is_empty() && removed.is_empty() && default.is_none() {
            Ok(None)
        } else {
            Ok(Some(InputCatalogChange {
                added,
                removed,
                default,
            }))
        }
    }
}

fn diff_devices(
    previous: &[InputDeviceInfo],
    current: &[InputDeviceInfo],
) -> (Vec<InputDeviceInfo>, Vec<InputDeviceInfo>) {
    let added = current
        .iter()
        .filter(|device| !previous.iter().any(|old| old.id == device.id))
        .cloned()
        .collect();
    let removed = previous
        .iter()
        .filter(|device| !current.iter().any(|new| new.id == device.id))
        .cloned()
        .collect();
    (added, removed)
}

fn observe_default(
    previous: &mut Option<Option<String>>,
    current: Option<String>,
) -> Option<DefaultInputChange> {
    match previous.replace(current.clone()) {
        None => None,
        Some(old) if old == current => None,
        Some(old) => Some(DefaultInputChange {
            previous: old,
            current,
        }),
    }
}

/// Negotiated configuration for a live input stream.
#[derive(Debug, Clone)]
pub struct InputStreamInfo {
    /// Human-readable input endpoint name.
    pub device_name: String,
    /// Stable CPAL endpoint identifier.
    pub device_id: String,
    /// Captured sample rate in frames per second.
    pub sample_rate: u32,
    /// Interleaved input channel count.
    pub channels: usize,
    /// Native CPAL sample representation.
    pub sample_format: SampleFormat,
}

/// Input-specific events drained from the control thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureEvent {
    /// CPAL accepted the start request (not proof of callback delivery).
    Started,
    /// Input callback delivery stopped by the caller.
    Stopped,
    /// Endpoint disappeared or CPAL invalidated the stream.
    Disconnected,
    /// CPAL reported another stream error.
    StreamError,
    /// Whole input frames were discarded because the bounded ring was full.
    Overflow {
        /// Samples dropped since the previous overflow event was polled.
        dropped_samples: u64,
    },
    /// A replacement stream opened successfully on the control thread.
    Reconnected,
    /// One bounded recovery attempt failed.
    RecoveryFailed {
        /// Number of failed attempts in the current incident.
        attempts: u32,
    },
    /// The bounded recovery budget was exhausted.
    RecoveryExhausted {
        /// Number of failed attempts in the terminal incident.
        attempts: u32,
    },
}

/// One sequenced input lifecycle record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureEventRecord {
    /// Unique issuance ticket for this input instance, including dropped records.
    ///
    /// Concurrent producers can enqueue out of ticket order. This is not a
    /// wall-clock timestamp or a replay ordering guarantee.
    pub sequence: u64,
    /// Stream generation that produced or observed the event.
    ///
    /// Overflow is aggregated across the shared ring and tagged with the
    /// generation at poll time, not necessarily the generation that lost data.
    pub generation: u64,
    /// Event payload.
    pub event: CaptureEvent,
}

/// Bounded event-log counters visible to the control thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureEventLogStats {
    /// Fixed number of records retained by the queue.
    pub capacity: usize,
    /// Records currently waiting to be drained.
    pub queued: usize,
    /// Records rejected because the queue was full.
    pub dropped: u64,
}

struct CaptureEventLog {
    queue: ArrayQueue<CaptureEventRecord>,
    next_sequence: AtomicU64,
    dropped: AtomicU64,
}

impl CaptureEventLog {
    fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity.max(1)),
            next_sequence: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
        }
    }

    fn record(&self, generation: u64, event: CaptureEvent) -> CaptureEventRecord {
        let record = self.issue(generation, event);
        if self.queue.push(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        record
    }

    fn issue(&self, generation: u64, event: CaptureEvent) -> CaptureEventRecord {
        CaptureEventRecord {
            sequence: self.next_sequence.fetch_add(1, Ordering::Relaxed),
            generation,
            event,
        }
    }

    fn pop(&self) -> Option<CaptureEventRecord> {
        self.queue.pop()
    }

    fn stats(&self) -> CaptureEventLogStats {
        CaptureEventLogStats {
            capacity: self.queue.capacity(),
            queued: self.queue.len(),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

fn poll_capture_record(
    log: &CaptureEventLog,
    generation: u64,
    sink: &CaptureSink,
    reported_drops: &mut u64,
) -> Option<CaptureEventRecord> {
    if let Some(record) = log.pop() {
        return Some(record);
    }
    let dropped = sink.dropped_samples();
    if dropped > *reported_drops {
        let delta = dropped - *reported_drops;
        *reported_drops = dropped;
        // Return directly: queueing as well would deliver the event twice.
        return Some(log.issue(
            generation,
            CaptureEvent::Overflow {
                dropped_samples: delta,
            },
        ));
    }
    None
}

/// Recovery policy for one input stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputRecovery {
    state: DeviceState,
    attempts: u32,
    max_attempts: u32,
}

impl InputRecovery {
    /// Creates a stopped input recovery state machine.
    #[must_use]
    pub const fn new(max_attempts: u32) -> Self {
        Self {
            state: DeviceState::Stopped,
            attempts: 0,
            max_attempts,
        }
    }

    /// Returns the current input lifecycle state.
    #[must_use]
    pub const fn state(self) -> DeviceState {
        self.state
    }

    /// Returns failed attempts in the current recovery incident.
    #[must_use]
    pub const fn attempts(self) -> u32 {
        self.attempts
    }

    /// Returns the configured recovery budget.
    #[must_use]
    pub const fn max_attempts(self) -> u32 {
        self.max_attempts
    }

    fn started(&mut self) {
        self.state = DeviceState::Running;
        self.attempts = 0;
    }

    fn stopped(&mut self) {
        self.state = DeviceState::Stopped;
        self.attempts = 0;
    }

    fn failed(&mut self) {
        if self.state != DeviceState::Failed {
            self.state = DeviceState::Recovering;
        }
    }

    fn recovery_failed(&mut self) -> bool {
        self.attempts = self.attempts.saturating_add(1);
        if self.attempts >= self.max_attempts.max(1) {
            self.state = DeviceState::Failed;
            true
        } else {
            self.state = DeviceState::Recovering;
            false
        }
    }

    fn recovered(&mut self, running: bool) {
        self.state = if running {
            DeviceState::Running
        } else {
            DeviceState::Stopped
        };
        self.attempts = 0;
    }
}

/// Authoritative control state, independent of the lossy diagnostic log.
struct InputControl {
    recovery: InputRecovery,
    wants_running: bool,
}

impl InputControl {
    fn new() -> Self {
        Self {
            recovery: InputRecovery::new(3),
            wants_running: false,
        }
    }

    fn started(&mut self) {
        self.wants_running = true;
        self.recovery.started();
    }

    fn stopped(&mut self) {
        self.wants_running = false;
        self.recovery.stopped();
    }

    fn observe_fault(&mut self, fault: &AtomicBool) -> DeviceState {
        if fault.swap(false, Ordering::AcqRel) {
            self.recovery.failed();
        }
        self.recovery.state()
    }

    fn snapshot(&self, fault: &AtomicBool) -> InputRecovery {
        let mut recovery = self.recovery;
        if fault.load(Ordering::Acquire) {
            recovery.failed();
        }
        recovery
    }

    // Every failure, including route lookup/build/play, consumes one attempt.
    fn attempt<T>(
        &mut self,
        open: impl FnOnce(bool) -> Result<T, DeviceError>,
    ) -> Result<T, DeviceError> {
        if self.recovery.state() != DeviceState::Recovering {
            return Err(DeviceError::InvalidState {
                operation: "recover input",
                state: self.recovery.state(),
            });
        }
        match open(self.wants_running) {
            Ok(value) => {
                self.recovery.recovered(self.wants_running);
                Ok(value)
            }
            Err(error) => {
                self.recovery.recovery_failed();
                Err(error)
            }
        }
    }
}

/// A CPAL input stream feeding a lock-free [`CaptureSink`].
///
/// Keep the value alive while capture is active. Dropping it closes the stream;
/// `stop` pauses delivery without discarding the generation. After a stream
/// error the state becomes [`DeviceState::Recovering`], even if its diagnostic
/// record is dropped. Draining records never replays state transitions.
/// The control thread may
/// call [`CpalInput::recover`] to perform one bounded reopen attempt.
///
/// The capture ring must have one active producer: do not feed clones of its
/// sink from other callbacks. Reopening retains buffered samples without
/// timestamps, generation tags, gap compensation, or sample-rate conversion.
pub struct CpalInput {
    stream: Option<cpal::Stream>,
    info: InputStreamInfo,
    sink: CaptureSink,
    errors: Arc<AtomicU64>,
    event_log: Arc<CaptureEventLog>,
    fault: Arc<AtomicBool>,
    generation: u64,
    route: InputRoute,
    control: InputControl,
    reported_drops: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputRoute {
    Fixed(String),
    FollowDefault,
}

impl CpalInput {
    /// Returns the last successfully negotiated input endpoint configuration.
    ///
    /// After a failed reopen this still describes the previous stream, not an
    /// active endpoint. Check `state` before treating it as live configuration.
    #[must_use]
    pub fn info(&self) -> &InputStreamInfo {
        &self.info
    }

    /// Number of CPAL stream errors observed since opening.
    #[must_use]
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Current input-stream lifecycle state.
    #[must_use]
    pub fn state(&self) -> DeviceState {
        self.recovery().state()
    }

    /// Current stream generation, starting at one for the initial stream.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Current bounded recovery state.
    #[must_use]
    pub fn recovery(&self) -> InputRecovery {
        self.control.snapshot(&self.fault)
    }

    /// Returns event-log capacity, queue depth, and explicit drop count.
    #[must_use]
    pub fn event_log_stats(&self) -> CaptureEventLogStats {
        self.event_log.stats()
    }

    /// Number of samples discarded by the bounded capture ring since opening.
    #[must_use]
    pub fn dropped_samples(&self) -> u64 {
        self.sink.dropped_samples()
    }

    /// Starts callback delivery.
    pub fn start(&mut self) -> Result<(), DeviceError> {
        let state = self.control.observe_fault(&self.fault);
        if state != DeviceState::Stopped {
            return Err(DeviceError::InvalidState {
                operation: "start input",
                state,
            });
        }
        // A stop during recovery retired the broken stream. An explicit start
        // may open a fresh generation, but never silently retries on failure.
        if self.stream.is_none() {
            self.generation = self.generation.saturating_add(1);
            self.fault = Arc::new(AtomicBool::new(false));
            let (stream, info) = self.open_replacement()?;
            self.stream = Some(stream);
            self.info = info;
        }
        self.stream
            .as_ref()
            .expect("a stopped input has a stream after explicit reopen")
            .play()
            .map_err(|error| DeviceError::PlayStream(error.to_string()))?;
        self.control.started();
        self.event_log
            .record(self.generation, CaptureEvent::Started);
        Ok(())
    }

    /// Pauses delivery, or retires a broken stream and cancels recovery.
    pub fn stop(&mut self) -> Result<(), DeviceError> {
        let state = self.control.observe_fault(&self.fault);
        if !matches!(state, DeviceState::Running | DeviceState::Recovering) {
            return Err(DeviceError::InvalidState {
                operation: "stop input",
                state,
            });
        }
        if state == DeviceState::Recovering {
            drop(self.stream.take());
            self.fault = Arc::new(AtomicBool::new(false));
        } else if let Some(stream) = &self.stream {
            stream
                .pause()
                .map_err(|error| DeviceError::StopStream(error.to_string()))?;
        }
        self.control.stopped();
        self.event_log
            .record(self.generation, CaptureEvent::Stopped);
        Ok(())
    }

    /// Drains one sequenced event from the bounded control-side event log.
    #[must_use]
    pub fn poll_event_record(&mut self) -> Option<CaptureEventRecord> {
        self.control.observe_fault(&self.fault);
        poll_capture_record(
            &self.event_log,
            self.generation,
            &self.sink,
            &mut self.reported_drops,
        )
    }

    /// Drains one lifecycle, stream-error, or ring-overflow event.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<CaptureEvent> {
        self.poll_event_record().map(|record| record.event)
    }

    /// Reopens the current endpoint or the current default endpoint once.
    ///
    /// This method is control-thread only. It never retries forever: after
    /// `max_attempts` failures the input enters terminal `Failed` and callers
    /// must explicitly open a new input instance. The current budget is three
    /// attempts, counting route lookup, configuration, build, and play failures.
    /// Follow-default is evaluated on reopen only; catalog polling never
    /// migrates an otherwise healthy stream.
    pub fn recover(&mut self) -> Result<(), DeviceError> {
        // Authorize from the consumed control state, not `state()`'s racy
        // snapshot: a new fault after this observation is for the next call.
        let state = self.control.observe_fault(&self.fault);
        if state != DeviceState::Recovering {
            return Err(DeviceError::InvalidState {
                operation: "recover input",
                state,
            });
        }
        // Retire before building: two generations must not produce into the
        // interleaved ring concurrently. Old fault flags cannot affect new ones.
        drop(self.stream.take());
        self.generation = self.generation.saturating_add(1);
        self.fault = Arc::new(AtomicBool::new(false));
        let Self {
            control,
            route,
            sink,
            errors,
            event_log,
            fault,
            generation,
            ..
        } = self;
        let result = control.attempt(|running| {
            let target_id = resolve_route(route, current_default_input_id)?;
            let (stream, info) = open_replacement(
                &target_id,
                sink.clone(),
                Arc::clone(errors),
                Arc::clone(event_log),
                Arc::clone(fault),
                *generation,
            )?;
            if running {
                stream
                    .play()
                    .map_err(|error| DeviceError::PlayStream(error.to_string()))?;
            }
            Ok((stream, info))
        });
        match result {
            Ok((stream, info)) => {
                self.stream = Some(stream);
                self.info = info;
                self.event_log
                    .record(self.generation, CaptureEvent::Reconnected);
                Ok(())
            }
            Err(error) => {
                let recovery = self.control.recovery;
                self.event_log.record(
                    self.generation,
                    if recovery.state() == DeviceState::Failed {
                        CaptureEvent::RecoveryExhausted {
                            attempts: recovery.attempts(),
                        }
                    } else {
                        CaptureEvent::RecoveryFailed {
                            attempts: recovery.attempts(),
                        }
                    },
                );
                Err(error)
            }
        }
    }

    fn open_replacement(&self) -> Result<(cpal::Stream, InputStreamInfo), DeviceError> {
        let target_id = resolve_route(&self.route, current_default_input_id)?;
        open_replacement(
            &target_id,
            self.sink.clone(),
            Arc::clone(&self.errors),
            Arc::clone(&self.event_log),
            Arc::clone(&self.fault),
            self.generation,
        )
    }
}

impl core::fmt::Debug for CpalInput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpalInput")
            .field("info", &self.info)
            .field("state", &self.state())
            .field("errors", &self.error_count())
            .field("dropped_samples", &self.sink.dropped_samples())
            .finish_non_exhaustive()
    }
}

/// Opens and starts the platform default input endpoint.
pub fn open_default_input(sink: CaptureSink) -> Result<CpalInput, DeviceError> {
    let mut input = open_default_input_paused(sink)?;
    input.start()?;
    Ok(input)
}

/// Opens the default input endpoint, leaving callback delivery stopped.
pub fn open_default_input_paused(sink: CaptureSink) -> Result<CpalInput, DeviceError> {
    let device_id = current_default_input_id()?;
    open_input_device_paused_with_route(&device_id, sink, InputRoute::FollowDefault)
}

fn current_default_input_id() -> Result<String, DeviceError> {
    cpal::default_host()
        .default_input_device()
        .ok_or(DeviceError::NoInputDevice)?
        .id()
        .map(|id| id.to_string())
        .map_err(|error| DeviceError::ConfigQuery(error.to_string()))
}

fn resolve_route(
    route: &InputRoute,
    default_id: impl FnOnce() -> Result<String, DeviceError>,
) -> Result<String, DeviceError> {
    match route {
        InputRoute::Fixed(id) => Ok(id.clone()),
        InputRoute::FollowDefault => default_id(),
    }
}

/// Opens and starts an enumerated input endpoint by its stable CPAL device ID.
///
/// Resolve the ID from [`CpalInputCatalog::devices`]. Names are intentionally
/// not accepted as selectors because they are not guaranteed unique. The ID is
/// looked up again at open time; a removed/stale endpoint returns
/// [`DeviceError::InputDeviceNotFound`].
pub fn open_input_device(device_id: &str, sink: CaptureSink) -> Result<CpalInput, DeviceError> {
    let mut input = open_input_device_paused(device_id, sink)?;
    input.start()?;
    Ok(input)
}

/// Opens an enumerated input endpoint by CPAL ID without starting delivery.
pub fn open_input_device_paused(
    device_id: &str,
    sink: CaptureSink,
) -> Result<CpalInput, DeviceError> {
    open_input_device_paused_with_route(device_id, sink, InputRoute::Fixed(device_id.to_string()))
}

fn open_input_device_paused_with_route(
    device_id: &str,
    sink: CaptureSink,
    route: InputRoute,
) -> Result<CpalInput, DeviceError> {
    let catalog = CpalInputCatalog::new();
    let devices = catalog.devices()?;
    let selected = resolve_catalog_device(&devices, device_id)?.clone();
    let parsed_id = device_id
        .parse::<cpal::DeviceId>()
        .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
    let device = catalog
        .host
        .device_by_id(&parsed_id)
        .ok_or_else(|| DeviceError::InputDeviceNotFound(device_id.to_string()))?;
    open_device_paused(device, selected, sink, route)
}

fn resolve_catalog_device<'a>(
    devices: &'a [InputDeviceInfo],
    requested_id: &str,
) -> Result<&'a InputDeviceInfo, DeviceError> {
    devices
        .iter()
        .find(|device| device.id == requested_id)
        .ok_or_else(|| DeviceError::InputDeviceNotFound(requested_id.to_string()))
}

fn open_device_paused(
    device: cpal::Device,
    selected: InputDeviceInfo,
    sink: CaptureSink,
    route: InputRoute,
) -> Result<CpalInput, DeviceError> {
    let device_id = selected.id;
    let device_name = selected.name;
    let default_config = device
        .default_input_config()
        .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
    let sample_format = default_config.sample_format();
    ensure_supported_input_format(sample_format)?;
    let channels = default_config.channels() as usize;
    if channels == 0 || channels != sink.channels() {
        return Err(DeviceError::CaptureChannelCountMismatch {
            sink: sink.channels(),
            device: channels,
        });
    }
    let info = InputStreamInfo {
        device_name,
        device_id: device_id.clone(),
        sample_rate: default_config.sample_rate(),
        channels,
        sample_format,
    };
    let errors = Arc::new(AtomicU64::new(0));
    let event_log = Arc::new(CaptureEventLog::new(DEFAULT_EVENT_LOG_CAPACITY));
    let fault = Arc::new(AtomicBool::new(false));
    let stream = build_input_stream(
        &device,
        &default_config.into(),
        sample_format,
        sink.clone(),
        Arc::clone(&errors),
        Arc::clone(&event_log),
        Arc::clone(&fault),
        1,
    )?;
    Ok(CpalInput {
        stream: Some(stream),
        info,
        sink,
        errors,
        event_log,
        fault,
        generation: 1,
        route,
        control: InputControl::new(),
        reported_drops: 0,
    })
}

fn ensure_supported_input_format(format: SampleFormat) -> Result<(), DeviceError> {
    match format {
        SampleFormat::F32
        | SampleFormat::F64
        | SampleFormat::I8
        | SampleFormat::I16
        | SampleFormat::I24
        | SampleFormat::I32
        | SampleFormat::I64
        | SampleFormat::U8
        | SampleFormat::U16
        | SampleFormat::U24
        | SampleFormat::U32
        | SampleFormat::U64 => Ok(()),
        other => Err(DeviceError::UnsupportedSampleFormat(format!("{other:?}"))),
    }
}

#[inline]
fn convert_input_sample<T>(sample: T) -> f32
where
    f32: FromSample<T>,
{
    f32::from_sample(sample)
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: SampleFormat,
    sink: CaptureSink,
    errors: Arc<AtomicU64>,
    event_log: Arc<CaptureEventLog>,
    fault: Arc<AtomicBool>,
    generation: u64,
) -> Result<cpal::Stream, DeviceError> {
    macro_rules! build {
        ($ty:ty) => {
            build_typed_input::<$ty>(device, config, sink, errors, event_log, fault, generation)
        };
    }
    match format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        other => Err(DeviceError::UnsupportedSampleFormat(format!("{other:?}"))),
    }
}

fn build_typed_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sink: CaptureSink,
    errors: Arc<AtomicU64>,
    event_log: Arc<CaptureEventLog>,
    fault: Arc<AtomicBool>,
    generation: u64,
) -> Result<cpal::Stream, DeviceError>
where
    T: SizedSample + Copy + Send + 'static,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _info: &cpal::InputCallbackInfo| {
                sink.push_interleaved_with(data, convert_input_sample::<T>);
            },
            move |error| {
                errors.fetch_add(1, Ordering::Relaxed);
                fault.store(true, Ordering::Release);
                let event = match error {
                    cpal::StreamError::DeviceNotAvailable
                    | cpal::StreamError::StreamInvalidated => CaptureEvent::Disconnected,
                    cpal::StreamError::BufferUnderrun
                    | cpal::StreamError::BackendSpecific { .. } => CaptureEvent::StreamError,
                };
                event_log.record(generation, event);
            },
            None,
        )
        .map_err(|error| DeviceError::BuildStream(error.to_string()))
}

fn open_replacement(
    device_id: &str,
    sink: CaptureSink,
    errors: Arc<AtomicU64>,
    event_log: Arc<CaptureEventLog>,
    fault: Arc<AtomicBool>,
    generation: u64,
) -> Result<(cpal::Stream, InputStreamInfo), DeviceError> {
    let catalog = CpalInputCatalog::new();
    let devices = catalog.devices()?;
    let selected = resolve_catalog_device(&devices, device_id)?.clone();
    let parsed_id = device_id
        .parse::<cpal::DeviceId>()
        .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
    let device = catalog
        .host
        .device_by_id(&parsed_id)
        .ok_or_else(|| DeviceError::InputDeviceNotFound(device_id.to_string()))?;
    let default_config = device
        .default_input_config()
        .map_err(|error| DeviceError::ConfigQuery(error.to_string()))?;
    let sample_format = default_config.sample_format();
    ensure_supported_input_format(sample_format)?;
    let channels = default_config.channels() as usize;
    if channels == 0 || channels != sink.channels() {
        return Err(DeviceError::CaptureChannelCountMismatch {
            sink: sink.channels(),
            device: channels,
        });
    }
    let info = InputStreamInfo {
        device_name: selected.name,
        device_id: selected.id,
        sample_rate: default_config.sample_rate(),
        channels,
        sample_format,
    };
    let stream = build_input_stream(
        &device,
        &default_config.into(),
        sample_format,
        sink,
        errors,
        event_log,
        fault,
        generation,
    )?;
    Ok((stream, info))
}

#[cfg(test)]
mod tests {
    use super::{
        convert_input_sample, diff_devices, ensure_supported_input_format, observe_default,
        resolve_catalog_device, CaptureEvent, InputDeviceInfo, InputRecovery,
    };
    use crate::capture::capture_ring;
    use cpal::SampleFormat;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    #[test]
    fn native_pcm_conversion_feeds_the_capture_ring_without_intermediate_storage() {
        let (sink, mut consumer) = capture_ring(2, 4);
        let captured = [i16::MIN, 0, i16::MAX, -16_384];
        assert_eq!(
            sink.push_interleaved_with(&captured, convert_input_sample::<i16>),
            captured.len()
        );
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 2);
        assert_eq!(consumer.drain_into(&mut output), 2);
        assert_eq!(output.channel(0)[0], -1.0);
        assert_eq!(output.channel(1)[0], 0.0);
        assert_eq!(output.channel(0)[1], i16::MAX as f32 / 32_768.0);
        assert_eq!(output.channel(1)[1], -0.5);
    }

    #[test]
    fn input_format_policy_accepts_pcm_and_rejects_encoded_formats() {
        for format in [
            SampleFormat::F32,
            SampleFormat::F64,
            SampleFormat::I8,
            SampleFormat::I16,
            SampleFormat::I24,
            SampleFormat::I32,
            SampleFormat::I64,
            SampleFormat::U8,
            SampleFormat::U16,
            SampleFormat::U24,
            SampleFormat::U32,
            SampleFormat::U64,
        ] {
            assert!(ensure_supported_input_format(format).is_ok(), "{format:?}");
        }
        for format in [
            SampleFormat::DsdU8,
            SampleFormat::DsdU16,
            SampleFormat::DsdU32,
        ] {
            assert!(ensure_supported_input_format(format).is_err(), "{format:?}");
        }
    }

    #[test]
    fn default_route_tracker_uses_first_poll_as_baseline_and_reports_changes() {
        let mut previous = None;
        assert_eq!(
            observe_default(&mut previous, Some("host:mic-a".into())),
            None
        );
        assert_eq!(
            observe_default(&mut previous, Some("host:mic-a".into())),
            None
        );
        assert_eq!(
            observe_default(&mut previous, Some("host:mic-b".into())),
            Some(super::DefaultInputChange {
                previous: Some("host:mic-a".into()),
                current: Some("host:mic-b".into()),
            })
        );
        assert_eq!(
            observe_default(&mut previous, None),
            Some(super::DefaultInputChange {
                previous: Some("host:mic-b".into()),
                current: None,
            })
        );
        assert_eq!(observe_default(&mut previous, None), None);
    }

    #[test]
    fn selected_device_resolution_uses_exact_stable_ids_not_display_names() {
        let devices = [
            InputDeviceInfo {
                id: "host:device-b".into(),
                name: "Studio Mic".into(),
                is_default: false,
            },
            InputDeviceInfo {
                id: "host:device-a".into(),
                name: "Studio Mic".into(),
                is_default: true,
            },
        ];
        assert_eq!(
            resolve_catalog_device(&devices, "host:device-b")
                .unwrap()
                .id,
            "host:device-b"
        );
        assert!(matches!(
            resolve_catalog_device(&devices, "Studio Mic"),
            Err(crate::DeviceError::InputDeviceNotFound(_))
        ));
        assert!(matches!(
            resolve_catalog_device(&devices, "host:stale"),
            Err(crate::DeviceError::InputDeviceNotFound(_))
        ));
    }

    #[test]
    fn catalog_delta_reports_added_and_removed_ids_deterministically() {
        let previous = [InputDeviceInfo {
            id: "host:a".into(),
            name: "A".into(),
            is_default: true,
        }];
        let current = [InputDeviceInfo {
            id: "host:b".into(),
            name: "B".into(),
            is_default: true,
        }];
        let (added, removed) = diff_devices(&previous, &current);
        assert_eq!(added, current);
        assert_eq!(removed, previous);
    }

    #[test]
    fn input_recovery_is_bounded_and_manual_stop_does_not_recover() {
        let mut recovery = InputRecovery::new(2);
        recovery.started();
        recovery.stopped();
        assert_eq!(recovery.state(), crate::DeviceState::Stopped);
        recovery.failed();
        assert_eq!(recovery.state(), crate::DeviceState::Recovering);
        assert!(!recovery.recovery_failed());
        assert!(recovery.recovery_failed());
        assert_eq!(recovery.state(), crate::DeviceState::Failed);
    }

    #[test]
    fn event_log_records_sequences_and_explicit_overflow() {
        let log = super::CaptureEventLog::new(2);
        let first = log.record(1, CaptureEvent::Started);
        let second = log.record(1, CaptureEvent::Stopped);
        let third = log.record(1, CaptureEvent::StreamError);
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(third.sequence, 3);
        assert_eq!(log.stats().dropped, 1);
        assert_eq!(log.pop(), Some(first));
        assert_eq!(log.pop(), Some(second));
        assert_eq!(log.pop(), None);
    }

    #[test]
    fn overflow_record_is_delivered_once_not_returned_and_enqueued() {
        let log = super::CaptureEventLog::new(2);
        let (sink, _consumer) = capture_ring(2, 1);
        sink.push_interleaved(&[1.0, 2.0, 3.0, 4.0]);
        let mut reported = 0;
        let record = super::poll_capture_record(&log, 4, &sink, &mut reported).unwrap();
        assert_eq!(record.generation, 4);
        assert_eq!(record.event, CaptureEvent::Overflow { dropped_samples: 2 });
        assert!(super::poll_capture_record(&log, 4, &sink, &mut reported).is_none());
        assert_eq!(log.stats().queued, 0);
    }

    #[test]
    fn route_lookup_build_and_play_failures_share_one_terminal_budget() {
        use crate::{DeviceError, DeviceState};
        let mut control = super::InputControl::new();
        control.started();
        control.recovery.failed();
        let errors = [
            DeviceError::NoInputDevice,
            DeviceError::BuildStream("injected build failure".into()),
            DeviceError::PlayStream("injected play failure".into()),
        ];
        for (index, error) in errors.into_iter().enumerate() {
            assert!(control
                .attempt::<()>(|running| {
                    assert!(running);
                    Err(error)
                })
                .is_err());
            assert_eq!(control.recovery.attempts(), index as u32 + 1);
        }
        assert_eq!(control.recovery.state(), DeviceState::Failed);
        control.recovery.failed(); // More callback faults cannot revive Failed.
        assert!(control
            .attempt::<()>(|_| panic!("terminal input must not reopen"))
            .is_err());
        assert_eq!(control.recovery.attempts(), 3);
    }

    #[test]
    fn missing_follow_default_route_is_a_counted_recovery_failure() {
        let mut control = super::InputControl::new();
        control.recovery.failed();
        for _ in 0..3 {
            assert!(control
                .attempt(|_| {
                    super::resolve_route(&super::InputRoute::FollowDefault, || {
                        Err(crate::DeviceError::NoInputDevice)
                    })
                })
                .is_err());
        }
        assert_eq!(control.recovery.state(), crate::DeviceState::Failed);
    }

    #[test]
    fn route_policy_resolves_fixed_ids_and_follows_default_only_on_request() {
        assert_eq!(
            super::resolve_route(&super::InputRoute::Fixed("fixed".into()), || panic!(
                "must not query default"
            ))
            .unwrap(),
            "fixed"
        );
        for id in ["default-a", "default-b"] {
            assert_eq!(
                super::resolve_route(&super::InputRoute::FollowDefault, || Ok(id.into())).unwrap(),
                id
            );
        }
    }

    #[test]
    fn dropped_fault_records_and_repeated_errors_do_not_lose_resume_intent() {
        use core::sync::atomic::{AtomicBool, Ordering};
        let log = super::CaptureEventLog::new(1);
        let mut control = super::InputControl::new();
        let fault = AtomicBool::new(false);
        control.started();
        log.record(1, CaptureEvent::Started);
        // Full diagnostic queue, but the separate fault latch still works.
        for _ in 0..2 {
            fault.store(true, Ordering::Release);
            log.record(1, CaptureEvent::Disconnected);
            assert_eq!(
                control.snapshot(&fault).state(),
                crate::DeviceState::Recovering
            );
            control.observe_fault(&fault);
        }
        assert_eq!(log.stats().dropped, 2);
        assert!(control
            .attempt(|running| {
                assert!(
                    running,
                    "repeated faults must preserve original running intent"
                );
                Ok(())
            })
            .is_ok());
        assert_eq!(control.recovery.state(), crate::DeviceState::Running);
        assert_eq!(control.recovery.attempts(), 0);
        // Old diagnostics are observations only and cannot reset new state.
        assert_eq!(log.pop().unwrap().event, CaptureEvent::Started);
        assert_eq!(control.recovery.state(), crate::DeviceState::Running);
    }

    #[test]
    fn stop_cancels_recovery_and_old_generation_faults_are_isolated() {
        use core::sync::atomic::{AtomicBool, Ordering};
        let mut control = super::InputControl::new();
        let old_fault = AtomicBool::new(true);
        control.started();
        control.observe_fault(&old_fault);
        control.stopped();
        let new_fault = AtomicBool::new(false);
        old_fault.store(true, Ordering::Release);
        control.observe_fault(&new_fault);
        assert_eq!(
            control.snapshot(&new_fault).state(),
            crate::DeviceState::Stopped
        );
        assert!(!control.wants_running);
        assert!(control
            .attempt::<()>(|_| panic!("manual stop must not auto-recover"))
            .is_err());
        control.recovery.failed(); // A paused stream may also fail.
        control
            .attempt(|running| {
                assert!(!running);
                Ok(())
            })
            .unwrap();
        assert_eq!(control.recovery.state(), crate::DeviceState::Stopped);
    }

    #[test]
    fn concurrent_event_log_has_unique_tickets_and_explicit_drop_accounting() {
        let log = std::sync::Arc::new(super::CaptureEventLog::new(256));
        std::thread::scope(|scope| {
            for generation in 1..=4 {
                let log = std::sync::Arc::clone(&log);
                scope.spawn(move || {
                    for _ in 0..100 {
                        log.record(generation, CaptureEvent::StreamError);
                    }
                });
            }
        });
        let mut tickets = Vec::new();
        while let Some(record) = log.pop() {
            tickets.push(record.sequence);
        }
        assert_eq!(tickets.len(), 256);
        assert_eq!(log.stats().dropped, 144);
        tickets.sort_unstable();
        tickets.dedup();
        assert_eq!(tickets.len(), 256);
    }

    #[test]
    fn late_callback_fault_does_not_authorize_reopen_from_a_snapshot() {
        use core::sync::atomic::{AtomicBool, Ordering};
        let mut control = super::InputControl::new();
        let fault = AtomicBool::new(false);
        control.started();
        let observed = control.observe_fault(&fault);
        fault.store(true, Ordering::Release); // Callback between control reads.
        assert_eq!(observed, crate::DeviceState::Running);
        assert_eq!(
            control.snapshot(&fault).state(),
            crate::DeviceState::Recovering
        );
        assert!(control
            .attempt::<()>(|_| panic!("snapshot must not authorize retiring a live stream"))
            .is_err());
        assert_eq!(
            control.observe_fault(&fault),
            crate::DeviceState::Recovering
        );
        control
            .attempt(|running| {
                assert!(running);
                Ok(())
            })
            .unwrap();
        assert_eq!(control.recovery.state(), crate::DeviceState::Running);
    }
}
