//! Presentation feedback → time prediction → pre-input begin-work gating.
//!
//! A runner polls backend feedback, calls [`PresentationPacing::feedback`],
//! then calls [`App::update_with_presentation`] **before sampling frame input**.
//! `Wait`/`QueueFull` leave the App untouched; arrange an event-loop wakeup or
//! wait for feedback instead of spinning. Use one explicitly chosen pacing
//! target for a shared App update, not one update per window. Other windows
//! keep independent timelines in the same resource. The render bridge calls
//! `submitted` only after successful submission and `retire_unpresented` for
//! discarded frames or completion when actual feedback is unavailable.
//!
//! All timestamps are nanoseconds in an explicitly identified monotonic clock
//! epoch. Platform adapters must convert native timestamps into that epoch;
//! tagging unrelated raw clocks with the same id is invalid. Surface recreation
//! must use a new generation. Suspension must call `set_active(false, now)`.
//!
//! Actual feedback is an explicit backend capability. In particular, pinned
//! wgpu 30.0.1's public `Queue::present` returns no display timestamp. Configure
//! it as [`FeedbackSupport::Unsupported`]. Returning from present, render/GPU
//! completion, predictions and VRR/LFC policy are never actual display evidence.

use alloc::collections::BTreeMap;
use core::fmt;

#[cfg(feature = "std")]
use alloc::collections::VecDeque;
#[cfg(feature = "std")]
use alloc::sync::Arc;

use prism_ecs::prelude::Resource;
use prism_time::{BeginDecision, FramePacer, PresentInfo, ReflexGate, VrrPresent, VrrWindow};

use crate::App;

/// Identity of a monotonic clock **and its epoch**, allocated by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationClockDomain(pub u64);

/// A timestamp whose clock epoch is explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationTimestamp {
    /// Monotonic clock and epoch.
    pub domain: PresentationClockDomain,
    /// Nanoseconds since that epoch.
    pub ns: u64,
}

/// A window/surface identity plus its incarnation; independent of OS handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationTarget {
    /// Host-assigned stable target id.
    pub id: u64,
    /// Increment on surface recreation or clock/source changes.
    pub generation: u64,
}

/// API providing actual display/compositor presentation evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationSource {
    /// Wayland presentation-time feedback (not a frame callback).
    WaylandPresentationTime,
    /// Metal drawable's presented-time callback.
    MetalPresentedTime,
    /// DXGI actual frame statistics.
    DxgiFrameStatistics,
    /// Vulkan display-timing / present-timing feedback.
    VulkanDisplayTiming,
    /// A host-defined actual presentation provider; id identifies its contract.
    External(u64),
}

/// Why actual presentation feedback cannot currently be obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedFeedback {
    /// Backend's public API has no actual presentation timestamp.
    BackendApi,
    /// The platform/provider does not support actual timestamps.
    Platform,
    /// There is no physical presentation target.
    Offscreen,
}

/// Actual feedback support; never inferred from vsync or GPU timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackSupport {
    /// An actual provider, in the target's configured clock domain.
    Actual(PresentationSource),
    /// Explicitly unsupported, with reason.
    Unsupported(UnsupportedFeedback),
}

/// Per-target begin-work policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationConfig {
    /// Actual backend capability.
    pub feedback: FeedbackSupport,
    /// Desired frame interval, clamped to at least one nanosecond.
    pub interval_ns: u64,
    /// Maximum submitted frames awaiting actual feedback or explicit retirement.
    pub max_frames_in_flight: u32,
    /// Extra time reserved before the predicted present deadline.
    pub safety_margin_ns: u64,
    /// Confirmed panel range. `None` if unknown; modes/refresh alone do not prove VRR.
    pub vrr: Option<VrrWindow>,
}

/// Provenance of a deadline. Every deadline is a prediction, never a timestamp measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineBasis {
    /// Predicted from validated actual display feedback.
    ActualFeedback(PresentationSource),
    /// Predicted from a host estimate, not actual presentation evidence.
    Estimated,
    /// Waiting for the first actual sample; cadence fallback is active.
    AwaitingActual,
    /// Backend cannot report actual feedback; cadence fallback is active.
    Unsupported(UnsupportedFeedback),
}

/// Runner's next-frame plan; query again after feedback or at the requested wakeup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationPlan {
    /// Reflex-style decision, or `QueueFull` while target is inactive.
    pub decision: BeginDecision,
    /// Predicted display deadline in the configured clock epoch.
    pub deadline: PresentationTimestamp,
    /// Latest safe begin-work time in that epoch.
    pub begin_at: PresentationTimestamp,
    /// Evidence on which this prediction is based.
    pub basis: DeadlineBasis,
    /// Submitted frames still awaiting actual feedback/explicit retirement.
    pub frames_in_flight: u32,
    /// VRR policy classification only; does not assert driver LFC was performed.
    pub vrr: Option<VrrPresent>,
    /// Whether the target is active (false means no input/update/render).
    pub active: bool,
}

/// Actual backend feedback, correlated with a previously submitted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationFeedback {
    /// Target and surface incarnation.
    pub target: PresentationTarget,
    /// Strictly increasing host frame serial, not an App or GPU timer counter.
    pub frame: u64,
    /// Actual timestamp provider.
    pub source: PresentationSource,
    /// Actual presentation timestamp, already converted into the host epoch.
    pub presented_at: PresentationTimestamp,
}

/// Cumulative observability; only accepted actual feedback produces display latency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentationStats {
    /// Accepted actual presentation samples.
    pub actual_samples: u64,
    /// Accepted estimates (never counted as actual).
    pub estimated_samples: u64,
    /// Rejected actual samples; predictor and queue remain unchanged.
    pub rejected_samples: u64,
    /// Explicit retirements without actual presentation evidence.
    pub unpresented_retirements: u64,
    /// Latest measured input-to-actual-presentation duration, if input was supplied.
    pub input_to_present_ns: Option<u64>,
    /// Latest interval/jitter/prediction from the time kernel.
    pub last_present: Option<PresentInfo>,
}

/// Deterministic validation failures; no partially applied feedback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationError {
    /// Target id is not registered.
    MissingTarget,
    /// A live target already uses this id; unregister it before recreating.
    DuplicateTarget,
    /// Feedback belongs to another incarnation of this surface.
    StaleGeneration,
    /// Timestamp clock/epoch differs from registration.
    ClockDomain,
    /// Backend explicitly cannot supply actual timestamps.
    Unsupported,
    /// Provider differs from the registered actual source.
    Source,
    /// Frame was not submitted or has already been retired.
    UnknownFrame,
    /// Submitted frame ids or presentation timestamps do not increase.
    NonMonotonic,
    /// Actual timestamp is later than the feedback poll time.
    FutureTimestamp,
    /// Input/presentation timestamp precedes the corresponding earlier event.
    TimestampOrder,
    /// Submission would exceed the configured queue bound.
    QueueFull,
    /// Target is suspended/occluded.
    Inactive,
    /// App has no presentation pacing resource installed.
    NotInstalled,
}

impl fmt::Display for PresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "presentation validation failed: {self:?}")
    }
}

impl core::error::Error for PresentationError {}

#[derive(Clone, Copy, Debug)]
struct Submission {
    at: u64,
    input: Option<u64>,
}

#[derive(Debug)]
struct TargetState {
    target: PresentationTarget,
    domain: PresentationClockDomain,
    config: PresentationConfig,
    pacer: FramePacer,
    gate: ReflexGate,
    pending: BTreeMap<u64, Submission>,
    last_submitted: Option<(u64, u64)>,
    last_actual: Option<(u64, u64)>,
    estimate: Option<(u64, u64)>,
    fallback_deadline: u64,
    active: bool,
    stats: PresentationStats,
}

impl TargetState {
    fn validate_time(&self, time: PresentationTimestamp) -> Result<u64, PresentationError> {
        if time.domain != self.domain {
            return Err(PresentationError::ClockDomain);
        }
        Ok(time.ns)
    }

    fn timestamp(&self, ns: u64) -> PresentationTimestamp {
        PresentationTimestamp {
            domain: self.domain,
            ns,
        }
    }

    fn accept_feedback(
        &mut self,
        sample: PresentationFeedback,
        now: PresentationTimestamp,
    ) -> Result<PresentInfo, PresentationError> {
        let source = match self.config.feedback {
            FeedbackSupport::Actual(source) => source,
            FeedbackSupport::Unsupported(_) => return Err(PresentationError::Unsupported),
        };
        if sample.source != source {
            return Err(PresentationError::Source);
        }
        let now = self.validate_time(now)?;
        let at = self.validate_time(sample.presented_at)?;
        if at > now {
            return Err(PresentationError::FutureTimestamp);
        }
        let submission = *self
            .pending
            .get(&sample.frame)
            .ok_or(PresentationError::UnknownFrame)?;
        if at < submission.at {
            return Err(PresentationError::TimestampOrder);
        }
        if self
            .last_actual
            .is_some_and(|(frame, time)| sample.frame <= frame || at <= time)
        {
            return Err(PresentationError::NonMonotonic);
        }
        // A newer displayed frame supersedes older queued frames (e.g. mailbox).
        // Those older frames contribute no actual timestamp or latency sample.
        let retired = self
            .pending
            .keys()
            .filter(|&&frame| frame < sample.frame)
            .count() as u64;
        self.pending.retain(|frame, _| *frame > sample.frame);
        self.stats.unpresented_retirements =
            self.stats.unpresented_retirements.saturating_add(retired);
        self.last_actual = Some((sample.frame, at));
        self.estimate = self.estimate.filter(|(frame, _)| *frame > sample.frame);
        let info = self.pacer.on_present(at);
        self.stats.actual_samples = self.stats.actual_samples.saturating_add(1);
        self.stats.input_to_present_ns = submission.input.map(|input| at - input);
        self.stats.last_present = Some(info);
        Ok(info)
    }
}

/// App-owned per-target presentation timelines; usable with `no_std + alloc`.
///
/// Install with `app.insert_resource(PresentationPacing::default())`. The host
/// registers each live window/surface and feeds it through the methods below.
/// No global clock, renderer completion, or cross-window cadence is inferred.
#[derive(Debug, Default, Resource)]
pub struct PresentationPacing {
    targets: BTreeMap<u64, TargetState>,
}

impl PresentationPacing {
    /// Register a new surface. First frame can begin immediately at `now`.
    pub fn register(
        &mut self,
        target: PresentationTarget,
        config: PresentationConfig,
        now: PresentationTimestamp,
    ) -> Result<(), PresentationError> {
        if self.targets.contains_key(&target.id) {
            return Err(PresentationError::DuplicateTarget);
        }
        let interval = config
            .interval_ns
            .max(1)
            .max(config.vrr.map_or(1, |vrr| vrr.min_interval_ns()));
        self.targets.insert(
            target.id,
            TargetState {
                target,
                domain: now.domain,
                config,
                pacer: FramePacer::new(interval),
                gate: ReflexGate::new(config.max_frames_in_flight, config.safety_margin_ns),
                pending: BTreeMap::new(),
                last_submitted: None,
                last_actual: None,
                estimate: None,
                fallback_deadline: now.ns,
                active: true,
                stats: PresentationStats::default(),
            },
        );
        Ok(())
    }

    fn state(&self, target: PresentationTarget) -> Result<&TargetState, PresentationError> {
        let state = self
            .targets
            .get(&target.id)
            .ok_or(PresentationError::MissingTarget)?;
        if state.target != target {
            return Err(PresentationError::StaleGeneration);
        }
        Ok(state)
    }

    fn state_mut(
        &mut self,
        target: PresentationTarget,
    ) -> Result<&mut TargetState, PresentationError> {
        let state = self
            .targets
            .get_mut(&target.id)
            .ok_or(PresentationError::MissingTarget)?;
        if state.target != target {
            return Err(PresentationError::StaleGeneration);
        }
        Ok(state)
    }

    /// Forget a destroyed surface, including queued frames and its clock epoch.
    pub fn unregister(&mut self, target: PresentationTarget) -> Result<(), PresentationError> {
        self.state(target)?;
        self.targets.remove(&target.id);
        Ok(())
    }

    /// Pause/resume a target. Transitions clear cadence/queue, preserving serial
    /// monotonicity and cumulative counters. Resume's first frame can start now.
    pub fn set_active(
        &mut self,
        target: PresentationTarget,
        active: bool,
        now: PresentationTimestamp,
    ) -> Result<(), PresentationError> {
        let state = self.state_mut(target)?;
        state.validate_time(now)?;
        if state.active != active {
            state.stats.unpresented_retirements = state
                .stats
                .unpresented_retirements
                .saturating_add(state.pending.len() as u64);
            state.pending.clear();
            state.pacer.reset();
            state.last_actual = None;
            state.estimate = None;
            state.stats.last_present = None;
            state.stats.input_to_present_ns = None;
            state.fallback_deadline = now.ns;
            state.active = active;
        }
        Ok(())
    }

    /// Record successful submission. Input must have been sampled at/before it.
    /// This consumes queue capacity but provides no actual presentation evidence.
    pub fn submitted(
        &mut self,
        target: PresentationTarget,
        frame: u64,
        at: PresentationTimestamp,
        input: Option<PresentationTimestamp>,
    ) -> Result<(), PresentationError> {
        let state = self.state_mut(target)?;
        let at = state.validate_time(at)?;
        let input = input.map(|time| state.validate_time(time)).transpose()?;
        if !state.active {
            return Err(PresentationError::Inactive);
        }
        if input.is_some_and(|time| time > at) {
            return Err(PresentationError::TimestampOrder);
        }
        if state
            .last_submitted
            .is_some_and(|(serial, time)| frame <= serial || at < time)
        {
            return Err(PresentationError::NonMonotonic);
        }
        if state.pending.len() >= state.gate.max_frames_in_flight() as usize {
            return Err(PresentationError::QueueFull);
        }
        state.pending.insert(frame, Submission { at, input });
        state.last_submitted = Some((frame, at));
        state.fallback_deadline = at.saturating_add(state.pacer.target_interval_ns());
        Ok(())
    }

    /// Retire a dropped frame, or a GPU-completed frame on an unsupported
    /// backend. Releases capacity without feeding the actual-present predictor.
    pub fn retire_unpresented(
        &mut self,
        target: PresentationTarget,
        frame: u64,
    ) -> Result<(), PresentationError> {
        let state = self.state_mut(target)?;
        state
            .pending
            .remove(&frame)
            .ok_or(PresentationError::UnknownFrame)?;
        state.estimate = state.estimate.filter(|(serial, _)| *serial != frame);
        state.stats.unpresented_retirements = state.stats.unpresented_retirements.saturating_add(1);
        Ok(())
    }

    /// Validate and consume an actual backend sample. Rejections do not change
    /// the predictor, queue, estimates or latency (only the rejection counter).
    pub fn feedback(
        &mut self,
        sample: PresentationFeedback,
        now: PresentationTimestamp,
    ) -> Result<PresentInfo, PresentationError> {
        let state = self.state_mut(sample.target)?;
        let result = state.accept_feedback(sample, now);
        if result.is_err() {
            state.stats.rejected_samples = state.stats.rejected_samples.saturating_add(1);
        }
        result
    }

    /// Store an explicitly estimated display time for an outstanding frame.
    /// This neither retires it nor enters the actual-present smoother/latency.
    pub fn estimate(
        &mut self,
        target: PresentationTarget,
        frame: u64,
        at: PresentationTimestamp,
    ) -> Result<(), PresentationError> {
        let state = self.state_mut(target)?;
        let at = state.validate_time(at)?;
        let submission = state
            .pending
            .get(&frame)
            .ok_or(PresentationError::UnknownFrame)?;
        if at < submission.at {
            return Err(PresentationError::TimestampOrder);
        }
        if state
            .estimate
            .is_some_and(|(serial, time)| frame < serial || at < time)
        {
            return Err(PresentationError::NonMonotonic);
        }
        state.estimate = Some((frame, at));
        state.stats.estimated_samples = state.stats.estimated_samples.saturating_add(1);
        Ok(())
    }

    /// Predict the next frame's display deadline and ask the time kernel when
    /// work should start. `estimated_work_ns` includes CPU and GPU work.
    pub fn plan(
        &self,
        target: PresentationTarget,
        now: PresentationTimestamp,
        estimated_work_ns: u64,
    ) -> Result<PresentationPlan, PresentationError> {
        let state = self.state(target)?;
        let now = state.validate_time(now)?;
        let interval = state.pacer.target_interval_ns();
        let count = state.pending.len() as u32;
        let (deadline, basis) = if let Some((_, last)) = state.last_actual {
            let predicted = state
                .pacer
                .next_present_ns()
                .saturating_add(interval.saturating_mul(u64::from(count)));
            let predicted = state
                .config
                .vrr
                .map_or(predicted, |vrr| vrr.earliest_present_ns(predicted, last));
            let FeedbackSupport::Actual(source) = state.config.feedback else {
                unreachable!("validated actual feedback requires a provider")
            };
            (predicted, DeadlineBasis::ActualFeedback(source))
        } else if let Some((frame, estimated)) = state.estimate {
            let later = state
                .pending
                .keys()
                .filter(|&&serial| serial > frame)
                .count() as u64;
            (
                estimated.saturating_add(interval.saturating_mul(later.saturating_add(1))),
                DeadlineBasis::Estimated,
            )
        } else {
            let basis = match state.config.feedback {
                FeedbackSupport::Actual(_) => DeadlineBasis::AwaitingActual,
                FeedbackSupport::Unsupported(reason) => DeadlineBasis::Unsupported(reason),
            };
            (state.fallback_deadline, basis)
        };
        // Saturating time arithmetic also handles a missed/past deadline: start
        // now, rather than continually moving the target and starving updates.
        let decision = if state.active {
            state.gate.decide(now, deadline, estimated_work_ns, count)
        } else {
            BeginDecision::QueueFull
        };
        Ok(PresentationPlan {
            decision,
            deadline: state.timestamp(deadline),
            begin_at: state.timestamp(state.gate.latest_begin_ns(deadline, estimated_work_ns)),
            basis,
            frames_in_flight: count,
            vrr: state
                .config
                .vrr
                .map(|vrr| vrr.classify(state.config.interval_ns.max(1))),
            active: state.active,
        })
    }

    /// Read cumulative evidence and validation statistics for this target.
    pub fn stats(
        &self,
        target: PresentationTarget,
    ) -> Result<PresentationStats, PresentationError> {
        Ok(self.state(target)?.stats)
    }
}

/// Optional window-runner policy. The render bridge selects the pacing window
/// explicitly and uses `clock` for submissions/feedback poll timestamps.
/// Without this resource the window runner retains its ordinary frame limiter.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Resource)]
pub struct PresentationRunner {
    /// One target driving the shared App frame, independently of secondary windows.
    pub target: PresentationTarget,
    /// CPU + GPU work estimate used by the time kernel's begin-work gate.
    pub estimated_work_ns: u64,
    /// Shared monotonic epoch used by this host and the platform adapter.
    pub clock: PresentationHostClock,
    /// Last query result, observable even when the App frame was deferred.
    pub last_plan: Option<PresentationPlan>,
    /// Last validation error. Errors stop updates and request an error exit.
    pub last_error: Option<PresentationError>,
}

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
enum PresentationBridgeEvent {
    Register {
        target: PresentationTarget,
        config: PresentationConfig,
        now: PresentationTimestamp,
    },
    Unregister {
        target: PresentationTarget,
    },
    Active {
        target: PresentationTarget,
        active: bool,
        now: PresentationTimestamp,
    },
    Submitted {
        target: PresentationTarget,
        frame: u64,
        at: PresentationTimestamp,
        input: Option<PresentationTimestamp>,
    },
    RetireUnpresented {
        target: PresentationTarget,
        frame: u64,
    },
    Feedback {
        sample: PresentationFeedback,
        now: PresentationTimestamp,
    },
}

/// Cross-world presentation event bridge.
///
/// The render sub-app owns surface state and emits lifecycle/submission events;
/// the main App drains them immediately before its next begin-work decision.
/// This keeps the pacing resource in the App world while allowing completion
/// callbacks to run on a render/backend thread. Events carry the host clock
/// domain and are never interpreted as actual display feedback unless they are
/// submitted through [`Self::feedback`].
#[cfg(feature = "std")]
#[derive(Clone, Resource)]
pub struct PresentationBridge {
    clock: PresentationHostClock,
    events: Arc<std::sync::Mutex<VecDeque<PresentationBridgeEvent>>>,
}

#[cfg(feature = "std")]
impl PresentationBridge {
    /// Creates a bridge using the same monotonic host epoch as the runner.
    #[must_use]
    pub fn new(clock: PresentationHostClock) -> Self {
        Self {
            clock,
            events: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        }
    }

    /// The shared host clock used for submission and lifecycle timestamps.
    #[must_use]
    pub const fn clock(&self) -> PresentationHostClock {
        self.clock
    }

    /// Register a surface incarnation and its pacing/capability contract.
    pub fn register(&self, target: PresentationTarget, config: PresentationConfig) {
        self.push(PresentationBridgeEvent::Register {
            target,
            config,
            now: self.clock.now(),
        });
    }

    /// Remove a surface incarnation after its backend resources are no longer
    /// referenced. Stale feedback is rejected after this point.
    pub fn unregister(&self, target: PresentationTarget) {
        self.push(PresentationBridgeEvent::Unregister { target });
    }

    /// Mark a surface active or inactive (hidden, occluded, suspended, or
    /// restored). Inactive surfaces do not admit App input or simulation.
    pub fn set_active(&self, target: PresentationTarget, active: bool) {
        self.push(PresentationBridgeEvent::Active {
            target,
            active,
            now: self.clock.now(),
        });
    }

    /// Record a successful backend submission. This is not display feedback.
    pub fn submitted(
        &self,
        target: PresentationTarget,
        frame: u64,
        input: Option<PresentationTimestamp>,
    ) {
        self.push(PresentationBridgeEvent::Submitted {
            target,
            frame,
            at: self.clock.now(),
            input,
        });
    }

    /// Release pacing capacity for a discarded frame or a completed frame when
    /// no actual display provider exists. This never updates the present
    /// predictor or actual latency statistics.
    pub fn retire_unpresented(&self, target: PresentationTarget, frame: u64) {
        self.push(PresentationBridgeEvent::RetireUnpresented { target, frame });
    }

    /// Supply a real platform/compositor timestamp after converting it into the
    /// bridge's host clock domain. Do not call this for queue submission or GPU
    /// completion events.
    pub fn feedback(&self, sample: PresentationFeedback) {
        self.push(PresentationBridgeEvent::Feedback {
            sample,
            now: self.clock.now(),
        });
    }

    fn push(&self, event: PresentationBridgeEvent) {
        let mut events = self
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        events.push_back(event);
    }

    /// Drain render/backend events into the App pacing resource in FIFO order.
    /// Processing continues after an invalid event so one stale surface cannot
    /// block unrelated windows. The last rejection is returned for observability.
    pub fn drain_into(&self, pacing: &mut PresentationPacing) -> PresentationBridgeReport {
        let events = {
            let mut queued = self
                .events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            core::mem::take(&mut *queued)
        };
        let mut report = PresentationBridgeReport::default();
        for event in events {
            let result = match event {
                PresentationBridgeEvent::Register {
                    target,
                    config,
                    now,
                } => pacing.register(target, config, now),
                PresentationBridgeEvent::Unregister { target } => pacing.unregister(target),
                PresentationBridgeEvent::Active {
                    target,
                    active,
                    now,
                } => pacing.set_active(target, active, now),
                PresentationBridgeEvent::Submitted {
                    target,
                    frame,
                    at,
                    input,
                } => pacing.submitted(target, frame, at, input),
                PresentationBridgeEvent::RetireUnpresented { target, frame } => {
                    pacing.retire_unpresented(target, frame)
                }
                PresentationBridgeEvent::Feedback { sample, now } => {
                    pacing.feedback(sample, now).map(|_| ())
                }
            };
            match result {
                Ok(()) => report.applied = report.applied.saturating_add(1),
                Err(error) => {
                    report.rejected = report.rejected.saturating_add(1);
                    report.last_error = Some(error);
                }
            }
        }
        report
    }
}

/// Result of draining one render-to-App bridge batch.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentationBridgeReport {
    /// Events applied to the pacing state.
    pub applied: u64,
    /// Events rejected by deterministic lifecycle/timestamp validation.
    pub rejected: u64,
    /// Last rejection, if any.
    pub last_error: Option<PresentationError>,
}

/// A copyable host epoch, shareable with the renderer's timestamp adapter.
/// Native clock conversion must be calibrated by that adapter; this clock
/// does not interpret GPU timestamps or native display-clock ticks.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
pub struct PresentationHostClock {
    domain: PresentationClockDomain,
    epoch: std::time::Instant,
}

#[cfg(feature = "std")]
impl PresentationHostClock {
    /// Start a new monotonic epoch. Host must assign an unused domain id.
    pub fn new(domain: PresentationClockDomain) -> Self {
        Self {
            domain,
            epoch: std::time::Instant::now(),
        }
    }

    /// Current host timestamp, saturating only after more than 584 years.
    pub fn now(&self) -> PresentationTimestamp {
        PresentationTimestamp {
            domain: self.domain,
            ns: self.epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        }
    }

    /// Host epoch for an adapter that has explicitly calibrated its native
    /// display clock to `std::time::Instant`. Never use callback arrival time
    /// as a substitute for the actual presented timestamp.
    pub fn epoch(&self) -> std::time::Instant {
        self.epoch
    }

    /// Convert a timestamp in this clock domain back to an OS wakeup instant.
    /// Returns `None` for a foreign clock domain or an unrepresentable duration.
    pub fn instant_at(&self, timestamp: PresentationTimestamp) -> Option<std::time::Instant> {
        if timestamp.domain != self.domain {
            return None;
        }
        self.epoch
            .checked_add(core::time::Duration::from_nanos(timestamp.ns))
    }
}

#[cfg(feature = "std")]
impl PresentationRunner {
    /// Select a target and its host epoch. Register the target separately in
    /// [`PresentationPacing`] with this same clock before starting the runner.
    pub fn new(
        target: PresentationTarget,
        clock: PresentationHostClock,
        estimated_work_ns: u64,
    ) -> Self {
        Self {
            target,
            estimated_work_ns,
            clock,
            last_plan: None,
            last_error: None,
        }
    }
}

/// Window runner's narrow integration seam. Returns true when presentation
/// pacing is installed (so the legacy frame-end limiter must be skipped).
/// Poll render-bridge feedback before this call. OS events continue to be
/// processed while ECS input/simulation/render updates are held by the gate.
///
#[cfg(feature = "std")]
pub(crate) fn run_window_frame(app: &mut App) -> bool {
    let Some(runner) = app.world().get_resource::<PresentationRunner>().copied() else {
        app.update();
        return false;
    };
    let bridge_report =
        if let Some(bridge) = app.world().get_resource::<PresentationBridge>().cloned() {
            if let Some(pacing) = app.world_mut().get_resource_mut::<PresentationPacing>() {
                Some(bridge.drain_into(pacing))
            } else {
                None
            }
        } else {
            None
        };
    if let Some(report) = bridge_report.filter(|report| report.last_error.is_some()) {
        if let Some(state) = app.world_mut().get_resource_mut::<PresentationRunner>() {
            state.last_error = report.last_error;
        }
        if let Some(request) = app.world_mut().get_resource_mut::<crate::AppExitRequest>() {
            request.send_error();
        }
        return true;
    }
    let result = app.update_with_presentation(
        runner.target,
        runner.clock.now(),
        runner.estimated_work_ns,
        |_| {},
    );
    if let Some(state) = app.world_mut().get_resource_mut::<PresentationRunner>() {
        state.last_plan = result.as_ref().ok().copied();
        state.last_error = result.as_ref().err().copied();
    }
    match result {
        Ok(plan) => {
            let _ = plan;
        }
        Err(_) => {
            if let Some(request) = app.world_mut().get_resource_mut::<crate::AppExitRequest>() {
                request.send_error();
            }
        }
    }
    true
}

/// Return the next OS wakeup requested by the presentation gate. The winit
/// adapter installs this as `ControlFlow::WaitUntil`; no thread sleep or 1 ms
/// polling loop is used. A full queue wakes at its predicted fallback/display
/// deadline and can also be interrupted by platform events.
#[cfg(feature = "std")]
pub(crate) fn next_window_wakeup(app: &App) -> Option<std::time::Instant> {
    let runner = app.world().get_resource::<PresentationRunner>()?;
    let plan = runner.last_plan?;
    let now = runner.clock.now();
    let timestamp = match plan.decision {
        BeginDecision::Wait { .. } => plan.begin_at,
        BeginDecision::QueueFull | BeginDecision::Begin => plan.deadline,
    };
    let timestamp = if timestamp.ns <= now.ns {
        PresentationTimestamp {
            domain: now.domain,
            ns: now.ns.saturating_add(1),
        }
    } else {
        timestamp
    };
    runner.clock.instant_at(timestamp)
}

impl App {
    /// Runner entry point: query the selected window's gate before sampling
    /// input. On `Begin`, invoke `sample_input` then the ordinary App frame
    /// (including fixed loop and SubApps); on `Wait`/`QueueFull`, do neither.
    /// Poll backend feedback before calling. Publish a successful submission
    /// through [`PresentationPacing::submitted`] in the renderer bridge.
    ///
    /// The unpaced `update()` remains available for headless/manual runners.
    pub fn update_with_presentation(
        &mut self,
        target: PresentationTarget,
        now: PresentationTimestamp,
        estimated_work_ns: u64,
        sample_input: impl FnOnce(&mut App),
    ) -> Result<PresentationPlan, PresentationError> {
        let plan = self
            .world()
            .get_resource::<PresentationPacing>()
            .ok_or(PresentationError::NotInstalled)?
            .plan(target, now, estimated_work_ns)?;
        if plan.decision == BeginDecision::Begin {
            sample_input(self);
            self.update();
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAIN: PresentationClockDomain = PresentationClockDomain(7);
    const TARGET: PresentationTarget = PresentationTarget {
        id: 3,
        generation: 1,
    };

    fn ts(ns: u64) -> PresentationTimestamp {
        PresentationTimestamp { domain: DOMAIN, ns }
    }

    fn config(feedback: FeedbackSupport) -> PresentationConfig {
        PresentationConfig {
            feedback,
            interval_ns: 10,
            max_frames_in_flight: 2,
            safety_margin_ns: 1,
            vrr: None,
        }
    }

    fn pacing(feedback: FeedbackSupport) -> PresentationPacing {
        let mut pacing = PresentationPacing::default();
        pacing.register(TARGET, config(feedback), ts(0)).unwrap();
        pacing
    }

    #[test]
    fn actual_feedback_drives_prediction_without_using_submit_time_as_present() {
        let mut pacing = pacing(FeedbackSupport::Actual(
            PresentationSource::MetalPresentedTime,
        ));
        pacing.submitted(TARGET, 11, ts(0), Some(ts(0))).unwrap();
        pacing
            .feedback(
                PresentationFeedback {
                    target: TARGET,
                    frame: 11,
                    source: PresentationSource::MetalPresentedTime,
                    presented_at: ts(10),
                },
                ts(10),
            )
            .unwrap();

        let stats = pacing.stats(TARGET).unwrap();
        assert_eq!(stats.actual_samples, 1);
        assert_eq!(stats.input_to_present_ns, Some(10));
        assert!(stats.last_present.is_some());
        let plan = pacing.plan(TARGET, ts(10), 0).unwrap();
        assert_eq!(
            plan.basis,
            DeadlineBasis::ActualFeedback(PresentationSource::MetalPresentedTime)
        );
        assert_eq!(plan.deadline.ns, 20);
    }

    #[test]
    fn feedback_rejections_preserve_predictor_and_queue() {
        let mut pacing = pacing(FeedbackSupport::Actual(
            PresentationSource::MetalPresentedTime,
        ));
        pacing.submitted(TARGET, 1, ts(5), None).unwrap();
        let before = pacing.plan(TARGET, ts(5), 0).unwrap();

        let wrong_source = pacing.feedback(
            PresentationFeedback {
                target: TARGET,
                frame: 1,
                source: PresentationSource::WaylandPresentationTime,
                presented_at: ts(6),
            },
            ts(6),
        );
        assert_eq!(wrong_source, Err(PresentationError::Source));

        let wrong_domain = pacing.feedback(
            PresentationFeedback {
                target: TARGET,
                frame: 1,
                source: PresentationSource::MetalPresentedTime,
                presented_at: PresentationTimestamp {
                    domain: PresentationClockDomain(8),
                    ns: 6,
                },
            },
            ts(6),
        );
        assert_eq!(wrong_domain, Err(PresentationError::ClockDomain));

        let future = pacing.feedback(
            PresentationFeedback {
                target: TARGET,
                frame: 1,
                source: PresentationSource::MetalPresentedTime,
                presented_at: ts(7),
            },
            ts(6),
        );
        assert_eq!(future, Err(PresentationError::FutureTimestamp));

        let after = pacing.plan(TARGET, ts(5), 0).unwrap();
        assert_eq!(before, after);
        assert_eq!(pacing.stats(TARGET).unwrap().rejected_samples, 3);
        assert_eq!(after.frames_in_flight, 1);
    }

    #[test]
    fn unsupported_feedback_is_distinct_from_estimates_and_retirement() {
        let mut pacing = pacing(FeedbackSupport::Unsupported(
            UnsupportedFeedback::BackendApi,
        ));
        pacing.submitted(TARGET, 1, ts(0), None).unwrap();
        assert_eq!(
            pacing.feedback(
                PresentationFeedback {
                    target: TARGET,
                    frame: 1,
                    source: PresentationSource::MetalPresentedTime,
                    presented_at: ts(10),
                },
                ts(10),
            ),
            Err(PresentationError::Unsupported)
        );
        pacing.estimate(TARGET, 1, ts(10)).unwrap();
        assert_eq!(pacing.stats(TARGET).unwrap().actual_samples, 0);
        assert_eq!(pacing.stats(TARGET).unwrap().estimated_samples, 1);
        assert_eq!(
            pacing.plan(TARGET, ts(10), 0).unwrap().basis,
            DeadlineBasis::Estimated
        );
        pacing.retire_unpresented(TARGET, 1).unwrap();
        let stats = pacing.stats(TARGET).unwrap();
        assert_eq!(stats.unpresented_retirements, 1);
        assert_eq!(stats.input_to_present_ns, None);
    }

    #[test]
    fn queue_backpressure_and_surface_generation_are_isolated() {
        let mut pacing = pacing(FeedbackSupport::Unsupported(UnsupportedFeedback::Offscreen));
        pacing.submitted(TARGET, 1, ts(0), None).unwrap();
        pacing.submitted(TARGET, 2, ts(1), None).unwrap();
        assert_eq!(
            pacing.submitted(TARGET, 3, ts(2), None),
            Err(PresentationError::QueueFull)
        );
        assert_eq!(
            pacing.set_active(
                PresentationTarget {
                    id: TARGET.id,
                    generation: 2
                },
                true,
                ts(3),
            ),
            Err(PresentationError::StaleGeneration)
        );
        pacing.set_active(TARGET, false, ts(3)).unwrap();
        assert_eq!(
            pacing.plan(TARGET, ts(3), 0).unwrap().decision,
            BeginDecision::QueueFull
        );
        pacing.set_active(TARGET, true, ts(4)).unwrap();
        assert_eq!(pacing.plan(TARGET, ts(4), 0).unwrap().frames_in_flight, 0);
        pacing.submitted(TARGET, 4, ts(4), None).unwrap();
        assert_eq!(pacing.plan(TARGET, ts(4), 0).unwrap().frames_in_flight, 1);
    }

    #[test]
    fn begin_gate_holds_input_until_the_reflex_boundary() {
        let mut app = App::new();
        app.insert_resource(pacing(FeedbackSupport::Unsupported(
            UnsupportedFeedback::Offscreen,
        )));
        let sampled = std::cell::Cell::new(0_u32);

        let first = app
            .update_with_presentation(TARGET, ts(0), 0, |_| sampled.set(sampled.get() + 1))
            .unwrap();
        assert_eq!(first.decision, BeginDecision::Begin);
        assert_eq!(sampled.get(), 1);

        app.world_mut()
            .resource_mut::<PresentationPacing>()
            .submitted(TARGET, 1, ts(0), None)
            .unwrap();
        let held = app
            .update_with_presentation(TARGET, ts(0), 1, |_| sampled.set(sampled.get() + 1))
            .unwrap();
        assert!(matches!(held.decision, BeginDecision::Wait { .. }));
        assert_eq!(sampled.get(), 1);

        let late = app
            .update_with_presentation(TARGET, ts(10), 1, |_| sampled.set(sampled.get() + 1))
            .unwrap();
        assert_eq!(late.decision, BeginDecision::Begin);
        assert_eq!(sampled.get(), 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn bridge_applies_fifo_lifecycle_and_submission_events() {
        let clock = PresentationHostClock::new(DOMAIN);
        let bridge = PresentationBridge::new(clock);
        let mut pacing = PresentationPacing::default();
        bridge.register(
            TARGET,
            config(FeedbackSupport::Unsupported(UnsupportedFeedback::Offscreen)),
        );
        bridge.set_active(TARGET, true);
        bridge.submitted(TARGET, 1, None);
        let report = bridge.drain_into(&mut pacing);
        assert_eq!(report.rejected, 0);
        assert_eq!(report.applied, 3);
        assert_eq!(
            pacing
                .plan(TARGET, clock.now(), 0)
                .unwrap()
                .frames_in_flight,
            1
        );

        bridge.retire_unpresented(TARGET, 1);
        let report = bridge.drain_into(&mut pacing);
        assert_eq!(
            report,
            PresentationBridgeReport {
                applied: 1,
                rejected: 0,
                last_error: None
            }
        );
        assert_eq!(pacing.stats(TARGET).unwrap().unpresented_retirements, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn host_clock_rejects_foreign_domain_when_building_wakeup() {
        let clock = PresentationHostClock::new(DOMAIN);
        assert!(clock.instant_at(ts(1)).is_some());
        assert!(clock
            .instant_at(PresentationTimestamp {
                domain: PresentationClockDomain(99),
                ns: 1,
            })
            .is_none());
    }
}
