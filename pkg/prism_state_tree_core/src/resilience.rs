//! Fixed-step budget, circuit-breaker, and per-tree isolation contracts.
//!
//! The contract is deliberately independent from the StateTree cursor. A host
//! charges migrations, cascade depth, and work units while it executes a
//! frame, then commits one structured result. No wall clock is consulted:
//! circuit recovery is driven only by the caller-supplied fixed-step frame
//! index. Safe mode tells the host to retain its last known-good active
//! configuration; it does not own or clone that configuration.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

/// Stable identity used to isolate resilience state and diagnostics per tree.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TreeId(pub u64);

/// A fixed-step frame identifier supplied by the host.
pub type FrameIndex = u64;

/// Hard upper bound for one event buffer.
pub const MAX_EVENT_CAPACITY: usize = 256;

/// A tree's fixed-step resource limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameBudgetLimits {
    migrations: u32,
    cascade_depth: u32,
    work_units: u64,
}

impl FrameBudgetLimits {
    /// Creates limits for migrations, maximum cascade depth, and work units.
    ///
    /// A zero value is valid and intentionally disables that category of
    /// work. This makes a zero budget useful for a deliberate probe or a
    /// caller that only wants to observe admission without executing work.
    #[must_use]
    pub const fn new(migrations: u32, cascade_depth: u32, work_units: u64) -> Self {
        Self {
            migrations,
            cascade_depth,
            work_units,
        }
    }

    /// Returns the maximum migrations allowed in one frame.
    #[must_use]
    pub const fn migrations(self) -> u32 {
        self.migrations
    }

    /// Returns the maximum nested cascade depth allowed in one frame.
    #[must_use]
    pub const fn cascade_depth(self) -> u32 {
        self.cascade_depth
    }

    /// Returns the maximum work units allowed in one frame.
    #[must_use]
    pub const fn work_units(self) -> u64 {
        self.work_units
    }
}

/// Work already charged to a fixed-step frame budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetUsage {
    /// Number of migrations charged.
    pub migrations: u32,
    /// Greatest cascade depth observed.
    pub cascade_depth: u32,
    /// Number of abstract work units charged.
    pub work_units: u64,
}

/// A fixed-step frame budget with no time source or platform clock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameBudget {
    limits: FrameBudgetLimits,
    frame: Option<FrameIndex>,
    usage: BudgetUsage,
}

impl FrameBudget {
    /// Creates an unused budget with the supplied limits.
    #[must_use]
    pub const fn new(limits: FrameBudgetLimits) -> Self {
        Self {
            limits,
            frame: None,
            usage: BudgetUsage {
                migrations: 0,
                cascade_depth: 0,
                work_units: 0,
            },
        }
    }

    /// Returns the immutable limits.
    #[must_use]
    pub const fn limits(&self) -> FrameBudgetLimits {
        self.limits
    }

    /// Returns the frame for which the budget was most recently reset.
    #[must_use]
    pub const fn frame(&self) -> Option<FrameIndex> {
        self.frame
    }

    /// Returns current usage without changing the budget.
    #[must_use]
    pub const fn usage(&self) -> BudgetUsage {
        self.usage
    }

    /// Returns remaining work units in the current frame.
    #[must_use]
    pub const fn remaining_work_units(&self) -> u64 {
        self.limits.work_units.saturating_sub(self.usage.work_units)
    }

    /// Resets all counters for a new fixed-step frame.
    pub fn begin_frame(&mut self, frame: FrameIndex) {
        self.frame = Some(frame);
        self.usage = BudgetUsage::default();
    }

    /// Charges one migration and returns a structured limit failure if full.
    pub fn charge_migration(&mut self) -> Result<(), BudgetExhaustion> {
        if self.usage.migrations >= self.limits.migrations {
            return Err(BudgetExhaustion::Migrations {
                limit: self.limits.migrations,
                attempted: self.usage.migrations.saturating_add(1),
            });
        }
        self.usage.migrations += 1;
        Ok(())
    }

    /// Records a cascade depth, rejecting a depth above the configured limit.
    ///
    /// Depth is a peak measurement rather than a cumulative counter. Reusing
    /// the method while unwinding a cascade is therefore allocation-free and
    /// does not make shallow sibling branches consume one another's depth.
    pub fn enter_cascade(&mut self, depth: u32) -> Result<(), BudgetExhaustion> {
        if depth > self.limits.cascade_depth {
            return Err(BudgetExhaustion::CascadeDepth {
                limit: self.limits.cascade_depth,
                attempted: depth,
            });
        }
        self.usage.cascade_depth = self.usage.cascade_depth.max(depth);
        Ok(())
    }

    /// Charges abstract work units without referring to elapsed time.
    pub fn charge_work(&mut self, units: u64) -> Result<(), BudgetExhaustion> {
        let Some(total) = self.usage.work_units.checked_add(units) else {
            return Err(BudgetExhaustion::WorkUnits {
                limit: self.limits.work_units,
                attempted: u64::MAX,
                consumed: self.usage.work_units,
            });
        };
        if total > self.limits.work_units {
            return Err(BudgetExhaustion::WorkUnits {
                limit: self.limits.work_units,
                attempted: total,
                consumed: self.usage.work_units,
            });
        }
        self.usage.work_units = total;
        Ok(())
    }
}

/// The category of a fixed-step budget failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetExhaustion {
    /// The next migration would exceed the frame migration limit.
    Migrations {
        /// Configured migration limit.
        limit: u32,
        /// Count that the rejected migration would have produced.
        attempted: u32,
    },
    /// A cascade entered deeper than the configured limit.
    CascadeDepth {
        /// Configured depth limit.
        limit: u32,
        /// Rejected depth.
        attempted: u32,
    },
    /// Work units would exceed the configured total.
    WorkUnits {
        /// Configured work-unit limit.
        limit: u64,
        /// Total that the rejected charge would have produced.
        attempted: u64,
        /// Work units already consumed before rejection.
        consumed: u64,
    },
}

/// Failure while charging work through a per-tree coordinator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetChargeError {
    /// No frame was admitted, so the charge has no deterministic frame owner.
    NoActiveFrame,
    /// The charge would exceed the admitted frame's fixed budget.
    Exhausted(BudgetExhaustion),
}

/// Configures the fixed-step circuit breaker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CircuitBreakerConfig {
    failure_threshold: u32,
    open_for_frames: u64,
}

impl CircuitBreakerConfig {
    /// Creates a breaker configuration.
    ///
    /// `failure_threshold` must be non-zero. `open_for_frames` is a fixed-step
    /// cooldown; zero still waits until the next fixed-step frame before a
    /// probe, preventing repeated probes in one frame.
    pub const fn new(
        failure_threshold: u32,
        open_for_frames: u64,
    ) -> Result<Self, CircuitConfigError> {
        if failure_threshold == 0 {
            return Err(CircuitConfigError::ZeroFailureThreshold);
        }
        Ok(Self {
            failure_threshold,
            open_for_frames,
        })
    }

    /// Returns the consecutive-failure count required to open the breaker.
    #[must_use]
    pub const fn failure_threshold(self) -> u32 {
        self.failure_threshold
    }

    /// Returns the fixed-step number of frames before a probe is admitted.
    #[must_use]
    pub const fn open_for_frames(self) -> u64 {
        self.open_for_frames
    }

    const fn retry_at(self, opened_at: FrameIndex) -> FrameIndex {
        let cooldown = if self.open_for_frames == 0 {
            1
        } else {
            self.open_for_frames
        };
        opened_at.saturating_add(cooldown)
    }
}

/// Failure while constructing a circuit breaker policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CircuitConfigError {
    /// A breaker must require at least one consecutive failure.
    ZeroFailureThreshold,
}

/// Observable circuit-breaker state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CircuitState {
    /// Normal execution; the value tracks consecutive failed frames.
    Closed {
        /// Number of consecutive failures since the last successful frame.
        consecutive_failures: u32,
    },
    /// Execution is isolated until the fixed-step cooldown expires.
    Open {
        /// Frame at which the breaker opened.
        opened_at: FrameIndex,
        /// Consecutive failures that caused the opening.
        failures: u32,
    },
    /// One bounded recovery probe is allowed.
    HalfOpen {
        /// Whether the one probe for this half-open interval is claimed.
        probe_claimed: bool,
    },
}

/// Admission result from a circuit breaker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CircuitAdmission {
    /// Normal execution is allowed.
    Run,
    /// A single recovery probe is allowed.
    Probe,
    /// The tree remains isolated until the reported frame.
    Denied {
        /// First frame at which another admission check may probe.
        retry_at: FrameIndex,
    },
}

/// A failure category that can trip a tree's breaker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureReason {
    /// The frame exceeded one of its fixed-step budgets.
    Budget(BudgetExhaustion),
    /// The host detected a semantic or runtime invariant violation.
    Invariant {
        /// Stable host-defined diagnostic code.
        code: u32,
    },
    /// The host rejected execution for a stable, structured reason.
    Host {
        /// Stable host-defined diagnostic code.
        code: u32,
    },
}

/// A permit returned for one admitted frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramePermit {
    frame: FrameIndex,
    probe: bool,
}

impl FramePermit {
    /// Returns the fixed-step frame represented by this permit.
    #[must_use]
    pub const fn frame(self) -> FrameIndex {
        self.frame
    }

    /// Returns whether this permit is the one bounded recovery probe.
    #[must_use]
    pub const fn is_probe(self) -> bool {
        self.probe
    }
}

/// Admission result for one tree at the beginning of a frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameAdmission {
    /// Execute the tree normally with this permit.
    Run(FramePermit),
    /// Execute a bounded recovery probe with this permit.
    Probe(FramePermit),
    /// Keep the last known-good configuration frozen.
    Isolated {
        /// First frame at which a probe may be attempted.
        retry_at: FrameIndex,
    },
}

/// Safe-mode state for one tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SafeMode {
    active: bool,
    entered_at: Option<FrameIndex>,
    last_good_frame: Option<FrameIndex>,
    reason: Option<FailureReason>,
}

impl SafeMode {
    /// Creates an inactive safe mode with no recorded good frame.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: false,
            entered_at: None,
            last_good_frame: None,
            reason: None,
        }
    }

    /// Returns whether the host must freeze the last known-good configuration.
    #[must_use]
    pub const fn is_active(self) -> bool {
        self.active
    }

    /// Returns the frame that entered safe mode, if any.
    #[must_use]
    pub const fn entered_at(self) -> Option<FrameIndex> {
        self.entered_at
    }

    /// Returns the most recent successfully completed frame.
    #[must_use]
    pub const fn last_good_frame(self) -> Option<FrameIndex> {
        self.last_good_frame
    }

    /// Returns the failure that caused safe mode.
    #[must_use]
    pub const fn reason(self) -> Option<FailureReason> {
        self.reason
    }

    /// Records a successful frame as the new freeze point.
    pub const fn mark_good(&mut self, frame: FrameIndex) {
        self.last_good_frame = Some(frame);
    }

    fn enter(&mut self, frame: FrameIndex, reason: FailureReason) -> bool {
        let changed = !self.active;
        if changed {
            self.active = true;
            self.entered_at = Some(frame);
        }
        self.reason = Some(reason);
        changed
    }

    fn exit(&mut self) -> bool {
        let changed = self.active;
        self.active = false;
        self.entered_at = None;
        self.reason = None;
        changed
    }
}

impl Default for SafeMode {
    fn default() -> Self {
        Self::new()
    }
}

/// A bounded structured resilience event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResilienceEventKind {
    /// A frame was admitted for normal execution.
    FrameStarted,
    /// A frame was admitted as a recovery probe.
    ProbeStarted,
    /// A frame was completed successfully.
    FrameSucceeded {
        /// Whether the successful frame was a probe.
        probe: bool,
        /// Budget usage at completion.
        usage: BudgetUsage,
    },
    /// Execution failed and the failure was recorded.
    FrameFailed {
        /// Whether the failed frame was a probe.
        probe: bool,
        /// Structured failure reason.
        reason: FailureReason,
    },
    /// A budget charge was rejected by the current frame budget.
    BudgetExhausted(BudgetExhaustion),
    /// The breaker entered Open state.
    CircuitOpened {
        /// Number of failures that opened the circuit.
        failures: u32,
    },
    /// The breaker entered HalfOpen state and awaits one probe.
    CircuitHalfOpened,
    /// A recovery probe succeeded and the breaker closed.
    ProbeSucceeded,
    /// A recovery probe failed and isolation was extended.
    ProbeFailed(FailureReason),
    /// The tree entered isolated safe mode.
    SafeModeEntered(FailureReason),
    /// A successful probe released the tree from safe mode.
    SafeModeExited,
    /// Admission was denied while the tree was isolated.
    FrameIsolated {
        /// First frame at which a probe may be attempted.
        retry_at: FrameIndex,
    },
}

/// One event with a stable per-tree sequence number.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResilienceEvent {
    /// Tree that emitted the event.
    pub tree: TreeId,
    /// Fixed-step frame associated with the event.
    pub frame: FrameIndex,
    /// Monotonic sequence within this controller until integer saturation.
    pub sequence: u64,
    /// Structured event payload.
    pub kind: ResilienceEventKind,
}

/// Bounded event storage that never grows during emission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventBuffer {
    events: Vec<ResilienceEvent>,
    capacity: usize,
    next_sequence: u64,
    overflowed: bool,
}

impl EventBuffer {
    /// Creates a buffer with a capacity no greater than [`MAX_EVENT_CAPACITY`].
    pub fn new(capacity: usize) -> Result<Self, EventBufferError> {
        if capacity > MAX_EVENT_CAPACITY {
            return Err(EventBufferError::CapacityLimit {
                requested: capacity,
                maximum: MAX_EVENT_CAPACITY,
            });
        }
        Ok(Self {
            events: Vec::with_capacity(capacity),
            capacity,
            next_sequence: 0,
            overflowed: false,
        })
    }

    /// Returns the configured capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns retained events in emission order.
    #[must_use]
    pub fn events(&self) -> &[ResilienceEvent] {
        &self.events
    }

    /// Returns whether an event was dropped after the fixed capacity filled.
    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Clears retained events while preserving the sequence domain.
    pub fn clear(&mut self) {
        self.events.clear();
        self.overflowed = false;
    }

    fn push(&mut self, tree: TreeId, frame: FrameIndex, kind: ResilienceEventKind) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.events.len() == self.capacity {
            self.overflowed = true;
            return;
        }
        self.events.push(ResilienceEvent {
            tree,
            frame,
            sequence,
            kind,
        });
    }
}

/// Failure while constructing bounded event storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventBufferError {
    /// The requested capacity exceeds the module's hard bound.
    CapacityLimit {
        /// Requested capacity.
        requested: usize,
        /// Maximum accepted capacity.
        maximum: usize,
    },
}

/// A circuit breaker driven only by fixed-step frame indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
}

impl CircuitBreaker {
    /// Creates a closed breaker.
    #[must_use]
    pub const fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed {
                consecutive_failures: 0,
            },
        }
    }

    /// Returns the configured policy.
    #[must_use]
    pub const fn config(&self) -> CircuitBreakerConfig {
        self.config
    }

    /// Returns the current state.
    #[must_use]
    pub const fn state(&self) -> CircuitState {
        self.state
    }

    /// Attempts to admit a frame or the single half-open probe.
    pub fn admit(&mut self, frame: FrameIndex) -> CircuitAdmission {
        match self.state {
            CircuitState::Closed { .. } => CircuitAdmission::Run,
            CircuitState::Open { opened_at, .. } => {
                let retry_at = self.config.retry_at(opened_at);
                if frame < retry_at {
                    CircuitAdmission::Denied { retry_at }
                } else {
                    self.state = CircuitState::HalfOpen {
                        probe_claimed: true,
                    };
                    CircuitAdmission::Probe
                }
            }
            CircuitState::HalfOpen { probe_claimed } => {
                if probe_claimed {
                    CircuitAdmission::Denied {
                        retry_at: frame.saturating_add(1),
                    }
                } else {
                    self.state = CircuitState::HalfOpen {
                        probe_claimed: true,
                    };
                    CircuitAdmission::Probe
                }
            }
        }
    }

    /// Records a successful admitted frame and closes a successful probe.
    pub fn record_success(&mut self) -> CircuitTransition {
        match self.state {
            CircuitState::Closed { .. } => {
                self.state = CircuitState::Closed {
                    consecutive_failures: 0,
                };
                CircuitTransition::None
            }
            CircuitState::HalfOpen { .. } => {
                self.state = CircuitState::Closed {
                    consecutive_failures: 0,
                };
                CircuitTransition::Closed
            }
            CircuitState::Open { .. } => CircuitTransition::None,
        }
    }

    /// Records a failed admitted frame and opens after the configured threshold.
    pub fn record_failure(&mut self, frame: FrameIndex) -> CircuitTransition {
        match self.state {
            CircuitState::Closed {
                consecutive_failures,
            } => {
                let failures = consecutive_failures.saturating_add(1);
                if failures >= self.config.failure_threshold {
                    self.state = CircuitState::Open {
                        opened_at: frame,
                        failures,
                    };
                    CircuitTransition::Opened { failures }
                } else {
                    self.state = CircuitState::Closed {
                        consecutive_failures: failures,
                    };
                    CircuitTransition::None
                }
            }
            CircuitState::HalfOpen { .. } => {
                self.state = CircuitState::Open {
                    opened_at: frame,
                    failures: self.config.failure_threshold,
                };
                CircuitTransition::Opened {
                    failures: self.config.failure_threshold,
                }
            }
            CircuitState::Open { .. } => CircuitTransition::None,
        }
    }
}

/// State transition produced by a breaker result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CircuitTransition {
    /// The state remained in its current class.
    None,
    /// The breaker opened after repeated failures.
    Opened {
        /// Failure count at opening.
        failures: u32,
    },
    /// A successful probe closed the breaker.
    Closed,
}

/// Errors from frame admission and completion sequencing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResilienceError {
    /// A second frame was started before the previous one completed.
    FrameAlreadyActive,
    /// Frame indices must advance strictly for one tree controller.
    FrameNotMonotonic {
        /// Most recently observed frame.
        previous: FrameIndex,
        /// Requested frame that was rejected.
        requested: FrameIndex,
    },
    /// A completion was submitted without an active frame.
    NoActiveFrame,
    /// A budget charge failed and the frame cannot be completed as successful.
    BudgetFailureUnresolved {
        /// The budget failure that must be reported or handled.
        reason: BudgetExhaustion,
    },
    /// A completion used a permit from another frame or tree controller.
    PermitMismatch,
}

/// Result supplied when an admitted frame completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameResult {
    /// The tree reached a stable, known-good completion point.
    Succeeded,
    /// The tree failed for a structured reason and must not commit new state.
    Failed(FailureReason),
}

/// Summary returned after an admitted frame is completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameReport {
    /// Fixed-step frame that was completed.
    pub frame: FrameIndex,
    /// Whether the completed frame was a recovery probe.
    pub probe: bool,
    /// Completion result.
    pub result: FrameResult,
    /// Budget usage at completion.
    pub usage: BudgetUsage,
    /// Circuit state after recording the result.
    pub circuit: CircuitState,
    /// Safe-mode state after recording the result.
    pub safe_mode: SafeMode,
}

/// Per-tree resilience coordinator.
///
/// Construct one value for each independently scheduled tree. It owns only
/// budget, breaker, safe-mode, and diagnostics state; the host still owns the
/// tree's active configuration and freezes that configuration when
/// [`SafeMode::is_active`] is true.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeResilience {
    tree: TreeId,
    budget: FrameBudget,
    breaker: CircuitBreaker,
    safe_mode: SafeMode,
    last_failure: Option<FailureReason>,
    events: EventBuffer,
    active_permit: Option<FramePermit>,
    last_frame: Option<FrameIndex>,
    budget_failure: Option<BudgetExhaustion>,
}

impl TreeResilience {
    /// Creates an isolated coordinator for one tree.
    pub fn new(
        tree: TreeId,
        budget: FrameBudgetLimits,
        breaker: CircuitBreakerConfig,
        event_capacity: usize,
    ) -> Result<Self, EventBufferError> {
        Ok(Self {
            tree,
            budget: FrameBudget::new(budget),
            breaker: CircuitBreaker::new(breaker),
            safe_mode: SafeMode::new(),
            last_failure: None,
            events: EventBuffer::new(event_capacity)?,
            active_permit: None,
            last_frame: None,
            budget_failure: None,
        })
    }

    /// Returns this coordinator's tree identity.
    #[must_use]
    pub const fn tree(&self) -> TreeId {
        self.tree
    }

    /// Returns the current frame budget.
    #[must_use]
    pub const fn budget(&self) -> &FrameBudget {
        &self.budget
    }

    /// Returns the current circuit breaker.
    #[must_use]
    pub const fn breaker(&self) -> &CircuitBreaker {
        &self.breaker
    }

    /// Returns the current safe-mode state.
    #[must_use]
    pub const fn safe_mode(&self) -> SafeMode {
        self.safe_mode
    }

    /// Returns the bounded event buffer.
    #[must_use]
    pub const fn events(&self) -> &EventBuffer {
        &self.events
    }

    /// Returns whether a frame is currently admitted and awaiting completion.
    #[must_use]
    pub const fn has_active_frame(&self) -> bool {
        self.active_permit.is_some()
    }

    /// Starts one fixed-step frame for this tree.
    pub fn begin_frame(&mut self, frame: FrameIndex) -> Result<FrameAdmission, ResilienceError> {
        if self.active_permit.is_some() {
            return Err(ResilienceError::FrameAlreadyActive);
        }
        if let Some(previous) = self.last_frame
            && frame <= previous
        {
            return Err(ResilienceError::FrameNotMonotonic {
                previous,
                requested: frame,
            });
        }
        self.last_frame = Some(frame);
        self.budget_failure = None;
        let admission = self.breaker.admit(frame);
        let decision = match admission {
            CircuitAdmission::Run => {
                if self.safe_mode.is_active() {
                    let retry_at = frame.saturating_add(1);
                    self.events.push(
                        self.tree,
                        frame,
                        ResilienceEventKind::FrameIsolated { retry_at },
                    );
                    FrameAdmission::Isolated { retry_at }
                } else {
                    let permit = FramePermit {
                        frame,
                        probe: false,
                    };
                    self.budget.begin_frame(frame);
                    self.active_permit = Some(permit);
                    self.events
                        .push(self.tree, frame, ResilienceEventKind::FrameStarted);
                    FrameAdmission::Run(permit)
                }
            }
            CircuitAdmission::Probe => {
                let permit = FramePermit { frame, probe: true };
                self.budget.begin_frame(frame);
                self.active_permit = Some(permit);
                self.events
                    .push(self.tree, frame, ResilienceEventKind::CircuitHalfOpened);
                self.events
                    .push(self.tree, frame, ResilienceEventKind::ProbeStarted);
                FrameAdmission::Probe(permit)
            }
            CircuitAdmission::Denied { retry_at } => {
                let reason = self.last_failure.unwrap_or(FailureReason::Host { code: 0 });
                if self.safe_mode.enter(frame, reason) {
                    self.events.push(
                        self.tree,
                        frame,
                        ResilienceEventKind::SafeModeEntered(reason),
                    );
                }
                self.events.push(
                    self.tree,
                    frame,
                    ResilienceEventKind::FrameIsolated { retry_at },
                );
                FrameAdmission::Isolated { retry_at }
            }
        };
        Ok(decision)
    }

    /// Charges one migration for the admitted frame.
    pub fn charge_migration(&mut self) -> Result<(), BudgetChargeError> {
        if self.active_permit.is_none() {
            return Err(BudgetChargeError::NoActiveFrame);
        }
        if let Some(reason) = self.budget_failure {
            return Err(BudgetChargeError::Exhausted(reason));
        }
        let result = self.budget.charge_migration();
        if let Err(reason) = result {
            self.budget_failure = Some(reason);
            let event = ResilienceEventKind::BudgetExhausted(reason);
            self.events
                .push(self.tree, self.budget.frame().unwrap_or_default(), event);
            return Err(BudgetChargeError::Exhausted(reason));
        }
        Ok(())
    }

    /// Records cascade depth for the admitted frame.
    pub fn enter_cascade(&mut self, depth: u32) -> Result<(), BudgetChargeError> {
        if self.active_permit.is_none() {
            return Err(BudgetChargeError::NoActiveFrame);
        }
        if let Some(reason) = self.budget_failure {
            return Err(BudgetChargeError::Exhausted(reason));
        }
        let result = self.budget.enter_cascade(depth);
        if let Err(reason) = result {
            self.budget_failure = Some(reason);
            self.events.push(
                self.tree,
                self.budget.frame().unwrap_or_default(),
                ResilienceEventKind::BudgetExhausted(reason),
            );
            return Err(BudgetChargeError::Exhausted(reason));
        }
        Ok(())
    }

    /// Charges abstract work units for the admitted frame.
    pub fn charge_work(&mut self, units: u64) -> Result<(), BudgetChargeError> {
        if self.active_permit.is_none() {
            return Err(BudgetChargeError::NoActiveFrame);
        }
        if let Some(reason) = self.budget_failure {
            return Err(BudgetChargeError::Exhausted(reason));
        }
        let result = self.budget.charge_work(units);
        if let Err(reason) = result {
            self.budget_failure = Some(reason);
            self.events.push(
                self.tree,
                self.budget.frame().unwrap_or_default(),
                ResilienceEventKind::BudgetExhausted(reason),
            );
            return Err(BudgetChargeError::Exhausted(reason));
        }
        Ok(())
    }

    /// Completes an admitted frame and updates breaker and safe-mode state.
    pub fn finish_frame(
        &mut self,
        permit: FramePermit,
        result: FrameResult,
    ) -> Result<FrameReport, ResilienceError> {
        if self.active_permit != Some(permit) {
            return Err(if self.active_permit.is_some() {
                ResilienceError::PermitMismatch
            } else {
                ResilienceError::NoActiveFrame
            });
        }
        if matches!(result, FrameResult::Succeeded)
            && let Some(reason) = self.budget_failure
        {
            return Err(ResilienceError::BudgetFailureUnresolved { reason });
        }
        self.active_permit = None;
        self.budget_failure = None;
        let usage = self.budget.usage();
        match result {
            FrameResult::Succeeded => {
                let transition = self.breaker.record_success();
                self.safe_mode.mark_good(permit.frame);
                self.events.push(
                    self.tree,
                    permit.frame,
                    ResilienceEventKind::FrameSucceeded {
                        probe: permit.probe,
                        usage,
                    },
                );
                if permit.probe {
                    self.events
                        .push(self.tree, permit.frame, ResilienceEventKind::ProbeSucceeded);
                }
                if matches!(transition, CircuitTransition::Closed) && self.safe_mode.exit() {
                    self.events
                        .push(self.tree, permit.frame, ResilienceEventKind::SafeModeExited);
                }
            }
            FrameResult::Failed(reason) => {
                self.last_failure = Some(reason);
                self.events.push(
                    self.tree,
                    permit.frame,
                    ResilienceEventKind::FrameFailed {
                        probe: permit.probe,
                        reason,
                    },
                );
                if permit.probe {
                    self.events.push(
                        self.tree,
                        permit.frame,
                        ResilienceEventKind::ProbeFailed(reason),
                    );
                }
                if let CircuitTransition::Opened { failures } =
                    self.breaker.record_failure(permit.frame)
                {
                    let entered = self.safe_mode.enter(permit.frame, reason);
                    self.events.push(
                        self.tree,
                        permit.frame,
                        ResilienceEventKind::CircuitOpened { failures },
                    );
                    if entered {
                        self.events.push(
                            self.tree,
                            permit.frame,
                            ResilienceEventKind::SafeModeEntered(reason),
                        );
                    }
                }
            }
        }
        Ok(FrameReport {
            frame: permit.frame,
            probe: permit.probe,
            result,
            usage,
            circuit: self.breaker.state(),
            safe_mode: self.safe_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BudgetChargeError, BudgetExhaustion, CircuitAdmission, CircuitBreaker,
        CircuitBreakerConfig, CircuitState, EventBuffer, EventBufferError, FailureReason,
        FrameAdmission, FrameBudget, FrameBudgetLimits, FrameResult, ResilienceEventKind, SafeMode,
        TreeId, TreeResilience,
    };

    fn config() -> CircuitBreakerConfig {
        CircuitBreakerConfig::new(2, 2).unwrap()
    }

    fn tree(id: u64) -> TreeResilience {
        TreeResilience::new(TreeId(id), FrameBudgetLimits::new(2, 3, 10), config(), 32).unwrap()
    }

    #[test]
    fn fixed_step_budget_tracks_all_categories_without_clock() {
        let mut budget = FrameBudget::new(FrameBudgetLimits::new(1, 2, 5));
        budget.begin_frame(41);
        assert_eq!(budget.frame(), Some(41));
        budget.charge_migration().unwrap();
        assert_eq!(
            budget.charge_migration(),
            Err(BudgetExhaustion::Migrations {
                limit: 1,
                attempted: 2,
            })
        );
        budget.enter_cascade(2).unwrap();
        assert_eq!(
            budget.enter_cascade(3),
            Err(BudgetExhaustion::CascadeDepth {
                limit: 2,
                attempted: 3,
            })
        );
        budget.charge_work(5).unwrap();
        assert_eq!(
            budget.charge_work(1),
            Err(BudgetExhaustion::WorkUnits {
                limit: 5,
                attempted: 6,
                consumed: 5,
            })
        );
        budget.begin_frame(42);
        assert_eq!(budget.usage().migrations, 0);
        assert_eq!(budget.usage().cascade_depth, 0);
        assert_eq!(budget.usage().work_units, 0);
    }

    #[test]
    fn breaker_progresses_closed_open_half_open_and_closed() {
        let mut breaker = CircuitBreaker::new(config());
        assert_eq!(breaker.admit(0), CircuitAdmission::Run);
        assert_eq!(breaker.record_failure(0), super::CircuitTransition::None);
        assert_eq!(breaker.admit(1), CircuitAdmission::Run);
        assert_eq!(
            breaker.record_failure(1),
            super::CircuitTransition::Opened { failures: 2 }
        );
        assert_eq!(
            breaker.state(),
            CircuitState::Open {
                opened_at: 1,
                failures: 2
            }
        );
        assert_eq!(breaker.admit(2), CircuitAdmission::Denied { retry_at: 3 });
        assert_eq!(breaker.admit(3), CircuitAdmission::Probe);
        assert_eq!(
            breaker.state(),
            CircuitState::HalfOpen {
                probe_claimed: true
            }
        );
        assert_eq!(breaker.admit(3), CircuitAdmission::Denied { retry_at: 4 });
        assert_eq!(breaker.record_success(), super::CircuitTransition::Closed);
        assert_eq!(
            breaker.state(),
            CircuitState::Closed {
                consecutive_failures: 0
            }
        );
    }

    #[test]
    fn failed_probe_reopens_without_wall_clock() {
        let mut breaker = CircuitBreaker::new(config());
        breaker.admit(0);
        breaker.record_failure(0);
        breaker.admit(1);
        breaker.record_failure(1);
        assert_eq!(breaker.admit(3), CircuitAdmission::Probe);
        assert_eq!(
            breaker.record_failure(3),
            super::CircuitTransition::Opened { failures: 2 }
        );
        assert_eq!(breaker.admit(4), CircuitAdmission::Denied { retry_at: 5 });
    }

    #[test]
    fn zero_cooldown_still_waits_for_the_next_fixed_frame() {
        let policy = CircuitBreakerConfig::new(1, 0).unwrap();
        let mut breaker = CircuitBreaker::new(policy);
        assert_eq!(breaker.admit(12), CircuitAdmission::Run);
        assert_eq!(
            breaker.record_failure(12),
            super::CircuitTransition::Opened { failures: 1 }
        );
        assert_eq!(breaker.admit(12), CircuitAdmission::Denied { retry_at: 13 });
        assert_eq!(breaker.admit(13), CircuitAdmission::Probe);
    }

    #[test]
    fn event_buffer_is_bounded_and_structured() {
        let mut buffer = EventBuffer::new(1).unwrap();
        buffer.push(TreeId(9), 7, ResilienceEventKind::FrameStarted);
        buffer.push(
            TreeId(9),
            7,
            ResilienceEventKind::FrameIsolated { retry_at: 8 },
        );
        assert_eq!(buffer.events().len(), 1);
        assert!(buffer.overflowed());
        assert_eq!(buffer.events()[0].tree, TreeId(9));
        assert_eq!(buffer.events()[0].sequence, 0);
        assert_eq!(
            EventBuffer::new(super::MAX_EVENT_CAPACITY + 1),
            Err(EventBufferError::CapacityLimit {
                requested: super::MAX_EVENT_CAPACITY + 1,
                maximum: super::MAX_EVENT_CAPACITY,
            })
        );
    }

    #[test]
    fn tree_failure_isolated_and_successful_probe_releases_safe_mode() {
        let mut first = tree(1);
        let mut second = tree(2);
        let first_permit = match first.begin_frame(0).unwrap() {
            FrameAdmission::Run(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        first.charge_work(10).unwrap();
        let report = first
            .finish_frame(
                first_permit,
                FrameResult::Failed(FailureReason::Budget(BudgetExhaustion::WorkUnits {
                    limit: 10,
                    attempted: 11,
                    consumed: 10,
                })),
            )
            .unwrap();
        assert!(!report.safe_mode.is_active());

        let second_permit = match second.begin_frame(0).unwrap() {
            FrameAdmission::Run(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        second
            .finish_frame(second_permit, FrameResult::Succeeded)
            .unwrap();
        assert_eq!(
            second.breaker().state(),
            CircuitState::Closed {
                consecutive_failures: 0
            }
        );

        let first_permit = match first.begin_frame(1).unwrap() {
            FrameAdmission::Run(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        first
            .finish_frame(
                first_permit,
                FrameResult::Failed(FailureReason::Invariant { code: 77 }),
            )
            .unwrap();
        assert!(first.safe_mode().is_active());

        assert_eq!(
            first.begin_frame(2).unwrap(),
            FrameAdmission::Isolated { retry_at: 3 }
        );
        assert_eq!(first.safe_mode().entered_at(), Some(1));
        let probe = match first.begin_frame(3).unwrap() {
            FrameAdmission::Probe(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        let report = first.finish_frame(probe, FrameResult::Succeeded).unwrap();
        assert!(report.probe);
        assert!(!first.safe_mode().is_active());
        assert!(first
            .events()
            .events()
            .iter()
            .any(|event| event.kind == ResilienceEventKind::SafeModeExited));
        assert!(second
            .events()
            .events()
            .iter()
            .all(|event| event.tree == TreeId(2)));
    }

    #[test]
    fn permit_and_frame_lifecycle_are_checked() {
        let mut controller = tree(3);
        assert_eq!(
            controller.charge_work(1),
            Err(BudgetChargeError::NoActiveFrame)
        );
        let permit = match controller.begin_frame(8).unwrap() {
            FrameAdmission::Run(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        assert_eq!(
            controller.begin_frame(9),
            Err(super::ResilienceError::FrameAlreadyActive)
        );
        let wrong = super::FramePermit {
            frame: 9,
            probe: false,
        };
        assert_eq!(
            controller.finish_frame(wrong, FrameResult::Succeeded),
            Err(super::ResilienceError::PermitMismatch)
        );
        controller
            .finish_frame(permit, FrameResult::Succeeded)
            .unwrap();
        assert_eq!(
            controller.begin_frame(8),
            Err(super::ResilienceError::FrameNotMonotonic {
                previous: 8,
                requested: 8,
            })
        );
        assert_eq!(
            controller.finish_frame(permit, FrameResult::Succeeded),
            Err(super::ResilienceError::NoActiveFrame)
        );
    }

    #[test]
    fn budget_failure_must_be_committed_as_failure() {
        let mut controller = tree(4);
        let permit = match controller.begin_frame(0).unwrap() {
            FrameAdmission::Run(permit) => permit,
            other => panic!("unexpected admission: {other:?}"),
        };
        let reason = BudgetExhaustion::WorkUnits {
            limit: 10,
            attempted: 11,
            consumed: 0,
        };
        assert_eq!(
            controller.charge_work(11),
            Err(BudgetChargeError::Exhausted(reason))
        );
        assert_eq!(
            controller.finish_frame(permit, FrameResult::Succeeded),
            Err(super::ResilienceError::BudgetFailureUnresolved { reason })
        );
        assert!(controller.has_active_frame());
        let report = controller
            .finish_frame(permit, FrameResult::Failed(FailureReason::Budget(reason)))
            .unwrap();
        assert_eq!(
            report.result,
            FrameResult::Failed(FailureReason::Budget(reason))
        );
        assert!(!controller.has_active_frame());
    }

    #[test]
    fn safe_mode_remembers_last_good_frame() {
        let mut safe = SafeMode::new();
        safe.mark_good(12);
        assert_eq!(safe.last_good_frame(), Some(12));
        assert!(!safe.is_active());
        assert!(safe.enter(13, FailureReason::Host { code: 5 }));
        assert!(safe.is_active());
        assert_eq!(safe.entered_at(), Some(13));
        assert_eq!(safe.reason(), Some(FailureReason::Host { code: 5 }));
        assert!(safe.exit());
        assert_eq!(safe.last_good_frame(), Some(12));
    }
}
