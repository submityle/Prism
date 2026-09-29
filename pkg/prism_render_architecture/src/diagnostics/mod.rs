//! Feature health, breadcrumbs, and performance counters.
//!
//! A shipping AAA renderer runs dozens of optional subsystems (virtual
//! shadows, temporal upscaling, hair, cloth, ray scenes, ...) that each spin
//! up, warm caches, occasionally degrade, and sometimes have to be tripped
//! offline when they misbehave. The renderer needs a small, deterministic,
//! `CPU`-verifiable contract to answer three operational questions every
//! frame:
//!
//! 1. *Is a subsystem healthy, and did its lifecycle follow the rules?* — the
//!    validated health state machine in [`health`] tracks each feature through
//!    `Unavailable -> Initializing -> Warming -> Active`, the
//!    `Active <-> Degraded` oscillation, the "any state can be tripped to
//!    `Quarantined`" fault path, and the controlled `Quarantined ->
//!    Initializing` recovery. Illegal transitions are rejected rather than
//!    silently applied.
//! 2. *What happened recently, in order?* — the fixed-capacity breadcrumb ring
//!    in [`breadcrumb`] records frame-stamped diagnostic events, overwriting
//!    the oldest entry on overflow while preserving chronological iteration and
//!    "most recent N" queries. It is fully deterministic.
//! 3. *How is the frame pacing trending?* — the sliding-window statistics in
//!    [`counters`] fold the most recent [`FrameCounters`] samples into average,
//!    peak, and `p95` aggregates for `CPU` frame time, `GPU` frame time, and
//!    `GPU` memory residency, using a sort-based `p95` with no external
//!    dependency and `NaN`-safe `f32` handling.
//!
//! The by-name [`registry`] ties the pieces together: features register a
//! [`FeatureStatus`] under a stable name, callers query and transition their
//! health through the same validated rules, and the registry aggregates an
//! overall system health (any `Quarantined` feature degrades the system).
//!
//! All arithmetic here is basic (`+ - * /`, `sort`, integer index math, and
//! `clamp`/`abs`-style branches); there are no transcendental functions, so
//! every result is a deterministic function of the input sequence. The `GPU`
//! side that would feed real timings and memory figures into [`FrameCounters`]
//! is out of scope and pending the `GPU` backend; this layer defines and
//! verifies the contract those figures flow through.

pub mod breadcrumb;
pub mod counters;
pub mod health;
pub mod registry;

pub use breadcrumb::{Breadcrumb, BreadcrumbSeverity, BreadcrumbTrail};
pub use counters::{Aggregate, FrameStats, FrameWindow, MemoryAggregate};
pub use health::{is_valid_transition, HealthMachine, TransitionError};
pub use registry::{FeatureRegistry, RegistryError};

/// Small epsilon for comparing derived `f32` quantities without relying on
/// exact bit equality. Shared with the sibling architecture subsystems.
pub const EPS: f32 = 1e-6;

/// Lifecycle state of an optional rendering feature.
///
/// The permitted transitions between these states are defined by
/// [`is_valid_transition`] and enforced by [`HealthMachine`]; this enum only
/// names the states.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FeatureHealth {
    /// The feature is not present or has never been initialized.
    Unavailable,
    /// The feature is spinning up (allocating, compiling, ...).
    Initializing,
    /// The feature is live but still warming caches / history; results may be
    /// lower quality until it reaches [`FeatureHealth::Active`].
    Warming,
    /// The feature is fully operational.
    Active,
    /// The feature is running but has self-reported reduced capability (for
    /// example a fallback path); it can recover back to
    /// [`FeatureHealth::Active`].
    Degraded,
    /// The feature has been tripped offline after a fault and must go through a
    /// controlled recovery (`Quarantined -> Initializing`) before it can serve
    /// again.
    Quarantined,
}

impl FeatureHealth {
    /// Returns `true` when the feature is fully operational.
    #[must_use]
    pub const fn is_operational(self) -> bool {
        matches!(self, FeatureHealth::Active)
    }

    /// Returns `true` when the feature is live in any capacity (warming,
    /// active, or degraded) as opposed to offline / spinning up.
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(
            self,
            FeatureHealth::Warming | FeatureHealth::Active | FeatureHealth::Degraded
        )
    }

    /// Operational severity, from `0` (best) to `5` (worst).
    ///
    /// This orders states by how alarming they are for system health rather
    /// than by lifecycle position, so it does not follow the enum's declaration
    /// order: `Active` is best, a tripped `Quarantined` feature is worst. It
    /// backs the aggregation in [`FeatureRegistry::overall_health`].
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            FeatureHealth::Active => 0,
            FeatureHealth::Warming => 1,
            FeatureHealth::Initializing => 2,
            FeatureHealth::Unavailable => 3,
            FeatureHealth::Degraded => 4,
            FeatureHealth::Quarantined => 5,
        }
    }
}

/// The named health record of a single rendering feature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureStatus {
    /// Stable identifier for the feature (for example `"virtual_shadow"`).
    pub name: &'static str,
    /// Current lifecycle health.
    pub health: FeatureHealth,
    /// Optional human-readable reason for the current state (for example why a
    /// feature degraded or was quarantined).
    pub reason: Option<String>,
}

impl FeatureStatus {
    /// Creates a status for `name` in the initial [`FeatureHealth::Unavailable`]
    /// state with no reason.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            health: FeatureHealth::Unavailable,
            reason: None,
        }
    }

    /// Returns a copy of this status with `health` set, dropping any reason.
    #[must_use]
    pub fn with_health(mut self, health: FeatureHealth) -> Self {
        self.health = health;
        self.reason = None;
        self
    }

    /// Returns a copy of this status carrying `reason`.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Per-frame `CPU`/`GPU` timing and memory sample.
///
/// A stream of these is fed into [`FrameWindow`] to derive rolling frame-pacing
/// statistics. The timing fields are milliseconds; `gpu_memory_bytes` is the
/// resident allocation at sample time. The `GPU`-sourced fields are placeholder
/// zero until the `GPU` backend populates them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameCounters {
    /// Monotonic frame index this sample was taken on.
    pub frame_index: u64,
    /// `CPU`-side frame time in milliseconds.
    pub cpu_frame_ms: f32,
    /// `GPU`-side frame time in milliseconds.
    pub gpu_frame_ms: f32,
    /// Resident `GPU` memory in bytes at sample time.
    pub gpu_memory_bytes: u64,
}

impl FrameCounters {
    /// Creates a sample for `frame_index` with the given timings and memory.
    #[must_use]
    pub const fn new(
        frame_index: u64,
        cpu_frame_ms: f32,
        gpu_frame_ms: f32,
        gpu_memory_bytes: u64,
    ) -> Self {
        Self {
            frame_index,
            cpu_frame_ms,
            gpu_frame_ms,
            gpu_memory_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_orders_worst_last() {
        assert!(FeatureHealth::Active.severity() < FeatureHealth::Warming.severity());
        assert!(FeatureHealth::Degraded.severity() < FeatureHealth::Quarantined.severity());
        assert!(FeatureHealth::Warming.severity() < FeatureHealth::Degraded.severity());
        assert_eq!(FeatureHealth::Quarantined.severity(), 5);
    }

    #[test]
    fn operational_and_live_flags() {
        assert!(FeatureHealth::Active.is_operational());
        assert!(!FeatureHealth::Warming.is_operational());
        assert!(FeatureHealth::Warming.is_live());
        assert!(FeatureHealth::Degraded.is_live());
        assert!(!FeatureHealth::Unavailable.is_live());
        assert!(!FeatureHealth::Quarantined.is_live());
    }

    #[test]
    fn feature_status_builders() {
        let s = FeatureStatus::new("hair");
        assert_eq!(s.name, "hair");
        assert_eq!(s.health, FeatureHealth::Unavailable);
        assert!(s.reason.is_none());

        let s = s.with_health(FeatureHealth::Active).with_reason("ready");
        assert_eq!(s.health, FeatureHealth::Active);
        assert_eq!(s.reason.as_deref(), Some("ready"));

        // with_health clears any previous reason.
        let s = s.with_health(FeatureHealth::Warming);
        assert!(s.reason.is_none());
    }

    #[test]
    fn frame_counters_new_matches_default_fields() {
        let c = FrameCounters::new(7, 5.0, 6.0, 1024);
        assert_eq!(c.frame_index, 7);
        assert_eq!(c.cpu_frame_ms, 5.0);
        assert_eq!(c.gpu_frame_ms, 6.0);
        assert_eq!(c.gpu_memory_bytes, 1024);
        assert_eq!(FrameCounters::default(), FrameCounters::new(0, 0.0, 0.0, 0));
    }
}
