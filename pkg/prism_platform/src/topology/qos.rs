//! Engine lane -> OS quality-of-service class mapping (design §24.3).
//!
//! The engine runs work in a small set of priority *lanes* ([`EngineQos`],
//! mirroring the `prism_tasks` `QoS` lanes). Each OS exposes its own
//! quality-of-service vocabulary — Apple's `QOS_CLASS_*`, Windows' process /
//! thread "Quality of Service" power-throttling tiers, Linux's `nice` / cgroup
//! weighting — and the kernel uses it to decide which physical cores a thread
//! lands on and how aggressively it is clocked.
//!
//! This module is **pure data**: it maps an [`EngineQos`] lane plus an [`Os`]
//! to an [`OsQosHint`] describing the native class name and a normalized
//! priority. It issues no syscalls; the actual `pthread_set_qos_class_self_np`
//! / `SetThreadInformation` / `setpriority` call belongs in the thread backend,
//! which can consume these hints verbatim.

use crate::platform::Os;

/// An engine scheduling lane. Ordered from most to least latency-sensitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EngineQos {
    /// Hard real-time critical path (audio mixing, main-thread present).
    Critical,
    /// User-interactive frame work that must land this frame.
    Interactive,
    /// Default simulation / gameplay work.
    Default,
    /// Latency-tolerant utility work (streaming decode, prefetch).
    Utility,
    /// Best-effort background work (telemetry flush, bake, GC).
    Background,
}

impl EngineQos {
    /// All lanes, most to least important.
    pub const ALL: [EngineQos; 5] = [
        EngineQos::Critical,
        EngineQos::Interactive,
        EngineQos::Default,
        EngineQos::Utility,
        EngineQos::Background,
    ];

    /// A normalized importance in `0..=100`, where 100 is the most important.
    /// Useful for platforms with no named classes (map onto `nice` or weights).
    #[must_use]
    pub const fn importance(self) -> u8 {
        match self {
            EngineQos::Critical => 100,
            EngineQos::Interactive => 80,
            EngineQos::Default => 60,
            EngineQos::Utility => 30,
            EngineQos::Background => 10,
        }
    }

    /// Whether work in this lane should prefer efficiency cores on a hybrid
    /// machine when the scheduler is free to choose.
    #[must_use]
    pub const fn prefers_efficiency_core(self) -> bool {
        matches!(self, EngineQos::Utility | EngineQos::Background)
    }
}

/// A platform-neutral description of the OS `QoS` class an [`EngineQos`] maps to.
///
/// `class` is a stable, byte-for-byte identifier naming the native class (for
/// logging and for the thread backend to switch on); `importance` is the
/// normalized `0..=100` priority; `throttlable` records whether the OS may
/// power-throttle (down-clock / park) threads in this class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsQosHint {
    /// Stable native-class identifier, e.g. `"QOS_CLASS_USER_INTERACTIVE"` or
    /// `"WIN_QOS_HIGH"`.
    pub class: &'static str,
    /// Normalized importance in `0..=100`.
    pub importance: u8,
    /// Whether the OS may power-throttle threads in this class.
    pub throttlable: bool,
}

/// Map an engine lane plus an OS onto the native `QoS` class to request.
///
/// The mapping is deliberately conservative and documented per platform; it is
/// the single place policy lives so the thread backend stays a thin syscall.
#[must_use]
pub fn qos_hint(os: Os, lane: EngineQos) -> OsQosHint {
    let importance = lane.importance();
    let throttlable = matches!(lane, EngineQos::Utility | EngineQos::Background);
    let class = match os {
        Os::Apple => match lane {
            EngineQos::Critical | EngineQos::Interactive => "QOS_CLASS_USER_INTERACTIVE",
            EngineQos::Default => "QOS_CLASS_DEFAULT",
            EngineQos::Utility => "QOS_CLASS_UTILITY",
            EngineQos::Background => "QOS_CLASS_BACKGROUND",
        },
        Os::Windows => match lane {
            EngineQos::Critical | EngineQos::Interactive => "WIN_QOS_HIGH",
            EngineQos::Default => "WIN_QOS_MEDIUM",
            EngineQos::Utility => "WIN_QOS_LOW",
            EngineQos::Background => "WIN_QOS_ECO",
        },
        Os::Linux | Os::Android => match lane {
            EngineQos::Critical => "NICE_-15",
            EngineQos::Interactive => "NICE_-10",
            EngineQos::Default => "NICE_0",
            EngineQos::Utility => "NICE_10",
            EngineQos::Background => "NICE_19",
        },
        Os::Web | Os::Unknown => "QOS_UNSUPPORTED",
    };
    OsQosHint {
        class,
        importance,
        throttlable,
    }
}
