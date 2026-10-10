//! Runtime capability probing and quality tiering (design §3, §17, §24.4).
//!
//! A shipping engine must scale the *same* assembly from a headless CI runner
//! or a dedicated server, through a battery-constrained mobile device, up to a
//! high-end desktop. The design calls this **档位化**: a three-gate model where
//! the running engine is shaped by
//!
//! 1. a runtime-probed **[`Capabilities`]** set — the hardware / environment
//!    facts we can actually observe (logical core count, whether a display is
//!    reachable, timer granularity, whether we are on a mobile OS);
//! 2. a derived **[`QualityTier`]** — the coarse running form
//!    (`Server` / `Mobile` / `Desktop`) a build selects based on those facts;
//!    and
//! 3. compile-time **feature flags** (`std` / `multi_thread` / `pipelined` /
//!    `headless` / …), which this module does not own.
//!
//! This module implements the first two gates as a real, self-contained probe
//! plus a pure derivation. It performs **no faking**: every field is either a
//! genuine measurement (core count, timer resolution) or an explicitly
//! documented, overridable heuristic (display reachability), and the no-std
//! fallback is a clearly conservative constant rather than a pretend probe.
//!
//! The probe feeds two consumers in this crate:
//!
//! * [`App`](crate::App) installs a [`Capabilities`] and a [`QualityTier`]
//!   resource at construction, so capability-driven assembly (design §17
//!   "特性探测装配：按 capability 在装配期选择 system 集/子应用拓扑") and the
//!   [`RunMode`](crate::run_mode::RunMode) default selection can read them.
//! * [`QualityTier`] is the natural source of the
//!   [`PlatformTier`](crate::settings::SettingsLayer::PlatformTier) settings
//!   layer (design §14 "平台档位（quality tier）").
//!
//! # Honesty of the display probe
//!
//! Reliably detecting an attached display without a window backend is a
//! platform-specific operation that properly belongs to `prism_window` (absent
#![cfg_attr(
    feature = "std",
    doc = "today). Rather than pretend otherwise, [`Capabilities::detect`] uses a"
)]
#![cfg_attr(
    not(feature = "std"),
    doc = "today). Rather than pretend otherwise, `Capabilities::detect` uses a"
)]
//! conservative, documented heuristic (environment-variable probe on X11 /
//! Wayland, an assume-present default on Windows / macOS, always present on
//! mobile) and honors an explicit `PRISM_HEADLESS` override. Callers that know
//! better — most importantly a future window plugin — can override the field
//! with [`Capabilities::with_display`] before the tier is derived.
//!
//! [`App`]: crate::app::App

/// Observable hardware / environment facts the engine scales against
/// (design §3 `capability`).
///
#[cfg_attr(
    feature = "std",
    doc = "Produced by [`detect`](Capabilities::detect) at runtime (std) or by the"
)]
#[cfg_attr(
    not(feature = "std"),
    doc = "Produced by `detect` at runtime (std) or by the"
)]
/// conservative [`headless`](Capabilities::headless) constant (no-std / tests).
/// Stored as a main-world resource by [`App::new`](crate::App::new); read it to
/// gate assembly on real hardware facts rather than compile-time guesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// The number of logical CPU cores available to this process.
    ///
    /// On std this is [`std::thread::available_parallelism`] (which already
    /// accounts for cgroup / affinity limits), clamped to at least `1`. The
    /// conservative fallback is `1`.
    pub logical_cores: usize,
    /// Whether a display / windowing surface is reachable (design §3
    /// "是否有窗口/显示器").
    ///
    /// This is a heuristic, not a guarantee — see the module docs. `false`
    /// steers tiering toward a headless [`Server`](QualityTier::Server) form.
    pub has_display: bool,
    /// Whether the process is running on a mobile OS (Android / iOS).
    ///
    /// A compile-time fact (`cfg!(target_os = …)`), surfaced here so tiering
    /// and lifecycle code read one uniform source.
    pub is_mobile: bool,
    /// Whether a high-resolution monotonic timer is available (design §3
    /// "是否支持高精度计时器").
    ///
    #[cfg_attr(
        feature = "std",
        doc = "Measured by [`probe_timer_resolution`](Capabilities::probe_timer_resolution) on std; a frame pacer (§13) needs this to be `true` to pace reliably."
    )]
    #[cfg_attr(
        not(feature = "std"),
        doc = "Measured by `probe_timer_resolution` on std; a frame pacer (§13) needs this to be `true` to pace reliably."
    )]
    pub high_resolution_timer: bool,
}

impl Capabilities {
    /// The conservative fallback: a single-threaded, headless, non-mobile
    /// profile with no high-resolution timer assumed.
    ///
    /// Used as the no-std default and as a deterministic baseline in tests. It
    /// deliberately under-claims every capability so that code gating on it
    /// degrades to the safest path rather than assuming hardware it cannot see.
    pub const fn headless() -> Self {
        Self {
            logical_cores: 1,
            has_display: false,
            is_mobile: false,
            high_resolution_timer: false,
        }
    }

    /// Probe the real environment for the current process's capabilities.
    ///
    /// Combines [`std::thread::available_parallelism`] (core count), a
    /// documented display heuristic (see [`probe_display`](Capabilities::probe_display)),
    /// the `target_os` mobile fact, and a live timer-granularity measurement
    /// ([`probe_timer_resolution`](Capabilities::probe_timer_resolution)).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        Self {
            logical_cores,
            has_display: Self::probe_display(),
            is_mobile: cfg!(any(target_os = "android", target_os = "ios")),
            high_resolution_timer: Self::probe_timer_resolution()
                <= core::time::Duration::from_micros(1),
        }
    }

    /// The documented display-reachability heuristic (see the module docs).
    ///
    /// Precedence: an explicit `PRISM_HEADLESS` truthy value forces `false`;
    /// otherwise mobile is always `true`; on X11 / Wayland unix a non-empty
    /// `DISPLAY` or `WAYLAND_DISPLAY` means `true`; on Windows / macOS the
    /// default is `true` (assume-present, overridable). This never pretends to
    /// a platform query it cannot make.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn probe_display() -> bool {
        if let Ok(v) = std::env::var("PRISM_HEADLESS")
            && is_truthy(&v)
        {
            return false;
        }
        if cfg!(any(target_os = "android", target_os = "ios")) {
            return true;
        }
        if cfg!(any(
            target_os = "linux",
            target_os = "freebsd",
            target_os = "openbsd",
            target_os = "netbsd",
            target_os = "dragonfly"
        )) {
            let present = |k: &str| std::env::var(k).map(|v| !v.is_empty()).unwrap_or(false);
            return present("DISPLAY") || present("WAYLAND_DISPLAY");
        }
        // Windows / macOS: no cheap backend-free query; assume present and let a
        // window plugin or PRISM_HEADLESS correct it.
        true
    }

    /// Measure the smallest observable positive step of [`std::time::Instant`].
    ///
    /// Samples the clock in a tight loop until the reading advances, repeating
    /// a few times and keeping the smallest nonzero delta. This is a genuine
    /// measurement of the platform clock's effective granularity, used to set
    /// [`high_resolution_timer`](Capabilities::high_resolution_timer).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn probe_timer_resolution() -> core::time::Duration {
        use std::time::Instant;
        let mut best = core::time::Duration::from_secs(1);
        for _ in 0..5 {
            let start = Instant::now();
            // Spin until the clock visibly advances.
            let delta = loop {
                let d = start.elapsed();
                if !d.is_zero() {
                    break d;
                }
            };
            if delta < best {
                best = delta;
            }
        }
        best
    }

    /// Return a copy with [`has_display`](Capabilities::has_display) overridden.
    ///
    /// The intended caller is a window plugin that has authoritative knowledge
    /// the heuristic cannot have; it corrects the field before the tier is
    /// derived.
    #[must_use]
    pub const fn with_display(mut self, has_display: bool) -> Self {
        self.has_display = has_display;
        self
    }

    /// Whether more than one logical core is available, i.e. whether
    /// multi-threaded execution can actually overlap work.
    #[must_use]
    pub const fn is_multicore(&self) -> bool {
        self.logical_cores > 1
    }
}

impl Default for Capabilities {
    #[cfg_attr(
        feature = "std",
        doc = "The real [`detect`](Capabilities::detect) probe on std, the conservative"
    )]
    #[cfg_attr(
        not(feature = "std"),
        doc = "The real `detect` probe on std, the conservative"
    )]
    /// [`headless`](Capabilities::headless) constant otherwise.
    fn default() -> Self {
        #[cfg(feature = "std")]
        {
            Self::detect()
        }
        #[cfg(not(feature = "std"))]
        {
            Self::headless()
        }
    }
}

impl prism_ecs::resource::Resource for Capabilities {}

/// The coarse running form a build selects from its [`Capabilities`]
/// (design §3 `quality tier`).
///
/// The tier is the single knob that decides how much the shell pays for: a
/// [`Server`](QualityTier::Server) build pays only the fixed-step heartbeat; a
/// [`Mobile`](QualityTier::Mobile) build adds a power-aware variable step and
/// lifecycle handling; a [`Desktop`](QualityTier::Desktop) build adds the
/// sub-app pipeline and frame pacing (design §3 target paragraph).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QualityTier {
    /// Headless fixed-step server: no display, minimal cost (design §3
    /// "server（无头定帧）").
    Server,
    /// Mobile: power-aware variable step plus suspend / resume lifecycle
    /// (design §3 "mobile（省电可变步 + 生命周期）").
    Mobile,
    /// Desktop: sub-app pipelining plus frame pacing (design §3
    /// "desktop（流水线 + frame pacing）").
    Desktop,
}

impl QualityTier {
    /// Derive the tier from probed [`Capabilities`].
    ///
    /// Mobile OS wins first (a phone with a screen is still the mobile tier);
    /// otherwise a reachable display selects [`Desktop`](QualityTier::Desktop)
    /// and its absence selects the headless [`Server`](QualityTier::Server)
    /// tier. This mirrors the design's server / mobile / desktop split (§3).
    #[must_use]
    pub const fn from_capabilities(caps: &Capabilities) -> Self {
        if caps.is_mobile {
            QualityTier::Mobile
        } else if caps.has_display {
            QualityTier::Desktop
        } else {
            QualityTier::Server
        }
    }

    /// Whether this tier is expected to drive rendering / a display.
    ///
    /// Only [`Desktop`](QualityTier::Desktop) and [`Mobile`](QualityTier::Mobile)
    /// present; [`Server`](QualityTier::Server) is headless.
    #[must_use]
    pub const fn presents(self) -> bool {
        matches!(self, QualityTier::Desktop | QualityTier::Mobile)
    }
}

impl prism_ecs::resource::Resource for QualityTier {}

/// Parse the common truthy spellings of an environment flag.
///
/// Accepts `1` / `true` / `yes` / `on` (case-insensitive); everything else,
/// including the empty string, is false. Shared by the `PRISM_HEADLESS` probe.
#[cfg(feature = "std")]
fn is_truthy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}
