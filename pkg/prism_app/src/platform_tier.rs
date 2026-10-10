//! Per-[`QualityTier`] platform settings profile (design §3, §14).
//!
//! Design §14 defines the configuration cascade
//! *engine default → **platform tier** → user → command line → runtime*, and
//! design §3 says each quality tier pays for a different running form: a
//! [`Server`](QualityTier::Server) build pays only the fixed-step heartbeat, a
//! [`Mobile`](QualityTier::Mobile) build adds a power-aware variable step and
//! suspend/resume lifecycle, and a [`Desktop`](QualityTier::Desktop) build adds
//! the sub-app pipeline and frame pacing.
//!
//! This module is the concrete bridge between those two chapters: it turns the
//! [`QualityTier`] the app derived from its probed [`Capabilities`] into the set
//! of settings the [`PlatformTier`](SettingsLayer::PlatformTier) layer
//! contributes to the cascade. It is intentionally the *only* place that knows
//! "what a tier means" as settings, so the mapping is reviewable in one spot and
//! the rest of the shell reads resolved values through [`Settings`] without
//! branching on the tier.
//!
//! # The profile
//!
//! [`PlatformTierProfile`] is a small, fully-typed struct — one field per knob —
//! so the tier→settings decision is made in real Rust, not in loose string keys.
//! [`entries`](PlatformTierProfile::entries) then serialises it into the
//! dynamically-typed [`SettingValue`] pairs the layered [`Settings`] store
//! holds, each under a stable, documented key constant from this module. Writing
//! the profile into the [`PlatformTier`](SettingsLayer::PlatformTier) layer
//! leaves the lower [`EngineDefault`](SettingsLayer::EngineDefault) layer intact
//! and lets any higher layer (user / command line / runtime) override a single
//! knob without disturbing the rest (design §14 "后者覆盖前者").
//!
//! # Honest scope
//!
//! These keys are the *engine shell's* own tier knobs — the ones `prism_app`
//! itself can act on (presentation on/off, the frame-limit cap, whether to opt
//! into pipelined rendering, the mobile power-aware step and lifecycle). They are
//! real resolved settings in the cascade that subsystems read; this module does
//! **not** invent renderer/audio/network keys owned by crates that do not exist
//! yet. Downstream crates layer their own tier keys on top of the same cascade
//! as they land, without changing this contract.
//!
//! [`Capabilities`]: crate::capability::Capabilities
//! [`QualityTier`]: crate::capability::QualityTier

use crate::app::App;
use crate::capability::QualityTier;
use crate::settings::{SettingValue, Settings, SettingsLayer};

/// Settings key for whether this tier drives presentation / rendering (`bool`).
///
/// Mirrors [`QualityTier::presents`](crate::capability::QualityTier::presents):
/// a headless [`Server`](QualityTier::Server) tier resolves this to `false`, and
/// a presenting [`Mobile`](QualityTier::Mobile) / [`Desktop`](QualityTier::Desktop)
/// tier to `true`.
pub const KEY_RENDER_PRESENT: &str = "render.present";

/// Settings key for the tier's default frame-rate cap in FPS (`i64`).
///
/// `0` means *unlimited* — the loop presents as fast as it runs (relying on
/// vsync or deliberate uncapping). A positive value is a software frame limiter
/// in frames-per-second (design §13).
#[cfg_attr(
    feature = "std",
    doc = "[`PlatformTierProfile::frame_limit`] turns this cap into a \
           [`FrameLimit`](crate::pacing::FrameLimit) for the pacing layer."
)]
pub const KEY_FRAME_LIMIT_FPS: &str = "pacing.frame_limit_fps";

/// Settings key for whether the tier opts into pipelined rendering (`bool`).
///
/// Only the [`Desktop`](QualityTier::Desktop) tier resolves this to `true`
/// (design §3 "desktop（流水线 + frame pacing）"). It is advisory: the actual
/// cross-thread overlap is still gated behind the `pipelined` cargo feature and
/// an explicit [`App::enable_pipelined_rendering`](crate::app::App) opt-in.
pub const KEY_PIPELINED_RENDERING: &str = "pipeline.pipelined_rendering";

/// Settings key for the mobile power-aware variable timestep (`bool`).
///
/// Only the [`Mobile`](QualityTier::Mobile) tier resolves this to `true`
/// (design §3 "mobile（省电可变步 + 生命周期）").
pub const KEY_POWER_AWARE_VARIABLE_STEP: &str = "loop.power_aware_variable_step";

/// Settings key for whether the tier expects suspend/resume lifecycle (`bool`).
///
/// Only the [`Mobile`](QualityTier::Mobile) tier resolves this to `true`: a
/// phone backgrounds and foregrounds and must pause simulation and release
/// transient GPU resources (design §3 / §12).
pub const KEY_SUSPEND_RESUME_LIFECYCLE: &str = "lifecycle.suspend_resume";

/// Every platform-tier settings key, in a stable order.
///
/// Useful for clearing a previously applied profile from the
/// [`PlatformTier`](SettingsLayer::PlatformTier) layer, or for enumerating the
/// knobs the tier contributes. The order matches the field order of
/// [`PlatformTierProfile`].
pub const ALL_KEYS: [&str; 5] = [
    KEY_RENDER_PRESENT,
    KEY_FRAME_LIMIT_FPS,
    KEY_PIPELINED_RENDERING,
    KEY_POWER_AWARE_VARIABLE_STEP,
    KEY_SUSPEND_RESUME_LIFECYCLE,
];

/// The settings a [`QualityTier`] contributes to the
/// [`PlatformTier`](SettingsLayer::PlatformTier) layer (design §3, §14).
///
/// One field per shell knob, all resolved up-front from the tier so the mapping
/// is a single, reviewable decision. Build it with
/// [`for_tier`](PlatformTierProfile::for_tier) (or the
/// [`QualityTier::platform_profile`] shortcut), inspect the typed fields, and
/// serialise it into the layered store with [`entries`](PlatformTierProfile::entries).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformTierProfile {
    /// Whether the tier drives presentation / rendering
    /// ([`KEY_RENDER_PRESENT`]).
    pub presents_display: bool,
    /// The default frame-rate cap in FPS, `0` for unlimited
    /// ([`KEY_FRAME_LIMIT_FPS`]).
    pub frame_limit_fps: u32,
    /// Whether the tier opts into pipelined rendering
    /// ([`KEY_PIPELINED_RENDERING`]).
    pub pipelined_rendering: bool,
    /// Whether the tier uses a power-aware variable timestep
    /// ([`KEY_POWER_AWARE_VARIABLE_STEP`]).
    pub power_aware_variable_step: bool,
    /// Whether the tier expects suspend/resume lifecycle
    /// ([`KEY_SUSPEND_RESUME_LIFECYCLE`]).
    pub suspend_resume_lifecycle: bool,
}

impl PlatformTierProfile {
    /// The recommended profile for `tier` (design §3 server/mobile/desktop split).
    ///
    /// * [`Server`](QualityTier::Server): headless — no presentation, no frame
    ///   cap (the fixed-step tickrate governs instead), no pipeline, no mobile
    ///   lifecycle.
    /// * [`Mobile`](QualityTier::Mobile): presents, caps to 60 FPS to save power,
    ///   runs the power-aware variable step and the suspend/resume lifecycle, no
    ///   pipeline.
    /// * [`Desktop`](QualityTier::Desktop): presents, uncapped (vsync / frame
    ///   pacing governs), opts into the render pipeline, no mobile lifecycle.
    #[must_use]
    pub const fn for_tier(tier: QualityTier) -> Self {
        match tier {
            QualityTier::Server => Self {
                presents_display: false,
                frame_limit_fps: 0,
                pipelined_rendering: false,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            },
            QualityTier::Mobile => Self {
                presents_display: true,
                frame_limit_fps: 60,
                pipelined_rendering: false,
                power_aware_variable_step: true,
                suspend_resume_lifecycle: true,
            },
            QualityTier::Desktop => Self {
                presents_display: true,
                frame_limit_fps: 0,
                pipelined_rendering: true,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            },
        }
    }

    /// The `(key, value)` pairs this profile contributes, in [`ALL_KEYS`] order.
    ///
    /// This is the serialisation from the typed fields into the dynamically-typed
    /// [`SettingValue`]s the layered [`Settings`] store holds. The FPS cap is
    /// stored as an [`i64`](SettingValue::Int) (its natural config form); the
    /// rest are [`bool`](SettingValue::Bool)s.
    #[must_use]
    pub fn entries(&self) -> [(&'static str, SettingValue); 5] {
        [
            (
                KEY_RENDER_PRESENT,
                SettingValue::Bool(self.presents_display),
            ),
            (
                KEY_FRAME_LIMIT_FPS,
                SettingValue::Int(i64::from(self.frame_limit_fps)),
            ),
            (
                KEY_PIPELINED_RENDERING,
                SettingValue::Bool(self.pipelined_rendering),
            ),
            (
                KEY_POWER_AWARE_VARIABLE_STEP,
                SettingValue::Bool(self.power_aware_variable_step),
            ),
            (
                KEY_SUSPEND_RESUME_LIFECYCLE,
                SettingValue::Bool(self.suspend_resume_lifecycle),
            ),
        ]
    }

    /// Write this profile into the [`PlatformTier`](SettingsLayer::PlatformTier)
    /// layer of `settings`, returning the number of keys whose *resolved* value
    /// changed.
    ///
    /// Lower-level counterpart of [`App::apply_platform_tier`]: it mutates a
    /// [`Settings`] store directly and does not broadcast events, so it is usable
    /// outside an [`App`] (e.g. in tests or tooling). A key only counts as
    /// changed when the newly resolved value differs — a higher-precedence layer
    /// already overriding the key keeps its resolved value, so that key does not
    /// count.
    pub fn write_into(&self, settings: &mut Settings) -> usize {
        let mut changed = 0;
        for (key, value) in self.entries() {
            if settings
                .set(SettingsLayer::PlatformTier, key, value)
                .is_some()
            {
                changed += 1;
            }
        }
        changed
    }
}

#[cfg(feature = "std")]
impl PlatformTierProfile {
    /// The [`FrameLimit`](crate::pacing::FrameLimit) implied by
    /// [`frame_limit_fps`](PlatformTierProfile::frame_limit_fps) (design §13).
    ///
    /// `0` FPS maps to [`FrameLimit::Off`](crate::pacing::FrameLimit::Off)
    /// (unlimited); a positive cap maps to
    /// [`FrameLimit::Fps`](crate::pacing::FrameLimit::Fps). This is the concrete
    /// hook a [`FramePacer`](crate::pacing::FramePacer) consumes, so the mobile
    /// 60 FPS power cap actually paces the loop rather than being advisory.
    #[must_use]
    pub fn frame_limit(&self) -> crate::pacing::FrameLimit {
        crate::pacing::FrameLimit::from_fps(self.frame_limit_fps)
    }
}

impl QualityTier {
    /// The [`PlatformTierProfile`] recommended for this tier.
    ///
    /// Shortcut for [`PlatformTierProfile::for_tier`] (design §3, §14).
    #[must_use]
    pub const fn platform_profile(self) -> PlatformTierProfile {
        PlatformTierProfile::for_tier(self)
    }
}

impl App {
    /// Populate the [`PlatformTier`](SettingsLayer::PlatformTier) settings layer
    /// from `tier`'s [`PlatformTierProfile`], broadcasting a
    /// [`SettingChanged`](crate::settings::SettingChanged) event for every key
    /// whose resolved value changed (design §3, §14).
    ///
    /// Auto-initialises the [`Settings`] store and the change event (like the
    /// other settings helpers). Because the values land in the
    /// [`PlatformTier`](SettingsLayer::PlatformTier) layer — second-lowest in the
    /// cascade — any [`User`](SettingsLayer::User),
    /// [`CommandLine`](SettingsLayer::CommandLine), or
    /// [`Runtime`](SettingsLayer::Runtime) override already present keeps winning,
    /// and that key reports no change. Re-applying the same tier is idempotent.
    pub fn apply_platform_tier(&mut self, tier: QualityTier) -> &mut Self {
        self.init_settings();
        let profile = tier.platform_profile();
        let mut changes = Vec::with_capacity(ALL_KEYS.len());
        {
            let settings = self.world_mut().resource_mut::<Settings>();
            for (key, value) in profile.entries() {
                if let Some(change) = settings.set(SettingsLayer::PlatformTier, key, value) {
                    changes.push(change);
                }
            }
        }
        for change in changes {
            self.send_event(change);
        }
        self
    }

    /// Populate the [`PlatformTier`](SettingsLayer::PlatformTier) layer from the
    /// tier the app derived from its probed [`Capabilities`](crate::capability::Capabilities) at
    /// [`App::new`](crate::app::App::new) (design §3, §14).
    ///
    /// Convenience over [`apply_platform_tier`](App::apply_platform_tier) using
    /// [`App::quality_tier`](crate::app::App::quality_tier).
    pub fn apply_detected_platform_tier(&mut self) -> &mut Self {
        let tier = self.quality_tier();
        self.apply_platform_tier(tier)
    }
}
