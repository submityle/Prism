//! **§24.4 — frame-budget-driven adaptive quality.** A pure-CPU closed-loop
//! controller that turns the frame-timing feedback signal ("how much of the
//! budget did this frame use?") into a graded quality decision (step **down**,
//! step **up**, or **hold**).
//!
//! The controller is the deterministic *control model* the design doc §24.4
//! calls for; it does not touch any renderer. A consumer maps the integer
//! quality level this emits onto concrete settings (dynamic resolution, LOD
//! bias, shadow-cascade count, particle caps, background-lane deferral, ...)
//! — that mapping is the caller's job (see the honest boundary below).
//!
//! ## Control law
//! Each [`observe`](AdaptiveQualityController::observe) compares the frame time
//! against the budget as an integer **utilisation** in parts-per-million (ppm)
//! of the budget, then applies a hysteresis band plus asymmetric patience:
//!
//! - **Hysteresis dead-band.** A frame counts as *over* only above
//!   [`downgrade_ppm`](AdaptiveQualityConfig::downgrade_ppm) and *under* only
//!   below [`upgrade_ppm`](AdaptiveQualityConfig::upgrade_ppm); utilisation
//!   between the two is *neutral* and nudges nothing. Keeping the two
//!   thresholds apart (e.g. `80%` up, `100%` down) stops the controller
//!   oscillating around a single set-point.
//! - **Fast down, slow up.** A downgrade needs only
//!   [`downgrade_patience`](AdaptiveQualityConfig::downgrade_patience)
//!   consecutive over-budget frames (default `2`) so the controller protects
//!   frame-rate quickly; an upgrade needs
//!   [`upgrade_patience`](AdaptiveQualityConfig::upgrade_patience) consecutive
//!   under-budget frames (default `30`), so quality rises only once there is
//!   durable headroom.
//! - **Cooldown.** After any change the controller holds for
//!   [`cooldown_frames`](AdaptiveQualityConfig::cooldown_frames) frames before
//!   another normal adjustment, damping ping-pong.
//! - **Severe-overrun escape hatch.** A frame above
//!   [`severe_ppm`](AdaptiveQualityConfig::severe_ppm) (default `150%`) drops
//!   [`severe_step`](AdaptiveQualityConfig::severe_step) levels at once and
//!   bypasses both patience and cooldown — the fast bail-out a hitch /
//!   death-spiral needs.
//!
//! Every computation is integer / fixed-point (`u128` ppm ratios); no floating
//! point and no wall-clock read enters the decision, so a given sequence of
//! `(frame, budget)` pairs yields a bit-identical level trajectory across runs
//! — safe to drive from a deterministic replay. `no_std + alloc`, no `unsafe`.
//!
//! ## Honest boundary
//! This layer only decides a quality *level*. Measuring per-stage frame costs
//! (`prism_profiler` / §16 observability) and applying the chosen level to real
//! render settings live in the renderer / gameplay layers, not here (design doc
//! §24.9). The controller never allocates and never reads a clock.

use crate::Duration;

/// Parts-per-million denominator; `1_000_000` ppm == `100%` of the budget.
const PPM: u128 = 1_000_000;

/// Utilisation of `frame` against `budget`, in parts-per-million of the budget
/// (`1_000_000` == exactly on budget). Returns `0` when `budget` is zero (no
/// meaningful budget to compare against). Saturates at [`u64::MAX`].
#[inline]
#[must_use]
pub fn utilization_ppm(frame: Duration, budget: Duration) -> u64 {
    let budget_ns = budget.as_nanos();
    if budget_ns == 0 {
        return 0;
    }
    let scaled = frame.as_nanos().saturating_mul(PPM);
    (scaled / budget_ns).min(u64::MAX as u128) as u64
}

/// Tuning for an [`AdaptiveQualityController`]. Build with [`new`](Self::new)
/// (sensible defaults) then adjust via the `const fn` setters; the controller's
/// constructor re-clamps every field so an invalid combination can never put
/// the control law into an inconsistent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveQualityConfig {
    /// Lowest selectable quality level (inclusive).
    pub min_level: u32,
    /// Highest selectable quality level (inclusive). Clamped to be `>= min_level`.
    pub max_level: u32,
    /// Utilisation (ppm) at or above which a frame counts as over budget.
    pub downgrade_ppm: u32,
    /// Utilisation (ppm) at or below which a frame counts as comfortably under
    /// budget. Clamped to be `<= downgrade_ppm` so a hysteresis gap always
    /// exists.
    pub upgrade_ppm: u32,
    /// Utilisation (ppm) at or above which a frame is a severe overrun and
    /// triggers the immediate multi-step bail-out. Clamped to be
    /// `>= downgrade_ppm`.
    pub severe_ppm: u32,
    /// Levels to drop on a severe overrun. Clamped to be `>= 1`.
    pub severe_step: u32,
    /// Consecutive over-budget frames needed to step down. Clamped to `>= 1`.
    pub downgrade_patience: u32,
    /// Consecutive under-budget frames needed to step up. Clamped to `>= 1`.
    pub upgrade_patience: u32,
    /// Frames to hold after any change before the next normal adjustment.
    pub cooldown_frames: u32,
}

impl AdaptiveQualityConfig {
    /// Default over-budget threshold: `100%` of the budget.
    pub const DEFAULT_DOWNGRADE_PPM: u32 = 1_000_000;
    /// Default comfortably-under threshold: `80%` of the budget.
    pub const DEFAULT_UPGRADE_PPM: u32 = 800_000;
    /// Default severe-overrun threshold: `150%` of the budget.
    pub const DEFAULT_SEVERE_PPM: u32 = 1_500_000;

    /// A configuration with AAA-sensible defaults over the level range
    /// `min_level..=max_level`: `100%` down / `80%` up hysteresis, severe at
    /// `150%` dropping `2` levels, patience `2` down / `30` up, `8`-frame
    /// cooldown.
    ///
    /// # Panics
    /// Never panics; `max_level` is clamped to be at least `min_level` by the
    /// controller constructor.
    #[inline]
    #[must_use]
    pub const fn new(min_level: u32, max_level: u32) -> Self {
        Self {
            min_level,
            max_level,
            downgrade_ppm: Self::DEFAULT_DOWNGRADE_PPM,
            upgrade_ppm: Self::DEFAULT_UPGRADE_PPM,
            severe_ppm: Self::DEFAULT_SEVERE_PPM,
            severe_step: 2,
            downgrade_patience: 2,
            upgrade_patience: 30,
            cooldown_frames: 8,
        }
    }

    /// Set the hysteresis band (over / under thresholds in ppm).
    #[inline]
    #[must_use]
    pub const fn with_hysteresis(mut self, upgrade_ppm: u32, downgrade_ppm: u32) -> Self {
        self.upgrade_ppm = upgrade_ppm;
        self.downgrade_ppm = downgrade_ppm;
        self
    }

    /// Set the severe-overrun threshold (ppm) and how many levels it drops.
    #[inline]
    #[must_use]
    pub const fn with_severe(mut self, severe_ppm: u32, severe_step: u32) -> Self {
        self.severe_ppm = severe_ppm;
        self.severe_step = severe_step;
        self
    }

    /// Set the consecutive-frame patience for down / up steps.
    #[inline]
    #[must_use]
    pub const fn with_patience(mut self, downgrade_patience: u32, upgrade_patience: u32) -> Self {
        self.downgrade_patience = downgrade_patience;
        self.upgrade_patience = upgrade_patience;
        self
    }

    /// Set the post-change cooldown length in frames.
    #[inline]
    #[must_use]
    pub const fn with_cooldown(mut self, cooldown_frames: u32) -> Self {
        self.cooldown_frames = cooldown_frames;
        self
    }

    /// Re-clamp every field into a self-consistent state: `max_level >=
    /// min_level`, `upgrade_ppm <= downgrade_ppm <= severe_ppm`, and the step /
    /// patience fields `>= 1`.
    #[inline]
    #[must_use]
    const fn clamped(mut self) -> Self {
        if self.max_level < self.min_level {
            self.max_level = self.min_level;
        }
        if self.upgrade_ppm > self.downgrade_ppm {
            self.upgrade_ppm = self.downgrade_ppm;
        }
        if self.severe_ppm < self.downgrade_ppm {
            self.severe_ppm = self.downgrade_ppm;
        }
        if self.severe_step < 1 {
            self.severe_step = 1;
        }
        if self.downgrade_patience < 1 {
            self.downgrade_patience = 1;
        }
        if self.upgrade_patience < 1 {
            self.upgrade_patience = 1;
        }
        self
    }
}

impl Default for AdaptiveQualityConfig {
    /// A `0..=4` five-level ladder with the default thresholds.
    #[inline]
    fn default() -> Self {
        Self::new(0, 4)
    }
}

/// The decision an [`observe`](AdaptiveQualityController::observe) produced.
/// Higher level == higher quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityAdjustment {
    /// No change this frame.
    Hold,
    /// Quality stepped down (frame-rate protection).
    Downgrade {
        /// Level before the change.
        from: u32,
        /// Level after the change.
        to: u32,
    },
    /// Quality stepped up (durable headroom reclaimed).
    Upgrade {
        /// Level before the change.
        from: u32,
        /// Level after the change.
        to: u32,
    },
}

impl QualityAdjustment {
    /// Whether the level actually changed.
    #[inline]
    #[must_use]
    pub const fn changed(&self) -> bool {
        !matches!(self, Self::Hold)
    }

    /// The resulting level, if the decision carries one.
    #[inline]
    #[must_use]
    pub const fn to_level(&self) -> Option<u32> {
        match self {
            Self::Hold => None,
            Self::Downgrade { to, .. } | Self::Upgrade { to, .. } => Some(*to),
        }
    }

    /// Signed level change (`+` up, `-` down, `0` hold).
    #[inline]
    #[must_use]
    pub const fn delta(&self) -> i64 {
        match self {
            Self::Hold => 0,
            Self::Downgrade { from, to } | Self::Upgrade { from, to } => {
                *to as i64 - *from as i64
            }
        }
    }
}

/// A deterministic frame-budget-driven adaptive quality controller.
///
/// Feed it `(frame_time, budget)` once per frame via
/// [`observe`](Self::observe); it returns the [`QualityAdjustment`] to apply
/// and updates its internal level. See the [module docs](self) for the full
/// control law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveQualityController {
    config: AdaptiveQualityConfig,
    level: u32,
    over_streak: u32,
    under_streak: u32,
    cooldown_remaining: u32,
}

impl AdaptiveQualityController {
    /// Create a controller starting at `start_level` (clamped into the config's
    /// range) with the given, re-clamped, [`AdaptiveQualityConfig`].
    #[inline]
    #[must_use]
    pub fn new(config: AdaptiveQualityConfig, start_level: u32) -> Self {
        let config = config.clamped();
        let level = start_level.clamp(config.min_level, config.max_level);
        Self {
            config,
            level,
            over_streak: 0,
            under_streak: 0,
            cooldown_remaining: 0,
        }
    }

    /// Create a controller at the highest quality level — the usual starting
    /// point, letting the controller back off under load.
    #[inline]
    #[must_use]
    pub fn at_max(config: AdaptiveQualityConfig) -> Self {
        let max = config.clamped().max_level;
        Self::new(config, max)
    }

    /// The current quality level.
    #[inline]
    #[must_use]
    pub const fn level(&self) -> u32 {
        self.level
    }

    /// The active configuration (already clamped).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &AdaptiveQualityConfig {
        &self.config
    }

    /// Whether the controller sits at its lowest quality level.
    #[inline]
    #[must_use]
    pub const fn is_at_min(&self) -> bool {
        self.level == self.config.min_level
    }

    /// Whether the controller sits at its highest quality level.
    #[inline]
    #[must_use]
    pub const fn is_at_max(&self) -> bool {
        self.level == self.config.max_level
    }

    /// Frames remaining in the post-change cooldown.
    #[inline]
    #[must_use]
    pub const fn cooldown_remaining(&self) -> u32 {
        self.cooldown_remaining
    }

    /// Current consecutive over-budget frame count.
    #[inline]
    #[must_use]
    pub const fn over_streak(&self) -> u32 {
        self.over_streak
    }

    /// Current consecutive under-budget frame count.
    #[inline]
    #[must_use]
    pub const fn under_streak(&self) -> u32 {
        self.under_streak
    }

    /// Force the level (clamped to range) and clear streaks and cooldown. For a
    /// hard quality override (user setting, scene change) outside the control
    /// loop.
    #[inline]
    pub fn set_level(&mut self, level: u32) {
        self.level = level.clamp(self.config.min_level, self.config.max_level);
        self.over_streak = 0;
        self.under_streak = 0;
        self.cooldown_remaining = 0;
    }

    /// Reset to `start_level` with cleared streaks and cooldown, keeping the
    /// configuration.
    #[inline]
    pub fn reset(&mut self, start_level: u32) {
        self.set_level(start_level);
    }

    /// Observe one frame and return the adjustment to apply.
    ///
    /// `frame` is the measured frame time; `budget` is the target frame time
    /// (e.g. `1/60 s`). A zero `budget` yields [`QualityAdjustment::Hold`] with
    /// no state change (nothing meaningful to compare against).
    pub fn observe(&mut self, frame: Duration, budget: Duration) -> QualityAdjustment {
        if budget.is_zero() {
            return QualityAdjustment::Hold;
        }
        let util = utilization_ppm(frame, budget);

        // Classify the frame and update consecutive-signal streaks. A neutral
        // frame (inside the hysteresis band) breaks both streaks, so only
        // genuinely consecutive signals accumulate.
        if util >= self.config.downgrade_ppm as u64 {
            self.over_streak = self.over_streak.saturating_add(1);
            self.under_streak = 0;
        } else if util <= self.config.upgrade_ppm as u64 {
            self.under_streak = self.under_streak.saturating_add(1);
            self.over_streak = 0;
        } else {
            self.over_streak = 0;
            self.under_streak = 0;
        }

        // Severe overrun: immediate multi-step bail-out, bypassing patience and
        // cooldown. Only acts when there is still room below.
        if util >= self.config.severe_ppm as u64 && self.level > self.config.min_level {
            let from = self.level;
            let drop = self.config.severe_step;
            let to = from.saturating_sub(drop).max(self.config.min_level);
            self.level = to;
            self.over_streak = 0;
            self.under_streak = 0;
            self.cooldown_remaining = self.config.cooldown_frames;
            return QualityAdjustment::Downgrade { from, to };
        }

        // Normal adjustments are suppressed during the cooldown window.
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            return QualityAdjustment::Hold;
        }

        // Fast downgrade.
        if self.over_streak >= self.config.downgrade_patience && self.level > self.config.min_level
        {
            let from = self.level;
            let to = from - 1;
            self.level = to;
            self.over_streak = 0;
            self.under_streak = 0;
            self.cooldown_remaining = self.config.cooldown_frames;
            return QualityAdjustment::Downgrade { from, to };
        }

        // Slow upgrade.
        if self.under_streak >= self.config.upgrade_patience && self.level < self.config.max_level {
            let from = self.level;
            let to = from + 1;
            self.level = to;
            self.over_streak = 0;
            self.under_streak = 0;
            self.cooldown_remaining = self.config.cooldown_frames;
            return QualityAdjustment::Upgrade { from, to };
        }

        QualityAdjustment::Hold
    }
}

impl Default for AdaptiveQualityController {
    /// A controller at the top of the default `0..=4` ladder.
    #[inline]
    fn default() -> Self {
        Self::at_max(AdaptiveQualityConfig::default())
    }
}
