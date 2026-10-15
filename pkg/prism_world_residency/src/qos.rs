use core::fmt;

/// Discrete quality levels used by the world scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QualityTier {
    /// Minimum work rate.
    Low,
    /// Reduced work rate.
    Medium,
    /// Normal work rate.
    High,
    /// Aggressive work rate when sustained headroom exists.
    Ultra,
}

impl QualityTier {
    /// Returns a stable ordinal for comparisons and telemetry.
    pub const fn rank(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
            Self::Ultra => 3,
        }
    }

    /// Returns the scheduler work-rate scale in per-mille.
    pub const fn work_rate_milli(self) -> u16 {
        match self {
            Self::Low => 500,
            Self::Medium => 750,
            Self::High => 1_000,
            Self::Ultra => 1_250,
        }
    }

    const fn step_down(self) -> Self {
        match self {
            Self::Low => Self::Low,
            Self::Medium => Self::Low,
            Self::High => Self::Medium,
            Self::Ultra => Self::High,
        }
    }

    const fn step_up(self) -> Self {
        match self {
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Ultra,
            Self::Ultra => Self::Ultra,
        }
    }
}

/// Hysteresis and rate-limiting parameters for [`QosGovernor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QosConfig {
    /// Target frame duration in microseconds.
    pub target_frame_micros: u32,
    /// Extra duration tolerated before a frame is classified as slow.
    pub slow_margin_micros: u32,
    /// Headroom required before a frame is classified as fast.
    pub fast_margin_micros: u32,
    /// Consecutive slow frames needed for one downshift.
    pub downshift_frames: u16,
    /// Consecutive fast frames needed for one upshift.
    pub upshift_frames: u16,
    /// Frames to hold after a tier change.
    pub hold_frames: u16,
}

impl QosConfig {
    /// Creates a validated governor configuration.
    pub const fn new(
        target_frame_micros: u32,
        slow_margin_micros: u32,
        fast_margin_micros: u32,
        downshift_frames: u16,
        upshift_frames: u16,
        hold_frames: u16,
    ) -> Option<Self> {
        if target_frame_micros == 0 || downshift_frames == 0 || upshift_frames == 0 {
            None
        } else {
            Some(Self {
                target_frame_micros,
                slow_margin_micros,
                fast_margin_micros,
                downshift_frames,
                upshift_frames,
                hold_frames,
            })
        }
    }

    /// A practical 60 Hz configuration with three-frame hysteresis.
    pub const fn sixty_hz() -> Self {
        Self {
            target_frame_micros: 16_666,
            slow_margin_micros: 1_000,
            fast_margin_micros: 1_000,
            downshift_frames: 3,
            upshift_frames: 8,
            hold_frames: 4,
        }
    }
}

/// The result of observing one frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QosDecision {
    /// Tier before the observation.
    pub previous: QualityTier,
    /// Tier after the observation.
    pub current: QualityTier,
    /// Whether this observation changed the tier.
    pub changed: bool,
    /// Current work-rate scale in per-mille.
    pub work_rate_milli: u16,
}

/// Frame-time feedback controller with slow/fast hysteresis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QosGovernor {
    config: QosConfig,
    tier: QualityTier,
    slow_streak: u16,
    fast_streak: u16,
    hold_remaining: u16,
}

impl QosGovernor {
    /// Creates a governor at the requested initial tier.
    pub const fn new(config: QosConfig, initial: QualityTier) -> Self {
        Self {
            config,
            tier: initial,
            slow_streak: 0,
            fast_streak: 0,
            hold_remaining: 0,
        }
    }

    /// Returns the current quality tier.
    pub const fn tier(self) -> QualityTier {
        self.tier
    }

    /// Returns the immutable feedback configuration.
    pub const fn config(self) -> QosConfig {
        self.config
    }

    /// Observes one completed frame and applies at most one tier step.
    pub fn observe(&mut self, frame_time_micros: u32) -> QosDecision {
        let previous = self.tier;
        if self.hold_remaining > 0 {
            self.hold_remaining -= 1;
            self.slow_streak = 0;
            self.fast_streak = 0;
            return self.decision(previous);
        }
        let slow_limit = self
            .config
            .target_frame_micros
            .saturating_add(self.config.slow_margin_micros);
        let fast_limit = self
            .config
            .target_frame_micros
            .saturating_sub(self.config.fast_margin_micros);
        if frame_time_micros > slow_limit {
            self.slow_streak = self.slow_streak.saturating_add(1);
            self.fast_streak = 0;
            if self.slow_streak >= self.config.downshift_frames {
                self.tier = self.tier.step_down();
                self.slow_streak = 0;
                self.hold_remaining = self.config.hold_frames;
            }
        } else if frame_time_micros < fast_limit {
            self.fast_streak = self.fast_streak.saturating_add(1);
            self.slow_streak = 0;
            if self.fast_streak >= self.config.upshift_frames {
                self.tier = self.tier.step_up();
                self.fast_streak = 0;
                self.hold_remaining = self.config.hold_frames;
            }
        } else {
            self.slow_streak = 0;
            self.fast_streak = 0;
        }
        self.decision(previous)
    }

    fn decision(self, previous: QualityTier) -> QosDecision {
        QosDecision {
            previous,
            current: self.tier,
            changed: previous != self.tier,
            work_rate_milli: self.tier.work_rate_milli(),
        }
    }
}

impl fmt::Display for QualityTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => f.write_str("low"),
            Self::Medium => f.write_str("medium"),
            Self::High => f.write_str("high"),
            Self::Ultra => f.write_str("ultra"),
        }
    }
}
