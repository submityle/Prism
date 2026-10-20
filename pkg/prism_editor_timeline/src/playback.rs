//! Forward integer-tick playback independent of clocks and runtime domains.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

use crate::{Timeline, TimelineError, TimelineEvent, TrackId};

/// Behavior at the end of a closed playback interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaybackMode {
    /// Hold the end tick and finish after delivering its events.
    Once,
    /// Visit the start tick one tick after the end tick, indefinitely.
    Loop,
}

/// Immutable bounds and work budget for a player.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaybackConfig {
    /// Number of timeline ticks per second at rate one; must be nonzero.
    pub ticks_per_second: u64,
    /// First tick in the closed playback interval.
    pub start_tick: u64,
    /// Last tick in the interval; must be at least `start_tick`.
    pub end_tick: u64,
    /// End-of-interval policy.
    pub mode: PlaybackMode,
    /// Maximum event occurrences returned by one successful advance.
    /// Zero permits advances that encounter no events.
    pub max_events: usize,
}

/// A positive, reduced rational speed multiplier; reverse playback is excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaybackRate {
    numerator: u64,
    denominator: u64,
}

impl PlaybackRate {
    /// Validates and reduces a positive rational speed.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, PlaybackError> {
        if numerator == 0 || denominator == 0 {
            return Err(PlaybackError::InvalidRate);
        }
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Ok(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }

    /// Reduced numerator and denominator.
    #[must_use]
    pub const fn ratio(self) -> (u64, u64) {
        (self.numerator, self.denominator)
    }
}

impl Default for PlaybackRate {
    fn default() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }
}

/// Transport status, including an explicitly completed one-shot traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaybackStatus {
    /// Advances sample the current position without consuming elapsed time.
    Paused,
    /// Advances consume elapsed time and deliver crossed events.
    Playing,
    /// A one-shot player reached the end; seek before playing again.
    Finished,
}

/// One actual timeline event occurrence, borrowing its source data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaybackEvent<'a> {
    /// Source event; same-tick occurrences retain ascending event-ID order.
    pub event: &'a TimelineEvent,
    /// Zero-based loop traversal, reset by seek.
    pub iteration: u128,
}

/// Samples at the destination and all events encountered in traversal order.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackFrame<'a> {
    /// Current integer tick.
    pub tick: u64,
    /// Current zero-based loop traversal.
    pub iteration: u128,
    /// Status after this operation.
    pub status: PlaybackStatus,
    /// Actual sampled numeric channels, keyed by stable track identity.
    /// Tracks before their first key are absent, as in `Timeline::sample`.
    pub samples: BTreeMap<TrackId, Vec<f64>>,
    /// Actual crossed event occurrences, bounded by the configured budget.
    pub events: Vec<PlaybackEvent<'a>>,
}

/// Rejected playback operation; errors never partially advance the player.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlaybackError {
    /// Tick frequency is zero, or the closed interval is reversed.
    InvalidConfig,
    /// A speed numerator or denominator is zero.
    InvalidRate,
    /// Seek destination lies outside the closed interval.
    SeekOutOfBounds,
    /// Time, position arithmetic, or the loop generation cannot fit in u128.
    ArithmeticOverflow,
    /// The complete traversal would exceed the event occurrence budget.
    EventBudgetExceeded {
        /// Configured per-advance maximum.
        limit: usize,
    },
    /// Real track evaluation failed, for example cubic sample overflow.
    Sample(TimelineError),
}

impl fmt::Display for PlaybackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid playback interval or frequency"),
            Self::InvalidRate => formatter.write_str("playback rate must be positive"),
            Self::SeekOutOfBounds => formatter.write_str("seek lies outside playback interval"),
            Self::ArithmeticOverflow => formatter.write_str("playback integer arithmetic overflow"),
            Self::EventBudgetExceeded { limit } => {
                write!(formatter, "playback event budget {limit} exceeded")
            }
            Self::Sample(error) => write!(formatter, "playback sample failed: {error}"),
        }
    }
}

impl core::error::Error for PlaybackError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Sample(error) => Some(error),
            _ => None,
        }
    }
}

/// Caller-clocked, forward player with transactional advancement.
///
/// Creation and successful seek arm the current tick's events for the first
/// successful playing advance, including a zero-duration advance. Thereafter
/// only newly visited ticks emit events. Paused advances never emit events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Playback {
    config: PlaybackConfig,
    rate: PlaybackRate,
    tick: u64,
    iteration: u128,
    remainder: u128,
    status: PlaybackStatus,
    origin_pending: bool,
}

impl Playback {
    /// Creates a paused player at the start tick, with rate one.
    pub fn new(config: PlaybackConfig) -> Result<Self, PlaybackError> {
        if config.ticks_per_second == 0 || config.start_tick > config.end_tick {
            return Err(PlaybackError::InvalidConfig);
        }
        Ok(Self {
            tick: config.start_tick,
            config,
            rate: PlaybackRate::default(),
            iteration: 0,
            remainder: 0,
            status: PlaybackStatus::Paused,
            origin_pending: true,
        })
    }

    /// Immutable interval, tick frequency, and event budget.
    #[must_use]
    pub const fn config(&self) -> PlaybackConfig {
        self.config
    }

    /// Current tick.
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Current loop traversal, reset by seek.
    #[must_use]
    pub const fn iteration(&self) -> u128 {
        self.iteration
    }

    /// Current transport status.
    #[must_use]
    pub const fn status(&self) -> PlaybackStatus {
        self.status
    }

    /// Current rational playback rate.
    #[must_use]
    pub const fn rate(&self) -> PlaybackRate {
        self.rate
    }

    /// Exact unconsumed fraction of a tick, as numerator and denominator.
    #[must_use]
    pub const fn fractional_tick(&self) -> (u128, u128) {
        (
            self.remainder,
            1_000_000_000 * self.rate.denominator as u128,
        )
    }

    /// Changes speed and discards the old fractional tick.
    /// An unchanged rate preserves the fraction. Position and event cursor stay put.
    pub fn set_rate(&mut self, rate: PlaybackRate) {
        if self.rate != rate {
            self.rate = rate;
            self.remainder = 0;
        }
    }

    /// Starts or resumes. Returns false for a finished one-shot player.
    pub fn play(&mut self) -> bool {
        if self.status == PlaybackStatus::Finished {
            return false;
        }
        self.status = PlaybackStatus::Playing;
        true
    }

    /// Pauses a playing player, preserving position, fraction, and event cursor.
    pub fn pause(&mut self) {
        if self.status == PlaybackStatus::Playing {
            self.status = PlaybackStatus::Paused;
        }
    }

    /// Samples an exact destination without emitting crossed events.
    ///
    /// On success, resets the fraction and loop generation and re-arms destination
    /// events. Playing stays playing; paused or finished becomes paused. Sampling
    /// or bounds failure leaves every field unchanged.
    pub fn seek<'a>(
        &mut self,
        timeline: &'a Timeline,
        tick: u64,
    ) -> Result<PlaybackFrame<'a>, PlaybackError> {
        if !(self.config.start_tick..=self.config.end_tick).contains(&tick) {
            return Err(PlaybackError::SeekOutOfBounds);
        }
        let samples = timeline.sample(tick).map_err(PlaybackError::Sample)?;
        self.tick = tick;
        self.iteration = 0;
        self.remainder = 0;
        self.origin_pending = true;
        if self.status == PlaybackStatus::Finished {
            self.status = PlaybackStatus::Paused;
        }
        Ok(self.frame(samples, Vec::new()))
    }

    /// Consumes caller-supplied elapsed time and evaluates the real timeline.
    ///
    /// All event counts, integer arithmetic, and sampling succeed before any
    /// state is committed. An error consumes no time and emits no partial batch;
    /// callers must retry that time, split it into smaller advances, or stop.
    /// Event-free whole loops are skipped without iterating over their ticks.
    pub fn advance<'a>(
        &mut self,
        timeline: &'a Timeline,
        elapsed: Duration,
    ) -> Result<PlaybackFrame<'a>, PlaybackError> {
        let mut next = self.clone();
        let mut segments = Vec::with_capacity(4);
        if self.status == PlaybackStatus::Playing {
            let (ticks, remainder) = self.elapsed_ticks(elapsed)?;
            next.remainder = remainder;
            if self.origin_pending {
                segments.push(Segment::new(self.tick, self.tick, self.iteration, 1));
            }
            next.origin_pending = false;
            self.plan(ticks, &mut next, &mut segments)?;
        }

        let mut count = 0;
        for segment in &mut segments {
            let remaining = self.config.max_events - count;
            for _ in timeline.events_in(segment.start..=segment.end) {
                if segment.event_count == remaining {
                    return Err(self.budget_error());
                }
                segment.event_count += 1;
            }
            if let Some(max_repeats) = remaining.checked_div(segment.event_count) {
                if segment.repeats > max_repeats as u128 {
                    return Err(self.budget_error());
                }
                // The budget comparison proves this conversion/product fit.
                count += segment.event_count * segment.repeats as usize;
            }
        }
        let samples = timeline.sample(next.tick).map_err(PlaybackError::Sample)?;
        let mut events = Vec::with_capacity(count);
        for segment in segments {
            if segment.event_count == 0 {
                continue;
            }
            for offset in 0..segment.repeats {
                for event in timeline.events_in(segment.start..=segment.end) {
                    events.push(PlaybackEvent {
                        event,
                        iteration: segment.iteration + offset,
                    });
                }
            }
        }
        *self = next;
        Ok(self.frame(samples, events))
    }

    fn budget_error(&self) -> PlaybackError {
        PlaybackError::EventBudgetExceeded {
            limit: self.config.max_events,
        }
    }

    fn frame<'a>(
        &self,
        samples: BTreeMap<TrackId, Vec<f64>>,
        events: Vec<PlaybackEvent<'a>>,
    ) -> PlaybackFrame<'a> {
        PlaybackFrame {
            tick: self.tick,
            iteration: self.iteration,
            status: self.status,
            samples,
            events,
        }
    }

    fn elapsed_ticks(&self, elapsed: Duration) -> Result<(u128, u128), PlaybackError> {
        let denominator = self.fractional_tick().1;
        let factor = self.config.ticks_per_second as u128 * self.rate.numerator as u128;
        let base_whole = elapsed.as_nanos() / denominator;
        let base_fraction = elapsed.as_nanos() % denominator;
        let (mut whole, mut fraction) = (0_u128, 0_u128);
        // Long multiplication of quotient/remainder pairs avoids a false
        // overflow from nanoseconds * frequency when the tick quotient fits.
        // Fractions stay below 3 * denominator, far below u128::MAX.
        for bit in (0..(128 - factor.leading_zeros())).rev() {
            whole = whole
                .checked_mul(2)
                .ok_or(PlaybackError::ArithmeticOverflow)?;
            fraction *= 2;
            if factor & (1_u128 << bit) != 0 {
                whole = whole
                    .checked_add(base_whole)
                    .ok_or(PlaybackError::ArithmeticOverflow)?;
                fraction += base_fraction;
            }
            whole = whole
                .checked_add(fraction / denominator)
                .ok_or(PlaybackError::ArithmeticOverflow)?;
            fraction %= denominator;
        }
        fraction += self.remainder;
        whole = whole
            .checked_add(fraction / denominator)
            .ok_or(PlaybackError::ArithmeticOverflow)?;
        Ok((whole, fraction % denominator))
    }

    fn plan(
        &self,
        ticks: u128,
        next: &mut Self,
        segments: &mut Vec<Segment>,
    ) -> Result<(), PlaybackError> {
        let start = self.config.start_tick;
        let end = self.config.end_tick;
        let length = end as u128 - start as u128 + 1;
        let offset = self.tick as u128 - start as u128;
        match self.config.mode {
            PlaybackMode::Once => {
                let distance = end as u128 - self.tick as u128;
                next.tick = self.tick + ticks.min(distance) as u64;
                if next.tick > self.tick {
                    segments.push(Segment::new(self.tick + 1, next.tick, self.iteration, 1));
                }
                if ticks >= distance {
                    next.status = PlaybackStatus::Finished;
                    next.remainder = 0;
                }
            }
            PlaybackMode::Loop => {
                let total = offset
                    .checked_add(ticks)
                    .ok_or(PlaybackError::ArithmeticOverflow)?;
                let wraps = total / length;
                next.tick = start + (total % length) as u64;
                next.iteration = self
                    .iteration
                    .checked_add(wraps)
                    .ok_or(PlaybackError::ArithmeticOverflow)?;
                if wraps == 0 {
                    if next.tick > self.tick {
                        segments.push(Segment::new(self.tick + 1, next.tick, self.iteration, 1));
                    }
                } else {
                    if self.tick < end {
                        segments.push(Segment::new(self.tick + 1, end, self.iteration, 1));
                    }
                    if wraps > 1 {
                        segments.push(Segment::new(start, end, self.iteration + 1, wraps - 1));
                    }
                    segments.push(Segment::new(start, next.tick, next.iteration, 1));
                }
            }
        }
        Ok(())
    }
}

struct Segment {
    start: u64,
    end: u64,
    iteration: u128,
    repeats: u128,
    event_count: usize,
}

impl Segment {
    fn new(start: u64, end: u64, iteration: u128, repeats: u128) -> Self {
        Self {
            start,
            end,
            iteration,
            repeats,
            event_count: 0,
        }
    }
}
