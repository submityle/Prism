//! Bounded snapshot interpolation and capped extrapolation (design §10.5).
//!
//! The buffer stores authoritative snapshots by simulation tick. A renderer
//! asks for a delayed tick cursor so normal jitter is absorbed by interpolation
//! between the nearest retained samples. If the next sample has not arrived,
//! the caller may opt into a short, explicitly capped extrapolation from the
//! last two samples. The module is pure `no_std + alloc`; transport arrival
//! timing, visual error smoothing, and transform-specific interpolation stay
//! outside this storage contract.

use alloc::collections::VecDeque;

/// A source snapshot at one authoritative simulation tick.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot<T> {
    /// Monotonic source tick.
    pub tick: u64,
    /// Application-defined replicated value.
    pub value: T,
}

/// A bounded snapshot-buffer policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolationConfig {
    /// Maximum number of snapshots retained.
    pub capacity: usize,
    /// Maximum distance beyond the newest sample that extrapolation may cover.
    pub max_extrapolation_ticks: u64,
}

impl InterpolationConfig {
    /// Construct a policy. Capacity must be non-zero.
    #[inline]
    pub const fn new(capacity: usize, max_extrapolation_ticks: u64) -> Self {
        Self {
            capacity,
            max_extrapolation_ticks,
        }
    }
}

/// The way a rendered value was produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleMode {
    /// The requested cursor landed exactly on a retained source tick.
    Exact,
    /// The value was produced between two retained source ticks.
    Interpolated,
    /// The value was projected past the newest retained source tick.
    Extrapolated,
}

/// A sampled value plus provenance useful for diagnostics and smoothing.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample<T> {
    /// The value returned by the caller's interpolation or extrapolation rule.
    pub value: T,
    /// How the value was produced.
    pub mode: SampleMode,
    /// Older source tick used by the operation.
    pub from_tick: u64,
    /// Newer source tick used by the operation. For extrapolation this is the
    /// newest retained tick.
    pub to_tick: u64,
}

/// Errors that preserve bounded, ordered snapshot semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationError {
    /// The buffer capacity was zero.
    InvalidCapacity,
    /// A snapshot arrived older than the newest accepted tick.
    NonMonotonicTick {
        /// Newest accepted tick.
        previous: u64,
        /// Tick that was rejected.
        received: u64,
    },
    /// The render fraction was NaN, infinite, or outside `[0, 1]`.
    InvalidFraction,
    /// No snapshot has been retained.
    Empty,
    /// The cursor predates the oldest retained snapshot.
    BeforeOldest {
        /// Oldest retained tick.
        oldest: u64,
        /// Requested integral tick.
        requested: u64,
    },
    /// The cursor is beyond the configured extrapolation window.
    ExtrapolationLimit {
        /// Newest retained tick.
        newest: u64,
        /// Requested integral tick.
        requested: u64,
        /// Configured maximum extrapolation distance.
        max_ticks: u64,
    },
    /// Extrapolation was requested but fewer than two samples are available.
    ExtrapolationUnavailable,
}

/// A bounded, monotonic snapshot buffer.
#[derive(Clone, Debug)]
pub struct SnapshotBuffer<T> {
    config: InterpolationConfig,
    snapshots: VecDeque<Snapshot<T>>,
}

impl<T> SnapshotBuffer<T> {
    /// Create an empty buffer.
    pub fn new(config: InterpolationConfig) -> Result<Self, InterpolationError> {
        if config.capacity == 0 {
            return Err(InterpolationError::InvalidCapacity);
        }
        Ok(Self {
            config,
            snapshots: VecDeque::new(),
        })
    }

    /// The active policy.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> InterpolationConfig {
        self.config
    }

    /// Number of retained snapshots.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether no snapshots are retained.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Oldest retained tick, if any.
    #[inline]
    #[must_use]
    pub fn oldest_tick(&self) -> Option<u64> {
        self.snapshots.front().map(|snapshot| snapshot.tick)
    }

    /// Newest retained tick, if any.
    #[inline]
    #[must_use]
    pub fn newest_tick(&self) -> Option<u64> {
        self.snapshots.back().map(|snapshot| snapshot.tick)
    }

    /// Insert a snapshot, replacing a duplicate tick in place.
    pub fn push(&mut self, tick: u64, value: T) -> Result<(), InterpolationError> {
        if let Some(previous) = self.newest_tick() {
            if tick < previous {
                return Err(InterpolationError::NonMonotonicTick {
                    previous,
                    received: tick,
                });
            }
            if tick == previous {
                *self.snapshots.back_mut().expect("newest tick exists") = Snapshot { tick, value };
                return Ok(());
            }
        }
        self.snapshots.push_back(Snapshot { tick, value });
        while self.snapshots.len() > self.config.capacity {
            self.snapshots.pop_front();
        }
        Ok(())
    }

    /// Remove all retained snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Sample a delayed render cursor.
    ///
    /// `render_tick - delay_ticks` is the integral cursor. `fraction` advances
    /// within that tick and must be in `[0, 1]`. The interpolation closure gets
    /// the two values and a ratio in `[0, 1]`; the extrapolation closure gets
    /// the last two values and a non-negative distance in ticks.
    pub fn sample(
        &self,
        render_tick: u64,
        fraction: f32,
        delay_ticks: u64,
        interpolate: impl FnOnce(&T, &T, f32) -> T,
        extrapolate: impl FnOnce(&T, &T, f32) -> T,
    ) -> Result<Sample<T>, InterpolationError>
    where
        T: Clone,
    {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(InterpolationError::InvalidFraction);
        }
        let target_tick = render_tick.saturating_sub(delay_ticks);
        self.sample_at(target_tick, fraction, interpolate, extrapolate)
    }

    fn sample_at(
        &self,
        tick: u64,
        fraction: f32,
        interpolate: impl FnOnce(&T, &T, f32) -> T,
        extrapolate: impl FnOnce(&T, &T, f32) -> T,
    ) -> Result<Sample<T>, InterpolationError>
    where
        T: Clone,
    {
        let oldest = self.oldest_tick().ok_or(InterpolationError::Empty)?;
        let newest = self.newest_tick().expect("oldest implies newest");
        if tick < oldest {
            return Err(InterpolationError::BeforeOldest {
                oldest,
                requested: tick,
            });
        }

        let lower = self.lower_bound(tick);
        let next_tick = tick
            .checked_add(u64::from(fraction > 0.0))
            .unwrap_or(u64::MAX);
        let upper = self.upper_bound(next_tick);

        if let Some(lower_index) = lower {
            let lower_snapshot = self.snapshots.get(lower_index).expect("lower index valid");
            if fraction == 0.0 && lower_snapshot.tick == tick {
                return Ok(Sample {
                    value: lower_snapshot.value.clone(),
                    mode: SampleMode::Exact,
                    from_tick: tick,
                    to_tick: tick,
                });
            }

            if let Some(upper_index) = upper {
                let upper_snapshot = self.snapshots.get(upper_index).expect("upper index valid");
                if upper_snapshot.tick > lower_snapshot.tick {
                    let cursor = tick as f32 + fraction;
                    let span = (upper_snapshot.tick - lower_snapshot.tick) as f32;
                    let ratio = ((cursor - lower_snapshot.tick as f32) / span).clamp(0.0, 1.0);
                    return Ok(Sample {
                        value: interpolate(&lower_snapshot.value, &upper_snapshot.value, ratio),
                        mode: SampleMode::Interpolated,
                        from_tick: lower_snapshot.tick,
                        to_tick: upper_snapshot.tick,
                    });
                }
            }
        }

        let Some(last_two_start) = self.snapshots.len().checked_sub(2) else {
            return Err(InterpolationError::ExtrapolationUnavailable);
        };
        if tick < newest {
            return Err(InterpolationError::ExtrapolationUnavailable);
        }
        let distance = tick.saturating_sub(newest) as f32 + fraction;
        if distance > self.config.max_extrapolation_ticks as f32 {
            return Err(InterpolationError::ExtrapolationLimit {
                newest,
                requested: tick,
                max_ticks: self.config.max_extrapolation_ticks,
            });
        }
        let previous = self
            .snapshots
            .get(last_two_start)
            .expect("second-last exists");
        let latest = self.snapshots.back().expect("newest exists");
        Ok(Sample {
            value: extrapolate(&previous.value, &latest.value, distance),
            mode: SampleMode::Extrapolated,
            from_tick: previous.tick,
            to_tick: latest.tick,
        })
    }

    fn lower_bound(&self, tick: u64) -> Option<usize> {
        let mut lo = 0;
        let mut hi = self.snapshots.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.snapshots.get(mid).expect("index valid").tick <= tick {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.checked_sub(1)
    }

    fn upper_bound(&self, tick: u64) -> Option<usize> {
        let mut lo = 0;
        let mut hi = self.snapshots.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.snapshots.get(mid).expect("index valid").tick < tick {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < self.snapshots.len()).then_some(lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interpolate(a: &i32, b: &i32, t: f32) -> i32 {
        (*a as f32 + (*b as f32 - *a as f32) * t).round() as i32
    }

    fn extrapolate(a: &i32, b: &i32, dt: f32) -> i32 {
        (*b as f32 + (*b as f32 - *a as f32) * dt).round() as i32
    }

    #[test]
    fn interpolation_uses_the_delayed_cursor() {
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(8, 2)).unwrap();
        buffer.push(10, 0).unwrap();
        buffer.push(12, 20).unwrap();
        let sample = buffer.sample(13, 0.0, 2, interpolate, extrapolate).unwrap();
        assert_eq!(sample.mode, SampleMode::Interpolated);
        assert_eq!(sample.from_tick, 10);
        assert_eq!(sample.to_tick, 12);
        assert_eq!(sample.value, 10);
    }

    #[test]
    fn fractional_cursor_interpolates_between_ticks() {
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(4, 1)).unwrap();
        buffer.push(3, 0).unwrap();
        buffer.push(4, 10).unwrap();
        let sample = buffer.sample(4, 0.25, 0, interpolate, extrapolate).unwrap();
        assert_eq!(sample.mode, SampleMode::Extrapolated);
        assert_eq!(sample.value, 13);
    }

    #[test]
    fn missing_next_snapshot_uses_only_capped_extrapolation() {
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(4, 2)).unwrap();
        buffer.push(5, 10).unwrap();
        buffer.push(6, 20).unwrap();
        let sample = buffer.sample(7, 0.5, 0, interpolate, extrapolate).unwrap();
        assert_eq!(sample.mode, SampleMode::Extrapolated);
        assert_eq!(sample.value, 35);
        assert_eq!(sample.from_tick, 5);
        assert_eq!(sample.to_tick, 6);
    }

    #[test]
    fn extrapolation_limit_and_oldest_boundary_are_explicit() {
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(2, 1)).unwrap();
        buffer.push(8, 0).unwrap();
        buffer.push(9, 1).unwrap();
        assert!(matches!(
            buffer.sample(7, 0.0, 0, interpolate, extrapolate),
            Err(InterpolationError::BeforeOldest {
                oldest: 8,
                requested: 7
            })
        ));
        assert!(matches!(
            buffer.sample(11, 0.0, 0, interpolate, extrapolate),
            Err(InterpolationError::ExtrapolationLimit {
                newest: 9,
                requested: 11,
                max_ticks: 1
            })
        ));
    }

    #[test]
    fn duplicate_ticks_replace_and_capacity_remains_bounded() {
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(2, 0)).unwrap();
        buffer.push(1, 1).unwrap();
        buffer.push(2, 2).unwrap();
        buffer.push(2, 20).unwrap();
        buffer.push(3, 3).unwrap();
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.oldest_tick(), Some(2));
        let sample = buffer.sample(2, 0.0, 0, interpolate, extrapolate).unwrap();
        assert_eq!(sample.value, 20);
    }

    #[test]
    fn malformed_inputs_do_not_mutate_the_buffer() {
        assert!(matches!(
            SnapshotBuffer::<i32>::new(InterpolationConfig::new(0, 0)),
            Err(InterpolationError::InvalidCapacity)
        ));
        let mut buffer = SnapshotBuffer::new(InterpolationConfig::new(4, 1)).unwrap();
        buffer.push(4, 1).unwrap();
        assert!(matches!(
            buffer.push(3, 0),
            Err(InterpolationError::NonMonotonicTick {
                previous: 4,
                received: 3
            })
        ));
        assert_eq!(buffer.newest_tick(), Some(4));
        assert_eq!(
            buffer.sample(4, f32::NAN, 0, interpolate, extrapolate),
            Err(InterpolationError::InvalidFraction)
        );
    }
}
