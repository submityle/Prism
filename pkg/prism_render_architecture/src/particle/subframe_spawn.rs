//! Sub-frame spawn interpolation for fast-moving emitters (design §8.2).
//!
//! When an emitter translates quickly, spawning every new particle at the
//! emitter's *end-of-frame* transform makes fireworks, muzzle flashes, and
//! rocket trails clump into a dotted line: all `N` particles for the frame
//! appear at one point instead of smeared along the path the emitter actually
//! swept during the frame's `dt`. Production `VFX` stacks (Unreal `Niagara`'s
//! *interpolated spawning*, Unity `VFX Graph`) fix this by distributing the
//! frame's spawns across *sub-frame* times and placing each one along the
//! interpolated emitter transform between the previous and current frame.
//!
//! This module owns the `CPU`-verifiable contract for that smearing, kept
//! deliberately distinct from [`super::emitter`]: the emitter module decides
//! *how many* particles to spawn and *where on the emission shape* they sit,
//! whereas this module decides, for a frame that already committed to `N`
//! spawns, *at what sub-frame instant* each one is born and *how to blend* the
//! emitter's previous and current per-frame state at that instant. It also owns
//! the fractional spawn accumulator that turns a continuous spawn rate into an
//! integer count without dropping or duplicating the fractional remainder.
//!
//! # No transcendental math, no float equality
//! Every routine uses only multiply, divide, subtract, comparison, and
//! `floor` — the `floor` extracts the integer part of the accumulator. There is
//! no `sin`/`exp`/`ceil`. Float values are compared with ordered `<`/`<=`
//! relations, never `==`/`!=`. Degenerate inputs (a zero or negative rate, a
//! zero or negative `dt`, a `NaN`, an empty schedule) never panic: they yield a
//! zero count, an unchanged accumulator, or an empty distribution.

use alloc::vec::Vec;

/// A fractional spawn-count accumulator (design §8.2).
///
/// A continuous spawn rate rarely lands on a whole number of particles per
/// frame: `rate * dt` might be `2.7`. Spawning `floor(2.7) = 2` every frame and
/// discarding the `0.7` would systematically under-emit. This accumulator keeps
/// the leftover fraction as `carry` so that fractional remainders add up across
/// frames and eventually release an extra whole particle, exactly matching the
/// requested rate over time.
///
/// `carry` is always in the half-open interval `[0.0, 1.0)` after
/// [`accumulate`](Self::accumulate).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpawnAccumulator {
    /// The fractional particle carried over to the next frame, in `[0.0, 1.0)`.
    pub carry: f32,
}

impl SpawnAccumulator {
    /// A fresh accumulator with no carried fraction.
    #[must_use]
    pub const fn new() -> Self {
        Self { carry: 0.0 }
    }

    /// Consumes `rate * dt + carry` particles for this frame.
    ///
    /// Returns the whole number of particles to spawn now and a new accumulator
    /// whose `carry` is the surviving fractional remainder. A negative or
    /// `NaN` `rate_per_second` or `dt` is treated as "emit nothing": the count
    /// is `0` and the accumulator is returned unchanged, so a paused or
    /// ill-formed frame never spawns and never loses its carry.
    ///
    /// The `floor` extracts the integer part; because `rate`, `dt`, and `carry`
    /// are all non-negative on the accumulating path, the total is non-negative
    /// and the saturating conversion to `u32` cannot wrap.
    #[must_use]
    pub fn accumulate(self, rate_per_second: f32, dt: f32) -> (u32, SpawnAccumulator) {
        if rate_per_second.is_nan() || dt.is_nan() || rate_per_second < 0.0 || dt < 0.0 {
            return (0, self);
        }
        let total = rate_per_second * dt + self.carry;
        let whole_f = total.floor();
        // `total` is non-negative here, so the truncating conversion saturates
        // toward `u32::MAX` at worst and never produces a negative or wrapped
        // count.
        let whole = whole_f as u32;
        let carry = total - whole_f;
        (whole, SpawnAccumulator { carry })
    }
}

/// A centered sub-frame schedule for a frame's committed spawn count.
///
/// Given `spawn_count` particles to emit this frame, the schedule assigns each
/// one a *sub-frame fraction* in `[0.0, 1.0)` marking when, within the frame's
/// `dt`, it is born. The fractions are centered — the `i`-th of `N` spawns
/// lands at `(i + 0.5) / N` — so the births are spread evenly across the frame
/// without piling onto either the start or the end boundary. The values are
/// strictly increasing and every one is strictly less than `1.0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubframeSchedule {
    /// The number of particles spawned this frame.
    pub spawn_count: u32,
}

impl SubframeSchedule {
    /// A schedule for `spawn_count` particles this frame.
    #[must_use]
    pub const fn new(spawn_count: u32) -> Self {
        Self { spawn_count }
    }

    /// The centered sub-frame fraction of the `i`-th spawn, in `[0.0, 1.0)`.
    ///
    /// Returns `(i + 0.5) / spawn_count`. An empty schedule (`spawn_count == 0`)
    /// has no spawns to place and returns `0.0`.
    #[must_use]
    pub fn fraction(&self, i: u32) -> f32 {
        if self.spawn_count == 0 {
            return 0.0;
        }
        // Spawn counts are small (a per-frame budget), and both casts are of
        // non-negative counts; the crate does not enable the pedantic cast
        // lints, matching the surrounding `as f32` count arithmetic.
        let numerator = i as f32 + 0.5;
        let denominator = self.spawn_count as f32;
        numerator / denominator
    }

    /// All centered sub-frame fractions for this frame, in spawn order.
    ///
    /// The returned vector has `spawn_count` entries, strictly increasing and
    /// all within `[0.0, 1.0)`; an empty schedule yields an empty vector.
    #[must_use]
    pub fn fractions(&self) -> Vec<f32> {
        (0..self.spawn_count).map(|i| self.fraction(i)).collect()
    }
}

/// Linearly interpolates an emitter position between two frames.
///
/// Blends `prev` (previous-frame transform origin) toward `curr`
/// (current-frame transform origin) by `frac`. Written as
/// `prev + (curr - prev) * frac` so the endpoints are exact: `frac == 0.0`
/// returns `prev` and `frac == 1.0` returns `curr`, both bit-for-bit.
#[must_use]
pub fn interpolate_position(prev: [f32; 3], curr: [f32; 3], frac: f32) -> [f32; 3] {
    [
        prev[0] + (curr[0] - prev[0]) * frac,
        prev[1] + (curr[1] - prev[1]) * frac,
        prev[2] + (curr[2] - prev[2]) * frac,
    ]
}

/// Linearly interpolates a scalar emitter property between two frames.
///
/// Blends `prev` toward `curr` by `frac` with exact endpoints: `frac == 0.0`
/// returns `prev` and `frac == 1.0` returns `curr`.
#[must_use]
pub fn interpolate_scalar(prev: f32, curr: f32, frac: f32) -> f32 {
    prev + (curr - prev) * frac
}

/// A one-shot burst that fires when a scheduled instant is crossed.
///
/// A burst releases `count` particles the first frame whose time interval
/// contains its scheduled `time`. Emission is driven off the same dilated clock
/// as motion (design §25), so `time` is measured in that clock's seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpawnBurst {
    /// The number of particles this burst releases when it fires.
    pub count: u32,
    /// The clock time at which the burst is scheduled to fire.
    pub time: f32,
}

impl SpawnBurst {
    /// A burst of `count` particles scheduled at `time`.
    #[must_use]
    pub const fn new(count: u32, time: f32) -> Self {
        Self { count, time }
    }

    /// Returns `true` when this frame's interval crosses the burst instant.
    ///
    /// The frame advances the clock from `prev_time` (exclusive) to `curr_time`
    /// (inclusive); the burst is due when `prev_time < time <= curr_time`. The
    /// half-open convention means a burst fires exactly once as the clock sweeps
    /// past it and never double-fires on the shared boundary between two frames.
    #[must_use]
    pub fn is_due(&self, prev_time: f32, curr_time: f32) -> bool {
        prev_time < self.time && self.time <= curr_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for comparing interpolated / accumulated floats in tests.
    const CMP_EPS: f32 = 1.0e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn accumulator_exact_when_product_is_whole() {
        let acc = SpawnAccumulator::new();
        let (whole, next) = acc.accumulate(2.0, 0.5);
        assert_eq!(whole, 1);
        assert!(close(next.carry, 0.0));
    }

    #[test]
    fn accumulator_preserves_fractional_carry() {
        let acc = SpawnAccumulator::new();
        let (whole, next) = acc.accumulate(2.5, 1.0);
        assert_eq!(whole, 2);
        assert!(close(next.carry, 0.5));
    }

    #[test]
    fn accumulator_carry_releases_extra_particle_across_frames() {
        // 0.3 particles/frame for ten frames == exactly 3.0 particles total.
        let mut acc = SpawnAccumulator::new();
        let mut total = 0u32;
        for _ in 0..10 {
            let (whole, next) = acc.accumulate(3.0, 0.1);
            total += whole;
            acc = next;
        }
        assert_eq!(total, 3);
        assert!(acc.carry < 1.0);
    }

    #[test]
    fn accumulator_matches_rate_over_many_frames() {
        // 7 particles/second at 1/60 s per frame over 600 frames == 70.
        let mut acc = SpawnAccumulator::new();
        let mut total = 0u32;
        for _ in 0..600 {
            let (whole, next) = acc.accumulate(7.0, 1.0 / 60.0);
            total += whole;
            acc = next;
        }
        assert_eq!(total, 70);
    }

    #[test]
    fn accumulator_zero_rate_and_zero_dt_emit_nothing() {
        let acc = SpawnAccumulator { carry: 0.4 };
        let (whole_rate, next_rate) = acc.accumulate(0.0, 1.0 / 60.0);
        assert_eq!(whole_rate, 0);
        assert!(close(next_rate.carry, 0.4));

        let (whole_dt, next_dt) = acc.accumulate(120.0, 0.0);
        assert_eq!(whole_dt, 0);
        assert!(close(next_dt.carry, 0.4));
    }

    #[test]
    fn accumulator_guards_negative_and_nan_inputs() {
        let acc = SpawnAccumulator { carry: 0.25 };

        let (neg_rate, after_neg_rate) = acc.accumulate(-5.0, 0.5);
        assert_eq!(neg_rate, 0);
        assert!(close(after_neg_rate.carry, 0.25));

        let (neg_dt, after_neg_dt) = acc.accumulate(5.0, -0.5);
        assert_eq!(neg_dt, 0);
        assert!(close(after_neg_dt.carry, 0.25));

        let (nan_rate, after_nan) = acc.accumulate(f32::NAN, 0.5);
        assert_eq!(nan_rate, 0);
        assert!(close(after_nan.carry, 0.25));
    }

    #[test]
    fn schedule_count_zero_is_empty() {
        let schedule = SubframeSchedule::new(0);
        assert!(schedule.fractions().is_empty());
        assert!(close(schedule.fraction(0), 0.0));
    }

    #[test]
    fn schedule_fractions_are_centered() {
        let schedule = SubframeSchedule::new(4);
        let fractions = schedule.fractions();
        assert_eq!(fractions.len(), 4);
        assert!(close(fractions[0], 0.125));
        assert!(close(fractions[1], 0.375));
        assert!(close(fractions[2], 0.625));
        assert!(close(fractions[3], 0.875));
    }

    #[test]
    fn schedule_fractions_monotonic_and_in_unit_range() {
        let schedule = SubframeSchedule::new(16);
        let fractions = schedule.fractions();
        assert_eq!(fractions.len(), 16);
        let mut prev = -1.0f32;
        for f in fractions {
            assert!(f >= 0.0);
            assert!(f < 1.0);
            assert!(f > prev);
            prev = f;
        }
    }

    #[test]
    fn schedule_single_spawn_sits_at_frame_center() {
        let schedule = SubframeSchedule::new(1);
        assert!(close(schedule.fraction(0), 0.5));
    }

    #[test]
    fn interpolate_position_hits_endpoints() {
        let prev = [1.0, 2.0, 3.0];
        let curr = [5.0, -2.0, 9.0];
        let at_start = interpolate_position(prev, curr, 0.0);
        let at_end = interpolate_position(prev, curr, 1.0);
        for k in 0..3 {
            assert!(close(at_start[k], prev[k]));
            assert!(close(at_end[k], curr[k]));
        }
    }

    #[test]
    fn interpolate_position_midpoint_is_average() {
        let prev = [0.0, 0.0, 0.0];
        let curr = [4.0, 8.0, -12.0];
        let mid = interpolate_position(prev, curr, 0.5);
        assert!(close(mid[0], 2.0));
        assert!(close(mid[1], 4.0));
        assert!(close(mid[2], -6.0));
    }

    #[test]
    fn interpolate_scalar_hits_endpoints_and_midpoint() {
        assert!(close(interpolate_scalar(3.0, 7.0, 0.0), 3.0));
        assert!(close(interpolate_scalar(3.0, 7.0, 1.0), 7.0));
        assert!(close(interpolate_scalar(3.0, 7.0, 0.25), 4.0));
    }

    #[test]
    fn burst_is_due_only_inside_the_interval() {
        let burst = SpawnBurst::new(32, 1.5);
        assert!(burst.is_due(1.0, 2.0));
        assert!(!burst.is_due(2.0, 3.0));
        assert!(!burst.is_due(0.0, 1.0));
    }

    #[test]
    fn burst_boundary_fires_once_on_upper_edge() {
        let burst = SpawnBurst::new(8, 2.0);
        // Inclusive upper edge fires; the next frame's exclusive lower edge does
        // not, so the burst releases exactly once across the shared boundary.
        assert!(burst.is_due(1.5, 2.0));
        assert!(!burst.is_due(2.0, 2.5));
    }
}
