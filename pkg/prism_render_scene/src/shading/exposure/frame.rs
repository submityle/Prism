//! CPU-side exposure dispatch admission and timing.

use std::time::Instant;

/// Both pipeline/bind-group pairs for a view. Kept generic so admission can be
/// tested without constructing a device or pretending to submit GPU work.
pub(crate) struct ExposurePassPair<H, A> {
    pub histogram: Option<H>,
    pub average: Option<A>,
}

impl<H, A> ExposurePassPair<H, A> {
    /// Histogram accumulation is safe only if the resolve can consume and
    /// clear it in this same frame. Both passes use this identical gate.
    fn ready(self, enabled: bool, extent: [u32; 2]) -> Option<(H, A)> {
        if !enabled || extent[0] == 0 || extent[1] == 0 {
            return None;
        }
        self.histogram.zip(self.average)
    }

    pub fn histogram(self, enabled: bool, extent: [u32; 2]) -> Option<H> {
        self.ready(enabled, extent).map(|(histogram, _)| histogram)
    }

    pub fn average(self, enabled: bool, extent: [u32; 2]) -> Option<A> {
        self.ready(enabled, extent).map(|(_, average)| average)
    }
}

/// Immutable time sample shared by every view in one render frame. Reading it
/// cannot advance the clock, regardless of view order or recording duration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExposureFrameSnapshot {
    delta_seconds: f32,
}

impl ExposureFrameSnapshot {
    pub fn delta_seconds(self) -> f32 {
        self.delta_seconds
    }
}

#[derive(Default)]
pub(crate) struct ExposureFrameClock {
    last: Option<Instant>,
}

impl ExposureFrameClock {
    /// Called once by PrepareResources, even with no views, disabled exposure,
    /// or pending pipelines. GPU readiness never determines the clock step.
    pub fn advance(&mut self, now: Instant) -> ExposureFrameSnapshot {
        let delta_seconds = match self.last.replace(now) {
            Some(previous) => now
                .saturating_duration_since(previous)
                .as_secs_f32()
                .clamp(0.0, 1.0),
            None => 1.0 / 60.0,
        };
        ExposureFrameSnapshot { delta_seconds }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn pair(mask: u8) -> ExposurePassPair<((), ()), ((), ())> {
        ExposurePassPair {
            histogram: (mask & 1 != 0)
                .then_some(())
                .zip((mask & 2 != 0).then_some(())),
            average: (mask & 4 != 0)
                .then_some(())
                .zip((mask & 8 != 0).then_some(())),
        }
    }

    #[test]
    fn both_passes_require_both_pipelines_and_both_view_bind_groups() {
        for mask in 0..16 {
            let expected = mask == 15;
            assert_eq!(
                pair(mask).histogram(true, [32, 32]).is_some(),
                expected,
                "histogram mask {mask}"
            );
            assert_eq!(
                pair(mask).average(true, [32, 32]).is_some(),
                expected,
                "average mask {mask}"
            );
        }
    }

    #[test]
    fn two_views_in_one_frame_use_the_same_delta() {
        let start = Instant::now();
        let mut clock = ExposureFrameClock::default();
        clock.advance(start);
        let frame = start + Duration::from_millis(16);
        let snapshot = clock.advance(frame);
        let first_view = snapshot.delta_seconds();
        let second_view = snapshot.delta_seconds();
        assert_eq!(first_view, 0.016);
        assert_eq!(first_view, second_view);
    }

    #[test]
    fn disabled_and_empty_views_admit_neither_pass() {
        for (enabled, extent) in [
            (false, [32, 32]),
            (true, [0, 32]),
            (true, [32, 0]),
            (true, [0, 0]),
        ] {
            assert!(pair(15).histogram(enabled, extent).is_none());
            assert!(pair(15).average(enabled, extent).is_none());
        }
    }

    #[test]
    fn pipeline_warmup_admits_no_unpaired_histogram() {
        // Histogram warms up first, then resolve; either bind group can also
        // disappear independently. The admission decisions must stay paired.
        for mask in [0, 3, 7, 15, 11, 15, 12, 15] {
            let histogram = pair(mask).histogram(true, [32, 32]);
            let average = pair(mask).average(true, [32, 32]);
            assert_eq!(histogram.is_some(), average.is_some(), "mask {mask}");
        }
    }

    #[test]
    fn frame_clock_advances_even_when_no_view_dispatches() {
        let start = Instant::now();
        let mut clock = ExposureFrameClock::default();
        assert_eq!(clock.advance(start).delta_seconds(), 1.0 / 60.0);
        for frame in 1..=120 {
            // No view consumes these snapshots (disabled, absent, or waiting
            // for compilation); PrepareResources still publishes every frame.
            clock.advance(start + Duration::from_millis(frame * 16));
        }
        let resumed = clock.advance(start + Duration::from_millis(121 * 16));
        assert_eq!(resumed.delta_seconds(), 0.016);
    }

    #[test]
    fn snapshot_stays_frozen_when_next_frame_advances() {
        let start = Instant::now();
        let mut clock = ExposureFrameClock::default();
        clock.advance(start);
        let first = clock.advance(start + Duration::from_millis(16));
        let second = clock.advance(start + Duration::from_millis(48));
        assert_eq!(first.delta_seconds(), 0.016);
        assert_eq!(second.delta_seconds(), 0.032);
    }

    #[test]
    fn clock_clamps_long_or_backwards_steps() {
        let start = Instant::now();
        let mut clock = ExposureFrameClock::default();
        clock.advance(start);
        assert_eq!(
            clock
                .advance(start + Duration::from_secs(3))
                .delta_seconds(),
            1.0
        );
        assert_eq!(clock.advance(start).delta_seconds(), 0.0);
    }
}
