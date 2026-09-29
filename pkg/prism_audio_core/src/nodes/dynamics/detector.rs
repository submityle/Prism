//! Shared building blocks for the dynamics family: level detection, attack /
//! release ballistics, and the static gain-computer curves that compressors,
//! limiters, gates, and duckers all share.
//!
//! Keeping these primitives in one place means every dynamics processor uses
//! the *same*, well-tested envelope math, so their sound is consistent and the
//! per-node code stays focused on routing and mixing.

use bevy_math::ops;

use crate::math::{Sample, linear_to_db};

/// How the side-chain signal level is measured before the gain computer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DetectionMode {
    /// Instantaneous absolute peak — fast, transient-accurate, used by
    /// limiters and percussive compression.
    Peak,
    /// Root-mean-square power average — smoother and closer to perceived
    /// loudness, used for musical bus compression.
    Rms,
}

/// Converts a time constant in milliseconds to a one-pole smoothing
/// coefficient at `sample_rate`.
///
/// The coefficient `a = exp(-1 / (t * sr))` is the classic analog-style time
/// constant: after `t` seconds a step response reaches `1 - 1/e` (~63 %) of the
/// way to its target. A zero (or negative) time yields `0.0`, i.e. an
/// instantaneous response.
#[inline]
#[must_use]
pub fn time_to_coef(time_ms: Sample, sample_rate: u32) -> Sample {
    let t = time_ms.max(0.0) * 0.001;
    if t <= 0.0 {
        0.0
    } else {
        let sr = (sample_rate.max(1)) as Sample;
        ops::exp(-1.0 / (t * sr))
    }
}

/// A running level detector producing a signal level in decibels.
///
/// In [`DetectionMode::Peak`] it reports the instantaneous rectified level; in
/// [`DetectionMode::Rms`] it maintains a one-pole average of the squared
/// signal so the reported level tracks power rather than individual samples.
#[derive(Debug, Clone, Copy)]
pub struct LevelDetector {
    /// Detection mode (peak or RMS).
    mode: DetectionMode,
    /// One-pole coefficient for the RMS power average.
    rms_coef: Sample,
    /// Running mean-square estimate (only used in RMS mode).
    mean_square: Sample,
}

impl LevelDetector {
    /// Builds a detector. `rms_window_ms` sets the averaging window used in
    /// [`DetectionMode::Rms`] (ignored for peak detection).
    #[must_use]
    pub fn new(mode: DetectionMode, rms_window_ms: Sample, sample_rate: u32) -> Self {
        Self {
            mode,
            rms_coef: time_to_coef(rms_window_ms, sample_rate),
            mean_square: 0.0,
        }
    }

    /// Feeds one (already channel-combined) linear sample and returns the
    /// current level in decibels.
    #[inline]
    pub fn level_db(&mut self, x: Sample) -> Sample {
        match self.mode {
            DetectionMode::Peak => linear_to_db(x.abs()),
            DetectionMode::Rms => {
                let sq = x * x;
                self.mean_square = self.rms_coef * self.mean_square + (1.0 - self.rms_coef) * sq;
                // 10*log10(power) == 20*log10(amplitude); linear_to_db expects
                // amplitude, so feed sqrt of the mean square.
                linear_to_db(ops::sqrt(self.mean_square))
            }
        }
    }

    /// Clears the internal power average.
    #[inline]
    pub fn reset(&mut self) {
        self.mean_square = 0.0;
    }
}

/// A "smooth decoupled peak detector" applied to a gain-reduction control
/// signal (Giannoulis, Massberg & Reiss, 2012).
///
/// It is fed the *desired* gain reduction in positive decibels (0 = no
/// reduction, larger = quieter) and returns a time-smoothed reduction so the
/// gain moves quickly when clamping down (attack) and eases back gently when
/// releasing. The decoupled topology avoids the ripple that a naive branching
/// one-pole produces on sustained material.
#[derive(Debug, Clone, Copy)]
pub struct GainBallistics {
    /// Attack coefficient (smaller = faster clamp-down).
    attack_coef: Sample,
    /// Release coefficient (larger = slower recovery).
    release_coef: Sample,
    /// First (release) smoothing stage state.
    y1: Sample,
    /// Second (attack) smoothing stage state — the applied reduction in dB.
    y: Sample,
}

impl GainBallistics {
    /// Builds ballistics from attack / release times in milliseconds.
    #[must_use]
    pub fn new(attack_ms: Sample, release_ms: Sample, sample_rate: u32) -> Self {
        Self {
            attack_coef: time_to_coef(attack_ms, sample_rate),
            release_coef: time_to_coef(release_ms, sample_rate),
            y1: 0.0,
            y: 0.0,
        }
    }

    /// Updates the attack / release times in place (state preserved).
    #[inline]
    pub fn set_times(&mut self, attack_ms: Sample, release_ms: Sample, sample_rate: u32) {
        self.attack_coef = time_to_coef(attack_ms, sample_rate);
        self.release_coef = time_to_coef(release_ms, sample_rate);
    }

    /// Advances one sample with target reduction `control_db` (>= 0) and
    /// returns the smoothed reduction in decibels (>= 0).
    #[inline]
    pub fn process(&mut self, control_db: Sample) -> Sample {
        let c = control_db.max(0.0);
        // Release stage: peak-hold that decays towards the control.
        self.y1 = c.max(self.release_coef * self.y1 + (1.0 - self.release_coef) * c);
        // Attack stage: smooth approach to the held peak.
        self.y = self.attack_coef * self.y + (1.0 - self.attack_coef) * self.y1;
        self.y
    }

    /// Returns the current smoothed reduction in decibels.
    #[inline]
    #[must_use]
    pub fn current_db(&self) -> Sample {
        self.y
    }

    /// Clears both smoothing stages.
    #[inline]
    pub fn reset(&mut self) {
        self.y1 = 0.0;
        self.y = 0.0;
    }
}

/// Static downward-compression curve.
///
/// Given the side-chain `level_db`, the `threshold_db`, a compression `ratio`
/// (`>= 1`), and a soft `knee_db` width, returns the gain **reduction** to
/// apply as a non-negative number of decibels (0 = below threshold, larger =
/// quieter). This is the standard soft-knee compression curve used throughout the digital audio effects literature (Reiss, 2014).
#[inline]
#[must_use]
pub fn compressor_reduction_db(
    level_db: Sample,
    threshold_db: Sample,
    ratio: Sample,
    knee_db: Sample,
) -> Sample {
    let ratio = ratio.max(1.0);
    let slope = 1.0 - 1.0 / ratio;
    let over = level_db - threshold_db;
    let knee = knee_db.max(0.0);

    let output_over = if knee > 0.0 && 2.0 * over > -knee && 2.0 * over < knee {
        // Quadratic soft-knee interpolation region.
        let t = over + knee * 0.5;
        over - slope * (t * t) / (2.0 * knee)
    } else if 2.0 * over <= -knee {
        // Fully below the knee: no compression.
        over
    } else {
        // Fully above the knee: linear ratio.
        over - slope * over
    };

    // Reduction is how much we pulled the level down (>= 0).
    (over - output_over).max(0.0)
}

/// Static downward-expansion / gate curve.
///
/// Given the side-chain `level_db` and a downward-expansion `ratio` (`>= 1`),
/// returns the gain **reduction** in non-negative decibels applied *below* the
/// threshold, limited to at most `range_db`. Above the threshold (plus soft
/// knee) the signal passes untouched.
#[inline]
#[must_use]
pub fn expander_reduction_db(
    level_db: Sample,
    threshold_db: Sample,
    ratio: Sample,
    knee_db: Sample,
    range_db: Sample,
) -> Sample {
    let ratio = ratio.max(1.0);
    let range = range_db.max(0.0);
    let knee = knee_db.max(0.0);
    // Distance below threshold (positive when quieter than the threshold).
    let under = threshold_db - level_db;

    let reduction = if under <= -knee * 0.5 {
        // Comfortably above threshold: open, no reduction.
        0.0
    } else if knee > 0.0 && under < knee * 0.5 {
        // Soft-knee interpolation around the threshold.
        let t = under + knee * 0.5;
        (ratio - 1.0) * (t * t) / (2.0 * knee)
    } else {
        // Fully below threshold: linear downward expansion.
        (ratio - 1.0) * under
    };

    reduction.clamp(0.0, range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coef_zero_time_is_instant() {
        assert_eq!(time_to_coef(0.0, 48_000), 0.0);
    }

    #[test]
    fn compressor_below_threshold_is_transparent() {
        let gr = compressor_reduction_db(-30.0, -20.0, 4.0, 0.0);
        assert!(gr.abs() < 1e-6, "{gr}");
    }

    #[test]
    fn compressor_above_threshold_reduces_by_ratio() {
        // 10 dB over threshold at 4:1 hard knee -> output 2.5 dB over ->
        // reduction 7.5 dB.
        let gr = compressor_reduction_db(-10.0, -20.0, 4.0, 0.0);
        assert!((gr - 7.5).abs() < 1e-4, "{gr}");
    }

    #[test]
    fn compressor_soft_knee_is_continuous() {
        // Just inside the knee the reduction should be small and positive.
        let gr = compressor_reduction_db(-20.0, -20.0, 4.0, 6.0);
        assert!(gr > 0.0 && gr < 1.0, "{gr}");
    }

    #[test]
    fn expander_above_threshold_is_open() {
        let gr = expander_reduction_db(-10.0, -40.0, 2.0, 0.0, 60.0);
        assert!(gr.abs() < 1e-6, "{gr}");
    }

    #[test]
    fn expander_below_threshold_reduces_and_clamps() {
        // 20 dB under at 2:1 -> 20 dB reduction, clamped to range 12.
        let gr = expander_reduction_db(-60.0, -40.0, 2.0, 0.0, 12.0);
        assert!((gr - 12.0).abs() < 1e-4, "{gr}");
    }

    #[test]
    fn ballistics_attack_then_release() {
        let mut b = GainBallistics::new(1.0, 100.0, 48_000);
        // Drive with a step of 6 dB reduction.
        for _ in 0..4_800 {
            b.process(6.0);
        }
        let attacked = b.current_db();
        assert!(attacked > 5.0, "attack did not reach target: {attacked}");
        // Release back toward zero.
        for _ in 0..48_000 {
            b.process(0.0);
        }
        assert!(b.current_db() < 1.0, "release did not recover: {}", b.current_db());
    }

    #[test]
    fn rms_detector_reports_power() {
        let mut d = LevelDetector::new(DetectionMode::Rms, 1.0, 48_000);
        let mut last = f32::NEG_INFINITY;
        for _ in 0..48_000 {
            last = d.level_db(0.5);
        }
        // Steady 0.5 amplitude -> about -6 dBFS.
        assert!((last + 6.0206).abs() < 0.5, "{last}");
    }
}
