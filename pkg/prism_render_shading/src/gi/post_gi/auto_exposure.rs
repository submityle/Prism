//! Histogram-based automatic exposure — backend-neutral CPU golden.
//!
//! Automatic exposure measures how bright the resolved, pre-exposed linear HDR
//! scene is and chooses the single scalar multiplier the shading pass applies
//! before display tone mapping (see [`super`] for the pipeline tail). The
//! classic AAA technique (UE's `AutoExposureHistogram`, Frostbite's physically
//! based exposure) bins the *log* luminance of the frame, rejects the darkest
//! and brightest tails by percentile, and averages what remains. That average
//! log luminance is converted to an EV100 exposure value, clamped to the
//! artist's range, and turned into a linear exposure multiplier. Finally the
//! exposure is eased toward its target across frames (eye adaptation) with
//! separate brightening / darkening speeds, because a real eye adapts to bright
//! light faster than it recovers in the dark.
//!
//! Two complementary conversions are offered:
//!
//! * **Metering (EV100).** `EV100 = log2(L * S / K)` for average luminance `L`,
//!   sensitivity `S = 100` and reflected-light calibration `K = 12.5`. The
//!   linear multiplier is `1 / (q_factor * 2^EV100)`, so a surface at the
//!   sensor's max luminance maps to `1.0`.
//! * **Key value.** The grey-world shortcut `exposure = key / L`, which scales
//!   the metered average straight onto a target middle grey (`0.18` by
//!   default), the way many engines drive their simplest auto-exposure.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Luminances and exposures are non-negative; log luminance is base-2.
//! * Defensive clamping everywhere: non-positive luminance is floored to a tiny
//!   epsilon so logs stay finite, empty / collapsed histograms fall back to a
//!   defined value, and no path can emit `NaN` or `inf`.
//! * `f32` storage mirrors the layout the WESL/GPU twin consumes.

use alloc::vec::Vec;
use bevy_math::ops;

/// Rec. 709 luminance weights (linear sRGB primaries).
pub const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Reflected-light meter calibration constant `K` (Canon/Nikon 12.5).
pub const METER_CALIBRATION_K: f32 = 12.5;

/// Standard lens/sensor factor folding `q = 0.65` into `Lmax = 1.2 * 2^EV100`.
pub const MAX_LUMINANCE_FACTOR: f32 = 1.2;

/// Default photographic middle grey (18 % reflectance) used as the key value.
pub const DEFAULT_KEY_VALUE: f32 = 0.18;

/// Smallest luminance fed to a logarithm, so `log2` of a black pixel stays
/// finite instead of collapsing to `-inf`.
pub const LUMINANCE_EPSILON: f32 = 1.0e-6;

/// Rec. 709 relative luminance of a linear RGB radiance sample, floored to
/// zero so a stray negative channel never yields a negative luminance.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    let l = rgb[0] * LUMINANCE_WEIGHTS[0]
        + rgb[1] * LUMINANCE_WEIGHTS[1]
        + rgb[2] * LUMINANCE_WEIGHTS[2];
    l.max(0.0)
}

/// The log-luminance window a histogram covers, expressed as base-2 exponents.
///
/// Bin `0` starts at `2^min_log2` and the last bin ends at `2^max_log2`. The
/// default `[-10, 10]` spans roughly a starlit floor to a bright overcast sky,
/// matching the range AAA auto-exposure histograms use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogLuminanceRange {
    /// Base-2 log of the darkest luminance the histogram resolves.
    pub min_log2: f32,
    /// Base-2 log of the brightest luminance the histogram resolves.
    pub max_log2: f32,
}

impl Default for LogLuminanceRange {
    fn default() -> Self {
        Self {
            min_log2: -10.0,
            max_log2: 10.0,
        }
    }
}

impl LogLuminanceRange {
    /// Width of the window in stops (`max_log2 - min_log2`), floored to a tiny
    /// positive value so bin mapping never divides by zero for a collapsed
    /// range.
    #[must_use]
    pub fn span(&self) -> f32 {
        (self.max_log2 - self.min_log2).max(LUMINANCE_EPSILON)
    }

    /// Maps a linear luminance to its histogram bin in `[0, bin_count)`.
    ///
    /// The luminance is floored to [`LUMINANCE_EPSILON`], converted to log2,
    /// placed on the normalised `[0, 1]` window, scaled by `bin_count`, floored
    /// and clamped. A `bin_count` of `0` degrades to bin `0`.
    #[must_use]
    pub fn luminance_to_bin(&self, luminance: f32, bin_count: u32) -> u32 {
        if bin_count == 0 {
            return 0;
        }
        let log2_l = ops::log2(luminance.max(LUMINANCE_EPSILON));
        let t = ((log2_l - self.min_log2) / self.span()).clamp(0.0, 1.0);
        let idx = ops::floor(t * bin_count as f32) as i32;
        idx.clamp(0, bin_count as i32 - 1) as u32
    }

    /// Base-2 log luminance at the *center* of `bin` for an `bin_count`-wide
    /// histogram. The inverse of [`luminance_to_bin`](Self::luminance_to_bin)
    /// up to within half a bin.
    #[must_use]
    pub fn bin_center_log2(&self, bin: u32, bin_count: u32) -> f32 {
        if bin_count == 0 {
            return self.min_log2;
        }
        let frac = (bin as f32 + 0.5) / bin_count as f32;
        self.min_log2 + frac * self.span()
    }

    /// Linear luminance at the center of `bin`: `2^bin_center_log2`.
    #[must_use]
    pub fn bin_center_luminance(&self, bin: u32, bin_count: u32) -> f32 {
        ops::exp2(self.bin_center_log2(bin, bin_count))
    }
}

/// Builds a log-luminance histogram over `samples`, returning `bin_count`
/// counters. Each sample's Rec. 709 luminance is binned via
/// [`LogLuminanceRange::luminance_to_bin`]. Returns an empty `Vec` when
/// `bin_count` is `0`.
#[must_use]
pub fn build_histogram(
    samples: &[[f32; 3]],
    range: LogLuminanceRange,
    bin_count: u32,
) -> Vec<u32> {
    let mut bins = alloc::vec![0_u32; bin_count as usize];
    if bin_count == 0 {
        return bins;
    }
    for &rgb in samples {
        let bin = range.luminance_to_bin(luminance(rgb), bin_count);
        bins[bin as usize] = bins[bin as usize].saturating_add(1);
    }
    bins
}

/// Low / high percentile cut applied to a histogram before averaging. Both are
/// fractions in `[0, 1)` of the *total* population; the darkest `low` fraction
/// and brightest `high` fraction are discarded so fireflies and black borders
/// do not drag the metered average.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramClip {
    /// Fraction of the darkest samples to reject (e.g. `0.5` = bottom half).
    pub low: f32,
    /// Fraction of the brightest samples to reject (e.g. `0.2` = top fifth).
    pub high: f32,
}

impl Default for HistogramClip {
    fn default() -> Self {
        // UE-like defaults: trim the darkest half and brightest fifth.
        Self {
            low: 0.5,
            high: 0.2,
        }
    }
}

/// Average *log2* luminance of a histogram after discarding the dark/bright
/// tails named by `clip`.
///
/// The surviving counts are weighted by each bin's center log2 luminance. An
/// empty histogram returns `range.min_log2` (the darkest resolvable value). If
/// the clip window collapses (`low + high >= 1`, or rounding leaves it empty)
/// the function falls back to the *untrimmed* count-weighted mean rather than
/// returning a meaningless value.
#[must_use]
pub fn average_log_luminance(
    bins: &[u32],
    range: LogLuminanceRange,
    clip: HistogramClip,
) -> f32 {
    let bin_count = bins.len() as u32;
    if bin_count == 0 {
        return range.min_log2;
    }
    let total: u64 = bins.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return range.min_log2;
    }

    let low = clip.low.clamp(0.0, 1.0);
    let high = clip.high.clamp(0.0, 1.0);
    let lo_cut = (low * total as f32) as u64;
    let hi_cut = (high * total as f32) as u64;

    // Walk the cumulative population, keeping only the window (lo_cut, total - hi_cut].
    let hi_keep = total.saturating_sub(hi_cut);
    let mut cumulative: u64 = 0;
    let mut weighted = 0.0_f64;
    let mut kept = 0.0_f64;
    for (bin, &count) in bins.iter().enumerate() {
        let count = count as u64;
        if count == 0 {
            continue;
        }
        let bin_start = cumulative;
        let bin_end = cumulative + count;
        cumulative = bin_end;
        // Overlap of [bin_start, bin_end) with the kept window (lo_cut, hi_keep].
        let keep_lo = bin_start.max(lo_cut);
        let keep_hi = bin_end.min(hi_keep);
        if keep_hi > keep_lo {
            let w = (keep_hi - keep_lo) as f64;
            let center = range.bin_center_log2(bin as u32, bin_count) as f64;
            weighted += center * w;
            kept += w;
        }
    }

    if kept > 0.0 {
        return (weighted / kept) as f32;
    }

    // Collapsed window: fall back to the untrimmed count-weighted mean.
    let mut weighted = 0.0_f64;
    for (bin, &count) in bins.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let center = range.bin_center_log2(bin as u32, bin_count) as f64;
        weighted += center * count as f64;
    }
    (weighted / total as f64) as f32
}

/// Average *linear* luminance of a clipped histogram: `2^average_log_luminance`.
#[must_use]
pub fn average_luminance(
    bins: &[u32],
    range: LogLuminanceRange,
    clip: HistogramClip,
) -> f32 {
    ops::exp2(average_log_luminance(bins, range, clip))
}

/// EV100 implied by an average luminance: `log2(L * 100 / K)`.
///
/// Non-positive luminance is floored to [`LUMINANCE_EPSILON`] so the log stays
/// finite.
#[must_use]
pub fn ev100_from_luminance(average_luminance: f32) -> f32 {
    let l = average_luminance.max(LUMINANCE_EPSILON);
    ops::log2(l * 100.0 / METER_CALIBRATION_K)
}

/// Linear exposure multiplier for an EV100: `1 / (1.2 * 2^EV100)`.
#[must_use]
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    let max_luminance = MAX_LUMINANCE_FACTOR * ops::exp2(ev100);
    (1.0 / max_luminance.max(LUMINANCE_EPSILON)).max(0.0)
}

/// Inverse of [`exposure_from_ev100`]: the EV100 that produces `exposure`.
///
/// Non-positive exposures floor to [`LUMINANCE_EPSILON`] so the log is finite.
#[must_use]
pub fn ev100_from_exposure(exposure: f32) -> f32 {
    let e = exposure.max(LUMINANCE_EPSILON);
    // exposure = 1 / (1.2 * 2^EV) => 2^EV = 1 / (1.2 * exposure).
    ops::log2(1.0 / (MAX_LUMINANCE_FACTOR * e))
}

/// Grey-world exposure that scales `average_luminance` onto `key_value`:
/// `exposure = key / L`.
///
/// Both arguments are floored to [`LUMINANCE_EPSILON`] (key) and a tiny epsilon
/// (luminance) so the ratio is finite and non-negative. This is the simplest
/// auto-exposure response and is handy as a reference against the EV100 path.
#[must_use]
pub fn exposure_from_key_value(average_luminance: f32, key_value: f32) -> f32 {
    let key = key_value.max(0.0);
    let l = average_luminance.max(LUMINANCE_EPSILON);
    (key / l).max(0.0)
}

/// Artist controls for the auto-exposure resolve.
///
/// `Default` meters a scene, clamps EV100 to `[-8, 16]` (a very dark room to a
/// sunlit exterior), applies no compensation and targets 18 % middle grey for
/// the key-value path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoExposureSettings {
    /// Lower EV100 clamp (brightest allowed exposure).
    pub min_ev100: f32,
    /// Upper EV100 clamp (darkest allowed exposure).
    pub max_ev100: f32,
    /// Exposure compensation in stops, added to the metered EV100.
    pub compensation_stops: f32,
    /// Target middle grey for [`Self::exposure_by_key_value`].
    pub key_value: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            min_ev100: -8.0,
            max_ev100: 16.0,
            compensation_stops: 0.0,
            key_value: DEFAULT_KEY_VALUE,
        }
    }
}

impl AutoExposureSettings {
    /// EV100 the settings resolve for `average_luminance`: metered, offset by
    /// `compensation_stops`, then clamped into `[min_ev100, max_ev100]`.
    ///
    /// An inverted range (`min > max`) is treated symmetrically by clamping to
    /// the ordered pair so the result is always finite.
    #[must_use]
    pub fn resolve_ev100(&self, average_luminance: f32) -> f32 {
        let ev = ev100_from_luminance(average_luminance) + self.compensation_stops;
        let lo = self.min_ev100.min(self.max_ev100);
        let hi = self.min_ev100.max(self.max_ev100);
        ev.clamp(lo, hi)
    }

    /// Linear exposure multiplier for `average_luminance` via the EV100 path.
    #[must_use]
    pub fn exposure(&self, average_luminance: f32) -> f32 {
        exposure_from_ev100(self.resolve_ev100(average_luminance))
    }

    /// Linear exposure multiplier for `average_luminance` via the key-value
    /// (grey-world) path, with `compensation_stops` applied as a `2^stops`
    /// gain and the result clamped to the EV100 range for consistency.
    #[must_use]
    pub fn exposure_by_key_value(&self, average_luminance: f32) -> f32 {
        let base = exposure_from_key_value(average_luminance, self.key_value);
        let compensated = base * ops::exp2(self.compensation_stops);
        // Clamp to the exposures the EV range permits so both paths agree on bounds.
        let lo = exposure_from_ev100(self.max_ev100.max(self.min_ev100));
        let hi = exposure_from_ev100(self.min_ev100.min(self.max_ev100));
        compensated.clamp(lo, hi)
    }
}

/// Separate brightening / darkening adaptation speeds (per second). A real eye
/// adapts to a brighter target faster than it recovers in the dark, so
/// `speed_up` is typically larger than `speed_down`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeAdaptation {
    /// Response rate when the target exposure state is *larger* than current.
    pub speed_up: f32,
    /// Response rate when the target exposure state is *smaller* than current.
    pub speed_down: f32,
}

impl Default for EyeAdaptation {
    fn default() -> Self {
        Self {
            speed_up: 3.0,
            speed_down: 1.0,
        }
    }
}

impl EyeAdaptation {
    /// Exponentially eases `current` toward `target` over `delta_seconds`.
    ///
    /// Uses the time-correct response `current + (target - current) * (1 -
    /// exp(-speed * dt))`, which is frame-rate independent and never
    /// overshoots. `delta_seconds <= 0` is a no-op; a non-positive speed holds
    /// `current` (no adaptation). Values are left in whatever space the caller
    /// adapts (linear exposure or EV100); the maths is identical.
    #[must_use]
    pub fn adapt(&self, current: f32, target: f32, delta_seconds: f32) -> f32 {
        if delta_seconds <= 0.0 || !current.is_finite() || !target.is_finite() {
            return current;
        }
        let speed = if target >= current {
            self.speed_up
        } else {
            self.speed_down
        }
        .max(0.0);
        if speed == 0.0 {
            return current;
        }
        let blend = 1.0 - ops::exp(-speed * delta_seconds);
        current + (target - current) * blend.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    #[test]
    fn luminance_matches_rec709_and_floors_negatives() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([-5.0, -5.0, -5.0]), 0.0);
    }

    #[test]
    fn bin_mapping_is_monotonic_and_clamped() {
        let range = LogLuminanceRange::default();
        let count = 64;
        assert_eq!(range.luminance_to_bin(0.0, count), 0);
        assert_eq!(range.luminance_to_bin(1.0e18, count), count - 1);
        let dark = range.luminance_to_bin(0.01, count);
        let bright = range.luminance_to_bin(100.0, count);
        assert!(bright > dark);
    }

    #[test]
    fn bin_center_round_trips() {
        let range = LogLuminanceRange::default();
        let count = 64;
        for bin in [0_u32, 1, 17, 63] {
            let l = range.bin_center_luminance(bin, count);
            assert_eq!(range.luminance_to_bin(l, count), bin);
        }
    }

    #[test]
    fn full_bucket_meters_that_luminance() {
        // A histogram that is one fully-populated bin should meter that bin's
        // luminance regardless of (symmetric) trimming.
        let range = LogLuminanceRange::default();
        let count = 32_u32;
        let mut bins = alloc::vec![0_u32; count as usize];
        bins[20] = 5000;
        let clip = HistogramClip { low: 0.1, high: 0.1 };
        approx(
            average_luminance(&bins, range, clip),
            range.bin_center_luminance(20, count),
        );
    }

    #[test]
    fn empty_histogram_falls_back_to_min() {
        let bins = [0_u32; 16];
        let range = LogLuminanceRange::default();
        approx(
            average_log_luminance(&bins, range, HistogramClip::default()),
            range.min_log2,
        );
    }

    #[test]
    fn percentile_trim_rejects_dark_and_bright_tails() {
        let range = LogLuminanceRange::default();
        let count = 8_u32;
        let mut bins = alloc::vec![0_u32; count as usize];
        bins[0] = 100;
        bins[4] = 100;
        bins[7] = 100;
        // Discard the darkest and brightest thirds -> only the middle bin.
        let clip = HistogramClip { low: 0.34, high: 0.34 };
        approx(
            average_luminance(&bins, range, clip),
            range.bin_center_luminance(4, count),
        );
    }

    #[test]
    fn collapsed_window_uses_untrimmed_mean() {
        let range = LogLuminanceRange::default();
        let count = 4_u32;
        let mut bins = alloc::vec![0_u32; count as usize];
        bins[1] = 10;
        bins[2] = 10;
        let clip = HistogramClip { low: 0.6, high: 0.6 };
        let got = average_log_luminance(&bins, range, clip);
        let expect = (range.bin_center_log2(1, count) + range.bin_center_log2(2, count)) / 2.0;
        approx(got, expect);
    }

    #[test]
    fn build_histogram_counts_samples() {
        let range = LogLuminanceRange::default();
        let count = 16_u32;
        let samples = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.001, 0.001, 0.001]];
        let bins = build_histogram(&samples, range, count);
        let total: u32 = bins.iter().sum();
        assert_eq!(total, 3);
        let bright = range.luminance_to_bin(1.0, count);
        assert_eq!(bins[bright as usize], 2);
    }

    #[test]
    fn build_histogram_zero_bins_is_empty() {
        assert!(build_histogram(&[[1.0, 1.0, 1.0]], LogLuminanceRange::default(), 0).is_empty());
    }

    #[test]
    fn ev100_exposure_round_trips() {
        for ev in [-6.0_f32, -1.0, 0.0, 3.5, 12.0] {
            let e = exposure_from_ev100(ev);
            approx(ev100_from_exposure(e), ev);
        }
    }

    #[test]
    fn ev100_meters_middle_grey() {
        // 0.18 at K=12.5, S=100 -> EV100 ~= log2(0.18*8) = log2(1.44).
        let ev = ev100_from_luminance(0.18);
        approx(ev, ops::log2(0.18 * 100.0 / 12.5));
    }

    #[test]
    fn key_value_scales_to_target() {
        // exposure * L == key for the raw key-value response.
        let l = 0.5;
        let key = 0.18;
        approx(exposure_from_key_value(l, key) * l, key);
    }

    #[test]
    fn key_value_handles_black() {
        let e = exposure_from_key_value(0.0, 0.18);
        assert!(e.is_finite() && e >= 0.0);
    }

    #[test]
    fn settings_clamp_ev_range() {
        let s = AutoExposureSettings {
            min_ev100: 4.0,
            max_ev100: 8.0,
            compensation_stops: 0.0,
            key_value: DEFAULT_KEY_VALUE,
        };
        let bright = METER_CALIBRATION_K / 100.0 * ops::exp2(20.0);
        approx(s.resolve_ev100(bright), 8.0);
        let dark = METER_CALIBRATION_K / 100.0 * ops::exp2(-10.0);
        approx(s.resolve_ev100(dark), 4.0);
    }

    #[test]
    fn compensation_shifts_metered_ev() {
        let base = AutoExposureSettings::default();
        let comp = AutoExposureSettings {
            compensation_stops: 2.0,
            ..AutoExposureSettings::default()
        };
        let l = METER_CALIBRATION_K / 100.0 * ops::exp2(6.0);
        approx(base.resolve_ev100(l), 6.0);
        approx(comp.resolve_ev100(l), 8.0);
    }

    #[test]
    fn exposure_decreases_with_brightness() {
        let s = AutoExposureSettings::default();
        let dim = s.exposure(0.05);
        let bright = s.exposure(5.0);
        assert!(dim > bright);
        assert!(bright > 0.0);
    }

    #[test]
    fn adaptation_eases_toward_target_without_overshoot() {
        let eye = EyeAdaptation::default();
        let next = eye.adapt(1.0, 2.0, 0.1);
        assert!(next > 1.0 && next < 2.0);
        approx(eye.adapt(1.0, 2.0, 0.0), 1.0);
    }

    #[test]
    fn adaptation_respects_asymmetric_speeds_and_converges() {
        let eye = EyeAdaptation {
            speed_up: 4.0,
            speed_down: 1.0,
        };
        let up = eye.adapt(1.0, 2.0, 0.25) - 1.0;
        let down = 2.0 - eye.adapt(2.0, 1.0, 0.25);
        assert!(up > down);
        let mut state = 1.0_f32;
        for _ in 0..400 {
            state = eye.adapt(state, 5.0, 0.05);
        }
        approx(state, 5.0);
    }

    #[test]
    fn adaptation_guards_non_finite_inputs() {
        let eye = EyeAdaptation::default();
        // A non-finite target is a documented no-op: current is returned.
        approx(eye.adapt(1.0, f32::INFINITY, 0.1), 1.0);
        approx(eye.adapt(2.0, f32::NAN, 0.1), 2.0);
        // Zero/negative speed holds the current value.
        let frozen = EyeAdaptation { speed_up: 0.0, speed_down: 0.0 };
        approx(frozen.adapt(1.0, 9.0, 1.0), 1.0);
    }
}
