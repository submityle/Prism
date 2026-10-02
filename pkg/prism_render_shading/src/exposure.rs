//! Backend-neutral CPU golden for physically-based exposure and eye adaptation.
//!
//! The visibility -> classify -> shade -> resolve chain writes *pre-exposed*
//! linear HDR radiance (`resolve.rs` documents its illuminance as "already
//! pre-exposed"), but nothing in the crate computed that exposure multiplier.
//! This module is that missing source: it turns either a physical camera
//! (aperture / shutter / ISO) or a measured scene luminance into the single
//! scalar the shading pass multiplies radiance by before it lands in the HDR
//! buffer. Pre-exposure keeps HDR values in a well-conditioned float range and
//! is what drives temporal *eye adaptation*; it is distinct from any display
//! tone-map curve applied later.
//!
//! The model follows Lagarde & de Rousiers, "Moving Frostbite to Physically
//! Based Rendering" (the same maths UE's auto-exposure uses):
//!
//! * **EV100** is the exposure value normalised to ISO 100. A physical camera
//!   gives `EV100 = log2(N^2 / t) - log2(S / 100)` for aperture `N` (f-stops),
//!   shutter time `t` (seconds) and sensitivity `S` (ISO). A light meter gives
//!   `EV100 = log2(L * 100 / K)` for average luminance `L` and reflected-light
//!   calibration `K` (12.5, the Canon/Nikon constant).
//! * The **max sensor luminance** for a given EV100 is `Lmax = 1.2 * 2^EV100`
//!   (the `1.2` folds in the standard lens/vignette factor `q = 0.65` via
//!   `Lmax = 78 / (S * q) * 2^EV100` at `S = 100`). The **exposure multiplier**
//!   is `1 / Lmax`, so a surface at `Lmax` maps to `1.0`.
//! * **Auto-exposure** measures average luminance from a log-luminance
//!   histogram (rejecting the darkest/brightest tails by percentile, exactly
//!   like UE's `AutoExposureHistogram`), converts it to EV100, clamps to the
//!   artist's `[min, max]` range and applies exposure compensation in stops.
//! * **Eye adaptation** eases the exposed luminance toward its target with an
//!   exponential response, using separate brightening/darkening speeds so the
//!   eye adapts to bright light faster than it recovers in the dark.
//!
//! Everything is pure `f32` maths mirrored arm-for-arm by
//! `shaders/exposure.wesl`, so the CPU golden and the GPU twin agree.

use bevy_math::ops;

/// Reflected-light meter calibration constant `K` (Canon/Nikon 12.5). Relates a
/// measured average luminance to EV100 via `EV100 = log2(L * 100 / K)`.
pub const METER_CALIBRATION_K: f32 = 12.5;

/// Standard lens/sensor factor folding `q = 0.65` into the max-luminance
/// relation `Lmax = 1.2 * 2^EV100` (equivalently `78 / (100 * 0.65)`).
pub const MAX_LUMINANCE_FACTOR: f32 = 1.2;

/// Rec. 709 luminance weights (linear sRGB primaries).
pub const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Rec. 709 relative luminance of a linear RGB radiance sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMINANCE_WEIGHTS[0] + rgb[1] * LUMINANCE_WEIGHTS[1] + rgb[2] * LUMINANCE_WEIGHTS[2]
}

/// A physical camera whose triangle of settings fixes the exposure. Defaults to
/// the "sunny 16" reference (f/16, 1/125 s, ISO 100 -> EV100 ~= 15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalCamera {
    /// Relative aperture `N` in f-stops (e.g. `16.0` for f/16). Larger stops
    /// admit less light and raise EV100.
    pub aperture: f32,
    /// Shutter time `t` in seconds (e.g. `1.0 / 125.0`). Shorter exposures
    /// admit less light and raise EV100.
    pub shutter_time: f32,
    /// Sensor sensitivity `S` in ISO (e.g. `100.0`). Higher ISO lowers EV100.
    pub iso: f32,
}

impl Default for PhysicalCamera {
    fn default() -> Self {
        Self {
            aperture: 16.0,
            shutter_time: 1.0 / 125.0,
            iso: 100.0,
        }
    }
}

impl PhysicalCamera {
    /// Exposure value at ISO 100: `log2(N^2 / t) - log2(S / 100)`.
    ///
    /// Degenerate settings (non-positive aperture/shutter/ISO) fall back to the
    /// safe reference EV100 of `0.0` rather than producing a NaN/inf.
    #[must_use]
    pub fn ev100(&self) -> f32 {
        if self.aperture <= 0.0 || self.shutter_time <= 0.0 || self.iso <= 0.0 {
            return 0.0;
        }
        ops::log2(self.aperture * self.aperture / self.shutter_time) - ops::log2(self.iso / 100.0)
    }
}

/// EV100 implied by a measured average luminance: `log2(L * 100 / K)`.
///
/// Non-positive luminance clamps to a very dark floor so the log stays finite.
#[must_use]
pub fn ev100_from_average_luminance(average_luminance: f32) -> f32 {
    let l = average_luminance.max(1.0e-6);
    ops::log2(l * 100.0 / METER_CALIBRATION_K)
}

/// Maximum sensor luminance `Lmax = 1.2 * 2^EV100` for a given EV100.
#[must_use]
pub fn max_luminance_for_ev100(ev100: f32) -> f32 {
    MAX_LUMINANCE_FACTOR * ops::exp2(ev100)
}

/// Linear exposure multiplier for a given EV100: `1 / (1.2 * 2^EV100)`.
///
/// Radiance is multiplied by this before it is written to the HDR buffer, so a
/// surface at `Lmax` lands on `1.0`.
#[must_use]
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    1.0 / max_luminance_for_ev100(ev100)
}

/// A log-luminance histogram description shared by the reduction that builds it
/// on the GPU and the average it feeds back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramRange {
    /// Lowest log2 luminance the histogram resolves (bin 0's lower edge).
    pub min_log2_luminance: f32,
    /// Highest log2 luminance the histogram resolves (last bin's upper edge).
    pub max_log2_luminance: f32,
}

impl Default for HistogramRange {
    /// A wide `[2^-10, 2^12]` luminance window, matching typical AAA defaults.
    fn default() -> Self {
        Self {
            min_log2_luminance: -10.0,
            max_log2_luminance: 12.0,
        }
    }
}

impl HistogramRange {
    /// Log2-luminance span of the histogram (guaranteed strictly positive so
    /// the bin maths never divides by zero).
    #[must_use]
    pub fn span(&self) -> f32 {
        (self.max_log2_luminance - self.min_log2_luminance).max(1.0e-6)
    }

    /// Maps a linear luminance to a normalised `[0, 1]` histogram position.
    /// Non-positive luminance maps to `0.0` (the darkest bin).
    #[must_use]
    pub fn luminance_to_unit(&self, luminance: f32) -> f32 {
        if luminance <= 0.0 {
            return 0.0;
        }
        let log2_lum = ops::log2(luminance);
        ((log2_lum - self.min_log2_luminance) / self.span()).clamp(0.0, 1.0)
    }

    /// Bin index (in `[0, bin_count - 1]`) a luminance falls into.
    #[must_use]
    pub fn luminance_to_bin(&self, luminance: f32, bin_count: u32) -> u32 {
        if bin_count == 0 {
            return 0;
        }
        let unit = self.luminance_to_unit(luminance);
        let scaled = unit * bin_count as f32;
        let idx = scaled as u32;
        idx.min(bin_count - 1)
    }

    /// Representative linear luminance of a bin's centre.
    #[must_use]
    pub fn bin_center_luminance(&self, bin: u32, bin_count: u32) -> f32 {
        if bin_count == 0 {
            return 0.0;
        }
        let unit = (bin as f32 + 0.5) / bin_count as f32;
        let log2_lum = self.min_log2_luminance + unit * self.span();
        ops::exp2(log2_lum)
    }
}

/// Percentile band used to reject the darkest/brightest pixels when averaging a
/// histogram, matching UE's `AutoExposure` low/high percent knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramPercentiles {
    /// Fraction of the darkest pixels to discard, in `[0, 1)`.
    pub low: f32,
    /// Fraction of the brightest pixels to discard, in `[0, 1)`.
    pub high: f32,
}

impl Default for HistogramPercentiles {
    /// Discard the darkest 50% and brightest 10%, a common AAA starting point
    /// that ignores dark background and tiny specular highlights.
    fn default() -> Self {
        Self {
            low: 0.5,
            high: 0.1,
        }
    }
}

/// Average linear luminance of a log-luminance histogram, discarding the
/// darkest `low` and brightest `high` fraction of samples (count-weighted mean
/// of the surviving bins' centre luminances).
///
/// Mirrors UE's histogram auto-exposure: the tails are trimmed so background
/// darkness and pinprick highlights do not drag the metering. An empty (or
/// fully trimmed) histogram returns `0.0`.
#[must_use]
pub fn average_luminance_from_histogram(
    bins: &[u32],
    range: HistogramRange,
    percentiles: HistogramPercentiles,
) -> f32 {
    let bin_count = bins.len() as u32;
    if bin_count == 0 {
        return 0.0;
    }
    let total: u64 = bins.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f32;
    let low = percentiles.low.clamp(0.0, 1.0);
    let high = percentiles.high.clamp(0.0, 1.0);
    // Sample window that survives the trim, expressed as cumulative counts.
    let drop_low = low * total_f;
    let keep_high = total_f - high * total_f;

    let mut cumulative = 0.0_f32;
    let mut weighted_sum = 0.0_f32;
    let mut weight = 0.0_f32;
    for (bin, &count) in bins.iter().enumerate() {
        let count_f = count as f32;
        let bin_start = cumulative;
        let bin_end = cumulative + count_f;
        cumulative = bin_end;
        // Portion of this bin inside the [drop_low, keep_high] window.
        let lo = bin_start.max(drop_low);
        let hi = bin_end.min(keep_high);
        let surviving = (hi - lo).max(0.0);
        if surviving <= 0.0 {
            continue;
        }
        let center = range.bin_center_luminance(bin as u32, bin_count);
        weighted_sum += center * surviving;
        weight += surviving;
    }

    if weight <= 0.0 {
        // The whole window collapsed (e.g. low + high >= 1): fall back to the
        // untrimmed count-weighted mean so metering never returns zero light.
        let mut fallback_sum = 0.0_f32;
        for (bin, &count) in bins.iter().enumerate() {
            fallback_sum += range.bin_center_luminance(bin as u32, bin_count) * count as f32;
        }
        return fallback_sum / total_f;
    }

    weighted_sum / weight
}

/// Artist controls that turn a measured luminance into a final exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoExposureSettings {
    /// Lowest EV100 the metering may settle to (bright-scene clamp).
    pub min_ev100: f32,
    /// Highest EV100 the metering may settle to (dark-scene clamp).
    pub max_ev100: f32,
    /// Exposure compensation in stops, added to the metered EV100 (positive
    /// darkens, matching photographic convention where higher EV = less light).
    pub compensation_stops: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            min_ev100: -8.0,
            max_ev100: 16.0,
            compensation_stops: 0.0,
        }
    }
}

impl AutoExposureSettings {
    /// Clamped, compensated EV100 for a metered average luminance.
    #[must_use]
    pub fn resolve_ev100(&self, average_luminance: f32) -> f32 {
        let (lo, hi) = if self.min_ev100 <= self.max_ev100 {
            (self.min_ev100, self.max_ev100)
        } else {
            (self.max_ev100, self.min_ev100)
        };
        let metered = ev100_from_average_luminance(average_luminance);
        (metered + self.compensation_stops).clamp(lo, hi)
    }

    /// Target linear exposure multiplier for a metered average luminance.
    #[must_use]
    pub fn resolve_exposure(&self, average_luminance: f32) -> f32 {
        exposure_from_ev100(self.resolve_ev100(average_luminance))
    }
}

/// Temporal eye-adaptation response with separate brightening/darkening rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeAdaptation {
    /// Adaptation speed when the target is *brighter* than the current state
    /// (pupil contracting). Higher adapts faster. Units are 1/seconds.
    pub speed_up: f32,
    /// Adaptation speed when the target is *darker* than the current state
    /// (pupil dilating). Higher adapts faster. Units are 1/seconds.
    pub speed_down: f32,
}

impl Default for EyeAdaptation {
    /// Brighten faster than darken, mirroring human vision and UE defaults.
    fn default() -> Self {
        Self {
            speed_up: 3.0,
            speed_down: 1.0,
        }
    }
}

impl EyeAdaptation {
    /// Eases `current` toward `target` over `delta_seconds` with an exponential
    /// response `current + (target - current) * (1 - exp(-dt * speed))`.
    ///
    /// The brightening speed is used when the target exceeds the current value,
    /// the darkening speed otherwise. A non-positive time step returns the
    /// current value unchanged so the result is frame-rate independent and
    /// never overshoots.
    #[must_use]
    pub fn adapt(&self, current: f32, target: f32, delta_seconds: f32) -> f32 {
        if delta_seconds <= 0.0 {
            return current;
        }
        let speed = if target > current {
            self.speed_up
        } else {
            self.speed_down
        }
        .max(0.0);
        let factor = 1.0 - ops::exp(-delta_seconds * speed);
        current + (target - current) * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    #[test]
    fn luminance_matches_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    #[test]
    fn sunny_16_is_about_ev15() {
        // f/16, 1/125 s, ISO 100 is the canonical EV100 ~= 15 reference.
        let camera = PhysicalCamera::default();
        approx(camera.ev100(), ops::log2(16.0 * 16.0 * 125.0));
        assert!((camera.ev100() - 15.0).abs() < 0.05);
    }

    #[test]
    fn iso_and_shutter_move_ev_the_expected_direction() {
        let base = PhysicalCamera::default().ev100();
        // Doubling ISO gathers one more stop of light -> EV100 drops by 1.
        let hi_iso = PhysicalCamera {
            iso: 200.0,
            ..PhysicalCamera::default()
        };
        approx(hi_iso.ev100(), base - 1.0);
        // Halving shutter time (faster) admits one less stop -> EV100 rises by 1.
        let fast = PhysicalCamera {
            shutter_time: 1.0 / 250.0,
            ..PhysicalCamera::default()
        };
        approx(fast.ev100(), base + 1.0);
    }

    #[test]
    fn degenerate_camera_is_safe() {
        let bad = PhysicalCamera {
            aperture: 0.0,
            shutter_time: 0.0,
            iso: 0.0,
        };
        approx(bad.ev100(), 0.0);
    }

    #[test]
    fn exposure_maps_max_luminance_to_one() {
        let ev = 10.0;
        let exposure = exposure_from_ev100(ev);
        let lmax = max_luminance_for_ev100(ev);
        approx(exposure * lmax, 1.0);
    }

    #[test]
    fn one_stop_halves_exposure() {
        // +1 EV means half the light, so the exposure multiplier halves.
        approx(exposure_from_ev100(11.0), exposure_from_ev100(10.0) * 0.5);
    }

    #[test]
    fn meter_round_trips_middle_grey() {
        // 18% grey under EV100 L should meter back to the same EV100.
        let ev = 12.0;
        let l = 12.5 / 100.0 * max_luminance_for_ev100(ev) / MAX_LUMINANCE_FACTOR;
        // Reconstruct the luminance that meters to `ev`: L = K/100 * 2^ev.
        let metered_l = METER_CALIBRATION_K / 100.0 * ops::exp2(ev);
        approx(ev100_from_average_luminance(metered_l), ev);
        let _ = l;
    }

    #[test]
    fn histogram_bin_mapping_is_monotonic_and_clamped() {
        let range = HistogramRange::default();
        let count = 64;
        assert_eq!(range.luminance_to_bin(0.0, count), 0);
        // Far above the max maps to the last bin, not out of range.
        assert_eq!(range.luminance_to_bin(1.0e9, count), count - 1);
        let dark = range.luminance_to_bin(0.01, count);
        let bright = range.luminance_to_bin(100.0, count);
        assert!(bright > dark);
    }

    #[test]
    fn bin_center_inverts_bin_mapping() {
        let range = HistogramRange::default();
        let count = 64;
        for bin in [0_u32, 5, 31, 63] {
            let center = range.bin_center_luminance(bin, count);
            assert_eq!(range.luminance_to_bin(center, count), bin);
        }
    }

    #[test]
    fn empty_histogram_averages_to_zero() {
        let bins = [0_u32; 16];
        approx(
            average_luminance_from_histogram(
                &bins,
                HistogramRange::default(),
                HistogramPercentiles::default(),
            ),
            0.0,
        );
    }

    #[test]
    fn single_populated_bin_returns_its_center() {
        let range = HistogramRange::default();
        let count = 32_u32;
        let mut bins = vec![0_u32; count as usize];
        bins[20] = 1000;
        // No trimming so the sole bin survives.
        let none = HistogramPercentiles {
            low: 0.0,
            high: 0.0,
        };
        approx(
            average_luminance_from_histogram(&bins, range, none),
            range.bin_center_luminance(20, count),
        );
    }

    #[test]
    fn percentile_trim_rejects_dark_and_bright_tails() {
        let range = HistogramRange::default();
        let count = 8_u32;
        // Equal weight in the darkest, a middle, and the brightest bin.
        let mut bins = vec![0_u32; count as usize];
        bins[0] = 100;
        bins[4] = 100;
        bins[7] = 100;
        // Trim the darkest third and brightest third -> only the middle bin.
        let trim = HistogramPercentiles {
            low: 0.34,
            high: 0.34,
        };
        let avg = average_luminance_from_histogram(&bins, range, trim);
        approx(avg, range.bin_center_luminance(4, count));
    }

    #[test]
    fn full_trim_falls_back_to_untrimmed_mean() {
        let range = HistogramRange::default();
        let count = 4_u32;
        let mut bins = vec![0_u32; count as usize];
        bins[1] = 10;
        bins[2] = 10;
        // low + high >= 1 collapses the window; fall back rather than return 0.
        let collapse = HistogramPercentiles {
            low: 0.6,
            high: 0.6,
        };
        let avg = average_luminance_from_histogram(&bins, range, collapse);
        let expect =
            (range.bin_center_luminance(1, count) + range.bin_center_luminance(2, count)) / 2.0;
        approx(avg, expect);
    }

    #[test]
    fn auto_exposure_clamps_ev_range() {
        let settings = AutoExposureSettings {
            min_ev100: 4.0,
            max_ev100: 8.0,
            compensation_stops: 0.0,
        };
        // Very bright scene wants a high EV but is clamped to max.
        let bright_l = METER_CALIBRATION_K / 100.0 * ops::exp2(20.0);
        approx(settings.resolve_ev100(bright_l), 8.0);
        // Very dark scene clamps to min.
        let dark_l = METER_CALIBRATION_K / 100.0 * ops::exp2(-10.0);
        approx(settings.resolve_ev100(dark_l), 4.0);
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
    fn eye_adaptation_eases_toward_target() {
        let eye = EyeAdaptation::default();
        // One step moves partway, never overshoots.
        let next = eye.adapt(1.0, 2.0, 0.1);
        assert!(next > 1.0 && next < 2.0);
        // Zero time step is a no-op.
        approx(eye.adapt(1.0, 2.0, 0.0), 1.0);
    }

    #[test]
    fn eye_adaptation_converges_and_respects_speeds() {
        let eye = EyeAdaptation {
            speed_up: 4.0,
            speed_down: 1.0,
        };
        // Brightening uses the faster speed.
        let up = eye.adapt(1.0, 2.0, 0.25);
        // Darkening the same magnitude with the slower speed moves less.
        let down = eye.adapt(2.0, 1.0, 0.25);
        let up_delta = up - 1.0;
        let down_delta = 2.0 - down;
        assert!(up_delta > down_delta);

        // Iterating drives the state to the target.
        let mut state = 1.0_f32;
        for _ in 0..200 {
            state = eye.adapt(state, 5.0, 0.05);
        }
        approx(state, 5.0);
    }
}
