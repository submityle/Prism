//! Per-path propagation Doppler.
//!
//! A [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//! resolves, once per control-rate query, the set of
//! [`PropagationPath`]s a source takes to the listener, each carrying its own
//! [`delay_seconds`](PropagationPath::delay_seconds). As the scene moves —
//! the source walks away, the listener turns, a reflector slides past — each
//! arrival's path length, and therefore its delay, changes from one query to
//! the next. That changing delay *is* the Doppler effect: a shrinking delay
//! (an approaching arrival) compresses the waveform in time and raises its
//! pitch, a growing delay (a receding arrival) stretches it and lowers the
//! pitch.
//!
//! Crucially, every arrival Dopplers on its own. The direct line of sight, a
//! specular bounce off a wall sweeping toward the listener, and a diffracted
//! bend around a corner each change length at a different rate, so each needs
//! its own pitch factor. Modelling one Doppler for the whole source (from the
//! source-to-listener distance alone, as a first-pass spatialiser does) is
//! audibly wrong the moment a reflection moves differently from the direct
//! sound. This module supplies the per-path treatment a shipping engine needs:
//! a stateful [`PerPathDoppler`] that remembers each tracked arrival's delay
//! from the previous query and, every update, turns that arrival's delay rate
//! of change into a per-path pitch factor the real-time voice applies as a
//! resampling ratio.
//!
//! # Pitch-factor convention
//!
//! The received signal at wall-clock time `t` is the signal the source emitted
//! at `t - delay(t)`, so the rate at which emission time advances per unit of
//! reception time — the playback (pitch) ratio — is
//! `1 - d(delay)/dt`. An approaching arrival (delay falling, `d(delay)/dt < 0`)
//! yields a factor above `1.0` (higher pitch); a receding arrival (delay
//! rising) yields a factor below `1.0` (lower pitch); a steady arrival yields
//! exactly `1.0` ([`UNISON`]). The factor is clamped to
//! [`DopplerLimits::min_factor`]..=[`DopplerLimits::max_factor`] so a one-frame
//! delay discontinuity (an arrival that re-sorts, a reflector that pops into
//! view) cannot produce a runaway pitch, and the raw per-frame estimate is
//! eased with a one-pole glide so quantisation jitter in the delay does not
//! chatter the pitch.
//!
//! A newly appeared arrival has no previous delay to difference against, so it
//! starts at [`UNISON`] rather than guessing a slope from one sample; a
//! vanished arrival is retired immediately, since there is no pitch to fade —
//! the arrival itself is gone.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! factor is the textbook moving-observer Doppler ratio expressed through the
//! path delay (`factor = 1 - d(delay)/dt`); the glide is the same one-pole
//! parameter smoother ([`crate::path_smoothing`]) uses, and the clamp is a
//! plain range limit.
//!
//! # Relationship
//!
//! Consumes the same [`PropagationPath`] set the real-time voice renders — the
//! raw [`crate::backend::GeometricBackend`] output, or more usefully the
//! time-coherent set [`crate::path_smoothing::PathSmoother`] produces, so the
//! pitch tracks the very delay trajectory the voice's delay lines follow — and
//! writes one pitch factor per path into a caller-provided buffer, index for
//! index. Like [`crate::path_smoothing::PathSmoother`] it is a stateful,
//! per-voice sibling of the stateless per-path colouring terms
//! ([`crate::source_directivity`], [`crate::receiver_directivity`],
//! [`crate::air_absorption`]): it holds the previous delay across queries, keys
//! tracked arrivals by the same `(kind, ordinal)` correspondence the smoother
//! uses, and is ticked once per control-rate update with the elapsed
//! wall-clock step. It reuses
//! [`TRACKED_PATH_CAPACITY`](crate::path_smoothing::TRACKED_PATH_CAPACITY) so a
//! voice's Doppler and smoother track exactly the same arrivals.
//!
//! It is deliberately decoupled from the delay-line *length* the voice renders:
//! the delay positions the arrival in time, while this factor drives an
//! independent per-path pitch/resample stage (design sections 15 and 34), so a
//! quantised or interpolation-order-limited delay line does not dictate the
//! Doppler and the two can be tuned separately.

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_spatial::propagation::{PathKind, PropagationPath};

use crate::path_smoothing::TRACKED_PATH_CAPACITY;

/// The neutral pitch factor: no Doppler shift (emission and reception time
/// advance together).
pub const UNISON: Sample = 1.0;

/// Default lower bound on the pitch factor (one octave down). Clamps the shift
/// a fast-receding or discontinuously re-sorted arrival can apply.
pub const DEFAULT_MIN_FACTOR: Sample = 0.5;

/// Default upper bound on the pitch factor (one octave up). Clamps the shift a
/// fast-approaching or discontinuously re-sorted arrival can apply.
pub const DEFAULT_MAX_FACTOR: Sample = 2.0;

/// Default glide time constant (seconds) for the pitch factor. About `50 ms`
/// smooths the frame-to-frame delay-rate estimate — which is only as precise
/// as the delay quantisation — into a pitch that glides rather than chatters,
/// while staying fast enough that a genuine acceleration is heard promptly.
pub const DEFAULT_SMOOTHING_SECONDS: Sample = 0.05;

/// The per-path pitch factor computed from the raw delay change over one time
/// step, before clamping or smoothing: `1 - (current - previous) / dt`.
///
/// Returns [`UNISON`] when `dt_seconds` is not positive (no time has elapsed,
/// so no rate is defined). An approaching arrival (`current < previous`)
/// returns a factor above `1.0`; a receding arrival returns one below `1.0`.
#[must_use]
pub fn doppler_factor(
    previous_delay_seconds: Sample,
    current_delay_seconds: Sample,
    dt_seconds: Sample,
) -> Sample {
    if dt_seconds <= 0.0 {
        return UNISON;
    }
    let rate = (current_delay_seconds - previous_delay_seconds) / dt_seconds;
    UNISON - rate
}

/// The clamp range and glide time a [`PerPathDoppler`] applies to the raw
/// per-path factor.
///
/// A `smoothing_seconds` of zero (or less) disables the glide: the clamped raw
/// factor is reported directly each update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerLimits {
    /// Lower bound on the reported pitch factor.
    pub min_factor: Sample,
    /// Upper bound on the reported pitch factor.
    pub max_factor: Sample,
    /// Glide time constant (seconds) easing the factor toward the clamped raw
    /// estimate; zero or less snaps straight to it.
    pub smoothing_seconds: Sample,
}

impl DopplerLimits {
    /// The documented defaults: a one-octave clamp either way and a `50 ms`
    /// glide.
    pub const DEFAULT: Self = Self {
        min_factor: DEFAULT_MIN_FACTOR,
        max_factor: DEFAULT_MAX_FACTOR,
        smoothing_seconds: DEFAULT_SMOOTHING_SECONDS,
    };

    /// Limits that disable the glide (the clamped raw factor is reported each
    /// update) while keeping the default clamp. Useful for an offline render
    /// that wants the un-eased per-frame factor.
    pub const INSTANT: Self = Self {
        min_factor: DEFAULT_MIN_FACTOR,
        max_factor: DEFAULT_MAX_FACTOR,
        smoothing_seconds: 0.0,
    };

    /// Returns these limits with the clamp widened or narrowed to
    /// `[min_factor, max_factor]`, keeping the glide. The bounds are ordered so
    /// a swapped pair cannot invert the clamp.
    #[must_use]
    pub fn with_clamp(mut self, min_factor: Sample, max_factor: Sample) -> Self {
        let lo = min_factor.min(max_factor).max(0.0);
        let hi = min_factor.max(max_factor);
        self.min_factor = lo;
        self.max_factor = hi;
        self
    }

    /// Returns these limits with the glide time constant replaced.
    #[must_use]
    pub fn with_smoothing_seconds(mut self, smoothing_seconds: Sample) -> Self {
        self.smoothing_seconds = smoothing_seconds;
        self
    }

    /// Clamps a raw factor to this range.
    #[inline]
    #[must_use]
    fn clamp(&self, factor: Sample) -> Sample {
        factor.clamp(self.min_factor, self.max_factor)
    }
}

impl Default for DopplerLimits {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One tracked arrival: its stable identity, the delay it last reported, and
/// the pitch factor currently rendered.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Whether this slot holds a live arrival.
    active: bool,
    /// Whether an incoming path matched this slot on the current update.
    matched: bool,
    /// The mechanism identity half of the correspondence key.
    kind: PathKind,
    /// The ordinal (index among same-kind arrivals) half of the key.
    ordinal: usize,
    /// The delay (seconds) this arrival reported on the previous update, used
    /// to difference against the current delay.
    previous_delay: Sample,
    /// The smoothed pitch factor currently reported for this arrival.
    factor: Sample,
}

impl Slot {
    /// An empty, inactive slot used to pre-fill the tracking table.
    const EMPTY: Self = Self {
        active: false,
        matched: false,
        kind: PathKind::Direct,
        ordinal: 0,
        previous_delay: 0.0,
        factor: UNISON,
    };
}

/// A stateful per-voice tracker that turns a backend's per-query
/// [`PropagationPath`] delays into one pitch factor per arrival.
///
/// Create one per source voice, then call [`PerPathDoppler::update`] once per
/// control-rate tick with the voice's current paths, the elapsed time since the
/// previous tick, and an output buffer. Entry `i` of the output is the pitch
/// factor for path `i`: multiply the per-path resampler's read rate by it (or
/// equivalently add `12 * log2(factor)` semitones) to render that arrival's
/// Doppler.
#[derive(Debug, Clone)]
pub struct PerPathDoppler {
    limits: DopplerLimits,
    slots: [Slot; TRACKED_PATH_CAPACITY],
}

impl PerPathDoppler {
    /// Creates a tracker with the given clamp/glide limits and no tracked
    /// arrivals yet.
    #[must_use]
    pub fn new(limits: DopplerLimits) -> Self {
        Self {
            limits,
            slots: [Slot::EMPTY; TRACKED_PATH_CAPACITY],
        }
    }

    /// Returns the clamp/glide limits in effect.
    #[inline]
    #[must_use]
    pub fn limits(&self) -> DopplerLimits {
        self.limits
    }

    /// Replaces the clamp/glide limits; takes effect on the next update.
    #[inline]
    pub fn set_limits(&mut self, limits: DopplerLimits) {
        self.limits = limits;
    }

    /// Forgets all tracked arrivals, so the next update starts every arrival at
    /// [`UNISON`]. Call this on a teleport or scene cut where differencing the
    /// delay across the discontinuity would fabricate a huge false Doppler.
    pub fn reset(&mut self) {
        self.slots = [Slot::EMPTY; TRACKED_PATH_CAPACITY];
    }

    /// The number of arrivals currently tracked.
    #[inline]
    #[must_use]
    pub fn tracked_len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.active).count()
    }

    /// Advances every tracked arrival one control-rate step and writes the
    /// per-path pitch factor into `factors`, index for index with `paths`.
    ///
    /// Returns how many entries of `factors` were populated (the smaller of
    /// `paths.len()` and `factors.len()`). Arrivals present in `paths` but not
    /// tracked before start at [`UNISON`]; arrivals tracked before but absent
    /// from `paths` are retired. Every path is keyed and tracked even when the
    /// output buffer is shorter than `paths`, so the delay history stays
    /// correct across frames; only the leading `factors.len()` factors are
    /// written.
    pub fn update(
        &mut self,
        paths: &[PropagationPath],
        dt_seconds: Sample,
        factors: &mut [Sample],
    ) -> usize {
        let dt = dt_seconds.max(0.0);
        let alpha = one_pole_alpha(dt, self.limits.smoothing_seconds);

        for slot in &mut self.slots {
            slot.matched = false;
        }

        for (index, path) in paths.iter().enumerate() {
            let ordinal = paths[..index]
                .iter()
                .filter(|earlier| earlier.kind == path.kind)
                .count();
            let factor = self.track(path, ordinal, dt, alpha);
            if let Some(out) = factors.get_mut(index) {
                *out = factor;
            }
        }

        self.retire_unmatched();
        paths.len().min(factors.len())
    }

    /// Continues an existing arrival with the same `(kind, ordinal)` key —
    /// differencing its delay and easing its factor — or registers a new one at
    /// [`UNISON`]. Returns the factor to report for this arrival.
    fn track(&mut self, path: &PropagationPath, ordinal: usize, dt: Sample, alpha: Sample) -> Sample {
        let limits = self.limits;
        if let Some(slot) = self.slots.iter_mut().find(|slot| {
            slot.active && !slot.matched && slot.kind == path.kind && slot.ordinal == ordinal
        }) {
            slot.matched = true;
            if dt > 0.0 {
                let raw = doppler_factor(slot.previous_delay, path.delay_seconds, dt);
                let clamped = limits.clamp(raw);
                slot.factor += (clamped - slot.factor) * alpha;
                slot.previous_delay = path.delay_seconds;
            }
            return slot.factor;
        }

        let Some(index) = self.slots.iter().position(|slot| !slot.active) else {
            // Tracking table full (only possible past TRACKED_PATH_CAPACITY live
            // arrivals): report no shift rather than evicting a tracked arrival.
            return UNISON;
        };
        let slot = &mut self.slots[index];
        slot.active = true;
        slot.matched = true;
        slot.kind = path.kind;
        slot.ordinal = ordinal;
        slot.previous_delay = path.delay_seconds;
        slot.factor = UNISON;
        UNISON
    }

    /// Deactivates every active slot that no incoming path matched: a vanished
    /// arrival has no pitch to fade, so it is retired at once.
    fn retire_unmatched(&mut self) {
        for slot in &mut self.slots {
            if slot.active && !slot.matched {
                slot.active = false;
            }
        }
    }
}

impl Default for PerPathDoppler {
    #[inline]
    fn default() -> Self {
        Self::new(DopplerLimits::DEFAULT)
    }
}

/// The one-pole glide coefficient for a step of `dt` seconds with time constant
/// `tau`. A non-positive `tau` (or an actual update with no elapsed time) snaps
/// to the target. The exponential routes through [`bevy_math::ops`] so the
/// glide is bit-reproducible across targets.
#[must_use]
fn one_pole_alpha(dt: Sample, tau: Sample) -> Sample {
    if tau <= 0.0 || dt <= 0.0 {
        return if dt <= 0.0 && tau > 0.0 { 0.0 } else { 1.0 };
    }
    (1.0 - ops::exp(-dt / tau)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;
    use prism_audio_spatial::propagation::MAX_PROPAGATION_PATHS;

    /// Builds a path of the given kind at the given delay; the other fields do
    /// not affect the Doppler and are left neutral.
    fn path(kind: PathKind, delay_seconds: Sample) -> PropagationPath {
        PropagationPath {
            kind,
            delay_seconds,
            gain: 1.0,
            cutoff_hz: 20_000.0,
            bands: prism_audio_spatial::band_spectrum::BandGains::UNITY,
            direction: Vec3::NEG_Z,
        }
    }

    const DT: Sample = 1.0 / 60.0;

    #[test]
    fn new_arrival_starts_at_unison() {
        let mut doppler = PerPathDoppler::default();
        let paths = [path(PathKind::Direct, 0.01)];
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let count = doppler.update(&paths, DT, &mut factors);
        assert_eq!(count, 1);
        assert!((factors[0] - UNISON).abs() < 1.0e-9);
        assert_eq!(doppler.tracked_len(), 1);
    }

    #[test]
    fn approaching_arrival_raises_pitch() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let mut delay = 0.030;
        // First update registers the arrival at unison.
        doppler.update(&[path(PathKind::Direct, delay)], DT, &mut factors);
        // Then the delay shrinks each frame (the source approaches).
        for _ in 0..40 {
            delay -= 0.0005;
            doppler.update(&[path(PathKind::Direct, delay)], DT, &mut factors);
        }
        assert!(factors[0] > UNISON, "approaching should raise pitch, got {}", factors[0]);
        assert!(factors[0] <= DEFAULT_MAX_FACTOR + 1.0e-6);
    }

    #[test]
    fn receding_arrival_lowers_pitch() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let mut delay = 0.010;
        doppler.update(&[path(PathKind::Direct, delay)], DT, &mut factors);
        for _ in 0..40 {
            delay += 0.0005;
            doppler.update(&[path(PathKind::Direct, delay)], DT, &mut factors);
        }
        assert!(factors[0] < UNISON, "receding should lower pitch, got {}", factors[0]);
        assert!(factors[0] >= DEFAULT_MIN_FACTOR - 1.0e-6);
    }

    #[test]
    fn static_arrival_converges_to_unison() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let paths = [path(PathKind::Direct, 0.020)];
        for _ in 0..200 {
            doppler.update(&paths, DT, &mut factors);
        }
        assert!((factors[0] - UNISON).abs() < 1.0e-4);
    }

    #[test]
    fn factor_is_clamped_on_a_delay_discontinuity() {
        let mut doppler = PerPathDoppler::new(DopplerLimits::INSTANT);
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        // Register far away, then jump to the listener in one frame: a huge
        // negative delay rate the clamp must cap at max_factor.
        doppler.update(&[path(PathKind::Direct, 0.100)], DT, &mut factors);
        doppler.update(&[path(PathKind::Direct, 0.000)], DT, &mut factors);
        assert!((factors[0] - DEFAULT_MAX_FACTOR).abs() < 1.0e-6);

        // And the reverse jump caps at min_factor.
        let mut other = PerPathDoppler::new(DopplerLimits::INSTANT);
        other.update(&[path(PathKind::Direct, 0.000)], DT, &mut factors);
        other.update(&[path(PathKind::Direct, 0.100)], DT, &mut factors);
        assert!((factors[0] - DEFAULT_MIN_FACTOR).abs() < 1.0e-6);
    }

    #[test]
    fn distinct_kinds_track_independently() {
        let mut doppler = PerPathDoppler::new(DopplerLimits::INSTANT);
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let mut direct = 0.010;
        let mut reflection = 0.030;
        doppler.update(
            &[path(PathKind::Direct, direct), path(PathKind::Reflection, reflection)],
            DT,
            &mut factors,
        );
        // Direct approaches, reflection recedes, in the same frame.
        direct -= 0.0005;
        reflection += 0.0005;
        doppler.update(
            &[path(PathKind::Direct, direct), path(PathKind::Reflection, reflection)],
            DT,
            &mut factors,
        );
        assert!(factors[0] > UNISON, "direct approaching");
        assert!(factors[1] < UNISON, "reflection receding");
        assert_eq!(doppler.tracked_len(), 2);
    }

    #[test]
    fn ordinal_keys_same_kind_arrivals() {
        let mut doppler = PerPathDoppler::new(DopplerLimits::INSTANT);
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        // Two reflections: the first approaches, the second recedes.
        doppler.update(
            &[path(PathKind::Reflection, 0.020), path(PathKind::Reflection, 0.040)],
            DT,
            &mut factors,
        );
        doppler.update(
            &[path(PathKind::Reflection, 0.019), path(PathKind::Reflection, 0.041)],
            DT,
            &mut factors,
        );
        assert!(factors[0] > UNISON, "reflection ordinal 0 approaching");
        assert!(factors[1] < UNISON, "reflection ordinal 1 receding");
    }

    #[test]
    fn vanished_arrival_is_retired() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        doppler.update(&[path(PathKind::Direct, 0.01)], DT, &mut factors);
        assert_eq!(doppler.tracked_len(), 1);
        // No paths this frame: the arrival vanishes and is retired at once.
        doppler.update(&[], DT, &mut factors);
        assert_eq!(doppler.tracked_len(), 0);
    }

    #[test]
    fn reappearing_arrival_restarts_at_unison() {
        let mut doppler = PerPathDoppler::new(DopplerLimits::INSTANT);
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        doppler.update(&[path(PathKind::Direct, 0.020)], DT, &mut factors);
        doppler.update(&[path(PathKind::Direct, 0.010)], DT, &mut factors);
        assert!(factors[0] > UNISON);
        // The arrival vanishes, then returns at a new delay: no false Doppler
        // from the gap, it restarts at unison.
        doppler.update(&[], DT, &mut factors);
        let count = doppler.update(&[path(PathKind::Direct, 0.050)], DT, &mut factors);
        assert_eq!(count, 1);
        assert!((factors[0] - UNISON).abs() < 1.0e-9);
    }

    #[test]
    fn instant_limits_skip_the_glide() {
        let mut doppler = PerPathDoppler::new(DopplerLimits::INSTANT);
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        doppler.update(&[path(PathKind::Direct, 0.020)], DT, &mut factors);
        // delay falls by 0.0005 over DT: rate = -0.03, raw factor = 1.03,
        // reported immediately with no easing.
        doppler.update(&[path(PathKind::Direct, 0.0195)], DT, &mut factors);
        let expected = doppler_factor(0.020, 0.0195, DT);
        assert!((factors[0] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn zero_dt_reports_unison_without_nan() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        let count = doppler.update(&[path(PathKind::Direct, 0.02)], 0.0, &mut factors);
        assert_eq!(count, 1);
        assert!(factors[0].is_finite());
        assert!((factors[0] - UNISON).abs() < 1.0e-9);
    }

    #[test]
    fn doppler_factor_matches_the_moving_observer_ratio() {
        // Delay falls from 0.010 to 0.005 over 0.1 s: rate = -0.05, factor 1.05.
        assert!((doppler_factor(0.010, 0.005, 0.1) - 1.05).abs() < 1.0e-6);
        // Delay rises: factor below unison.
        assert!((doppler_factor(0.005, 0.010, 0.1) - 0.95).abs() < 1.0e-6);
        // No elapsed time: unison.
        assert!((doppler_factor(0.010, 0.005, 0.0) - UNISON).abs() < 1.0e-9);
    }

    #[test]
    fn output_count_is_capped_to_the_buffer() {
        let mut doppler = PerPathDoppler::default();
        let paths = [
            path(PathKind::Direct, 0.01),
            path(PathKind::Reflection, 0.02),
            path(PathKind::Diffraction, 0.03),
        ];
        let mut one = [0.0; 1];
        let count = doppler.update(&paths, DT, &mut one);
        assert_eq!(count, 1);
        // All three are still tracked even though only one factor was written.
        assert_eq!(doppler.tracked_len(), 3);
    }

    #[test]
    fn reset_forgets_all_tracked_arrivals() {
        let mut doppler = PerPathDoppler::default();
        let mut factors = [0.0; MAX_PROPAGATION_PATHS];
        doppler.update(&[path(PathKind::Direct, 0.02)], DT, &mut factors);
        doppler.update(&[path(PathKind::Direct, 0.01)], DT, &mut factors);
        assert_eq!(doppler.tracked_len(), 1);
        doppler.reset();
        assert_eq!(doppler.tracked_len(), 0);
        // After reset the arrival is reborn at unison rather than inheriting the
        // old approaching slope.
        doppler.update(&[path(PathKind::Direct, 0.005)], DT, &mut factors);
        assert!((factors[0] - UNISON).abs() < 1.0e-9);
    }

    #[test]
    fn with_clamp_orders_swapped_bounds() {
        let limits = DopplerLimits::DEFAULT.with_clamp(2.0, 0.25);
        assert!((limits.min_factor - 0.25).abs() < 1.0e-9);
        assert!((limits.max_factor - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn is_deterministic_across_instances() {
        let script = [0.030, 0.028, 0.025, 0.025, 0.027, 0.031];
        let run = || {
            let mut doppler = PerPathDoppler::default();
            let mut factors = [0.0; MAX_PROPAGATION_PATHS];
            let mut last = 0.0;
            for &delay in &script {
                doppler.update(&[path(PathKind::Direct, delay)], DT, &mut factors);
                last = factors[0];
            }
            last
        };
        assert_eq!(run(), run());
    }
}
