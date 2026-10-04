//! Frame-to-frame smoothing of resolved propagation paths.
//!
//! A [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//! resolves the arrivals of a source at control rate: once per query it reports
//! a fresh set of [`PropagationPath`]s. Between two such queries the geometry
//! can change abruptly — a door swings shut and occludes the direct wave, a
//! reflector slides into view and a new specular bounce appears, the listener
//! turns and the arrivals re-sort — and feeding those discontinuous parameters
//! straight to the real-time voice is audible: a jumped gain clicks (zipper
//! noise), a jumped delay snaps the pitch, and a jumped filter corner pops.
//!
//! This module supplies the missing glue a shipping engine always has between
//! the control-rate solver and the audio-rate renderer: a stateful
//! [`PathSmoother`] that remembers the parameters it rendered last time and,
//! every update, eases each tracked arrival toward the backend's newest target
//! with a per-parameter one-pole glide. Arrivals that appear swell up from
//! silence; arrivals that vanish fade down to silence and are then retired, so
//! the voice never hears a hard edge. At steady state (unchanging geometry) the
//! smoothed parameters converge exactly onto the backend's targets, so the
//! model adds motion only while the scene is actually changing.
//!
//! # Gain convention
//!
//! Unlike the per-path colouring terms ([`crate::source_directivity`],
//! [`crate::receiver_directivity`], [`crate::air_absorption`]), which deliberately
//! touch only the per-band [`BandGains`] and leave the scalar
//! [`gain`](prism_audio_spatial::propagation::PropagationPath::gain) alone, this
//! module is the one place that eases the scalar `gain` over time — that is
//! precisely its job, since an un-eased `gain` step is the loudest artifact of
//! all. The easing is bias-free: every parameter is only ever moved *toward* the
//! backend's reported target, and once the target stops moving the smoothed
//! value settles onto it, so a steady scene renders bit-for-bit what the backend
//! asked for. The smoother never invents loudness: a tracked arrival's gain is
//! bounded by the targets it has been given and can only decay once its target
//! becomes zero (a vanished path).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The glide
//! is the textbook one-pole parameter smoother (`y += (x - y) * (1 - exp(-dt /
//! tau))`); the delay and filter corner are eased where a naive linear blend
//! would be perceptually wrong (delay as time, corner in the log-frequency
//! domain), and the arrival direction by renormalised linear interpolation.
//!
//! # Relationship
//!
//! Consumes the [`PropagationPath`] set produced by
//! [`crate::backend::GeometricBackend`] (or any other
//! [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend),
//! including the spatial crate's reference backends) and produces a like-shaped,
//! time-coherent set for the real-time voice. It holds state across queries, so
//! unlike the stateless per-path terms it is owned by the caller that drives one
//! source's voice, and ticked once per control-rate update with the elapsed
//! wall-clock step. It reuses [`prism_audio_spatial::band_spectrum`] for the
//! three propagation bands and nothing else.

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;
use prism_audio_spatial::band_spectrum::{BandGains, PROPAGATION_BAND_COUNT};
use prism_audio_spatial::propagation::{
    PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ, MAX_PROPAGATION_PATHS,
};

/// Default glide time constant (seconds) for the scalar broadband gain. About
/// `15 ms` removes zipper noise from a gain step while staying fast enough that
/// an occlusion change feels immediate.
pub const DEFAULT_GAIN_TIME_SECONDS: Sample = 0.015;

/// Default glide time constant (seconds) for the propagation delay. Delay is
/// eased more slowly (`40 ms`) because the ear is sensitive to the transient
/// pitch glide a moving delay line produces, so a gentler ramp sounds natural
/// rather than like an abrupt sweep.
pub const DEFAULT_DELAY_TIME_SECONDS: Sample = 0.040;

/// Default glide time constant (seconds) for the spectral shape: both the
/// single-pole [`cutoff_hz`](PropagationPath::cutoff_hz) corner (eased in the
/// log-frequency domain) and the per-band [`BandGains`]. `15 ms` matches the
/// gain so colour and level move together.
pub const DEFAULT_SPECTRUM_TIME_SECONDS: Sample = 0.015;

/// Default glide time constant (seconds) for the arrival direction. `15 ms`
/// keeps a panning arrival gliding smoothly as the listener or source turns.
pub const DEFAULT_DIRECTION_TIME_SECONDS: Sample = 0.015;

/// A faded-out arrival is retired once its smoothed gain falls below this linear
/// threshold (`-80 dB`), well under any audibility floor, freeing its slot.
pub const RETIRE_GAIN_EPSILON: Sample = 1.0e-4;

/// Lowest filter corner (Hz) the log-domain cutoff glide will represent, so the
/// logarithm stays finite even as an arrival darkens toward silence.
pub const MIN_CUTOFF_HZ: Sample = 20.0;

/// How many arrivals the smoother can track at once. Sized to twice the
/// backend's per-query ceiling ([`MAX_PROPAGATION_PATHS`]) so a full set can be
/// fading out while a full fresh set fades in during a hard scene cut, without
/// dropping either crossfade.
pub const TRACKED_PATH_CAPACITY: usize = 2 * MAX_PROPAGATION_PATHS;

/// Smallest squared length an interpolated direction may have and still define a
/// bearing; below it the glide snaps to the target rather than normalising a
/// near-zero vector.
const DIRECTION_EPSILON_SQ: Sample = 1.0e-12;

/// The per-parameter glide time constants (seconds) a [`PathSmoother`] eases
/// with. A time constant of zero (or less) means "no smoothing": that parameter
/// snaps straight to the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothingTimes {
    /// Time constant for the scalar broadband gain.
    pub gain_seconds: Sample,
    /// Time constant for the propagation delay.
    pub delay_seconds: Sample,
    /// Time constant for the spectral shape (cutoff corner and per-band gains).
    pub spectrum_seconds: Sample,
    /// Time constant for the arrival direction.
    pub direction_seconds: Sample,
}

impl SmoothingTimes {
    /// The documented defaults, tuned to remove artifacts while keeping scene
    /// changes responsive.
    pub const DEFAULT: Self = Self {
        gain_seconds: DEFAULT_GAIN_TIME_SECONDS,
        delay_seconds: DEFAULT_DELAY_TIME_SECONDS,
        spectrum_seconds: DEFAULT_SPECTRUM_TIME_SECONDS,
        direction_seconds: DEFAULT_DIRECTION_TIME_SECONDS,
    };

    /// Time constants that disable smoothing entirely: every parameter snaps to
    /// its target on each update. Useful for a teleport frame or an offline
    /// render that wants the raw backend output.
    pub const INSTANT: Self = Self {
        gain_seconds: 0.0,
        delay_seconds: 0.0,
        spectrum_seconds: 0.0,
        direction_seconds: 0.0,
    };
}

impl Default for SmoothingTimes {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One tracked arrival: its stable identity, the smoothed parameters currently
/// being rendered, and the backend targets being eased toward.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Whether this slot holds a live arrival.
    active: bool,
    /// Whether an incoming target matched this slot on the current update.
    matched: bool,
    /// Whether this arrival has vanished and is fading out before retirement.
    dying: bool,
    /// The mechanism identity half of the correspondence key.
    kind: PathKind,
    /// The ordinal (index among same-kind arrivals) half of the key.
    ordinal: usize,
    /// The parameters currently rendered (smoothed state).
    current: PropagationPath,
    /// Target broadband gain to ease toward.
    target_gain: Sample,
    /// Target delay (seconds) to ease toward.
    target_delay: Sample,
    /// Target filter corner (Hz) to ease toward.
    target_cutoff_hz: Sample,
    /// Target per-band gains to ease toward.
    target_bands: BandGains,
    /// Target arrival direction to ease toward.
    target_direction: Vec3,
}

impl Slot {
    /// An empty, inactive slot used to pre-fill the tracking table.
    const EMPTY: Self = Self {
        active: false,
        matched: false,
        dying: false,
        kind: PathKind::Direct,
        ordinal: 0,
        current: PropagationPath::SILENT,
        target_gain: 0.0,
        target_delay: 0.0,
        target_cutoff_hz: FULL_BAND_CUTOFF_HZ,
        target_bands: BandGains::SILENT,
        target_direction: Vec3::NEG_Z,
    };

    /// The current rendered loudness, used to rank arrivals when more are live
    /// than the caller's output buffer can hold.
    #[inline]
    fn loudness(&self) -> Sample {
        self.current.effective_bands().broadband_rms()
    }
}

/// A stateful smoother that turns a backend's per-query [`PropagationPath`] sets
/// into a time-coherent stream for the real-time voice.
///
/// Create one per source voice, then call [`PathSmoother::update`] once per
/// control-rate tick with the backend's latest paths, the elapsed time since the
/// previous tick, and an output buffer. The output holds the smoothed arrivals,
/// loudest first, ready to drive the per-path delay + gain + filter chain.
#[derive(Debug, Clone)]
pub struct PathSmoother {
    times: SmoothingTimes,
    slots: [Slot; TRACKED_PATH_CAPACITY],
}

impl PathSmoother {
    /// Creates a smoother with the given glide time constants and no tracked
    /// arrivals yet.
    #[must_use]
    pub fn new(times: SmoothingTimes) -> Self {
        Self {
            times,
            slots: [Slot::EMPTY; TRACKED_PATH_CAPACITY],
        }
    }

    /// Returns the glide time constants in effect.
    #[inline]
    #[must_use]
    pub fn times(&self) -> SmoothingTimes {
        self.times
    }

    /// Replaces the glide time constants; takes effect on the next update.
    #[inline]
    pub fn set_times(&mut self, times: SmoothingTimes) {
        self.times = times;
    }

    /// Forgets all tracked arrivals, so the next update starts every arrival
    /// from a fresh fade-in. Call this on a teleport or scene cut where easing
    /// across the discontinuity would be wrong.
    pub fn reset(&mut self) {
        self.slots = [Slot::EMPTY; TRACKED_PATH_CAPACITY];
    }

    /// The number of arrivals currently tracked (live or fading out).
    #[inline]
    #[must_use]
    pub fn tracked_len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.active).count()
    }

    /// Eases the tracked arrivals one control-rate step toward `targets`
    /// (the backend's freshest paths), advancing by `dt_seconds` of wall-clock
    /// time, and writes the smoothed arrivals into `out` loudest first.
    ///
    /// Returns how many entries of `out` were populated (never more than
    /// `out.len()`). Arrivals present in `targets` but absent before fade in
    /// from silence; arrivals absent from `targets` but tracked before fade out
    /// and are retired once inaudible. When more arrivals are live than `out`
    /// can hold, the quietest are omitted from the output (but stay tracked, so
    /// they can return to the output as they swell).
    pub fn update(
        &mut self,
        targets: &[PropagationPath],
        dt_seconds: Sample,
        out: &mut [PropagationPath],
    ) -> usize {
        let dt = dt_seconds.max(0.0);

        for slot in &mut self.slots {
            slot.matched = false;
        }

        for (index, target) in targets.iter().enumerate() {
            let ordinal = targets[..index]
                .iter()
                .filter(|earlier| earlier.kind == target.kind)
                .count();
            self.assign_target(target, ordinal);
        }

        self.retire_unmatched();
        self.advance(dt);
        self.collect(out)
    }

    /// Routes a backend target to its tracked slot, continuing an existing
    /// arrival with the same `(kind, ordinal)` key or being born into a free
    /// slot (fading in from silence).
    fn assign_target(&mut self, target: &PropagationPath, ordinal: usize) {
        if let Some(slot) = self.slots.iter_mut().find(|slot| {
            slot.active && !slot.matched && slot.kind == target.kind && slot.ordinal == ordinal
        }) {
            set_targets(slot, target);
            slot.matched = true;
            slot.dying = false;
            return;
        }

        let Some(index) = self.free_slot_index() else {
            return;
        };
        let slot = &mut self.slots[index];
        slot.active = true;
        slot.matched = true;
        slot.dying = false;
        slot.kind = target.kind;
        slot.ordinal = ordinal;
        // Born in place: correct delay/colour/direction immediately, but silent,
        // so the arrival swells up rather than sweeping its delay in from zero.
        slot.current = *target;
        slot.current.gain = 0.0;
        set_targets(slot, target);
    }

    /// Finds a slot for a new arrival: a truly free one if any, otherwise the
    /// quietest fading-out slot (reclaimed because an audible fresh arrival
    /// matters more than finishing an already-inaudible fade). Returns `None`
    /// only when every slot holds a live, non-dying arrival.
    fn free_slot_index(&self) -> Option<usize> {
        if let Some(index) = self.slots.iter().position(|slot| !slot.active) {
            return Some(index);
        }
        let mut best: Option<(usize, Sample)> = None;
        for (index, slot) in self.slots.iter().enumerate() {
            if !slot.dying {
                continue;
            }
            let loudness = slot.loudness();
            match best {
                Some((_, best_loudness)) if best_loudness <= loudness => {}
                _ => best = Some((index, loudness)),
            }
        }
        best.map(|(index, _)| index)
    }

    /// Marks every active slot that no incoming target matched as fading out:
    /// its target gain drops to zero while its other targets hold, so it decays
    /// in place.
    fn retire_unmatched(&mut self) {
        for slot in &mut self.slots {
            if slot.active && !slot.matched {
                slot.dying = true;
                slot.target_gain = 0.0;
            }
        }
    }

    /// Eases every active slot one step toward its targets and retires any
    /// fading-out slot that has become inaudible.
    fn advance(&mut self, dt: Sample) {
        let gain_alpha = one_pole_alpha(dt, self.times.gain_seconds);
        let delay_alpha = one_pole_alpha(dt, self.times.delay_seconds);
        let spectrum_alpha = one_pole_alpha(dt, self.times.spectrum_seconds);
        let direction_alpha = one_pole_alpha(dt, self.times.direction_seconds);

        for slot in &mut self.slots {
            if !slot.active {
                continue;
            }

            let path = &mut slot.current;
            path.kind = slot.kind;
            path.gain = ease(path.gain, slot.target_gain, gain_alpha).clamp(0.0, 1.0);
            path.delay_seconds = ease(path.delay_seconds, slot.target_delay, delay_alpha).max(0.0);
            path.cutoff_hz = ease_cutoff(path.cutoff_hz, slot.target_cutoff_hz, spectrum_alpha);
            path.bands = ease_bands(path.bands, slot.target_bands, spectrum_alpha);
            path.direction = ease_direction(path.direction, slot.target_direction, direction_alpha);

            if slot.dying && path.gain <= RETIRE_GAIN_EPSILON {
                *slot = Slot::EMPTY;
            }
        }
    }

    /// Writes the live arrivals into `out`, loudest first, capped at
    /// `out.len()`, and returns the count written. Ties break deterministically
    /// on `(kind, ordinal)` so the output order is reproducible.
    fn collect(&self, out: &mut [PropagationPath]) -> usize {
        let capacity = out.len();
        if capacity == 0 {
            return 0;
        }

        let mut written = 0;
        let mut taken = [false; TRACKED_PATH_CAPACITY];

        while written < capacity {
            let mut best: Option<usize> = None;
            for (index, slot) in self.slots.iter().enumerate() {
                if !slot.active || taken[index] {
                    continue;
                }
                best = Some(match best {
                    Some(current) if !is_louder(slot, &self.slots[current]) => current,
                    _ => index,
                });
            }
            let Some(index) = best else {
                break;
            };
            taken[index] = true;
            out[written] = self.slots[index].current;
            written += 1;
        }

        written
    }
}

impl Default for PathSmoother {
    #[inline]
    fn default() -> Self {
        Self::new(SmoothingTimes::DEFAULT)
    }
}

/// Copies a backend target's parameters into a slot's target fields, leaving the
/// slot's smoothed `current` state untouched so the glide eases toward them.
fn set_targets(slot: &mut Slot, target: &PropagationPath) {
    slot.target_gain = target.gain.clamp(0.0, 1.0);
    slot.target_delay = target.delay_seconds.max(0.0);
    slot.target_cutoff_hz = target.cutoff_hz;
    slot.target_bands = target.bands;
    slot.target_direction = target.direction;
}

/// Whether `a` is strictly louder than `b`, breaking exact-loudness ties on the
/// correspondence key so ranking is deterministic across targets.
fn is_louder(a: &Slot, b: &Slot) -> bool {
    let (la, lb) = (a.loudness(), b.loudness());
    if la != lb {
        return la > lb;
    }
    (kind_rank(a.kind), a.ordinal) < (kind_rank(b.kind), b.ordinal)
}

/// A stable ordering rank for each [`PathKind`], used only to break loudness
/// ties deterministically.
#[inline]
fn kind_rank(kind: PathKind) -> u8 {
    match kind {
        PathKind::Direct => 0,
        PathKind::Transmission => 1,
        PathKind::Reflection => 2,
        PathKind::Diffraction => 3,
    }
}

/// The one-pole smoothing coefficient for a step of `dt` seconds toward a target
/// with time constant `tau` seconds: `1 - exp(-dt / tau)`, clamped to `[0, 1]`.
/// A non-positive `tau` (or a `dt` long enough to converge) yields `1.0`, i.e.
/// snap to the target. The exponential routes through [`bevy_math::ops`] so the
/// glide is bit-reproducible across targets.
#[must_use]
fn one_pole_alpha(dt: Sample, tau: Sample) -> Sample {
    if tau <= 0.0 || dt <= 0.0 {
        return if dt <= 0.0 && tau > 0.0 { 0.0 } else { 1.0 };
    }
    (1.0 - ops::exp(-dt / tau)).clamp(0.0, 1.0)
}

/// Eases `current` toward `target` by `alpha`: `current + (target - current) *
/// alpha`.
#[inline]
#[must_use]
fn ease(current: Sample, target: Sample, alpha: Sample) -> Sample {
    current + (target - current) * alpha
}

/// Eases a filter corner in the log-frequency domain, where a halving of
/// frequency is a constant perceptual step, so a glide from "unfiltered" down to
/// a low corner sounds like a smooth darkening rather than a late collapse. Both
/// endpoints are clamped to `[MIN_CUTOFF_HZ, FULL_BAND_CUTOFF_HZ]` so the
/// logarithm stays finite.
#[must_use]
fn ease_cutoff(current_hz: Sample, target_hz: Sample, alpha: Sample) -> Sample {
    let current = clamp_cutoff(current_hz);
    let target = clamp_cutoff(target_hz);
    let eased = ease(ops::ln(current), ops::ln(target), alpha);
    clamp_cutoff(ops::exp(eased))
}

/// Clamps a filter corner into the representable log-domain range.
#[inline]
#[must_use]
fn clamp_cutoff(hz: Sample) -> Sample {
    hz.clamp(MIN_CUTOFF_HZ, FULL_BAND_CUTOFF_HZ)
}

/// Eases each propagation band toward its target gain independently, keeping the
/// result in `[0, 1]`.
#[must_use]
fn ease_bands(current: BandGains, target: BandGains, alpha: Sample) -> BandGains {
    let mut bands = [0.0; PROPAGATION_BAND_COUNT];
    for (band, slot) in bands.iter_mut().enumerate() {
        *slot = ease(current.band(band), target.band(band), alpha);
    }
    BandGains::new(bands)
}

/// Eases a listener-local arrival direction toward its target by renormalised
/// linear interpolation (a cheap, deterministic stand-in for a great-circle
/// slerp that is accurate for the small per-step angles a control-rate glide
/// produces). Degenerate interpolants fall back to the (normalised) target, then
/// the current direction, so the result is always a finite unit vector.
#[must_use]
fn ease_direction(current: Vec3, target: Vec3, alpha: Sample) -> Vec3 {
    let blended = current + (target - current) * alpha;
    normalize(blended)
        .or_else(|| normalize(target))
        .or_else(|| normalize(current))
        .unwrap_or(Vec3::NEG_Z)
}

/// Normalises `v`, returning `None` when it is non-finite or shorter than
/// [`DIRECTION_EPSILON_SQ`]. The length routes through [`bevy_math::ops`] so the
/// result is bit-reproducible across targets.
fn normalize(v: Vec3) -> Option<Vec3> {
    let len_sq = v.dot(v);
    if !len_sq.is_finite() || len_sq <= DIRECTION_EPSILON_SQ {
        return None;
    }
    Some(v / ops::sqrt(len_sq))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(kind: PathKind, gain: Sample, delay: Sample, direction: Vec3) -> PropagationPath {
        PropagationPath {
            kind,
            delay_seconds: delay,
            gain,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            bands: BandGains::UNITY,
            direction,
        }
    }

    #[test]
    fn instant_times_pass_targets_through_unchanged() {
        let mut smoother = PathSmoother::new(SmoothingTimes::INSTANT);
        let targets = [path(PathKind::Direct, 0.8, 0.01, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let count = smoother.update(&targets, 1.0 / 60.0, &mut out);
        assert_eq!(count, 1);
        assert!((out[0].gain - 0.8).abs() < 1.0e-6);
        assert!((out[0].delay_seconds - 0.01).abs() < 1.0e-6);
    }

    #[test]
    fn a_new_arrival_fades_in_from_silence() {
        let mut smoother = PathSmoother::default();
        let targets = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];

        // First tick: the arrival is tracked but still rising from zero.
        let count = smoother.update(&targets, 1.0 / 60.0, &mut out);
        assert_eq!(count, 1);
        assert!(out[0].gain > 0.0 && out[0].gain < 1.0);
        // Its delay, however, is already correct (born in place, not swept).
        assert!((out[0].delay_seconds - 0.01).abs() < 1.0e-6);
    }

    #[test]
    fn a_steady_scene_converges_onto_the_target() {
        let mut smoother = PathSmoother::default();
        let targets = [path(PathKind::Direct, 0.5, 0.02, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        for _ in 0..1_000 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
        }
        assert!((out[0].gain - 0.5).abs() < 1.0e-4);
        assert!((out[0].delay_seconds - 0.02).abs() < 1.0e-4);
    }

    #[test]
    fn a_vanished_arrival_fades_out_and_is_retired() {
        let mut smoother = PathSmoother::default();
        let present = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        // Settle the arrival.
        for _ in 0..1_000 {
            smoother.update(&present, 1.0 / 60.0, &mut out);
        }
        assert_eq!(smoother.tracked_len(), 1);
        // It vanishes: no targets at all now.
        let gone: [PropagationPath; 0] = [];
        let mut last = 1.0;
        for _ in 0..1_000 {
            let count = smoother.update(&gone, 1.0 / 60.0, &mut out);
            if count == 0 {
                break;
            }
            // Monotonically decaying while it fades.
            assert!(out[0].gain <= last + 1.0e-6);
            last = out[0].gain;
        }
        assert_eq!(smoother.tracked_len(), 0);
    }

    #[test]
    fn gain_moves_monotonically_toward_a_louder_target() {
        let mut smoother = PathSmoother::default();
        let targets = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let mut previous = 0.0;
        for _ in 0..20 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
            assert!(out[0].gain >= previous - 1.0e-9);
            previous = out[0].gain;
        }
    }

    #[test]
    fn distinct_kinds_track_independently() {
        let mut smoother = PathSmoother::default();
        let targets = [
            path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z),
            path(PathKind::Reflection, 0.5, 0.03, Vec3::X),
        ];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        for _ in 0..1_000 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
        }
        assert_eq!(smoother.tracked_len(), 2);
        // Loudest first: the direct path outranks the quieter reflection.
        assert_eq!(out[0].kind, PathKind::Direct);
        assert_eq!(out[1].kind, PathKind::Reflection);
        assert!((out[1].gain - 0.5).abs() < 1.0e-4);
    }

    #[test]
    fn output_is_capped_to_the_buffer_keeping_the_loudest() {
        let mut smoother = PathSmoother::default();
        let targets = [
            path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z),
            path(PathKind::Reflection, 0.2, 0.03, Vec3::X),
        ];
        for _ in 0..1_000 {
            let mut scratch = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
            smoother.update(&targets, 1.0 / 60.0, &mut scratch);
        }
        let mut one = [PropagationPath::SILENT; 1];
        let count = smoother.update(&targets, 1.0 / 60.0, &mut one);
        assert_eq!(count, 1);
        // The single slot keeps the louder direct arrival.
        assert_eq!(one[0].kind, PathKind::Direct);
    }

    #[test]
    fn delay_eases_without_overshoot() {
        let mut smoother = PathSmoother::default();
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        // Settle at a short delay.
        let near = [path(PathKind::Direct, 1.0, 0.005, Vec3::NEG_Z)];
        for _ in 0..1_000 {
            smoother.update(&near, 1.0 / 60.0, &mut out);
        }
        // Jump the target far away; the delay should rise monotonically toward
        // it and never exceed it.
        let far = [path(PathKind::Direct, 1.0, 0.080, Vec3::NEG_Z)];
        let mut previous = out[0].delay_seconds;
        for _ in 0..5 {
            smoother.update(&far, 1.0 / 60.0, &mut out);
            assert!(out[0].delay_seconds >= previous - 1.0e-9);
            assert!(out[0].delay_seconds <= 0.080 + 1.0e-6);
            previous = out[0].delay_seconds;
        }
    }

    #[test]
    fn direction_glide_stays_a_unit_vector() {
        let mut smoother = PathSmoother::default();
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let front = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        smoother.update(&front, 1.0 / 60.0, &mut out);
        // Flip the target direction and glide; the rendered direction must stay
        // normalised at every step.
        let behind = [path(PathKind::Direct, 1.0, 0.01, Vec3::Z)];
        for _ in 0..200 {
            smoother.update(&behind, 1.0 / 60.0, &mut out);
            assert!((out[0].direction.length() - 1.0).abs() < 1.0e-5);
        }
        // And ends up pointing at the new target.
        assert!(out[0].direction.dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn cutoff_eases_in_the_log_domain_without_leaving_the_valid_range() {
        let mut smoother = PathSmoother::default();
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let open = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        smoother.update(&open, 1.0 / 60.0, &mut out);
        // Target a dark, low corner.
        let mut dark = path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z);
        dark.cutoff_hz = 500.0;
        let targets = [dark];
        for _ in 0..5 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
            assert!(out[0].cutoff_hz >= MIN_CUTOFF_HZ);
            assert!(out[0].cutoff_hz <= FULL_BAND_CUTOFF_HZ);
        }
        // Converges toward the target corner.
        for _ in 0..1_000 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
        }
        assert!((out[0].cutoff_hz - 500.0).abs() < 1.0);
    }

    #[test]
    fn reset_forgets_all_tracked_arrivals() {
        let mut smoother = PathSmoother::default();
        let targets = [path(PathKind::Direct, 1.0, 0.01, Vec3::NEG_Z)];
        let mut out = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        for _ in 0..1_000 {
            smoother.update(&targets, 1.0 / 60.0, &mut out);
        }
        assert_eq!(smoother.tracked_len(), 1);
        smoother.reset();
        assert_eq!(smoother.tracked_len(), 0);
        // After reset the arrival is reborn from silence.
        let count = smoother.update(&targets, 1.0 / 60.0, &mut out);
        assert_eq!(count, 1);
        assert!(out[0].gain < 1.0);
    }
}
