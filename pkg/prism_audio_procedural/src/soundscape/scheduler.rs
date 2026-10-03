//! The seeded soundscape scheduler: scatters palette one-shots round the ear.
//!
//! This is the heart of section 37. Each block it walks the active palette and,
//! for every element, advances a per-element clock by the block duration. Every
//! time an element's interval elapses it draws one scheduling slot: a seeded
//! coin flip (weighted by the element's day/night curve and the global density)
//! decides whether the slot sounds, and if it does the one-shot is placed at a
//! uniformly random point on the disc around the listener, with a random pitch
//! and gain inside the element's ranges. A pluggable [`SpatialFilter`] then
//! vetoes or attenuates placements for geometry, occlusion, and room ownership
//! (so outdoor birds never leak into a cave), and both a per-element and a
//! global, externally governable concurrency cap bound the voice count. The
//! scheduler never plays a looping bed: it emits a stream of discrete
//! [`ScatteredOneShot`] descriptors the host turns into real voices. Because
//! every random choice comes from the crate's seeded generator, an identical
//! seed and an identical state/block sequence reproduce the scatter exactly.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The phase-clock
//! scheduling and disc scattering are standard, publicly documented techniques.
//!
//! # Relationship
//! Implements the procedural scheduler of design section 37; walks a
//! [`crate::soundscape::palette::SoundscapePalette`] under a
//! [`crate::soundscape::state::SoundscapeState`], draws from [`crate::rng`], and
//! emits [`ScatteredOneShot`]s for the host to spawn. The concurrency cap is the
//! governance hook the quality governor drives.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::ops;

use crate::rng::ProceduralRng;
use crate::soundscape::element::SoundscapeElement;
use crate::soundscape::palette::SoundscapePalette;
use crate::soundscape::state::SoundscapeState;
use prism_audio_core::math::Sample;

/// Upper bound on one-shots a single element may emit in one block.
///
/// This guards the inner scheduling loop against pathologically small
/// intervals so a block can never spin unbounded; it is far above any musically
/// sensible scatter density.
const MAX_EVENTS_PER_ELEMENT: u32 = 256;

/// A spatial descriptor of one scheduled ambience one-shot.
///
/// The scheduler does not render audio; it emits these so the host can spawn an
/// actual voice (for example a granular or sampled ambience sound) at the given
/// world position, pitch, and gain, aligned to the sample offset inside the
/// block.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatteredOneShot {
    /// The palette element that produced this one-shot.
    pub element_id: u32,
    /// World-space position `[x, y, z]` the one-shot should play at.
    pub position: [Sample; 3],
    /// Pitch offset in semitones inside the element's range.
    pub pitch_semitones: Sample,
    /// Linear gain inside the element's range, after spatial attenuation.
    pub gain: Sample,
    /// Sample offset inside the current block the one-shot starts at.
    pub sample_offset: u32,
}

/// A hook for vetoing or attenuating scattered one-shots by scene geometry.
///
/// The scheduler calls [`SpatialFilter::accept`] with each candidate world
/// position. Returning `None` rejects the placement (occluded, in the wrong
/// room, out of range); returning `Some(gain)` accepts it and multiplies the
/// one-shot's gain by `gain` (for distance or obstruction attenuation). The
/// default [`NullSpatialFilter`] accepts everything at unit gain.
pub trait SpatialFilter {
    /// Decides whether a one-shot at `position` may sound, and with what gain
    /// multiplier.
    fn accept(&self, position: [Sample; 3]) -> Option<Sample>;
}

/// A [`SpatialFilter`] that accepts every placement at unit gain.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullSpatialFilter;

impl SpatialFilter for NullSpatialFilter {
    #[inline]
    fn accept(&self, _position: [Sample; 3]) -> Option<Sample> {
        Some(1.0)
    }
}

/// Per-element scheduling state carried across blocks.
#[derive(Clone, Copy, Debug, Default)]
struct ElementClock {
    time_to_next_s: Sample,
    active_estimate: Sample,
    primed: bool,
}

/// The deterministic procedural soundscape scheduler.
#[derive(Clone, Debug)]
pub struct SoundscapeScheduler {
    rng: ProceduralRng,
    clocks: Vec<ElementClock>,
    pending: Vec<ScatteredOneShot>,
    listener: [Sample; 3],
    sample_rate: u32,
    max_concurrent: u32,
}

impl SoundscapeScheduler {
    /// Creates a scheduler seeded with `seed` at `sample_rate`, preallocating
    /// room for `element_capacity` element clocks and a matching one-shot
    /// buffer so the steady-state render path does not allocate.
    #[must_use]
    pub fn new(sample_rate: u32, element_capacity: usize, seed: u64) -> Self {
        Self {
            rng: ProceduralRng::new(seed),
            clocks: Vec::with_capacity(element_capacity),
            pending: Vec::with_capacity(element_capacity.saturating_mul(4).max(16)),
            listener: [0.0, 0.0, 0.0],
            sample_rate: sample_rate.max(1),
            max_concurrent: 64,
        }
    }

    /// Sets the global concurrency cap, the governance hook an external quality
    /// governor drives to trade density for budget. Floored at `1`.
    #[inline]
    pub fn set_max_concurrent(&mut self, cap: u32) {
        self.max_concurrent = cap.max(1);
    }

    /// Returns the current global concurrency cap.
    #[inline]
    #[must_use]
    pub fn max_concurrent(&self) -> u32 {
        self.max_concurrent
    }

    /// Sets the listener world position one-shots are scattered around.
    #[inline]
    pub fn set_listener(&mut self, position: [Sample; 3]) {
        self.listener = position;
    }

    /// Returns the listener world position.
    #[inline]
    #[must_use]
    pub fn listener(&self) -> [Sample; 3] {
        self.listener
    }

    /// Returns the estimated number of concurrently sounding one-shots across
    /// all elements.
    #[inline]
    #[must_use]
    pub fn active_estimate(&self) -> Sample {
        self.clocks.iter().map(|c| c.active_estimate).sum()
    }

    /// Advances the scheduler by `frames` samples under the given palette and
    /// state, returning the one-shots scheduled in this block.
    ///
    /// The returned slice is owned by the scheduler and valid until the next
    /// call. Placements are filtered by `filter`; pass [`NullSpatialFilter`]
    /// for no filtering. The call is deterministic: identical seed, palette,
    /// and state/`frames` sequence reproduce the identical one-shot stream.
    pub fn process(
        &mut self,
        palette: &SoundscapePalette,
        state: SoundscapeState,
        frames: usize,
        filter: &dyn SpatialFilter,
    ) -> &[ScatteredOneShot] {
        self.pending.clear();
        let elements = palette.elements();
        self.sync_clocks(elements.len());
        if frames == 0 || elements.is_empty() {
            return &self.pending;
        }

        let fs = self.sample_rate as Sample;
        let dt = frames as Sample / fs;
        let daylight = state.daylight();
        let density = state.density();
        let indoor = state.is_indoor();
        let max_total = self.max_concurrent as Sample;

        // Running total of active voices, updated as placements are emitted so
        // the global cap accounts for one-shots fired earlier in this block.
        let mut global_active: Sample = self.clocks.iter().map(|c| c.active_estimate).sum();

        for (element, clock) in elements.iter().zip(self.clocks.iter_mut()) {
            Self::decay_clock(clock, element, dt, fs);
            let trigger = element.effective_trigger_prob(daylight, density);
            let permitted = element.permitted_indoors(indoor);
            let per_cap = element.concurrency_cap() as Sample;
            let (plo, phi) = element.pitch_range();
            let (glo, ghi) = element.gain_range();

            let mut consumed = 0.0;
            let mut events = 0;
            while events < MAX_EVENTS_PER_ELEMENT {
                let remaining = dt - consumed;
                if clock.time_to_next_s > remaining {
                    clock.time_to_next_s -= remaining;
                    break;
                }
                consumed += clock.time_to_next_s;
                events += 1;

                let offset = Self::offset_of(consumed, fs, frames);
                let fired = self.rng.chance(trigger);
                if fired {
                    let (dx, dz) = self.rng.next_in_disc(element.scatter_radius_m());
                    let pitch = self.rng.next_range(plo, phi);
                    let base_gain = self.rng.next_range(glo, ghi);
                    let under_cap = clock.active_estimate + 1.0 <= per_cap
                        && global_active + 1.0 <= max_total;
                    if permitted && under_cap {
                        let position = [
                            self.listener[0] + dx,
                            self.listener[1],
                            self.listener[2] + dz,
                        ];
                        if let Some(spatial_gain) = filter.accept(position) {
                            self.pending.push(ScatteredOneShot {
                                element_id: element.id(),
                                position,
                                pitch_semitones: pitch,
                                gain: base_gain * spatial_gain.max(0.0),
                                sample_offset: offset,
                            });
                            clock.active_estimate += 1.0;
                            global_active += 1.0;
                        }
                    }
                }

                clock.time_to_next_s = Self::draw_interval(&mut self.rng, element);
            }
        }

        &self.pending
    }

    /// Silences all scheduling state (clocks and the pending buffer) while
    /// keeping the random stream position, so a fresh block starts cleanly.
    pub fn reset(&mut self) {
        for clock in &mut self.clocks {
            *clock = ElementClock::default();
        }
        self.pending.clear();
    }

    /// Resizes the per-element clock list to match the palette, priming new
    /// clocks lazily so a freshly added element does not all fire at `t = 0`.
    fn sync_clocks(&mut self, len: usize) {
        if self.clocks.len() > len {
            self.clocks.truncate(len);
        } else {
            while self.clocks.len() < len {
                self.clocks.push(ElementClock::default());
            }
        }
    }

    /// Decays an element's active-voice estimate for the elapsed block and
    /// primes its first interval if it has not scheduled yet.
    fn decay_clock(clock: &mut ElementClock, element: &SoundscapeElement, dt: Sample, fs: Sample) {
        if !clock.primed {
            // Schedule the first event one mean interval out so a freshly added
            // element does not fire at t = 0; elements with distinct means then
            // naturally desynchronise instead of bursting together.
            clock.time_to_next_s = element.mean_interval_s();
            clock.primed = true;
        }
        let duration_samples = element.nominal_duration_s() * fs;
        if duration_samples > 0.0 {
            let decay = (dt * fs) / duration_samples;
            clock.active_estimate = (clock.active_estimate - decay).max(0.0);
        }
    }

    /// Draws the next interval in seconds for an element, honouring its jitter.
    fn draw_interval(rng: &mut ProceduralRng, element: &SoundscapeElement) -> Sample {
        let mean = element.mean_interval_s();
        let jitter = element.interval_jitter();
        let scale = 1.0 + rng.next_bipolar() * jitter;
        (mean * scale).max(mean * 0.01)
    }

    /// Converts a time offset in seconds into a clamped in-block sample offset.
    fn offset_of(seconds: Sample, fs: Sample, frames: usize) -> u32 {
        let raw = ops::floor(seconds * fs);
        let max = frames.saturating_sub(1) as Sample;
        let clamped = raw.clamp(0.0, max);
        clamped as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundscape::element::SoundscapeElement;

    fn busy_palette() -> SoundscapePalette {
        SoundscapePalette::new().with_element(
            SoundscapeElement::new(1)
                .with_interval(0.01, 0.0)
                .with_trigger_prob(1.0)
                .with_scatter_radius(10.0)
                .with_pitch_range(-3.0, 3.0)
                .with_gain_range(0.5, 1.0)
                .with_concurrency_cap(1000)
                .with_nominal_duration(0.05),
        )
    }

    #[test]
    fn emits_one_shots() {
        let palette = busy_palette();
        let mut sched = SoundscapeScheduler::new(48_000, 4, 1);
        sched.set_max_concurrent(1000);
        let state = SoundscapeState::outdoor_noon();
        let filter = NullSpatialFilter;
        // Advance a few blocks so clocks pass their first interval.
        let mut total = 0;
        for _ in 0..20 {
            total += sched.process(&palette, state, 1_024, &filter).len();
        }
        assert!(total > 0, "expected scattered one-shots, got {total}");
    }

    #[test]
    fn deterministic_for_same_seed_and_state() {
        let palette = busy_palette();
        let state = SoundscapeState::outdoor_noon();
        let run = || {
            let mut sched = SoundscapeScheduler::new(48_000, 4, 7);
            sched.set_max_concurrent(1000);
            let filter = NullSpatialFilter;
            let mut shots = Vec::new();
            for _ in 0..16 {
                shots.extend_from_slice(sched.process(&palette, state, 512, &filter));
            }
            shots
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn global_cap_bounds_concurrency() {
        let palette = busy_palette();
        let mut sched = SoundscapeScheduler::new(48_000, 4, 3);
        sched.set_max_concurrent(5);
        let state = SoundscapeState::outdoor_noon();
        let filter = NullSpatialFilter;
        for _ in 0..64 {
            sched.process(&palette, state, 1_024, &filter);
            assert!(
                sched.active_estimate() <= 5.0 + 1e-3,
                "active={}",
                sched.active_estimate()
            );
        }
    }

    #[test]
    fn indoor_mutes_outdoor_only_element() {
        let palette = SoundscapePalette::new().with_element(
            SoundscapeElement::new(9)
                .with_interval(0.01, 0.0)
                .with_trigger_prob(1.0)
                .with_concurrency_cap(1000)
                .with_environments(false, true),
        );
        let mut sched = SoundscapeScheduler::new(48_000, 2, 5);
        sched.set_max_concurrent(1000);
        let indoor = SoundscapeState::new(0.5, 1.0, true);
        let filter = NullSpatialFilter;
        let mut total = 0;
        for _ in 0..20 {
            total += sched.process(&palette, indoor, 1_024, &filter).len();
        }
        assert_eq!(total, 0, "outdoor element should be muted indoors");
    }

    #[test]
    fn spatial_filter_can_reject() {
        struct RejectAll;
        impl SpatialFilter for RejectAll {
            fn accept(&self, _position: [Sample; 3]) -> Option<Sample> {
                None
            }
        }
        let palette = busy_palette();
        let mut sched = SoundscapeScheduler::new(48_000, 2, 11);
        sched.set_max_concurrent(1000);
        let state = SoundscapeState::outdoor_noon();
        let filter = RejectAll;
        let mut total = 0;
        for _ in 0..20 {
            total += sched.process(&palette, state, 1_024, &filter).len();
        }
        assert_eq!(total, 0);
    }

    #[test]
    fn scatter_stays_within_radius() {
        let palette = busy_palette();
        let mut sched = SoundscapeScheduler::new(48_000, 2, 2);
        sched.set_max_concurrent(1000);
        sched.set_listener([5.0, 1.0, -2.0]);
        let state = SoundscapeState::outdoor_noon();
        let filter = NullSpatialFilter;
        for _ in 0..16 {
            for shot in sched.process(&palette, state, 1_024, &filter) {
                let dx = shot.position[0] - 5.0;
                let dz = shot.position[2] + 2.0;
                assert!(ops::sqrt(dx * dx + dz * dz) <= 10.0 + 1e-3);
                assert_eq!(shot.position[1], 1.0);
            }
        }
    }
}
