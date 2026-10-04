//! Entity LOD / update-frequency banding (design §13.2).
//!
//! In a large world most entities are far from the camera and do not need to
//! simulate every frame. Unreal's MassEntity, for instance, sorts agents into
//! *LOD bands* by distance and ticks the distant ones at a reduced cadence (or
//! collapses them to a cheap proxy), so the per-frame cost scales with the
//! number of **nearby** entities rather than the total population.
//!
//! This module is the pure, `World`-independent CPU half of that scheme. A
//! [`LodSchedule`] holds an ordered list of [`LodBand`]s (nearest first); given
//! an entity's *squared* distance to the viewer it answers two questions:
//!
//! * **Which detail level** does the entity belong to right now
//!   ([`LodSchedule::level_for`])? Level `0` is the highest detail / full rate;
//!   higher levels are progressively coarser and cheaper.
//! * **Should it tick this frame** ([`LodSchedule::should_tick`])? Each band
//!   carries a `tick_period`: a band with period `N` only updates on frames
//!   that are multiples of `N`, so period `1` is every frame, period `4` is
//!   once every four frames, and so on.
//!
//! All distance comparisons use *squared* metres so the hot path needs no
//! `sqrt` — this mirrors the `no_std`, float-intrinsic-free style of the
//! sibling [`floating_origin`](super::floating_origin) module, and pairs
//! naturally with its rebased [`LocalPos`](super::floating_origin::LocalPos):
//! feed [`distance_sq`] two rebased positions and route the result straight
//! into [`LodSchedule::evaluate`].
//!
//! # Boundary convention
//!
//! A band is active for every squared distance **up to and including** its
//! `max_distance_sq` (the upper edge is *inclusive*). Bands are tested nearest
//! first, so the first band whose `max_distance_sq` is `>=` the queried
//! distance wins. An entity beyond the last band is handled by the schedule's
//! [`OutOfRange`] policy.
//!
//! # Example
//!
//! ```ignore
//! use prism_ecs::partition::lod::*;
//! let sched = LodSchedule::from_sorted_pairs(&[
//!     (100.0, 1),      // < 10 m : full rate
//!     (2_500.0, 4),    // < 50 m : every 4th frame
//!     (40_000.0, 16),  // < 200 m: every 16th frame
//! ]);
//! let d = distance_sq([0.0, 0.0, 0.0], [30.0, 0.0, 0.0]); // 900
//! let decision = sched.evaluate(d, 8);
//! assert_eq!(decision.level, Some(LodLevel(1)));
//! assert!(decision.tick_this_frame); // 8 % 4 == 0
//! ```

use alloc::vec::Vec;

/// A level-of-detail band index for an entity (design §13.2).
///
/// `0` is the highest detail / full-rate level, assigned to the nearest
/// entities; each increment is a coarser, cheaper band further from the
/// viewer. Ordering follows the numeric value, so `LodLevel(0) < LodLevel(1)`
/// means "more detailed than". The type is small and `Copy` so it can double
/// as a per-entity [`Component`](crate::component::Component) tag.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct LodLevel(pub u8);

impl LodLevel {
    /// The highest-detail / full-rate level (`LodLevel(0)`), assigned to the
    /// nearest entities.
    pub const HIGHEST: Self = Self(0);
}

impl crate::component::Component for LodLevel {}

/// A per-band *simulation precision / quality tier* for an entity
/// (design §23.7 分档精度).
///
/// §23.7 bands scale updates along **two independent axes**: *frequency*
/// (how often an entity ticks, carried by [`LodBand::tick_period`]) and
/// *precision* (how much work each tick does — e.g. full IK + perception vs. a
/// cheap positional approximation). This type is that second axis, kept
/// deliberately separate from both the cadence and the band index so content
/// can decouple them: two distance bands may share one quality tier, or a
/// full-rate band may still run reduced-precision work (e.g. near-but-occluded
/// agents).
///
/// `0` is the highest precision / most expensive tier (the near default); each
/// increment is a coarser, cheaper tier. Ordering follows the numeric value, so
/// `LodQuality(0) < LodQuality(1)` means "more precise than". The type is small
/// and `Copy` so it can double as a per-entity
/// [`Component`](crate::component::Component) tag the owner writes back from a
/// [`LodDecision`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct LodQuality(pub u8);

impl LodQuality {
    /// The highest-precision / most expensive tier (`LodQuality(0)`), the
    /// default for the nearest entities.
    pub const HIGHEST: Self = Self(0);
}

impl crate::component::Component for LodQuality {}

/// One distance band in a [`LodSchedule`] (design §13.2).
///
/// A band is active for entities whose *squared* distance to the viewer is at
/// most `max_distance_sq` (inclusive upper edge) and that fall into no nearer
/// band. Entities in the band tick once every `tick_period` frames.
///
/// Bands are always stored **nearest first** inside a [`LodSchedule`], i.e. by
/// ascending `max_distance_sq`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LodBand {
    /// Inclusive upper bound of the band, as a *squared* distance in metres²
    /// (squared so the hot path never needs `sqrt`).
    pub max_distance_sq: f32,
    /// Update cadence: entities in this band tick once every `tick_period`
    /// frames. `1` means every frame; must be `>= 1`.
    pub tick_period: u32,
    /// Simulation precision tier for entities in this band (design §23.7). The
    /// cadence axis above says *how often* they tick; this says *how much work*
    /// each tick does. Independent of the band index, so precision can be tuned
    /// separately from distance and frequency.
    pub quality: LodQuality,
}

impl LodBand {
    /// Creates a band from its inclusive squared-distance bound and tick
    /// period, at the highest precision tier ([`LodQuality::HIGHEST`]).
    ///
    /// Use [`with_quality`](Self::with_quality) (or [`quality`](Self::quality))
    /// to assign a coarser precision tier to farther bands.
    pub const fn new(max_distance_sq: f32, tick_period: u32) -> Self {
        Self {
            max_distance_sq,
            tick_period,
            quality: LodQuality::HIGHEST,
        }
    }

    /// Creates a band with an explicit precision tier (design §23.7).
    pub const fn with_quality(
        max_distance_sq: f32,
        tick_period: u32,
        quality: LodQuality,
    ) -> Self {
        Self {
            max_distance_sq,
            tick_period,
            quality,
        }
    }

    /// Returns this band with its precision tier replaced (chainable builder).
    pub const fn quality(mut self, quality: LodQuality) -> Self {
        self.quality = quality;
        self
    }
}

/// What to do with an entity that lies beyond the last (farthest) band of a
/// [`LodSchedule`] (design §13.2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum OutOfRange {
    /// Clamp the entity to the last, coarsest band: it keeps simulating at that
    /// band's reduced cadence instead of being dropped. This is the default.
    #[default]
    ClampToLast,
    /// Treat the entity as dormant: [`LodSchedule::level_for`] returns `None`
    /// and the entity never ticks until it moves back into range.
    Dormant,
}

/// The outcome of evaluating a [`LodSchedule`] for one entity on one frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LodDecision {
    /// The selected detail band, or `None` when the entity is dormant (only
    /// possible under the [`OutOfRange::Dormant`] policy, beyond the last
    /// band).
    pub level: Option<LodLevel>,
    /// Whether the entity should run its simulation update on this frame. Always
    /// `false` when `level` is `None`.
    pub tick_this_frame: bool,
    /// The precision tier of the selected band (design §23.7). The owner reads
    /// this to pick which simulation path to run for a ticking entity (e.g.
    /// full vs. approximate). Defaults to [`LodQuality::HIGHEST`] when the
    /// entity is dormant (`level` is `None`).
    pub quality: LodQuality,
}

/// An ordered set of distance bands plus an out-of-range policy that maps an
/// entity's squared distance to a detail level and update cadence
/// (design §13.2).
///
/// Bands are kept **nearest first** (ascending `max_distance_sq`); the band's
/// position in the list *is* its [`LodLevel`]. See the [module
/// docs](self#boundary-convention) for the inclusive-upper-edge convention.
#[derive(Clone, PartialEq, Debug)]
pub struct LodSchedule {
    /// Bands ordered nearest first (ascending `max_distance_sq`).
    bands: Vec<LodBand>,
    /// Behaviour for entities beyond the last band.
    beyond_last: OutOfRange,
}

impl LodSchedule {
    /// Creates a schedule from a nearest-first list of bands, using the default
    /// [`OutOfRange::ClampToLast`] policy.
    ///
    /// # Contract
    /// Panics unless `bands` is non-empty, sorted by ascending
    /// `max_distance_sq`, and every `tick_period` is `>= 1`.
    pub fn new(bands: Vec<LodBand>) -> Self {
        Self::with_out_of_range(bands, OutOfRange::ClampToLast)
    }

    /// Creates a schedule from a nearest-first list of bands and an explicit
    /// out-of-range policy.
    ///
    /// # Contract
    /// Panics unless `bands` is non-empty, sorted by ascending
    /// `max_distance_sq`, and every `tick_period` is `>= 1`.
    pub fn with_out_of_range(bands: Vec<LodBand>, beyond_last: OutOfRange) -> Self {
        assert!(!bands.is_empty(), "LodSchedule needs at least one band");
        let mut prev = f32::NEG_INFINITY;
        for (i, b) in bands.iter().enumerate() {
            assert!(
                b.tick_period >= 1,
                "band {i} tick_period must be >= 1, got {}",
                b.tick_period
            );
            assert!(
                b.max_distance_sq >= prev,
                "bands must be sorted nearest-first by ascending max_distance_sq \
                 (band {i} = {} < previous {prev})",
                b.max_distance_sq
            );
            prev = b.max_distance_sq;
        }
        Self { bands, beyond_last }
    }

    /// Convenience constructor from `(max_distance_sq, tick_period)` pairs,
    /// nearest first. Same contract as [`new`](Self::new).
    pub fn from_sorted_pairs(pairs: &[(f32, u32)]) -> Self {
        let bands = pairs
            .iter()
            .map(|&(max_distance_sq, tick_period)| LodBand::new(max_distance_sq, tick_period))
            .collect();
        Self::new(bands)
    }

    /// Convenience constructor from `(max_distance_sq, tick_period, quality)`
    /// triples, nearest first, assigning each band an explicit precision tier
    /// (design §23.7). Same contract as [`new`](Self::new).
    pub fn from_sorted_triples(triples: &[(f32, u32, LodQuality)]) -> Self {
        let bands = triples
            .iter()
            .map(|&(max_distance_sq, tick_period, quality)| {
                LodBand::with_quality(max_distance_sq, tick_period, quality)
            })
            .collect();
        Self::new(bands)
    }

    /// Returns a copy of this schedule with its out-of-range policy replaced.
    pub fn with_policy(mut self, beyond_last: OutOfRange) -> Self {
        self.beyond_last = beyond_last;
        self
    }

    /// Sets the out-of-range policy in place.
    pub fn set_out_of_range(&mut self, beyond_last: OutOfRange) {
        self.beyond_last = beyond_last;
    }

    /// The current out-of-range policy.
    #[inline]
    pub fn out_of_range(&self) -> OutOfRange {
        self.beyond_last
    }

    /// The number of bands in the schedule (`>= 1`).
    #[inline]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Read-only view of the bands, nearest first.
    #[inline]
    pub fn bands(&self) -> &[LodBand] {
        &self.bands
    }

    /// Returns the detail [`LodLevel`] for the given *squared* distance, or
    /// `None` when the entity is dormant.
    ///
    /// Bands are tested nearest first and the upper edge is inclusive: the
    /// first band whose `max_distance_sq >= distance_sq` is selected. For a
    /// distance beyond the last band the result follows the schedule's
    /// [`OutOfRange`] policy — the last level under
    /// [`ClampToLast`](OutOfRange::ClampToLast), or `None` under
    /// [`Dormant`](OutOfRange::Dormant).
    pub fn level_for(&self, distance_sq: f32) -> Option<LodLevel> {
        for (i, b) in self.bands.iter().enumerate() {
            if distance_sq <= b.max_distance_sq {
                return Some(LodLevel(i as u8));
            }
        }
        match self.beyond_last {
            OutOfRange::ClampToLast => Some(LodLevel((self.bands.len() - 1) as u8)),
            OutOfRange::Dormant => None,
        }
    }

    /// The tick period of a band, clamped to the last band if `level` is out of
    /// range. Always returns `>= 1`.
    pub fn tick_period(&self, level: LodLevel) -> u32 {
        let idx = (level.0 as usize).min(self.bands.len() - 1);
        self.bands[idx].tick_period
    }

    /// The precision tier of a band, clamped to the last band if `level` is out
    /// of range (design §23.7). Pairs with [`tick_period`](Self::tick_period):
    /// one gives the cadence, the other the per-tick precision.
    pub fn quality(&self, level: LodLevel) -> LodQuality {
        let idx = (level.0 as usize).min(self.bands.len() - 1);
        self.bands[idx].quality
    }

    /// Whether a band's entities should tick on `frame`.
    ///
    /// True exactly when `frame` is a multiple of the band's `tick_period`, so
    /// period `1` ticks every frame and period `4` ticks on frames
    /// `0, 4, 8, …`.
    #[inline]
    pub fn should_tick(&self, level: LodLevel, frame: u64) -> bool {
        self.should_tick_phased(level, frame, 0)
    }

    /// Phase-offset variant of [`should_tick`](Self::should_tick): ticks when
    /// `(frame + phase) % tick_period == 0`.
    ///
    /// Giving entities in the same band different `phase` values spreads their
    /// updates across different frames, avoiding a thundering-herd spike where
    /// a whole band ticks together.
    #[inline]
    pub fn should_tick_phased(&self, level: LodLevel, frame: u64, phase: u64) -> bool {
        let period = self.tick_period(level) as u64;
        frame.wrapping_add(phase).is_multiple_of(period)
    }

    /// Full per-entity evaluation: selects the band for `distance_sq` and
    /// decides whether it ticks on `frame`.
    ///
    /// A dormant entity (level `None`) never ticks.
    #[inline]
    pub fn evaluate(&self, distance_sq: f32, frame: u64) -> LodDecision {
        self.evaluate_phased(distance_sq, frame, 0)
    }

    /// Phase-offset variant of [`evaluate`](Self::evaluate), using
    /// [`should_tick_phased`](Self::should_tick_phased) for the cadence test.
    #[inline]
    pub fn evaluate_phased(&self, distance_sq: f32, frame: u64, phase: u64) -> LodDecision {
        match self.level_for(distance_sq) {
            Some(level) => LodDecision {
                level: Some(level),
                tick_this_frame: self.should_tick_phased(level, frame, phase),
                quality: self.quality(level),
            },
            None => LodDecision {
                level: None,
                tick_this_frame: false,
                quality: LodQuality::HIGHEST,
            },
        }
    }
}

/// The squared Euclidean distance between two points, in metres² (no `sqrt`).
///
/// Comparing squared distances is monotonic with the true distance, so LOD
/// banding never needs the real length — keeping the kernel free of `std`
/// float intrinsics. Pairs with rebased
/// [`LocalPos`](super::floating_origin::LocalPos) coordinates.
#[inline]
pub fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> LodSchedule {
        // < 10 m, < 50 m, < 200 m (as squared metres).
        LodSchedule::from_sorted_pairs(&[(100.0, 1), (2_500.0, 4), (40_000.0, 16)])
    }

    #[test]
    fn highest_level_is_zero() {
        assert_eq!(LodLevel::HIGHEST, LodLevel(0));
        assert!(LodLevel(0) < LodLevel(1));
        assert_eq!(LodLevel::default(), LodLevel(0));
    }

    #[test]
    fn level_selection_nearest_first() {
        let s = sched();
        assert_eq!(s.level_for(0.0), Some(LodLevel(0)));
        assert_eq!(s.level_for(50.0), Some(LodLevel(0)));
        assert_eq!(s.level_for(101.0), Some(LodLevel(1)));
        assert_eq!(s.level_for(2_499.0), Some(LodLevel(1)));
        assert_eq!(s.level_for(2_501.0), Some(LodLevel(2)));
        assert_eq!(s.level_for(40_000.0), Some(LodLevel(2)));
    }

    #[test]
    fn boundary_upper_edge_is_inclusive() {
        let s = sched();
        // Exactly at a band's max_distance_sq stays in that band.
        assert_eq!(s.level_for(100.0), Some(LodLevel(0)));
        assert_eq!(s.level_for(2_500.0), Some(LodLevel(1)));
        // Just past it falls through to the next band.
        let eps = 0.001;
        assert_eq!(s.level_for(100.0 + eps), Some(LodLevel(1)));
        assert_eq!(s.level_for(2_500.0 + eps), Some(LodLevel(2)));
    }

    #[test]
    fn clamp_to_last_beyond_range() {
        let s = sched(); // default ClampToLast
        assert_eq!(s.out_of_range(), OutOfRange::ClampToLast);
        assert_eq!(s.level_for(1_000_000.0), Some(LodLevel(2)));
    }

    #[test]
    fn dormant_beyond_range() {
        let s = sched().with_policy(OutOfRange::Dormant);
        assert_eq!(s.out_of_range(), OutOfRange::Dormant);
        assert_eq!(s.level_for(40_000.0), Some(LodLevel(2))); // still in range
        assert_eq!(s.level_for(40_001.0), None); // dormant beyond last band
    }

    #[test]
    fn set_out_of_range_in_place() {
        let mut s = sched();
        s.set_out_of_range(OutOfRange::Dormant);
        assert_eq!(s.level_for(1_000_000.0), None);
        s.set_out_of_range(OutOfRange::ClampToLast);
        assert_eq!(s.level_for(1_000_000.0), Some(LodLevel(2)));
    }

    #[test]
    fn tick_period_lookup_and_clamp() {
        let s = sched();
        assert_eq!(s.tick_period(LodLevel(0)), 1);
        assert_eq!(s.tick_period(LodLevel(1)), 4);
        assert_eq!(s.tick_period(LodLevel(2)), 16);
        // Out-of-range level clamps to the last band.
        assert_eq!(s.tick_period(LodLevel(99)), 16);
    }

    #[test]
    fn period_one_ticks_every_frame() {
        let s = sched();
        for frame in 0..10 {
            assert!(s.should_tick(LodLevel(0), frame));
        }
    }

    #[test]
    fn period_four_ticks_on_multiples() {
        let s = sched();
        let expected = [true, false, false, false, true, false, false, false, true];
        for (frame, &want) in expected.iter().enumerate() {
            assert_eq!(
                s.should_tick(LodLevel(1), frame as u64),
                want,
                "frame {frame}"
            );
        }
    }

    #[test]
    fn phase_offset_spreads_load() {
        let s = sched();
        // Period 4: phase 0 ticks on 0,4,8; phase 1 ticks on 3,7,11; etc.
        assert!(s.should_tick_phased(LodLevel(1), 0, 0));
        assert!(!s.should_tick_phased(LodLevel(1), 0, 1));
        assert!(s.should_tick_phased(LodLevel(1), 3, 1)); // (3+1)%4==0
        assert!(s.should_tick_phased(LodLevel(1), 2, 2)); // (2+2)%4==0
                                                          // Four distinct phases cover four consecutive frames exactly once each.
        for frame in 0u64..4 {
            let hits = (0u64..4)
                .filter(|&p| s.should_tick_phased(LodLevel(1), frame, p))
                .count();
            assert_eq!(
                hits, 1,
                "frame {frame} should be covered by exactly one phase"
            );
        }
    }

    #[test]
    fn evaluate_combines_level_and_cadence() {
        let s = sched();
        // Distance 900 (= 30 m) -> band 1, period 4.
        let d = distance_sq([0.0, 0.0, 0.0], [30.0, 0.0, 0.0]);
        assert_eq!(d, 900.0);
        let on = s.evaluate(d, 8);
        assert_eq!(on.level, Some(LodLevel(1)));
        assert!(on.tick_this_frame); // 8 % 4 == 0
        let off = s.evaluate(d, 9);
        assert_eq!(off.level, Some(LodLevel(1)));
        assert!(!off.tick_this_frame); // 9 % 4 != 0
    }

    #[test]
    fn evaluate_dormant_never_ticks() {
        let s = sched().with_policy(OutOfRange::Dormant);
        let dec = s.evaluate(1_000_000.0, 0); // frame 0 would otherwise tick
        assert_eq!(dec.level, None);
        assert!(!dec.tick_this_frame);
    }

    #[test]
    fn evaluate_phased_matches_parts() {
        let s = sched();
        let d = 2_000.0; // band 1, period 4
        let dec = s.evaluate_phased(d, 1, 3); // (1+3)%4==0
        assert_eq!(dec.level, Some(LodLevel(1)));
        assert!(dec.tick_this_frame);
    }

    #[test]
    fn distance_sq_is_correct() {
        assert_eq!(distance_sq([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]), 25.0);
        assert_eq!(distance_sq([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0);
        assert_eq!(distance_sq([0.0, 0.0, 0.0], [2.0, 3.0, 6.0]), 49.0);
        // Order-independent.
        assert_eq!(
            distance_sq([-1.0, -2.0, -3.0], [2.0, 2.0, 9.0]),
            distance_sq([2.0, 2.0, 9.0], [-1.0, -2.0, -3.0])
        );
    }

    #[test]
    fn builder_accessors() {
        let s = sched();
        assert_eq!(s.band_count(), 3);
        assert_eq!(s.bands().len(), 3);
        assert_eq!(s.bands()[0], LodBand::new(100.0, 1));
    }

    #[test]
    fn quality_tier_defaults_to_highest_and_is_ordered() {
        // Bands built via the frequency-only path default to full precision.
        let s = sched();
        assert_eq!(s.quality(LodLevel(0)), LodQuality::HIGHEST);
        assert_eq!(s.quality(LodLevel(2)), LodQuality::HIGHEST);
        assert_eq!(LodQuality::default(), LodQuality(0));
        assert!(LodQuality(0) < LodQuality(1));
    }

    #[test]
    fn quality_is_independent_of_cadence_and_band_index() {
        // Decouple precision from both distance and frequency: a full-rate near
        // band may still request reduced precision, and two bands can share a
        // tier regardless of their index.
        let s = LodSchedule::from_sorted_triples(&[
            (100.0, 1, LodQuality(1)),   // near, every frame, but coarse precision
            (2_500.0, 4, LodQuality(1)), // shares tier 1 with the nearer band
            (40_000.0, 16, LodQuality(3)),
        ]);
        assert_eq!(s.quality(LodLevel(0)), LodQuality(1));
        assert_eq!(s.quality(LodLevel(1)), LodQuality(1));
        assert_eq!(s.quality(LodLevel(2)), LodQuality(3));
        // Cadence axis is still independent.
        assert_eq!(s.tick_period(LodLevel(0)), 1);
        assert_eq!(s.tick_period(LodLevel(2)), 16);
    }

    #[test]
    fn quality_clamps_past_last_band() {
        let s = LodSchedule::from_sorted_triples(&[
            (100.0, 1, LodQuality(0)),
            (2_500.0, 4, LodQuality(2)),
        ]);
        // Out-of-range level clamps to the last band's tier.
        assert_eq!(s.quality(LodLevel(9)), LodQuality(2));
    }

    #[test]
    fn evaluate_reports_selected_band_quality() {
        let s = LodSchedule::from_sorted_triples(&[
            (100.0, 1, LodQuality(0)),
            (2_500.0, 4, LodQuality(2)),
            (40_000.0, 16, LodQuality(4)),
        ]);
        // Mid band (d² 900 → level 1), frame 8 ticks (8 % 4 == 0).
        let d = s.evaluate(900.0, 8);
        assert_eq!(d.level, Some(LodLevel(1)));
        assert!(d.tick_this_frame);
        assert_eq!(d.quality, LodQuality(2));

        // A ticking far entity still reports the coarse tier.
        let far = s.evaluate(39_000.0, 16);
        assert_eq!(far.level, Some(LodLevel(2)));
        assert_eq!(far.quality, LodQuality(4));
    }

    #[test]
    fn dormant_decision_reports_highest_quality_default() {
        let s = LodSchedule::with_out_of_range(
            alloc::vec![LodBand::with_quality(100.0, 1, LodQuality(3))],
            OutOfRange::Dormant,
        );
        let d = s.evaluate(1_000.0, 0); // beyond last band, dormant
        assert_eq!(d.level, None);
        assert!(!d.tick_this_frame);
        assert_eq!(d.quality, LodQuality::HIGHEST);
    }

    #[test]
    fn with_quality_builder_matches_field() {
        let b = LodBand::new(100.0, 2).quality(LodQuality(5));
        assert_eq!(b.quality, LodQuality(5));
        assert_eq!(b.tick_period, 2);
        assert_eq!(b.max_distance_sq, 100.0);
        assert_eq!(LodBand::with_quality(100.0, 2, LodQuality(5)), b);
    }

    #[test]
    #[should_panic]
    fn unsorted_bands_panic() {
        let _ = LodSchedule::from_sorted_pairs(&[(2_500.0, 4), (100.0, 1)]);
    }

    #[test]
    #[should_panic]
    fn zero_tick_period_panics() {
        let _ = LodSchedule::from_sorted_pairs(&[(100.0, 0)]);
    }

    #[test]
    #[should_panic]
    fn empty_schedule_panics() {
        let _ = LodSchedule::new(Vec::new());
    }
}
