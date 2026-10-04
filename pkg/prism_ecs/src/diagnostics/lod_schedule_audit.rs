//! LOD schedule well-formedness audit (design §13.2 / §23.7 / §16.6).
//!
//! A [`LodSchedule`](crate::partition::lod::LodSchedule) is the band table that
//! turns an entity's squared distance into an update *cadence*
//! ([`tick_period`](crate::partition::lod::LodBand::tick_period)) and a
//! simulation *precision* tier
//! ([`quality`](crate::partition::lod::LodBand::quality)) — the mechanism
//! behind MassEntity-style processor LOD (design §13.2 / §23.7). Its
//! constructor enforces only the hard structural contract: at least one band,
//! bands sorted nearest-first by ascending `max_distance_sq`, and every
//! `tick_period >= 1`.
//!
//! It does **not** enforce the *semantic* conventions a healthy schedule should
//! obey, and getting those wrong silently inverts the whole point of LOD:
//!
//! * **Reachability.** Because [`level_for`](crate::partition::lod::LodSchedule::level_for)
//!   picks the first band whose inclusive upper edge covers the distance, a
//!   band whose `max_distance_sq` does not exceed the previous band's can never
//!   be selected — it is dead configuration that will never run.
//! * **Cadence monotonicity.** Nearer bands should tick at least as often as
//!   farther ones. A farther band with a *smaller* `tick_period` than a nearer
//!   band makes distant entities update more frequently than close ones — the
//!   opposite of the intended CPU saving.
//! * **Quality monotonicity.** Likewise a farther band should be no more
//!   precise than a nearer one; a farther band with a *lower*
//!   [`LodQuality`](crate::partition::lod::LodQuality) value spends more work on
//!   distant entities than close ones.
//!
//! This report walks the band table once and surfaces each of those smells per
//! band plus schedule-wide roll-ups, so a tuning tool or CI gate can reject a
//! misconfigured schedule before it ships. It is a pure read of the schedule
//! (`O(bands)`), touches no world state, and is deterministic.

use alloc::vec::Vec;

use crate::partition::lod::{LodLevel, LodQuality, LodSchedule, OutOfRange};

/// Per-band audit of a [`LodSchedule`], pairing the band's raw parameters with
/// the semantic-convention smells detected against the previous (nearer) band
/// (design §13.2 / §23.7).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LodBandAudit {
    /// The band's detail level (its index in the schedule, nearest first).
    pub level: LodLevel,
    /// Inclusive upper bound of the band, as a squared distance in metres².
    pub max_distance_sq: f32,
    /// Width of the band's distance interval: this band's `max_distance_sq`
    /// minus the previous band's (or minus `0.0` for the nearest band).
    pub band_width_sq: f32,
    /// Update cadence: entities in this band tick once every `tick_period`
    /// frames.
    pub tick_period: u32,
    /// Simulation precision tier for the band (design §23.7).
    pub quality: LodQuality,
    /// Whether the band can never be selected because its upper edge does not
    /// exceed the previous band's (a zero- or negative-width interval). Always
    /// `false` for the nearest band.
    pub is_unreachable: bool,
    /// Whether this (farther) band ticks *more* often than the previous nearer
    /// band — a backwards cadence. Always `false` for the nearest band.
    pub cadence_regresses: bool,
    /// Whether this (farther) band is *more* precise than the previous nearer
    /// band — a backwards precision tier. Always `false` for the nearest band.
    pub quality_regresses: bool,
}

impl LodBandAudit {
    /// Whether the band ticks every frame (`tick_period == 1`).
    #[inline]
    pub fn is_full_rate(&self) -> bool {
        self.tick_period == 1
    }

    /// Whether the band is free of every detected smell (reachable, and neither
    /// cadence nor quality regresses against the nearer band).
    #[inline]
    pub fn is_healthy(&self) -> bool {
        !self.is_unreachable && !self.cadence_regresses && !self.quality_regresses
    }
}

/// Read-only well-formedness audit of a whole [`LodSchedule`] band table
/// (design §13.2 / §23.7 / §16.6).
#[derive(Clone, Debug)]
pub struct LodScheduleAudit {
    /// Per-band audits, nearest first (band index is the [`LodLevel`]).
    entries: Vec<LodBandAudit>,
    /// The schedule's out-of-range policy for entities beyond the last band.
    beyond_last: OutOfRange,
}

impl LodScheduleAudit {
    /// Audit `schedule`. Read-only; walks the band table once comparing each
    /// band to its nearer neighbour to detect unreachable bands and cadence /
    /// quality regressions (design §13.2 / §23.7).
    pub fn from_schedule(schedule: &LodSchedule) -> Self {
        let bands = schedule.bands();
        let mut entries: Vec<LodBandAudit> = Vec::with_capacity(bands.len());

        let mut prev_max = 0.0f32;
        let mut prev_period: Option<u32> = None;
        let mut prev_quality: Option<u8> = None;
        for (index, band) in bands.iter().enumerate() {
            let band_width_sq = band.max_distance_sq - prev_max;
            let is_unreachable = index > 0 && band_width_sq <= 0.0;
            let cadence_regresses = prev_period.is_some_and(|p| band.tick_period < p);
            let quality_regresses = prev_quality.is_some_and(|q| band.quality.0 < q);

            entries.push(LodBandAudit {
                level: LodLevel(index as u8),
                max_distance_sq: band.max_distance_sq,
                band_width_sq,
                tick_period: band.tick_period,
                quality: band.quality,
                is_unreachable,
                cadence_regresses,
                quality_regresses,
            });

            prev_max = band.max_distance_sq;
            prev_period = Some(band.tick_period);
            prev_quality = Some(band.quality.0);
        }

        Self {
            entries,
            beyond_last: schedule.out_of_range(),
        }
    }

    /// The per-band audits, nearest first.
    #[inline]
    pub fn entries(&self) -> &[LodBandAudit] {
        &self.entries
    }

    /// Number of bands in the audited schedule (always `>= 1`).
    #[inline]
    pub fn band_count(&self) -> usize {
        self.entries.len()
    }

    /// The schedule's out-of-range policy.
    #[inline]
    pub fn out_of_range(&self) -> OutOfRange {
        self.beyond_last
    }

    /// Number of bands that can never be selected (zero- / negative-width
    /// interval against the nearer band).
    pub fn unreachable_band_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_unreachable)
            .count()
    }

    /// Whether any band is unreachable.
    #[inline]
    pub fn has_unreachable_band(&self) -> bool {
        self.unreachable_band_count() > 0
    }

    /// Number of bands whose cadence regresses against the nearer band.
    pub fn cadence_regression_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.cadence_regresses)
            .count()
    }

    /// Number of bands whose precision tier regresses against the nearer band.
    pub fn quality_regression_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.quality_regresses)
            .count()
    }

    /// Whether cadence is monotonic non-decreasing from near to far (no band
    /// ticks more often than a nearer band).
    #[inline]
    pub fn is_monotonic_cadence(&self) -> bool {
        self.cadence_regression_count() == 0
    }

    /// Whether precision is monotonic non-increasing from near to far (no band
    /// is more precise than a nearer band).
    #[inline]
    pub fn is_monotonic_quality(&self) -> bool {
        self.quality_regression_count() == 0
    }

    /// Whether the schedule is free of every detected smell: all bands
    /// reachable, cadence monotonic, and quality monotonic.
    #[inline]
    pub fn is_well_formed(&self) -> bool {
        !self.has_unreachable_band() && self.is_monotonic_cadence() && self.is_monotonic_quality()
    }

    /// Number of bands that tick every frame (`tick_period == 1`).
    pub fn full_rate_band_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_full_rate())
            .count()
    }

    /// The largest `tick_period` across all bands (the coarsest cadence).
    /// Returns `1` for the degenerate empty case that the schedule contract
    /// forbids.
    pub fn max_tick_period(&self) -> u32 {
        self.entries
            .iter()
            .map(|entry| entry.tick_period)
            .max()
            .unwrap_or(1)
    }

    /// The smallest `tick_period` across all bands (the finest cadence).
    /// Returns `1` for the degenerate empty case that the schedule contract
    /// forbids.
    pub fn min_tick_period(&self) -> u32 {
        self.entries
            .iter()
            .map(|entry| entry.tick_period)
            .min()
            .unwrap_or(1)
    }

    /// Number of distinct precision tiers ([`LodQuality`]) used across the
    /// bands.
    pub fn distinct_quality_tiers(&self) -> usize {
        let mut tiers: Vec<u8> = self.entries.iter().map(|entry| entry.quality.0).collect();
        tiers.sort_unstable();
        tiers.dedup();
        tiers.len()
    }

    /// The farthest reachable squared distance: the last band's upper edge.
    /// Returns `0.0` for the degenerate empty case that the schedule contract
    /// forbids.
    pub fn farthest_distance_sq(&self) -> f32 {
        self.entries
            .last()
            .map(|entry| entry.max_distance_sq)
            .unwrap_or(0.0)
    }

    /// Look up a band audit by its detail level. `None` when out of range.
    pub fn entry(&self, level: LodLevel) -> Option<&LodBandAudit> {
        self.entries.get(level.0 as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::lod::LodSchedule;

    #[test]
    fn healthy_schedule_is_well_formed() {
        // Ascending distance, non-decreasing cadence and quality.
        let schedule = LodSchedule::from_sorted_triples(&[
            (10.0, 1, LodQuality(0)),
            (100.0, 2, LodQuality(0)),
            (1000.0, 4, LodQuality(1)),
        ]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert_eq!(audit.band_count(), 3);
        assert!(audit.is_well_formed());
        assert!(audit.is_monotonic_cadence());
        assert!(audit.is_monotonic_quality());
        assert!(!audit.has_unreachable_band());
        assert!(audit.entries().iter().all(|entry| entry.is_healthy()));
        assert_eq!(audit.full_rate_band_count(), 1);
        assert_eq!(audit.max_tick_period(), 4);
        assert_eq!(audit.min_tick_period(), 1);
        assert_eq!(audit.distinct_quality_tiers(), 2);
    }

    #[test]
    fn duplicate_upper_edge_flags_unreachable_band() {
        // Band 1 shares band 0's upper edge, so level_for can never pick it.
        let schedule = LodSchedule::from_sorted_pairs(&[(10.0, 1), (10.0, 2), (100.0, 4)]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert!(audit.has_unreachable_band());
        assert_eq!(audit.unreachable_band_count(), 1);
        assert!(audit.entry(LodLevel(1)).unwrap().is_unreachable);
        assert!(!audit.entry(LodLevel(0)).unwrap().is_unreachable);
        assert!(!audit.entry(LodLevel(2)).unwrap().is_unreachable);
        assert!(!audit.is_well_formed());
    }

    #[test]
    fn cadence_regression_is_detected() {
        // Farther band ticks more often (1) than the nearer band (4).
        let schedule = LodSchedule::from_sorted_pairs(&[(10.0, 4), (100.0, 1)]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert_eq!(audit.cadence_regression_count(), 1);
        assert!(!audit.is_monotonic_cadence());
        assert!(audit.entry(LodLevel(1)).unwrap().cadence_regresses);
        assert!(!audit.is_well_formed());
    }

    #[test]
    fn quality_regression_is_detected() {
        // Farther band is more precise (quality 0) than the nearer band (2).
        let schedule = LodSchedule::from_sorted_triples(&[
            (10.0, 1, LodQuality(2)),
            (100.0, 2, LodQuality(0)),
        ]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert_eq!(audit.quality_regression_count(), 1);
        assert!(!audit.is_monotonic_quality());
        assert!(audit.entry(LodLevel(1)).unwrap().quality_regresses);
        assert!(!audit.is_well_formed());
    }

    #[test]
    fn band_widths_and_levels_are_computed() {
        let schedule = LodSchedule::from_sorted_pairs(&[(10.0, 1), (100.0, 2)]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        let b0 = audit.entry(LodLevel(0)).unwrap();
        let b1 = audit.entry(LodLevel(1)).unwrap();
        assert_eq!(b0.level, LodLevel(0));
        assert_eq!(b1.level, LodLevel(1));
        assert!((b0.band_width_sq - 10.0).abs() < f32::EPSILON);
        assert!((b1.band_width_sq - 90.0).abs() < f32::EPSILON);
        assert!((audit.farthest_distance_sq() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn out_of_range_policy_is_preserved() {
        let schedule =
            LodSchedule::from_sorted_pairs(&[(10.0, 1)]).with_policy(OutOfRange::Dormant);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert_eq!(audit.out_of_range(), OutOfRange::Dormant);
        assert_eq!(audit.band_count(), 1);
        // A single nearest band is always reachable and healthy.
        assert!(audit.is_well_formed());
    }

    #[test]
    fn entry_lookup_out_of_range_is_none() {
        let schedule = LodSchedule::from_sorted_pairs(&[(10.0, 1), (100.0, 2)]);
        let audit = LodScheduleAudit::from_schedule(&schedule);
        assert!(audit.entry(LodLevel(2)).is_none());
        assert!(audit.entry(LodLevel(99)).is_none());
    }
}
