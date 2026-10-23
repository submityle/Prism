//! Entity LOD processor: drives a population through a [`LodSchedule`] and
//! transitions dormancy in one deterministic pass (design §13.2; §23.7).
//!
//! [`LodSchedule`](crate::partition::lod::LodSchedule) answers *per entity*
//! ("what band, does it tick this frame?") and [`DormancySet`] tracks *which
//! entities are asleep* — but nothing links the two: an entity whose LOD band
//! falls off the end of the schedule ([`OutOfRange::Dormant`]) should be put to
//! sleep, and one that moves back into range should be woken. [`EntityLodProcessor`]
//! is that link. Each frame it takes a batch of `(entity, position)` samples
//! and a viewpoint and, in one pass:
//!
//! * evaluates every entity's [`LodDecision`];
//! * sleeps entities that just went dormant and wakes entities that just
//!   re-entered range, driving the [`DormancySet`] authoritatively;
//! * collects the awake entities that should run their simulation this frame.
//!
//! This is the "processor" form from design §23.7. [`drive_candidates`](Self::drive_candidates)
//! lets a spatial-index owner bound work to a candidate population, while
//! [`drive`](Self::drive) remains the convenience full-population path. Far
//! entities can collapse to a reduced cadence or dormancy. It is deliberately
//! `World`-independent — the owner
//! extracts `(entity, position)` from its transform components and applies the
//! returned [`LodLevel`](crate::partition::lod::LodLevel) tags / tick list back to the world — matching the pure
//! bookkeeping style of the rest of [`crate::partition`].
//!
//! Determinism (design §14): all ordered outputs are sorted by
//! [`Entity::to_bits`], so a given population and frame yield a frame-stable,
//! platform-independent result regardless of input ordering.

use alloc::vec::Vec;

use crate::entity::Entity;
use crate::partition::dormant::DormancySet;
use crate::partition::lod::{distance_sq, LodDecision, LodSchedule};

/// How an [`EntityLodProcessor`] assigns the per-entity tick *phase* that
/// spreads a band's updates across frames (design §23.7 anti-thundering-herd).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PhasePolicy {
    /// Every entity in a band shares phase `0`, so the whole band ticks on the
    /// same frames. Simplest and fully deterministic; fine for small bands.
    #[default]
    Synchronized,
    /// Each entity's phase is derived from its handle
    /// ([`Entity::index`]), spreading a band's updates across its period so a
    /// large band never ticks all at once. Still deterministic: the same entity
    /// always gets the same phase.
    PerEntity,
}

impl PhasePolicy {
    /// The tick phase this policy assigns to `entity`.
    #[inline]
    fn phase(self, entity: Entity) -> u64 {
        match self {
            PhasePolicy::Synchronized => 0,
            PhasePolicy::PerEntity => entity.index() as u64,
        }
    }
}

/// The outcome of one [`EntityLodProcessor::drive`] pass over a population
/// (design §13.2).
///
/// `to_tick` is the active set for this frame; `slept`/`woken` are the dormancy
/// transitions the processor applied to the [`DormancySet`]; `decisions`
/// carries every entity's [`LodDecision`] in input order so the owner can write
/// back [`LodLevel`](crate::partition::lod::LodLevel) tags. All of `to_tick`, `slept`, and `woken` are sorted by
/// [`Entity::to_bits`] for a deterministic schedule (design §14).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LodTickResult {
    /// Awake entities whose band says they should run their simulation on this
    /// frame, sorted by handle. This is the per-frame active set.
    pub to_tick: Vec<Entity>,
    /// Entities put to sleep this pass because their LOD went dormant (level
    /// `None`), sorted by handle.
    pub slept: Vec<Entity>,
    /// Entities woken this pass because they re-entered a live band after being
    /// dormant, sorted by handle.
    pub woken: Vec<Entity>,
    /// Every evaluated entity paired with its decision, in the order the
    /// population batch was supplied (not sorted), so the owner can apply
    /// [`LodLevel`](crate::partition::lod::LodLevel) tags without a second evaluation.
    pub decisions: Vec<(Entity, LodDecision)>,
}

impl LodTickResult {
    /// Whether nothing ticks and no dormancy transition happened this pass.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.to_tick.is_empty() && self.slept.is_empty() && self.woken.is_empty()
    }

    /// Number of entities scheduled to tick this frame.
    #[inline]
    pub fn tick_count(&self) -> usize {
        self.to_tick.len()
    }
}

/// Drives a population of entities through a [`LodSchedule`], transitioning a
/// [`DormancySet`] and producing the per-frame active set (design §13.2;
/// §23.7).
///
/// The processor owns the schedule and a [`PhasePolicy`]; it does not own the
/// dormancy set (the owner keeps that across many processors / systems and
/// drains its wake events centrally). Call [`drive`](Self::drive) once per
/// frame with the current viewpoint and the entities' positions.
#[derive(Clone, Debug)]
pub struct EntityLodProcessor {
    schedule: LodSchedule,
    phase: PhasePolicy,
}

impl EntityLodProcessor {
    /// Builds a processor with the given schedule and a
    /// [`PhasePolicy::Synchronized`] cadence.
    #[inline]
    pub fn new(schedule: LodSchedule) -> Self {
        Self {
            schedule,
            phase: PhasePolicy::Synchronized,
        }
    }

    /// Sets the [`PhasePolicy`] used to spread band updates across frames.
    #[inline]
    pub fn with_phase_policy(mut self, phase: PhasePolicy) -> Self {
        self.phase = phase;
        self
    }

    /// The schedule this processor evaluates against.
    #[inline]
    pub fn schedule(&self) -> &LodSchedule {
        &self.schedule
    }

    /// The active phase policy.
    #[inline]
    pub fn phase_policy(&self) -> PhasePolicy {
        self.phase
    }

    /// Evaluates one entity at `position` against `viewpoint` on `frame`,
    /// without touching any dormancy state. Pure helper exposed for callers
    /// that only want the decision.
    #[inline]
    pub fn evaluate(
        &self,
        viewpoint: [f32; 3],
        position: [f32; 3],
        entity: Entity,
        frame: u64,
    ) -> LodDecision {
        let d = distance_sq(viewpoint, position);
        self.schedule
            .evaluate_phased(d, frame, self.phase.phase(entity))
    }

    /// Drives the whole `population` for one frame (design §13.2).
    ///
    /// For each `(entity, position)`:
    /// * evaluate its [`LodDecision`] relative to `viewpoint`;
    /// * if the entity is now dormant (level `None`), [`sleep`](DormancySet::sleep)
    ///   it and record the transition in [`LodTickResult::slept`];
    /// * otherwise, if it was dormant, [`wake`](DormancySet::wake) it and record
    ///   it in [`LodTickResult::woken`]; and if its band says it ticks this
    ///   frame, add it to [`LodTickResult::to_tick`].
    ///
    /// The dormancy set is mutated in place; its own wake-event log (drained via
    /// [`DormancySet::drain_woken`]) therefore stays authoritative for the
    /// scheduler. All ordered outputs are sorted by handle before returning.
    pub fn drive(
        &self,
        viewpoint: [f32; 3],
        population: &[(Entity, [f32; 3])],
        frame: u64,
        dormancy: &mut DormancySet,
    ) -> LodTickResult {
        self.drive_candidates(viewpoint, population, frame, dormancy)
    }

    /// Drives a candidate subset for one frame (design §13.2 / §23.7).
    ///
    /// This is the indexed-population entry point: callers may pass only the
    /// entities selected by a spatial query instead of materializing and
    /// evaluating every entity in the world. To preserve transitions, the
    /// candidate set must include **all entities within the schedule's
    /// farthest active distance** and every previously-awake entity that may
    /// have moved out of range since the last pass, with each entity appearing
    /// at most once. Entities omitted from this
    /// slice are neither evaluated nor changed in `dormancy`; in particular,
    /// dormant entities outside the query bounds are expected to remain asleep
    /// until a later spatial query selects them. Entity creation/destruction
    /// and stale handles remain the caller's responsibility.
    ///
    /// The work performed here is `O(candidates)` (plus sorting emitted
    /// results), independent of the total world population. Completeness and
    /// freshness of the spatial candidate query are part of the caller's
    /// contract; use [`drive`](Self::drive) when no such index is available.
    pub fn drive_candidates(
        &self,
        viewpoint: [f32; 3],
        candidates: &[(Entity, [f32; 3])],
        frame: u64,
        dormancy: &mut DormancySet,
    ) -> LodTickResult {
        let mut to_tick = Vec::new();
        let mut slept = Vec::new();
        let mut woken = Vec::new();
        let mut decisions = Vec::with_capacity(candidates.len());

        for &(entity, position) in candidates {
            let decision = self.evaluate(viewpoint, position, entity, frame);
            match decision.level {
                None => {
                    // Beyond the schedule under `OutOfRange::Dormant`: pull the
                    // entity out of the active set. `sleep` returns false if it
                    // was already dormant, so a steady-state dormant entity is
                    // not re-reported.
                    if dormancy.sleep(entity) {
                        slept.push(entity);
                    }
                }
                Some(_) => {
                    // Live band: re-admit if it was asleep, then schedule it if
                    // the cadence fires this frame.
                    if dormancy.is_dormant(entity) && dormancy.wake(entity) {
                        woken.push(entity);
                    }
                    if decision.tick_this_frame {
                        to_tick.push(entity);
                    }
                }
            }
            decisions.push((entity, decision));
        }

        to_tick.sort_unstable_by_key(|e| e.to_bits());
        slept.sort_unstable_by_key(|e| e.to_bits());
        woken.sort_unstable_by_key(|e| e.to_bits());

        LodTickResult {
            to_tick,
            slept,
            woken,
            decisions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::lod::{LodLevel, LodQuality, OutOfRange};

    fn ent(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64)
            .expect("test entity needs a nonzero generation")
    }

    /// Three bands, dormant beyond the last: < 100 full rate, < 2500 every 4th,
    /// < 40000 every 16th, else dormant.
    fn schedule() -> LodSchedule {
        LodSchedule::from_sorted_pairs(&[(100.0, 1), (2_500.0, 4), (40_000.0, 16)])
            .with_policy(OutOfRange::Dormant)
    }

    #[test]
    fn evaluate_matches_schedule() {
        let proc = EntityLodProcessor::new(schedule());
        let e = ent(1, 1);
        // Distance 30 → d² 900 → band 1 (< 2500), ticks on frame 8 (8 % 4 == 0).
        let d = proc.evaluate([0.0; 3], [30.0, 0.0, 0.0], e, 8);
        assert_eq!(d.level, Some(LodLevel(1)));
        assert!(d.tick_this_frame);
    }

    #[test]
    fn drive_schedules_only_ticking_entities() {
        let proc = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let near = ent(1, 1); // band 0, period 1 → always ticks
        let mid = ent(2, 1); // band 1, period 4
        let pop = [
            (near, [5.0, 0.0, 0.0]), // d² 25
            (mid, [40.0, 0.0, 0.0]), // d² 1600
        ];

        // Frame 1: near ticks (period 1), mid does not (1 % 4 != 0).
        let r = proc.drive([0.0; 3], &pop, 1, &mut dormancy);
        assert_eq!(r.to_tick, alloc::vec![near]);
        assert!(r.slept.is_empty());
        assert!(r.woken.is_empty());

        // Frame 4: both tick (4 % 4 == 0, 4 % 1 == 0).
        let r = proc.drive([0.0; 3], &pop, 4, &mut dormancy);
        assert_eq!(r.to_tick, alloc::vec![near, mid]);
    }

    #[test]
    fn candidate_drive_only_evaluates_spatial_candidates_and_preserves_transitions() {
        let proc = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();
        let leaving_range = ent(1, 1);
        let entering_range = ent(2, 1);
        let near = ent(3, 1);

        // A previous awake entity must remain a candidate long enough to be
        // put to sleep after leaving range. A dormant entity is reconsidered
        // when the spatial query selects it again.
        let first = proc.drive_candidates(
            [0.0; 3],
            &[
                (leaving_range, [10.0, 0.0, 0.0]),
                (entering_range, [1000.0, 0.0, 0.0]),
                (near, [1.0, 0.0, 0.0]),
            ],
            0,
            &mut dormancy,
        );
        assert_eq!(first.decisions.len(), 3);
        assert!(dormancy.is_dormant(entering_range));

        let second = proc.drive_candidates(
            [0.0; 3],
            &[
                // Spatial query includes recently-awake entities as an exit
                // guard, even when they no longer intersect the active radius.
                (leaving_range, [1000.0, 0.0, 0.0]),
                // This dormant entity is now inside the spatial query bounds.
                (entering_range, [5.0, 0.0, 0.0]),
                (near, [2.0, 0.0, 0.0]),
            ],
            1,
            &mut dormancy,
        );
        assert_eq!(second.decisions.len(), 3);
        assert_eq!(second.slept, alloc::vec![leaving_range]);
        assert_eq!(second.woken, alloc::vec![entering_range]);
        assert!(dormancy.is_dormant(leaving_range));
        assert!(!dormancy.is_dormant(entering_range));

        // Unrelated off-index population is not passed to or evaluated by the
        // processor; output work is bounded by the candidate slice.
        let sparse = proc.drive_candidates([0.0; 3], &[(near, [3.0, 0.0, 0.0])], 2, &mut dormancy);
        assert_eq!(sparse.decisions.len(), 1);
        assert_eq!(sparse.to_tick, alloc::vec![near]);
    }

    #[test]
    fn drive_sleeps_out_of_range_entities() {
        let proc = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let far = ent(7, 1);
        let pop = [(far, [1000.0, 0.0, 0.0])]; // d² 1_000_000 > 40_000 → dormant

        let r = proc.drive([0.0; 3], &pop, 0, &mut dormancy);
        assert_eq!(r.slept, alloc::vec![far]);
        assert!(r.to_tick.is_empty());
        assert!(dormancy.is_dormant(far));

        // Still out of range next frame: no duplicate sleep report.
        let r = proc.drive([0.0; 3], &pop, 1, &mut dormancy);
        assert!(r.slept.is_empty());
        assert!(dormancy.is_dormant(far));
    }

    #[test]
    fn drive_wakes_entities_returning_into_range() {
        let proc = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let e = ent(3, 1);
        // Far away → sleep.
        let r = proc.drive([0.0; 3], &[(e, [1000.0, 0.0, 0.0])], 0, &mut dormancy);
        assert_eq!(r.slept, alloc::vec![e]);
        assert!(dormancy.is_dormant(e));

        // Back in range on frame 4 → wake and (band 0, period 1) tick.
        let r = proc.drive([0.0; 3], &[(e, [5.0, 0.0, 0.0])], 4, &mut dormancy);
        assert_eq!(r.woken, alloc::vec![e]);
        assert_eq!(r.to_tick, alloc::vec![e]);
        assert!(!dormancy.is_dormant(e));
        // The dormancy set logged the wake event for the scheduler to drain.
        assert_eq!(dormancy.drain_woken(), alloc::vec![e]);
    }

    #[test]
    fn outputs_are_sorted_by_handle() {
        let proc = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let a = ent(9, 1);
        let b = ent(2, 1);
        let c = ent(5, 1);
        // All near, band 0, period 1 → all tick; supplied out of order.
        let pop = [
            (a, [1.0, 0.0, 0.0]),
            (b, [1.0, 0.0, 0.0]),
            (c, [1.0, 0.0, 0.0]),
        ];
        let r = proc.drive([0.0; 3], &pop, 0, &mut dormancy);
        assert_eq!(r.to_tick, alloc::vec![b, c, a]); // sorted by index 2,5,9
                                                     // decisions preserve input order.
        let order: Vec<Entity> = r.decisions.iter().map(|&(e, _)| e).collect();
        assert_eq!(order, alloc::vec![a, b, c]);
    }

    #[test]
    fn decisions_carry_the_band_precision_tier() {
        // A schedule whose bands carry distinct precision tiers; the processor
        // must surface each entity's tier in its decision so the owner can pick
        // a full vs. approximate simulation path (design §23.7).
        let sched = LodSchedule::from_sorted_triples(&[
            (100.0, 1, LodQuality(0)),   // near: full precision
            (2_500.0, 4, LodQuality(2)), // mid: coarser
            (40_000.0, 16, LodQuality(4)),
        ])
        .with_policy(OutOfRange::Dormant);
        let proc = EntityLodProcessor::new(sched);
        let mut dormancy = DormancySet::new();

        let near = ent(1, 1);
        let mid = ent(2, 1);
        let pop = [
            (near, [5.0, 0.0, 0.0]), // d² 25  → band 0, tier 0
            (mid, [40.0, 0.0, 0.0]), // d² 1600 → band 1, tier 2
        ];
        let r = proc.drive([0.0; 3], &pop, 0, &mut dormancy);

        // decisions keep input order; check each entity's reported tier.
        let near_q = r
            .decisions
            .iter()
            .find(|&&(e, _)| e == near)
            .unwrap()
            .1
            .quality;
        let mid_q = r
            .decisions
            .iter()
            .find(|&&(e, _)| e == mid)
            .unwrap()
            .1
            .quality;
        assert_eq!(near_q, LodQuality(0));
        assert_eq!(mid_q, LodQuality(2));

        // The pure evaluate() path agrees.
        assert_eq!(
            proc.evaluate([0.0; 3], [5.0, 0.0, 0.0], near, 0).quality,
            LodQuality(0)
        );
        assert_eq!(
            proc.evaluate([0.0; 3], [40.0, 0.0, 0.0], mid, 0).quality,
            LodQuality(2)
        );
    }

    #[test]
    fn per_entity_phase_spreads_a_band_across_frames() {
        let proc = EntityLodProcessor::new(schedule()).with_phase_policy(PhasePolicy::PerEntity);
        assert_eq!(proc.phase_policy(), PhasePolicy::PerEntity);
        let mut dormancy = DormancySet::new();

        // Two entities in band 1 (period 4) with different indices → their tick
        // frames are offset, so they do not both fire on the same frame.
        let e0 = ent(4, 1); // phase 4 → ticks when (frame + 4) % 4 == 0 → frame % 4 == 0
        let e1 = ent(5, 1); // phase 5 → ticks when (frame + 5) % 4 == 0 → frame % 4 == 3
        let pop = [(e0, [40.0, 0.0, 0.0]), (e1, [40.0, 0.0, 0.0])];

        let f0 = proc.drive([0.0; 3], &pop, 0, &mut dormancy);
        assert_eq!(f0.to_tick, alloc::vec![e0]);
        let f3 = proc.drive([0.0; 3], &pop, 3, &mut dormancy);
        assert_eq!(f3.to_tick, alloc::vec![e1]);
    }
}
