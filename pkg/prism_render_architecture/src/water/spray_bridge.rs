//! Crest-spray spawn-event bridge: hands the breaking-wave spray plan to the
//! `Ember` particle engine over the shared cross-system data channel.
//!
//! Per design §6b ("破碎浪与飞沫喷发") and the subsystem split (§3), the water
//! engine never owns a particle pool: it classifies a breaking crest
//! ([`super::breaking`]), plans an arc-spray burst
//! ([`super::breaking::SprayEmission`]), and *emits a spawn event* that `Ember`
//! consumes one frame later. This module is that emitter. It packs each
//! world-space spray burst into a
//! [`ChannelRecord`](crate::particle::events::ChannelRecord) and publishes it to
//! a dedicated channel in `Ember`'s
//! [`EventRouter`](crate::particle::events::EventRouter), the §14 cross-system
//! data channel that the spark/shockwave/smoke consumers already read.
//!
//! The bridge is a deterministic pure scheduler: it walks the bursts in input
//! order, skips calm bursts, admits at most
//! [`WaterBudget::spray_bursts_per_frame`](super::WaterBudget) of them, and
//! reports exactly what happened (admitted / skipped / over-budget / dropped /
//! missing-channel) so the host can surface back-pressure. All arithmetic is
//! add/mul plus `u32` saturation; there is no float equality.
//!
//! ## Record encoding
//! Each admitted burst becomes one [`ChannelRecord`]:
//! * `position` — the world-space crest point the spray erupts from,
//! * `velocity` — the planned launch velocity (crest tangent + upward jet),
//! * `value`    — the spray particle count (as `f32`), so the `Ember` consumer
//!   knows how many whitewater particles to release for this trigger,
//! * `tag`      — [`SPRAY_CHANNEL_TAG`], so a consumer multiplexing one channel
//!   across producers can pick out crest spray.

use super::breaking::SprayEmission;
use super::{Vec3, WaterBudget};
use crate::particle::events::{AppendOutcome, ChannelRecord, EventRouter};
use crate::particle::Vec3 as EmberVec3;

/// Classification tag stamped on every spray [`ChannelRecord`] so an `Ember`
/// consumer sharing a channel can tell crest spray apart from other producers.
///
/// The value is the ASCII bytes `b"watr"` packed big-endian, chosen only to be a
/// stable, human-recognisable constant; consumers compare against this symbol
/// rather than the literal.
pub const SPRAY_CHANNEL_TAG: u32 = 0x7761_7472;

/// A world-space crest-spray burst awaiting delivery to `Ember`.
///
/// Produced by pairing a breaking-wave sample's world position with the
/// [`SprayEmission`] that [`super::breaking::plan_spray`] (or
/// [`super::surface_fx::plan_surface_fx`]) planned for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSprayBurst {
    /// World-space crest position the burst erupts from.
    pub position: Vec3,
    /// The planned spray burst (particle count + launch velocity).
    pub emission: SprayEmission,
}

impl WaterSprayBurst {
    /// Builds a burst from a world position and a planned emission.
    #[must_use]
    pub const fn new(position: Vec3, emission: SprayEmission) -> Self {
        Self { position, emission }
    }

    /// Whether this burst actually releases any spray this frame.
    ///
    /// A calm or merely cresting sample plans [`SprayEmission::NONE`] (zero
    /// count), which the bridge skips without spending budget.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.emission.count > 0
    }
}

/// Outcome tally from one [`publish_spray_bursts`] pass.
///
/// Every active burst lands in exactly one of `admitted` / `over_budget` /
/// `dropped` / `missing_channel`; every calm burst lands in `skipped_calm`.
/// The four buckets plus `skipped_calm` therefore sum to the input length, which
/// the unit tests assert.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SprayPublishStats {
    /// Active bursts packed and accepted into the channel.
    pub admitted: u32,
    /// Spray particles represented by the admitted bursts (sum of their counts,
    /// saturating at [`u32::MAX`]).
    pub particles_admitted: u32,
    /// Calm (zero-count) bursts skipped before the budget was consulted.
    pub skipped_calm: u32,
    /// Active bursts left for a later frame because the per-frame budget was
    /// already spent.
    pub over_budget: u32,
    /// Active bursts the channel refused because it was full (back-pressure).
    pub dropped: u32,
    /// Active bursts that found no channel registered under the requested id.
    pub missing_channel: u32,
}

/// Publishes a frame's crest-spray bursts to `Ember` over `router`'s channel
/// `channel_id`, returning the delivery tally.
///
/// The bursts are walked in input order (deterministic). A calm burst
/// ([`WaterSprayBurst::is_active`] is `false`) is tallied as `skipped_calm` and
/// never consumes budget. Each active burst consumes one unit of
/// `budget.spray_bursts_per_frame` the moment it is *selected* for delivery,
/// independent of the channel's response, so the admission schedule stays a pure
/// function of the inputs and never depends on downstream channel state. Once
/// the budget is spent, the remaining active bursts are tallied as `over_budget`
/// and left for a later frame.
///
/// A selected burst is packed into a [`ChannelRecord`] (world position, launch
/// velocity, particle count as the scalar `value`, [`SPRAY_CHANNEL_TAG`]) and
/// published. A full channel is reported as `dropped` and an unregistered id as
/// `missing_channel`; neither panics, matching the engine-wide "越界跳过不
/// panic" rule.
#[must_use]
pub fn publish_spray_bursts(
    bursts: &[WaterSprayBurst],
    budget: &WaterBudget,
    router: &mut EventRouter,
    channel_id: u32,
) -> SprayPublishStats {
    let mut stats = SprayPublishStats::default();
    let mut remaining = budget.spray_bursts_per_frame;
    for burst in bursts {
        if !burst.is_active() {
            stats.skipped_calm = stats.skipped_calm.saturating_add(1);
            continue;
        }
        if remaining == 0 {
            stats.over_budget = stats.over_budget.saturating_add(1);
            continue;
        }
        remaining -= 1;
        let record = ChannelRecord::new(
            EmberVec3::new(burst.position.x, burst.position.y, burst.position.z),
            EmberVec3::new(
                burst.emission.velocity.x,
                burst.emission.velocity.y,
                burst.emission.velocity.z,
            ),
            burst.emission.count as f32,
            SPRAY_CHANNEL_TAG,
        );
        match router.publish(channel_id, record) {
            AppendOutcome::Accepted => {
                stats.admitted = stats.admitted.saturating_add(1);
                stats.particles_admitted = stats
                    .particles_admitted
                    .saturating_add(burst.emission.count);
            }
            AppendOutcome::Dropped => stats.dropped = stats.dropped.saturating_add(1),
            AppendOutcome::NoSuchChannel => {
                stats.missing_channel = stats.missing_channel.saturating_add(1);
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL: u32 = 7;

    fn burst(x: f32, count: u32, vy: f32) -> WaterSprayBurst {
        WaterSprayBurst::new(
            Vec3::new(x, 0.0, 0.0),
            SprayEmission {
                count,
                velocity: Vec3::new(0.0, vy, 0.0),
            },
        )
    }

    fn budget(bursts: u32) -> WaterBudget {
        WaterBudget {
            spray_bursts_per_frame: bursts,
            ..WaterBudget::default()
        }
    }

    fn router_with_capacity(capacity: u32) -> EventRouter {
        let mut router = EventRouter::new();
        assert!(router.register_channel(CHANNEL, capacity));
        router
    }

    #[test]
    fn calm_bursts_are_skipped_and_never_spend_budget() {
        let bursts = [burst(0.0, 0, 0.0), burst(1.0, 0, 0.0)];
        let mut router = router_with_capacity(8);
        let stats = publish_spray_bursts(&bursts, &budget(4), &mut router, CHANNEL);
        assert_eq!(stats.skipped_calm, 2);
        assert_eq!(stats.admitted, 0);
        assert_eq!(stats.particles_admitted, 0);
        assert_eq!(router.pending_len(CHANNEL), Some(0));
    }

    #[test]
    fn active_bursts_are_packed_with_count_and_velocity() {
        let bursts = [burst(2.0, 5, 3.0)];
        let mut router = router_with_capacity(8);
        let stats = publish_spray_bursts(&bursts, &budget(4), &mut router, CHANNEL);
        assert_eq!(stats.admitted, 1);
        assert_eq!(stats.particles_admitted, 5);
        // Reads see the previous frame's publish; make this frame's writes
        // readable, then inspect the record.
        router.publish_frame();
        let snapshot = router.snapshot(CHANNEL).expect("channel exists");
        assert_eq!(snapshot.len(), 1);
        let record = snapshot[0];
        assert_eq!(record.tag, SPRAY_CHANNEL_TAG);
        assert!((record.value - 5.0).abs() < 1e-6);
        assert!((record.position.x - 2.0).abs() < 1e-6);
        assert!((record.velocity.y - 3.0).abs() < 1e-6);
    }

    #[test]
    fn budget_caps_admitted_bursts_in_input_order() {
        let bursts = [burst(0.0, 1, 1.0), burst(1.0, 2, 1.0), burst(2.0, 3, 1.0)];
        let mut router = router_with_capacity(8);
        let stats = publish_spray_bursts(&bursts, &budget(2), &mut router, CHANNEL);
        assert_eq!(stats.admitted, 2);
        assert_eq!(stats.over_budget, 1);
        // First two (counts 1 + 2) are the admitted ones; the budget is spent in
        // input order, so the third (count 3) is deferred.
        assert_eq!(stats.particles_admitted, 3);
    }

    #[test]
    fn full_channel_reports_back_pressure_without_panicking() {
        let bursts = [burst(0.0, 1, 1.0), burst(1.0, 1, 1.0), burst(2.0, 1, 1.0)];
        let mut router = router_with_capacity(1);
        let stats = publish_spray_bursts(&bursts, &budget(8), &mut router, CHANNEL);
        assert_eq!(stats.admitted, 1);
        assert_eq!(stats.dropped, 2);
        assert_eq!(router.dropped_total(CHANNEL), Some(2));
    }

    #[test]
    fn missing_channel_is_counted_not_panicked() {
        let bursts = [burst(0.0, 1, 1.0)];
        let mut router = EventRouter::new();
        let stats = publish_spray_bursts(&bursts, &budget(8), &mut router, CHANNEL);
        assert_eq!(stats.missing_channel, 1);
        assert_eq!(stats.admitted, 0);
    }

    #[test]
    fn buckets_partition_every_input_burst() {
        let bursts = [
            burst(0.0, 0, 0.0), // calm
            burst(1.0, 2, 1.0), // admitted
            burst(2.0, 3, 1.0), // over budget
        ];
        let mut router = router_with_capacity(8);
        let stats = publish_spray_bursts(&bursts, &budget(1), &mut router, CHANNEL);
        let total = stats.admitted
            + stats.skipped_calm
            + stats.over_budget
            + stats.dropped
            + stats.missing_channel;
        assert_eq!(total as usize, bursts.len());
    }

    #[test]
    fn empty_input_delivers_nothing() {
        let mut router = router_with_capacity(8);
        let stats = publish_spray_bursts(&[], &budget(8), &mut router, CHANNEL);
        assert_eq!(stats, SprayPublishStats::default());
    }
}
