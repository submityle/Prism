//! Bounded server-side lag compensation (design §11, feature `lag_comp`).
//!
//! [`LagCompensator`] keeps a bounded, per-entity history of world-space
//! axis-aligned hit boxes. A hit proposal names the simulation tick the client
//! saw; the server can then run a deterministic ray query against the retained
//! historical boxes instead of mutating the live ECS world. The module is a
//! pure `no_std + alloc` kernel: entity ownership, authoritative event
//! emission, anti-cheat policy, and narrow-phase shape providers remain in the
//! server layer.
//!
//! The implementation deliberately uses exact tick samples rather than
//! interpolating between them. Hit validation must be reproducible across
//! machines, and a caller that needs a sub-tick rule can quantize its input to
//! a tick before calling [`LagCompensator::query_ray`]. Requests outside the
//! configured rewind window are rejected instead of silently falling back to
//! the current state.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;

use prism_math::{Aabb3, Vec3};

/// Stable application-owned identity for an entity in the lag-compensation
/// history. It is intentionally not an ECS entity handle: network/session
/// identity must survive ECS allocation changes.
pub type EntityId = u64;

/// Configuration for a bounded historical hit-test window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LagCompConfig {
    /// Maximum number of tick samples retained for each entity.
    pub history_ticks: usize,
    /// Maximum number of ticks an attack may rewind from the current server
    /// tick. This must be smaller than `history_ticks`.
    pub max_rewind_ticks: u64,
}

impl LagCompConfig {
    /// Construct a bounded configuration.
    #[inline]
    pub const fn new(history_ticks: usize, max_rewind_ticks: u64) -> Self {
        Self {
            history_ticks,
            max_rewind_ticks,
        }
    }
}

/// A normalized ray used by the historical hit query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// Ray origin in world space.
    pub origin: Vec3,
    /// Unit-length ray direction.
    pub direction: Vec3,
}

impl Ray {
    /// Build a ray and normalize its direction.
    ///
    /// A zero, non-finite, or otherwise unusable direction is rejected rather
    /// than turning into a broad-phase query that can never terminate
    /// meaningfully.
    pub fn new(origin: Vec3, direction: Vec3) -> Result<Self, RayError> {
        if !origin.is_finite() || !direction.is_finite() {
            return Err(RayError::NonFinite);
        }
        let direction = direction.normalize_or_zero();
        if direction.length_squared() <= f32::EPSILON {
            return Err(RayError::ZeroDirection);
        }
        Ok(Self { origin, direction })
    }
}

/// Failure while constructing a [`Ray`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayError {
    /// The origin or direction contained NaN or infinity.
    NonFinite,
    /// The direction had no usable length.
    ZeroDirection,
}

/// A historical hit result, sorted by increasing ray distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Entity whose retained historical hit box was intersected.
    pub entity: EntityId,
    /// Historical tick used for this hit.
    pub tick: u64,
    /// Distance along the normalized ray to the first intersection point.
    pub distance: f32,
    /// First intersection point in world space.
    pub point: Vec3,
}

/// Result of a historical ray query.
#[derive(Clone, Debug, PartialEq)]
pub struct HitReport {
    /// Tick requested by the client.
    pub requested_tick: u64,
    /// Number of entities whose exact sample was tested.
    pub tested_entities: usize,
    /// Entities that had no sample at the requested tick, for example because
    /// they spawned after that tick or their history was sparse.
    pub skipped_entities: usize,
    /// All intersections, nearest first and entity-id stable on ties.
    pub hits: Vec<Hit>,
}

/// Errors that refuse an unsafe or ambiguous history operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LagCompError {
    /// `history_ticks` must be non-zero and larger than `max_rewind_ticks`.
    InvalidConfig {
        /// Requested sample capacity.
        history_ticks: usize,
        /// Requested rewind window.
        max_rewind_ticks: u64,
    },
    /// The hit box contained a non-finite coordinate.
    NonFiniteBounds(EntityId),
    /// A frame was submitted older than the already accepted global tick.
    NonMonotonicFrame {
        /// Newest frame already accepted.
        current: u64,
        /// Older frame that was submitted.
        received: u64,
    },
    /// An entity's samples were submitted out of order.
    NonMonotonicEntity {
        /// Entity whose history was invalid.
        entity: EntityId,
        /// Last retained tick for that entity.
        previous: u64,
        /// Newly submitted tick.
        received: u64,
    },
    /// A ray query was submitted before any server frame was recorded.
    NoHistory,
    /// The requested tick is in the future relative to the server.
    FutureTick {
        /// Newest server tick already accepted.
        current: u64,
        /// Future tick that was requested.
        requested: u64,
    },
    /// The requested tick exceeds the configured rewind policy.
    RewindTooOld {
        /// Earliest tick permitted by the policy.
        earliest: u64,
        /// Requested tick.
        requested: u64,
    },
    /// The query distance was invalid.
    InvalidDistance,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HistorySample {
    tick: u64,
    bounds: Aabb3,
}

#[derive(Clone, Debug, Default)]
struct EntityHistory {
    samples: VecDeque<HistorySample>,
}

impl EntityHistory {
    fn last_tick(&self) -> Option<u64> {
        self.samples.back().map(|sample| sample.tick)
    }

    fn push(&mut self, sample: HistorySample, capacity: usize) {
        if self.last_tick() == Some(sample.tick) {
            *self.samples.back_mut().expect("last tick exists") = sample;
            return;
        }
        self.samples.push_back(sample);
        while self.samples.len() > capacity {
            self.samples.pop_front();
        }
    }

    fn at(&self, tick: u64) -> Option<HistorySample> {
        let first = self.samples.front()?.tick;
        let last = self.samples.back()?.tick;
        if tick < first || tick > last {
            return None;
        }
        let mut lo = 0;
        let mut hi = self.samples.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.samples.get(mid).map(|sample| sample.tick) {
                Some(value) if value < tick => lo = mid + 1,
                Some(value) if value > tick => hi = mid,
                Some(_) => return self.samples.get(mid).copied(),
                None => return None,
            }
        }
        None
    }
}

/// Bounded historical hit-test storage for an authoritative server.
#[derive(Clone, Debug)]
pub struct LagCompensator {
    config: LagCompConfig,
    current_tick: Option<u64>,
    histories: BTreeMap<EntityId, EntityHistory>,
}

impl LagCompensator {
    /// Create an empty compensator.
    pub fn new(config: LagCompConfig) -> Result<Self, LagCompError> {
        if config.history_ticks == 0 || config.max_rewind_ticks >= config.history_ticks as u64 {
            return Err(LagCompError::InvalidConfig {
                history_ticks: config.history_ticks,
                max_rewind_ticks: config.max_rewind_ticks,
            });
        }
        Ok(Self {
            config,
            current_tick: None,
            histories: BTreeMap::new(),
        })
    }

    /// The configured retention and rewind policy.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> LagCompConfig {
        self.config
    }

    /// The newest server tick accepted so far.
    #[inline]
    #[must_use]
    pub const fn current_tick(&self) -> Option<u64> {
        self.current_tick
    }

    /// Number of entities currently retained.
    #[inline]
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.histories.len()
    }

    /// Record one tick containing zero or more entity hit boxes.
    ///
    /// The input slice is validated before any history is mutated. Duplicate
    /// entity ids in one frame are rejected by [`LagCompError::NonMonotonicEntity`]
    /// semantics (the duplicate is not a second authoritative sample).
    pub fn record_frame(
        &mut self,
        tick: u64,
        samples: &[(EntityId, Aabb3)],
    ) -> Result<(), LagCompError> {
        if let Some(current) = self.current_tick
            && tick < current
        {
            return Err(LagCompError::NonMonotonicFrame {
                current,
                received: tick,
            });
        }

        let mut seen = BTreeSet::new();
        for &(entity, bounds) in samples {
            if !seen.insert(entity) {
                let previous = self
                    .histories
                    .get(&entity)
                    .and_then(EntityHistory::last_tick)
                    .unwrap_or(tick);
                return Err(LagCompError::NonMonotonicEntity {
                    entity,
                    previous,
                    received: tick,
                });
            }
            if !bounds.min.is_finite() || !bounds.max.is_finite() {
                return Err(LagCompError::NonFiniteBounds(entity));
            }
            if let Some(previous) = self
                .histories
                .get(&entity)
                .and_then(EntityHistory::last_tick)
                && tick < previous
            {
                return Err(LagCompError::NonMonotonicEntity {
                    entity,
                    previous,
                    received: tick,
                });
            }
        }

        self.current_tick = Some(self.current_tick.map_or(tick, |current| current.max(tick)));
        for &(entity, bounds) in samples {
            self.histories
                .entry(entity)
                .or_default()
                .push(HistorySample { tick, bounds }, self.config.history_ticks);
        }
        Ok(())
    }

    /// Record one entity's hit box for a tick.
    #[inline]
    pub fn record(
        &mut self,
        entity: EntityId,
        tick: u64,
        bounds: Aabb3,
    ) -> Result<(), LagCompError> {
        self.record_frame(tick, &[(entity, bounds)])
    }

    /// Remove an entity from all retained historical frames.
    #[inline]
    pub fn remove(&mut self, entity: EntityId) -> bool {
        self.histories.remove(&entity).is_some()
    }

    /// Query retained historical hit boxes with a normalized ray.
    pub fn query_ray(
        &self,
        requested_tick: u64,
        ray: Ray,
        max_distance: f32,
    ) -> Result<HitReport, LagCompError> {
        let current = self.current_tick.ok_or(LagCompError::NoHistory)?;
        if requested_tick > current {
            return Err(LagCompError::FutureTick {
                current,
                requested: requested_tick,
            });
        }
        let earliest = current.saturating_sub(self.config.max_rewind_ticks);
        if requested_tick < earliest {
            return Err(LagCompError::RewindTooOld {
                earliest,
                requested: requested_tick,
            });
        }
        if !max_distance.is_finite() || max_distance < 0.0 {
            return Err(LagCompError::InvalidDistance);
        }

        let mut hits = Vec::new();
        let mut tested_entities = 0;
        let mut skipped_entities = 0;
        for (&entity, history) in &self.histories {
            let Some(sample) = history.at(requested_tick) else {
                skipped_entities += 1;
                continue;
            };
            tested_entities += 1;
            if let Some(distance) = ray_aabb_distance(ray, sample.bounds, max_distance) {
                hits.push(Hit {
                    entity,
                    tick: requested_tick,
                    distance,
                    point: ray.origin + ray.direction * distance,
                });
            }
        }
        hits.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.entity.cmp(&right.entity))
        });
        Ok(HitReport {
            requested_tick,
            tested_entities,
            skipped_entities,
            hits,
        })
    }
}

fn ray_aabb_distance(ray: Ray, bounds: Aabb3, max_distance: f32) -> Option<f32> {
    let mut near: f32 = 0.0;
    let mut far: f32 = max_distance;
    for (origin, direction, min, max) in [
        (ray.origin.x, ray.direction.x, bounds.min.x, bounds.max.x),
        (ray.origin.y, ray.direction.y, bounds.min.y, bounds.max.y),
        (ray.origin.z, ray.direction.z, bounds.min.z, bounds.max.z),
    ] {
        if direction.abs() <= f32::EPSILON {
            if origin < min || origin > max {
                return None;
            }
            continue;
        }
        let inv = 1.0 / direction;
        let mut t0 = (min - origin) * inv;
        let mut t1 = (max - origin) * inv;
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
        }
        near = near.max(t0);
        far = far.min(t1);
        if near > far {
            return None;
        }
    }
    Some(near.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn box_at(x: f32) -> Aabb3 {
        Aabb3::from_center_half_extents(Vec3::new(x, 0.0, 0.0), Vec3::splat(0.5))
    }

    fn ray() -> Ray {
        Ray::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X).expect("valid ray")
    }

    #[test]
    fn rejects_unbounded_or_empty_config() {
        assert!(matches!(
            LagCompensator::new(LagCompConfig::new(0, 0)),
            Err(LagCompError::InvalidConfig {
                history_ticks: 0,
                max_rewind_ticks: 0,
            })
        ));
        assert!(LagCompensator::new(LagCompConfig::new(4, 4)).is_err());
    }

    #[test]
    fn queries_the_requested_historical_tick_not_the_live_pose() {
        let mut lag = LagCompensator::new(LagCompConfig::new(8, 5)).unwrap();
        lag.record(7, 10, box_at(0.0)).unwrap();
        lag.record(7, 11, box_at(5.0)).unwrap();

        let old = lag.query_ray(10, ray(), 20.0).unwrap();
        assert_eq!(old.hits[0].entity, 7);
        assert!((old.hits[0].distance - 9.5).abs() < 1.0e-6);

        let live = lag.query_ray(11, ray(), 12.0).unwrap();
        assert!(live.hits.is_empty());
    }

    #[test]
    fn retains_a_bounded_window_and_reports_sparse_entities() {
        let mut lag = LagCompensator::new(LagCompConfig::new(3, 2)).unwrap();
        lag.record(1, 1, box_at(0.0)).unwrap();
        lag.record_frame(2, &[]).unwrap();
        lag.record(1, 3, box_at(0.0)).unwrap();
        lag.record(2, 3, box_at(2.0)).unwrap();
        lag.record(1, 4, box_at(0.0)).unwrap();

        let report = lag.query_ray(2, ray(), 20.0).unwrap();
        assert_eq!(report.tested_entities, 0);
        assert_eq!(report.skipped_entities, 2);
        assert!(report.hits.is_empty());
    }

    #[test]
    fn sorts_hits_by_distance_and_entity_id() {
        let mut lag = LagCompensator::new(LagCompConfig::new(4, 3)).unwrap();
        lag.record(20, 1, box_at(3.0)).unwrap();
        lag.record(10, 1, box_at(0.0)).unwrap();
        let report = lag.query_ray(1, ray(), 20.0).unwrap();
        assert_eq!(
            report.hits.iter().map(|hit| hit.entity).collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn rejects_future_old_and_invalid_queries() {
        let mut lag = LagCompensator::new(LagCompConfig::new(4, 2)).unwrap();
        lag.record(1, 10, box_at(0.0)).unwrap();
        assert_eq!(
            lag.query_ray(11, ray(), 20.0),
            Err(LagCompError::FutureTick {
                current: 10,
                requested: 11,
            })
        );
        assert_eq!(
            lag.query_ray(7, ray(), 20.0),
            Err(LagCompError::RewindTooOld {
                earliest: 8,
                requested: 7,
            })
        );
        assert_eq!(
            lag.query_ray(10, ray(), f32::NAN),
            Err(LagCompError::InvalidDistance)
        );
    }

    #[test]
    fn rejects_non_monotonic_frames_and_non_finite_bounds_without_mutation() {
        let mut lag = LagCompensator::new(LagCompConfig::new(4, 2)).unwrap();
        lag.record(1, 5, box_at(0.0)).unwrap();
        assert!(matches!(
            lag.record(1, 4, box_at(0.0)),
            Err(LagCompError::NonMonotonicFrame { .. })
        ));
        let invalid = Aabb3::from_center_half_extents(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ONE);
        assert_eq!(
            lag.record(2, 6, invalid),
            Err(LagCompError::NonFiniteBounds(2))
        );
        assert_eq!(lag.current_tick(), Some(5));
    }

    #[test]
    fn ray_constructor_rejects_invalid_directions() {
        assert_eq!(
            Ray::new(Vec3::ZERO, Vec3::ZERO),
            Err(RayError::ZeroDirection)
        );
        assert_eq!(
            Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X),
            Err(RayError::NonFinite)
        );
    }
}
