//! Emission: spawn budgeting, distribution shapes, and slot allocation.
//!
//! An emitter turns an authored spawn rate and a set of bursts into a concrete
//! count of new particles this frame, samples a *distribution shape* for each
//! new particle's local spawn position and emission direction, inherits a
//! fraction of the emitter's own velocity, and draws slots from the owning
//! [`ParticlePool`]'s free list (design §8.2). Everything here is a pure,
//! deterministic contract: the same inputs (including the same stream of unit
//! random samples) always produce the same spawns, so the `CPU` reference and a
//! future `GPU` spawn kernel agree bit for bit.
//!
//! Determinism without transcendental math: the distribution shapes need points
//! on a disk, in a ball, or on a sphere, which the textbook formulations build
//! from `sin`/`cos`/`cbrt`. This crate forbids transcendental functions (only
//! `sqrt` is allowed), so the shapes are sampled by *rejection* inside a cube or
//! square — exactly uniform, using only comparisons and a final
//! [`Vec3::normalize_or_zero`]. Rejection consumes a bounded number of samples
//! from a wrapping [`UnitCursor`]; the same cursor drives the `GPU` kernel from
//! the shared hash stream (design §29), keeping the two paths in lockstep.

use alloc::vec::Vec;

use super::pool::{CapacityPolicy, ParticlePool};
use super::{EmitterHandle, SimSpace, Vec3};

/// Maximum rejection attempts before a shape sampler falls back to its axis.
///
/// A cube's inscribed ball fills `pi/6 ≈ 52%` of its volume, so eight draws
/// reject with probability `< 0.5^8 < 0.4%`; the deterministic fallback keeps
/// the sampler total (never looping forever, never panicking) for the rare
/// all-reject case and for a caller that passes too few samples.
const MAX_REJECTION_TRIES: u32 = 8;

/// A wrapping cursor over a slice of unit-interval random samples.
///
/// The backing slice is the deterministic hash-RNG output for one spawn (design
/// §29). Reads wrap so a sampler can never run past the end; an empty slice
/// yields a constant `0.5`, which keeps shape sampling total even when a caller
/// supplies no randomness (the spawn then lands at the shape's canonical point).
#[derive(Clone, Copy, Debug)]
pub struct UnitCursor<'a> {
    values: &'a [f32],
    idx: usize,
}

impl<'a> UnitCursor<'a> {
    /// Wraps a slice of samples, each expected in `0..=1`.
    #[must_use]
    pub fn new(values: &'a [f32]) -> Self {
        Self { values, idx: 0 }
    }

    /// Next sample in `[0, 1]`, wrapping; `0.5` when the slice is empty.
    pub fn next_unit(&mut self) -> f32 {
        if self.values.is_empty() {
            return 0.5;
        }
        let v = self.values[self.idx % self.values.len()];
        self.idx += 1;
        v
    }

    /// Next sample mapped to `[-1, 1]`.
    pub fn next_signed(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

/// A point in the unit disk `x^2 + y^2 <= 1`, sampled by rejection.
///
/// Falls back to the disk center after [`MAX_REJECTION_TRIES`] rejects.
fn sample_unit_disk(cursor: &mut UnitCursor<'_>) -> (f32, f32) {
    for _ in 0..MAX_REJECTION_TRIES {
        let x = cursor.next_signed();
        let y = cursor.next_signed();
        if x * x + y * y <= 1.0 {
            return (x, y);
        }
    }
    (0.0, 0.0)
}

/// A point in the unit ball `|p| <= 1`, sampled by rejection.
///
/// Falls back to the ball center after [`MAX_REJECTION_TRIES`] rejects.
fn sample_unit_ball(cursor: &mut UnitCursor<'_>) -> Vec3 {
    for _ in 0..MAX_REJECTION_TRIES {
        let p = Vec3::new(
            cursor.next_signed(),
            cursor.next_signed(),
            cursor.next_signed(),
        );
        if p.length_squared() <= 1.0 {
            return p;
        }
    }
    Vec3::ZERO
}

/// The spatial distribution a particle spawns from, in emitter-local space
/// (design §8.2). The emission axis is local `+Z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterShape {
    /// All particles spawn at the origin and emit along `+Z`.
    Point,
    /// Spawn inside (or, when `surface_only`, on) a ball of `radius`, emitting
    /// radially outward.
    Sphere {
        /// Ball radius; non-positive collapses to a point.
        radius: f32,
        /// `true` samples the shell, `false` fills the volume.
        surface_only: bool,
    },
    /// Spawn inside an axis-aligned box, emitting along `+Z`.
    Box {
        /// Half-extents on each axis; the full box is `[-h, h]` per axis.
        half_extents: Vec3,
    },
    /// Emit from the apex in a cone of directions opening toward `+Z`, defined
    /// by the base disk of `base_radius` at `height`.
    Cone {
        /// Radius of the cone's base disk.
        base_radius: f32,
        /// Distance from apex to base plane along `+Z`; non-positive collapses
        /// to `+Z`.
        height: f32,
    },
}

/// One sampled spawn placement in emitter-local space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnSample {
    /// Local spawn position offset from the emitter origin.
    pub position: Vec3,
    /// Unit emission direction, or `+Z` for a degenerate shape.
    pub direction: Vec3,
}

/// Local `+Z`, the canonical emission axis and degenerate-shape fallback.
const EMIT_AXIS: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// Samples a spawn placement for `shape`, consuming unit samples from `cursor`.
///
/// The result is fully determined by `shape` and the `cursor` contents, so the
/// spawn is reproducible whenever the hash-RNG stream is.
#[must_use]
pub fn sample_shape(shape: EmitterShape, cursor: &mut UnitCursor<'_>) -> SpawnSample {
    match shape {
        EmitterShape::Point => SpawnSample {
            position: Vec3::ZERO,
            direction: EMIT_AXIS,
        },
        EmitterShape::Sphere {
            radius,
            surface_only,
        } => {
            let p = sample_unit_ball(cursor);
            let dir = if p == Vec3::ZERO {
                EMIT_AXIS
            } else {
                p.normalize_or_zero()
            };
            let position = if surface_only {
                dir.scale(radius.max(0.0))
            } else {
                p.scale(radius.max(0.0))
            };
            SpawnSample {
                position,
                direction: dir,
            }
        }
        EmitterShape::Box { half_extents } => {
            let position = Vec3::new(
                cursor.next_signed() * half_extents.x,
                cursor.next_signed() * half_extents.y,
                cursor.next_signed() * half_extents.z,
            );
            SpawnSample {
                position,
                direction: EMIT_AXIS,
            }
        }
        EmitterShape::Cone {
            base_radius,
            height,
        } => {
            let (dx, dy) = sample_unit_disk(cursor);
            let base = Vec3::new(dx * base_radius, dy * base_radius, height.max(0.0));
            let direction = if base == Vec3::ZERO {
                EMIT_AXIS
            } else {
                base.normalize_or_zero()
            };
            SpawnSample {
                position: Vec3::ZERO,
                direction,
            }
        }
    }
}

/// Fractional-carry accumulator turning a continuous spawn rate into an integer
/// count per frame (design §8.2).
///
/// A rate of `r` particles per second over a frame of `dt` seconds owes
/// `r * dt` particles; the fractional remainder is carried to the next frame so
/// the long-run average rate is exact and no particles are lost to truncation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpawnAccumulator {
    carry: f32,
}

impl SpawnAccumulator {
    /// A fresh accumulator with no carried remainder.
    #[must_use]
    pub fn new() -> Self {
        Self { carry: 0.0 }
    }

    /// The remainder carried toward the next frame, in `[0, 1)`.
    #[must_use]
    pub fn carry(self) -> f32 {
        self.carry
    }

    /// Accumulates `rate * dt` particles and returns the whole count to spawn,
    /// carrying the fractional part.
    ///
    /// Non-finite or non-positive `rate`/`dt` contribute nothing and leave the
    /// carry untouched, so a paused or malformed emitter never spawns.
    pub fn accumulate(&mut self, rate_per_second: f32, dt: f32) -> u32 {
        if rate_per_second > 0.0 && dt > 0.0 {
            self.carry += rate_per_second * dt;
            // `as u32` truncates toward zero; the carry is non-negative so this
            // is floor. Saturates rather than wrapping on an extreme rate.
            let whole = if self.carry >= u32::MAX as f32 {
                u32::MAX
            } else {
                self.carry as u32
            };
            self.carry -= whole as f32;
            whole
        } else {
            // A paused, malformed (`NaN`), or backward step spawns nothing and
            // leaves the carry untouched.
            0
        }
    }
}

/// A one-shot spawn of `count` particles at simulation time `time`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BurstSpawn {
    /// Emitter-local time (seconds) at which the burst fires.
    pub time: f32,
    /// Particles released by the burst.
    pub count: u32,
}

/// Sums the bursts firing in the half-open window `(start, end]`.
///
/// The window is half-open so a burst fires exactly once as time advances past
/// its timestamp, never on the frame whose interval merely ends at `start`.
/// A backward or empty window (`end <= start`) fires nothing.
#[must_use]
pub fn bursts_in_window(bursts: &[BurstSpawn], start: f32, end: f32) -> u32 {
    if end > start {
        let mut total: u32 = 0;
        for burst in bursts {
            if burst.time > start && burst.time <= end {
                total = total.saturating_add(burst.count);
            }
        }
        total
    } else {
        // A backward, empty, or `NaN` window fires nothing.
        0
    }
}

/// Authored per-spawn parameters bound onto each new particle (design §8.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnParams {
    /// Initial speed along the sampled emission direction.
    pub speed: f32,
    /// Fraction of the emitter's velocity added to each particle (`0` detaches,
    /// `1` fully inherits).
    pub inherit_velocity: f32,
    /// Whether spawns stay in emitter-local space or detach into world space
    /// (design §26).
    pub sim_space: SimSpace,
}

impl Default for SpawnParams {
    fn default() -> Self {
        Self {
            speed: 0.0,
            inherit_velocity: 0.0,
            sim_space: SimSpace::World,
        }
    }
}

/// The initial state written into a particle's slot at spawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnState {
    /// Initial position (world space unless the emitter is [`SimSpace::Local`]).
    pub position: Vec3,
    /// Initial velocity: emission plus inherited emitter motion.
    pub velocity: Vec3,
}

/// Velocity a spawn inherits from its emitter.
#[must_use]
pub fn inherited_velocity(emitter_velocity: Vec3, factor: f32) -> Vec3 {
    emitter_velocity.scale(factor)
}

/// Builds the initial [`SpawnState`] for one particle.
///
/// The sampled local offset is placed relative to the emitter `origin` for
/// world/hybrid spaces, or kept as a local offset for [`SimSpace::Local`]. The
/// velocity is the emission direction scaled by [`SpawnParams::speed`] plus the
/// inherited emitter velocity.
#[must_use]
pub fn build_spawn(
    origin: Vec3,
    emitter_velocity: Vec3,
    sample: SpawnSample,
    params: SpawnParams,
) -> SpawnState {
    let position = match params.sim_space {
        SimSpace::Local => sample.position,
        SimSpace::World | SimSpace::Hybrid => origin.add(sample.position),
    };
    let velocity = sample.direction.scale(params.speed).add(inherited_velocity(
        emitter_velocity,
        params.inherit_velocity,
    ));
    SpawnState { position, velocity }
}

/// Allocates up to `count` pool slots for this frame's spawns.
///
/// Slots come from the pool's free list in allocation order; allocation stops
/// early only when [`CapacityPolicy::DiscardNewest`] hits a full pool, so the
/// returned length is the number of particles that actually spawned. With
/// [`CapacityPolicy::RecycleOldest`] every request is satisfied by evicting the
/// oldest live particle.
#[must_use]
pub fn allocate_spawns(pool: &mut ParticlePool, count: u32, policy: CapacityPolicy) -> Vec<u32> {
    let mut slots = Vec::new();
    for _ in 0..count {
        match pool.spawn(policy) {
            Some(slot) => slots.push(slot),
            None => break,
        }
    }
    slots
}

/// A configured emitter: its handle, distribution shape, spawn parameters, and
/// the running rate accumulator.
///
/// This bundles the per-frame emission decision so a system can advance many
/// emitters uniformly. It owns no particle storage; slots are drawn from the
/// [`ParticlePool`] passed to [`Emitter::allocate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// Stable emitter identity within its system.
    pub handle: EmitterHandle,
    /// Distribution the emitter spawns from.
    pub shape: EmitterShape,
    /// Per-spawn authored parameters.
    pub params: SpawnParams,
    accumulator: SpawnAccumulator,
}

impl Emitter {
    /// Builds an emitter with a fresh (zeroed) rate accumulator.
    #[must_use]
    pub fn new(handle: EmitterHandle, shape: EmitterShape, params: SpawnParams) -> Self {
        Self {
            handle,
            shape,
            params,
            accumulator: SpawnAccumulator::new(),
        }
    }

    /// The remainder carried by the rate accumulator toward the next frame.
    #[must_use]
    pub fn rate_carry(&self) -> f32 {
        self.accumulator.carry()
    }

    /// Total particles to spawn this frame: the continuous rate contribution
    /// plus every burst firing in `(time - dt, time]`.
    pub fn spawn_count(
        &mut self,
        rate_per_second: f32,
        dt: f32,
        bursts: &[BurstSpawn],
        time: f32,
    ) -> u32 {
        let from_rate = self.accumulator.accumulate(rate_per_second, dt);
        let window_start = time - dt;
        let from_bursts = bursts_in_window(bursts, window_start, time);
        from_rate.saturating_add(from_bursts)
    }

    /// Allocates `count` slots from `pool` under the given capacity policy.
    #[must_use]
    pub fn allocate(
        &self,
        pool: &mut ParticlePool,
        count: u32,
        policy: CapacityPolicy,
    ) -> Vec<u32> {
        allocate_spawns(pool, count, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rate_and_dt_spawn_nothing() {
        let mut acc = SpawnAccumulator::new();
        assert_eq!(acc.accumulate(0.0, 0.016), 0);
        assert_eq!(acc.accumulate(100.0, 0.0), 0);
        assert_eq!(acc.accumulate(f32::NAN, 0.016), 0);
        assert_eq!(acc.carry(), 0.0);
    }

    #[test]
    fn rate_accumulates_fractional_carry_exactly() {
        let mut acc = SpawnAccumulator::new();
        // 30 particles/sec at 60 fps => 0.5 per frame: 0,1,0,1,...
        assert_eq!(acc.accumulate(30.0, 1.0 / 60.0), 0);
        assert!((acc.carry() - 0.5).abs() < 1e-6);
        assert_eq!(acc.accumulate(30.0, 1.0 / 60.0), 1);
        assert!(acc.carry().abs() < 1e-6);
    }

    #[test]
    fn rate_long_run_average_is_exact() {
        let mut acc = SpawnAccumulator::new();
        let mut total = 0u32;
        for _ in 0..600 {
            total += acc.accumulate(100.0, 1.0 / 60.0);
        }
        // 100/s for 10 simulated seconds => 1000 particles.
        assert_eq!(total, 1000);
    }

    #[test]
    fn bursts_fire_once_in_half_open_window() {
        let bursts = [
            BurstSpawn {
                time: 0.5,
                count: 10,
            },
            BurstSpawn {
                time: 1.0,
                count: 5,
            },
        ];
        // Window (0.4, 0.6] catches only the first burst.
        assert_eq!(bursts_in_window(&bursts, 0.4, 0.6), 10);
        // Boundary at start is excluded, boundary at end is included.
        assert_eq!(bursts_in_window(&bursts, 0.5, 1.0), 5);
        // Backward window fires nothing.
        assert_eq!(bursts_in_window(&bursts, 1.0, 0.5), 0);
    }

    #[test]
    fn point_shape_spawns_at_origin_along_axis() {
        let mut cursor = UnitCursor::new(&[]);
        let s = sample_shape(EmitterShape::Point, &mut cursor);
        assert_eq!(s.position, Vec3::ZERO);
        assert_eq!(s.direction, EMIT_AXIS);
    }

    #[test]
    fn sphere_surface_sample_lands_on_radius() {
        let samples = [0.9, 0.1, 0.8];
        let mut cursor = UnitCursor::new(&samples);
        let s = sample_shape(
            EmitterShape::Sphere {
                radius: 3.0,
                surface_only: true,
            },
            &mut cursor,
        );
        assert!((s.position.length() - 3.0).abs() < 1e-5);
        assert!((s.direction.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_volume_sample_stays_within_radius() {
        let samples = [0.9, 0.1, 0.8];
        let mut cursor = UnitCursor::new(&samples);
        let s = sample_shape(
            EmitterShape::Sphere {
                radius: 3.0,
                surface_only: false,
            },
            &mut cursor,
        );
        assert!(s.position.length() <= 3.0 + 1e-5);
    }

    #[test]
    fn box_sample_stays_within_half_extents() {
        let samples = [1.0, 0.0, 0.5];
        let mut cursor = UnitCursor::new(&samples);
        let h = Vec3::new(2.0, 4.0, 6.0);
        let s = sample_shape(EmitterShape::Box { half_extents: h }, &mut cursor);
        assert!(s.position.x.abs() <= h.x + 1e-6);
        assert!(s.position.y.abs() <= h.y + 1e-6);
        assert!(s.position.z.abs() <= h.z + 1e-6);
        assert_eq!(s.direction, EMIT_AXIS);
    }

    #[test]
    fn cone_direction_is_unit_and_within_cone() {
        let samples = [0.75, 0.5];
        let mut cursor = UnitCursor::new(&samples);
        let s = sample_shape(
            EmitterShape::Cone {
                base_radius: 1.0,
                height: 1.0,
            },
            &mut cursor,
        );
        assert!((s.direction.length() - 1.0).abs() < 1e-6);
        // Height==base_radius => half-angle 45deg => z component >= cos(45).
        let cos_45 = 0.5_f32.sqrt();
        assert!(s.direction.z >= cos_45 - 1e-4);
    }

    #[test]
    fn shape_sampling_is_deterministic() {
        let samples = [0.3, 0.7, 0.2, 0.9, 0.55];
        let shape = EmitterShape::Sphere {
            radius: 2.0,
            surface_only: false,
        };
        let mut a = UnitCursor::new(&samples);
        let mut b = UnitCursor::new(&samples);
        assert_eq!(sample_shape(shape, &mut a), sample_shape(shape, &mut b));
    }

    #[test]
    fn velocity_inheritance_and_emission_compose() {
        let params = SpawnParams {
            speed: 5.0,
            inherit_velocity: 0.5,
            sim_space: SimSpace::World,
        };
        let state = build_spawn(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            SpawnSample {
                position: Vec3::new(0.0, 0.0, 0.0),
                direction: Vec3::new(0.0, 0.0, 1.0),
            },
            params,
        );
        assert_eq!(state.position, Vec3::new(1.0, 0.0, 0.0));
        // 5 along +Z emission, plus half of the emitter's 10 along +Y.
        assert_eq!(state.velocity, Vec3::new(0.0, 5.0, 5.0));
    }

    #[test]
    fn local_space_keeps_spawn_offset_relative() {
        let params = SpawnParams {
            sim_space: SimSpace::Local,
            ..SpawnParams::default()
        };
        let state = build_spawn(
            Vec3::new(9.0, 9.0, 9.0),
            Vec3::ZERO,
            SpawnSample {
                position: Vec3::new(0.5, 0.0, 0.0),
                direction: EMIT_AXIS,
            },
            params,
        );
        assert_eq!(state.position, Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn allocate_discards_when_pool_full() {
        let mut pool = ParticlePool::with_capacity(2);
        let slots = allocate_spawns(&mut pool, 5, CapacityPolicy::DiscardNewest);
        assert_eq!(slots, [0, 1]);
        assert!(pool.is_full());
    }

    #[test]
    fn allocate_recycle_satisfies_every_request() {
        let mut pool = ParticlePool::with_capacity(2);
        let slots = allocate_spawns(&mut pool, 4, CapacityPolicy::RecycleOldest);
        assert_eq!(slots.len(), 4);
        assert!(pool.is_full());
    }

    #[test]
    fn emitter_combines_rate_and_bursts() {
        let mut emitter = Emitter::new(
            EmitterHandle(1),
            EmitterShape::Point,
            SpawnParams::default(),
        );
        let bursts = [BurstSpawn {
            time: 0.5,
            count: 7,
        }];
        // 60/s at dt=1/60 => 1 from rate, plus the burst at t=0.5 in (0.484, 0.5].
        let n = emitter.spawn_count(60.0, 1.0 / 60.0, &bursts, 0.5);
        assert_eq!(n, 1 + 7);
    }
}
