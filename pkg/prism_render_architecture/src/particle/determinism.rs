//! Determinism and networking contracts for the Ember particle engine
//! (design §29).
//!
//! Two properties make a `GPU`-driven particle engine safe to use in
//! gameplay-affecting, networked, or recorded contexts:
//!
//! 1. **Bit-reproducible randomness.** Every random draw is a pure function of
//!    stable integer inputs — `particle_id`, a global seed, a *stream*
//!    namespace, a frame index, and a per-draw sub-index — rather than a
//!    mutable cursor. A `CPU` reference and a `GPU` compute kernel that hash the
//!    same inputs produce the same bits regardless of thread scheduling, and a
//!    replay can re-derive any historical value without storing it.
//! 2. **An explicit sync policy.** Most particles are pure presentation and
//!    must never consume network bandwidth; the few that drive gameplay run on
//!    a deterministic `CPU` path so every peer agrees. [`NetworkSyncPolicy`]
//!    and [`select_sync_policy`] make that split a first-class decision.
//!
//! The randomness core is a *stateless hash* `RNG` built from a `SplitMix` /
//! `PCG`-style integer finalizer (pure `u32`/`u64` bit operations). No
//! transcendental function is used anywhere: floating point appears only for
//! the final integer-to-`f32` mapping and for the `sqrt`-based normalization of
//! sampled directions (both allowed by the workspace determinism lint), so the
//! `dt`-fixed simulation stays reproducible against a future `GPU` kernel.

use super::{EmitterHandle, ParticleSystemHandle, Vec3, EPS_LEN_SQ};

/// Golden-ratio odd constant seeding the hash avalanche.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
/// Odd multiplier decorrelating the global-seed input.
const MUL_SEED: u64 = 0xff51_afd7_ed55_8ccd;
/// Odd multiplier decorrelating the particle-identity input.
const MUL_ID: u64 = 0xc4ce_b9fe_1a85_ec53;
/// Odd multiplier decorrelating the stream-namespace input.
const MUL_STREAM: u64 = 0xd6e8_feb8_6659_fd93;
/// Odd multiplier decorrelating the frame input.
const MUL_FRAME: u64 = 0xa076_1d64_78bd_642f;
/// Odd multiplier decorrelating the per-draw sub-index input.
const MUL_INDEX: u64 = 0x8ebc_6af0_9c88_c6e3;

/// Reciprocal of `2^24`, used to scale a 24-bit integer into `[0, 1)`.
///
/// A 24-bit payload is exactly representable in an `f32` mantissa, so the
/// multiply is exact and the result is uniform. Multiplying by a constant
/// reciprocal is ordinary arithmetic, never a transcendental call.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// `SplitMix64` finalizer: a strong integer bit-mixer with good avalanche.
///
/// This is the same family of mixer used by `PCG`-style generators; it takes an
/// arbitrary `u64` and scrambles every input bit across the output using only
/// shifts, xors, and odd multiplies. It is a pure function, so it is trivially
/// reproducible on any backend.
#[must_use]
const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Maps a 32-bit hash word to a uniform `f32` in `[0, 1)`.
///
/// Only the high 24 bits are used so the value lands exactly on a multiple of
/// `2^-24`; this keeps the mapping uniform and exactly representable without a
/// division by a transcendental quantity.
#[must_use]
pub fn unit_f32_from_bits(bits: u32) -> f32 {
    (bits >> 8) as f32 * INV_2POW24
}

/// A named random-stream namespace (design §29).
///
/// Independent particle attributes must draw from *decorrelated* streams so
/// that, for example, jittering a particle's lifetime never nudges its spawn
/// color. Each variant maps to a distinct namespace constant folded into the
/// hash; [`StreamId::Custom`] lets a user `WESL` module carve out its own
/// namespace without colliding with the built-in channels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StreamId {
    /// Initial spawn position offset / distribution sampling.
    InitialPosition,
    /// Initial velocity direction and speed.
    InitialVelocity,
    /// Spawn color / palette selection.
    Color,
    /// Per-particle lifetime jitter.
    LifetimeJitter,
    /// Per-particle size jitter.
    SizeJitter,
    /// Initial rotation / angular velocity.
    Rotation,
    /// Sub-`UV` / texture-sheet frame selection.
    SubUv,
    /// A user-defined stream namespace (design §19 custom modules).
    Custom(u32),
}

impl StreamId {
    /// Folds the stream into its `u32` namespace constant.
    ///
    /// Built-in streams occupy a small reserved block; [`StreamId::Custom`]
    /// values are lifted above it so a user namespace can never alias a
    /// built-in one.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        match self {
            StreamId::InitialPosition => 0x0001_0000,
            StreamId::InitialVelocity => 0x0002_0000,
            StreamId::Color => 0x0003_0000,
            StreamId::LifetimeJitter => 0x0004_0000,
            StreamId::SizeJitter => 0x0005_0000,
            StreamId::Rotation => 0x0006_0000,
            StreamId::SubUv => 0x0007_0000,
            StreamId::Custom(id) => 0x8000_0000 ^ id,
        }
    }
}

/// A stateless hash-`RNG` key: the complete set of stable inputs from which any
/// random value is re-derived (design §29).
///
/// The key carries no mutable cursor. Every draw is `hash(particle_id,
/// global_seed, stream_id, frame, index)`, so the same key always yields the
/// same value and any historical draw can be replayed on demand. The
/// per-draw `index` lets one logical event (a spawn) consume several
/// decorrelated values without allocating extra streams.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RngKey {
    /// Stable per-particle identity (a pool slot's stable id, not its index).
    particle_id: u32,
    /// The effect-wide deterministic seed (see [`DeterminismConfig`]).
    global_seed: u32,
    /// The stream namespace (see [`StreamId`]).
    stream_id: u32,
    /// The simulation frame index.
    frame: u32,
}

impl RngKey {
    /// Builds a key from raw inputs. `stream_id` is usually
    /// [`StreamId::as_u32`], but a raw namespace is accepted for callers that
    /// derive their own.
    #[must_use]
    pub const fn new(particle_id: u32, global_seed: u32, stream_id: u32, frame: u32) -> Self {
        Self {
            particle_id,
            global_seed,
            stream_id,
            frame,
        }
    }

    /// Builds a key for a named [`StreamId`].
    #[must_use]
    pub const fn for_stream(
        particle_id: u32,
        global_seed: u32,
        stream: StreamId,
        frame: u32,
    ) -> Self {
        Self::new(particle_id, global_seed, stream.as_u32(), frame)
    }

    /// Builds a key scoped to a specific system/emitter pair.
    ///
    /// The [`ParticleSystemHandle`] and [`EmitterHandle`] are folded into the
    /// seed so two emitters that share a `particle_id` and stream still draw
    /// decorrelated values — a single global seed then reproduces an entire
    /// multi-emitter effect.
    #[must_use]
    pub fn for_emitter(
        system: ParticleSystemHandle,
        emitter: EmitterHandle,
        particle_id: u32,
        global_seed: u32,
        stream: StreamId,
        frame: u32,
    ) -> Self {
        let scope = (u64::from(system.0) << 32) | u64::from(emitter.0);
        let mixed_seed = (mix64(u64::from(global_seed) ^ scope) >> 32) as u32;
        Self::for_stream(particle_id, mixed_seed, stream, frame)
    }

    /// The particle identity this key draws for.
    #[must_use]
    pub const fn particle_id(self) -> u32 {
        self.particle_id
    }

    /// The global deterministic seed this key draws under.
    #[must_use]
    pub const fn global_seed(self) -> u32 {
        self.global_seed
    }

    /// The stream namespace this key draws from.
    #[must_use]
    pub const fn stream_id(self) -> u32 {
        self.stream_id
    }

    /// The frame index this key draws at.
    #[must_use]
    pub const fn frame(self) -> u32 {
        self.frame
    }

    /// The raw 64-bit hash word for draw `index`.
    #[must_use]
    fn word(self, index: u32) -> u64 {
        let mut h = GOLDEN;
        h = mix64(h ^ u64::from(self.global_seed).wrapping_mul(MUL_SEED));
        h = mix64(h ^ u64::from(self.particle_id).wrapping_mul(MUL_ID));
        h = mix64(h ^ u64::from(self.stream_id).wrapping_mul(MUL_STREAM));
        h = mix64(h ^ u64::from(self.frame).wrapping_mul(MUL_FRAME));
        mix64(h ^ u64::from(index).wrapping_mul(MUL_INDEX))
    }

    /// A full 64-bit random word for draw `index`.
    #[must_use]
    pub fn next_u64(self, index: u32) -> u64 {
        self.word(index)
    }

    /// A 32-bit random word for draw `index` (the high half of [`RngKey::next_u64`]).
    #[must_use]
    pub fn next_u32(self, index: u32) -> u32 {
        (self.word(index) >> 32) as u32
    }

    /// A uniform `f32` in `[0, 1)` for draw `index`.
    #[must_use]
    pub fn unit_f32(self, index: u32) -> f32 {
        unit_f32_from_bits(self.next_u32(index))
    }

    /// A uniform `f32` in `[min, max)` for draw `index`.
    ///
    /// When `max <= min` the (degenerate) interval collapses and `min` is
    /// returned, keeping the function total.
    #[must_use]
    pub fn range_f32(self, index: u32, min: f32, max: f32) -> f32 {
        let span = max - min;
        if span > 0.0 {
            min + self.unit_f32(index) * span
        } else {
            min
        }
    }

    /// A uniformly distributed direction on the unit sphere.
    ///
    /// Uses bounded rejection sampling inside the unit cube: each attempt draws
    /// three uniforms in `[-1, 1)` and keeps the first point inside the open
    /// unit ball, then normalizes it with `sqrt` (the only allowed
    /// non-arithmetic operation). No `sin`/`cos` is used. Eight attempts make
    /// an all-miss outcome astronomically unlikely; the deterministic
    /// `+Z` fallback keeps the function total.
    #[must_use]
    pub fn unit_sphere(self, first_index: u32) -> Vec3 {
        const ATTEMPTS: u32 = 8;
        for attempt in 0..ATTEMPTS {
            let base = first_index.wrapping_add(attempt.wrapping_mul(3));
            let x = self.range_f32(base, -1.0, 1.0);
            let y = self.range_f32(base.wrapping_add(1), -1.0, 1.0);
            let z = self.range_f32(base.wrapping_add(2), -1.0, 1.0);
            let candidate = Vec3::new(x, y, z);
            let len_sq = candidate.length_squared();
            if len_sq > EPS_LEN_SQ && len_sq <= 1.0 {
                return candidate.normalize_or_zero();
            }
        }
        Vec3::new(0.0, 0.0, 1.0)
    }

    /// A uniformly distributed direction on the hemisphere around `normal`.
    ///
    /// Samples the full sphere and reflects any point in the wrong half onto
    /// `normal`'s side. When `normal` is (numerically) zero the full-sphere
    /// sample is returned unchanged.
    #[must_use]
    pub fn unit_hemisphere(self, first_index: u32, normal: Vec3) -> Vec3 {
        let axis = normal.normalize_or_zero();
        let dir = self.unit_sphere(first_index);
        if axis.length_squared() <= EPS_LEN_SQ {
            return dir;
        }
        if dir.dot(axis) < 0.0 {
            dir.scale(-1.0)
        } else {
            dir
        }
    }

    /// An approximately Gaussian `f32` with the given `mean` and `std_dev`.
    ///
    /// Uses the Irwin–Hall central-limit approximation: the sum of twelve
    /// independent uniforms on `[0, 1)` has mean `6` and variance `1`, so
    /// `sum - 6` approximates a standard normal. This needs no `exp`/`log`, so
    /// it stays transcendental-free. The result is bounded to
    /// `[mean - 6·std_dev, mean + 6·std_dev]`.
    #[must_use]
    pub fn normal_approx(self, first_index: u32, mean: f32, std_dev: f32) -> f32 {
        const TERMS: u32 = 12;
        let mut sum = 0.0f32;
        for k in 0..TERMS {
            sum += self.unit_f32(first_index.wrapping_add(k));
        }
        let z = sum - (TERMS as f32) * 0.5;
        mean + std_dev * z
    }
}

/// A convenience cursor over an [`RngKey`] for callers that want sequential
/// draws without threading an index by hand.
///
/// The cursor is *not* hidden state that breaks reproducibility: it merely
/// walks the deterministic `index` axis of a key. Resetting the cursor (or
/// rebuilding the sampler) replays the exact same sequence, which is what makes
/// record/replay possible.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Sampler {
    /// The immutable key all draws hash against.
    key: RngKey,
    /// The next draw index to consume.
    cursor: u32,
}

impl Sampler {
    /// Builds a sampler starting at draw index `0`.
    #[must_use]
    pub const fn new(key: RngKey) -> Self {
        Self { key, cursor: 0 }
    }

    /// The key this sampler draws from.
    #[must_use]
    pub const fn key(self) -> RngKey {
        self.key
    }

    /// The next draw index this sampler will consume.
    #[must_use]
    pub const fn cursor(self) -> u32 {
        self.cursor
    }

    /// Rewinds the cursor to the beginning, replaying the same sequence.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Advances the cursor by `count` draws without consuming values.
    pub fn skip(&mut self, count: u32) {
        self.cursor = self.cursor.wrapping_add(count);
    }

    /// Consumes one 32-bit draw.
    pub fn next_u32(&mut self) -> u32 {
        let value = self.key.next_u32(self.cursor);
        self.cursor = self.cursor.wrapping_add(1);
        value
    }

    /// Consumes one 64-bit draw.
    pub fn next_u64(&mut self) -> u64 {
        let value = self.key.next_u64(self.cursor);
        self.cursor = self.cursor.wrapping_add(1);
        value
    }

    /// Consumes one uniform `[0, 1)` draw.
    pub fn unit_f32(&mut self) -> f32 {
        let value = self.key.unit_f32(self.cursor);
        self.cursor = self.cursor.wrapping_add(1);
        value
    }

    /// Consumes one uniform `[min, max)` draw.
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        let value = self.key.range_f32(self.cursor, min, max);
        self.cursor = self.cursor.wrapping_add(1);
        value
    }

    /// Consumes a unit-sphere direction (three underlying draws per attempt).
    pub fn unit_sphere(&mut self) -> Vec3 {
        // Reserve a fixed span so subsequent draws never alias the rejection
        // attempts, keeping the cursor sequence stable across platforms.
        let value = self.key.unit_sphere(self.cursor);
        self.cursor = self.cursor.wrapping_add(24);
        value
    }

    /// Consumes a Gaussian draw (twelve underlying uniforms).
    pub fn normal_approx(&mut self, mean: f32, std_dev: f32) -> f32 {
        let value = self.key.normal_approx(self.cursor, mean, std_dev);
        self.cursor = self.cursor.wrapping_add(12);
        value
    }
}

/// The deterministic-execution configuration for a particle world (design §29).
///
/// Determinism requires three things to be pinned: a fixed simulation `dt` so
/// the integrator never sees a variable step, a fixed `global_seed` so the
/// hash-`RNG` re-derives identical draws, and a stable stage-execution order so
/// reductions and constraint batches accumulate in the same sequence on every
/// peer. When all three hold, the same configuration replayed against the same
/// frame sequence is bit-for-bit identical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeterminismConfig {
    /// The fixed simulation timestep in seconds.
    pub fixed_dt: f32,
    /// The effect-wide `RNG` seed.
    pub global_seed: u32,
    /// Whether stages are guaranteed to run in a stable, replayable order.
    pub stable_stage_order: bool,
}

impl DeterminismConfig {
    /// Builds a configuration.
    #[must_use]
    pub const fn new(fixed_dt: f32, global_seed: u32, stable_stage_order: bool) -> Self {
        Self {
            fixed_dt,
            global_seed,
            stable_stage_order,
        }
    }

    /// The fixed timestep.
    #[must_use]
    pub const fn dt(self) -> f32 {
        self.fixed_dt
    }

    /// Whether this configuration can guarantee bit-reproducible replay.
    ///
    /// Requires a positive fixed `dt` and a stable stage order; a non-positive
    /// `dt` means the caller has not actually pinned the step.
    #[must_use]
    pub fn is_deterministic(self) -> bool {
        self.stable_stage_order && self.fixed_dt > 0.0
    }
}

/// A rolling digest of a stream of random draws, used to compare two replays.
///
/// Recording every sampled value would be expensive; instead a replay folds its
/// draws into this 64-bit digest and two runs compare digests. Equal digests
/// mean the sampled sequences matched (up to the vanishing collision
/// probability of a strong mixer). This is the mechanism behind
/// [`replay_sample_digest`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayDigest(u64);

impl ReplayDigest {
    /// A fresh, empty digest.
    #[must_use]
    pub const fn new() -> Self {
        Self(GOLDEN)
    }

    /// Folds a 32-bit value into the digest.
    pub fn absorb_u32(&mut self, value: u32) {
        self.0 = mix64(self.0 ^ u64::from(value).wrapping_mul(MUL_INDEX));
    }

    /// Folds a 64-bit value into the digest.
    pub fn absorb_u64(&mut self, value: u64) {
        self.0 = mix64(self.0 ^ value.wrapping_mul(MUL_SEED));
    }

    /// Folds an `f32` (by its exact bit pattern) into the digest.
    pub fn absorb_f32(&mut self, value: f32) {
        self.absorb_u32(value.to_bits());
    }

    /// The finalized 64-bit digest value.
    #[must_use]
    pub fn finish(self) -> u64 {
        mix64(self.0)
    }
}

impl Default for ReplayDigest {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes a replay digest over a frame window for a set of particles and
/// streams (design §29).
///
/// For every frame in `[start_frame, end_frame)`, every particle, and every
/// stream, this folds one 32-bit draw and one uniform `[0, 1)` draw into a
/// [`ReplayDigest`]. Because every draw is a pure hash of stable inputs, two
/// runs with the same [`DeterminismConfig`] and the same window produce an
/// identical digest — the core record/replay verification tool. When the
/// configuration is not deterministic the digest is still well-defined but
/// carries no replay guarantee.
#[must_use]
pub fn replay_sample_digest(
    config: &DeterminismConfig,
    particle_ids: &[u32],
    streams: &[StreamId],
    start_frame: u32,
    end_frame: u32,
) -> ReplayDigest {
    let mut digest = ReplayDigest::new();
    let mut frame = start_frame;
    while frame < end_frame {
        for &particle_id in particle_ids {
            for &stream in streams {
                let key = RngKey::for_stream(particle_id, config.global_seed, stream, frame);
                digest.absorb_u32(key.next_u32(0));
                digest.absorb_f32(key.unit_f32(1));
            }
        }
        frame = frame.wrapping_add(1);
    }
    digest
}

/// Returns `true` when two replay digests match (i.e. the runs are bit-equal).
#[must_use]
pub fn replays_match(a: ReplayDigest, b: ReplayDigest) -> bool {
    a.finish() == b.finish()
}

/// How a particle effect participates in network synchronization (design §29).
///
/// The overwhelming majority of particles are pure eye-candy: smoke, sparks,
/// and debris that peers may render slightly differently with no gameplay
/// consequence. Synchronizing them would waste bandwidth, so they use
/// [`NetworkSyncPolicy::PresentationOnly`]. The rare effect that *drives*
/// gameplay — a projectile whose particles are the hitbox, or a hazard cloud
/// that deals damage — must agree across peers, so it takes
/// [`NetworkSyncPolicy::DeterministicCpu`] and runs the deterministic `CPU`
/// reference path rather than the free-running `GPU` presentation path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NetworkSyncPolicy {
    /// Presentation-only: never synchronized, free-running on the `GPU`.
    PresentationOnly,
    /// Gameplay-relevant: simulated on the deterministic `CPU` path so every
    /// peer derives the same state from the same inputs.
    DeterministicCpu,
}

impl NetworkSyncPolicy {
    /// Whether particles under this policy take part in network sync.
    #[must_use]
    pub fn participates_in_sync(self) -> bool {
        matches!(self, NetworkSyncPolicy::DeterministicCpu)
    }

    /// Whether particles under this policy run on the deterministic `CPU` path.
    ///
    /// Presentation-only effects stay on the free-running `GPU` path.
    #[must_use]
    pub fn runs_on_cpu(self) -> bool {
        matches!(self, NetworkSyncPolicy::DeterministicCpu)
    }
}

/// Selects a [`NetworkSyncPolicy`] from whether an effect affects gameplay
/// (design §29).
///
/// This is the single decision point: an effect that affects gameplay must be
/// deterministic and synchronized ([`NetworkSyncPolicy::DeterministicCpu`]);
/// everything else stays presentation-only to save bandwidth.
#[must_use]
pub fn select_sync_policy(affects_gameplay: bool) -> NetworkSyncPolicy {
    if affects_gameplay {
        NetworkSyncPolicy::DeterministicCpu
    } else {
        NetworkSyncPolicy::PresentationOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for `f32` comparisons that are otherwise exact.
    const EPS: f32 = 1e-6;
    /// Looser tolerance for statistical (Monte-Carlo) comparisons.
    const STAT_EPS: f32 = 5e-2;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn same_inputs_produce_same_values() {
        let key = RngKey::for_stream(7, 0xdead_beef, StreamId::Color, 42);
        assert_eq!(key.next_u32(0), key.next_u32(0));
        assert_eq!(key.next_u64(3), key.next_u64(3));
        assert!(approx(key.unit_f32(1), key.unit_f32(1), 0.0));
        assert!(approx(
            key.range_f32(2, -3.0, 9.0),
            key.range_f32(2, -3.0, 9.0),
            0.0,
        ));
        let s0 = key.unit_sphere(5);
        let s1 = key.unit_sphere(5);
        assert!(approx(s0.x, s1.x, 0.0));
        assert!(approx(s0.y, s1.y, 0.0));
        assert!(approx(s0.z, s1.z, 0.0));
        assert!(approx(
            key.normal_approx(10, 2.0, 0.5),
            key.normal_approx(10, 2.0, 0.5),
            0.0,
        ));
    }

    #[test]
    fn distinct_keys_produce_distinct_values() {
        let a = RngKey::for_stream(1, 100, StreamId::InitialPosition, 0);
        let b = RngKey::for_stream(2, 100, StreamId::InitialPosition, 0);
        let c = RngKey::for_stream(1, 101, StreamId::InitialPosition, 0);
        let d = RngKey::for_stream(1, 100, StreamId::InitialPosition, 1);
        assert_ne!(a.next_u32(0), b.next_u32(0));
        assert_ne!(a.next_u32(0), c.next_u32(0));
        assert_ne!(a.next_u32(0), d.next_u32(0));
    }

    #[test]
    fn unit_f32_stays_in_unit_interval_and_buckets_evenly() {
        const BUCKETS: usize = 8;
        const SAMPLES: u32 = 64;
        let mut counts = [0u32; BUCKETS];
        let mut total = 0u32;
        for id in 0..SAMPLES {
            for frame in 0..SAMPLES {
                let key = RngKey::for_stream(id, 0x1234_5678, StreamId::SizeJitter, frame);
                let u = key.unit_f32(0);
                assert!((0.0..1.0).contains(&u));
                let mut bucket = (u * BUCKETS as f32) as usize;
                if bucket >= BUCKETS {
                    bucket = BUCKETS - 1;
                }
                counts[bucket] += 1;
                total += 1;
            }
        }
        assert_eq!(total, SAMPLES * SAMPLES);
        // Expected 512 per bucket; a uniform mixer keeps every bucket well
        // within a wide integer band (no floating-point threshold needed).
        let expected = total / BUCKETS as u32;
        for &count in &counts {
            assert!(count > expected / 2, "under-filled bucket: {count}");
            assert!(count < expected * 3 / 2, "over-filled bucket: {count}");
        }
    }

    #[test]
    fn streams_are_decorrelated() {
        const N: u32 = 4096;
        let mut sum_a = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut equal_bits = 0u32;
        let mut both_high = 0u32;
        for i in 0..N {
            let ka = RngKey::for_stream(i, 55, StreamId::InitialPosition, 0);
            let kb = RngKey::for_stream(i, 55, StreamId::InitialVelocity, 0);
            let a = ka.unit_f32(0);
            let b = kb.unit_f32(0);
            sum_a += a;
            sum_b += b;
            if a.to_bits() == b.to_bits() {
                equal_bits += 1;
            }
            if a > 0.5 && b > 0.5 {
                both_high += 1;
            }
        }
        // Means both near 0.5.
        assert!(approx(sum_a / N as f32, 0.5, STAT_EPS));
        assert!(approx(sum_b / N as f32, 0.5, STAT_EPS));
        // The two streams almost never coincide bitwise.
        assert!(equal_bits < N / 100, "streams too correlated: {equal_bits}");
        // Joint "both above 0.5" count sits near N/4 for independent streams.
        let quarter = N / 4;
        assert!(both_high > quarter - quarter / 4);
        assert!(both_high < quarter + quarter / 4);
    }

    #[test]
    fn frames_and_seeds_are_decorrelated() {
        const N: u32 = 2048;
        let mut equal_frames = 0u32;
        let mut equal_seeds = 0u32;
        for i in 0..N {
            let base = RngKey::for_stream(i, 9, StreamId::Rotation, 0);
            let next_frame = RngKey::for_stream(i, 9, StreamId::Rotation, 1);
            let next_seed = RngKey::for_stream(i, 10, StreamId::Rotation, 0);
            if base.next_u32(0) == next_frame.next_u32(0) {
                equal_frames += 1;
            }
            if base.next_u32(0) == next_seed.next_u32(0) {
                equal_seeds += 1;
            }
        }
        assert!(equal_frames < N / 100);
        assert!(equal_seeds < N / 100);
    }

    #[test]
    fn unit_sphere_directions_are_normalized() {
        for i in 0..1024u32 {
            let key = RngKey::for_stream(i, 3, StreamId::InitialVelocity, i);
            let dir = key.unit_sphere(0);
            assert!(approx(dir.length(), 1.0, 1e-5), "not unit length");
        }
    }

    #[test]
    fn hemisphere_directions_face_the_normal() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        for i in 0..1024u32 {
            let key = RngKey::for_stream(i, 77, StreamId::InitialVelocity, i);
            let dir = key.unit_hemisphere(0, normal);
            assert!(approx(dir.length(), 1.0, 1e-5));
            assert!(
                dir.dot(normal) >= -EPS,
                "wrong hemisphere: {}",
                dir.dot(normal)
            );
        }
    }

    #[test]
    fn hemisphere_with_zero_normal_falls_back_to_full_sphere() {
        let key = RngKey::for_stream(1, 1, StreamId::InitialVelocity, 1);
        let full = key.unit_sphere(0);
        let hemi = key.unit_hemisphere(0, Vec3::ZERO);
        assert!(approx(full.x, hemi.x, 0.0));
        assert!(approx(full.y, hemi.y, 0.0));
        assert!(approx(full.z, hemi.z, 0.0));
    }

    #[test]
    fn range_stays_within_bounds() {
        let min = -2.5f32;
        let max = 7.5f32;
        for i in 0..4096u32 {
            let key = RngKey::for_stream(i, 4, StreamId::LifetimeJitter, 0);
            let v = key.range_f32(0, min, max);
            assert!((min..max).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn degenerate_range_returns_min() {
        let key = RngKey::for_stream(1, 1, StreamId::Color, 0);
        assert!(approx(key.range_f32(0, 5.0, 5.0), 5.0, 0.0));
        assert!(approx(key.range_f32(0, 9.0, 1.0), 9.0, 0.0));
    }

    #[test]
    fn normal_approx_matches_mean_and_bounds() {
        const N: u32 = 8192;
        let mean = 3.0f32;
        let std_dev = 2.0f32;
        let mut sum = 0.0f32;
        for i in 0..N {
            let key = RngKey::for_stream(i, 8, StreamId::SizeJitter, 0);
            let z = key.normal_approx(0, mean, std_dev);
            // Irwin-Hall bounds: mean +/- 6*std_dev.
            assert!(z >= mean - 6.0 * std_dev - EPS);
            assert!(z <= mean + 6.0 * std_dev + EPS);
            sum += z;
        }
        assert!(approx(sum / N as f32, mean, STAT_EPS));
    }

    #[test]
    fn sampler_replays_deterministically() {
        let key = RngKey::for_stream(11, 21, StreamId::SubUv, 3);
        let mut a = Sampler::new(key);
        let first = [a.next_u32(), a.next_u32(), a.next_u32()];
        a.reset();
        let second = [a.next_u32(), a.next_u32(), a.next_u32()];
        assert_eq!(first, second);

        let mut b = Sampler::new(key);
        b.skip(1);
        assert_eq!(b.next_u32(), key.next_u32(1));
    }

    #[test]
    fn replay_digest_is_stable_and_seed_sensitive() {
        let ids = [1u32, 2, 3, 4];
        let streams = [StreamId::InitialPosition, StreamId::Color];
        let cfg = DeterminismConfig::new(1.0 / 60.0, 0xabcd, true);
        let a = replay_sample_digest(&cfg, &ids, &streams, 0, 16);
        let b = replay_sample_digest(&cfg, &ids, &streams, 0, 16);
        assert_eq!(a, b);
        assert!(replays_match(a, b));

        let other = DeterminismConfig::new(1.0 / 60.0, 0x1234, true);
        let c = replay_sample_digest(&other, &ids, &streams, 0, 16);
        assert_ne!(a.finish(), c.finish());
        assert!(!replays_match(a, c));
    }

    #[test]
    fn determinism_config_reports_reproducibility() {
        assert!(DeterminismConfig::new(0.016, 1, true).is_deterministic());
        assert!(!DeterminismConfig::new(0.0, 1, true).is_deterministic());
        assert!(!DeterminismConfig::new(-0.016, 1, true).is_deterministic());
        assert!(!DeterminismConfig::new(0.016, 1, false).is_deterministic());
        assert!(approx(
            DeterminismConfig::new(0.02, 0, true).dt(),
            0.02,
            EPS
        ));
    }

    #[test]
    fn replay_digest_default_matches_new() {
        assert_eq!(ReplayDigest::default(), ReplayDigest::new());
    }

    #[test]
    fn stream_ids_are_distinct() {
        let ids = [
            StreamId::InitialPosition.as_u32(),
            StreamId::InitialVelocity.as_u32(),
            StreamId::Color.as_u32(),
            StreamId::LifetimeJitter.as_u32(),
            StreamId::SizeJitter.as_u32(),
            StreamId::Rotation.as_u32(),
            StreamId::SubUv.as_u32(),
            StreamId::Custom(0).as_u32(),
            StreamId::Custom(1).as_u32(),
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "stream namespace collision");
            }
        }
    }

    #[test]
    fn emitter_scoping_decorrelates_emitters() {
        let sys = ParticleSystemHandle(4);
        let e0 = EmitterHandle(0);
        let e1 = EmitterHandle(1);
        let k0 = RngKey::for_emitter(sys, e0, 9, 1234, StreamId::Color, 5);
        let k1 = RngKey::for_emitter(sys, e1, 9, 1234, StreamId::Color, 5);
        assert_ne!(k0.next_u32(0), k1.next_u32(0));
        // Same emitter scope reproduces the same key.
        let k0b = RngKey::for_emitter(sys, e0, 9, 1234, StreamId::Color, 5);
        assert_eq!(k0, k0b);
    }

    #[test]
    fn sync_policy_follows_gameplay_impact() {
        assert_eq!(
            select_sync_policy(true),
            NetworkSyncPolicy::DeterministicCpu
        );
        assert_eq!(
            select_sync_policy(false),
            NetworkSyncPolicy::PresentationOnly
        );
        assert!(NetworkSyncPolicy::DeterministicCpu.participates_in_sync());
        assert!(NetworkSyncPolicy::DeterministicCpu.runs_on_cpu());
        assert!(!NetworkSyncPolicy::PresentationOnly.participates_in_sync());
        assert!(!NetworkSyncPolicy::PresentationOnly.runs_on_cpu());
    }
}
