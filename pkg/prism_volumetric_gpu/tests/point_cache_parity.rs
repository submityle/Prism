//! Real-device parity for the per-point point-cache sample twin:
//! [`GpuPointCache`](prism_volumetric_gpu::point_cache::GpuPointCache) must
//! reproduce the `CPU` golden
//! [`PointCache`](prism_render_architecture::particle::point_cache::PointCache)
//! accessors
//! ([`position_at`](prism_render_architecture::particle::point_cache::PointCache::position_at),
//! [`velocity_at`](prism_render_architecture::particle::point_cache::PointCache::velocity_at)
//! and
//! [`scalar_at`](prism_render_architecture::particle::point_cache::PointCache::scalar_at))
//! across all three
//! [`PlaybackMode`](prism_render_architecture::particle::point_cache::PlaybackMode)
//! boundary policies (`Clamp`, `Loop`, `PingPong`).
//!
//! The fixtures cover a scalar-carrying multi-frame cache, a scalar-less cache,
//! a single-frame (`F == 1`) cache and a single-point (`N == 1`) cache, each
//! sampled at playback times whose `time * fps` lands clear of a frame boundary
//! (so the chosen frame pair is unambiguous) and that span negative times,
//! in-range interiors and past-the-end times under every mode. All channels are
//! exact `0.1`-step jitter shared bit-for-bit between the `CPU` reference and the
//! `GPU` inputs, and the whole batch is one shared cache plus one dispatch,
//! exercising the one-thread-per-query path. A separate test drives the
//! empty-batch host short-circuit.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer `has_scalar` flag is compared with an exact `==`. The
//! interpolated `position`, `velocity` and `scalar` thread through a subtract, a
//! multiply and an add, so they are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::point_cache::{PlaybackMode, PointCache};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::point_cache::{
    GpuPointCache, PointCacheQuery, PointCacheSample, POINT_CACHE_CLAMP, POINT_CACHE_LOOP,
    POINT_CACHE_PINGPONG,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for a continuous interpolated channel.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for a continuous interpolated channel.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// The three playback modes paired with their kernel `u32` code.
const MODES: [(PlaybackMode, u32); 3] = [
    (PlaybackMode::Clamp, POINT_CACHE_CLAMP),
    (PlaybackMode::Loop, POINT_CACHE_LOOP),
    (PlaybackMode::PingPong, POINT_CACHE_PINGPONG),
];

/// Frame-unit sample points, each with a fractional part clear of `0` and `1`
/// so the floored frame pair is unambiguous: two negative, three in-range and
/// two past-the-end positions (relative to a short cache).
const FRAME_UNITS: [f32; 7] = [-1.4, -0.6, 0.3, 1.4, 2.6, 3.7, 6.5];

/// Mixed absolute / relative tolerance comparison for one `f32` value.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Component-wise tolerance comparison for two [`Vec3`] values.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Deterministic host-side `u64` LCG (numerical-recipes constants) producing a
/// repeatable stream of simple-decimal jitter; no transcendental math is
/// involved.
struct Lcg {
    /// Current state word.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns a signed `0.1`-step jitter in
    /// `[-0.5, 0.5]`, built from integer arithmetic so the value is an exact
    /// small decimal shared bit-for-bit between the `CPU` and `GPU` inputs.
    fn jitter(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        // Integer in 0..=10, centred to -5..=5, then scaled to a 0.1-step grid.
        let step = (bits % 11) as i32 - 5;
        step as f32 / 10.0
    }
}

/// Builds a frame-major cache of `frame_count` frames and `point_count` points
/// at `fps`, with jittered positions and velocities and, when `with_scalar`,
/// a jittered scalar channel. The channels are exact `0.1`-step decimals.
fn build_cache(
    frame_count: u32,
    point_count: u32,
    fps: f32,
    with_scalar: bool,
    rng: &mut Lcg,
) -> PointCache {
    let count = (frame_count as usize) * (point_count as usize);
    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    let mut scalars = Vec::new();
    if with_scalar {
        scalars.reserve(count);
    }
    for frame in 0..frame_count {
        for point in 0..point_count {
            // A deterministic integer base keeps the frames well separated so a
            // mislocated frame pair is caught, with sub-unit jitter on top.
            let fbase = frame as f32;
            let pbase = point as f32;
            positions.push(Vec3::new(
                fbase + rng.jitter(),
                pbase + rng.jitter(),
                fbase - pbase + rng.jitter(),
            ));
            velocities.push(Vec3::new(
                rng.jitter(),
                fbase + rng.jitter(),
                pbase + rng.jitter(),
            ));
            if with_scalar {
                scalars.push(fbase * 2.0 + pbase + rng.jitter());
            }
        }
    }
    PointCache::from_frames(
        frame_count,
        point_count,
        fps,
        positions,
        velocities,
        scalars,
    )
    .expect("fixture cache invariants hold")
}

/// One golden case: the time, mode, point and the query index in the batch.
struct Case {
    /// Playback time in seconds.
    time: f32,
    /// Reference boundary mode.
    mode: PlaybackMode,
    /// Sampled point identity.
    point: u32,
    /// Index of this query in the flattened batch.
    index: usize,
}

/// Appends every `(point, time, mode)` query for one cache to the shared batch,
/// recording the parallel golden cases.
fn add_cases(cache: &PointCache, queries: &mut Vec<PointCacheQuery>, cases: &mut Vec<Case>) {
    let fps = cache.fps();
    for point in 0..cache.point_count() {
        for &unit in &FRAME_UNITS {
            let time = unit / fps;
            for &(mode, code) in &MODES {
                let index = queries.len();
                queries.push(PointCacheQuery::new(time, code, point));
                cases.push(Case {
                    time,
                    mode,
                    point,
                    index,
                });
            }
        }
    }
}

/// Asserts the `GPU` samples match the golden accessors for one cache's cases.
fn check_cache(cache: &PointCache, samples: &[PointCacheSample], cases: &[Case]) {
    for case in cases {
        let got = &samples[case.index];
        let want_pos = cache.position_at(case.point, case.time, case.mode);
        let want_vel = cache.velocity_at(case.point, case.time, case.mode);
        let want_scalar = cache.scalar_at(case.point, case.time, case.mode);
        assert!(
            approx_vec(got.position, want_pos),
            "position mismatch (mode {:?}, point {}, time {}): gpu {:?} vs cpu {:?}",
            case.mode,
            case.point,
            case.time,
            got.position,
            want_pos
        );
        assert!(
            approx_vec(got.velocity, want_vel),
            "velocity mismatch (mode {:?}, point {}, time {}): gpu {:?} vs cpu {:?}",
            case.mode,
            case.point,
            case.time,
            got.velocity,
            want_vel
        );
        assert_eq!(
            got.has_scalar,
            want_scalar.is_some(),
            "has_scalar mismatch (mode {:?}, point {}, time {})",
            case.mode,
            case.point,
            case.time
        );
        if let Some(want) = want_scalar {
            assert!(
                approx(got.scalar, want),
                "scalar mismatch (mode {:?}, point {}, time {}): gpu {} vs cpu {}",
                case.mode,
                case.point,
                case.time,
                got.scalar,
                want
            );
        }
    }
}

#[test]
fn point_cache_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointCache::new(&ctx);
    let mut rng = Lcg::new(0x5151_f0cd_1234_abcd);

    // Four caches exercising scalar / scalar-less channels and the F == 1 and
    // N == 1 degeneracies. Each is dispatched independently against its own
    // shared frame buffers.
    let caches = [
        build_cache(4, 3, 24.0, true, &mut rng),
        build_cache(3, 2, 30.0, false, &mut rng),
        build_cache(1, 1, 60.0, true, &mut rng),
        build_cache(5, 1, 24.0, false, &mut rng),
    ];

    for cache in &caches {
        let mut queries = Vec::new();
        let mut cases = Vec::new();
        add_cases(cache, &mut queries, &mut cases);
        let samples = gpu.sample(&queries, cache);
        assert_eq!(samples.len(), queries.len());
        check_cache(cache, &samples, &cases);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointCache::new(&ctx);
    // A valid, non-empty cache isolates the query short-circuit: no queries
    // means no dispatch and an empty result vector.
    let mut rng = Lcg::new(0x0bad_c0de_dead_beef);
    let cache = build_cache(3, 2, 24.0, true, &mut rng);
    let out = gpu.sample(&[], &cache);
    assert!(out.is_empty());
}
