//! Real-device parity for the particle-emission twin:
//! [`GpuEmitter`](prism_volumetric_gpu::emitter::GpuEmitter) must reproduce the
//! `CPU` golden
//! [`emitter`](prism_render_architecture::particle::emitter) across every
//! numeric term it exposes: the rejection-sampled unit disk and unit ball (the
//! internals of
//! [`sample_shape`](prism_render_architecture::particle::emitter::sample_shape)),
//! the full shape sampler
//! ([`sample_shape`](prism_render_architecture::particle::emitter::sample_shape)),
//! the fractional-carry rate accumulator
//! ([`SpawnAccumulator::accumulate`](prism_render_architecture::particle::emitter::SpawnAccumulator::accumulate)),
//! the half-open burst-window reduction
//! ([`bursts_in_window`](prism_render_architecture::particle::emitter::bursts_in_window)),
//! the inherited-velocity scale
//! ([`inherited_velocity`](prism_render_architecture::particle::emitter::inherited_velocity)),
//! the initial-state composition
//! ([`build_spawn`](prism_render_architecture::particle::emitter::build_spawn)),
//! and the combined per-frame spawn count
//! ([`Emitter::spawn_count`](prism_render_architecture::particle::emitter::Emitter::spawn_count)).
//!
//! The fixtures exercise one dedicated query per term plus a randomized mixed
//! batch compared element for element, and one stateful-chain test that drives a
//! real [`SpawnAccumulator`](prism_render_architecture::particle::emitter::SpawnAccumulator)
//! across many frames and feeds each frame's carry back into the twin, so the
//! pure carry-in / carry-out twin is pinned to the observable behaviour of the
//! stateful reference. Every scalar is an interior value held well away from its
//! guard branch: spawn rates and steps are comfortably positive with the product
//! `rate * dt` kept clear of an integer so the floored whole count is robust,
//! radii and heights are positive, burst timestamps sit strictly inside or
//! outside the window, and the sampler inputs land the first rejection candidate
//! comfortably inside the disk or ball so the `CPU` and `GPU` cursors consume the
//! same samples and walk the same branch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous term is a polynomial or rational function of its inputs with
//! at most one `sqrt` (the robust normalize inside the shape sampler), so `CPU`
//! and `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous `f32` lane; the integer spawn counts
//! are compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the guard cracks: unit samples map through the
//! `[-1, 1]` cursor to components comfortably inside the unit disk and ball so
//! the first rejection candidate is accepted (identical sample consumption on
//! both paths), spawn rates and steps are positive with `rate * dt` away from an
//! integer so the floored count is unambiguous, burst timestamps are separated
//! from the window edges, radii and heights are positive, and the sampled
//! direction is non-degenerate (or decisively zero for the `+Z` fallback case)
//! so no fixture lands on a classification tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::emitter::{
    build_spawn, bursts_in_window, inherited_velocity, sample_shape, BurstSpawn, Emitter,
    EmitterShape, SpawnAccumulator, SpawnParams, SpawnSample, UnitCursor,
};
use prism_render_architecture::particle::{EmitterHandle, SimSpace, Vec3};
use prism_volumetric_gpu::emitter::{
    EmitterQuery, EmitterResult, GpuEmitter, MAX_EMITTER_BURSTS, MAX_EMITTER_SAMPLES,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Rejection-attempt bound mirroring the reference `MAX_REJECTION_TRIES`.
const MAX_TRIES: u32 = 8;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Converts a packed component triple into a [`Vec3`].
fn v3(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0], c[1], c[2])
}

/// Copies `vals` into a budgeted, zero-padded unit-sample array.
fn samples_from(vals: &[f32]) -> [f32; MAX_EMITTER_SAMPLES] {
    let mut out = [0.0_f32; MAX_EMITTER_SAMPLES];
    out[..vals.len()].copy_from_slice(vals);
    out
}

/// Copies `vals` into a budgeted, zero-padded burst array.
fn bursts_from(vals: &[BurstSpawn]) -> [BurstSpawn; MAX_EMITTER_BURSTS] {
    let mut out = [BurstSpawn {
        time: 0.0,
        count: 0,
    }; MAX_EMITTER_BURSTS];
    out[..vals.len()].copy_from_slice(vals);
    out
}

/// Reproduces the reference unit-disk rejection sampler using the public
/// [`UnitCursor`], whose private `sample_unit_disk` is not callable directly.
/// Only comparisons and the wrapping cursor are used, matching the reference
/// term for term.
fn golden_disk(samples: &[f32; MAX_EMITTER_SAMPLES], sample_len: u32) -> (f32, f32) {
    let mut cursor = UnitCursor::new(&samples[..sample_len as usize]);
    let mut i = 0;
    while i < MAX_TRIES {
        let x = cursor.next_signed();
        let y = cursor.next_signed();
        if x * x + y * y <= 1.0 {
            return (x, y);
        }
        i += 1;
    }
    (0.0, 0.0)
}

/// Reproduces the reference unit-ball rejection sampler using the public
/// [`UnitCursor`], whose private `sample_unit_ball` is not callable directly.
fn golden_ball(samples: &[f32; MAX_EMITTER_SAMPLES], sample_len: u32) -> [f32; 3] {
    let mut cursor = UnitCursor::new(&samples[..sample_len as usize]);
    let mut i = 0;
    while i < MAX_TRIES {
        let x = cursor.next_signed();
        let y = cursor.next_signed();
        let z = cursor.next_signed();
        if x * x + y * y + z * z <= 1.0 {
            return [x, y, z];
        }
        i += 1;
    }
    [0.0, 0.0, 0.0]
}

/// Reproduces the reference fractional-carry transition. The reference
/// [`SpawnAccumulator`]'s `carry` field is private, so the documented pure
/// transition `carry_out = (carry_in + rate * dt) - whole` is reproduced here;
/// the `accumulate_matches_stateful_reference` test ties this reproduction to
/// the real stateful accumulator across many frames.
fn golden_accumulate(carry_in: f32, rate: f32, dt: f32) -> (u32, f32) {
    if rate > 0.0 && dt > 0.0 {
        let carry = carry_in + rate * dt;
        let whole = if carry >= u32::MAX as f32 {
            u32::MAX
        } else {
            carry as u32
        };
        (whole, carry - whole as f32)
    } else {
        (0, carry_in)
    }
}

/// Recomputes the expected [`EmitterResult`] by calling the `CPU` golden for
/// `query` (or, for the two private rejection samplers and the private-state
/// accumulator, the faithful public-API reproductions above).
fn golden_result(query: &EmitterQuery) -> EmitterResult {
    match *query {
        EmitterQuery::SampleUnitDisk {
            samples,
            sample_len,
        } => {
            let (x, y) = golden_disk(&samples, sample_len);
            EmitterResult::SampleUnitDisk { point: [x, y] }
        }
        EmitterQuery::SampleUnitBall {
            samples,
            sample_len,
        } => EmitterResult::SampleUnitBall {
            point: golden_ball(&samples, sample_len),
        },
        EmitterQuery::SampleShape {
            shape,
            samples,
            sample_len,
        } => {
            let mut cursor = UnitCursor::new(&samples[..sample_len as usize]);
            let s = sample_shape(shape, &mut cursor);
            EmitterResult::SampleShape {
                position: [s.position.x, s.position.y, s.position.z],
                direction: [s.direction.x, s.direction.y, s.direction.z],
            }
        }
        EmitterQuery::Accumulate {
            carry_in,
            rate_per_second,
            dt,
        } => {
            let (count, carry) = golden_accumulate(carry_in, rate_per_second, dt);
            EmitterResult::Accumulate { count, carry }
        }
        EmitterQuery::BurstsInWindow {
            bursts,
            burst_count,
            start,
            end,
        } => EmitterResult::BurstsInWindow {
            count: bursts_in_window(&bursts[..burst_count as usize], start, end),
        },
        EmitterQuery::InheritedVelocity {
            emitter_velocity,
            factor,
        } => {
            let v = inherited_velocity(emitter_velocity, factor);
            EmitterResult::InheritedVelocity {
                velocity: [v.x, v.y, v.z],
            }
        }
        EmitterQuery::BuildSpawn {
            origin,
            emitter_velocity,
            sample,
            params,
        } => {
            let state = build_spawn(origin, emitter_velocity, sample, params);
            EmitterResult::BuildSpawn {
                position: [state.position.x, state.position.y, state.position.z],
                velocity: [state.velocity.x, state.velocity.y, state.velocity.z],
            }
        }
        EmitterQuery::SpawnCount {
            carry_in: _,
            rate_per_second,
            dt,
            bursts,
            burst_count,
            time,
        } => {
            // The reference `spawn_count` starts from a fresh (zeroed) carry, so
            // the fixtures pass `carry_in == 0`; a fresh emitter reproduces it.
            let mut emitter = Emitter::new(
                EmitterHandle(1),
                EmitterShape::Point,
                SpawnParams::default(),
            );
            let count =
                emitter.spawn_count(rate_per_second, dt, &bursts[..burst_count as usize], time);
            EmitterResult::SpawnCount {
                count,
                carry: emitter.rate_carry(),
            }
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: each lane the
/// variant carries must agree within the parity bound (continuous lanes within
/// tolerance, integer counts exactly).
fn pin(idx: usize, query: &EmitterQuery, got: &EmitterResult) {
    let want = golden_result(query);
    match (got, want) {
        (
            EmitterResult::SampleUnitDisk { point: g },
            EmitterResult::SampleUnitDisk { point: w },
        ) => {
            for k in 0..2 {
                assert!(
                    close(g[k], w[k]),
                    "query {idx} disk[{k}]: gpu {} vs cpu {}",
                    g[k],
                    w[k]
                );
            }
        }
        (
            EmitterResult::SampleUnitBall { point: g },
            EmitterResult::SampleUnitBall { point: w },
        ) => {
            for k in 0..3 {
                assert!(
                    close(g[k], w[k]),
                    "query {idx} ball[{k}]: gpu {} vs cpu {}",
                    g[k],
                    w[k]
                );
            }
        }
        (
            EmitterResult::SampleShape {
                position: gp,
                direction: gd,
            },
            EmitterResult::SampleShape {
                position: wp,
                direction: wd,
            },
        ) => {
            for k in 0..3 {
                assert!(
                    close(gp[k], wp[k]),
                    "query {idx} shape position[{k}]: gpu {} vs cpu {}",
                    gp[k],
                    wp[k]
                );
                assert!(
                    close(gd[k], wd[k]),
                    "query {idx} shape direction[{k}]: gpu {} vs cpu {}",
                    gd[k],
                    wd[k]
                );
            }
        }
        (
            EmitterResult::InheritedVelocity { velocity: g },
            EmitterResult::InheritedVelocity { velocity: w },
        ) => {
            for k in 0..3 {
                assert!(
                    close(g[k], w[k]),
                    "query {idx} inherited velocity[{k}]: gpu {} vs cpu {}",
                    g[k],
                    w[k]
                );
            }
        }
        (
            EmitterResult::BuildSpawn {
                position: gp,
                velocity: gv,
            },
            EmitterResult::BuildSpawn {
                position: wp,
                velocity: wv,
            },
        ) => {
            for k in 0..3 {
                assert!(
                    close(gp[k], wp[k]),
                    "query {idx} spawn position[{k}]: gpu {} vs cpu {}",
                    gp[k],
                    wp[k]
                );
                assert!(
                    close(gv[k], wv[k]),
                    "query {idx} spawn velocity[{k}]: gpu {} vs cpu {}",
                    gv[k],
                    wv[k]
                );
            }
        }
        (
            EmitterResult::Accumulate {
                count: gc,
                carry: gy,
            },
            EmitterResult::Accumulate {
                count: wc,
                carry: wy,
            },
        )
        | (
            EmitterResult::SpawnCount {
                count: gc,
                carry: gy,
            },
            EmitterResult::SpawnCount {
                count: wc,
                carry: wy,
            },
        ) => {
            assert_eq!(*gc, wc, "query {idx} count: gpu {gc} vs cpu {wc}");
            assert!(close(*gy, wy), "query {idx} carry: gpu {gy} vs cpu {wy}");
        }
        (
            EmitterResult::BurstsInWindow { count: g },
            EmitterResult::BurstsInWindow { count: w },
        ) => {
            assert_eq!(*g, w, "query {idx} burst count: gpu {g} vs cpu {w}");
        }
        (g, w) => panic!("query {idx} result variant mismatch: gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuEmitter, queries: &[EmitterQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn sample_unit_disk_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    // Samples map to components in [-0.4, 0.4]; the first candidate lies well
    // inside the disk so both cursors accept it on the first try.
    let queries = vec![
        EmitterQuery::SampleUnitDisk {
            samples: samples_from(&[0.6, 0.4, 0.55, 0.45]),
            sample_len: 4,
        },
        EmitterQuery::SampleUnitDisk {
            samples: samples_from(&[0.3, 0.7]),
            sample_len: 2,
        },
        // Empty slice: the cursor yields 0.5 => signed 0 => the disk center.
        EmitterQuery::SampleUnitDisk {
            samples: samples_from(&[]),
            sample_len: 0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn sample_unit_ball_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let queries = vec![
        EmitterQuery::SampleUnitBall {
            samples: samples_from(&[0.6, 0.4, 0.55]),
            sample_len: 3,
        },
        EmitterQuery::SampleUnitBall {
            samples: samples_from(&[0.35, 0.65, 0.45, 0.5]),
            sample_len: 4,
        },
        EmitterQuery::SampleUnitBall {
            samples: samples_from(&[]),
            sample_len: 0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn sample_shape_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let samples = samples_from(&[0.6, 0.4, 0.55, 0.45, 0.5, 0.6]);
    let queries = vec![
        EmitterQuery::SampleShape {
            shape: EmitterShape::Point,
            samples,
            sample_len: 6,
        },
        EmitterQuery::SampleShape {
            shape: EmitterShape::Sphere {
                radius: 3.0,
                surface_only: false,
            },
            samples,
            sample_len: 6,
        },
        EmitterQuery::SampleShape {
            shape: EmitterShape::Sphere {
                radius: 3.0,
                surface_only: true,
            },
            samples,
            sample_len: 6,
        },
        EmitterQuery::SampleShape {
            shape: EmitterShape::Box {
                half_extents: v3([2.0, 4.0, 6.0]),
            },
            samples,
            sample_len: 6,
        },
        EmitterQuery::SampleShape {
            shape: EmitterShape::Cone {
                base_radius: 1.0,
                height: 2.0,
            },
            samples,
            sample_len: 6,
        },
        // Empty slice drives the sphere to its +Z degenerate fallback.
        EmitterQuery::SampleShape {
            shape: EmitterShape::Sphere {
                radius: 3.0,
                surface_only: false,
            },
            samples: samples_from(&[]),
            sample_len: 0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn accumulate_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    // `rate * dt` kept clear of an integer so the floored whole count is robust.
    let queries = vec![
        EmitterQuery::Accumulate {
            carry_in: 0.0,
            rate_per_second: 33.0,
            dt: 0.1,
        },
        EmitterQuery::Accumulate {
            carry_in: 0.0,
            rate_per_second: 10.0,
            dt: 0.25,
        },
        EmitterQuery::Accumulate {
            carry_in: 0.3,
            rate_per_second: 10.0,
            dt: 0.25,
        },
        // Paused emitter: zero rate spawns nothing and leaves the carry intact.
        EmitterQuery::Accumulate {
            carry_in: 0.42,
            rate_per_second: 0.0,
            dt: 0.1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn accumulate_matches_stateful_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    // Drive the real stateful accumulator across many frames and feed each
    // frame's carry back into the pure twin. The per-frame increment 2.73 keeps
    // every partial sum clear of an integer so the floored counts never straddle
    // a boundary.
    let rate = 27.3_f32;
    let dt = 0.1_f32;
    let mut gold = SpawnAccumulator::new();
    let mut carry_in = 0.0_f32;
    for frame in 0..10 {
        let gold_count = gold.accumulate(rate, dt);
        let gold_carry = gold.carry();
        let got = gpu.evaluate(
            &ctx,
            &[EmitterQuery::Accumulate {
                carry_in,
                rate_per_second: rate,
                dt,
            }],
        );
        match got[0] {
            EmitterResult::Accumulate { count, carry } => {
                assert_eq!(
                    count, gold_count,
                    "frame {frame} count: gpu {count} vs cpu {gold_count}"
                );
                assert!(
                    close(carry, gold_carry),
                    "frame {frame} carry: gpu {carry} vs cpu {gold_carry}"
                );
                carry_in = carry;
            }
            other => panic!("frame {frame} unexpected result {other:?}"),
        }
    }
}

#[test]
fn bursts_in_window_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let bursts = bursts_from(&[
        BurstSpawn {
            time: 0.3,
            count: 4,
        },
        BurstSpawn {
            time: 0.7,
            count: 9,
        },
        BurstSpawn {
            time: 1.5,
            count: 100,
        },
    ]);
    let queries = vec![
        // Window (0.0, 1.0] captures the first two bursts, not the third.
        EmitterQuery::BurstsInWindow {
            bursts,
            burst_count: 3,
            start: 0.0,
            end: 1.0,
        },
        // Backward window fires nothing.
        EmitterQuery::BurstsInWindow {
            bursts,
            burst_count: 3,
            start: 1.0,
            end: 0.5,
        },
        // Window (0.5, 2.0] captures the second and third bursts.
        EmitterQuery::BurstsInWindow {
            bursts,
            burst_count: 3,
            start: 0.5,
            end: 2.0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn inherited_velocity_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let queries = vec![
        EmitterQuery::InheritedVelocity {
            emitter_velocity: v3([0.0, 10.0, 0.0]),
            factor: 0.5,
        },
        EmitterQuery::InheritedVelocity {
            emitter_velocity: v3([3.0, -2.0, 4.0]),
            factor: 0.25,
        },
        EmitterQuery::InheritedVelocity {
            emitter_velocity: v3([1.5, 1.5, 1.5]),
            factor: 0.0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn build_spawn_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let queries = vec![
        // World space: the local offset is placed relative to the origin.
        EmitterQuery::BuildSpawn {
            origin: v3([1.0, 0.0, 0.0]),
            emitter_velocity: v3([0.0, 10.0, 0.0]),
            sample: SpawnSample {
                position: v3([0.0, 0.0, 0.0]),
                direction: v3([0.0, 0.0, 1.0]),
            },
            params: SpawnParams {
                speed: 5.0,
                inherit_velocity: 0.5,
                sim_space: SimSpace::World,
            },
        },
        // Local space keeps the spawn offset relative.
        EmitterQuery::BuildSpawn {
            origin: v3([9.0, 9.0, 9.0]),
            emitter_velocity: v3([0.0, 0.0, 0.0]),
            sample: SpawnSample {
                position: v3([0.5, 0.0, 0.0]),
                direction: v3([0.0, 0.0, 1.0]),
            },
            params: SpawnParams {
                speed: 0.0,
                inherit_velocity: 0.0,
                sim_space: SimSpace::Local,
            },
        },
        // Hybrid space offsets like world space.
        EmitterQuery::BuildSpawn {
            origin: v3([2.0, -1.0, 3.0]),
            emitter_velocity: v3([1.0, 2.0, 3.0]),
            sample: SpawnSample {
                position: v3([0.25, 0.5, -0.5]),
                direction: v3([0.0, 1.0, 0.0]),
            },
            params: SpawnParams {
                speed: 4.0,
                inherit_velocity: 0.75,
                sim_space: SimSpace::Hybrid,
            },
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn spawn_count_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let bursts = bursts_from(&[BurstSpawn {
        time: 0.45,
        count: 7,
    }]);
    // 33/s over dt=0.1 => 3 from the rate (carry 0.3), plus the burst at t=0.45
    // in the window (0.4, 0.5].
    let queries = vec![
        EmitterQuery::SpawnCount {
            carry_in: 0.0,
            rate_per_second: 33.0,
            dt: 0.1,
            bursts,
            burst_count: 1,
            time: 0.5,
        },
        // A frame whose window misses the burst: only the rate contributes.
        EmitterQuery::SpawnCount {
            carry_in: 0.0,
            rate_per_second: 33.0,
            dt: 0.1,
            bursts,
            burst_count: 1,
            time: 0.9,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEmitter::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_0f0f ^ 0x9e37_79b9_7f4a_7c15;
    let mut queries = Vec::new();
    for _ in 0..12 {
        // Samples in [0.3, 0.7] => signed components in [-0.4, 0.4], so the first
        // disk/ball candidate is comfortably accepted on both paths.
        let s = samples_from(&[
            ranged(&mut state, 0.3, 0.7),
            ranged(&mut state, 0.3, 0.7),
            ranged(&mut state, 0.3, 0.7),
            ranged(&mut state, 0.3, 0.7),
            ranged(&mut state, 0.3, 0.7),
            ranged(&mut state, 0.3, 0.7),
        ]);
        queries.push(EmitterQuery::SampleUnitDisk {
            samples: s,
            sample_len: 6,
        });
        queries.push(EmitterQuery::SampleUnitBall {
            samples: s,
            sample_len: 6,
        });
        queries.push(EmitterQuery::SampleShape {
            shape: EmitterShape::Sphere {
                radius: ranged(&mut state, 1.0, 5.0),
                surface_only: lcg(&mut state) < 0.5,
            },
            samples: s,
            sample_len: 6,
        });
        queries.push(EmitterQuery::SampleShape {
            shape: EmitterShape::Box {
                half_extents: v3([
                    ranged(&mut state, 1.0, 4.0),
                    ranged(&mut state, 1.0, 4.0),
                    ranged(&mut state, 1.0, 4.0),
                ]),
            },
            samples: s,
            sample_len: 6,
        });
        queries.push(EmitterQuery::SampleShape {
            shape: EmitterShape::Cone {
                base_radius: ranged(&mut state, 1.0, 3.0),
                height: ranged(&mut state, 1.0, 3.0),
            },
            samples: s,
            sample_len: 6,
        });
        // `rate * dt` kept well clear of an integer (fractional part near 0.3).
        let dt = 0.1;
        let whole = (lcg(&mut state) * 5.0) as u32;
        let rate = (whole as f32 + 0.3) / dt;
        queries.push(EmitterQuery::Accumulate {
            carry_in: ranged(&mut state, 0.0, 0.4),
            rate_per_second: rate,
            dt,
        });
        // Bursts strictly inside or outside the window (0.5, 1.5].
        let bursts = bursts_from(&[
            BurstSpawn {
                time: ranged(&mut state, 0.6, 1.4),
                count: (lcg(&mut state) * 20.0) as u32,
            },
            BurstSpawn {
                time: ranged(&mut state, 1.6, 2.4),
                count: (lcg(&mut state) * 20.0) as u32,
            },
        ]);
        queries.push(EmitterQuery::BurstsInWindow {
            bursts,
            burst_count: 2,
            start: 0.5,
            end: 1.5,
        });
        queries.push(EmitterQuery::InheritedVelocity {
            emitter_velocity: v3([
                ranged(&mut state, -5.0, 5.0),
                ranged(&mut state, -5.0, 5.0),
                ranged(&mut state, -5.0, 5.0),
            ]),
            factor: ranged(&mut state, 0.0, 1.0),
        });
        let space = match (lcg(&mut state) * 3.0) as u32 {
            0 => SimSpace::Local,
            1 => SimSpace::World,
            _ => SimSpace::Hybrid,
        };
        queries.push(EmitterQuery::BuildSpawn {
            origin: v3([
                ranged(&mut state, -3.0, 3.0),
                ranged(&mut state, -3.0, 3.0),
                ranged(&mut state, -3.0, 3.0),
            ]),
            emitter_velocity: v3([
                ranged(&mut state, -4.0, 4.0),
                ranged(&mut state, -4.0, 4.0),
                ranged(&mut state, -4.0, 4.0),
            ]),
            sample: SpawnSample {
                position: v3([
                    ranged(&mut state, -1.0, 1.0),
                    ranged(&mut state, -1.0, 1.0),
                    ranged(&mut state, -1.0, 1.0),
                ]),
                direction: v3([
                    ranged(&mut state, -1.0, 1.0),
                    ranged(&mut state, -1.0, 1.0),
                    ranged(&mut state, -1.0, 1.0),
                ]),
            },
            params: SpawnParams {
                speed: ranged(&mut state, 0.0, 6.0),
                inherit_velocity: ranged(&mut state, 0.0, 1.0),
                sim_space: space,
            },
        });
        // Fresh-emitter spawn count: rate contributes a clear whole plus a burst
        // placed inside the window (time - dt, time].
        let whole2 = (lcg(&mut state) * 5.0) as u32;
        let rate2 = (whole2 as f32 + 0.3) / dt;
        let time = 1.0_f32;
        let count = (lcg(&mut state) * 20.0) as u32;
        let sc_bursts = bursts_from(&[BurstSpawn {
            time: ranged(&mut state, time - dt + 0.01, time - 0.005),
            count,
        }]);
        queries.push(EmitterQuery::SpawnCount {
            carry_in: 0.0,
            rate_per_second: rate2,
            dt,
            bursts: sc_bursts,
            burst_count: 1,
            time,
        });
    }
    check(&ctx, &gpu, &queries);
}
