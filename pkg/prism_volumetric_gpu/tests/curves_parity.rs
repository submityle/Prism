//! Real-device parity for the per-`t` curve-sample twin:
//! [`GpuCurve`](prism_volumetric_gpu::curves::GpuCurve) must reproduce the
//! `CPU` golden
//! [`Curve::sample`](prism_render_architecture::particle::curves::Curve::sample)
//! across all five
//! [`InterpolationMode`](prism_render_architecture::particle::curves::InterpolationMode)
//! families (`Step`, `Linear`, `Hermite`, `CatmullRom`, `Bezier`).
//!
//! The fixtures cover key counts of `1`, `2`, `3` and `5` under every mode, each
//! sampled below the first key, on the two endpoints, between keys and exactly
//! on an interior key (`t` of `-0.1`, `0`, `0.25`, `0.5`, `0.75`, `1.0`, `1.1`).
//! Key times are distinct, well-spread simple decimals so no segment collapses
//! (the local parameter stays clear of its zero-width guard) and the
//! `partition_point` boundary is never an ambiguous tie; values and tangents are
//! exact `0.1`-step jitter shared bit-for-bit between the `CPU` and `GPU`
//! inputs. All curves are flattened into one shared keyframe buffer and one
//! dispatch, exercising the one-thread-per-query path. A separate test drives
//! the empty-batch host short-circuit.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The sampled value threads through multiplies, adds and one guarded division,
//! so it is compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::curves`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::curves::{Curve, InterpolationMode, Keyframe};
use prism_volumetric_gpu::curves::{
    CurveSampleQuery, GpuCurve, GpuKeyframe, CURVE_BEZIER, CURVE_CATMULL_ROM, CURVE_HERMITE,
    CURVE_LINEAR, CURVE_STEP,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous sampled value.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous sampled value.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Sample parameters: below the first key, the two endpoints, two between-key
/// interiors and one exactly on an interior key, plus above the last key.
const T_SAMPLES: [f32; 7] = [-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1];

/// The five interpolation modes paired with their kernel `u32` code, exercised
/// for every multi-key curve.
const MODES: [(InterpolationMode, u32); 5] = [
    (InterpolationMode::Step, CURVE_STEP),
    (InterpolationMode::Linear, CURVE_LINEAR),
    (InterpolationMode::Hermite, CURVE_HERMITE),
    (InterpolationMode::CatmullRom, CURVE_CATMULL_ROM),
    (InterpolationMode::Bezier, CURVE_BEZIER),
];

/// Mixed absolute / relative tolerance comparison for one `f32` value.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
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
    /// `[-0.3, 0.3]`, built from integer arithmetic so the value is an exact
    /// small decimal shared bit-for-bit between the `CPU` and `GPU` inputs.
    fn jitter(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        // Integer in 0..=6, centred to -3..=3, then scaled to a 0.1-step grid.
        let step = (bits % 7) as i32 - 3;
        step as f32 / 10.0
    }
}

/// One sampled comparison: the golden curve, the parameter and the index of its
/// query in the flattened batch.
struct Case {
    /// Golden reference curve.
    curve: Curve,
    /// Sample parameter.
    t: f32,
    /// Index of this `(curve, t)` query in the flattened batch.
    index: usize,
}

/// Accumulates keyframes, queries and golden cases for a single batched
/// dispatch.
struct Batch {
    /// Shared flattened keyframes for every query, in `Curve`-sorted order.
    keys: Vec<GpuKeyframe>,
    /// One query per `(curve, t)` pair.
    queries: Vec<CurveSampleQuery>,
    /// Golden cases parallel to the queries.
    cases: Vec<Case>,
    /// Jitter source for the key values and tangents.
    rng: Lcg,
}

impl Batch {
    /// Starts an empty batch seeded for deterministic key payloads.
    fn new(seed: u64) -> Batch {
        Batch {
            keys: Vec::new(),
            queries: Vec::new(),
            cases: Vec::new(),
            rng: Lcg::new(seed),
        }
    }

    /// Builds `n` keyframes with distinct, well-spread times in `(0.15, 0.85)`
    /// and `0.1`-step jittered values and tangents. The times are exact simple
    /// decimals, so no two keys collide and no segment width collapses.
    fn make_keys(&mut self, n: usize) -> Vec<Keyframe> {
        let mut keys = Vec::with_capacity(n);
        for i in 0..n {
            let time = if n == 1 {
                0.5
            } else {
                0.15 + (i as f32) * (0.7 / ((n - 1) as f32))
            };
            let value = 1.0 + (i as f32) * 0.5 + self.rng.jitter();
            let in_tangent = 0.5 + self.rng.jitter();
            let out_tangent = -0.5 + self.rng.jitter();
            keys.push(Keyframe::with_tangents(
                time,
                value,
                in_tangent,
                out_tangent,
            ));
        }
        keys
    }

    /// Adds a curve of `n` keys under `mode`/`code`, queued at every parameter
    /// in [`T_SAMPLES`]. The keyframes are uploaded in the `Curve`-sorted order
    /// the reference evaluates, so the `GPU` sub-slice matches it exactly.
    fn add(&mut self, n: usize, mode: InterpolationMode, code: u32) {
        let authored = self.make_keys(n);
        let curve = Curve::from_keys(mode, authored);
        let offset = self.keys.len() as u32;
        for k in curve.keys() {
            self.keys.push(GpuKeyframe::new(
                k.time,
                k.value,
                k.in_tangent,
                k.out_tangent,
            ));
        }
        for &t in &T_SAMPLES {
            let index = self.queries.len();
            self.queries
                .push(CurveSampleQuery::new(offset, n as u32, code, t));
            self.cases.push(Case {
                curve: curve.clone(),
                t,
                index,
            });
        }
    }
}

#[test]
fn parity_across_modes_and_key_counts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCurve::new(&ctx);
    let mut batch = Batch::new(0x5101_2357_9bdf_1113);
    // Single-key curves ignore the mode and hold the lone value everywhere; one
    // per mode confirms the n == 1 short-circuit.
    for &(mode, code) in &MODES {
        batch.add(1, mode, code);
    }
    // Multi-key curves under every mode, at key counts 2, 3 and 5.
    for &n in &[2_usize, 3, 5] {
        for &(mode, code) in &MODES {
            batch.add(n, mode, code);
        }
    }

    let results = gpu.sample(&batch.queries, &batch.keys);
    assert_eq!(results.len(), batch.queries.len());
    for case in &batch.cases {
        let cpu = case.curve.sample(case.t);
        let got = results[case.index];
        assert!(
            approx(got, cpu),
            "sample mismatch (mode {:?}, t={}): gpu {got} vs cpu {cpu}",
            case.curve.mode(),
            case.t
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCurve::new(&ctx);
    // No queries: no dispatch is issued and the result vector is empty. The
    // shared keyframe buffer is non-empty so this isolates the query
    // short-circuit.
    let keys = [
        GpuKeyframe::new(0.0, 1.0, 0.0, 0.0),
        GpuKeyframe::new(1.0, 2.0, 0.0, 0.0),
    ];
    let out = gpu.sample(&[], &keys);
    assert!(out.is_empty());
}
