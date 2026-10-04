//! Real-device parity for the painted max-distance clamp twin:
//! [`GpuClothClampMaxDistance`](prism_volumetric_gpu::cloth_clamp_max_distance::GpuClothClampMaxDistance)
//! must reproduce the `CPU` golden `clamp_max_distance` composed with
//! `PaintedConstraint::clamped` of `prism_render_architecture::cloth`, which
//! sanitises a painted `max_distance`, then skips pinned / uncapped /
//! inside-sphere vertices and projects an over-limit vertex radially back onto
//! its max-distance sphere (welding to the anchor when the drift is vanishing).
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! `clamped` sanitiser (`NaN` collapses to `0`, negatives lift to `0`, `+inf`
//! preserved), the pinned / non-finite / inside-sphere skips, the radial
//! projection and the length-epsilon weld — so the test never imports
//! `prism_render_architecture`.
//!
//! The fixtures cover a pinned vertex, a `NaN` `max_distance` that welds to the
//! anchor, a `+inf` `max_distance` left free, a vertex inside its sphere, a
//! vertex projected onto the sphere, a negative `max_distance` lifted to `0`, a
//! vanishing drift that welds to the anchor, and a batch of at least two
//! distinct queries that catches any `std430` stride aliasing. A sweep over
//! random `(particle, anchor, max_distance, pinned)` follows, with knee-point
//! rejection sampling, plus an empty batch the host short-circuits with no
//! dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous position channel threads through subtracts, a dot product, a
//! `sqrt` and a divide, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly and is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted` 与
//! `prism_render_architecture::cloth::asset`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_clamp_max_distance::{
    ClothClampMaxDistanceQuery, GpuClothClampMaxDistance,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared length epsilon matching the golden cloth `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent oracle for the `clamped` sanitiser: a `NaN` collapses to `0`, a
/// negative value lifts to `0`, and a finite non-negative value (including the
/// `+inf` "no cap" default) is otherwise preserved.
fn clamped_max_distance(max_distance: f32) -> f32 {
    if max_distance.is_nan() {
        0.0
    } else {
        max_distance.max(0.0)
    }
}

/// The independent oracle for one query: the clamped `(position, valid)`.
fn oracle(q: &ClothClampMaxDistanceQuery) -> ([f32; 3], u32) {
    let md = clamped_max_distance(q.max_distance);
    if q.pinned != 0 {
        return (q.particle, 1);
    }
    if !md.is_finite() {
        return (q.particle, 1);
    }
    let drift = [
        q.particle[0] - q.anchor[0],
        q.particle[1] - q.anchor[1],
        q.particle[2] - q.anchor[2],
    ];
    let dist_sq = drift[0] * drift[0] + drift[1] * drift[1] + drift[2] * drift[2];
    let max_sq = md * md;
    if dist_sq <= max_sq {
        return (q.particle, 1);
    }
    if dist_sq > EPS_LEN_SQ {
        let dist = dist_sq.sqrt();
        let scale = md / dist;
        let position = [
            q.anchor[0] + drift[0] * scale,
            q.anchor[1] + drift[1] * scale,
            q.anchor[2] + drift[2] * scale,
        ];
        return (position, 1);
    }
    (q.anchor, 1)
}

/// Dispatches one query and asserts every position channel plus the validity
/// flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuClothClampMaxDistance, q: ClothClampMaxDistanceQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (position, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    for (axis, (g, c)) in r.position.iter().zip(position.iter()).enumerate() {
        assert!(
            close(*g, *c),
            "position[{axis}] mismatch: gpu={g} cpu={c} query={q:?}"
        );
    }
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuClothClampMaxDistance,
    queries: &[ClothClampMaxDistanceQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (position, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        for (axis, (g, c)) in r.position.iter().zip(position.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "batch position[{axis}] mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}

#[test]
fn pinned_vertex_is_untouched() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A pinned vertex keeps its position even when it is far outside the sphere.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([5.0, -3.0, 2.0], [0.0, 0.0, 0.0], 0.5, 1),
    );
}

#[test]
fn nan_max_distance_welds_to_anchor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A NaN max_distance sanitises to 0 (a weld-to-anchor pin), so the over-limit
    // vertex collapses onto the anchor.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([2.0, 1.0, -1.0], [0.5, 0.5, 0.5], f32::NAN, 0),
    );
}

#[test]
fn infinite_max_distance_leaves_free() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // The +inf "no cap" default leaves the vertex free regardless of its drift.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([9.0, 9.0, 9.0], [0.0, 0.0, 0.0], f32::INFINITY, 0),
    );
}

#[test]
fn inside_sphere_is_untouched() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A drift of length ~0.5 inside a 2.0 cap stays untouched.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([0.3, 0.4, 0.0], [0.0, 0.0, 0.0], 2.0, 0),
    );
}

#[test]
fn over_limit_projects_onto_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A drift of length 5 against a 1.0 cap projects onto the unit sphere about
    // the anchor; the result sits at distance 1 along the drift direction.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([3.0, 4.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0),
    );
}

#[test]
fn negative_max_distance_lifts_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A negative cap lifts to 0, welding the drifted vertex to the anchor.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], -4.0, 0),
    );
}

#[test]
fn vanishing_drift_welds_to_anchor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A drift far below the length epsilon with a zero cap welds to the anchor
    // rather than normalising a vanishing vector. 1e-7 squared is 1e-14 < 1e-12,
    // so dist_sq is at or below EPS_LEN_SQ while still exceeding max_sq = 0.
    assert_parity(
        &ctx,
        &gpu,
        ClothClampMaxDistanceQuery::new([1.0e-7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        ClothClampMaxDistanceQuery::new([3.0, 4.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0),
        ClothClampMaxDistanceQuery::new([0.1, 0.1, 0.1], [0.0, 0.0, 0.0], 2.0, 0),
        ClothClampMaxDistanceQuery::new([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], f32::INFINITY, 0),
        ClothClampMaxDistanceQuery::new([2.0, -2.0, 1.0], [0.0, 0.0, 0.0], 0.5, 1),
    ];
    assert_batch(&ctx, &gpu, &queries);
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    let mut rng = Lcg::new(0x0C_1A_7D_51);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let anchor = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];
        let drift = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];
        let particle = [
            anchor[0] + drift[0],
            anchor[1] + drift[1],
            anchor[2] + drift[2],
        ];
        let max_distance = rng.next_range(0.1, 5.0);
        let pinned = (rng.next_u32() & 1) as u32;

        let dist_sq = drift[0] * drift[0] + drift[1] * drift[1] + drift[2] * drift[2];
        let max_sq = max_distance * max_distance;
        // Reject samples that sit on the inside/outside knee (dist_sq ~ max_sq)
        // or near the length-epsilon weld knee, so a last-bit rounding split can
        // never flip which branch the CPU and GPU take.
        let knee = (dist_sq - max_sq).abs();
        if knee <= 1.0e-2 {
            continue;
        }
        if dist_sq <= 1.0e-6 {
            continue;
        }
        queries.push(ClothClampMaxDistanceQuery::new(
            particle,
            anchor,
            max_distance,
            pinned,
        ));
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothClampMaxDistance::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
