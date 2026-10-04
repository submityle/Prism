//! Real-device parity for the plastic rest-length creep twin:
//! [`GpuClothPlasticRestLength`](prism_volumetric_gpu::cloth_plastic_rest_length::GpuClothPlasticRestLength)
//! must reproduce the `CPU` golden `plastic_rest_length` composed with
//! `PlasticParams::sanitized` of
//! `prism_physics_core::soft::damage::plasticity`, which sanitises the painted
//! plastic parameters, rejects a degenerate rest length and an edge inside the
//! yield band, creeps the rest length by the creep fraction of the beyond-yield
//! excess strain, floors it to a numerical epsilon, then clamps the residual
//! elastic strain against `max_strain`.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! `sanitized` rule (`NaN` to `0`, negatives lifted to `0`, `creep` clamped to
//! `[0, 1]`), the degenerate-rest and yield-band rejections, the creep update,
//! the epsilon floor and the residual cap — so the test never imports
//! `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover an edge inside the yield band (rejected), a tensile creep,
//! a compressive creep, a residual-capped edge, a creep that collapses to the
//! epsilon floor (rejected), a degenerate rest length (rejected), a `NaN`
//! parameter triple that sanitises to zeros, and a batch of at least two
//! distinct queries that catches any `std430` stride aliasing. A sweep over
//! random `(rest, length, yield, creep, max)` follows, with knee-point rejection
//! sampling, plus an empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The continuous `new_rest` channel threads through subtracts, divides and
//! multiplies, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a device may fuse a multiply-add the scalar reference leaves
//! separate). The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::plasticity`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_plastic_rest_length::{
    ClothPlasticRestLengthQuery, GpuClothPlasticRestLength,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Rest length at or below which an edge is inert, matching the golden EPS_REST.
const EPS_REST: f32 = 1.0e-9;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent oracle for `clamp_nonneg`: a `NaN` collapses to `0`, a negative
/// value lifts to `0`, and a finite non-negative value is preserved.
fn clamp_nonneg(v: f32) -> f32 {
    if v.is_nan() || v < 0.0 {
        0.0
    } else {
        v
    }
}

/// Independent oracle for the creep sanitiser: a `NaN` collapses to `0`,
/// otherwise the value is clamped into `[0, 1]`.
fn sanitize_creep(c: f32) -> f32 {
    if c.is_nan() {
        0.0
    } else {
        c.clamp(0.0, 1.0)
    }
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly: the crept rest length (or the epsilon placeholder when
/// rejected) and the validity flag.
fn oracle(q: &ClothPlasticRestLengthQuery) -> (f32, u32) {
    let yield_strain = clamp_nonneg(q.yield_strain);
    let creep = sanitize_creep(q.creep);
    let max_strain = clamp_nonneg(q.max_strain);
    let rest = q.rest_length;
    let length = q.length;

    let rest_ok = rest > EPS_REST;
    let safe_rest = if rest_ok { rest } else { 1.0 };
    let strain = (length - rest) / safe_rest;
    let beyond = strain.abs() > yield_strain;

    let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
    let excess = strain - sign * yield_strain;
    let mut new_rest = rest * (1.0 + creep * excess);
    if new_rest <= EPS_REST {
        new_rest = EPS_REST;
    }

    let residual = (length - new_rest) / new_rest;
    let residual_sign = if residual >= 0.0 { 1.0 } else { -1.0 };
    let capped = length / (1.0 + residual_sign * max_strain);
    if residual.abs() > max_strain {
        new_rest = capped;
    }

    let accepted = rest_ok && beyond && (new_rest > EPS_REST);
    if accepted {
        (new_rest, 1)
    } else {
        (EPS_REST, 0)
    }
}

/// Dispatches one query and asserts the crept rest length plus the validity
/// flag.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuClothPlasticRestLength,
    q: ClothPlasticRestLengthQuery,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (new_rest, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    assert!(
        close(r.new_rest, new_rest),
        "new_rest mismatch: gpu={} cpu={new_rest} query={q:?}",
        r.new_rest
    );
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuClothPlasticRestLength,
    queries: &[ClothPlasticRestLengthQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (new_rest, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.new_rest, new_rest),
            "batch new_rest mismatch: gpu={} cpu={new_rest} query={q:?}",
            r.new_rest
        );
    }
}

#[test]
fn within_yield_band_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // A strain of 0.05 stays inside the yield band of 0.1, so the edge is left
    // untouched and rejected (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0, 1.05, 0.1, 0.5, 1.0),
    );
}

#[test]
fn tensile_creep_moves_rest_length() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // strain = 1.0, excess = 0.9, new_rest = 1 * (1 + 0.5 * 0.9) = 1.45; the
    // residual is well within the generous cap, so it survives unchanged.
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0, 2.0, 0.1, 0.5, 10.0),
    );
}

#[test]
fn compressive_creep_moves_rest_length() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // strain = -0.5, excess = -0.4, new_rest = 2 * (1 + 0.5 * -0.4) = 1.6, well
    // within the cap.
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(2.0, 1.0, 0.1, 0.5, 10.0),
    );
}

#[test]
fn residual_cap_engages() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // strain = 1.0, excess = 0.9, new_rest = 1 * (1 + 0.1 * 0.9) = 1.09; the
    // residual (2 - 1.09) / 1.09 ~ 0.835 exceeds the 0.2 cap, so the rest length
    // is re-pinned to length / (1 + 0.2) = 1.6667.
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0, 2.0, 0.1, 0.1, 0.2),
    );
}

#[test]
fn creep_collapse_floored_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // A zero edge length drives strain = -1 with zero yield, so excess = -1 and
    // new_rest = 1 * (1 - 1) = 0 collapses onto the epsilon floor; the residual
    // (0 - EPS) / EPS = -1 is within the generous cap, so new_rest stays at the
    // floor and is rejected (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0, 0.0, 0.0, 1.0, 10.0),
    );
}

#[test]
fn degenerate_rest_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // A rest length below the epsilon makes the edge inert (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0e-12, 1.0, 0.1, 0.5, 10.0),
    );
}

#[test]
fn nan_params_sanitize_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // NaN yield / creep / max all sanitise to 0; strain = 1 clears the zero yield
    // band, creep = 0 leaves new_rest = rest = 1, then the zero cap re-pins to
    // length / (1 + 0) = length = 2.
    assert_parity(
        &ctx,
        &gpu,
        ClothPlasticRestLengthQuery::new(1.0, 2.0, f32::NAN, f32::NAN, f32::NAN),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        ClothPlasticRestLengthQuery::new(1.0, 2.0, 0.1, 0.5, 10.0),
        ClothPlasticRestLengthQuery::new(2.0, 1.0, 0.1, 0.5, 10.0),
        ClothPlasticRestLengthQuery::new(1.0, 1.05, 0.1, 0.5, 1.0),
        ClothPlasticRestLengthQuery::new(1.0, 2.0, 0.1, 0.1, 0.2),
        ClothPlasticRestLengthQuery::new(1.0e-12, 1.0, 0.1, 0.5, 10.0),
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
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    let mut rng = Lcg::new(0x0C_1A_7D_51);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let rest = rng.next_range(0.1, 5.0);
        let length = rng.next_range(0.0, 10.0);
        let yield_strain = rng.next_range(0.0, 0.5);
        let creep = rng.next_range(0.0, 1.0);
        let max_strain = rng.next_range(0.05, 0.8);

        // Reject samples sitting on a branch knee so a last-bit rounding split can
        // never flip which arm the CPU and GPU take. rest >= 0.1 is always a
        // valid divisor.
        let strain = (length - rest) / rest;
        if (strain.abs() - yield_strain).abs() <= 1.0e-2 {
            continue;
        }
        let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
        let excess = strain - sign * yield_strain;
        let new_rest = rest * (1.0 + creep * excess);
        // Stay well above the epsilon floor so the floor knee never fires.
        if new_rest <= 1.0e-3 {
            continue;
        }
        let residual = (length - new_rest) / new_rest;
        if (residual.abs() - max_strain).abs() <= 1.0e-2 {
            continue;
        }

        queries.push(ClothPlasticRestLengthQuery::new(
            rest,
            length,
            yield_strain,
            creep,
            max_strain,
        ));
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPlasticRestLength::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
