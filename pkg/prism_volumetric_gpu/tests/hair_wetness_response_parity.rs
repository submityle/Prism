//! Real-device parity for the wet-hair optical/physical response twin:
//! [`GpuHairWetnessResponse`](prism_volumetric_gpu::hair_wetness_response::GpuHairWetnessResponse)
//! must reproduce the `CPU` golden `sanitize_wetness`, `wet_hair_response`,
//! `WetHairResponse::apply_absorption` and `WetHairResponse::apply_roughness` of
//! `prism_render_architecture::hair::wetness`, which map a single water
//! saturation scalar to the pigment-absorption multiplier applied per `RGB`
//! channel and the additive, unit-clamped roughness offset applied to a base
//! roughness.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `[0, 1]` saturation sanitiser (non-finite collapses to fully dry), the
//! linearly interpolated `sigma_a` multiplier and roughness offset, the
//! per-channel absorption scale and the unit-clamped roughness — so the test
//! never imports `prism_render_architecture`.
//!
//! The fixtures cover the dry and fully-wet endpoints, interior saturation
//! midpoints, the degenerate `NaN` / `+inf` / `-inf` / out-of-range inputs that
//! all sanitise to `[0, 1]`, both ends of the roughness clamp, and a batch of
//! at least two distinct queries that catches any `std430` stride aliasing. A
//! sweep over random `(wetness, base_absorption, base_roughness)` follows, plus
//! an empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through multiplies and adds, so `CPU` and
//! `GPU` evaluate the same closed form but need not be bit-exact. The
//! continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly and is
//! always `1`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::wetness`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_wetness_response::{
    GpuHairWetnessResponse, HairWetnessResponseQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Pigment-absorption multiplier at full saturation; dry endpoint is `1.0`.
const SIGMA_A_MUL_WET: f32 = 1.5;
/// Additive roughness offset at full saturation; dry endpoint is `0.0`.
const ROUGHNESS_DELTA_WET: f32 = -0.35;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent oracle for the `[0, 1]` saturation sanitiser: non-finite inputs
/// collapse to fully dry (`0.0`).
fn sanitize_wetness(wetness: f32) -> f32 {
    if wetness.is_finite() {
        wetness.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Linear interpolation between the dry endpoint `dry` and the wet endpoint
/// `wet` by the saturation `w`.
fn lerp(dry: f32, wet: f32, w: f32) -> f32 {
    dry + (wet - dry) * w
}

/// The independent oracle for one query:
/// `(out_absorption, out_roughness, sanitized_wetness, valid)`.
fn oracle(q: &HairWetnessResponseQuery) -> ([f32; 3], f32, f32, u32) {
    let w = sanitize_wetness(q.wetness);
    let sigma_a_mul = lerp(1.0, SIGMA_A_MUL_WET, w);
    let roughness_delta = lerp(0.0, ROUGHNESS_DELTA_WET, w);
    let out_absorption = [
        q.base_absorption[0] * sigma_a_mul,
        q.base_absorption[1] * sigma_a_mul,
        q.base_absorption[2] * sigma_a_mul,
    ];
    let out_roughness = (q.base_roughness + roughness_delta).clamp(0.0, 1.0);
    (out_absorption, out_roughness, w, 1u32)
}

/// Dispatches one query and asserts every channel plus the validity flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairWetnessResponse, q: HairWetnessResponseQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (out_absorption, out_roughness, sanitized, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    for (channel, (g, c)) in r
        .out_absorption
        .iter()
        .zip(out_absorption.iter())
        .enumerate()
    {
        assert!(
            close(*g, *c),
            "out_absorption[{channel}] mismatch: gpu={g} cpu={c} query={q:?}"
        );
    }
    assert!(
        close(r.out_roughness, out_roughness),
        "out_roughness mismatch: gpu={} cpu={out_roughness} query={q:?}",
        r.out_roughness
    );
    assert!(
        close(r.sanitized_wetness, sanitized),
        "sanitized_wetness mismatch: gpu={} cpu={sanitized} query={q:?}",
        r.sanitized_wetness
    );
}

#[test]
fn dry_endpoint_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // Fully dry: every modifier is a no-op, so the outputs equal the inputs.
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(0.0, [0.3, 0.5, 0.8], 0.4),
    );
}

#[test]
fn fully_wet_endpoint_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // Full saturation: absorption scales by SIGMA_A_MUL_WET and roughness drops
    // by ROUGHNESS_DELTA_WET (clamped if it would go negative).
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(1.0, [0.2, 0.4, 0.6], 0.7),
    );
}

#[test]
fn saturation_midpoints_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // Interior saturations where both modifiers linearly interpolate.
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(0.25, [0.15, 0.35, 0.55], 0.5),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(0.5, [0.25, 0.45, 0.65], 0.6),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(0.75, [0.35, 0.55, 0.75], 0.8),
    );
}

#[test]
fn non_finite_wetness_sanitizes_to_dry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // NaN and both infinities collapse to fully dry, so outputs equal inputs.
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(f32::NAN, [0.3, 0.5, 0.7], 0.45),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(f32::INFINITY, [0.3, 0.5, 0.7], 0.45),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(f32::NEG_INFINITY, [0.3, 0.5, 0.7], 0.45),
    );
}

#[test]
fn out_of_range_wetness_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // Negative saturations clamp to 0 (dry); greater-than-one clamp to 1 (wet).
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(-3.0, [0.4, 0.5, 0.6], 0.5),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(2.5, [0.4, 0.5, 0.6], 0.5),
    );
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(9.0, [0.4, 0.5, 0.6], 0.5),
    );
}

#[test]
fn roughness_lower_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // A low base roughness, fully wet: base + ROUGHNESS_DELTA_WET would go
    // negative, so the output clamps to 0.
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(1.0, [0.2, 0.3, 0.4], 0.1),
    );
}

#[test]
fn roughness_upper_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // A high base roughness, fully dry: the offset is 0 so the output stays at
    // the base, confirming the upper end of the valid range.
    assert_parity(
        &ctx,
        &gpu,
        HairWetnessResponseQuery::new(0.0, [0.5, 0.6, 0.7], 1.0),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        HairWetnessResponseQuery::new(0.0, [0.1, 0.2, 0.3], 0.2),
        HairWetnessResponseQuery::new(0.6, [0.4, 0.5, 0.6], 0.7),
        HairWetnessResponseQuery::new(1.0, [0.7, 0.8, 0.9], 0.9),
        HairWetnessResponseQuery::new(0.33, [0.15, 0.25, 0.35], 0.5),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (out_absorption, out_roughness, sanitized, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        for (channel, (g, c)) in r
            .out_absorption
            .iter()
            .zip(out_absorption.iter())
            .enumerate()
        {
            assert!(
                close(*g, *c),
                "batch out_absorption[{channel}] mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
        assert!(
            close(r.out_roughness, out_roughness),
            "batch out_roughness mismatch: gpu={} cpu={out_roughness} query={q:?}",
            r.out_roughness
        );
        assert!(
            close(r.sanitized_wetness, sanitized),
            "batch sanitized_wetness mismatch: gpu={} cpu={sanitized} query={q:?}",
            r.sanitized_wetness
        );
    }
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
    let gpu = GpuHairWetnessResponse::new(&ctx);
    let mut rng = Lcg::new(0x5A_1D_9E_27);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Keep wetness off the clamp knees at 0 and 1 so the sanitiser is in its
        // interior linear regime; absorption and roughness span their ranges.
        let wetness = rng.next_range(0.02, 0.98);
        let base_absorption = [
            rng.next_range(0.0, 2.0),
            rng.next_range(0.0, 2.0),
            rng.next_range(0.0, 2.0),
        ];
        let base_roughness = rng.next_range(0.0, 1.0);
        queries.push(HairWetnessResponseQuery::new(
            wetness,
            base_absorption,
            base_roughness,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (out_absorption, out_roughness, sanitized, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        for (channel, (g, c)) in r
            .out_absorption
            .iter()
            .zip(out_absorption.iter())
            .enumerate()
        {
            assert!(
                close(*g, *c),
                "sweep out_absorption[{channel}] mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
        assert!(
            close(r.out_roughness, out_roughness),
            "sweep out_roughness mismatch: gpu={} cpu={out_roughness} query={q:?}",
            r.out_roughness
        );
        assert!(
            close(r.sanitized_wetness, sanitized),
            "sweep sanitized_wetness mismatch: gpu={} cpu={sanitized} query={q:?}",
            r.sanitized_wetness
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairWetnessResponse::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
