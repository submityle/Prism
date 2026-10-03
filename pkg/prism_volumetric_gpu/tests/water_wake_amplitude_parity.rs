//! Real-device parity for the wake-scalar twin:
//! [`GpuWaterWakeAmplitude`](prism_volumetric_gpu::water_wake_amplitude::GpuWaterWakeAmplitude)
//! must reproduce the three stateless scalar primitives of the `CPU` golden
//! [`wake`](prism_render_architecture::water::wake) —
//! [`transverse_wavelength`](prism_render_architecture::water::wake::transverse_wavelength),
//! [`froude_length`](prism_render_architecture::water::wake::froude_length) and
//! [`emission_amplitude`](prism_render_architecture::water::wake::emission_amplitude)
//! — across healthy fixtures, every degenerate branch, and a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden free functions
//! directly: `transverse_wavelength` and `froude_length` take scalars, and
//! `emission_amplitude` is fed a
//! [`HullMotion`](prism_render_architecture::water::wake::HullMotion) and a
//! [`WakeConfig`](prism_render_architecture::water::wake::WakeConfig) built from
//! the same query, so a `GPU == golden` pass is direct evidence the port is
//! faithful.
//!
//! # Parity criterion
//!
//! The three outputs thread through a `sqrt` and guarded divisions, so a `GPU`
//! `sqrt` or divide may land a few units in the last place from the scalar
//! reference; each is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`.
//!
//! # Conditioning
//!
//! The randomized "healthy" sweep rejection-samples every `EPS`-gated input far
//! from its threshold so parity is never pinned on a tie; the degenerate zero
//! paths are covered by dedicated deterministic fixtures that deliberately sit
//! on the far side of each gate.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wake`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::wake::{
    emission_amplitude, froude_length, transverse_wavelength, HullMotion, WakeConfig,
};
use prism_render_architecture::water::Vec2;
use prism_volumetric_gpu::water_wake_amplitude::{
    GpuWaterWakeAmplitude, WaterWakeAmplitudeQuery, WaterWakeAmplitudeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar output. A `GPU` `sqrt` or divide may land a
/// few units in the last place from the scalar reference; `1e-5` admits that
/// legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A linear congruential generator for host-side randomized fixtures; the
/// device never sees a `u64`.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        lo + (hi - lo) * u
    }
}

/// Builds a healthy, fully valid query far from every `EPS` gate.
fn base() -> WaterWakeAmplitudeQuery {
    WaterWakeAmplitudeQuery {
        speed: 5.0,
        gravity: 9.81,
        length_m: 12.0,
        hull_speed: 5.0,
        min_speed: 1.0,
        reference_speed: 6.0,
        base_amplitude: 1.0,
        draft_m: 1.5,
        source_spacing_m: 2.0,
        trail_length_m: 20.0,
        max_sources: 64,
    }
}

/// Reconstructs the expected result by calling the golden free functions.
fn oracle(q: &WaterWakeAmplitudeQuery) -> WaterWakeAmplitudeResult {
    let hull = HullMotion {
        position: Vec2::new(0.0, 0.0),
        forward: Vec2::new(1.0, 0.0),
        speed: q.hull_speed,
        beam_m: 1.0,
        draft_m: q.draft_m,
    };
    let cfg = WakeConfig {
        gravity: q.gravity,
        min_speed: q.min_speed,
        source_spacing_m: q.source_spacing_m,
        trail_length_m: q.trail_length_m,
        reference_speed: q.reference_speed,
        base_amplitude: q.base_amplitude,
        max_sources: q.max_sources,
    };
    WaterWakeAmplitudeResult {
        transverse_wavelength: transverse_wavelength(q.speed, q.gravity),
        froude_length: froude_length(q.speed, q.length_m, q.gravity),
        emission_amplitude: emission_amplitude(&hull, &cfg),
    }
}

/// Pins one `GPU` result against the golden oracle, output-for-output.
fn check_query(idx: usize, got: &WaterWakeAmplitudeResult, want: &WaterWakeAmplitudeResult) {
    assert!(
        close(got.transverse_wavelength, want.transverse_wavelength),
        "query {idx} transverse_wavelength: gpu {} vs cpu {}",
        got.transverse_wavelength,
        want.transverse_wavelength
    );
    assert!(
        close(got.froude_length, want.froude_length),
        "query {idx} froude_length: gpu {} vs cpu {}",
        got.froude_length,
        want.froude_length
    );
    assert!(
        close(got.emission_amplitude, want.emission_amplitude),
        "query {idx} emission_amplitude: gpu {} vs cpu {}",
        got.emission_amplitude,
        want.emission_amplitude
    );
}

/// Dispatches `queries` on the device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterWakeAmplitude, queries: &[WaterWakeAmplitudeQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_query(idx, g, &oracle(q));
    }
}

/// The deterministic degenerate fixtures that exercise every zero branch.
fn degenerate_fixtures() -> Vec<WaterWakeAmplitudeQuery> {
    vec![
        // gravity = 0: transverse_wavelength -> 0, froude denom -> 0, invalid cfg.
        WaterWakeAmplitudeQuery {
            gravity: 0.0,
            ..base()
        },
        // length_m = 0: froude denom -> 0 (wavelength and amplitude unaffected).
        WaterWakeAmplitudeQuery {
            length_m: 0.0,
            ..base()
        },
        // hull_speed below the cut-in: emission_amplitude -> 0.
        WaterWakeAmplitudeQuery {
            hull_speed: 0.5,
            min_speed: 1.0,
            ..base()
        },
        // Invalid cfg: source_spacing_m <= EPS.
        WaterWakeAmplitudeQuery {
            source_spacing_m: 0.0,
            ..base()
        },
        // Invalid cfg: trail_length_m <= EPS.
        WaterWakeAmplitudeQuery {
            trail_length_m: 0.0,
            ..base()
        },
        // Invalid cfg: reference_speed <= EPS.
        WaterWakeAmplitudeQuery {
            reference_speed: 0.0,
            ..base()
        },
        // Invalid cfg: base_amplitude < 0.
        WaterWakeAmplitudeQuery {
            base_amplitude: -1.0,
            ..base()
        },
        // Invalid cfg: min_speed < 0.
        WaterWakeAmplitudeQuery {
            min_speed: -1.0,
            ..base()
        },
        // Invalid cfg: max_sources = 0.
        WaterWakeAmplitudeQuery {
            max_sources: 0,
            ..base()
        },
        // Negative draft: draft_scale clamps the draft to zero -> scale 1.
        WaterWakeAmplitudeQuery {
            draft_m: -2.0,
            ..base()
        },
        // Negative speed: clamped to zero by the golden `max`.
        WaterWakeAmplitudeQuery {
            speed: -3.0,
            hull_speed: -3.0,
            ..base()
        },
    ]
}

#[test]
fn healthy_fixture_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWakeAmplitude::new(&ctx);
    check(&ctx, &gpu, &[base()]);
}

#[test]
fn degenerate_branches_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWakeAmplitude::new(&ctx);
    check(&ctx, &gpu, &degenerate_fixtures());
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWakeAmplitude::new(&ctx);
    // Zero queries short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWakeAmplitude::new(&ctx);
    let mut rng = Lcg::new(0x0f1e_2d3c_4b5a_6978);
    let mut queries = degenerate_fixtures();
    // Several workgroups' worth of valid, well-conditioned queries. Every
    // EPS-gated input is sampled far from its threshold so parity never lands
    // on a tie.
    for _ in 0..512 {
        queries.push(WaterWakeAmplitudeQuery {
            speed: rng.next_f32(0.5, 20.0),
            gravity: rng.next_f32(1.0, 20.0),
            length_m: rng.next_f32(1.0, 80.0),
            hull_speed: rng.next_f32(0.5, 20.0),
            min_speed: rng.next_f32(0.1, 2.0),
            reference_speed: rng.next_f32(1.0, 12.0),
            base_amplitude: rng.next_f32(0.1, 4.0),
            draft_m: rng.next_f32(0.1, 5.0),
            source_spacing_m: rng.next_f32(0.5, 5.0),
            trail_length_m: rng.next_f32(5.0, 50.0),
            max_sources: 1 + (rng.next_u32() % 128),
        });
    }
    check(&ctx, &gpu, &queries);
}
