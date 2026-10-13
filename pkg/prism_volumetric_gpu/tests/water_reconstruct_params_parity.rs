//! Real-device parity for the per-domain surface reconstruction scalar
//! parameter twin:
//! [`GpuWaterReconstructParams`](prism_volumetric_gpu::water_reconstruct_params::GpuWaterReconstructParams)
//! must reproduce the dependency-free `CPU` golden
//! [`surface_isovalue`](prism_render_architecture::water::reconstruct::surface_isovalue),
//! [`narrow_band_width_cells`](prism_render_architecture::water::reconstruct::narrow_band_width_cells)
//! and
//! [`screen_space_smoothing_iterations`](prism_render_architecture::water::reconstruct::screen_space_smoothing_iterations)
//! — the iso level, the narrow-band half-width in cells and the screen-space
//! smoothing iteration count — across the clamp / floor-of-one / near-far fade
//! regimes, several domain counts including non-multiples of the `256`-wide
//! workgroup, a bounded randomized sweep, and the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The three goldens are public and pure, so the expected parameters are built
//! in-host, domain by domain, and compared against the `GPU` readback. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! The isovalue is a single `max`-then-multiply, so it is bit-identical up to
//! the last-place slack of a `GPU` fused multiply-add; it is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The two integer counts are exact:
//! the fixtures keep each `f32` intermediate comfortably away (at least `1e-3`)
//! from the truncation seam, so a legal `GPU` divide (2.5 ULP in core `WGSL`)
//! cannot flip the rounded integer and the counts are asserted for exact
//! equality. Samples that land on a seam are skipped by the randomized sweep.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::reconstruct`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::reconstruct::{
    narrow_band_width_cells, screen_space_smoothing_iterations, surface_isovalue,
};
use prism_volumetric_gpu::water_reconstruct_params::{
    GpuWaterReconstructParams, ReconstructDomain, WaterReconstructParams,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous isovalue field.
const EPS_ABS: f32 = 1.0e-4;

/// Relative parity bound on the continuous isovalue field.
const EPS_REL: f32 = 1.0e-3;

/// Floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum distance of a `f32` intermediate from the truncation seam below
/// which the randomized sweep declines to assert the integer counts.
const SEAM_MARGIN: f32 = 1.0e-3;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS_ABS || rel <= EPS_REL
}

/// Mirrors the golden's floating-point intermediate before truncation, so a
/// randomized sample can be skipped when it sits on the seam.
fn band_intermediate(d: &ReconstructDomain) -> f32 {
    let cell = d.dx.max(1.0e-6);
    let radius = d.particle_radius.max(0.0);
    let factor = d.band_factor.max(0.0);
    (factor * radius / cell) + 1.0 - 1.0e-6
}

/// Mirrors the golden's float that is rounded for the smoothing count.
fn smoothing_intermediate(d: &ReconstructDomain) -> f32 {
    let max_iters = d.max_iterations.max(1);
    let near = d.near_distance.max(0.0);
    let far = d.far_distance.max(near + 1.0e-6);
    let distance = d.camera_distance.max(0.0);
    if distance <= near {
        return (max_iters - 1) as f32;
    }
    if distance >= far {
        return 0.0;
    }
    let t = (distance - near) / (far - near);
    let span = (max_iters - 1) as f32;
    span * (1.0 - t) + 0.5
}

/// Returns whether `x` lies at least `SEAM_MARGIN` away from the nearest integer.
fn off_integer_seam(x: f32) -> bool {
    (x - x.round()).abs() >= SEAM_MARGIN
}

/// Iterates the inputs, checking the continuous isovalue against the oracle,
/// and (when off the seam) the two integer counts for exact equality.
fn check_domain(idx: usize, d: &ReconstructDomain, got: &WaterReconstructParams) {
    let want_isovalue = surface_isovalue(d.rest_density, d.iso_fraction);
    assert!(
        close(got.isovalue[idx], want_isovalue),
        "domain {idx} isovalue: gpu {} vs cpu {want_isovalue}",
        got.isovalue[idx]
    );
    if off_integer_seam(band_intermediate(d)) {
        let want_band = narrow_band_width_cells(d.particle_radius, d.dx, d.band_factor);
        assert_eq!(
            got.narrow_band[idx], want_band,
            "domain {idx} narrow_band: gpu {} vs cpu {want_band}",
            got.narrow_band[idx]
        );
    }
    if off_integer_seam(smoothing_intermediate(d)) {
        let want_smoothing = screen_space_smoothing_iterations(
            d.camera_distance,
            d.max_iterations,
            d.near_distance,
            d.far_distance,
        );
        assert_eq!(
            got.smoothing_iterations[idx], want_smoothing,
            "domain {idx} smoothing_iterations: gpu {} vs cpu {want_smoothing}",
            got.smoothing_iterations[idx]
        );
    }
}

/// Dispatches `domains` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterReconstructParams, domains: &[ReconstructDomain]) {
    let got = gpu.evaluate(ctx, domains);
    assert_eq!(got.isovalue.len(), domains.len(), "one isovalue per domain");
    assert_eq!(got.narrow_band.len(), domains.len(), "one band per domain");
    assert_eq!(
        got.smoothing_iterations.len(),
        domains.len(),
        "one smoothing count per domain"
    );
    for (idx, d) in domains.iter().enumerate() {
        check_domain(idx, d, &got);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns empty parameter vectors.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(
        got.isovalue.is_empty(),
        "an empty batch produces no isovalues"
    );
    assert!(
        got.narrow_band.is_empty(),
        "an empty batch produces no bands"
    );
    assert!(
        got.smoothing_iterations.is_empty(),
        "an empty batch produces no smoothing counts"
    );
}

#[test]
fn isovalue_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // Negative rest density folds to zero; the fraction clamps to `0..=1`; a
    // mid fraction rides the linear ramp. The band and smoothing inputs are
    // fixed away from their seams so this test exercises the isovalue path.
    let domains = [
        domain(0.0, 0.5, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0),
        domain(1000.0, -0.25, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0),
        domain(1000.0, 1.75, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0),
        domain(1000.0, 0.25, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0),
        domain(-5.0, -3.0, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0),
    ];
    check(&ctx, &gpu, &domains);
}

#[test]
fn narrow_band_round_up_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // `cells = factor * radius / dx` swept across the round-up ramp (each value
    // kept off an integer seam), the floor-of-one under a tiny radius, a
    // negative radius folding to the one-cell floor, and a zero `dx` falling
    // back to the epsilon cell (kept off the seam).
    let domains = [
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0), // cells 2.6 -> 3
        domain(1000.0, 0.4, 0.27, 0.1, 2.0, 1.0, 8, 1.0, 5.0), // cells 5.4 -> 6
        domain(1000.0, 0.4, 0.01, 0.5, 1.0, 1.0, 8, 1.0, 5.0), // cells 0.02 -> 1
        domain(1000.0, 0.4, -0.5, 0.1, 2.0, 1.0, 8, 1.0, 5.0), // folds -> 1
        domain(1000.0, 0.4, 0.0, 0.1, 2.0, 1.0, 8, 1.0, 5.0),  // zero -> 1
    ];
    check(&ctx, &gpu, &domains);
}

#[test]
fn smoothing_fade_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // Near (at and inside), far (at and beyond) and two mid-ramp distances kept
    // off the rounding seam; plus a zero `max_iterations` floored to one and a
    // degenerate `far <= near` fade.
    let domains = [
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 0.5, 8, 1.0, 5.0), // <= near -> 8
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 1.0, 8, 1.0, 5.0), // == near -> 8
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 2.5, 8, 1.0, 5.0), // mid -> 5
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 3.5, 8, 1.0, 5.0), // mid -> 4
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 5.0, 8, 1.0, 5.0), // == far -> 1
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 9.0, 8, 1.0, 5.0), // >= far -> 1
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 2.5, 0, 1.0, 5.0), // floor to 1
        domain(1000.0, 0.4, 0.13, 0.1, 2.0, 2.5, 8, 5.0, 1.0), // far <= near
    ];
    check(&ctx, &gpu, &domains);
}

#[test]
fn non_multiple_of_workgroup_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // 257 domains exercises the last, partial workgroup (the shader guards
    // `idx >= count`), with inputs on a coarse grid kept off the seams.
    let mut domains = Vec::with_capacity(257);
    for i in 0..257u32 {
        let frac = i as f32 * 0.001_1;
        let radius = 0.13 + i as f32 * 0.013_1;
        let distance = 1.0 + (i % 40) as f32 * 0.101_3;
        domains.push(domain(
            1000.0, frac, radius, 0.1, 2.0, distance, 8, 1.0, 5.0,
        ));
    }
    check(&ctx, &gpu, &domains);
}

#[test]
fn randomized_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterReconstructParams::new(&ctx);
    // A bounded pseudo-random sweep across every input range. The integer
    // counts are asserted only off the seam (see `check_domain`); the isovalue
    // is continuous and always checked.
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut domains = Vec::with_capacity(512);
    for _ in 0..512 {
        let rest_density = uniform(lcg(&mut state), 0.0, 2000.0);
        let iso_fraction = uniform(lcg(&mut state), -0.5, 1.5);
        let particle_radius = uniform(lcg(&mut state), 0.0, 1.0);
        let dx = uniform(lcg(&mut state), 0.01, 0.5);
        let band_factor = uniform(lcg(&mut state), 0.0, 4.0);
        let camera_distance = uniform(lcg(&mut state), 0.0, 20.0);
        let max_iterations = 1 + lcg(&mut state) % 16;
        let near_distance = uniform(lcg(&mut state), 0.0, 5.0);
        let far_distance = near_distance + uniform(lcg(&mut state), 0.0, 20.0);
        domains.push(domain(
            rest_density,
            iso_fraction,
            particle_radius,
            dx,
            band_factor,
            camera_distance,
            max_iterations,
            near_distance,
            far_distance,
        ));
    }
    check(&ctx, &gpu, &domains);
}

/// Builds a reconstruction domain from explicit scalars.
#[expect(
    clippy::too_many_arguments,
    reason = "a domain carries its full input tuple, so the helper mirrors it"
)]
fn domain(
    rest_density: f32,
    iso_fraction: f32,
    particle_radius: f32,
    dx: f32,
    band_factor: f32,
    camera_distance: f32,
    max_iterations: u32,
    near_distance: f32,
    far_distance: f32,
) -> ReconstructDomain {
    ReconstructDomain {
        rest_density,
        iso_fraction,
        particle_radius,
        dx,
        band_factor,
        camera_distance,
        max_iterations,
        near_distance,
        far_distance,
    }
}
