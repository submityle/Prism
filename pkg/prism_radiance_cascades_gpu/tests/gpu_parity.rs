//! Real-device parity tests for the Radiance Cascades GPU twin.
//!
//! Each test acquires a best-effort headless `GPU` via
//! [`GpuContext::try_headless`]. On a machine with a usable adapter (e.g. Apple
//! `M`-series Metal) the kernels run for real and are compared against the
//! device-free CPU golden in
//! [`prism_render_architecture::lighting::radiance_cascades`]; where no adapter
//! exists the test prints a skip note and returns, so CI without a `GPU` stays
//! green.
//!
//! The sampler below is a byte-for-byte CPU twin of the WGSL `sample_interval`
//! in `src/shaders/gather.wgsl`: the exact same pure-rational arithmetic in the
//! exact same left-to-right operation order. Because both sides avoid
//! trigonometry (directions are uploaded) and transcendentals, the only
//! residual divergence is a possible device fused multiply-add, so the
//! comparisons use a small absolute tolerance rather than exact equality.

use glam::{Vec2, Vec3};
use prism_radiance_cascades_gpu::{GpuContext, GpuInterval, GpuRadianceCascades};
use prism_render_architecture::lighting::radiance_cascades::{
    Cascade, CascadeHierarchy, RadianceInterval, SceneSampler, resolve, solve,
};

/// CPU twin of the WGSL `sample_interval` — identical arithmetic and order.
struct RationalMedium;

impl SceneSampler for RationalMedium {
    fn sample_interval(&self, o: Vec2, d: Vec2, t0: f32, t1: f32) -> RadianceInterval {
        let m = o.x * 0.5 + o.y * 0.25 + d.x * 2.0 - d.y * 1.5 + t0 * 0.1;
        let base = m * m;
        let denom = 1.0 + base;
        let tr_raw = 1.0 / denom;
        let tr = tr_raw.clamp(0.0, 1.0);
        let k = (t1 - t0) * 0.05;
        RadianceInterval::new(Vec3::new(m + k, m * 0.5 + d.x, base * 0.25 + k), tr)
    }
}

/// A small hierarchy exercising multiple cascade levels and non-trivial
/// probe/angular dimensions without making the device dispatch expensive.
fn hierarchy() -> CascadeHierarchy {
    CascadeHierarchy {
        origin: Vec2::new(1.0, -2.0),
        base_spacing: 1.5,
        base_cols: 8,
        base_rows: 8,
        base_angular: 4,
        base_interval: 1.0,
        levels: 3,
    }
}

/// Acquires the device or prints a skip note and returns `None`.
#[expect(
    clippy::print_stderr,
    reason = "test diagnostics: explain a GPU-less skip so CI logs are legible"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!(
                "[prism_radiance_cascades_gpu] no headless GPU adapter available; \
                 skipping device parity test"
            );
            None
        }
    }
}

/// Largest component-wise gap between a device interval and the golden one.
fn interval_gap(gpu: GpuInterval, cpu: RadianceInterval) -> f32 {
    let dr = (gpu.radiance[0] - cpu.radiance.x).abs();
    let dg = (gpu.radiance[1] - cpu.radiance.y).abs();
    let db = (gpu.radiance[2] - cpu.radiance.z).abs();
    let dt = (gpu.transmittance - cpu.transmittance).abs();
    dr.max(dg).max(db).max(dt)
}

#[test]
fn gather_matches_golden_per_level() {
    let Some(ctx) = with_gpu() else { return };
    let h = hierarchy();
    let rc = GpuRadianceCascades::new(&ctx);
    for level in 0..h.levels {
        let device = rc.gather(&ctx, &h, level);
        let golden = Cascade::gather(&h, level, &RationalMedium);
        let (cols, _rows) = golden.dims();
        let angular = golden.angular();
        assert_eq!(
            device.len(),
            (h.rays(level)) as usize,
            "device ray count mismatch at level {level}"
        );
        let mut worst = 0.0f32;
        for (i, g) in device.iter().enumerate() {
            let dir = (i as u32) % angular;
            let probe = (i as u32) / angular;
            let col = probe % cols;
            let row = probe / cols;
            worst = worst.max(interval_gap(*g, golden.get(col, row, dir)));
        }
        assert!(
            worst <= 1e-5,
            "level {level} gather diverged from golden by {worst}"
        );
    }
}

#[test]
fn solve_matches_golden_cascade0() {
    let Some(ctx) = with_gpu() else { return };
    let h = hierarchy();
    let rc = GpuRadianceCascades::new(&ctx);
    let device = rc.solve(&ctx, &h);
    let golden = solve(&h, &RationalMedium);
    let (cols, _rows) = golden.dims();
    let angular = golden.angular();
    assert_eq!(device.len(), h.rays(0) as usize);
    let mut worst = 0.0f32;
    for (i, g) in device.iter().enumerate() {
        let dir = (i as u32) % angular;
        let probe = (i as u32) / angular;
        let col = probe % cols;
        let row = probe / cols;
        worst = worst.max(interval_gap(*g, golden.get(col, row, dir)));
    }
    assert!(worst <= 2e-4, "solve diverged from golden by {worst}");
}

#[test]
fn resolve_matches_golden_mean_radiance() {
    let Some(ctx) = with_gpu() else { return };
    let h = hierarchy();
    let rc = GpuRadianceCascades::new(&ctx);
    let device = rc.resolve(&ctx, &h);
    let golden = resolve::mean_radiance(&solve(&h, &RationalMedium));
    assert_eq!(device.len(), golden.len());
    let mut worst = 0.0f32;
    for (g, c) in device.iter().zip(golden.iter()) {
        worst = worst
            .max((g[0] - c.x).abs())
            .max((g[1] - c.y).abs())
            .max((g[2] - c.z).abs());
    }
    assert!(worst <= 2e-4, "resolve diverged from golden by {worst}");
}
