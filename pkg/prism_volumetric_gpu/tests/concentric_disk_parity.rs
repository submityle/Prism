//! Real-device parity for the concentric square-to-disk twin:
//! [`GpuConcentricDisk`](prism_volumetric_gpu::concentric_disk::GpuConcentricDisk)
//! must reproduce the `CPU` golden `concentric_disk` of
//! `prism_render_architecture::reference_pt::concentric`. The Shirley-Chiu map
//! folds the unit square into four wedges and sends concentric square rings to
//! concentric circles, so a uniform `[0, 1]^2` input becomes a uniform-by-area
//! sample on the unit disk while preserving the discrepancy of a low-discrepancy
//! (`QMC`) stream.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `[-1, 1]^2` remap, the centre short-circuit, the wedge selection, and
//! the golden truncated-`Taylor` `sin`/`cos` polynomials evaluated in `f64` —
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`.
//!
//! The fixtures cover the exact centre `(0.5, 0.5)` mapping to the origin, the
//! four square corners mapping to the unit circle's rim, a representative point
//! inside each wedge, a mixed batch of two or more elements that validates the
//! `std430` stride, and an empty batch the host short-circuits with no dispatch.
//! A sweep over random `[0, 1]^2` samples follows, rejecting the diagonal band
//! where `|a| ~= |b|` so a wedge-boundary disagreement cannot masquerade as a
//! parity pass.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The device evaluates the wedge angle's `sin`/`cos` with the native `f32`
//! intrinsics, while the oracle uses the golden `f64` truncated-`Taylor`
//! series; over the `[-pi/4, pi/4]` wedge the two differ by far less than the
//! parity tolerance. Each continuous `(disk_x, disk_y)` is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The map is total, so `valid` is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::concentric`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::concentric_disk::{
    ConcentricDiskQuery, ConcentricDiskResult, GpuConcentricDisk,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// A quarter of pi, matching the golden `core::f64::consts::FRAC_PI_4`.
const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;

/// Evaluates `sin(x)` for `x` in `[-pi/4, pi/4]` with the golden truncated
/// `Taylor` series, Horner-factored exactly as the reference does.
fn sin_quarter(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0
            + x2 * (1.0 / 120.0
                + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0 + x2 * (-1.0 / 39_916_800.0))))))
}

/// Evaluates `cos(x)` for `x` in `[-pi/4, pi/4]` with the golden truncated
/// `Taylor` series, Horner-factored exactly as the reference does.
fn cos_quarter(x: f64) -> f64 {
    let x2 = x * x;
    1.0 + x2
        * (-0.5
            + x2 * (1.0 / 24.0
                + x2 * (-1.0 / 720.0
                    + x2 * (1.0 / 40320.0
                        + x2 * (-1.0 / 3_628_800.0 + x2 * (1.0 / 479_001_600.0))))))
}

/// Independent host oracle: reproduces `concentric_disk` in the golden operator
/// order and `f64` precision, returning the disk coordinates and the (always
/// `1`) validity flag.
fn oracle(q: &ConcentricDiskQuery) -> (f32, f32, u32) {
    let a = 2.0 * f64::from(q.u) - 1.0;
    let b = 2.0 * f64::from(q.v) - 1.0;
    if a * a + b * b <= 0.0 {
        return (0.0, 0.0, 1);
    }
    if a * a > b * b {
        let phi = FRAC_PI_4 * (b / a);
        (
            (a * cos_quarter(phi)) as f32,
            (a * sin_quarter(phi)) as f32,
            1,
        )
    } else {
        let t = FRAC_PI_4 * (a / b);
        ((b * sin_quarter(t)) as f32, (b * cos_quarter(t)) as f32, 1)
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, and the two disk coordinates to tolerance.
fn assert_parity(gpu: &ConcentricDiskResult, q: &ConcentricDiskQuery, label: &str) {
    let (disk_x, disk_y, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    assert!(
        close(gpu.disk_x, disk_x),
        "{label}: disk_x mismatch gpu={} oracle={}",
        gpu.disk_x,
        disk_x
    );
    assert!(
        close(gpu.disk_y, disk_y),
        "{label}: disk_y mismatch gpu={} oracle={}",
        gpu.disk_y,
        disk_y
    );
}

#[test]
fn centre_maps_to_the_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    let q = ConcentricDiskQuery::new(0.5, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(out[0].disk_x.abs() < 1.0e-6, "disk_x={}", out[0].disk_x);
    assert!(out[0].disk_y.abs() < 1.0e-6, "disk_y={}", out[0].disk_y);
    assert_parity(&out[0], &q, "centre");
}

#[test]
fn corners_land_on_the_unit_circle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    let corners = vec![
        ConcentricDiskQuery::new(0.0, 0.0),
        ConcentricDiskQuery::new(1.0, 0.0),
        ConcentricDiskQuery::new(0.0, 1.0),
        ConcentricDiskQuery::new(1.0, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &corners);
    assert_eq!(out.len(), corners.len());
    for (i, (res, q)) in out.iter().zip(corners.iter()).enumerate() {
        let r = (res.disk_x * res.disk_x + res.disk_y * res.disk_y).sqrt();
        assert!((r - 1.0).abs() < 1.0e-5, "corner[{i}] radius={r}");
        assert_parity(res, q, &format!("corner[{i}]"));
    }
}

#[test]
fn horizontal_wedge_representative_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    // u far from 0.5 and v near 0.5 so |a| > |b| (horizontal wedge), away from
    // the diagonal.
    let q = ConcentricDiskQuery::new(0.95, 0.55);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "horizontal");
}

#[test]
fn vertical_wedge_representative_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    // v far from 0.5 and u near 0.5 so |b| > |a| (vertical wedge), away from
    // the diagonal.
    let q = ConcentricDiskQuery::new(0.55, 0.05);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "vertical");
}

#[test]
fn batch_mixes_wedges_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    let queries = vec![
        ConcentricDiskQuery::new(0.9, 0.5),
        ConcentricDiskQuery::new(0.5, 0.1),
        ConcentricDiskQuery::new(0.5, 0.5),
        ConcentricDiskQuery::new(0.2, 0.8),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for res in &out {
        assert_eq!(res.valid, 1);
        let r2 = res.disk_x * res.disk_x + res.disk_y * res.disk_y;
        assert!(r2 <= 1.0 + 1.0e-5, "point escaped the disk: r2={r2}");
    }
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConcentricDisk::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let u = lcg.next_unit();
        let v = lcg.next_unit();
        // Reject the diagonal band where |a| ~= |b|, so a wedge-boundary
        // disagreement cannot pass as a parity match.
        let a = 2.0 * u - 1.0;
        let b = 2.0 * v - 1.0;
        if (a * a - b * b).abs() < 0.05 {
            continue;
        }
        queries.push(ConcentricDiskQuery::new(u, v));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        let r2 = res.disk_x * res.disk_x + res.disk_y * res.disk_y;
        assert!(r2 <= 1.0 + 1.0e-5, "sweep[{i}] escaped the disk: r2={r2}");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
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
}
