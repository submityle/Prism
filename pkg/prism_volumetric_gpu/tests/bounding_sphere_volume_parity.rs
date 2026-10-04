//! Real-device parity for the bounding-sphere volume twin:
//! [`GpuBoundingSphereVolume`](prism_volumetric_gpu::bounding_sphere_volume::GpuBoundingSphereVolume)
//! must reproduce the `CPU` golden `BoundingSphere::volume` of
//! `prism_physics_core::collider::bounding_sphere`. The volume of a sphere of
//! radius `r` is `V = (4/3) * pi * r^3` when the radius is finite and
//! non-negative, and undefined (invalid) otherwise.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and non-negativity guards, then `(4/3) * pi * r * r * r` in
//! the golden operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`, and does not pull in
//! `glam`.
//!
//! The fixtures cover a zero radius (volume `0`), a unit radius, a fractional
//! radius, a large radius, a degenerate negative radius (invalid), non-finite
//! radii (invalid), a batch of two or more elements that mixes valid and invalid
//! radii to validate the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random strictly-positive radii
//! follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through a constant ratio, a constant
//! multiply and three `r` multiplies, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact (a `GPU` may contract a multiply-add).
//! The valid `volume` scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! fixtures keep radii strictly positive and well away from zero/non-finite so
//! the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_sphere_volume::{
    BoundingSphereVolumeQuery, BoundingSphereVolumeResult, GpuBoundingSphereVolume,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `volume` in the golden operator order,
/// returning the volume and the validity flag.
fn oracle(q: &BoundingSphereVolumeQuery) -> (f32, u32) {
    let r = q.radius;
    if !r.is_finite() || r < 0.0 {
        return (0.0, 0);
    }
    ((4.0 / 3.0) * std::f32::consts::PI * r * r * r, 1)
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
/// `valid` flag exactly, and the `volume` scalar to tolerance when valid.
fn assert_parity(gpu: &BoundingSphereVolumeResult, q: &BoundingSphereVolumeQuery, label: &str) {
    let (volume, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.volume, volume),
            "{label}: volume mismatch gpu={} oracle={}",
            gpu.volume,
            volume
        );
    }
}

#[test]
fn zero_radius_has_zero_volume() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let q = BoundingSphereVolumeQuery::new(0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].volume, 0.0));
    assert_parity(&out[0], &q, "zero");
}

#[test]
fn unit_radius_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let q = BoundingSphereVolumeQuery::new(1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // (4/3)*pi*1 = 4.18879...
    assert!(close(out[0].volume, (4.0 / 3.0) * std::f32::consts::PI));
    assert_parity(&out[0], &q, "unit");
}

#[test]
fn fractional_radius_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let q = BoundingSphereVolumeQuery::new(2.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "fractional");
}

#[test]
fn large_radius_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let q = BoundingSphereVolumeQuery::new(1.0e3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "large");
}

#[test]
fn negative_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let q = BoundingSphereVolumeQuery::new(-2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].volume, 0.0);
    assert_parity(&out[0], &q, "negative");
}

#[test]
fn non_finite_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let infinite = BoundingSphereVolumeQuery::new(f32::INFINITY);
    let not_a_number = BoundingSphereVolumeQuery::new(f32::NAN);
    let out = gpu.evaluate(&ctx, &[infinite, not_a_number]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "infinite radius should be invalid");
    assert_eq!(out[1].valid, 0, "NaN radius should be invalid");
    assert_eq!(out[0].volume, 0.0);
    assert_eq!(out[1].volume, 0.0);
    assert_parity(&out[0], &infinite, "infinite");
    assert_parity(&out[1], &not_a_number, "nan");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let queries = vec![
        BoundingSphereVolumeQuery::new(1.0),
        BoundingSphereVolumeQuery::new(-1.0),
        BoundingSphereVolumeQuery::new(4.0),
        BoundingSphereVolumeQuery::new(f32::NAN),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereVolume::new(&ctx);
    let mut lcg = Lcg::new(0x0B0D_15E7);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-positive radii well away from zero and non-finite so
        // validity is stable.
        let radius = lcg.next_range(0.01, 50.0);
        queries.push(BoundingSphereVolumeQuery::new(radius));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
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

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
