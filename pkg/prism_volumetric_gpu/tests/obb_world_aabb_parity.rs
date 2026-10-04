//! Real-device parity for the oriented-box world-`AABB` twin:
//! [`GpuObbWorldAabb`](prism_volumetric_gpu::obb_world_aabb::GpuObbWorldAabb)
//! must reproduce the `CPU` golden `Obb::aabb` of
//! `prism_physics_core::collider::obb`. An oriented bounding box is a center,
//! three orthonormal axes and three half-extents; its tightest world-space
//! axis-aligned bounding box has half-size `r` whose each component is the sum
//! of the absolute axis projections scaled by the matching half-extent, so the
//! box is `(center - r, center + r)`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the component-wise absolute value, the per-axis scaling by `half_extents`
//! (axis `a0` with `hx`, `a1` with `hy`, `a2` with `hz`) accumulated in the
//! same order the kernel uses, and the two corner offsets — written out
//! directly so the test never imports `prism_render_architecture` or
//! `prism_physics_core`. There is no branch and no degenerate path: `valid` is
//! always `1`, so a passing comparison is evidence the ported kernel computes
//! the same support radius the reference does, not merely that the shader
//! compiled.
//!
//! The fixtures cover an axis-aligned box (identity axes collapse the radius to
//! the half-extents), a box rotated 45° about a single axis, an arbitrary
//! orthonormal rotation, a non-zero center translation, a batch of two or more
//! elements that validates the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random orthonormal rotations
//! (built host-side from `std` `cos`/`sin` of random Euler angles) with random
//! positive half-extents and random centers follows. The closed form has no
//! degenerate knee, so no rejection sampling is required; the oracle simply
//! accumulates in the same order the kernel does.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component threads through absolute values, multiplies and adds, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact (a
//! `GPU` may contract a multiply-add). Each continuous output is compared with
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::obb_world_aabb::{
    GpuObbWorldAabb, ObbWorldAabbQuery, ObbWorldAabbResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale_ref = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale_ref <= REL_EPS
}

/// Independent host oracle: reproduces `Obb::aabb` in the same accumulation
/// order the kernel uses (`|a0|*hx + |a1|*hy + |a2|*hz`, axis `a0` then `a1`
/// then `a2`), returning the min corner, max corner and the always-`1` flag.
fn oracle(q: &ObbWorldAabbQuery) -> ([f32; 3], [f32; 3], u32) {
    let c = q.center;
    let a0 = q.axes[0];
    let a1 = q.axes[1];
    let a2 = q.axes[2];
    let hx = q.half_extents[0];
    let hy = q.half_extents[1];
    let hz = q.half_extents[2];
    let r = [
        a0[0].abs() * hx + a1[0].abs() * hy + a2[0].abs() * hz,
        a0[1].abs() * hx + a1[1].abs() * hy + a2[1].abs() * hz,
        a0[2].abs() * hx + a1[2].abs() * hy + a2[2].abs() * hz,
    ];
    let min = [c[0] - r[0], c[1] - r[1], c[2] - r[2]];
    let max = [c[0] + r[0], c[1] + r[1], c[2] + r[2]];
    (min, max, 1)
}

/// Asserts a single `GPU` result matches the independent oracle: the two
/// corners within tolerance and the `valid` flag exactly.
fn assert_parity(gpu: &ObbWorldAabbResult, q: &ObbWorldAabbQuery, label: &str) {
    let (min, max, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    for axis in 0..3 {
        assert!(
            close(gpu.min[axis], min[axis]),
            "{label}: min[{axis}] gpu={} oracle={}",
            gpu.min[axis],
            min[axis]
        );
        assert!(
            close(gpu.max[axis], max[axis]),
            "{label}: max[{axis}] gpu={} oracle={}",
            gpu.max[axis],
            max[axis]
        );
    }
}

/// The three identity (axis-aligned) orthonormal box axes.
fn identity_axes() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Builds the orthonormal column basis of the rotation `Rz(zr)·Ry(yr)·Rx(xr)`.
/// `std` `cos`/`sin` are used only here, host-side, to synthesize valid
/// orthonormal fixtures; the oracle body itself stays pure arithmetic.
fn euler_axes(xr: f32, yr: f32, zr: f32) -> [[f32; 3]; 3] {
    let (sx, cx) = (xr.sin(), xr.cos());
    let (sy, cy) = (yr.sin(), yr.cos());
    let (sz, cz) = (zr.sin(), zr.cos());
    // Rz * Ry * Rx, column-major: each column is one orthonormal axis.
    let col0 = [cz * cy, sz * cy, -sy];
    let col1 = [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx];
    let col2 = [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx];
    [col0, col1, col2]
}

#[test]
fn axis_aligned_collapses_to_half_extents() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let q = ObbWorldAabbQuery::new([0.0, 0.0, 0.0], identity_axes(), [2.0, 3.0, 4.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_parity(&out[0], &q, "axis_aligned");
    // Identity axes: the radius is exactly the half-extents.
    assert!(close(out[0].max[0], 2.0));
    assert!(close(out[0].max[1], 3.0));
    assert!(close(out[0].max[2], 4.0));
}

#[test]
fn rotated_45_about_single_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    // cos(45°) = sin(45°) = sqrt(2)/2, hard-coded so the fixture carries no
    // transcendental dependency.
    let h = 0.707_106_78_f32;
    let axes = [[h, h, 0.0], [-h, h, 0.0], [0.0, 0.0, 1.0]];
    let q = ObbWorldAabbQuery::new([1.0, -1.0, 0.5], axes, [1.0, 1.0, 1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_parity(&out[0], &q, "rotated_45");
}

#[test]
fn arbitrary_orthonormal_rotation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let axes = euler_axes(0.6, -1.2, 2.3);
    let q = ObbWorldAabbQuery::new([2.0, 5.0, -3.0], axes, [1.5, 0.5, 2.5]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_parity(&out[0], &q, "arbitrary_rotation");
}

#[test]
fn non_zero_center_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let center = [10.0, -20.0, 30.0];
    let q = ObbWorldAabbQuery::new(center, identity_axes(), [1.0, 2.0, 3.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_parity(&out[0], &q, "translated");
    // Pure translation of an axis-aligned box: min/max track the center.
    assert!(close(out[0].min[0], 9.0));
    assert!(close(out[0].max[2], 33.0));
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let queries = vec![
        ObbWorldAabbQuery::new([0.0, 0.0, 0.0], identity_axes(), [1.0, 1.0, 1.0]),
        ObbWorldAabbQuery::new([5.0, 6.0, 7.0], euler_axes(0.3, 0.9, -0.5), [2.0, 0.5, 1.0]),
        ObbWorldAabbQuery::new(
            [-4.0, 2.0, 1.0],
            euler_axes(-1.1, 0.2, 1.7),
            [0.5, 3.0, 1.5],
        ),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbWorldAabb::new(&ctx);
    let mut lcg = Lcg::new(0x51ED_600B);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Random orthonormal axes from random Euler angles.
        let xr = lcg.next_range(-3.141_592_7, 3.141_592_7);
        let yr = lcg.next_range(-3.141_592_7, 3.141_592_7);
        let zr = lcg.next_range(-3.141_592_7, 3.141_592_7);
        let axes = euler_axes(xr, yr, zr);
        let center = [
            lcg.next_range(-50.0, 50.0),
            lcg.next_range(-50.0, 50.0),
            lcg.next_range(-50.0, 50.0),
        ];
        // Strictly positive half-extents.
        let he = [
            lcg.next_range(0.05, 10.0),
            lcg.next_range(0.05, 10.0),
            lcg.next_range(0.05, 10.0),
        ];
        queries.push(ObbWorldAabbQuery::new(center, axes, he));
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
