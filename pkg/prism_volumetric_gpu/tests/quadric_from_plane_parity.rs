//! Real-device parity for the plane-quadric twin:
//! [`GpuQuadricFromPlane`](prism_volumetric_gpu::quadric_from_plane::GpuQuadricFromPlane)
//! must reproduce the `CPU` golden `Quadric::from_plane` of
//! `prism_physics_core::collider::quadric`, which packs the plane
//! `n . x + d = 0` into the ten upper-triangular coefficients of its symmetric
//! `4x4` quadric matrix with a fixed sequence of scalar multiplications.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the ten products `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2` evaluated in the
//! golden's order — written out directly in flat `f32` array math so the test
//! never imports `prism_physics_core`, `prism_render_architecture` or `glam`.
//! It mirrors the reference operation for operation.
//!
//! The fixtures cover the regimes the kernel must honor: the three axis-aligned
//! unit normals (checked against literal coefficient values), a tilted unit
//! normal, a plane through the origin (`d = 0`), a negative offset, a
//! multi-element mixed batch that validates the `std430` array stride end to
//! end, plus an empty batch the host short-circuits with no dispatch. A sweep
//! over random unit normals and offsets follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each coefficient is a single multiply, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on each of the
//! ten coefficients; the discrete `valid` flag is compared exactly. The closed
//! form has no degenerate branch, so `valid` is always `1` and no comparison
//! sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::from_plane`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_from_plane::{GpuQuadricFromPlane, QuadricFromPlaneQuery};
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
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent host re-implementation of the golden `Quadric::from_plane`,
/// returning the ten upper-triangular coefficients and the `valid` flag without
/// importing the golden crate or `glam`. The products are evaluated in the same
/// order as the kernel: `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
fn oracle(q: &QuadricFromPlaneQuery) -> ([f32; 10], u32) {
    let a = q.normal[0];
    let b = q.normal[1];
    let c = q.normal[2];
    let d = q.d;
    let coeffs = [
        a * a,
        a * b,
        a * c,
        a * d,
        b * b,
        b * c,
        b * d,
        c * c,
        c * d,
        d * d,
    ];
    (coeffs, 1)
}

/// Dispatches one plane and asserts the `GPU` result matches the oracle on all
/// ten coefficients (within tolerance) and the `valid` flag (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuQuadricFromPlane, q: QuadricFromPlaneQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (coeffs, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
        assert!(
            close(*g, *c),
            "coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
        );
    }
}

#[test]
fn axis_aligned_normals_match_literals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    // The +X plane at offset 2: only a2, ad and d2 are non-zero.
    let qx = QuadricFromPlaneQuery::new([1.0, 0.0, 0.0], 2.0);
    assert_parity(&ctx, &gpu, qx);
    let rx = gpu.evaluate(&ctx, std::slice::from_ref(&qx))[0];
    // a2=1, ab=0, ac=0, ad=2, b2=0, bc=0, bd=0, c2=0, cd=0, d2=4.
    let expected_x = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0];
    for (i, (g, e)) in rx.coeffs.iter().zip(expected_x.iter()).enumerate() {
        assert!(close(*g, *e), "+X coeff {i} mismatch: gpu={g} expected={e}");
    }

    // The +Y and +Z planes, checked through the oracle, exercise the other two
    // axis-aligned normals so a transposed coefficient order would show up.
    assert_parity(
        &ctx,
        &gpu,
        QuadricFromPlaneQuery::new([0.0, 1.0, 0.0], -1.5),
    );
    assert_parity(
        &ctx,
        &gpu,
        QuadricFromPlaneQuery::new([0.0, 0.0, 1.0], 3.25),
    );
}

#[test]
fn tilted_normal_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    // A tilted unit normal with distinct non-zero components, so every one of
    // the ten products is exercised against the oracle.
    let n = unit([0.3, -0.6, 0.74]);
    assert_parity(&ctx, &gpu, QuadricFromPlaneQuery::new(n, 1.7));
}

#[test]
fn plane_through_origin_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    // d = 0: the four coefficients that carry d (ad, bd, cd, d2) must all be 0.
    let n = unit([-0.5, 0.5, 0.70710677]);
    let q = QuadricFromPlaneQuery::new(n, 0.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    for idx in [3usize, 6, 8, 9] {
        assert!(
            close(r.coeffs[idx], 0.0),
            "d=0 coeff {idx} should be zero, got {}",
            r.coeffs[idx]
        );
    }
}

#[test]
fn negative_offset_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    // A negative offset flips the signs of ad, bd, cd while d2 stays positive.
    let n = unit([0.8, 0.1, -0.59]);
    assert_parity(&ctx, &gpu, QuadricFromPlaneQuery::new(n, -4.2));
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    // A multi-element mixed batch (axis-aligned, tilted, origin plane) exercises
    // the std430 array stride: every slot must decode at the right byte offset.
    let queries = [
        QuadricFromPlaneQuery::new([1.0, 0.0, 0.0], 2.0),
        QuadricFromPlaneQuery::new(unit([0.3, -0.6, 0.74]), 1.7),
        QuadricFromPlaneQuery::new(unit([-0.5, 0.5, 0.70710677]), 0.0),
        QuadricFromPlaneQuery::new(unit([0.8, 0.1, -0.59]), -4.2),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (coeffs, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "batch coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromPlane::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// Normalizes a 3-vector into a unit normal, keeping the fixture independent of
/// `glam`.
fn unit(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
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
    let gpu = GpuQuadricFromPlane::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_2E_C4);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Draw a random direction; reject tiny norms so normalization is
        // well-conditioned, then build a unit normal and a random offset.
        let x = rng.next_range(-1.0, 1.0);
        let y = rng.next_range(-1.0, 1.0);
        let z = rng.next_range(-1.0, 1.0);
        if x * x + y * y + z * z <= 0.04 {
            continue;
        }
        let normal = unit([x, y, z]);
        let d = rng.next_range(-5.0, 5.0);
        queries.push(QuadricFromPlaneQuery::new(normal, d));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (coeffs, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "sweep coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}
