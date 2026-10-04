//! Real-device parity for the quadric optimal-point twin:
//! [`GpuQuadricOptimalPoint`](prism_volumetric_gpu::quadric_optimal_point::GpuQuadricOptimalPoint)
//! must reproduce the `CPU` golden `Quadric::optimal_point` from
//! `prism_render_architecture`, which solves the symmetric `3x3` quadric-error
//! system `A x = -b` by an explicit cofactor (adjugate) inverse and rejects a
//! near-singular matrix.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! operator for operator in the golden's order: the first-row cofactors
//! `c00, c01, c02`, the determinant `det = m00*c00 + m01*c01 + m02*c02`, the
//! singular guard `|det| <= 1e-12`, the remaining symmetric cofactors
//! `c11, c12, c22`, the right-hand side `r = -(ad, bd, cd)`, and the three
//! adjugate dot products scaled by `1/det`. It is written out directly so the
//! test never imports `prism_render_architecture`, `prism_physics_core`, nor
//! `glam`. The constant term `d2` is carried for layout parity and does not
//! participate in the solve.
//!
//! The fixtures cover the branches the kernel must honor: a full-rank diagonal
//! system (three orthogonal planes, `det = 1`, a unique closed-form point); a
//! general well-conditioned symmetric positive-definite system; an all-zero
//! quadric and a rank-deficient quadric (both `det = 0`, `found = 0`); a
//! multi-element batch that validates the `std430` stride by mixing full-rank
//! and singular queries; and a `512`-step sweep over random diagonally
//! dominant symmetric matrices, kept clear of the `|det| = 1e-12` knee by
//! rejection sampling so the discrete `found` decision never flips under `f32`
//! noise. An empty batch is short-circuited on the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The point components are continuous `f32` outputs, so parity uses an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`, with a `1e-6`
//! relative floor so near-zero components compare on the absolute leg). The
//! `found` and `valid` words are discrete and compared exactly; `valid` is
//! always `1` since the layout carries it only for parity with the crate's
//! other twins.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture` 的 `Quadric::optimal_point`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_optimal_point::{
    GpuQuadricOptimalPoint, QuadricOptimalPointQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance leg for the continuous comparisons.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous comparisons.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero components fall back to the absolute
/// leg instead of demanding an impossible relative match.
const REL_FLOOR: f32 = 1.0e-6;

/// Absolute-or-relative closeness for a single `f32` lane.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_EPS * scale
}

/// Independent oracle for one query, returning `(point, found, valid)`.
///
/// Reproduces `Quadric::optimal_point` operator by operator, in the same order
/// as the kernel: the symmetric matrix comes from the upper triangle, the
/// determinant is the explicit first-row cofactor expansion, `|det| <= 1e-12`
/// is rejected as singular, and otherwise the symmetric adjugate multiplies the
/// negated linear term scaled by `1/det`. `valid` is always `1`.
fn oracle(q: &QuadricOptimalPointQuery) -> ([f32; 3], u32, u32) {
    let (m00, m01, m02) = (q.a2, q.ab, q.ac);
    let (m11, m12, m22) = (q.b2, q.bc, q.c2);
    let c00 = m11 * m22 - m12 * m12;
    let c01 = m02 * m12 - m01 * m22;
    let c02 = m01 * m12 - m02 * m11;
    let det = m00 * c00 + m01 * c01 + m02 * c02;
    if det.abs() <= 1.0e-12 {
        return ([0.0, 0.0, 0.0], 0, 1);
    }
    let inv_det = 1.0 / det;
    let c11 = m00 * m22 - m02 * m02;
    let c12 = m02 * m01 - m00 * m12;
    let c22 = m00 * m11 - m01 * m01;
    let (r0, r1, r2) = (-q.ad, -q.bd, -q.cd);
    let x = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det;
    let y = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det;
    let z = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det;
    if x.abs() < 3.0e38 && y.abs() < 3.0e38 && z.abs() < 3.0e38 {
        ([x, y, z], 1, 1)
    } else {
        ([0.0, 0.0, 0.0], 0, 1)
    }
}

/// Asserts one device result matches the oracle, lane by lane.
fn check(
    q: &QuadricOptimalPointQuery,
    r: &prism_volumetric_gpu::quadric_optimal_point::QuadricOptimalPointResult,
) {
    let (point, found, valid) = oracle(q);
    assert_eq!(r.found, found, "found mismatch: query={q:?}");
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.point_x, point[0]) && close(r.point_y, point[1]) && close(r.point_z, point[2]),
        "point mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.point_x,
        r.point_y,
        r.point_z,
        point[0],
        point[1],
        point[2]
    );
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuQuadricOptimalPoint, q: QuadricOptimalPointQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    check(&q, &results[0]);
}

#[test]
fn full_rank_identity_system() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // A = diag(1, 1, 1): three orthogonal unit planes, det = 1 (far from the
    // singular knee). The point is simply -(ad, bd, cd) = (-0.5, 1.25, -2.0).
    assert_parity(
        &ctx,
        &gpu,
        QuadricOptimalPointQuery::new(1.0, 0.0, 0.0, 0.5, 1.0, 0.0, -1.25, 1.0, 2.0, 0.0),
    );
}

#[test]
fn full_rank_scaled_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // A = diag(2, 4, 0.5): still diagonal and well conditioned, det = 4. The
    // point is (-ad/2, -bd/4, -cd/0.5) applied componentwise.
    assert_parity(
        &ctx,
        &gpu,
        QuadricOptimalPointQuery::new(2.0, 0.0, 0.0, -3.0, 4.0, 0.0, 2.0, 0.5, -1.0, 0.0),
    );
}

#[test]
fn full_rank_general_spd() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // A general symmetric positive-definite matrix with off-diagonal coupling:
    // diagonally dominant (det clearly positive), so the full adjugate solve is
    // exercised rather than a diagonal shortcut.
    assert_parity(
        &ctx,
        &gpu,
        QuadricOptimalPointQuery::new(2.0, 0.3, -0.2, 1.5, 1.8, 0.1, -0.75, 2.5, 0.9, 3.0),
    );
}

#[test]
fn all_zero_quadric_is_singular() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // Every coefficient zero: det = 0 exactly, so found = 0 with a zeroed point.
    assert_parity(
        &ctx,
        &gpu,
        QuadricOptimalPointQuery::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    );
}

#[test]
fn rank_deficient_quadric_is_singular() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // Only A00 is non-zero (a single plane direction): det = 0 exactly, so the
    // system is rank-deficient and found = 0. Linear terms are non-zero to prove
    // the reject path still zeroes the point.
    assert_parity(
        &ctx,
        &gpu,
        QuadricOptimalPointQuery::new(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 3.0, 0.0),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent 10-lane query
    // slots and 5-lane result slots must decode independently and in order,
    // mixing a full-rank diagonal, a singular all-zero quadric and a general
    // positive-definite system.
    let full_rank =
        QuadricOptimalPointQuery::new(1.0, 0.0, 0.0, 0.5, 1.0, 0.0, -1.25, 1.0, 2.0, 0.0);
    let singular = QuadricOptimalPointQuery::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
    let general =
        QuadricOptimalPointQuery::new(2.0, 0.3, -0.2, 1.5, 1.8, 0.1, -0.75, 2.5, 0.9, 3.0);
    let queries = [full_rank, singular, general];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
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
    let gpu = GpuQuadricOptimalPoint::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_3C_09);
    // Each query is a diagonally dominant symmetric matrix: off-diagonals in
    // [-0.3, 0.3] and diagonals in [1.0, 2.0] guarantee positive definiteness,
    // so the determinant is clearly positive and well clear of the 1e-12 knee.
    // Rejection sampling additionally requires |det| > 1e-3 so the discrete
    // found decision never flips between the host guard and the device guard.
    const MIN_DET: f32 = 1.0e-3;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let a2 = rng.next_range(1.0, 2.0);
        let b2 = rng.next_range(1.0, 2.0);
        let c2 = rng.next_range(1.0, 2.0);
        let ab = rng.next_range(-0.3, 0.3);
        let ac = rng.next_range(-0.3, 0.3);
        let bc = rng.next_range(-0.3, 0.3);
        let ad = rng.next_range(-2.0, 2.0);
        let bd = rng.next_range(-2.0, 2.0);
        let cd = rng.next_range(-2.0, 2.0);
        let d2 = rng.next_range(-1.0, 1.0);
        // Determinant of the symmetric 3x3 from the upper triangle.
        let c00 = b2 * c2 - bc * bc;
        let c01 = ac * bc - ab * c2;
        let c02 = ab * bc - ac * b2;
        let det = a2 * c00 + ab * c01 + ac * c02;
        if det.abs() <= MIN_DET {
            continue;
        }
        queries.push(QuadricOptimalPointQuery::new(
            a2, ab, ac, ad, b2, bc, bd, c2, cd, d2,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}
