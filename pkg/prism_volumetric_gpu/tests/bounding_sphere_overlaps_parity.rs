//! Real-device parity for the bounding-sphere overlap twin:
//! [`GpuBoundingSphereOverlaps`](prism_volumetric_gpu::bounding_sphere_overlaps::GpuBoundingSphereOverlaps)
//! must reproduce the `CPU` golden `BoundingSphere::overlaps` from
//! `prism_physics_core::collider::bounding_sphere`, which reports whether the
//! squared distance between two sphere centres is at most the squared sum of
//! their radii.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `dist_sq = |center_a - center_b|^2`, `rr = (radius_a + radius_b)^2`, and
//! `overlaps = dist_sq <= rr` — written out directly so the test never imports
//! `prism_physics_core`, `prism_render_architecture`, nor `glam`. Finiteness is
//! an ordered `abs(x) < 3.0e38` magnitude test (rejecting both infinities and
//! `NaN`) mirroring the kernel's guard, so the two never diverge; a non-finite
//! input yields `valid = 0`, `overlaps = 0` and `dist_sq = 0`.
//!
//! The fixtures cover the branches the kernel must honor: clearly overlapping
//! pairs, clearly disjoint pairs, concentric spheres (`dist_sq = 0`), non-finite
//! centre and radius inputs that must reject, a multi-element batch that
//! validates the `std430` stride by mixing overlap, disjoint and degenerate
//! queries, and a `512`-step sweep over random centres and radii, kept clear of
//! the decision knee by rejection sampling so the discrete overlap word never
//! flips under round-off. An empty batch is short-circuited on the host with no
//! dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! `dist_sq` is a continuous `f32` output, so parity uses an absolute-or-relative
//! tolerance (`abs <= 1e-4 || rel <= 1e-3`, with a `1e-6` relative floor so
//! near-zero values compare on the absolute leg). The `overlaps` and `valid`
//! words are discrete and compared exactly; because the overlap decision can
//! flip under round-off when `dist_sq` is within rounding of `rr`, every sweep
//! sample is held clear of that knee by rejection sampling.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_sphere_overlaps::{
    BoundingSphereOverlapsQuery, GpuBoundingSphereOverlaps,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance leg for the continuous comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous comparison.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero values fall back to the absolute leg
/// instead of demanding an impossible relative match.
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

/// Independent oracle for one query, returning `(overlaps, dist_sq, valid)`.
///
/// Reproduces `BoundingSphere::overlaps` operator by operator, in the same order
/// as the kernel: an ordered finiteness test over all eight inputs, then
/// `dist_sq = |center_a - center_b|^2`, `rr = (radius_a + radius_b)^2`, and
/// `overlaps = dist_sq <= rr`. A non-finite input rejects to `(0, 0.0, 0)`.
fn oracle(q: &BoundingSphereOverlapsQuery) -> (u32, f32, u32) {
    let fin = q.ax.abs() < 3.0e38
        && q.ay.abs() < 3.0e38
        && q.az.abs() < 3.0e38
        && q.ra.abs() < 3.0e38
        && q.bx.abs() < 3.0e38
        && q.by.abs() < 3.0e38
        && q.bz.abs() < 3.0e38
        && q.rb.abs() < 3.0e38;
    if !fin {
        return (0, 0.0, 0);
    }
    let dx = q.ax - q.bx;
    let dy = q.ay - q.by;
    let dz = q.az - q.bz;
    let dist_sq = dx * dx + dy * dy + dz * dz;
    let r = q.ra + q.rb;
    let rr = r * r;
    let overlaps = if dist_sq <= rr { 1 } else { 0 };
    (overlaps, dist_sq, 1)
}

/// Asserts one device result matches the oracle, lane by lane.
fn check(
    q: &BoundingSphereOverlapsQuery,
    r: &prism_volumetric_gpu::bounding_sphere_overlaps::BoundingSphereOverlapsResult,
) {
    let (overlaps, dist_sq, valid) = oracle(q);
    assert_eq!(r.overlaps, overlaps, "overlaps mismatch: query={q:?}");
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.dist_sq, dist_sq),
        "dist_sq mismatch: query={q:?} gpu={} oracle={}",
        r.dist_sq,
        dist_sq
    );
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuBoundingSphereOverlaps,
    q: BoundingSphereOverlapsQuery,
) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    check(&q, &results[0]);
}

#[test]
fn concentric_spheres_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // Same centre, dist_sq = 0 <= (0.5 + 0.75)^2: a clear overlap.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(1.0, -2.0, 0.5, 0.5, 1.0, -2.0, 0.5, 0.75),
    );
}

#[test]
fn touching_but_clear_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // Centre distance 2 along +X, radii sum 3: dist_sq = 4 <= 9, well inside.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(0.0, 0.0, 0.0, 1.5, 2.0, 0.0, 0.0, 1.5),
    );
}

#[test]
fn clearly_disjoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // Centre distance 5 along +X, radii sum 1.5: dist_sq = 25 > 2.25, disjoint.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(-2.5, 0.0, 0.0, 0.75, 2.5, 0.0, 0.0, 0.75),
    );
}

#[test]
fn diagonal_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // Off-axis diagonal: delta = (1, 1, 1), dist_sq = 3 <= (1 + 1)^2 = 4.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(-0.5, -0.5, -0.5, 1.0, 0.5, 0.5, 0.5, 1.0),
    );
}

#[test]
fn diagonal_disjoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // Off-axis diagonal: delta = (2, 2, 2), dist_sq = 12 > (1 + 0.5)^2 = 2.25.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.5),
    );
}

#[test]
fn non_finite_centre_rejects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // An infinite centre component rejects: valid = 0, overlaps = 0, dist_sq = 0.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(f32::INFINITY, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    );
}

#[test]
fn nan_radius_rejects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // A NaN radius rejects via the ordered magnitude guard: valid = 0.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(0.0, 0.0, 0.0, f32::NAN, 1.0, 0.0, 0.0, 1.0),
    );
}

#[test]
fn neg_infinite_radius_rejects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // A negative-infinite second radius rejects as non-finite.
    assert_parity(
        &ctx,
        &gpu,
        BoundingSphereOverlapsQuery::new(0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, f32::NEG_INFINITY),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent 4-lane result
    // slots must decode independently and in order, mixing an overlapping pair,
    // a disjoint pair and a degenerate (non-finite) query.
    let overlap = BoundingSphereOverlapsQuery::new(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0);
    let disjoint = BoundingSphereOverlapsQuery::new(-3.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.0, 0.5);
    let degenerate = BoundingSphereOverlapsQuery::new(f32::NAN, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let queries = [overlap, disjoint, degenerate];
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
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
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
    let gpu = GpuBoundingSphereOverlaps::new(&ctx);
    let mut rng = Lcg::new(0x51_7E_90_2B);
    // Each query uses random centres in [-2, 2] and radii in [0.1, 1.5].
    // Rejection sampling keeps every sample clear of the decision knee
    // (|dist_sq - rr| >= 0.05 * max(dist_sq, rr, 1.0)) so the discrete overlap
    // word never flips under f32/f64 round-off divergence.
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let ax = rng.next_range(-2.0, 2.0);
        let ay = rng.next_range(-2.0, 2.0);
        let az = rng.next_range(-2.0, 2.0);
        let ra = rng.next_range(0.1, 1.5);
        let bx = rng.next_range(-2.0, 2.0);
        let by = rng.next_range(-2.0, 2.0);
        let bz = rng.next_range(-2.0, 2.0);
        let rb = rng.next_range(0.1, 1.5);
        let dx = ax - bx;
        let dy = ay - by;
        let dz = az - bz;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = ra + rb;
        let rr = r * r;
        let margin = 0.05 * dist_sq.max(rr).max(1.0);
        if (dist_sq - rr).abs() < margin {
            continue;
        }
        queries.push(BoundingSphereOverlapsQuery::new(
            ax, ay, az, ra, bx, by, bz, rb,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}
