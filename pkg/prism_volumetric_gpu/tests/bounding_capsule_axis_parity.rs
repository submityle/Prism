//! Real-device parity for the bounding-capsule axis/height twin:
//! [`GpuBoundingCapsuleAxis`](prism_volumetric_gpu::bounding_capsule_axis::GpuBoundingCapsuleAxis)
//! must reproduce the `CPU` golden `BoundingCapsule::axis` and `::height` from
//! `prism_physics_core::collider::bounding_capsule`, which return the unit axis
//! direction from `center_a` to `center_b` and the distance between the two
//! cap centres.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `delta = center_b - center_a`, `len = length(delta)` as the height, and the
//! axis `delta / len` guarded by `len > 0` (the zero vector otherwise) to match
//! the reference `normalize_or_zero` — written out directly so the test never
//! imports `prism_physics_core`, `prism_render_architecture`, nor `glam`. The
//! reference `normalize_or_zero` normalizes only when the reciprocal length is
//! finite and positive, which for a finite segment is exactly `len > 0`; the
//! oracle mirrors the kernel's ordered guard and guarded divisor so the two
//! never diverge at the collapse point.
//!
//! The fixtures cover the branches the kernel must honor: normal axes along
//! `+X`, `+Y`, `+Z` and an off-axis diagonal; an exactly collapsed capsule
//! (`center_a == center_b`, axis zero and height zero); a very short but
//! non-zero segment that still normalizes; a multi-element batch that validates
//! the `std430` stride by mixing a normal and a degenerate query; and a
//! `512`-step sweep over random cap centres, kept clear of the collapse knee by
//! rejection sampling so the normalize branch stays decisive. An empty batch is
//! short-circuited on the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The axis components and height are continuous `f32` outputs, so parity uses
//! an absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`, with a
//! `1e-6` relative floor so near-zero components compare on the absolute leg).
//! The `valid` word is discrete and compared exactly; it is always `1` since
//! neither accessor has a rejection path.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_capsule`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_capsule_axis::{
    BoundingCapsuleAxisQuery, GpuBoundingCapsuleAxis,
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

/// Independent oracle for one query, returning `(axis, height, valid)`.
///
/// Reproduces `BoundingCapsule::axis` and `::height` operator by operator, in
/// the same order as the kernel: `delta = center_b - center_a`,
/// `len = length(delta)`, and `axis = delta / len` when `len > 0` else the zero
/// vector (matching `normalize_or_zero`). `valid` is always `1`.
fn oracle(q: &BoundingCapsuleAxisQuery) -> ([f32; 3], f32, u32) {
    let dx = q.bx - q.ax;
    let dy = q.by - q.ay;
    let dz = q.bz - q.az;
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        ([dx * inv, dy * inv, dz * inv], len, 1)
    } else {
        ([0.0, 0.0, 0.0], len, 1)
    }
}

/// Asserts one device result matches the oracle, lane by lane.
fn check(
    q: &BoundingCapsuleAxisQuery,
    r: &prism_volumetric_gpu::bounding_capsule_axis::BoundingCapsuleAxisResult,
) {
    let (axis, height, valid) = oracle(q);
    assert!(
        close(r.axis_x, axis[0]) && close(r.axis_y, axis[1]) && close(r.axis_z, axis[2]),
        "axis mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.axis_x,
        r.axis_y,
        r.axis_z,
        axis[0],
        axis[1],
        axis[2]
    );
    assert!(
        close(r.height, height),
        "height mismatch: query={q:?} gpu={} oracle={}",
        r.height,
        height
    );
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuBoundingCapsuleAxis, q: BoundingCapsuleAxisQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    check(&q, &results[0]);
}

#[test]
fn axis_along_plus_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // Segment of length 3 along +X: axis = (1, 0, 0), height = 3.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(-1.0, 0.5, 2.0, 2.0, 0.5, 2.0),
    );
}

#[test]
fn axis_along_plus_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // Segment of length 4 along +Y: axis = (0, 1, 0), height = 4.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(1.0, -2.0, -0.5, 1.0, 2.0, -0.5),
    );
}

#[test]
fn axis_along_plus_z() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // Segment of length 2.5 along +Z: axis = (0, 0, 1), height = 2.5.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(0.25, -1.0, -1.0, 0.25, -1.0, 1.5),
    );
}

#[test]
fn axis_off_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // A non-axis-aligned diagonal exercises all three normalized components.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(-0.5, -1.5, 0.25, 1.5, 2.0, -1.75),
    );
}

#[test]
fn collapsed_capsule_is_zero_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // center_a == center_b: len = 0, axis = (0, 0, 0), height = 0, valid = 1.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(0.75, -0.25, 1.25, 0.75, -0.25, 1.25),
    );
}

#[test]
fn very_short_segment_still_normalizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // A short but non-zero segment (length ~1e-3 along +X): len > 0 so the axis
    // normalizes to (1, 0, 0) despite the tiny magnitude.
    assert_parity(
        &ctx,
        &gpu,
        BoundingCapsuleAxisQuery::new(0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent 5-lane result
    // slots must decode independently and in order, mixing a normal segment, a
    // collapsed segment and an off-axis segment.
    let normal = BoundingCapsuleAxisQuery::new(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let collapsed = BoundingCapsuleAxisQuery::new(2.0, -3.0, 0.5, 2.0, -3.0, 0.5);
    let diagonal = BoundingCapsuleAxisQuery::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let queries = [normal, collapsed, diagonal];
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
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
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
    let gpu = GpuBoundingCapsuleAxis::new(&ctx);
    let mut rng = Lcg::new(0x6F_3A_11_57);
    // Each query uses random cap centres in [-2, 2]. Rejection sampling keeps the
    // segment length at least 1e-3 so the normalize branch is decisive and the
    // CPU/GPU guard (len > 0) never diverges near the collapse knee.
    const MIN_LEN: f32 = 1.0e-3;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let ax = rng.next_range(-2.0, 2.0);
        let ay = rng.next_range(-2.0, 2.0);
        let az = rng.next_range(-2.0, 2.0);
        let bx = rng.next_range(-2.0, 2.0);
        let by = rng.next_range(-2.0, 2.0);
        let bz = rng.next_range(-2.0, 2.0);
        let dx = bx - ax;
        let dy = by - ay;
        let dz = bz - az;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < MIN_LEN {
            continue;
        }
        queries.push(BoundingCapsuleAxisQuery::new(ax, ay, az, bx, by, bz));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}
