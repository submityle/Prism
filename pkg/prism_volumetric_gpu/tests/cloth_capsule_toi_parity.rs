//! Real-device parity for the cloth capsule-TOI twin:
//! [`GpuClothCapsuleToi`](prism_volumetric_gpu::cloth_capsule_toi::GpuClothCapsuleToi)
//! must reproduce the `CPU` golden `capsule_toi` of
//! `prism_physics_core::soft::collision::ccd`, which sweeps a particle segment
//! `prev -> curr` against a capsule (segment `p0`..`p1` inflated by `radius`)
//! and returns the earliest time of impact in `0..=1`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `first_entry_time` (the earliest root of `a t^2 + b t + c <= 0` with its
//! `c <= 0` already-inside, `a <= EPS_COEF` linear and quadratic branches),
//! `sphere_toi`, `cylinder_slab_toi` (the radial sub-`radius` interval
//! intersected with the axial slab interval and with `0..=1`), `earliest`, and
//! `capsule_toi` (non-positive radius inert, collapsed capsule degenerating to
//! a sphere, otherwise the earliest of the cylinder slab and the two end-cap
//! spheres) — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`. It mirrors the
//! reference branch for branch, so a passing comparison is evidence the ported
//! kernel took the same quadratic / interval-intersection branch, not merely
//! that the shader compiled.
//!
//! The reference expresses the infinite radial/axial interval bounds with
//! `f32::INFINITY`; the oracle uses the same true infinities, while the kernel
//! substitutes large finite sentinels that can never win the `max(.., 0.0)` /
//! `min(.., 1.0)` clamp, so both collapse to the identical returned time.
//!
//! The fixtures cover each branch the kernel must honor: a non-positive radius
//! (inert, miss), a collapsed capsule (`p0 == p1`, sphere fallback), a
//! cylindrical-side hit, an end-cap sphere hit, a clean miss, a point starting
//! inside (`t == 0`), a batch of two or more elements that validates the
//! `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random geometry follows, rejecting marginal
//! configurations (where a tiny perturbation would flip the hit flag or shift
//! the time across a knee) so the branch choice agrees on both sides despite
//! any last-bit difference.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every path threads through multiplies, adds, guarded divisions and a
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The continuous time `t` is
//! compared with `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `hit` and `valid` flags are compared exactly. Because the
//! closed form always classifies, `valid` is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::ccd`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_capsule_toi::{ClothCapsuleToiQuery, GpuClothCapsuleToi};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Numerical floor for treating a scalar coefficient as zero, matching the
/// reference `EPS_COEF`.
const EPS_COEF: f32 = 1.0e-12;
/// Squared-length floor below which the capsule axis is collapsed, matching the
/// reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Vector difference `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales a vector.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Independent host re-implementation of the reference `first_entry_time`: the
/// earliest root in `0..=1` of `a t^2 + b t + c <= 0` for a non-negative
/// leading coefficient, or [`None`]. Replicated branch for branch.
fn first_entry_time(a: f32, b: f32, c: f32) -> Option<f32> {
    if c <= 0.0 {
        return Some(0.0);
    }
    if a <= EPS_COEF {
        // Linear: b*t + c <= 0. With c > 0 this needs b < 0.
        if b >= -EPS_COEF {
            return None;
        }
        let t = -c / b;
        return if t <= 1.0 { Some(t.max(0.0)) } else { None };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    // With c > 0 and a > 0 the earlier root is the entry into the region.
    let t = ((-b) - root) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// Independent host re-implementation of the reference `sphere_toi`.
fn sphere_toi(prev: [f32; 3], curr: [f32; 3], center: [f32; 3], radius: f32) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let m = sub(curr, prev);
    let e = sub(prev, center);
    let a = dot(m, m);
    let b = 2.0 * dot(e, m);
    let c = dot(e, e) - radius * radius;
    first_entry_time(a, b, c)
}

/// Independent host re-implementation of the reference `cylinder_slab_toi`: the
/// radial sub-`radius` interval intersected with the axial slab `0..=len` and
/// with `0..=1`, returning the lower bound. The infinite interval bounds use
/// true `f32` infinities to match the golden exactly.
fn cylinder_slab_toi(
    prev: [f32; 3],
    curr: [f32; 3],
    p0: [f32; 3],
    axis: [f32; 3],
    radius: f32,
) -> Option<f32> {
    let len = dot(axis, axis).sqrt();
    if len <= EPS_COEF {
        return None;
    }
    let u = scale(axis, 1.0 / len);
    let e0 = sub(prev, p0);
    let m = sub(curr, prev);
    let mu = dot(m, u);
    let e0u = dot(e0, u);

    // Radial interval [rad_lo, rad_hi] where perpendicular distance <= radius.
    let a = dot(m, m) - mu * mu;
    let b = 2.0 * (dot(e0, m) - e0u * mu);
    let c = dot(e0, e0) - e0u * e0u - radius * radius;
    let (rad_lo, rad_hi) = if a > EPS_COEF {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let root = disc.sqrt();
        (((-b) - root) / (2.0 * a), ((-b) + root) / (2.0 * a))
    } else if c <= 0.0 {
        (f32::NEG_INFINITY, f32::INFINITY)
    } else {
        return None;
    };

    // Axial interval [ax_lo, ax_hi] where the projection lies in [0, len].
    let (ax_lo, ax_hi) = if mu.abs() > EPS_COEF {
        let t_at_zero = -e0u / mu;
        let t_at_len = (len - e0u) / mu;
        (t_at_zero.min(t_at_len), t_at_zero.max(t_at_len))
    } else if (0.0..=len).contains(&e0u) {
        (f32::NEG_INFINITY, f32::INFINITY)
    } else {
        return None;
    };

    let lo = rad_lo.max(ax_lo).max(0.0);
    let hi = rad_hi.min(ax_hi).min(1.0);
    if lo <= hi {
        Some(lo)
    } else {
        None
    }
}

/// Independent host re-implementation of the reference `earliest`.
fn earliest(lhs: Option<f32>, rhs: Option<f32>) -> Option<f32> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, rhs) => rhs,
    }
}

/// Independent host re-implementation of the reference `capsule_toi`.
fn capsule_toi(
    prev: [f32; 3],
    curr: [f32; 3],
    p0: [f32; 3],
    p1: [f32; 3],
    radius: f32,
) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let axis = sub(p1, p0);
    let len_sq = dot(axis, axis);
    if len_sq <= EPS_LEN_SQ {
        return sphere_toi(prev, curr, p0, radius);
    }
    let mut best = cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = earliest(best, sphere_toi(prev, curr, p0, radius));
    best = earliest(best, sphere_toi(prev, curr, p1, radius));
    best
}

/// The full host oracle for one query: `(hit, t, valid)`. The closed form
/// always classifies, so `valid` is always `1`.
fn oracle(q: &ClothCapsuleToiQuery) -> (u32, f32, u32) {
    let prev = [q.prev_x, q.prev_y, q.prev_z];
    let curr = [q.curr_x, q.curr_y, q.curr_z];
    let p0 = [q.p0_x, q.p0_y, q.p0_z];
    let p1 = [q.p1_x, q.p1_y, q.p1_z];
    match capsule_toi(prev, curr, p0, p1, q.radius) {
        Some(t) => (1, t, 1),
        None => (0, 0.0, 1),
    }
}

/// Dispatches one query and asserts the hit flag, the continuous time and the
/// validity flag against the independent oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuClothCapsuleToi, q: ClothCapsuleToiQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (hit, t, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.hit, hit, "hit flag mismatch: query={q:?}");
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    if hit == 1 {
        assert!(
            close(r.t, t),
            "time mismatch: gpu={} cpu={t} query={q:?}",
            r.t
        );
    }
}

#[test]
fn non_positive_radius_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // A zero or negative radius makes the capsule inert: always a miss, even
    // when the segment passes straight through the axis.
    for radius in [0.0_f32, -0.5] {
        let q = ClothCapsuleToiQuery::new(
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            radius,
        );
        let (hit, _, _) = oracle(&q);
        assert_eq!(hit, 0, "non-positive radius must miss on the oracle");
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn collapsed_capsule_behaves_as_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // p0 == p1 collapses the axis below EPS_LEN_SQ, so the capsule degenerates
    // to a single sphere at p0; the swept point crosses it.
    let q = ClothCapsuleToiQuery::new(
        [0.0, 3.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        1.0,
    );
    let (hit, _, _) = oracle(&q);
    assert_eq!(hit, 1, "collapsed capsule sphere must be hit on the oracle");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn cylindrical_side_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // Segment crosses the cylinder wall squarely inside the axis slab: the
    // cylinder branch wins with an interior time.
    let q = ClothCapsuleToiQuery::new(
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    let (hit, t, _) = oracle(&q);
    assert_eq!(hit, 1, "cylindrical side must be hit on the oracle");
    assert!(t > 0.0 && t < 1.0, "expected an interior time, got {t}");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn end_cap_sphere_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // Segment grazes past the p1 end cap beyond the axis slab, so the cylinder
    // branch misses and an end-cap sphere supplies the earliest time.
    let q = ClothCapsuleToiQuery::new(
        [1.4, 1.0, 0.0],
        [1.4, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    let (hit, t, _) = oracle(&q);
    assert_eq!(hit, 1, "end-cap sphere must be hit on the oracle");
    assert!(t > 0.0 && t < 1.0, "expected an interior time, got {t}");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn clean_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // Segment stays well above the capsule for its whole length: a clean miss.
    let q = ClothCapsuleToiQuery::new(
        [-3.0, 2.0, 0.0],
        [3.0, 2.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    let (hit, _, _) = oracle(&q);
    assert_eq!(hit, 0, "distant segment must miss on the oracle");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn start_inside_reports_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // The segment starts on the axis (inside the capsule), so the entry time is
    // exactly zero.
    let q = ClothCapsuleToiQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    let (hit, t, _) = oracle(&q);
    assert_eq!(hit, 1, "start-inside must be a hit on the oracle");
    assert!(t == 0.0, "start-inside must report t == 0, got {t}");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    // A multi-element batch catches any std430 stride mismatch between the host
    // query struct and the shader layout.
    let queries = [
        ClothCapsuleToiQuery::new(
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        ClothCapsuleToiQuery::new(
            [1.4, 1.0, 0.0],
            [1.4, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        ClothCapsuleToiQuery::new(
            [-3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        ClothCapsuleToiQuery::new(
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
        ),
        ClothCapsuleToiQuery::new(
            [0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let (hit, t, valid) = oracle(q);
        assert_eq!(r.hit, hit, "batch hit mismatch: query={q:?}");
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        if hit == 1 {
            assert!(
                close(r.t, t),
                "batch time mismatch: gpu={} cpu={t} query={q:?}",
                r.t
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
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

/// Perturbs a query slightly for conditioning checks: nudges each geometric
/// coordinate by `delta` and the radius by a small matching amount.
fn perturb(q: &ClothCapsuleToiQuery, delta: f32) -> ClothCapsuleToiQuery {
    ClothCapsuleToiQuery::new(
        [q.prev_x + delta, q.prev_y - delta, q.prev_z + delta],
        [q.curr_x - delta, q.curr_y + delta, q.curr_z - delta],
        [q.p0_x + delta, q.p0_y + delta, q.p0_z - delta],
        [q.p1_x - delta, q.p1_y - delta, q.p1_z + delta],
        q.radius + delta,
    )
}

/// Rejects marginal configurations where a tiny perturbation flips the hit flag
/// or shifts the impact time across a knee, so the branch choice (quadratic
/// discriminant sign, slab endpoint, `t == 0`/`t == 1` clamp) agrees on both
/// sides despite any last-bit difference between host and device.
fn well_conditioned(q: &ClothCapsuleToiQuery) -> bool {
    let (base_hit, base_t, _) = oracle(q);
    for &delta in &[1.0e-3_f32, -1.0e-3] {
        let (hit, t, _) = oracle(&perturb(q, delta));
        if hit != base_hit {
            return false;
        }
        if base_hit == 1 && (t - base_t).abs() > 2.0e-2 {
            return false;
        }
    }
    true
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCapsuleToi::new(&ctx);
    let mut rng = Lcg::new(0x2B_17_9C_41);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let prev = [
            rng.next_range(-2.5, 2.5),
            rng.next_range(-2.5, 2.5),
            rng.next_range(-2.5, 2.5),
        ];
        let curr = [
            rng.next_range(-2.5, 2.5),
            rng.next_range(-2.5, 2.5),
            rng.next_range(-2.5, 2.5),
        ];
        let p0 = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        // Build p1 from a non-degenerate axis so len_sq stays well above
        // EPS_LEN_SQ and the sweep exercises the cylinder path, not the
        // collapsed-capsule corner (covered by a named fixture).
        let axis = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        if dot(axis, axis) < 0.25 {
            continue;
        }
        let p1 = [p0[0] + axis[0], p0[1] + axis[1], p0[2] + axis[2]];
        let radius = rng.next_range(0.2, 1.0);
        let q = ClothCapsuleToiQuery::new(prev, curr, p0, p1, radius);
        // Reject marginal geometry so a last-bit host/device difference cannot
        // flip the branch choice.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    let mut hits = 0u32;
    for (q, r) in queries.iter().zip(results.iter()) {
        let (hit, t, valid) = oracle(q);
        assert_eq!(r.hit, hit, "sweep hit mismatch: query={q:?}");
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        if hit == 1 {
            hits += 1;
            assert!(
                close(r.t, t),
                "sweep time mismatch: gpu={} cpu={t} query={q:?}",
                r.t
            );
        }
    }
    // Guard against a degenerate all-miss sweep that would weakly test parity.
    assert!(hits > 0, "sweep produced no hits; geometry is too sparse");
}
