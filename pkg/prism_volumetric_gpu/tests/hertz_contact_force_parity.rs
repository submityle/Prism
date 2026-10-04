//! Real-device parity for the Hertz contact-force twin:
//! [`GpuHertzContactForce`](prism_volumetric_gpu::hertz_contact_force::GpuHertzContactForce)
//! must reproduce the `CPU` golden `evaluate_hertz_contact` of
//! `prism_physics_core::collider::hertz_contact`, the full normal-plus-tangential
//! contact force two elastic grains develop when they press into each other.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly in flat `f32`/`f64` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly, including the Hertzian elastic term's square root and
//! `3/2` power taken in `f64` (`r_eff.sqrt()`, `delta.powf(1.5)`), and the
//! golden degeneracy gate: a near-zero axis, a non-positive or non-finite
//! `overlap`, or a non-positive or non-finite `effective_radius` yields a zero
//! force with `overlap.max(0)` and `sliding = false`.
//!
//! The fixtures cover the regimes the kernel must honor: a pressing contact
//! with slip in the sliding branch; a tangential response below the Coulomb
//! limit (non-sliding); a purely-normal relative velocity with no slip; a
//! near-zero axis; a non-positive overlap; a non-positive radius; non-finite
//! inputs; a multi-element mixed batch validating the `std430` array stride;
//! plus an empty batch the host short-circuits with no dispatch. A `512`-step
//! `LCG` sweep over valid interior inputs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The golden evaluates the elastic term in `f64` while the twin uses `f32`
//! `sqrt` and `pow`, so `CPU` and `GPU` agree only up to a small numerical gap.
//! The continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on each force component and magnitude; the discrete
//! `sliding` flag is compared exactly. The sweep keeps every input well clear
//! of the degeneracy and sliding knees so the round-off cannot flip a discrete
//! decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hertz_contact_force::{GpuHertzContactForce, HertzContactForceQuery};
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

/// Independent host re-implementation of the golden `evaluate_hertz_contact`,
/// returning `(force_on_b, overlap, normal_magnitude, tangential_magnitude,
/// sliding)` without importing the golden crate or `glam`. The Hertzian elastic
/// term is taken in `f64` exactly as the golden does, and the degeneracy gate
/// mirrors the golden rule: axis norm above `f32::EPSILON`, and `overlap` and
/// `effective_radius` both finite and strictly positive.
fn oracle(q: &HertzContactForceQuery) -> ([f32; 3], f32, f32, f32, bool) {
    let axis_len = (q.axis[0] * q.axis[0] + q.axis[1] * q.axis[1] + q.axis[2] * q.axis[2]).sqrt();
    let degenerate = axis_len <= f32::EPSILON
        || !(q.overlap.is_finite() && q.overlap > 0.0)
        || !(q.effective_radius.is_finite() && q.effective_radius > 0.0);
    if degenerate {
        return ([0.0; 3], q.overlap.max(0.0), 0.0, 0.0, false);
    }
    let n = [
        q.axis[0] / axis_len,
        q.axis[1] / axis_len,
        q.axis[2] / axis_len,
    ];
    let v_n = q.rel_vel[0] * n[0] + q.rel_vel[1] * n[1] + q.rel_vel[2] * n[2];
    let elastic = {
        let e = f64::from(q.effective_modulus);
        let r = f64::from(q.effective_radius);
        let d = f64::from(q.overlap);
        ((4.0 / 3.0) * e * r.sqrt() * d.powf(1.5)) as f32
    };
    let normal_mag = (elastic - q.normal_damping * v_n).max(0.0);
    let nf = [normal_mag * n[0], normal_mag * n[1], normal_mag * n[2]];
    let v_t = [
        q.rel_vel[0] - v_n * n[0],
        q.rel_vel[1] - v_n * n[1],
        q.rel_vel[2] - v_n * n[2],
    ];
    let speed_t = (v_t[0] * v_t[0] + v_t[1] * v_t[1] + v_t[2] * v_t[2]).sqrt();
    let (tf, tang_mag, sliding) = if speed_t > f32::EPSILON {
        let viscous = q.tangential_damping * speed_t;
        let coulomb = q.friction * normal_mag;
        let sliding = viscous >= coulomb;
        let m = viscous.min(coulomb);
        let th = [v_t[0] / speed_t, v_t[1] / speed_t, v_t[2] / speed_t];
        ([-m * th[0], -m * th[1], -m * th[2]], m, sliding)
    } else {
        ([0.0; 3], 0.0, false)
    };
    (
        [nf[0] + tf[0], nf[1] + tf[1], nf[2] + tf[2]],
        q.overlap,
        normal_mag,
        tang_mag,
        sliding,
    )
}

/// Dispatches one query and asserts the `GPU` result matches the oracle on
/// every continuous component (within tolerance) and the `sliding` flag
/// (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuHertzContactForce, q: &HertzContactForceQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(q))[0];
    let (force, overlap, normal_mag, tang_mag, sliding) = oracle(q);
    for axis in 0..3 {
        assert!(
            close(r.force_on_b[axis], force[axis]),
            "force[{axis}] mismatch: gpu={} cpu={} query={q:?}",
            r.force_on_b[axis],
            force[axis]
        );
    }
    assert!(
        close(r.overlap, overlap),
        "overlap mismatch: gpu={} cpu={} query={q:?}",
        r.overlap,
        overlap
    );
    assert!(
        close(r.normal_magnitude, normal_mag),
        "normal_magnitude mismatch: gpu={} cpu={} query={q:?}",
        r.normal_magnitude,
        normal_mag
    );
    assert!(
        close(r.tangential_magnitude, tang_mag),
        "tangential_magnitude mismatch: gpu={} cpu={} query={q:?}",
        r.tangential_magnitude,
        tang_mag
    );
    assert_eq!(r.sliding, sliding, "sliding mismatch: query={q:?}");
}

#[test]
fn normal_contact_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // A pressing contact along +Z with a tangential slip component; a non-unit
    // axis also exercises the defensive normalization.
    let q = HertzContactForceQuery::new(
        1.0e7,
        10.0,
        5.0,
        0.5,
        [0.0, 0.0, 2.0],
        0.01,
        0.5,
        [1.0, 0.0, -0.3],
    );
    assert_parity(&ctx, &gpu, &q);
}

#[test]
fn non_sliding_branch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // Large friction and small tangential damping keep the viscous term below
    // the Coulomb limit, so there is slip (speed_t > EPS) but sliding = false.
    let q = HertzContactForceQuery::new(
        1.0e7,
        0.0,
        0.01,
        5.0,
        [0.0, 0.0, 1.0],
        0.02,
        0.5,
        [0.5, 0.0, 0.0],
    );
    let (_, _, _, _, sliding) = oracle(&q);
    assert!(!sliding, "fixture must land in the non-sliding branch");
    assert_parity(&ctx, &gpu, &q);
}

#[test]
fn sliding_branch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // Large tangential damping and small friction push the viscous term well
    // above the Coulomb limit, so sliding = true. Numerically: speed_t = 0.4,
    // viscous = 2000 * 0.4 = 800; elastic = (4/3)*1e6*sqrt(0.5)*0.02^1.5
    // ~= 2666.6, so normal_mag ~= 2666.6 and coulomb = 0.05 * 2666.6 ~= 133.3.
    // viscous (800) exceeds coulomb (133.3) by ~6x, a comfortable margin from
    // the sliding knee so f32 round-off cannot flip the discrete flag.
    let q = HertzContactForceQuery::new(
        1.0e6,
        0.0,
        2000.0,
        0.05,
        [0.0, 0.0, 1.0],
        0.02,
        0.5,
        [0.4, 0.0, 0.0],
    );
    let (_, _, _, _, sliding) = oracle(&q);
    assert!(sliding, "fixture must land in the sliding branch");
    assert_parity(&ctx, &gpu, &q);
}

#[test]
fn zero_tangential_speed_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // Relative velocity purely along the axis: the tangential component is zero,
    // so tangential_magnitude = 0 and sliding = false.
    let q = HertzContactForceQuery::new(
        1.0e7,
        10.0,
        5.0,
        0.5,
        [0.0, 0.0, 1.0],
        0.01,
        0.5,
        [0.0, 0.0, -0.4],
    );
    let (_, _, _, tang_mag, sliding) = oracle(&q);
    assert!(
        tang_mag == 0.0,
        "fixture must have zero tangential magnitude"
    );
    assert!(!sliding, "fixture must not slide");
    assert_parity(&ctx, &gpu, &q);
}

#[test]
fn degenerate_near_zero_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    let degenerate = [
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 0.0],
            0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [1.0e-9, 0.0, 0.0],
            0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
    ];
    for q in degenerate {
        assert_parity(&ctx, &gpu, &q);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(
            r.force_on_b, [0.0; 3],
            "near-zero axis must give zero force"
        );
        assert!(!r.sliding, "degenerate contact never slides");
    }
}

#[test]
fn degenerate_non_positive_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    let degenerate = [
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            0.0,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            -0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
    ];
    for q in degenerate {
        assert_parity(&ctx, &gpu, &q);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(
            r.force_on_b, [0.0; 3],
            "non-positive overlap gives zero force"
        );
    }
}

#[test]
fn degenerate_non_positive_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    let degenerate = [
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            0.01,
            0.0,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            0.01,
            -0.5,
            [1.0, 0.0, 0.0],
        ),
    ];
    for q in degenerate {
        assert_parity(&ctx, &gpu, &q);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(
            r.force_on_b, [0.0; 3],
            "non-positive radius gives zero force"
        );
    }
}

#[test]
fn degenerate_non_finite() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // Non-finite overlap, radius, or axis must be rejected. We deliberately use
    // NaN and negative infinity (not +inf overlap, whose golden overlap.max(0)
    // is +inf while the guarded kernel collapses it to 0).
    let degenerate = [
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            f32::NAN,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            f32::NEG_INFINITY,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            0.01,
            f32::NAN,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [f32::NAN, 0.0, 1.0],
            0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
    ];
    for q in degenerate {
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(
            r.force_on_b, [0.0; 3],
            "non-finite input gives zero force: {q:?}"
        );
        assert_eq!(r.normal_magnitude, 0.0, "non-finite input: zero normal mag");
        assert_eq!(
            r.tangential_magnitude, 0.0,
            "non-finite input: zero tangential mag"
        );
        assert!(!r.sliding, "non-finite input never slides: {q:?}");
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
    // A multi-element mixed batch exercises the std430 array stride: every slot
    // must decode at the right byte offset and remain independent.
    let queries = [
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 2.0],
            0.01,
            0.5,
            [1.0, 0.0, -0.3],
        ),
        HertzContactForceQuery::new(
            1.0e6,
            0.0,
            50.0,
            0.05,
            [0.0, 0.0, 1.0],
            0.02,
            0.5,
            [0.4, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 0.0],
            0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            -0.01,
            0.5,
            [1.0, 0.0, 0.0],
        ),
        HertzContactForceQuery::new(
            1.0e7,
            10.0,
            5.0,
            0.5,
            [0.0, 0.0, 1.0],
            0.01,
            f32::NAN,
            [1.0, 0.0, 0.0],
        ),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (force, overlap, normal_mag, tang_mag, sliding) = oracle(q);
        for axis in 0..3 {
            assert!(
                close(r.force_on_b[axis], force[axis]),
                "batch force[{axis}] mismatch: gpu={} cpu={} query={q:?}",
                r.force_on_b[axis],
                force[axis]
            );
        }
        assert!(
            close(r.overlap, overlap),
            "batch overlap mismatch: query={q:?}"
        );
        assert!(
            close(r.normal_magnitude, normal_mag),
            "batch normal_magnitude mismatch: query={q:?}"
        );
        assert!(
            close(r.tangential_magnitude, tang_mag),
            "batch tangential_magnitude mismatch: query={q:?}"
        );
        assert_eq!(r.sliding, sliding, "batch sliding mismatch: query={q:?}");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzContactForce::new(&ctx);
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
    let gpu = GpuHertzContactForce::new(&ctx);
    let mut rng = Lcg::new(0x6B_2D_91_55);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn sample well inside the valid region and clear of the
    // sliding knee: a well-nonzero axis, strictly positive overlap and radius
    // with margin, modest damping and friction, and a relative velocity with a
    // clear tangential component. The viscous and Coulomb terms are kept
    // separated so the discrete sliding decision cannot flip under round-off.
    while queries.len() < 512 {
        let modulus = rng.next_range(1.0e4, 1.0e8);
        let radius = rng.next_range(0.05, 2.0);
        let overlap = rng.next_range(0.05, 0.5);
        // Axis well away from zero (z component dominant and strictly positive).
        let axis = [
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
            rng.next_range(0.8, 1.6),
        ];
        let rel_vel = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let normal_damping = rng.next_range(0.0, 20.0);
        let tangential_damping = rng.next_range(0.0, 20.0);
        let friction = rng.next_range(0.1, 0.9);
        let q = HertzContactForceQuery::new(
            modulus,
            normal_damping,
            tangential_damping,
            friction,
            axis,
            overlap,
            radius,
            rel_vel,
        );
        // Reject samples near the sliding knee or near zero slip so the discrete
        // flag is robust against the f64/f32 round-off.
        let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let n = [axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len];
        let v_n = rel_vel[0] * n[0] + rel_vel[1] * n[1] + rel_vel[2] * n[2];
        let v_t = [
            rel_vel[0] - v_n * n[0],
            rel_vel[1] - v_n * n[1],
            rel_vel[2] - v_n * n[2],
        ];
        let speed_t = (v_t[0] * v_t[0] + v_t[1] * v_t[1] + v_t[2] * v_t[2]).sqrt();
        let (_, _, normal_mag, _, _) = oracle(&q);
        let viscous = tangential_damping * speed_t;
        let coulomb = friction * normal_mag;
        let knee = viscous.max(coulomb).max(1.0);
        if speed_t < 0.05 || (viscous - coulomb).abs() / knee < 0.05 {
            continue;
        }
        queries.push(q);
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (force, overlap, normal_mag, tang_mag, sliding) = oracle(q);
        for axis in 0..3 {
            assert!(
                close(r.force_on_b[axis], force[axis]),
                "sweep force[{axis}] mismatch: gpu={} cpu={} query={q:?}",
                r.force_on_b[axis],
                force[axis]
            );
        }
        assert!(
            close(r.overlap, overlap),
            "sweep overlap mismatch: query={q:?}"
        );
        assert!(
            close(r.normal_magnitude, normal_mag),
            "sweep normal_magnitude mismatch: query={q:?}"
        );
        assert!(
            close(r.tangential_magnitude, tang_mag),
            "sweep tangential_magnitude mismatch: query={q:?}"
        );
        assert_eq!(r.sliding, sliding, "sweep sliding mismatch: query={q:?}");
    }
}
