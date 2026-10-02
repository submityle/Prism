//! Real-device parity for the swept *axis-aligned bounding box* (`AABB`)
//! *time-of-impact* (`TOI`) twin:
//! [`GpuSweepAabb`](prism_volumetric_gpu::sweep_aabb::GpuSweepAabb) must
//! reproduce the `CPU` golden
//! [`swept_toi`](prism_render_architecture::particle::sweep_aabb::swept_toi)
//! across an empty batch, a head-on hit at a known fraction, a faster hit, a
//! grazing end-of-step contact, a contact strictly past the step, a receding
//! pair, two statically separated boxes, a static overlap, a moving overlap, an
//! approach from `+x`, an approach from `-y`, a relative-velocity hit driven by
//! both boxes moving, and a large pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each pair is a fixed, non-reorderable reduction to a relative velocity, three
//! guarded divisions and a running interval intersection, so `CPU` and `GPU`
//! evaluate the same closed form in the same associativity. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! `toi` and the normal lanes and asserts an *exact* match on the discrete hit
//! flag, the axis index and the `initially_overlapping` flag. For the random
//! batch a verdict disagreement is tolerated only when the contact window sits
//! inside a narrow tie band (`|t_entry|`, `|1 - t_entry|` or the window width
//! within `1e-2`), the only place where a legal `ULP` perturbation can flip a
//! `<=` verdict; the named fixtures are all placed clear of such boundaries so
//! they assert exact flags unconditionally.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sweep_aabb`；
//! standard separating-axis swept-`AABB` continuous-collision time-of-impact; no
//! third-party engine source or derived code.

use prism_render_architecture::particle::sweep_aabb::{
    swept_toi, Aabb, MovingAabb, SweepResult, Vec3,
};
use prism_volumetric_gpu::sweep_aabb::{GpuSweepAabb, SweepAabbQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on `toi` and the normal lanes. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the tie band inside which a `<=` verdict can legally flip under
/// a `ULP`-scale perturbation, so a boolean disagreement there is tolerated for
/// the random batch (never for the clear-of-boundary fixtures).
const TIE: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Returns whether two normals agree lane for lane within tolerance.
fn vec_close(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

/// The axis-aligned unit box `[0, 1]` on every axis, the fixture most of the
/// named cases probe.
fn unit() -> Aabb {
    Aabb::new(Vec3::ZERO, Vec3::splat(1.0))
}

/// Builds one query from two moving boxes.
fn query(a: MovingAabb, b: MovingAabb) -> SweepAabbQuery {
    SweepAabbQuery { a, b }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the hit verdict matches exactly, and when both hit the `toi`,
/// axis, overlap flag and normal all match (values within tolerance, discrete
/// fields exactly). Returns the `GPU` verdicts for extra per-test assertions.
/// Use only for fixtures placed clear of every boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSweepAabb,
    queries: &[SweepAabbQuery],
) -> Vec<Option<SweepResult>> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = swept_toi(q.a, q.b);
        assert_eq!(
            g.is_some(),
            want.is_some(),
            "lane {lane}: hit gpu {} vs cpu {}",
            g.is_some(),
            want.is_some()
        );
        if let (Some(gr), Some(wr)) = (g, want) {
            assert!(
                close(gr.toi, wr.toi),
                "lane {lane}: toi gpu {} vs cpu {}",
                gr.toi,
                wr.toi
            );
            assert_eq!(
                gr.axis, wr.axis,
                "lane {lane}: axis gpu {} vs cpu {}",
                gr.axis, wr.axis
            );
            assert_eq!(
                gr.initially_overlapping, wr.initially_overlapping,
                "lane {lane}: overlap gpu {} vs cpu {}",
                gr.initially_overlapping, wr.initially_overlapping
            );
            assert!(
                vec_close(gr.normal, wr.normal),
                "lane {lane}: normal gpu ({}, {}, {}) vs cpu ({}, {}, {})",
                gr.normal.x,
                gr.normal.y,
                gr.normal.z,
                wr.normal.x,
                wr.normal.y,
                wr.normal.z
            );
        }
    }
    got
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn head_on_hit_at_half_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Unit box at speed +2 toward a box one unit away: contact at t == 0.5 on x,
    // with the normal on b pointing back toward a.
    let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("head-on approach must hit");
    assert!(close(r.toi, 0.5), "toi {}", r.toi);
    assert_eq!(r.axis, 0, "axis {}", r.axis);
    assert!(!r.initially_overlapping);
    assert!(
        vec_close(r.normal, Vec3::new(-1.0, 0.0, 0.0)),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn faster_hit_at_quarter_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    let a = MovingAabb::new(unit(), Vec3::new(4.0, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("faster approach still hits");
    assert!(close(r.toi, 0.25), "toi {}", r.toi);
}

#[test]
fn contact_exactly_at_end_of_step_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Gap of 1.0 at speed 1.0 => contact at t == 1, inside the closed step.
    let a = MovingAabb::new(unit(), Vec3::new(1.0, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("contact at t == 1 is inside the step");
    assert!(close(r.toi, 1.0), "toi {}", r.toi);
}

#[test]
fn contact_beyond_step_does_not_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Gap of 1.0 at speed 0.5 => contact at t == 2.0, outside [0, 1].
    let a = MovingAabb::new(unit(), Vec3::new(0.5, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].is_none(), "contact past the step is a miss");
}

#[test]
fn receding_pair_never_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    let a = MovingAabb::new(unit(), Vec3::new(-1.0, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].is_none(), "a receding pair never touches");
}

#[test]
fn separated_static_boxes_do_not_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Both axes with no relative motion and far apart: the no-motion separated
    // guard rejects the pair.
    let a = MovingAabb::new(unit(), Vec3::ZERO);
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].is_none(), "separated static boxes never touch");
}

#[test]
fn static_overlap_reports_zero_toi() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Overlapping at rest: no relative motion on any axis, every axis takes the
    // unbounded sentinel interval, so t_entry == -inf clamps to toi == 0.
    let a = MovingAabb::new(unit(), Vec3::ZERO);
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("overlapping boxes contact at t == 0");
    assert!(close(r.toi, 0.0), "toi {}", r.toi);
    assert!(r.initially_overlapping);
    assert!(
        vec_close(r.normal, Vec3::ZERO),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn moving_overlap_reports_zero_toi() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Already overlapping at the start while moving: toi still clamps to zero
    // and the normal is the ambiguous resting-contact zero.
    let a = MovingAabb::new(unit(), Vec3::new(3.0, 1.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
        Vec3::ZERO,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("already overlapping at start");
    assert!(close(r.toi, 0.0), "toi {}", r.toi);
    assert!(r.initially_overlapping);
    assert!(
        vec_close(r.normal, Vec3::ZERO),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn approach_from_positive_x_has_positive_x_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    let a = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::new(-2.0, 0.0, 0.0),
    );
    let b = MovingAabb::new(unit(), Vec3::ZERO);
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("approaching from +x hits");
    assert!(close(r.toi, 0.5), "toi {}", r.toi);
    assert_eq!(r.axis, 0, "axis {}", r.axis);
    assert!(
        vec_close(r.normal, Vec3::new(1.0, 0.0, 0.0)),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn approach_from_below_has_negative_y_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    let a = MovingAabb::new(
        Aabb::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(1.0, -1.0, 1.0)),
        Vec3::new(0.0, 2.0, 0.0),
    );
    let b = MovingAabb::new(unit(), Vec3::ZERO);
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("approaching from -y hits");
    assert!(close(r.toi, 0.5), "toi {}", r.toi);
    assert_eq!(r.axis, 1, "axis {}", r.axis);
    assert!(
        vec_close(r.normal, Vec3::new(0.0, -1.0, 0.0)),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn relative_velocity_from_both_boxes_moving() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);
    // Both boxes move: a at +1 and b at -1 on x give a relative speed of 2 over
    // the unit gap, so contact lands at t == 0.5 exactly like the head-on case.
    let a = MovingAabb::new(unit(), Vec3::new(1.0, 0.0, 0.0));
    let b = MovingAabb::new(
        Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        Vec3::new(-1.0, 0.0, 0.0),
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    let r = got[0].expect("relative approach hits");
    assert!(close(r.toi, 0.5), "toi {}", r.toi);
    assert_eq!(r.axis, 0, "axis {}", r.axis);
    assert!(
        vec_close(r.normal, Vec3::new(-1.0, 0.0, 0.0)),
        "normal ({}, {}, {})",
        r.normal.x,
        r.normal.y,
        r.normal.z
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSweepAabb::new(&ctx);

    let mut saw_hit = false;
    let mut saw_miss = false;
    let mut saw_overlap = false;

    let mut state: u64 = 0x5eed_1234_abcd_9f01;
    for _ in 0..6 {
        let mut queries: Vec<SweepAabbQuery> = Vec::new();
        for _ in 0..128 {
            // Box a: a unit-ish box centered somewhere in [-3, 3], never
            // degenerate.
            let ac = Vec3::new(
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
            );
            let ah = Vec3::new(
                0.5 + lcg(&mut state) * 1.0,
                0.5 + lcg(&mut state) * 1.0,
                0.5 + lcg(&mut state) * 1.0,
            );
            let a_box = Aabb::from_center_half_extent(ac, ah);

            // Box b: another non-degenerate box in the same neighbourhood.
            let bc = Vec3::new(
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
            );
            let bh = Vec3::new(
                0.5 + lcg(&mut state) * 1.0,
                0.5 + lcg(&mut state) * 1.0,
                0.5 + lcg(&mut state) * 1.0,
            );
            let b_box = Aabb::from_center_half_extent(bc, bh);

            // Velocities wide enough that most pairs resolve clear of EPS, with
            // the pair closing or separating at a few units per step.
            let av = Vec3::new(
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
            );
            let bv = Vec3::new(
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
            );

            queries.push(query(
                MovingAabb::new(a_box, av),
                MovingAabb::new(b_box, bv),
            ));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let want = swept_toi(q.a, q.b);

            // Hit verdicts match exactly unless the contact window sits inside a
            // tie band, the only place a ULP perturbation can legally flip a
            // boundary `<=`.
            if g.is_some() != want.is_some() {
                let near_tie = match (g, &want) {
                    (Some(r), None) | (None, Some(r)) => {
                        r.toi.abs() <= TIE || (1.0 - r.toi).abs() <= TIE
                    }
                    _ => false,
                };
                assert!(
                    near_tie,
                    "lane {lane}: hit disagreement off the tie band: gpu {} cpu {}",
                    g.is_some(),
                    want.is_some()
                );
            }

            // The continuous toi is compared only where both agree on a hit.
            // The axis index, contact normal and the initial-overlap flag are
            // left to the named fixtures: near-equal entry times or a t_entry
            // sitting on zero form legal ties a ULP perturbation can flip, so
            // they are not re-asserted on random data.
            if let (Some(gr), Some(wr)) = (g, want) {
                assert!(
                    close(gr.toi, wr.toi),
                    "lane {lane}: toi gpu {} vs cpu {}",
                    gr.toi,
                    wr.toi
                );
                saw_overlap |= gr.initially_overlapping;
            }

            saw_hit |= want.is_some();
            saw_miss |= want.is_none();
        }
    }

    // A large random spread must exercise both verdict classes plus an initial
    // overlap, so the test is not trivially passing on an all-hit or all-miss
    // batch.
    assert!(
        saw_hit && saw_miss && saw_overlap,
        "random batch should produce hits, misses and an initial overlap"
    );
}
