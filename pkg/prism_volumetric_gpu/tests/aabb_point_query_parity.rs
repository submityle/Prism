//! Real-device parity for the `AABB` point-query twin:
//! [`GpuAabbPointQuery`](prism_volumetric_gpu::aabb_point_query::GpuAabbPointQuery)
//! must reproduce the `CPU` golden `prism_math::geom::Aabb3` point and
//! containment closed forms, one `u32`-selected op per thread.
//!
//! The oracle below is an independent `f32` re-implementation of those closed
//! forms, written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core`, `prism_math` or `glam`.
//!
//! This module deliberately twins only the `Aabb3` operations not already
//! covered by the `bounds` twin (surface area, center, half-extent,
//! union/merge, longest axis, point expansion). The seven ops exercised here
//! are volume, point containment, box containment, closest point, squared
//! distance, intersection and uniform expand/shrink.
//!
//! The fixtures cover an empty batch the host short-circuits, every op against
//! the oracle, inclusive boundary points (a point on a face is contained), an
//! inside point whose closest point is itself and whose squared distance is
//! zero, a disjoint intersection reporting `valid = 0`, a negative expand that
//! shrinks the box, an out-of-range op id reporting an all-zero invalid result,
//! a mixed-op batch validating the `std430` stride, and a `512`-step random
//! sweep that keeps the discrete-sensitive ops clear of their decision knees so
//! a containment or overlap flag cannot flip by round-off.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous output (`scalar`, `v0`, `v1`) is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `flag` and
//! `valid` outputs are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::geom::aabb`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::aabb_point_query::{
    AabbPointQueryQuery, AabbPointQueryResult, GpuAabbPointQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Inclusive component-wise containment, matching the golden closed form.
fn contains(lo: [f32; 3], hi: [f32; 3], pt: [f32; 3]) -> bool {
    pt[0] >= lo[0]
        && pt[0] <= hi[0]
        && pt[1] >= lo[1]
        && pt[1] <= hi[1]
        && pt[2] >= lo[2]
        && pt[2] <= hi[2]
}

/// Component-wise clamp of `p` into the box `[lo, hi]`.
fn clamp3(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
    [
        p[0].clamp(lo[0], hi[0]),
        p[1].clamp(lo[1], hi[1]),
        p[2].clamp(lo[2], hi[2]),
    ]
}

/// Independent host oracle: reproduces each `Aabb3` op in the golden `f32`
/// operator order. Inactive fields are zero; an out-of-range op id yields an
/// all-zero invalid result, as does a disjoint intersection.
fn oracle(q: &AabbPointQueryQuery) -> AabbPointQueryResult {
    let mut r = AabbPointQueryResult {
        scalar: 0.0,
        flag: 0,
        v0: [0.0; 3],
        v1: [0.0; 3],
        valid: 0,
    };
    let a_min = q.a_min;
    let a_max = q.a_max;
    match q.op {
        0 => {
            let e = [
                a_max[0] - a_min[0],
                a_max[1] - a_min[1],
                a_max[2] - a_min[2],
            ];
            r.scalar = e[0] * e[1] * e[2];
            r.valid = 1;
        }
        1 => {
            r.flag = u32::from(contains(a_min, a_max, q.p));
            r.valid = 1;
        }
        2 => {
            let inside = contains(a_min, a_max, q.b_min) && contains(a_min, a_max, q.b_max);
            r.flag = u32::from(inside);
            r.valid = 1;
        }
        3 => {
            r.v0 = clamp3(q.p, a_min, a_max);
            r.valid = 1;
        }
        4 => {
            let cl = clamp3(q.p, a_min, a_max);
            let d = [q.p[0] - cl[0], q.p[1] - cl[1], q.p[2] - cl[2]];
            r.scalar = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            r.valid = 1;
        }
        5 => {
            let lo = [
                a_min[0].max(q.b_min[0]),
                a_min[1].max(q.b_min[1]),
                a_min[2].max(q.b_min[2]),
            ];
            let hi = [
                a_max[0].min(q.b_max[0]),
                a_max[1].min(q.b_max[1]),
                a_max[2].min(q.b_max[2]),
            ];
            let overlap = lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2];
            if overlap {
                r.valid = 1;
                r.v0 = lo;
                r.v1 = hi;
            }
        }
        6 => {
            r.v0 = [
                a_min[0] - q.amount,
                a_min[1] - q.amount,
                a_min[2] - q.amount,
            ];
            r.v1 = [
                a_max[0] + q.amount,
                a_max[1] + q.amount,
                a_max[2] + q.amount,
            ];
            r.valid = 1;
        }
        _ => {}
    }
    r
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts one `GPU` result matches the independent oracle: the discrete `flag`
/// and `valid` fields exactly, and the continuous `scalar`, `v0`, `v1` to
/// tolerance.
fn assert_parity(gpu: &AabbPointQueryResult, q: &AabbPointQueryQuery, label: &str) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    assert_eq!(gpu.flag, want.flag, "{label}: flag mismatch");
    assert!(
        close(gpu.scalar, want.scalar),
        "{label}: scalar {} vs {}",
        gpu.scalar,
        want.scalar
    );
    for k in 0..3 {
        assert!(
            close(gpu.v0[k], want.v0[k]),
            "{label}: v0[{k}] {} vs {}",
            gpu.v0[k],
            want.v0[k]
        );
        assert!(
            close(gpu.v1[k], want.v1[k]),
            "{label}: v1[{k}] {} vs {}",
            gpu.v1[k],
            want.v1[k]
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn volume_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let q = AabbPointQueryQuery::volume([-1.0, 0.0, 2.0], [3.0, 4.0, 5.0]);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    // (3 - -1) * (4 - 0) * (5 - 2) = 4 * 4 * 3 = 48.
    assert!(close(out[0].scalar, 48.0), "volume {}", out[0].scalar);
    assert_parity(&out[0], &q, "volume");
}

#[test]
fn contains_point_inclusive_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let a_min = [0.0, 0.0, 0.0];
    let a_max = [2.0, 2.0, 2.0];
    let inside = AabbPointQueryQuery::contains_point(a_min, a_max, [1.0, 1.0, 1.0]);
    let on_min_face = AabbPointQueryQuery::contains_point(a_min, a_max, [0.0, 1.0, 1.0]);
    let on_corner = AabbPointQueryQuery::contains_point(a_min, a_max, [2.0, 2.0, 2.0]);
    let outside = AabbPointQueryQuery::contains_point(a_min, a_max, [2.5, 1.0, 1.0]);
    let queries = vec![inside, on_min_face, on_corner, outside];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].flag, 1, "inside");
    assert_eq!(out[1].flag, 1, "on min face is inclusive");
    assert_eq!(out[2].flag, 1, "on far corner is inclusive");
    assert_eq!(out[3].flag, 0, "outside");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("contains_point[{i}]"));
    }
}

#[test]
fn contains_aabb_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let a_min = [0.0, 0.0, 0.0];
    let a_max = [4.0, 4.0, 4.0];
    let inner = AabbPointQueryQuery::contains_aabb(a_min, a_max, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
    let edge = AabbPointQueryQuery::contains_aabb(a_min, a_max, [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let poking = AabbPointQueryQuery::contains_aabb(a_min, a_max, [1.0, 1.0, 1.0], [5.0, 3.0, 3.0]);
    let queries = vec![inner, edge, poking];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].flag, 1, "inner box contained");
    assert_eq!(out[1].flag, 1, "coincident box is inclusive");
    assert_eq!(out[2].flag, 0, "box poking outside");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("contains_aabb[{i}]"));
    }
}

#[test]
fn closest_point_inside_and_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let a_min = [0.0, 0.0, 0.0];
    let a_max = [2.0, 2.0, 2.0];
    let inside = AabbPointQueryQuery::closest_point(a_min, a_max, [1.0, 1.0, 1.0]);
    let outside = AabbPointQueryQuery::closest_point(a_min, a_max, [5.0, -3.0, 1.0]);
    let out = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(out.len(), 2);
    // Inside point is its own closest point.
    for k in 0..3 {
        assert!(close(out[0].v0[k], [1.0, 1.0, 1.0][k]), "inside v0[{k}]");
    }
    // Outside point clamps per axis to [0, 2].
    let want = [2.0, 0.0, 1.0];
    for k in 0..3 {
        assert!(close(out[1].v0[k], want[k]), "outside v0[{k}]");
    }
    assert_parity(&out[0], &inside, "closest_inside");
    assert_parity(&out[1], &outside, "closest_outside");
}

#[test]
fn distance_squared_inside_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let a_min = [0.0, 0.0, 0.0];
    let a_max = [2.0, 2.0, 2.0];
    let inside = AabbPointQueryQuery::distance_squared(a_min, a_max, [1.0, 1.5, 0.5]);
    let outside = AabbPointQueryQuery::distance_squared(a_min, a_max, [5.0, -1.0, 1.0]);
    let out = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(out.len(), 2);
    assert!(close(out[0].scalar, 0.0), "inside distance^2 is zero");
    // Clamp -> [2, 0, 1]; delta = [3, -1, 0]; squared = 9 + 1 = 10.
    assert!(
        close(out[1].scalar, 10.0),
        "outside distance^2 {}",
        out[1].scalar
    );
    assert_parity(&out[0], &inside, "dist_inside");
    assert_parity(&out[1], &outside, "dist_outside");
}

#[test]
fn intersection_overlap_and_disjoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let overlap = AabbPointQueryQuery::intersection(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
    );
    let disjoint = AabbPointQueryQuery::intersection(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    );
    let out = gpu.evaluate(&ctx, &[overlap, disjoint]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1, "overlap is valid");
    for k in 0..3 {
        assert!(close(out[0].v0[k], [1.0, 1.0, 1.0][k]), "overlap lo[{k}]");
        assert!(close(out[0].v1[k], [2.0, 2.0, 2.0][k]), "overlap hi[{k}]");
    }
    assert_eq!(out[1].valid, 0, "disjoint is invalid");
    for k in 0..3 {
        assert_eq!(out[1].v0[k], 0.0, "disjoint lo[{k}] zeroed");
        assert_eq!(out[1].v1[k], 0.0, "disjoint hi[{k}] zeroed");
    }
    assert_parity(&out[0], &overlap, "intersection_overlap");
    assert_parity(&out[1], &disjoint, "intersection_disjoint");
}

#[test]
fn expand_grows_and_shrinks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let a_min = [0.0, 0.0, 0.0];
    let a_max = [2.0, 2.0, 2.0];
    let grow = AabbPointQueryQuery::expand(a_min, a_max, 0.5);
    let shrink = AabbPointQueryQuery::expand(a_min, a_max, -0.5);
    let out = gpu.evaluate(&ctx, &[grow, shrink]);
    assert_eq!(out.len(), 2);
    for k in 0..3 {
        assert!(close(out[0].v0[k], -0.5), "grow min[{k}]");
        assert!(close(out[0].v1[k], 2.5), "grow max[{k}]");
        assert!(close(out[1].v0[k], 0.5), "shrink min[{k}]");
        assert!(close(out[1].v1[k], 1.5), "shrink max[{k}]");
    }
    assert_parity(&out[0], &grow, "expand_grow");
    assert_parity(&out[1], &shrink, "expand_shrink");
}

#[test]
fn out_of_range_op_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let make = |op: u32| AabbPointQueryQuery {
        op,
        a_min: [0.0, 0.0, 0.0],
        a_max: [1.0, 1.0, 1.0],
        b_min: [0.0, 0.0, 0.0],
        b_max: [0.0, 0.0, 0.0],
        p: [0.0, 0.0, 0.0],
        amount: 0.0,
    };
    let queries = vec![make(7), make(99)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "op {} should be invalid", q.op);
        assert_eq!(res.flag, 0, "invalid[{i}] flag zeroed");
        assert_eq!(res.scalar, 0.0, "invalid[{i}] scalar zeroed");
        for k in 0..3 {
            assert_eq!(res.v0[k], 0.0, "invalid[{i}] v0[{k}] zeroed");
            assert_eq!(res.v1[k], 0.0, "invalid[{i}] v1[{k}] zeroed");
        }
        assert_parity(res, q, &format!("out_of_range[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    // A mix of every op exercises the std430 stride end to end.
    let queries = vec![
        AabbPointQueryQuery::volume([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
        AabbPointQueryQuery::contains_point([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
        AabbPointQueryQuery::contains_aabb(
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ),
        AabbPointQueryQuery::closest_point([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [5.0, -1.0, 1.0]),
        AabbPointQueryQuery::distance_squared([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [5.0, -1.0, 1.0]),
        AabbPointQueryQuery::intersection(
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ),
        AabbPointQueryQuery::expand([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.25),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbPointQuery::new(&ctx);
    let mut lcg = Lcg::new(0x5EED_A1B2);
    let mut queries = Vec::with_capacity(512);
    let mut counts = [0u32; 7];
    // Keep discrete-sensitive ops clear of their decision knees so the
    // containment/overlap flag cannot flip by round-off.
    let margin = 0.05f32;
    while queries.len() < 512 {
        let op = lcg.next_u32() % 7;
        let a_min = [
            lcg.next_range(-5.0, 5.0),
            lcg.next_range(-5.0, 5.0),
            lcg.next_range(-5.0, 5.0),
        ];
        let a_max = [
            a_min[0] + lcg.next_range(0.1, 4.0),
            a_min[1] + lcg.next_range(0.1, 4.0),
            a_min[2] + lcg.next_range(0.1, 4.0),
        ];
        let b_min = [
            lcg.next_range(-5.0, 5.0),
            lcg.next_range(-5.0, 5.0),
            lcg.next_range(-5.0, 5.0),
        ];
        let b_max = [
            b_min[0] + lcg.next_range(0.1, 4.0),
            b_min[1] + lcg.next_range(0.1, 4.0),
            b_min[2] + lcg.next_range(0.1, 4.0),
        ];
        let p = [
            lcg.next_range(-7.0, 7.0),
            lcg.next_range(-7.0, 7.0),
            lcg.next_range(-7.0, 7.0),
        ];
        let amount = lcg.next_range(-1.0, 2.0);

        // Reject samples sitting on a discrete decision boundary.
        if op == 1 && near_face(p, a_min, a_max, margin) {
            continue;
        }
        if op == 2
            && (near_face(b_min, a_min, a_max, margin) || near_face(b_max, a_min, a_max, margin))
        {
            continue;
        }
        if op == 5 {
            let mut too_close = false;
            for k in 0..3 {
                let lo = a_min[k].max(b_min[k]);
                let hi = a_max[k].min(b_max[k]);
                if (lo - hi).abs() < margin {
                    too_close = true;
                }
            }
            if too_close {
                continue;
            }
        }

        counts[op as usize] += 1;
        queries.push(AabbPointQueryQuery {
            op,
            a_min,
            a_max,
            b_min,
            b_max,
            p,
            amount,
        });
    }
    for (op, &c) in counts.iter().enumerate() {
        assert!(c >= 1, "sweep should cover op {op}, got {c}");
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// Returns true when `pt` is within `margin` of any face of box `[lo, hi]` on
/// any axis, i.e. a tiny perturbation could flip inclusive containment.
fn near_face(pt: [f32; 3], lo: [f32; 3], hi: [f32; 3], margin: f32) -> bool {
    for k in 0..3 {
        if (pt[k] - lo[k]).abs() < margin || (pt[k] - hi[k]).abs() < margin {
            return true;
        }
    }
    false
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
