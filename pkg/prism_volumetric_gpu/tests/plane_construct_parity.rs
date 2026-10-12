//! Real-device parity for the plane-construction twin:
//! [`GpuPlaneConstruct`](prism_volumetric_gpu::plane_construct::GpuPlaneConstruct)
//! must reproduce the `CPU` golden `prism_math::geom::plane`'s `Plane` (Hessian
//! normal form, where a plane is the set of points `p` with `normal · p + d ==
//! 0`). Each query selects one `Plane` operation by an unsigned `op_id`, and one
//! thread resolves one query, so a passing parity run is direct evidence the
//! ported kernel reproduces the reference values.
//!
//! The oracle here is an independent re-implementation of every ported
//! operation, written out directly in `f32` with hand-rolled `[f32; 3]`
//! arithmetic, so the test never imports `prism_math`, `glam`,
//! `prism_physics_core` or `prism_render_architecture`.
//!
//! # Operations
//!
//! * `0` `from_point_normal(point, normal)`: `n = normalize(normal)`,
//!   `normal = n`, `d = -n · point`.
//! * `1` `from_points(a, b, c)`: `n = (b - a) × (c - a)`; collinear points
//!   (`|n| <= 1e-20`) are invalid (`valid = 0`).
//! * `2` `normalized(plane)`: scale `normal` and `d` by `1 / |normal|`.
//! * `3` `signed_distance(plane, p)`: `normal · p + d` (scalar).
//! * `4` `project_point(plane, p)`: `p - normal * signed_distance(p)` (vector).
//! * `5` `flipped(plane)`: negate `normal` and `d`.
//! * `6` `distance(plane, p)`: `|signed_distance(p)|` (scalar).
//!
//! # Degenerate inputs and knee avoidance
//!
//! The collinear `from_points` case and a zero-length normal fed to the
//! normalizing operations yield `valid = 0` with zeroed outputs. The `valid`
//! flag is a discrete decision keyed off a `1e-20` length threshold, so the
//! sweep keeps every construction normal well clear of zero length (and keeps
//! `from_points` cross products well clear of collinear) so `CPU` and `GPU`
//! never straddle the validity boundary.
//!
//! # Fixtures
//!
//! Named fixtures cover each operation against the oracle, a collinear
//! `from_points` triple (`valid = 0`), the sign of `signed_distance` on both
//! half-spaces, a projected point landing on the plane, the sign flip of
//! `flipped`, an empty batch the host short-circuits, and a mixed batch (with a
//! collinear invalid entry) that validates the `std430` stride. A `512`-step
//! LCG sweep follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators and a square root a `GPU`
//! may evaluate slightly differently, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact. The three `vec_out` components and the
//! scalar output are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::geom::plane`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::plane_construct::{
    GpuPlaneConstruct, PlaneConstructQuery, PlaneConstructResult, PlaneOp,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Length guard mirroring the golden/kernel: a normal or cross product at or
/// below this length is degenerate.
const EPS_LEN: f32 = 1.0e-20;

/// Three-component dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Three-component cross product.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Component-wise subtraction `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales a vector by a scalar.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean length.
fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Independent host oracle: reproduces the selected `Plane` operation in the
/// golden operator order, returning `(vec_out, scalar, valid)`. A degenerate
/// normalize (zero-length normal) or a collinear `from_points` triple is
/// invalid with zeroed outputs.
fn oracle(q: &PlaneConstructQuery) -> ([f32; 3], f32, u32) {
    match q.op_id {
        0 => {
            // from_point_normal: normalize the normal, d = -n . point.
            let len = length(q.normal);
            if len <= EPS_LEN {
                return ([0.0; 3], 0.0, 0);
            }
            let n = scale(q.normal, 1.0 / len);
            (n, -dot(n, q.a), 1)
        }
        1 => {
            // from_points: n = (b-a) x (c-a); collinear when |n| <= 1e-20.
            let e1 = sub(q.b, q.a);
            let e2 = sub(q.c, q.a);
            let n = cross(e1, e2);
            let len = length(n);
            if len <= EPS_LEN {
                return ([0.0; 3], 0.0, 0);
            }
            let nn = scale(n, 1.0 / len);
            (nn, -dot(nn, q.a), 1)
        }
        2 => {
            // normalized: scale normal and d by 1 / |normal|.
            let len = length(q.normal);
            if len <= EPS_LEN {
                return ([0.0; 3], 0.0, 0);
            }
            let inv = 1.0 / len;
            (scale(q.normal, inv), q.d * inv, 1)
        }
        3 => ([0.0; 3], dot(q.normal, q.p) + q.d, 1),
        4 => {
            let sd = dot(q.normal, q.p) + q.d;
            (sub(q.p, scale(q.normal, sd)), 0.0, 1)
        }
        5 => ([-q.normal[0], -q.normal[1], -q.normal[2]], -q.d, 1),
        6 => ([0.0; 3], (dot(q.normal, q.p) + q.d).abs(), 1),
        _ => ([0.0; 3], 0.0, 0),
    }
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
/// `valid` flag exactly, and the vector/scalar outputs to tolerance when valid,
/// or zeroed when invalid.
fn assert_parity(gpu: &PlaneConstructResult, q: &PlaneConstructQuery, label: &str) {
    let (vec_out, scalar, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        for k in 0..3 {
            assert!(
                close(gpu.vec_out[k], vec_out[k]),
                "{label}: vec_out[{k}] mismatch gpu={} oracle={}",
                gpu.vec_out[k],
                vec_out[k]
            );
        }
        assert!(
            close(gpu.scalar, scalar),
            "{label}: scalar mismatch gpu={} oracle={}",
            gpu.scalar,
            scalar
        );
    } else {
        for k in 0..3 {
            assert_eq!(
                gpu.vec_out[k], 0.0,
                "{label}: invalid vec_out[{k}] should be zero"
            );
        }
        assert_eq!(gpu.scalar, 0.0, "{label}: invalid scalar should be zero");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty(), "empty batch should return an empty vector");
}

#[test]
fn each_op_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    // A general, well-conditioned plane (normal need not be unit for ops 3..=6).
    let normal = [0.3, -0.5, 0.8];
    let unit = {
        let l = length(normal);
        scale(normal, 1.0 / l)
    };
    let d = 1.25;
    let p = [2.0, -1.5, 0.75];
    let queries = vec![
        PlaneConstructQuery::from_point_normal([1.0, 2.0, -3.0], normal),
        PlaneConstructQuery::from_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        PlaneConstructQuery::plane_only(PlaneOp::Normalized, normal, d),
        PlaneConstructQuery::plane_point(PlaneOp::SignedDistance, unit, d, p),
        PlaneConstructQuery::plane_point(PlaneOp::ProjectPoint, unit, d, p),
        PlaneConstructQuery::plane_only(PlaneOp::Flipped, unit, d),
        PlaneConstructQuery::plane_point(PlaneOp::Distance, unit, d, p),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "op[{i}] should be valid");
        assert_parity(res, q, &format!("op[{i}]"));
    }
}

#[test]
fn from_points_collinear_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    // Three collinear points: the cross product vanishes, so valid = 0.
    let q = PlaneConstructQuery::from_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0, "collinear from_points should be invalid");
    assert_parity(&out[0], &q, "collinear");
}

#[test]
fn signed_distance_sign_is_correct() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    // Plane z = 0 with normal +z, d = 0: points above have positive signed
    // distance, points below negative.
    let normal = [0.0, 0.0, 1.0];
    let above =
        PlaneConstructQuery::plane_point(PlaneOp::SignedDistance, normal, 0.0, [1.0, 2.0, 3.0]);
    let below =
        PlaneConstructQuery::plane_point(PlaneOp::SignedDistance, normal, 0.0, [1.0, 2.0, -3.0]);
    let out = gpu.evaluate(&ctx, &[above, below]);
    assert_eq!(out.len(), 2);
    assert!(out[0].scalar > 0.0, "point above plane: {}", out[0].scalar);
    assert!(out[1].scalar < 0.0, "point below plane: {}", out[1].scalar);
    assert_parity(&out[0], &above, "above");
    assert_parity(&out[1], &below, "below");
}

#[test]
fn project_point_lands_on_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    // Project with a unit-normal plane, then confirm the projected point's
    // signed distance is ~0.
    let unit = {
        let n = [0.3, -0.5, 0.8];
        scale(n, 1.0 / length(n))
    };
    let d = 0.7;
    let p = [2.0, -1.5, 0.75];
    let project = PlaneConstructQuery::plane_point(PlaneOp::ProjectPoint, unit, d, p);
    let out = gpu.evaluate(&ctx, &[project]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let proj = out[0].vec_out;
    let sd = dot(unit, proj) + d;
    assert!(sd.abs() <= 1.0e-4, "projected point off-plane, sd={sd}");
    assert_parity(&out[0], &project, "project");
}

#[test]
fn flipped_negates_normal_and_d() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    let normal = [0.3, -0.5, 0.8];
    let d = 1.25;
    let q = PlaneConstructQuery::plane_only(PlaneOp::Flipped, normal, d);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].vec_out[0], -normal[0]), "flip x");
    assert!(close(out[0].vec_out[1], -normal[1]), "flip y");
    assert!(close(out[0].vec_out[2], -normal[2]), "flip z");
    assert!(close(out[0].scalar, -d), "flip d");
    assert_parity(&out[0], &q, "flipped");
}

#[test]
fn batch_mixes_ops_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    let unit = {
        let n = [0.6, 0.0, 0.8];
        scale(n, 1.0 / length(n))
    };
    // Mix every op kind plus a collinear invalid entry to exercise std430
    // stride packing across a heterogeneous batch.
    let queries = vec![
        PlaneConstructQuery::from_point_normal([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]),
        PlaneConstructQuery::from_points([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
        PlaneConstructQuery::plane_only(PlaneOp::Normalized, [0.0, 3.0, 4.0], 2.0),
        PlaneConstructQuery::plane_point(PlaneOp::SignedDistance, unit, 0.5, [1.0, 1.0, 1.0]),
        PlaneConstructQuery::plane_point(PlaneOp::ProjectPoint, unit, 0.5, [1.0, 1.0, 1.0]),
        PlaneConstructQuery::plane_only(PlaneOp::Flipped, unit, 0.5),
        PlaneConstructQuery::plane_point(PlaneOp::Distance, unit, 0.5, [-2.0, 1.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    // Index 1 is collinear from_points -> invalid; everything else valid.
    assert_eq!(out[1].valid, 0, "collinear entry should be invalid");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneConstruct::new(&ctx);
    let mut lcg = Lcg::new(0x7A3C_1E9F);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let op = lcg.next_u32() % 7;
        let q = match op {
            0 => {
                // from_point_normal: well-conditioned normal (length > 0.5).
                let normal = random_nonzero_vec(&mut lcg, 0.5);
                let point = random_vec(&mut lcg, -5.0, 5.0);
                PlaneConstructQuery::from_point_normal(point, normal)
            }
            1 => {
                // from_points: rejection-sample so the cross product length is
                // comfortably away from the collinear knee.
                let (a, b, c) = loop {
                    let a = random_vec(&mut lcg, -5.0, 5.0);
                    let b = random_vec(&mut lcg, -5.0, 5.0);
                    let c = random_vec(&mut lcg, -5.0, 5.0);
                    let n = cross(sub(b, a), sub(c, a));
                    if length(n) > 0.5 {
                        break (a, b, c);
                    }
                };
                PlaneConstructQuery::from_points(a, b, c)
            }
            2 => {
                let normal = random_nonzero_vec(&mut lcg, 0.5);
                let d = lcg.next_range(-5.0, 5.0);
                PlaneConstructQuery::plane_only(PlaneOp::Normalized, normal, d)
            }
            op => {
                // Plane-plus-point ops (3,4,6) and flipped (5). Project uses a
                // unit normal so its land-on-plane invariant holds; others
                // tolerate any normal.
                let raw = random_nonzero_vec(&mut lcg, 0.5);
                let normal = scale(raw, 1.0 / length(raw));
                let d = lcg.next_range(-5.0, 5.0);
                let p = random_vec(&mut lcg, -5.0, 5.0);
                let plane_op = match op {
                    3 => PlaneOp::SignedDistance,
                    4 => PlaneOp::ProjectPoint,
                    5 => PlaneOp::Flipped,
                    _ => PlaneOp::Distance,
                };
                if plane_op == PlaneOp::Flipped {
                    PlaneConstructQuery::plane_only(plane_op, normal, d)
                } else {
                    PlaneConstructQuery::plane_point(plane_op, normal, d, p)
                }
            }
        };
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A random vector with each component in `[lo, hi)`.
fn random_vec(lcg: &mut Lcg, lo: f32, hi: f32) -> [f32; 3] {
    [
        lcg.next_range(lo, hi),
        lcg.next_range(lo, hi),
        lcg.next_range(lo, hi),
    ]
}

/// A random vector rejection-sampled so its length exceeds `min_len`, keeping
/// normalization well clear of the zero-length knee.
fn random_nonzero_vec(lcg: &mut Lcg, min_len: f32) -> [f32; 3] {
    loop {
        let v = random_vec(lcg, -2.0, 2.0);
        if length(v) > min_len {
            return v;
        }
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
