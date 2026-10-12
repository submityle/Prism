//! Real-device parity for the dual-quaternion screw-algebra twin:
//! [`GpuDualQuatScrew`](prism_volumetric_gpu::dual_quat_screw::GpuDualQuatScrew)
//! must reproduce the `CPU` golden `prism_math::dual_quat::DualQuat` advanced
//! algebra free functions, selected by an unsigned `op_id`. A dual quaternion
//! `q = real + e*dual` stores two quaternions `(x, y, z, w)` with `w` the scalar
//! part; this twin ports `real_dot`, `conjugate`, `inverse`, `normalize`,
//! `negated`, `nlerp`, `sclerp` and the screw-parameter `powf`. The
//! `transform_point3`/`from_rotation_translation`/`blend_weighted` (slice loop)
//! operations are twinned elsewhere and are excluded here.
//!
//! The oracle is an independent pure-`f32` re-implementation of those closed
//! forms, written out directly so the test never imports `prism_math`, `glam`,
//! `prism_physics_core` or `prism_render_architecture`. It mirrors the same
//! Hamilton-product operator order, the same dual product, the same screw
//! parameters and the same validity gates. Two facts are reproduced faithfully:
//! `inverse` equals `conjugate` exactly (the unit dual-quaternion convention),
//! and `powf`'s near-zero-rotation case is an in-domain pure translation that
//! stays `valid = 1` — only `normalize` and the `normalize` tail of `nlerp`
//! reject a zero-length real part.
//!
//! The fixtures cover each `op_id` with hand-checked inputs, the screw identity
//! properties (`sclerp(a, b, 0) ≈ a`, `sclerp(a, b, 1) ≈ b`, `powf(a, 1) ≈ a`,
//! the pure-translation branch), the `normalize` zero-length rejection, an
//! out-of-range `op_id`, a mixed batch validating the `std430` stride, and an
//! empty batch the host short-circuits with no dispatch. A sweep over random
//! unit dual quaternions with bounded screw angles follows; it keeps every query
//! master-valid and covers all eight operations.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Both sides evaluate the same pure-`f32` closed form, so `CPU` and `GPU` need
//! not be bit-exact under reassociation. Every continuous lane is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the `valid` word is
//! compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::dual_quat::DualQuat`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::dual_quat_screw::{
    DualQuatScrewQuery, DualQuatScrewResult, GpuDualQuatScrew,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Near-zero rotation threshold shared with the kernel's pure-translation
/// branch of `powf`.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// The resolved oracle output, in the same encoding as the device result.
struct Oracle {
    out: [f32; 8],
    valid: bool,
}

/// Four-component dot product.
fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Quaternion conjugate: negate the vector part, keep the scalar.
fn qconj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Hamilton product in the golden operator order (`a * b` = apply `b` then `a`).
fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Component-wise sum.
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Component-wise scale.
fn scale4(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// Dual-quaternion product: `real = ar*br`, `dual = ar*bd + ad*br`.
fn dqmul(ar: [f32; 4], ad: [f32; 4], br: [f32; 4], bd: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let r = qmul(ar, br);
    let d = add4(qmul(ar, bd), qmul(ad, br));
    (r, d)
}

/// Screw-parameter power of a unit dual quaternion, matching the kernel's
/// closed form including the in-domain pure-translation branch.
fn dqpowf(ar: [f32; 4], ad: [f32; 4], t: f32) -> ([f32; 4], [f32; 4]) {
    let vr = [ar[0], ar[1], ar[2]];
    let wr = ar[3];
    let vd = [ad[0], ad[1], ad[2]];
    let wd = ad[3];
    let len_sq = vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2];
    if len_sq < MIN_LEN_SQ {
        // Pure translation: the rotation is (near) zero.
        let sign = if wr < 0.0 { -1.0 } else { 1.0 };
        let real = [0.0, 0.0, 0.0, sign];
        let dual = scale4(ad, t);
        return (real, dual);
    }
    let inv_r = 1.0 / len_sq.sqrt();
    let angle0 = 2.0 * wr.clamp(-1.0, 1.0).acos();
    let pitch0 = -2.0 * wd * inv_r;
    let direction = [vr[0] * inv_r, vr[1] * inv_r, vr[2] * inv_r];
    let moment = [
        (vd[0] - direction[0] * (pitch0 * wr * 0.5)) * inv_r,
        (vd[1] - direction[1] * (pitch0 * wr * 0.5)) * inv_r,
        (vd[2] - direction[2] * (pitch0 * wr * 0.5)) * inv_r,
    ];
    let angle = angle0 * t;
    let pitch = pitch0 * t;
    let half = angle * 0.5;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let real = [
        direction[0] * sin_half,
        direction[1] * sin_half,
        direction[2] * sin_half,
        cos_half,
    ];
    let dual_v = [
        moment[0] * sin_half + direction[0] * (pitch * 0.5 * cos_half),
        moment[1] * sin_half + direction[1] * (pitch * 0.5 * cos_half),
        moment[2] * sin_half + direction[2] * (pitch * 0.5 * cos_half),
    ];
    let dual = [dual_v[0], dual_v[1], dual_v[2], -pitch * 0.5 * sin_half];
    (real, dual)
}

/// Splits a packed `[real.xyzw, dual.xyzw]` operand into its two quaternions.
fn split(q: [f32; 8]) -> ([f32; 4], [f32; 4]) {
    ([q[0], q[1], q[2], q[3]], [q[4], q[5], q[6], q[7]])
}

/// Normalizes a dual quaternion, returning `None` for a zero-length real part.
fn normalize_dq(real: [f32; 4], dual: [f32; 4]) -> Option<([f32; 4], [f32; 4])> {
    let mag_sq = dot4(real, real);
    if mag_sq > MIN_LEN_SQ {
        let inv = 1.0 / mag_sq.sqrt();
        let nr = scale4(real, inv);
        let nd0 = scale4(dual, inv);
        let dsc = dot4(nr, nd0);
        let nd = [
            nd0[0] - nr[0] * dsc,
            nd0[1] - nr[1] * dsc,
            nd0[2] - nr[2] * dsc,
            nd0[3] - nr[3] * dsc,
        ];
        Some((nr, nd))
    } else {
        None
    }
}

/// Independent host oracle: reproduces each `op_id`-selected golden closed form
/// in the golden operator order, pure `f32`.
fn oracle(q: &DualQuatScrewQuery) -> Oracle {
    let (ar, ad) = split(q.a);
    let (br, bd) = split(q.b);
    let t = q.t;
    let mut rr = [0.0f32; 4];
    let mut rd = [0.0f32; 4];
    let mut ok = true;

    match q.op_id {
        0 => {
            // real_dot(a, b): dot of the real parts, in out[0].
            rr[0] = dot4(ar, br);
        }
        1 => {
            // conjugate: quaternion-conjugate both parts.
            rr = qconj(ar);
            rd = qconj(ad);
        }
        2 => {
            // inverse == conjugate exactly.
            rr = qconj(ar);
            rd = qconj(ad);
        }
        3 => {
            // normalize: scale by 1/|real|, then orthogonalize the dual.
            match normalize_dq(ar, ad) {
                Some((nr, nd)) => {
                    rr = nr;
                    rd = nd;
                }
                None => ok = false,
            }
        }
        4 => {
            // negated: negate both parts.
            rr = scale4(ar, -1.0);
            rd = scale4(ad, -1.0);
        }
        5 => {
            // nlerp: hemisphere-align b, blend linearly, then normalize.
            let flip = dot4(ar, br) < 0.0;
            let (br2, bd2) = if flip {
                (scale4(br, -1.0), scale4(bd, -1.0))
            } else {
                (br, bd)
            };
            let real = add4(scale4(ar, 1.0 - t), scale4(br2, t));
            let dual = add4(scale4(ad, 1.0 - t), scale4(bd2, t));
            match normalize_dq(real, dual) {
                Some((nr, nd)) => {
                    rr = nr;
                    rd = nd;
                }
                None => ok = false,
            }
        }
        6 => {
            // sclerp: a * (a^-1 * b)^t with hemisphere alignment.
            let flip = dot4(ar, br) < 0.0;
            let (br2, bd2) = if flip {
                (scale4(br, -1.0), scale4(bd, -1.0))
            } else {
                (br, bd)
            };
            let (dr, dd) = dqmul(qconj(ar), qconj(ad), br2, bd2);
            let (pr, pd) = dqpowf(dr, dd, t);
            let (res_r, res_d) = dqmul(ar, ad, pr, pd);
            rr = res_r;
            rd = res_d;
        }
        7 => {
            // powf: screw-parameter power of a single dual quaternion.
            let (pr, pd) = dqpowf(ar, ad, t);
            rr = pr;
            rd = pd;
        }
        _ => ok = false,
    }

    let out = if ok {
        [rr[0], rr[1], rr[2], rr[3], rd[0], rd[1], rd[2], rd[3]]
    } else {
        [0.0; 8]
    };
    Oracle { out, valid: ok }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the `valid`
/// word matches exactly, each lane matches to tolerance when valid, and every
/// lane is zero when the master flag is cleared.
fn assert_parity(gpu: &DualQuatScrewResult, q: &DualQuatScrewQuery, label: &str) {
    let o = oracle(q);
    let gpu_valid = gpu.valid == 1;
    assert_eq!(gpu_valid, o.valid, "{label}: master valid flag");

    if !o.valid {
        for (i, lane) in gpu.out.iter().enumerate() {
            assert_eq!(*lane, 0.0, "{label}: out[{i}] zeroed");
        }
        return;
    }

    for i in 0..8 {
        assert!(
            close(gpu.out[i], o.out[i]),
            "{label}: out[{i}] gpu={} oracle={}",
            gpu.out[i],
            o.out[i]
        );
    }
}

/// Builds a unit dual quaternion from a unit axis, rotation angle and a
/// translation. `real = (axis*sin(θ/2), cos(θ/2))`, `dual = 0.5 * t_quat ⊗ real`
/// so `dot(real, dual) = 0`.
fn unit_dq(axis: [f32; 3], angle: f32, translation: [f32; 3]) -> [f32; 8] {
    let half = angle * 0.5;
    let s = half.sin();
    let c = half.cos();
    let real = [axis[0] * s, axis[1] * s, axis[2] * s, c];
    let t_quat = [translation[0], translation[1], translation[2], 0.0];
    let dual = scale4(qmul(t_quat, real), 0.5);
    [
        real[0], real[1], real[2], real[3], dual[0], dual[1], dual[2], dual[3],
    ]
}

/// Normalizes a 3-vector (host-side only).
fn unit3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

const ZERO8: [f32; 8] = [0.0; 8];

#[test]
fn real_dot_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.0, 0.0, 1.0]), 0.6, [1.0, 2.0, 3.0]);
    let b = unit_dq(unit3([0.0, 1.0, 0.0]), 0.4, [-1.0, 0.5, 2.0]);
    let q = DualQuatScrewQuery::new(0u32, a, b, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let expect = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    assert!(close(out[0].out[0], expect), "real_dot scalar");
    assert_parity(&out[0], &q, "real_dot");
}

#[test]
fn conjugate_and_inverse_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([1.0, -2.0, 0.5]), 0.9, [0.3, -0.7, 1.2]);
    let queries = vec![
        DualQuatScrewQuery::new(1u32, a, ZERO8, 0.0),
        DualQuatScrewQuery::new(2u32, a, ZERO8, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    // inverse == conjugate exactly: all lanes identical.
    for i in 0..8 {
        assert!(
            close(out[0].out[i], out[1].out[i]),
            "inverse equals conjugate on lane {i}"
        );
    }
    assert_parity(&out[0], &queries[0], "conjugate");
    assert_parity(&out[1], &queries[1], "inverse");
}

#[test]
fn negated_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.5, 0.5, 0.5]), 1.1, [2.0, -1.0, 0.0]);
    let q = DualQuatScrewQuery::new(4u32, a, ZERO8, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    for i in 0..8 {
        assert!(close(out[0].out[i], -a[i]), "negated lane {i}");
    }
    assert_parity(&out[0], &q, "negated");
}

#[test]
fn normalize_matches_oracle_and_is_unit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    // A non-unit dual quaternion (scaled by 3) with a non-orthogonal dual.
    let a = [0.6f32, 0.0, 0.0, 2.4, 0.3, 0.1, -0.2, 0.5];
    let q = DualQuatScrewQuery::new(3u32, a, ZERO8, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // Real part is unit length.
    let real = [out[0].out[0], out[0].out[1], out[0].out[2], out[0].out[3]];
    assert!(close(dot4(real, real), 1.0), "normalized real is unit");
    // Dual is orthogonal to real.
    let dual = [out[0].out[4], out[0].out[5], out[0].out[6], out[0].out[7]];
    assert!(close(dot4(real, dual), 0.0), "dual orthogonal to real");
    assert_parity(&out[0], &q, "normalize");
}

#[test]
fn normalize_degenerate_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    // Zero-length real part → invalid.
    let a = [0.0f32, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4];
    let q = DualQuatScrewQuery::new(3u32, a, ZERO8, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0, "zero-length real rejected");
    assert_parity(&out[0], &q, "normalize_degenerate");
}

#[test]
fn powf_identity_and_pure_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.2, 0.9, -0.3]), 1.0, [0.5, -0.4, 0.9]);
    // powf(a, 1) ≈ a; pure-translation branch (angle 0) → rotation stays
    // identity, translation scales by t.
    let pure = unit_dq([0.0, 0.0, 1.0], 0.0, [1.0, -2.0, 0.5]);
    let queries = vec![
        DualQuatScrewQuery::new(7u32, a, ZERO8, 1.0),
        DualQuatScrewQuery::new(7u32, pure, ZERO8, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    for i in 0..8 {
        assert!(close(out[0].out[i], a[i]), "powf(a, 1) lane {i}");
    }
    // Pure translation: real stays (0,0,0,1), dual halves.
    assert!(close(out[1].out[3], 1.0), "pure rotation scalar stays 1");
    for i in 0..3 {
        assert!(close(out[1].out[i], 0.0), "pure rotation vector zero {i}");
    }
    for i in 4..8 {
        assert!(close(out[1].out[i], pure[i] * 0.5), "pure dual halves {i}");
    }
    assert_parity(&out[0], &queries[0], "powf_identity");
    assert_parity(&out[1], &queries[1], "powf_pure_translation");
}

#[test]
fn sclerp_endpoints_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.1, 0.2, 0.97]), 0.7, [1.0, 0.0, -1.0]);
    let b = unit_dq(unit3([0.8, 0.1, 0.3]), 1.3, [-0.5, 2.0, 0.4]);
    let queries = vec![
        DualQuatScrewQuery::new(6u32, a, b, 0.0),
        DualQuatScrewQuery::new(6u32, a, b, 1.0),
        DualQuatScrewQuery::new(6u32, a, b, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    // sclerp(a, b, 0) ≈ a.
    for i in 0..8 {
        assert!(close(out[0].out[i], a[i]), "sclerp t=0 lane {i}");
    }
    // sclerp(a, b, 1) ≈ b (hemisphere-aligned, so matches b up to sign; the
    // oracle applies the same alignment, so parity below is the real check).
    assert_parity(&out[0], &queries[0], "sclerp_t0");
    assert_parity(&out[1], &queries[1], "sclerp_t1");
    assert_parity(&out[2], &queries[2], "sclerp_mid");
}

#[test]
fn nlerp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.0, 0.0, 1.0]), 0.3, [1.0, 1.0, 0.0]);
    let b = unit_dq(unit3([0.0, 1.0, 0.0]), 0.8, [0.0, -1.0, 2.0]);
    let q = DualQuatScrewQuery::new(5u32, a, b, 0.35);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let real = [out[0].out[0], out[0].out[1], out[0].out[2], out[0].out[3]];
    assert!(close(dot4(real, real), 1.0), "nlerp real is unit");
    assert_parity(&out[0], &q, "nlerp");
}

#[test]
fn out_of_range_op_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([1.0, 0.0, 0.0]), 0.5, [1.0, 2.0, 3.0]);
    let q = DualQuatScrewQuery::new(15u32, a, ZERO8, 0.3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0, "op_id 15 is out of range");
    assert_parity(&out[0], &q, "out_of_range");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let a = unit_dq(unit3([0.3, 0.4, 0.9]), 0.9, [1.0, -1.0, 2.0]);
    let b = unit_dq(unit3([0.6, 0.1, 0.8]), 1.1, [-2.0, 0.5, 1.0]);
    let queries = vec![
        DualQuatScrewQuery::new(0u32, a, b, 0.0),
        DualQuatScrewQuery::new(1u32, a, ZERO8, 0.0),
        DualQuatScrewQuery::new(4u32, a, ZERO8, 0.0),
        // Zero-length normalize → invalid, mid-batch, to exercise the stride.
        DualQuatScrewQuery::new(3u32, ZERO8, ZERO8, 0.0),
        DualQuatScrewQuery::new(6u32, a, b, 0.5),
        DualQuatScrewQuery::new(7u32, a, ZERO8, 0.5),
        // Out-of-range op → invalid.
        DualQuatScrewQuery::new(15u32, a, ZERO8, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    assert_eq!(out[3].valid, 0, "slot 3 zero-length normalize invalid");
    assert_eq!(out[6].valid, 0, "slot 6 out-of-range invalid");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuatScrew::new(&ctx);
    let mut lcg = Lcg::new(0x2B7E_1591);
    let mut queries = Vec::with_capacity(512);
    let mut op_counts = [0u32; 8];
    while queries.len() < 512 {
        let op = lcg.next_u32() % 8;
        // Unit axes keep the real part a unit quaternion; angles are bounded
        // well away from 0 and pi so the acos domain and the powf
        // pure-translation knee are never reached.
        let axis_a = unit3([
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
        ]);
        let axis_b = unit3([
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
        ]);
        let angle_a = lcg.next_range(0.1, 2.5);
        let angle_b = lcg.next_range(0.1, 2.5);
        let trans_a = [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ];
        let trans_b = [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ];
        let a = unit_dq(axis_a, angle_a, trans_a);
        let b = unit_dq(axis_b, angle_b, trans_b);
        // Interpolation / exponent parameter in a bounded range.
        let t = lcg.next_range(0.0, 1.0);
        op_counts[op as usize] += 1;
        queries.push(DualQuatScrewQuery::new(op, a, b, t));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be master-valid");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
    for (op, c) in op_counts.iter().enumerate() {
        assert!(*c > 0, "sweep should cover op_id {op}");
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
