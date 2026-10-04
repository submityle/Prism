//! Real-device parity for the quaternion-construction twin:
//! [`GpuQuatConstruct`](prism_volumetric_gpu::quat_construct::GpuQuatConstruct)
//! must reproduce the `CPU` golden `prism_math::quat::Quat` construction and
//! basic-algebra free functions, selected by an unsigned `op_id`. The golden
//! stores a quaternion as `(x, y, z, w)` with `w` the scalar part; this twin
//! ports `from_axis_angle`, `from_rotation_x`/`y`/`z`, `conjugate`, `inverse`,
//! `normalize`, `dot`, `length` and `length_squared`. The
//! rotate/`mul_vec3`/`nlerp`/`slerp` operations are twinned elsewhere and are
//! excluded here.
//!
//! The oracle is an independent re-implementation of those closed forms,
//! written out directly so the test never imports `prism_math`, `glam`,
//! `prism_physics_core` or `prism_render_architecture`. The golden is pure
//! `f32`, so the oracle is pure `f32` too and mirrors the same operator order
//! and the same validity gates. Two findings are reproduced faithfully:
//! `from_axis_angle` does **not** normalize the axis (it simply scales a unit
//! axis by `sin(angle/2)`), and `inverse` equals `conjugate` exactly (the
//! unit-quaternion convention), not `conjugate / length_squared`.
//!
//! The fixtures cover each `op_id` with hand-checked inputs, the `normalize`
//! zero-length and non-finite rejections, an out-of-range `op_id`, a mixed
//! batch validating the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random finite operands
//! follows; it keeps every query master-valid (rejecting the zero-length
//! `normalize` input) and covers all ten operations.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Both sides evaluate the same pure-`f32` closed form, so `CPU` and `GPU` need
//! not be bit-exact under reassociation. Every continuous scalar is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the `valid` word
//! is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::quat::Quat`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quat_construct::{
    GpuQuatConstruct, QuatConstructQuery, QuatConstructResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The resolved oracle outputs, in the same encoding as the device result.
struct Oracle {
    out: [f32; 4],
    valid: bool,
}

/// Four-component dot product in the golden operator order.
fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Independent host oracle: reproduces each `op_id`-selected golden closed form
/// in the golden operator order, pure `f32`.
fn oracle(q: &QuatConstructQuery) -> Oracle {
    let mut r = [0.0f32; 4];
    let mut ok = true;

    match q.op_id {
        0 => {
            // from_axis_angle: axis * s, w = c. Axis is NOT normalized.
            let half = q.angle * 0.5;
            let s = half.sin();
            let c = half.cos();
            r = [q.axis[0] * s, q.axis[1] * s, q.axis[2] * s, c];
        }
        1 => {
            let half = q.angle * 0.5;
            let s = half.sin();
            let c = half.cos();
            r = [s, 0.0, 0.0, c];
        }
        2 => {
            let half = q.angle * 0.5;
            let s = half.sin();
            let c = half.cos();
            r = [0.0, s, 0.0, c];
        }
        3 => {
            let half = q.angle * 0.5;
            let s = half.sin();
            let c = half.cos();
            r = [0.0, 0.0, s, c];
        }
        4 => {
            // conjugate.
            r = [-q.q0[0], -q.q0[1], -q.q0[2], q.q0[3]];
        }
        5 => {
            // inverse == conjugate exactly.
            r = [-q.q0[0], -q.q0[1], -q.q0[2], q.q0[3]];
        }
        6 => {
            // normalize: zero-length or non-finite → invalid.
            let ls = dot4(q.q0, q.q0);
            let finite = ls.abs() < 3.0e38;
            let nonzero = ls > 1.0e-30;
            let nok = finite && nonzero;
            if nok {
                let inv = 1.0 / ls.sqrt();
                r = [q.q0[0] * inv, q.q0[1] * inv, q.q0[2] * inv, q.q0[3] * inv];
            }
            ok = nok;
        }
        7 => {
            // dot(q0, q1) → scalar in out[0].
            r[0] = dot4(q.q0, q.q1);
        }
        8 => {
            // length(q0) → scalar in out[0].
            r[0] = dot4(q.q0, q.q0).sqrt();
        }
        9 => {
            // length_squared(q0) → scalar in out[0].
            r[0] = dot4(q.q0, q.q0);
        }
        _ => {
            ok = false;
        }
    }

    if !ok {
        r = [0.0, 0.0, 0.0, 0.0];
    }

    Oracle { out: r, valid: ok }
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
fn assert_parity(gpu: &QuatConstructResult, q: &QuatConstructQuery, label: &str) {
    let o = oracle(q);
    let gpu_valid = gpu.valid == 1;
    assert_eq!(gpu_valid, o.valid, "{label}: master valid flag");

    if !o.valid {
        for (i, lane) in gpu.out.iter().enumerate() {
            assert_eq!(*lane, 0.0, "{label}: out[{i}] zeroed");
        }
        return;
    }

    for i in 0..4 {
        assert!(
            close(gpu.out[i], o.out[i]),
            "{label}: out[{i}] gpu={} oracle={}",
            gpu.out[i],
            o.out[i]
        );
    }
}

/// Normalizes a 3-vector for `from_axis_angle` fixtures (host-side only).
fn unit3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

const ZERO4: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ZERO3: [f32; 3] = [0.0, 0.0, 0.0];

#[test]
fn from_axis_angle_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let axis = unit3([0.0, 0.0, 1.0]);
    // angle = pi/2 → (0, 0, sin(pi/4), cos(pi/4)) ≈ (0, 0, 0.7071, 0.7071).
    let q = QuatConstructQuery::new(0u32, axis, std::f32::consts::FRAC_PI_2, ZERO4, ZERO4);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].out[2], std::f32::consts::FRAC_1_SQRT_2));
    assert!(close(out[0].out[3], std::f32::consts::FRAC_1_SQRT_2));
    assert_parity(&out[0], &q, "from_axis_angle");
}

#[test]
fn from_rotation_axes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let angle = 0.9f32;
    let queries = vec![
        QuatConstructQuery::new(1u32, ZERO3, angle, ZERO4, ZERO4),
        QuatConstructQuery::new(2u32, ZERO3, angle, ZERO4, ZERO4),
        QuatConstructQuery::new(3u32, ZERO3, angle, ZERO4, ZERO4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "rotation axis[{i}] valid");
        assert_parity(res, q, &format!("from_rotation[{i}]"));
    }
    // from_rotation_x puts sin on lane 0, y on lane 1, z on lane 2.
    let s = (angle * 0.5).sin();
    assert!(close(out[0].out[0], s), "x axis sin on lane 0");
    assert!(close(out[1].out[1], s), "y axis sin on lane 1");
    assert!(close(out[2].out[2], s), "z axis sin on lane 2");
}

#[test]
fn conjugate_and_inverse_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let q0 = [0.2f32, -0.5, 0.7, 0.3];
    let queries = vec![
        QuatConstructQuery::new(4u32, ZERO3, 0.0, q0, ZERO4),
        QuatConstructQuery::new(5u32, ZERO3, 0.0, q0, ZERO4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    // inverse == conjugate exactly: both lanes identical.
    for i in 0..4 {
        assert!(
            close(out[0].out[i], out[1].out[i]),
            "inverse equals conjugate on lane {i}"
        );
    }
    assert_parity(&out[0], &queries[0], "conjugate");
    assert_parity(&out[1], &queries[1], "inverse");
}

#[test]
fn normalize_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let q0 = [1.0f32, 2.0, 3.0, 4.0];
    let q = QuatConstructQuery::new(6u32, ZERO3, 0.0, q0, ZERO4);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let len = (out[0].out[0] * out[0].out[0]
        + out[0].out[1] * out[0].out[1]
        + out[0].out[2] * out[0].out[2]
        + out[0].out[3] * out[0].out[3])
        .sqrt();
    assert!(close(len, 1.0), "normalized quat has unit length");
    assert_parity(&out[0], &q, "normalize");
}

#[test]
fn normalize_degenerate_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let queries = vec![
        // Zero-length quaternion.
        QuatConstructQuery::new(6u32, ZERO3, 0.0, ZERO4, ZERO4),
        // Non-finite component.
        QuatConstructQuery::new(6u32, ZERO3, 0.0, [f32::NAN, 0.0, 0.0, 1.0], ZERO4),
        QuatConstructQuery::new(6u32, ZERO3, 0.0, [f32::INFINITY, 0.0, 0.0, 0.0], ZERO4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "normalize degenerate[{i}] rejected");
        assert_parity(res, q, &format!("normalize_degenerate[{i}]"));
    }
}

#[test]
fn dot_length_length_squared_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let q0 = [0.5f32, -1.5, 2.0, 1.0];
    let q1 = [1.0f32, 2.0, -0.5, 0.25];
    let queries = vec![
        QuatConstructQuery::new(7u32, ZERO3, 0.0, q0, q1),
        QuatConstructQuery::new(8u32, ZERO3, 0.0, q0, ZERO4),
        QuatConstructQuery::new(9u32, ZERO3, 0.0, q0, ZERO4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    // dot = 0.5 - 3.0 - 1.0 + 0.25 = -3.25.
    assert!(close(out[0].out[0], -3.25), "dot scalar");
    // length_squared = 0.25 + 2.25 + 4.0 + 1.0 = 7.5.
    assert!(close(out[2].out[0], 7.5), "length_squared scalar");
    assert!(close(out[1].out[0], 7.5f32.sqrt()), "length scalar");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("scalar_op[{i}]"));
    }
}

#[test]
fn out_of_range_op_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let q = QuatConstructQuery::new(15u32, ZERO3, 0.0, [1.0, 2.0, 3.0, 4.0], ZERO4);
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
    let gpu = GpuQuatConstruct::new(&ctx);
    let axis = unit3([1.0, 1.0, 0.0]);
    let queries = vec![
        QuatConstructQuery::new(0u32, axis, 1.2, ZERO4, ZERO4),
        QuatConstructQuery::new(4u32, ZERO3, 0.0, [0.1, 0.2, 0.3, 0.4], ZERO4),
        QuatConstructQuery::new(9u32, ZERO3, 0.0, [1.0, 1.0, 1.0, 1.0], ZERO4),
        // Zero-length normalize → invalid, mid-batch, to exercise the stride.
        QuatConstructQuery::new(6u32, ZERO3, 0.0, ZERO4, ZERO4),
        QuatConstructQuery::new(7u32, ZERO3, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
        QuatConstructQuery::new(15u32, ZERO3, 0.0, ZERO4, ZERO4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    assert_eq!(out[3].valid, 0, "slot 3 zero-length normalize invalid");
    assert_eq!(out[5].valid, 0, "slot 5 out-of-range invalid");
    // slot 2: length_squared of (1,1,1,1) = 4.
    assert!(close(out[2].out[0], 4.0), "slot 2 length_squared");
    // slot 4: dot of orthogonal unit quats = 0.
    assert!(close(out[4].out[0], 0.0), "slot 4 dot");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatConstruct::new(&ctx);
    let mut lcg = Lcg::new(0x2B7E_1591);
    let mut queries = Vec::with_capacity(512);
    let mut op_counts = [0u32; 10];
    while queries.len() < 512 {
        let op = lcg.next_u32() % 10;
        // Angle kept bounded so sin/cos divergence at huge magnitude never
        // dominates the tolerance.
        let angle = lcg.next_range(-10.0, 10.0);
        // Random (not necessarily unit) axis for from_axis_angle; the golden
        // scales it without normalizing, so any finite axis is master-valid.
        let axis = [
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
        ];
        let mut q0 = [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ];
        let q1 = [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ];
        if op == 6 {
            // Keep the normalize input well away from the zero-length knee so
            // the validity decision cannot be flipped by round-off.
            let ls = q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3];
            if ls < 0.25 {
                q0[3] += 1.0;
            }
            let ls2 = q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3];
            if ls2 < 0.25 {
                continue;
            }
        }
        op_counts[op as usize] += 1;
        queries.push(QuatConstructQuery::new(op, axis, angle, q0, q1));
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
