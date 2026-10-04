//! Real-device parity for the isometric-bending projection twin:
//! [`GpuSoftIsometricBendingProject`](prism_volumetric_gpu::soft_isometric_bending_project::GpuSoftIsometricBendingProject)
//! must reproduce the `CPU` golden `project_isometric_bending` of
//! `prism_physics_core::soft::constraint::isometric_bending`. For each bending
//! stencil the reference couples the four particles through their weighted sum
//! `S = Σ wtᵢ·pᵢ` and drives `S` back toward zero (the flat configuration) with
//! a single compliant `XPBD` step, moving each free corner along its gradient.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the weighted sum `S`, the squared length `s_len_sq`, the bending energy, the
//! effective-mass sum `sum_w_grad`, the `alpha_tilde` compliance term, the three
//! no-op guards (`dt <= 0`, `s_len_sq <= EPS_LEN_SQ`, `denom <= 0`), the
//! `d_lambda` solution and the per-corner update — written out directly so the
//! test never imports `prism_render_architecture` or `prism_physics_core`. It
//! mirrors the reference branch for branch, so a passing comparison is evidence
//! the ported kernel took the same degenerate / projecting branch, not merely
//! that the shader compiled.
//!
//! The fixtures cover each branch the kernel must honor: a non-planar stencil
//! whose bending is projected, a flat stencil whose `S == 0` makes
//! `s_len_sq <= EPS_LEN_SQ` a no-op, a non-positive `dt` no-op, a stencil with a
//! pinned corner (verifying that corner stays put while the free ones move), a
//! batch of two or more elements that validates the `std430` stride, and an
//! empty batch the host short-circuits with no dispatch. A sweep over random
//! non-planar stencils follows, rejecting marginal geometry (where a tiny
//! perturbation would collapse `s_len_sq` toward the flat knee or `denom` toward
//! zero and flip the `valid` flag) so the branch choice agrees on both sides
//! despite any last-bit difference.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every path threads through multiplies, adds, a dot product and a guarded
//! division, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). Each continuous output is
//! compared with `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::isometric_bending`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_isometric_bending_project::{
    GpuSoftIsometricBendingProject, SoftIsometricBendingProjectQuery,
    SoftIsometricBendingProjectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// The flat-stencil knee, matching the kernel and golden `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale_ref = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale_ref <= REL_EPS
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Vector sum `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a vector.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Independent host re-implementation of `project_isometric_bending` for a
/// single bending stencil. Returns the four updated positions and the discrete
/// validity flag, replicated branch for branch in the same accumulation order
/// as the kernel (`p0..p3` for `S`, `t0..t3` for `sum_w_grad`).
fn oracle(q: &SoftIsometricBendingProjectQuery) -> ([[f32; 3]; 4], u32) {
    let p = q.positions;
    let w = q.inverse_masses;
    let wt = q.weights;
    let bend_scale = q.scale;

    // S = Sum wt_i * p_i; every stencil vertex contributes, including pinned.
    // Left-to-right accumulation matching the kernel's `p0..p3` order exactly.
    let s = add(
        add(
            add(scale(p[0], wt[0]), scale(p[1], wt[1])),
            scale(p[2], wt[2]),
        ),
        scale(p[3], wt[3]),
    );
    let s_len_sq = dot(s, s);
    let energy = 0.5 * bend_scale * s_len_sq;

    // grad_i = bend_scale * wt_i * S => |grad_i|^2 = (bend_scale*wt_i)^2 * s_len_sq.
    let g = [
        bend_scale * wt[0],
        bend_scale * wt[1],
        bend_scale * wt[2],
        bend_scale * wt[3],
    ];
    let term = |i: usize| -> f32 {
        if w[i] > 0.0 {
            w[i] * g[i] * g[i] * s_len_sq
        } else {
            0.0
        }
    };
    let sum_w_grad = term(0) + term(1) + term(2) + term(3);

    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let denom = sum_w_grad + alpha_tilde;

    let run = q.dt > 0.0 && s_len_sq > EPS_LEN_SQ && denom > 0.0;
    let safe_denom = if denom > 0.0 { denom } else { 1.0 };
    let d_lambda = -energy / safe_denom;

    let mut out = p;
    for (i, slot) in out.iter_mut().enumerate() {
        if run && w[i] > 0.0 {
            *slot = add(*slot, scale(s, w[i] * d_lambda * g[i]));
        }
    }
    (out, u32::from(run))
}

/// Asserts the device result matches the host oracle for one query: the
/// `valid` flag exactly and every continuous output within tolerance. A no-op
/// echoes the input positions, so the comparison also holds there.
fn assert_parity(
    gpu: &SoftIsometricBendingProjectResult,
    q: &SoftIsometricBendingProjectQuery,
    label: &str,
) {
    let (pos, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid mismatch query={q:?}");
    for corner in 0..4 {
        for axis in 0..3 {
            assert!(
                close(gpu.positions[corner][axis], pos[corner][axis]),
                "{label}: positions[{corner}][{axis}] mismatch gpu={} cpu={} query={q:?}",
                gpu.positions[corner][axis],
                pos[corner][axis]
            );
        }
    }
}

/// A non-planar bending stencil with a clearly non-zero weighted sum `S`, so a
/// projection runs. The weights sum to zero (a classic cotangent stencil), yet
/// the asymmetric positions keep `S` away from the flat knee.
fn bending_stencil() -> [[f32; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [0.5, 1.0, 0.0],
        [0.3, 0.4, 0.9],
    ]
}

#[test]
fn general_bend_projects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    let q = SoftIsometricBendingProjectQuery::new(
        bending_stencil(),
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        0.5,
        0.0,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].valid, 1, "non-planar stencil is projected");
    assert_parity(&r[0], &q, "general_bend_projects");
}

#[test]
fn compliant_bend_runs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    // Non-zero compliance exercises the alpha_tilde term alongside the mass sum.
    let q = SoftIsometricBendingProjectQuery::new(
        bending_stencil(),
        [1.0, 0.5, 2.0, 1.5],
        [1.0, -1.0, 1.0, -1.0],
        0.8,
        0.01,
        1.0 / 120.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "compliant bend runs");
    assert_parity(&r[0], &q, "compliant_bend_runs");
}

#[test]
fn flat_stencil_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    // All four corners coincide and the weights sum to zero, so S == 0 exactly
    // and s_len_sq <= EPS_LEN_SQ: the solver is a no-op and positions are
    // echoed unchanged.
    let q = SoftIsometricBendingProjectQuery::new(
        [[0.3, 0.3, 0.3]; 4],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        0.5,
        0.0,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "flat stencil is a no-op");
    assert_parity(&r[0], &q, "flat_stencil_is_no_op");
}

#[test]
fn non_positive_dt_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    // dt <= 0 => alpha_tilde undefined on the physics side => no-op branch; the
    // kernel echoes the input positions.
    let q = SoftIsometricBendingProjectQuery::new(
        bending_stencil(),
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        0.5,
        0.0,
        0.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "non-positive dt is a no-op");
    assert_parity(&r[0], &q, "non_positive_dt_is_no_op");
}

#[test]
fn pinned_vertex_stays() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    // Corner 0 is pinned (inverse mass 0): it must not move while the free
    // corners project. The remaining free masses keep denom > 0.
    let q = SoftIsometricBendingProjectQuery::new(
        bending_stencil(),
        [0.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        0.6,
        0.0,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(
        r[0].valid, 1,
        "stencil with one pinned corner still projects"
    );
    assert_parity(&r[0], &q, "pinned_vertex_stays");
    // The pinned corner is echoed unchanged.
    let p0 = bending_stencil()[0];
    for axis in 0..3 {
        assert!(
            close(r[0].positions[0][axis], p0[axis]),
            "pinned corner moved on axis {axis}: {} vs {}",
            r[0].positions[0][axis],
            p0[axis]
        );
    }
    // At least one free corner actually moved, so the fixture is meaningful.
    let moved = (1..4).any(|corner| {
        (0..3).any(|axis| {
            (r[0].positions[corner][axis] - bending_stencil()[corner][axis]).abs() > 1.0e-4
        })
    });
    assert!(moved, "no free corner moved; fixture is degenerate");
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    // Mixed batch exercises the std430 stride across projecting and no-op
    // branches at once.
    let queries = [
        SoftIsometricBendingProjectQuery::new(
            bending_stencil(),
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            0.5,
            0.0,
            1.0 / 60.0,
        ),
        SoftIsometricBendingProjectQuery::new(
            [[0.3, 0.3, 0.3]; 4],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            0.5,
            0.0,
            1.0 / 60.0,
        ),
        SoftIsometricBendingProjectQuery::new(
            [
                [-0.2, 0.1, 0.3],
                [0.9, -0.1, 0.2],
                [0.1, 1.1, -0.2],
                [0.2, 0.3, 1.2],
            ],
            [2.0, 1.0, 0.5, 1.5],
            [0.5, 0.5, -0.5, -0.5],
            0.9,
            0.02,
            1.0 / 90.0,
        ),
    ];
    let r = gpu.evaluate(&ctx, &queries);
    assert_eq!(r.len(), queries.len(), "one result per query");
    for (res, q) in r.iter().zip(queries.iter()) {
        assert_parity(res, q, "batch_of_two_or_more_validates_stride");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
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

/// Perturbs a query's stencil geometry slightly for conditioning checks.
fn perturb(q: &SoftIsometricBendingProjectQuery, delta: f32) -> SoftIsometricBendingProjectQuery {
    let mut positions = q.positions;
    for (corner, slot) in positions.iter_mut().enumerate() {
        let sign = if corner % 2 == 0 { 1.0 } else { -1.0 };
        slot[0] += delta * sign;
        slot[1] -= delta * sign;
        slot[2] += delta * sign;
    }
    SoftIsometricBendingProjectQuery::new(
        positions,
        q.inverse_masses,
        q.weights,
        q.scale,
        q.compliance,
        q.dt,
    )
}

/// Directly recomputes the two knee quantities `s_len_sq` and `denom` for the
/// margin check, in the kernel's accumulation order.
fn knees(q: &SoftIsometricBendingProjectQuery) -> (f32, f32) {
    let p = q.positions;
    let w = q.inverse_masses;
    let wt = q.weights;
    let bend_scale = q.scale;
    // Left-to-right accumulation matching the kernel's `p0..p3` order exactly.
    let s = add(
        add(
            add(scale(p[0], wt[0]), scale(p[1], wt[1])),
            scale(p[2], wt[2]),
        ),
        scale(p[3], wt[3]),
    );
    let s_len_sq = dot(s, s);
    let g = [
        bend_scale * wt[0],
        bend_scale * wt[1],
        bend_scale * wt[2],
        bend_scale * wt[3],
    ];
    let term = |i: usize| -> f32 {
        if w[i] > 0.0 {
            w[i] * g[i] * g[i] * s_len_sq
        } else {
            0.0
        }
    };
    let sum_w_grad = term(0) + term(1) + term(2) + term(3);
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    (s_len_sq, sum_w_grad + alpha_tilde)
}

/// Rejects marginal configurations where a tiny perturbation would push
/// `s_len_sq` toward the flat knee or `denom` toward zero and flip the `valid`
/// flag, so the branch choice agrees on both sides despite any last-bit
/// difference between host and device.
fn well_conditioned(q: &SoftIsometricBendingProjectQuery) -> bool {
    let (_, base_valid) = oracle(q);
    if base_valid != 1 {
        return false;
    }
    let (s_len_sq, denom) = knees(q);
    // Comfortable margins so neither no-op boundary is nearby.
    if s_len_sq < 1.0e-2 || denom < 1.0e-2 {
        return false;
    }
    for &delta in &[1.0e-3_f32, -1.0e-3] {
        let (_, valid) = oracle(&perturb(q, delta));
        if valid != base_valid {
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
    let gpu = GpuSoftIsometricBendingProject::new(&ctx);
    let mut rng = Lcg::new(0x51_3A_C4_17);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Offset the stencil from the origin with all-positive weights so the
        // weighted sum S stays comfortably away from the flat knee, and keep
        // every inverse mass positive so denom is well above zero.
        let base = [
            rng.next_range(0.4, 0.9),
            rng.next_range(0.4, 0.9),
            rng.next_range(0.4, 0.9),
        ];
        let mut positions = [[0.0f32; 3]; 4];
        for slot in positions.iter_mut() {
            slot[0] = base[0] + rng.next_range(-0.3, 0.3);
            slot[1] = base[1] + rng.next_range(-0.3, 0.3);
            slot[2] = base[2] + rng.next_range(-0.3, 0.3);
        }
        let inverse_masses = [
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
        ];
        let weights = [
            rng.next_range(0.4, 1.5),
            rng.next_range(0.4, 1.5),
            rng.next_range(0.4, 1.5),
            rng.next_range(0.4, 1.5),
        ];
        let bend_scale = rng.next_range(0.3, 1.5);
        let compliance = rng.next_range(0.0, 0.05);
        let dt = rng.next_range(1.0 / 240.0, 1.0 / 30.0);
        let q = SoftIsometricBendingProjectQuery::new(
            positions,
            inverse_masses,
            weights,
            bend_scale,
            compliance,
            dt,
        );
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    let mut valid_count = 0u32;
    for (q, r) in queries.iter().zip(results.iter()) {
        let (_, valid) = oracle(q);
        if valid == 1 {
            valid_count += 1;
        }
        assert_parity(r, q, "random_sweep_matches_oracle");
    }
    // The sweep is constructed to project every stencil; guard against a
    // degenerate sweep that would weakly test parity.
    assert!(valid_count > 0, "sweep produced no projections");
}
