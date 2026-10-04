//! Real-device parity for the tetrahedral volume-projection twin:
//! [`GpuSoftTetraVolumeProject`](prism_volumetric_gpu::soft_tetra_volume_project::GpuSoftTetraVolumeProject)
//! must reproduce the `CPU` golden `TetraVolumeConstraint::project` of
//! `prism_physics_core::soft::constraint::volume`. For each tetrahedron the
//! reference drives the signed volume back toward a rest value with a single
//! compliant `XPBD` step, moving each free corner along the volume gradient.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the cross-product gradients, the effective-mass sum `denom_mass`, the
//! `denom_mass <= 0` no-op guard, the signed volume, the `delta_lambda`
//! solution and the per-corner position update — written out directly so the
//! test never imports `prism_render_architecture` or `prism_physics_core`. It
//! mirrors the reference branch for branch, so a passing comparison is evidence
//! the ported kernel took the same degenerate / projecting branch, not merely
//! that the shader compiled.
//!
//! The fixtures cover each branch the kernel must honor: a non-degenerate
//! tetrahedron whose volume is projected toward rest, a fully pinned cell whose
//! zero inverse masses make `denom_mass == 0` a no-op, a collapsed coplanar
//! shape with vanishing gradients that is likewise a no-op, a batch of two or
//! more elements that validates the `std430` stride, and an empty batch the
//! host short-circuits with no dispatch. A sweep over random non-degenerate
//! tetrahedra follows, rejecting marginal geometry (where a tiny perturbation
//! would collapse `denom_mass` toward zero and flip the `valid` flag) so the
//! branch choice agrees on both sides despite any last-bit difference.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every path threads through multiplies, adds, cross products, dot products
//! and a guarded division, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact (a `GPU` may contract a multiply-add), and the
//! gradient magnitudes can be large so the relative test matters. Each
//! continuous output is compared with `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::volume`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_tetra_volume_project::{
    GpuSoftTetraVolumeProject, SoftTetraVolumeProjectQuery,
};
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
    let scale_ref = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale_ref <= REL_EPS
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Vector difference `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Vector sum `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a vector.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Independent host re-implementation of `TetraVolumeConstraint::project` for a
/// single tetrahedron. Returns the four updated positions, the updated
/// multiplier and the discrete validity flag, replicated branch for branch.
fn oracle(q: &SoftTetraVolumeProjectQuery) -> ([[f32; 3]; 4], f32, u32) {
    let p0 = q.positions[0];
    let p1 = q.positions[1];
    let p2 = q.positions[2];
    let p3 = q.positions[3];
    let w = q.inverse_masses;

    let e1 = sub(p1, p0);
    let e2 = sub(p2, p0);
    let e3 = sub(p3, p0);
    let grad1 = scale(cross(e2, e3), 1.0 / 6.0);
    let grad2 = scale(cross(e3, e1), 1.0 / 6.0);
    let grad3 = scale(cross(e1, e2), 1.0 / 6.0);
    let grad0 = scale(add(add(grad1, grad2), grad3), -1.0);

    let denom_mass = w[0] * dot(grad0, grad0)
        + w[1] * dot(grad1, grad1)
        + w[2] * dot(grad2, grad2)
        + w[3] * dot(grad3, grad3);
    if denom_mass <= 0.0 {
        return ([p0, p1, p2, p3], q.lambda, 0);
    }

    let volume = dot(e1, cross(e2, e3)) / 6.0;
    let c = volume - q.rest_volume;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let delta_lambda = (-c - alpha_tilde * q.lambda) / (denom_mass + alpha_tilde);
    let new_lambda = q.lambda + delta_lambda;

    let np0 = add(p0, scale(grad0, delta_lambda * w[0]));
    let np1 = add(p1, scale(grad1, delta_lambda * w[1]));
    let np2 = add(p2, scale(grad2, delta_lambda * w[2]));
    let np3 = add(p3, scale(grad3, delta_lambda * w[3]));
    ([np0, np1, np2, np3], new_lambda, 1)
}

/// Asserts the device result matches the host oracle for one query: the
/// `valid` flag exactly and every continuous output within tolerance. A no-op
/// echoes the input positions and lambda, so the comparison also holds there.
fn assert_parity(
    gpu: &prism_volumetric_gpu::soft_tetra_volume_project::SoftTetraVolumeProjectResult,
    q: &SoftTetraVolumeProjectQuery,
    label: &str,
) {
    let (pos, lambda, valid) = oracle(q);
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
    assert!(
        close(gpu.new_lambda, lambda),
        "{label}: new_lambda mismatch gpu={} cpu={} query={q:?}",
        gpu.new_lambda,
        lambda
    );
}

/// A standard non-degenerate corner tetrahedron with a small perturbation so
/// its volume differs from the rest value.
fn unit_tetra() -> [[f32; 3]; 4] {
    [
        [0.05, -0.02, 0.03],
        [1.1, 0.0, 0.0],
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 1.05],
    ]
}

#[test]
fn non_degenerate_projects_volume() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    // Free corners with a rest volume different from the current volume, so a
    // non-zero projection runs.
    let q = SoftTetraVolumeProjectQuery::new(
        unit_tetra(),
        [1.0, 1.0, 1.0, 1.0],
        0.1,
        0.0,
        0.0,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].valid, 1, "non-degenerate tetrahedron is projected");
    assert_parity(&r[0], &q, "non_degenerate_projects_volume");
}

#[test]
fn compliant_projection_runs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    // A non-zero compliance and a seeded lambda exercise the alpha_tilde term.
    let q = SoftTetraVolumeProjectQuery::new(
        unit_tetra(),
        [1.0, 0.5, 2.0, 1.5],
        0.15,
        0.01,
        0.3,
        1.0 / 120.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "compliant projection runs");
    assert_parity(&r[0], &q, "compliant_projection_runs");
}

#[test]
fn fully_pinned_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    // All inverse masses zero => denom_mass == 0 => no-op, positions and lambda
    // unchanged.
    let q = SoftTetraVolumeProjectQuery::new(
        unit_tetra(),
        [0.0, 0.0, 0.0, 0.0],
        0.1,
        0.0,
        0.7,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "fully pinned tetrahedron is a no-op");
    assert_parity(&r[0], &q, "fully_pinned_is_no_op");
}

#[test]
fn collinear_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    // All four corners are collinear on the x axis, so every edge is parallel:
    // all volume gradients (edge cross-products) vanish and denom_mass == 0 even
    // with free masses, so the solver is a no-op. (A merely coplanar tetrahedron
    // still has non-zero gradients and would project, so it is not degenerate.)
    let q = SoftTetraVolumeProjectQuery::new(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        [1.0, 1.0, 1.0, 1.0],
        0.1,
        0.0,
        0.2,
        1.0 / 60.0,
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "collinear tetrahedron is a no-op");
    assert_parity(&r[0], &q, "collinear_is_no_op");
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    // Mixed batch exercises the std430 stride across both branches at once.
    let queries = [
        SoftTetraVolumeProjectQuery::new(
            unit_tetra(),
            [1.0, 1.0, 1.0, 1.0],
            0.1,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftTetraVolumeProjectQuery::new(
            unit_tetra(),
            [0.0, 0.0, 0.0, 0.0],
            0.1,
            0.0,
            0.4,
            1.0 / 60.0,
        ),
        SoftTetraVolumeProjectQuery::new(
            [
                [-0.3, 0.2, 0.1],
                [0.8, -0.1, 0.2],
                [0.1, 1.2, -0.2],
                [0.2, 0.1, 1.3],
            ],
            [2.0, 1.0, 0.5, 1.5],
            0.2,
            0.02,
            0.1,
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
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
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

/// Perturbs a query's corner geometry slightly for conditioning checks.
fn perturb(q: &SoftTetraVolumeProjectQuery, delta: f32) -> SoftTetraVolumeProjectQuery {
    let mut positions = q.positions;
    for (corner, slot) in positions.iter_mut().enumerate() {
        let sign = if corner % 2 == 0 { 1.0 } else { -1.0 };
        slot[0] += delta * sign;
        slot[1] -= delta * sign;
        slot[2] += delta * sign;
    }
    SoftTetraVolumeProjectQuery::new(
        positions,
        q.inverse_masses,
        q.rest_volume,
        q.compliance,
        q.lambda,
        q.dt,
    )
}

/// Rejects marginal configurations where a tiny perturbation would collapse
/// `denom_mass` toward zero and flip the `valid` flag, so the branch choice
/// agrees on both sides despite any last-bit difference between host and
/// device. Also requires a comfortable `denom_mass` margin directly.
fn well_conditioned(q: &SoftTetraVolumeProjectQuery) -> bool {
    let (_, _, base_valid) = oracle(q);
    if base_valid != 1 {
        return false;
    }
    // Direct denom_mass margin so the no-op boundary is far away.
    let p0 = q.positions[0];
    let e1 = sub(q.positions[1], p0);
    let e2 = sub(q.positions[2], p0);
    let e3 = sub(q.positions[3], p0);
    let grad1 = scale(cross(e2, e3), 1.0 / 6.0);
    let grad2 = scale(cross(e3, e1), 1.0 / 6.0);
    let grad3 = scale(cross(e1, e2), 1.0 / 6.0);
    let grad0 = scale(add(add(grad1, grad2), grad3), -1.0);
    let w = q.inverse_masses;
    let denom = w[0] * dot(grad0, grad0)
        + w[1] * dot(grad1, grad1)
        + w[2] * dot(grad2, grad2)
        + w[3] * dot(grad3, grad3);
    if denom < 1.0e-2 {
        return false;
    }
    for &delta in &[1.0e-3_f32, -1.0e-3] {
        let (_, _, valid) = oracle(&perturb(q, delta));
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
    let gpu = GpuSoftTetraVolumeProject::new(&ctx);
    let mut rng = Lcg::new(0x2F_71_B6_05);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Start from a well-shaped corner tetrahedron and jitter each corner so
        // the cell stays non-degenerate (denom_mass well above zero).
        let positions = [
            [
                rng.next_range(-0.2, 0.2),
                rng.next_range(-0.2, 0.2),
                rng.next_range(-0.2, 0.2),
            ],
            [
                rng.next_range(0.8, 1.3),
                rng.next_range(-0.2, 0.2),
                rng.next_range(-0.2, 0.2),
            ],
            [
                rng.next_range(-0.2, 0.2),
                rng.next_range(0.8, 1.3),
                rng.next_range(-0.2, 0.2),
            ],
            [
                rng.next_range(-0.2, 0.2),
                rng.next_range(-0.2, 0.2),
                rng.next_range(0.8, 1.3),
            ],
        ];
        let inverse_masses = [
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
            rng.next_range(0.3, 2.0),
        ];
        let rest_volume = rng.next_range(0.0, 0.4);
        let compliance = rng.next_range(0.0, 0.05);
        let lambda = rng.next_range(-0.5, 0.5);
        let dt = rng.next_range(1.0 / 240.0, 1.0 / 30.0);
        let q = SoftTetraVolumeProjectQuery::new(
            positions,
            inverse_masses,
            rest_volume,
            compliance,
            lambda,
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
        let (_, _, valid) = oracle(q);
        if valid == 1 {
            valid_count += 1;
        }
        assert_parity(r, q, "random_sweep_matches_oracle");
    }
    // The sweep is constructed to project every cell; guard against a
    // degenerate sweep that would weakly test parity.
    assert!(valid_count > 0, "sweep produced no projections");
}
