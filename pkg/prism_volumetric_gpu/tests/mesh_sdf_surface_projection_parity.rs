//! Real-device parity for the signed-distance-field surface-*projection* twin:
//! [`GpuSdfSurfaceProjection`](prism_volumetric_gpu::mesh_sdf_surface_projection::GpuSdfSurfaceProjection)
//! must reproduce the `CPU` golden `project_to_surface` of
//! `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`, which
//! composes the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` with the
//! central-difference normal `sdf_normal` of
//! `prism_render_architecture::ray_scene::mesh_sdf_normal` and iterates the
//! Newton step `point -= normal * signed_distance` until the unsigned signed
//! distance drops to or below the tolerance or the iteration budget is spent.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, central-difference gradient, guarded normalization and
//! the Newton loop — written directly against a flat per-cell signed-distance
//! array so the test does not import `prism_render_architecture`. It operates on
//! the same field layout the twin consumes (`x` fastest, `clamp-to-border`
//! addressing) and tracks every value in `f32` so it follows the `GPU`
//! trajectory step for step.
//!
//! The fixtures cover the degenerate and boundary shapes the kernel must honor:
//! an already-on-surface start (huge tolerance, `iterations = 0`), a converging
//! interior point on a voxelized sphere field, a constant field (vanishing
//! gradient, `valid = 0`, zeroed result on both sides), a one-step budget that
//! runs out (`valid = 1`, `iterations = 1`), a sweep over random sphere fields
//! and start points, and an empty query batch (the host short-circuits with no
//! dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Sampling and the gradient are a fixed sequence of linear-interpolation adds,
//! a constant scale and one `sqrt` per iteration, so `CPU` and `GPU` evaluate
//! the same closed form but need not be bit-exact (a `GPU` may contract a
//! multiply-add). The continuous comparison on the projected point and the
//! residual is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`). The
//! `iterations` count and the `valid` flag are integer decisions and match
//! exactly; the random sweep rejects start points whose path grazes the
//! tolerance band or the zero-gradient guard so neither discrete decision flips
//! between the two sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_surface_projection::{
    GpuSdfSurfaceProjection, SdfSurfaceProjectionField, SdfSurfaceProjectionQuery,
    SdfSurfaceProjectionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Linear interpolation between `a` and `b` by `s`, matching the golden `lerp`.
fn lerp(a: f32, b: f32, s: f32) -> f32 {
    a + (b - a) * s
}

/// Independent oracle re-implementing the golden trilinear
/// `sample_signed_distance` against a flat per-cell signed-distance array.
fn sample(dims: [u32; 3], origin: [f32; 3], voxel: f32, dist: &[f32], point: [f32; 3]) -> f32 {
    let mut base = [0u32; 3];
    let mut frac = [0f32; 3];
    for (axis, slot) in base.iter_mut().enumerate() {
        if dims[axis] <= 1 {
            frac[axis] = 0.0;
            continue;
        }
        let last = (dims[axis] - 1) as f32;
        let continuous = (point[axis] - origin[axis]) / voxel - 0.5;
        let clamped = continuous.clamp(0.0, last);
        let lower = clamped.floor().clamp(0.0, last - 1.0);
        *slot = lower as u32;
        frac[axis] = clamped - lower;
    }
    let corner = |dx: u32, dy: u32, dz: u32| -> f32 {
        let x = (base[0] + dx).min(dims[0] - 1);
        let y = (base[1] + dy).min(dims[1] - 1);
        let z = (base[2] + dz).min(dims[2] - 1);
        let idx = ((z * dims[1] + y) * dims[0] + x) as usize;
        dist[idx]
    };
    let d000 = corner(0, 0, 0);
    let d100 = corner(1, 0, 0);
    let d010 = corner(0, 1, 0);
    let d110 = corner(1, 1, 0);
    let d001 = corner(0, 0, 1);
    let d101 = corner(1, 0, 1);
    let d011 = corner(0, 1, 1);
    let d111 = corner(1, 1, 1);
    let c00 = lerp(d000, d100, frac[0]);
    let c01 = lerp(d001, d101, frac[0]);
    let c10 = lerp(d010, d110, frac[0]);
    let c11 = lerp(d011, d111, frac[0]);
    let c0 = lerp(c00, c10, frac[1]);
    let c1 = lerp(c01, c11, frac[1]);
    lerp(c0, c1, frac[2])
}

/// Independent oracle re-implementing the golden `sdf_gradient` central
/// difference against a flat per-cell signed-distance array.
fn gradient(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> [f32; 3] {
    let inv = 1.0 / (2.0 * voxel);
    let mut g = [0f32; 3];
    for (axis, slot) in g.iter_mut().enumerate() {
        let mut forward = point;
        let mut backward = point;
        forward[axis] += voxel;
        backward[axis] -= voxel;
        let diff = sample(dims, origin, voxel, dist, forward)
            - sample(dims, origin, voxel, dist, backward);
        *slot = diff * inv;
    }
    g
}

/// Diagnostics collected along the oracle projection path, used only by the
/// random sweep to reject start points whose path grazes a discrete decision
/// boundary (so neither the iteration count nor the `valid` flag can flip
/// between the two sides of the parity comparison).
struct ProjectDiag {
    /// Minimum of `|abs(signed) - tolerance|` over every convergence check.
    min_tol_margin: f32,
    /// Minimum gradient length squared over every non-converged iteration.
    min_len2: f32,
}

/// Independent oracle for the golden `project_to_surface`, mirroring the kernel
/// step for step in `f32`. Returns the projection result plus path diagnostics.
///
/// Convergence wins even when the gradient would be degenerate: the unsigned
/// distance is tested before the gradient. A mid-loop vanishing gradient stops
/// with `valid = 0` and a zeroed result (the golden `None`); running out of
/// iterations returns the best-effort point with `valid = 1` and
/// `iterations = max_iterations`.
fn project(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
    max_iterations: u32,
    tolerance: f32,
) -> (SdfSurfaceProjectionResult, ProjectDiag) {
    let mut cur = point;
    let min_positive = f32::MIN_POSITIVE;
    let mut min_tol_margin = f32::INFINITY;
    let mut min_len2 = f32::INFINITY;

    let mut result = SdfSurfaceProjectionResult {
        point: [0.0, 0.0, 0.0],
        residual: 0.0,
        iterations: 0,
        valid: 0,
    };
    let mut done = false;
    let mut i = 0u32;
    while i < max_iterations {
        let signed = sample(dims, origin, voxel, dist, cur);
        let abs_signed = signed.abs();
        min_tol_margin = min_tol_margin.min((abs_signed - tolerance).abs());
        if abs_signed <= tolerance {
            result = SdfSurfaceProjectionResult {
                point: cur,
                residual: abs_signed,
                iterations: i,
                valid: 1,
            };
            done = true;
            break;
        }
        let g = gradient(dims, origin, voxel, dist, cur);
        let len2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        min_len2 = min_len2.min(len2);
        if len2 <= min_positive {
            // Gradient vanished before convergence: no direction to step, so
            // the result stays the zeroed, invalid default (golden `None`).
            done = true;
            break;
        }
        let len = len2.sqrt();
        cur = [
            cur[0] - (g[0] / len) * signed,
            cur[1] - (g[1] / len) * signed,
            cur[2] - (g[2] / len) * signed,
        ];
        i += 1;
    }
    if !done {
        let residual = sample(dims, origin, voxel, dist, cur).abs();
        result = SdfSurfaceProjectionResult {
            point: cur,
            residual,
            iterations: max_iterations,
            valid: 1,
        };
    }
    (
        result,
        ProjectDiag {
            min_tol_margin,
            min_len2,
        },
    )
}

/// Asserts the `GPU` results match the oracle: the projected point and residual
/// within the continuous tolerance, the iteration count and the `valid` flag
/// exactly.
fn assert_parity(
    gpu: &[SdfSurfaceProjectionResult],
    field: &SdfSurfaceProjectionField,
    queries: &[SdfSurfaceProjectionQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let (want, _) = project(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.point,
            q.max_iterations,
            q.tolerance,
        );
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
        assert_eq!(
            g.iterations, want.iterations,
            "{label}: query {i} iteration count GPU {} vs CPU {}",
            g.iterations, want.iterations
        );
        assert!(
            close(g.residual, want.residual),
            "{label}: query {i} residual GPU {} vs CPU {}",
            g.residual,
            want.residual
        );
        for k in 0..3 {
            assert!(
                close(g.point[k], want.point[k]),
                "{label}: query {i} point[{k}] GPU {} vs CPU {}",
                g.point[k],
                want.point[k]
            );
        }
    }
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

/// Builds a voxelized sphere field: each cell stores the Euclidean distance
/// from its world-space center to `center` minus `radius`, giving an oriented,
/// well-conditioned gradient that the Newton step converges on quickly.
fn sphere_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    center: [f32; 3],
    radius: f32,
) -> SdfSurfaceProjectionField {
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * voxel;
                let wy = origin[1] + y as f32 * voxel;
                let wz = origin[2] + z as f32 * voxel;
                let cx = wx - center[0];
                let cy = wy - center[1];
                let cz = wz - center[2];
                let r = (cx * cx + cy * cy + cz * cz).sqrt();
                dist.push(r - radius);
            }
        }
    }
    SdfSurfaceProjectionField::new(dims, origin, voxel, dist)
}

#[test]
fn already_on_surface_returns_zero_iterations() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([9, 9, 9], [-1.0, -1.0, -1.0], 0.25, [0.0, 0.0, 0.0], 0.5);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    // A huge tolerance makes the very first sample converge, so the Newton loop
    // returns at iteration zero with the start point unchanged.
    let queries = vec![
        SdfSurfaceProjectionQuery::new([0.3, -0.2, 0.1], 32, 1.0e6),
        SdfSurfaceProjectionQuery::new([-0.4, 0.25, 0.35], 32, 1.0e6),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "already_on_surface");
    for r in &out {
        assert_eq!(r.valid, 1, "huge tolerance must converge immediately");
        assert_eq!(r.iterations, 0, "no Newton step should be taken");
    }
}

#[test]
fn converges_on_sphere_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([9, 9, 9], [-1.0, -1.0, -1.0], 0.25, [0.0, 0.0, 0.0], 0.5);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    let tol = 0.02;
    let queries = vec![
        SdfSurfaceProjectionQuery::new([0.2, 0.1, -0.05], 32, tol),
        SdfSurfaceProjectionQuery::new([-0.3, 0.2, 0.15], 32, tol),
        SdfSurfaceProjectionQuery::new([0.35, -0.25, 0.3], 32, tol),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "sphere_converge");
    for r in &out {
        assert_eq!(r.valid, 1, "interior start must project onto the surface");
        assert!(
            r.residual <= tol,
            "residual {} must reach tolerance",
            r.residual
        );
    }
}

#[test]
fn constant_field_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [4u32, 4, 4];
    let dist = vec![1.5f32; 64];
    let field = SdfSurfaceProjectionField::new(dims, [-1.0, -1.0, -1.0], 0.5, dist);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    // The non-converging first iteration finds a zero gradient, so the twin
    // stops with the zeroed, invalid result (golden `None`).
    let queries = vec![
        SdfSurfaceProjectionQuery::new([0.0, 0.0, 0.0], 16, 0.02),
        SdfSurfaceProjectionQuery::new([-0.3, 0.2, 0.1], 16, 0.02),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "constant");
    for r in &out {
        assert_eq!(r.valid, 0, "constant field vanishes the gradient");
        assert_eq!(r.point, [0.0, 0.0, 0.0]);
        assert_eq!(r.iterations, 0);
    }
}

#[test]
fn single_step_budget_runs_out() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([9, 9, 9], [-1.0, -1.0, -1.0], 0.25, [0.0, 0.0, 0.0], 0.5);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    // A tight tolerance with a one-step budget cannot converge on the first
    // sample, so the loop runs out and returns the best-effort point with the
    // iteration count pinned to the budget.
    let queries = vec![
        SdfSurfaceProjectionQuery::new([0.9, 0.0, 0.0], 1, 1.0e-6),
        SdfSurfaceProjectionQuery::new([0.0, -0.85, 0.05], 1, 1.0e-6),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "single_step");
    for r in &out {
        assert_eq!(r.valid, 1, "ran-out return is still best-effort valid");
        assert_eq!(r.iterations, 1, "iteration count pins to the budget");
    }
}

#[test]
fn zero_iteration_budget_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([9, 9, 9], [-1.0, -1.0, -1.0], 0.25, [0.0, 0.0, 0.0], 0.5);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    // An empty loop returns the start point unchanged with the tail residual.
    let queries = vec![SdfSurfaceProjectionQuery::new([0.2, -0.1, 0.3], 0, 0.02)];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "zero_budget");
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].iterations, 0);
    for k in 0..3 {
        assert!(close(out[0].point[k], queries[0].point[k]));
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    let mut rng = Lcg::new(0x2BD1_77A3);
    let max_it = 32u32;
    let tol = 0.02f32;
    // Margins that keep every discrete decision clear of its boundary: the
    // tolerance band for the convergence test and the gradient length for the
    // zero-gradient guard. Both comfortably exceed any `CPU`/`GPU` drift.
    let tol_margin = 0.005f32;
    let len2_floor = 0.25f32;
    let mut built = 0u32;
    let mut attempts = 0u32;
    while built < 512 && attempts < 100_000 {
        attempts += 1;
        // A sphere field whose surface sits well inside the field extent.
        let dims = [9u32, 9, 9];
        let origin = [-1.0f32, -1.0, -1.0];
        let voxel = 0.25f32;
        let center = [
            rng.next_range(-0.1, 0.1),
            rng.next_range(-0.1, 0.1),
            rng.next_range(-0.1, 0.1),
        ];
        let radius = rng.next_range(0.4, 0.6);
        let field = sphere_field(dims, origin, voxel, center, radius);

        // A random start along a random direction from the center, kept inside
        // the field extent. The direction is normalized with `sqrt` only.
        let dir = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let dlen2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if dlen2 < 0.1 {
            continue;
        }
        let dlen = dlen2.sqrt();
        let span = rng.next_range(0.15, 0.8);
        let start = [
            center[0] + dir[0] / dlen * span,
            center[1] + dir[1] / dlen * span,
            center[2] + dir[2] / dlen * span,
        ];

        let (want, diag) = project(dims, origin, voxel, field.distances(), start, max_it, tol);
        // Only accept well-conditioned, genuinely converged paths that stay
        // clear of both discrete-decision boundaries.
        if want.valid != 1 || want.iterations >= max_it {
            continue;
        }
        if diag.min_tol_margin < tol_margin || diag.min_len2 < len2_floor {
            continue;
        }

        let queries = [SdfSurfaceProjectionQuery::new(start, max_it, tol)];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        built += 1;
    }
    assert_eq!(built, 512, "sweep should build the full sample budget");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([9, 9, 9], [-1.0, -1.0, -1.0], 0.25, [0.0, 0.0, 0.0], 0.5);
    let gpu = GpuSdfSurfaceProjection::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
