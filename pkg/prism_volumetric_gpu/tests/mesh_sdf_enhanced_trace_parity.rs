//! Real-device parity for the enhanced (over-relaxed) sphere-tracing twin:
//! [`GpuSdfEnhancedTrace`](prism_volumetric_gpu::mesh_sdf_enhanced_trace::GpuSdfEnhancedTrace)
//! must reproduce the `CPU` golden `enhanced_sphere_trace` of
//! `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`, which
//! marches a ray through a signed distance field with the Keinert et al. (2014)
//! over-relaxation scheme on top of the trilinear sampler
//! `sample_signed_distance`, the direction `normalize` and the slab clip
//! `ray_aabb` of `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `normalize`, slab `ray_aabb`, trilinear sampling and the over-relaxed march
//! loop — written directly against a flat per-cell signed-distance array so the
//! test does not import `prism_render_architecture`. It operates on the same
//! field layout the twin consumes (`x` fastest, `clamp-to-border` addressing)
//! and steps `f32`-for-`f32` in the same operation order as the kernel so the
//! discrete outputs match exactly.
//!
//! The fixtures cover the shapes the kernel must honor: a normal-incidence ray
//! into a padded cube field (converges with no over-relaxation back-offs), an
//! oblique over-relaxed ray, a degenerate zero-length direction (a miss), a ray
//! aimed away from the box (a miss), a sweep over random fields and rays with a
//! branch-boundary rejection filter, and an empty query batch (the host
//! short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The march is iterative and its over-relaxation back-off branch is
//! `f32`-sensitive, so the discrete outputs `hit`, `steps` and
//! `relaxation_resets` are asserted with exact integer equality while the
//! continuous outputs `t`, `position` and `distance` use
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`), since a `GPU`
//! may contract a multiply-add. The random sweep rejects any ray whose march
//! takes a branch decision within a guard margin of its boundary, so the
//! discrete counts never flip between the two sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_enhanced_trace::{
    GpuSdfEnhancedTrace, SdfEnhancedTraceField, SdfEnhancedTraceQuery, SdfEnhancedTraceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Rejection margin for the random sweep: a ray is skipped when any branch
/// decision along its march sits within this many units of its boundary, so a
/// `GPU` multiply-add contraction can never flip a discrete count.
const BRANCH_GUARD: f32 = 1.0e-3;

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

/// Independent oracle for the golden `normalize`: returns `None` when the
/// direction is too short to normalize, matching the kernel's miss guard.
fn normalize(d: [f32; 3]) -> Option<[f32; 3]> {
    let ls = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if ls <= f32::MIN_POSITIVE {
        return None;
    }
    let len = ls.sqrt();
    Some([d[0] / len, d[1] / len, d[2] / len])
}

/// The outcome of a march, mirroring [`SdfEnhancedTraceResult`] plus a
/// well-conditioned flag used only by the random sweep's rejection filter.
struct Outcome {
    result: SdfEnhancedTraceResult,
    conditioned: bool,
}

/// A clean miss with every output field zeroed.
fn miss(conditioned: bool) -> Outcome {
    Outcome {
        result: SdfEnhancedTraceResult {
            hit: 0,
            t: 0.0,
            position: [0.0, 0.0, 0.0],
            distance: 0.0,
            steps: 0,
            relaxation_resets: 0,
        },
        conditioned,
    }
}

/// Independent oracle re-implementing the golden `enhanced_sphere_trace`,
/// stepping `f32`-for-`f32` in the kernel's operation order. The returned
/// `conditioned` flag is `false` when any branch decision sits within
/// [`BRANCH_GUARD`] of its boundary, so the random sweep can reject fixtures
/// that might flip a discrete count under multiply-add contraction.
fn trace(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    q: &SdfEnhancedTraceQuery,
) -> Outcome {
    let mut conditioned = true;

    let Some(dir) = normalize(q.direction) else {
        return miss(true);
    };

    let grid_max = [
        origin[0] + dims[0] as f32 * voxel,
        origin[1] + dims[1] as f32 * voxel,
        origin[2] + dims[2] as f32 * voxel,
    ];

    // Slab clip against the field bounding box, threading (t0, t1) across axes.
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for axis in 0..3 {
        if dir[axis].abs() <= f32::MIN_POSITIVE {
            if q.origin[axis] < origin[axis] || q.origin[axis] > grid_max[axis] {
                return miss(conditioned);
            }
            continue;
        }
        let inv = 1.0 / dir[axis];
        let mut ta = (origin[axis] - q.origin[axis]) * inv;
        let mut tb = (grid_max[axis] - q.origin[axis]) * inv;
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if (t1 - t0).abs() < BRANCH_GUARD {
            conditioned = false;
        }
        if t0 > t1 {
            return miss(conditioned);
        }
    }

    let t_enter = t0;
    let t_exit = t1.min(q.max_distance);
    if (t_enter - t_exit).abs() < BRANCH_GUARD {
        conditioned = false;
    }
    if t_enter > t_exit {
        return miss(conditioned);
    }

    let omega = q.over_relaxation.clamp(1.0, 1.999_999);

    let entry_pos = [
        q.origin[0] + t_enter * dir[0],
        q.origin[1] + t_enter * dir[1],
        q.origin[2] + t_enter * dir[2],
    ];
    let entry_sample = sample(dims, origin, voxel, dist, entry_pos);
    if entry_sample.abs() < BRANCH_GUARD {
        conditioned = false;
    }
    let sign0 = if entry_sample < 0.0 { -1.0 } else { 1.0 };

    let mut t = t_enter;
    let mut previous_radius = 0.0f32;
    let mut step_length = 0.0f32;
    let mut omega_cur = omega;
    let mut relaxation_resets = 0u32;

    for step in 0..q.max_steps {
        let position = [
            q.origin[0] + t * dir[0],
            q.origin[1] + t * dir[1],
            q.origin[2] + t * dir[2],
        ];
        let signed = sign0 * sample(dims, origin, voxel, dist, position);
        let radius = signed.abs();

        let sor_failed = omega_cur > 1.0 && (radius + previous_radius) < step_length;
        if ((radius + previous_radius) - step_length).abs() < BRANCH_GUARD {
            conditioned = false;
        }
        if sor_failed {
            step_length -= omega_cur * step_length;
            omega_cur = 1.0;
            relaxation_resets += 1;
        } else {
            step_length = signed * omega_cur;
        }

        previous_radius = radius;

        if (radius - q.pixel_radius * t).abs() < BRANCH_GUARD {
            conditioned = false;
        }
        if !sor_failed && radius < q.pixel_radius * t {
            return Outcome {
                result: SdfEnhancedTraceResult {
                    hit: 1,
                    t,
                    position,
                    distance: sign0 * signed,
                    steps: step + 1,
                    relaxation_resets,
                },
                conditioned,
            };
        }

        t += step_length;
        if (t - t_exit).abs() < BRANCH_GUARD {
            conditioned = false;
        }
        if t > t_exit {
            return miss(conditioned);
        }
    }

    miss(conditioned)
}

/// Convenience wrapper returning just the oracle result.
fn oracle(field: &SdfEnhancedTraceField, q: &SdfEnhancedTraceQuery) -> SdfEnhancedTraceResult {
    trace(
        field.dims(),
        field.origin(),
        field.voxel_size(),
        field.distances(),
        q,
    )
    .result
}

/// Asserts the `GPU` results match the oracle: discrete counts exactly, the
/// continuous quantities within the module tolerance.
fn assert_parity(
    gpu: &[SdfEnhancedTraceResult],
    field: &SdfEnhancedTraceField,
    queries: &[SdfEnhancedTraceQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = oracle(field, q);
        assert_eq!(g.hit, want.hit, "{label}: query {i} hit flag");
        assert_eq!(g.steps, want.steps, "{label}: query {i} steps");
        assert_eq!(
            g.relaxation_resets, want.relaxation_resets,
            "{label}: query {i} relaxation_resets"
        );
        assert!(
            close(g.t, want.t),
            "{label}: query {i} t GPU {} vs CPU {}",
            g.t,
            want.t
        );
        for k in 0..3 {
            assert!(
                close(g.position[k], want.position[k]),
                "{label}: query {i} position[{k}] GPU {} vs CPU {}",
                g.position[k],
                want.position[k]
            );
        }
        assert!(
            close(g.distance, want.distance),
            "{label}: query {i} distance GPU {} vs CPU {}",
            g.distance,
            want.distance
        );
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

/// Analytic signed distance to the axis-aligned cube `[0, 1]^3`.
fn cube_sdf(point: [f32; 3]) -> f32 {
    let center = [0.5f32, 0.5, 0.5];
    let half = [0.5f32, 0.5, 0.5];
    let q = [
        (point[0] - center[0]).abs() - half[0],
        (point[1] - center[1]).abs() - half[1],
        (point[2] - center[2]).abs() - half[2],
    ];
    let outside = (q[0].max(0.0) * q[0].max(0.0)
        + q[1].max(0.0) * q[1].max(0.0)
        + q[2].max(0.0) * q[2].max(0.0))
    .sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

/// Builds a padded signed-distance field sampling the analytic cube `[0, 1]^3`
/// over `[-1, 3]^3`, with room above the cube for a descending ray to march.
fn padded_cube_field() -> SdfEnhancedTraceField {
    let dims = [17u32, 17, 17];
    let origin = [-1.0f32, -1.0, -1.0];
    let voxel = 0.25f32;
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let p = [
                    origin[0] + x as f32 * voxel,
                    origin[1] + y as f32 * voxel,
                    origin[2] + z as f32 * voxel,
                ];
                dist.push(cube_sdf(p));
            }
        }
    }
    SdfEnhancedTraceField::new(dims, origin, voxel, dist)
}

#[test]
fn normal_incidence_converges_without_resets() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = padded_cube_field();
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    // Straight-down ray onto the top face of the cube (z = 1); conservative
    // stepping (over_relaxation = 1.0) never backs off.
    let queries = vec![SdfEnhancedTraceQuery::new(
        [0.5, 0.5, 3.0],
        [0.0, 0.0, -1.0],
        10.0,
        1.0e-3,
        256,
        1.0,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "normal_incidence");
    let want = oracle(&field, &queries[0]);
    assert_eq!(want.hit, 1, "ray must converge on the cube");
    assert_eq!(
        want.relaxation_resets, 0,
        "conservative stepping takes no over-relaxation back-offs"
    );
}

#[test]
fn oblique_over_relaxed_ray_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = padded_cube_field();
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    // An oblique ray with over-relaxation enabled exercises the back-off path.
    let queries = vec![
        SdfEnhancedTraceQuery::new([2.5, 2.5, 2.5], [-1.0, -1.0, -1.0], 12.0, 1.0e-3, 256, 1.8),
        SdfEnhancedTraceQuery::new([-0.5, 0.5, 2.5], [0.3, 0.0, -1.0], 12.0, 1.0e-3, 256, 1.6),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "oblique_over_relaxed");
}

#[test]
fn degenerate_direction_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = padded_cube_field();
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    let queries = vec![SdfEnhancedTraceQuery::new(
        [0.5, 0.5, 3.0],
        [0.0, 0.0, 0.0],
        10.0,
        1.0e-3,
        64,
        1.0,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate_direction");
    assert_eq!(out[0].hit, 0, "a zero-length direction cannot hit");
}

#[test]
fn ray_pointing_away_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = padded_cube_field();
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    // Origin above the box aimed further up: the slab clip rejects it.
    let queries = vec![SdfEnhancedTraceQuery::new(
        [0.5, 0.5, 3.0],
        [0.0, 0.0, 1.0],
        10.0,
        1.0e-3,
        64,
        1.0,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "pointing_away");
    assert_eq!(out[0].hit, 0, "a ray aimed away from the box misses");
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    let mut rng = Lcg::new(0x0EA7_51D5);
    let mut built = 0u32;
    while built < 512 {
        // Random small field with random per-cell signed distances.
        let dims = [
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
        ];
        let voxel = rng.next_range(0.25, 1.0);
        let origin = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let mut dist = Vec::with_capacity(count);
        for _ in 0..count {
            dist.push(rng.next_range(-1.5, 1.5));
        }

        // Random ray; some clearly enter the box, some miss.
        let ray_origin = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let direction = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        // Choose either strictly conservative stepping or a clear over-relax
        // factor, never within the guard of the `omega > 1` boundary.
        let over_relaxation = if rng.next_u32() & 1 == 0 {
            1.0
        } else {
            rng.next_range(1.2, 1.9)
        };
        let max_distance = rng.next_range(2.0, 8.0);
        let pixel_radius = rng.next_range(1.0e-3, 5.0e-2);
        let max_steps = 8 + (rng.next_u32() % 24);

        let query = SdfEnhancedTraceQuery::new(
            ray_origin,
            direction,
            max_distance,
            pixel_radius,
            max_steps,
            over_relaxation,
        );

        // Reject rays whose direction is too short to normalize (covered by a
        // dedicated fixture) or whose march straddles a branch boundary, so the
        // discrete counts cannot flip under a multiply-add contraction.
        if normalize(direction).is_none() {
            continue;
        }
        let probe = trace(dims, origin, voxel, &dist, &query);
        if !probe.conditioned {
            continue;
        }

        let field = SdfEnhancedTraceField::new(dims, origin, voxel, dist);
        let queries = [query];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        built += 1;
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = padded_cube_field();
    let gpu = GpuSdfEnhancedTrace::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
