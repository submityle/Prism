//! Real-device parity for the per-particle mesh-instance transform-assembly
//! twin:
//! [`GpuMeshRenderer`](prism_volumetric_gpu::mesh_renderer::GpuMeshRenderer)
//! must reproduce the `CPU` golden
//! [`mesh_renderer`](prism_render_architecture::particle::mesh_renderer) across
//! the four orientation modes (`Identity`, velocity-aligned, the numeric
//! `(sin, cos)` fixed rotation and the minimal align-onto-axis rotation), the
//! folded `R * S` linear part, the world placement, the world-space `AABB` and
//! the `LOD` tier, plus a randomized batch of clearly-conditioned queries
//! compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each instance is a fixed, non-reorderable chain of guarded matrix
//! operations, so `CPU` and `GPU` evaluate the same algebra in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place, and the world bounds use an eight-corner `min`/`max`
//! reduction on-device versus the golden's algebraically identical
//! `abs(linear)` half-extent path. The comparison therefore allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous `f32` field (the
//! `rotation`, `linear`, `translation`, `world_aabb_min` and `world_aabb_max`
//! values) while the discrete `lod_tier` is compared bit-exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately clear of the twin's branch ties. The
//! velocity-aligned basis keeps the `velocity`-by-`up_reference` cross well off
//! the collapse (except the one deliberate parallel-fallback fixture, which uses
//! clean axis-aligned vectors so both sides take the identical fallback). The
//! align-onto-axis target keeps `|dot|` off the parallel/antiparallel tie
//! (except the deliberate parallel and antiparallel fixtures, again clean and
//! deterministic). Every normalize input stays well above the `EPS_LEN_SQ`
//! guard, local boxes keep `min <= max`, and the `LOD` coverage stays a safe
//! margin off every threshold so the discrete tier is exact by construction.
//!
//! Provenance: twinned from this repository's
//! [`mesh_renderer`](prism_render_architecture::particle::mesh_renderer); no
//! third-party engine source or derived code.

use prism_volumetric_gpu::mesh_renderer::{
    cpu_reference, GpuMeshRenderer, GpuMeshRendererQuery, GpuMeshRendererResult, LOCAL_AXIS_PLUS_X,
    LOCAL_AXIS_PLUS_Y, LOCAL_AXIS_PLUS_Z, MAX_LOD_THRESHOLDS, ORIENTATION_ALIGN_TO_AXIS,
    ORIENTATION_FIXED_ROTATION, ORIENTATION_IDENTITY, ORIENTATION_VELOCITY_ALIGNED,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two 3-vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]) && close(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random 3-vector with each component in `[-span, span)`.
fn rand_vec3(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// The squared length of a 3-vector.
fn len_sq(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// The dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The cross product of two 3-vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The unit vector in the direction of `v`, via `f32::sqrt` (no transcendental).
fn unit(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / len_sq(v).sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Rejection-samples a well-spread unit 3-vector, retrying until the raw draw is
/// clear of the near-zero region so the normalize is well defined.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let raw = rand_vec3(state, 1.0);
        if len_sq(raw) >= 0.09 {
            return unit(raw);
        }
    }
}

/// A numeric `(sin, cos)` pair from a Pythagorean parameterization, so the
/// fixture needs no trigonometry yet still feeds the kernel a valid unit pair:
/// `sin = 2m / (1 + m^2)`, `cos = (1 - m^2) / (1 + m^2)`.
fn pyth_sincos(state: &mut u64) -> (f32, f32) {
    let m = signed(state, 2.0);
    let d = 1.0 + m * m;
    (2.0 * m / d, (1.0 - m * m) / d)
}

/// The unit vector for a local-axis code, matching the twin's `local_axis_unit`
/// default (anything other than `+Y` / `+Z` folds to `+X`).
fn axis_unit(code: u32) -> [f32; 3] {
    if code == LOCAL_AXIS_PLUS_Y {
        [0.0, 1.0, 0.0]
    } else if code == LOCAL_AXIS_PLUS_Z {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// A clearly-conditioned descending `LOD` threshold ladder plus a coverage that
/// stays a safe margin off every active threshold, so the discrete tier is
/// exact. Returns the thresholds, the coverage and the valid threshold count.
fn rand_lod(state: &mut u64) -> ([f32; MAX_LOD_THRESHOLDS], f32, u32) {
    let thresholds = [0.8, 0.6, 0.4, 0.2];
    let count = (lcg(state) * 5.0) as u32;
    let n = (count as usize).min(MAX_LOD_THRESHOLDS);
    let coverage = loop {
        let c = lcg(state);
        let mut ok = true;
        for &t in &thresholds[..n] {
            if (c - t).abs() <= 0.03 {
                ok = false;
                break;
            }
        }
        if ok {
            break c;
        }
    };
    (thresholds, coverage, count)
}

/// A benign, clearly-conditioned base query that each fixture overrides field by
/// field via struct-update syntax. It is an identity orientation with a unit
/// scale, a symmetric local box and a mid-range coverage selecting a middle
/// tier a safe margin off its neighbours.
fn base_query() -> GpuMeshRendererQuery {
    GpuMeshRendererQuery {
        position: [1.0, -2.0, 3.0],
        velocity: [1.0, 0.0, 0.0],
        up_reference: [0.0, 1.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        target: [0.0, 1.0, 0.0],
        per_axis_scale: [1.0, 1.0, 1.0],
        local_aabb_min: [-1.0, -1.0, -1.0],
        local_aabb_max: [1.0, 1.0, 1.0],
        thresholds: [0.75, 0.5, 0.25, 0.1],
        rot_sin: 0.0,
        rot_cos: 1.0,
        uniform_scale: 1.0,
        size_over_life: 1.0,
        coverage: 0.6,
        orientation_mode: ORIENTATION_IDENTITY,
        local_axis: LOCAL_AXIS_PLUS_X,
        threshold_count: 4,
    }
}

/// Builds a clearly-conditioned random query by rejection sampling. The
/// orientation mode and local-axis code are drawn freely so the batch covers
/// every branch; the velocity-aligned inputs keep the cross well off the
/// collapse, the align target keeps `|dot|` off the parallel tie, the scale and
/// position stay moderate so the bounds parity holds, and the local box keeps
/// `min <= max`.
fn well_conditioned(state: &mut u64) -> GpuMeshRendererQuery {
    // Velocity-aligned inputs: a large velocity-by-up cross so the basis never
    // takes the degenerate fallback.
    let (velocity, up_reference) = loop {
        let v = rand_vec3(state, 3.0);
        let u = rand_vec3(state, 3.0);
        if len_sq(v) >= 0.5 && len_sq(cross(v, u)) >= 0.5 {
            break (v, u);
        }
    };

    // Fixed-rotation inputs: a unit axis and a numeric unit (sin, cos) pair.
    let axis = rand_unit(state);
    let (rot_sin, rot_cos) = pyth_sincos(state);

    // Align-to-axis inputs: a local-axis code and a target well off the parallel
    // and antiparallel ties.
    let local_axis = (lcg(state) * 3.0) as u32;
    let la_unit = axis_unit(local_axis);
    let target = loop {
        let t = rand_vec3(state, 3.0);
        if len_sq(t) >= 0.5 && dot(la_unit, unit(t)).abs() <= 0.9 {
            break t;
        }
    };

    // Moderate positive scale so the world bounds stay well inside the tolerance.
    let per_axis_scale = [
        0.5 + lcg(state) * 2.0,
        0.5 + lcg(state) * 2.0,
        0.5 + lcg(state) * 2.0,
    ];
    let uniform_scale = 0.5 + lcg(state) * 1.5;
    let size_over_life = 0.5 + lcg(state) * 1.5;

    // Local box with min <= max in every component.
    let center = rand_vec3(state, 2.0);
    let half = [
        0.25 + lcg(state) * 1.5,
        0.25 + lcg(state) * 1.5,
        0.25 + lcg(state) * 1.5,
    ];
    let local_aabb_min = [
        center[0] - half[0],
        center[1] - half[1],
        center[2] - half[2],
    ];
    let local_aabb_max = [
        center[0] + half[0],
        center[1] + half[1],
        center[2] + half[2],
    ];

    let (thresholds, coverage, threshold_count) = rand_lod(state);
    let orientation_mode = (lcg(state) * 4.0) as u32;
    let position = rand_vec3(state, 10.0);

    GpuMeshRendererQuery {
        position,
        velocity,
        up_reference,
        axis,
        target,
        per_axis_scale,
        local_aabb_min,
        local_aabb_max,
        thresholds,
        rot_sin,
        rot_cos,
        uniform_scale,
        size_over_life,
        coverage,
        orientation_mode,
        local_axis,
        threshold_count,
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the resolved
/// rotation matrix, the `R * S` linear part, the translation and the world-space
/// bounds must all agree within bound, and the discrete `lod_tier` bit-exactly.
fn pin(idx: usize, query: &GpuMeshRendererQuery, got: &GpuMeshRendererResult) {
    let want = cpu_reference(query);

    for row in 0..3 {
        close_vec("rotation", idx, got.rotation[row], want.rotation[row]);
        close_vec("linear", idx, got.linear[row], want.linear[row]);
    }
    close_vec("translation", idx, got.translation, want.translation);
    close_vec(
        "world_aabb_min",
        idx,
        got.world_aabb_min,
        want.world_aabb_min,
    );
    close_vec(
        "world_aabb_max",
        idx,
        got.world_aabb_max,
        want.world_aabb_max,
    );

    assert_eq!(
        got.lod_tier, want.lod_tier,
        "query {idx} lod_tier: gpu {} vs cpu {}",
        got.lod_tier, want.lod_tier
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuMeshRenderer, queries: &[GpuMeshRendererQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(
        got.is_empty(),
        "an empty input must return an empty result with no dispatch"
    );
}

#[test]
fn identity_orientation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let queries = [
        base_query(),
        GpuMeshRendererQuery {
            position: [5.0, 6.0, -7.0],
            per_axis_scale: [2.0, 0.5, 1.5],
            uniform_scale: 1.5,
            size_over_life: 0.75,
            local_aabb_min: [-0.5, -2.0, -1.0],
            local_aabb_max: [1.5, 0.5, 3.0],
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn velocity_aligned_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let queries = [
        // Well-conditioned: velocity and up are far from parallel.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_VELOCITY_ALIGNED,
            velocity: [2.0, 0.5, -1.0],
            up_reference: [0.0, 1.0, 0.0],
            ..base_query()
        },
        // Oblique velocity, non-axis up reference.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_VELOCITY_ALIGNED,
            velocity: [-1.0, 2.0, 0.5],
            up_reference: [0.2, 0.9, 0.3],
            ..base_query()
        },
        // Deliberate parallel fallback: velocity parallel to up, so the cross
        // collapses and both sides take the identical any-perpendicular path.
        // Clean axis-aligned values keep the branch unambiguous.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_VELOCITY_ALIGNED,
            velocity: [0.0, 3.0, 0.0],
            up_reference: [0.0, 1.0, 0.0],
            ..base_query()
        },
        // Deliberate zero velocity: the forward axis falls back to world +X.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_VELOCITY_ALIGNED,
            velocity: [0.0, 0.0, 0.0],
            up_reference: [0.0, 1.0, 0.0],
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixed_rotation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let queries = [
        // Zero angle (sin, cos) = (0, 1): the identity rotation.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: [0.0, 0.0, 1.0],
            rot_sin: 0.0,
            rot_cos: 1.0,
            ..base_query()
        },
        // Quarter turn (sin, cos) = (1, 0) about +Z.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: [0.0, 0.0, 1.0],
            rot_sin: 1.0,
            rot_cos: 0.0,
            ..base_query()
        },
        // A (0.6, 0.8) unit pair about a tilted axis.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: unit([1.0, 1.0, 0.0]),
            rot_sin: 0.6,
            rot_cos: 0.8,
            ..base_query()
        },
        // A (0.8, -0.6) unit pair (obtuse angle) about +X.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: [1.0, 0.0, 0.0],
            rot_sin: 0.8,
            rot_cos: -0.6,
            ..base_query()
        },
        // Deliberate zero axis: the rotation folds to the identity.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: [0.0, 0.0, 0.0],
            rot_sin: 0.6,
            rot_cos: 0.8,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn align_to_axis_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let queries = [
        // General rotation locking +X onto a tilted target.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_X,
            target: unit([0.3, 0.8, 0.5]),
            ..base_query()
        },
        // Lock +Y onto a tilted target.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Y,
            target: unit([0.7, -0.2, 0.6]),
            ..base_query()
        },
        // Lock +Z onto a tilted target.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Z,
            target: unit([-0.5, 0.6, 0.4]),
            ..base_query()
        },
        // Deliberate parallel: target already on +Y, so the rotation is the
        // identity (clean axis-aligned values keep the branch unambiguous).
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Y,
            target: [0.0, 2.0, 0.0],
            ..base_query()
        },
        // Deliberate antiparallel: target opposes +Y, so a half turn about a
        // deterministic perpendicular (both sides take the identical path).
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Y,
            target: [0.0, -3.0, 0.0],
            ..base_query()
        },
        // Deliberate zero target: nothing to align to, so the identity.
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Z,
            target: [0.0, 0.0, 0.0],
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn transform_aabb_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    // A rotated, non-uniformly-scaled, offset instance over an asymmetric local
    // box: the eight-corner on-device reduction must match the golden's
    // abs(linear) half-extent bounds within tolerance.
    let queries = [
        GpuMeshRendererQuery {
            position: [3.5, -2.0, 1.25],
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: [0.0, 0.0, 1.0],
            rot_sin: 0.6,
            rot_cos: 0.8,
            per_axis_scale: [1.5, 0.75, 2.0],
            uniform_scale: 1.25,
            size_over_life: 0.9,
            local_aabb_min: [-0.5, -1.0, -0.25],
            local_aabb_max: [1.5, 0.5, 0.75],
            ..base_query()
        },
        GpuMeshRendererQuery {
            position: [-6.0, 4.0, -3.0],
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_X,
            target: unit([0.4, 0.7, -0.5]),
            per_axis_scale: [0.8, 1.2, 1.6],
            uniform_scale: 1.1,
            size_over_life: 1.0,
            local_aabb_min: [-2.0, -0.5, -1.5],
            local_aabb_max: [0.5, 1.5, 1.0],
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn select_mesh_lod_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let thresholds = [0.8, 0.6, 0.4, 0.2];
    // A coverage sweep hitting every tier a safe margin off each threshold, both
    // clamp sides, and the empty-threshold (count = 0) case.
    let queries = [
        // Above the top threshold: tier 0.
        GpuMeshRendererQuery {
            coverage: 0.9,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Between the first two thresholds: tier 1.
        GpuMeshRendererQuery {
            coverage: 0.7,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Tier 2.
        GpuMeshRendererQuery {
            coverage: 0.5,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Tier 3.
        GpuMeshRendererQuery {
            coverage: 0.3,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Below every threshold: the lowest tier (threshold_count).
        GpuMeshRendererQuery {
            coverage: 0.1,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Over-range coverage clamps to one: tier 0.
        GpuMeshRendererQuery {
            coverage: 1.5,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Under-range coverage clamps to zero: the lowest tier.
        GpuMeshRendererQuery {
            coverage: -0.5,
            thresholds,
            threshold_count: 4,
            ..base_query()
        },
        // Fewer active thresholds: only the first two are read, tier 2 below.
        GpuMeshRendererQuery {
            coverage: 0.5,
            thresholds,
            threshold_count: 2,
            ..base_query()
        },
        // No thresholds: always tier 0.
        GpuMeshRendererQuery {
            coverage: 0.5,
            thresholds,
            threshold_count: 0,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing a few deterministic fixtures with many clearly-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-wise.
    let mut queries = vec![
        base_query(),
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_VELOCITY_ALIGNED,
            velocity: [1.0, 2.0, -0.5],
            up_reference: [0.0, 1.0, 0.0],
            ..base_query()
        },
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_FIXED_ROTATION,
            axis: unit([0.0, 1.0, 1.0]),
            rot_sin: 0.6,
            rot_cos: 0.8,
            ..base_query()
        },
        GpuMeshRendererQuery {
            orientation_mode: ORIENTATION_ALIGN_TO_AXIS,
            local_axis: LOCAL_AXIS_PLUS_Z,
            target: unit([0.2, 0.5, 0.8]),
            ..base_query()
        },
    ];
    for _ in 0..64 {
        queries.push(well_conditioned(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshRenderer::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned queries (several workgroups' worth)
    // pins every reported field across many random orientations, scales, boxes
    // and LOD ladders.
    let queries: Vec<GpuMeshRendererQuery> =
        (0..256).map(|_| well_conditioned(&mut state)).collect();
    check(&ctx, &gpu, &queries);

    // The random stream must span more than one orientation mode and more than
    // one LOD tier, proving the discrete classifiers are genuinely exercised
    // rather than stuck on a single branch.
    let mut seen_modes = [false; 4];
    let mut seen_tier0 = false;
    let mut seen_other_tier = false;
    let results = gpu.eval(&ctx, &queries);
    for (query, result) in queries.iter().zip(results.iter()) {
        let mode = query.orientation_mode.min(3) as usize;
        seen_modes[mode] = true;
        if result.lod_tier == 0 {
            seen_tier0 = true;
        } else {
            seen_other_tier = true;
        }
    }
    assert!(
        seen_modes.iter().all(|&m| m),
        "random sweep should cover every orientation mode: {seen_modes:?}"
    );
    assert!(
        seen_tier0 && seen_other_tier,
        "random sweep should cover more than one LOD tier (tier0 {seen_tier0}, other {seen_other_tier})"
    );
}
