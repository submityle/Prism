//! Real-device parity for the orientation-basis twin:
//! [`GpuOrientationBasis`](prism_volumetric_gpu::orientation_basis::GpuOrientationBasis)
//! must reproduce the `CPU` golden
//! [`orientation_basis`](prism_render_architecture::particle::orientation_basis)
//! across all five facing modes with generic geometry, every degenerate
//! fallback the reference takes (a camera sitting on the particle, a `world_up`
//! or `fixed_axis` parallel to the view direction, a zero velocity, a zero
//! `world_up`), and a randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each basis is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` axis component.
//!
//! # Conditioning
//!
//! The random batch is kept well away from every branch crack: the view
//! direction is forced far from degenerate (the camera is well separated from
//! the particle), the velocity magnitude stays comfortably above `MIN_LENGTH`,
//! and the `cross(up, view)` that selects the `right` axis is accepted only when
//! its length is comfortably large, so `CPU` and `GPU` stay on the same side of
//! every fallback branch regardless of a few units in the last place of slack.
//! The degenerate fixtures use exact-zero or exactly-parallel inputs so both
//! devices classify the collapse identically.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::orientation_basis`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::orientation_basis::{FacingMode, OrientationBasis};
use prism_volumetric_gpu::orientation_basis::{golden, GpuOrientationBasis, OrientationQuery};
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

/// World-space Y axis, the reference's default `world_up`.
const WORLD_Y: [f32; 3] = [0.0, 1.0, 0.0];

/// Every facing mode, so each degenerate test can sweep the whole enum.
const ALL_MODES: [FacingMode; 5] = [
    FacingMode::Billboard,
    FacingMode::HorizontalBillboard,
    FacingMode::VerticalBillboard,
    FacingMode::VelocityAligned,
    FacingMode::FixedAxis,
];

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Squared length of `v`.
fn length_squared(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// `a - b` component-wise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Builds a clearly-conditioned velocity-aligned query by rejection sampling:
/// the camera is well separated from the particle (a sharp view direction), the
/// velocity magnitude is comfortably above `MIN_LENGTH`, and the `cross(up,
/// view)` that selects the `right` axis is accepted only when its length is
/// comfortably large, so the fallback branch is never on a tie.
fn rand_query(state: &mut u64) -> OrientationQuery {
    loop {
        let pos = rand_vec(state, 3.0);
        let cam_pos = rand_vec(state, 3.0);
        let vel = rand_vec(state, 2.0);
        let view = sub(cam_pos, pos);
        if length_squared(view) < 1.0 {
            continue;
        }
        if length_squared(vel) < 0.5 {
            continue;
        }
        // Guard the right-axis cross: reject near-parallel up/view so both
        // devices keep the primary branch instead of the perpendicular fallback.
        if length_squared(cross(vel, view)) < 0.25 {
            continue;
        }
        return OrientationQuery::new(
            FacingMode::VelocityAligned,
            pos,
            cam_pos,
            vel,
            WORLD_Y,
            [1.0, 0.0, 0.0],
        );
    }
}

/// Pins one `GPU` basis against the `CPU` golden for `query`: both the `right`
/// and `up` axes must agree component-for-component within bound.
fn pin(idx: usize, query: &OrientationQuery, got: &OrientationBasis) {
    let want = golden(query);
    for k in 0..3 {
        assert!(
            close(got.right[k], want.right[k]),
            "query {idx} right[{k}]: gpu {} vs cpu {}",
            got.right[k],
            want.right[k]
        );
        assert!(
            close(got.up[k], want.up[k]),
            "query {idx} up[{k}]: gpu {} vs cpu {}",
            got.up[k],
            want.up[k]
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every basis against the reference.
fn check(ctx: &GpuContext, gpu: &GpuOrientationBasis, queries: &[OrientationQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, basis)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, basis);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn billboard_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // Camera off to the side so the view direction is well away from world up:
    // the right axis is cross(world_up, view) and up is cross(view, right).
    let query = OrientationQuery::new(
        FacingMode::Billboard,
        [0.0, 0.0, 0.0],
        [3.0, 1.0, 5.0],
        [0.0, 0.0, 0.0],
        WORLD_Y,
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn horizontal_billboard_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // The up axis locks to world up; right turns toward the camera.
    let query = OrientationQuery::new(
        FacingMode::HorizontalBillboard,
        [0.0, 0.0, 0.0],
        [4.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        WORLD_Y,
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn vertical_billboard_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // Ground-aligned: the plane normal is world up, the in-plane axes face the
    // camera.
    let query = OrientationQuery::new(
        FacingMode::VerticalBillboard,
        [0.0, 0.0, 0.0],
        [4.0, 9.0, 2.0],
        [0.0, 0.0, 0.0],
        WORLD_Y,
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn velocity_aligned_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // The up axis follows the velocity; right turns toward the camera.
    let query = OrientationQuery::new(
        FacingMode::VelocityAligned,
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
        WORLD_Y,
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn fixed_axis_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // The up axis locks to the supplied constraint axis; right turns toward the
    // camera.
    let query = OrientationQuery::new(
        FacingMode::FixedAxis,
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
        WORLD_Y,
        [0.0, 0.0, 1.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn camera_on_particle_falls_back_for_all_modes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // Camera exactly on the particle: the view direction is zero, so every mode
    // takes its deterministic fallback — both devices classify the collapse the
    // same way because the length is exactly zero.
    let same = [2.0, 2.0, 2.0];
    let queries: Vec<OrientationQuery> = ALL_MODES
        .iter()
        .map(|&mode| OrientationQuery::new(mode, same, same, [0.0, 0.0, 0.0], WORLD_Y, WORLD_Y))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn world_up_parallel_to_view_falls_back_for_all_modes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // Camera directly above the particle: the view is exactly parallel to world
    // up, so cross(world_up, view) is exactly zero and both devices take the
    // perpendicular fallback.
    let pos = [0.0, 0.0, 0.0];
    let cam_pos = [0.0, 7.0, 0.0];
    let queries: Vec<OrientationQuery> = ALL_MODES
        .iter()
        .map(|&mode| OrientationQuery::new(mode, pos, cam_pos, [0.0, 0.0, 0.0], WORLD_Y, WORLD_Y))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixed_axis_parallel_to_view_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // The constraint axis is exactly parallel to the view direction, so the
    // right-axis cross collapses to zero and falls back to a perpendicular.
    let query = OrientationQuery::new(
        FacingMode::FixedAxis,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0],
        [0.0, 0.0, 0.0],
        WORLD_Y,
        [0.0, 0.0, 1.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn zero_world_up_falls_back_for_all_modes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    // A zero world up (and zero velocity / fixed axis) forces every mode's
    // nested normalize fallbacks; both devices classify the exact-zero collapse
    // identically.
    let queries: Vec<OrientationQuery> = ALL_MODES
        .iter()
        .map(|&mode| {
            OrientationQuery::new(
                mode,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        OrientationQuery::new(
            FacingMode::Billboard,
            [0.0, 0.0, 0.0],
            [3.0, 1.0, 5.0],
            [0.0, 0.0, 0.0],
            WORLD_Y,
            [1.0, 0.0, 0.0],
        ),
        OrientationQuery::new(
            FacingMode::VelocityAligned,
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            WORLD_Y,
            [1.0, 0.0, 0.0],
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrientationBasis::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the frame across many
    // random particle states.
    let queries: Vec<OrientationQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
