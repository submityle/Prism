//! Real-device parity for the screen-motion-vector twin:
//! [`GpuMotionVectors`](prism_volumetric_gpu::motion_vectors::GpuMotionVectors)
//! must reproduce the `CPU` golden
//! [`screen_motion_vector`](prism_render_architecture::particle::motion_vectors::screen_motion_vector)
//! across spawned particles, front-facing particles, random depths and random
//! camera matrices, both `NDC`-to-`UV` conventions and degenerate homogeneous
//! `w`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies and adds (the
//! `4x4` transform, the perspective divide, the jitter subtraction and the `UV`
//! map), so `CPU` and `GPU` evaluate the same closed form in the same order.
//! The comparison allows `abs_diff <= 1e-5` or `rel_diff <= 1e-5` — loose
//! enough to admit a legal fused multiply-add contraction across the four terms
//! of each transformed component, yet tight enough to fail a wrong port (a
//! transposed matrix read, a dropped jitter term, a flipped `V` axis, a missing
//! `w` guard). The validity decision must match exactly: a lane the reference
//! rejects as [`None`] (a spawned particle or a point at/behind the camera
//! plane) must read back invalid, and vice versa.
//!
//! Provenance: standard clip-space reprojection screen motion vector; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::motion_vectors::{
    screen_motion_vector, CameraMotionState, Mat4, NdcConvention, PrevParticleState,
    ScreenMotionVector, Vec2,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::motion_vectors::{GpuMotionVectors, MotionVectorQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate, perturbing the low mantissa bits by a few units
/// in the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Builds a perspective-style view-projection matrix in the reference's
/// column-major (`cols[col][row]`) storage, where `w = -z` so a point with
/// `z < 0` sits in front of the camera. The focal length, aspect, near and far
/// are passed as literal constants so no transcendental is needed to build it.
fn perspective(focal: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
    let depth = (zfar + znear) / (znear - zfar);
    let depth_bias = (2.0 * zfar * znear) / (znear - zfar);
    // cols[col][row]: column 2 carries the depth remap and the `-1` that makes
    // `w = -z`; column 3 carries the depth bias.
    Mat4::from_cols([
        [focal / aspect, 0.0, 0.0, 0.0],
        [0.0, focal, 0.0, 0.0],
        [0.0, 0.0, depth, -1.0],
        [0.0, 0.0, depth_bias, 0.0],
    ])
}

/// Builds a fully random matrix in `cols[col][row]` order with entries in
/// roughly `[-scale, scale)`. Used to exercise arbitrary (non-physical)
/// view-projection transforms where the reference, not any assumption about the
/// projection, defines the ground truth.
fn random_matrix(state: &mut u64, scale: f32) -> Mat4 {
    let mut cols = [[0.0_f32; 4]; 4];
    for col in &mut cols {
        for value in col.iter_mut() {
            *value = lcg(state) * scale;
        }
    }
    Mat4::from_cols(cols)
}

/// Runs the `GPU` dispatch and compares every lane against the `CPU` golden
/// [`screen_motion_vector`], returning the `GPU` results for any extra
/// per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMotionVectors,
    camera: CameraMotionState,
    queries: &[MotionVectorQuery],
) -> Vec<Option<ScreenMotionVector>> {
    let got = gpu.eval(ctx, camera, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        let expected = screen_motion_vector(camera, query.cur_world, query.prev);
        match (expected, result) {
            (None, None) => {}
            (Some(exp), Some(act)) => {
                assert!(
                    close(act.uv_delta.x, exp.uv_delta.x)
                        && close(act.uv_delta.y, exp.uv_delta.y),
                    "uv delta mismatch at particle {idx}: gpu ({}, {}), cpu ({}, {})",
                    act.uv_delta.x,
                    act.uv_delta.y,
                    exp.uv_delta.x,
                    exp.uv_delta.y
                );
            }
            (exp, act) => panic!(
                "validity mismatch at particle {idx}: cpu {}, gpu {}",
                exp.is_some(),
                act.is_some()
            ),
        }
    }
    got
}

/// A front-facing camera with no jitter under the given convention.
fn camera_no_jitter(convention: NdcConvention) -> CameraMotionState {
    CameraMotionState {
        cur_view_proj: perspective(1.5, 1.0, -0.1, -100.0),
        // The previous frame used a slightly different focal length, so a
        // perfectly static particle still produces a non-zero motion vector.
        prev_view_proj: perspective(1.4, 1.0, -0.1, -100.0),
        cur_jitter: Vec2::ZERO,
        prev_jitter: Vec2::ZERO,
        convention,
    }
}

/// A particle with valid previous-frame history.
fn moving(cur: Vec3, prev: Vec3) -> MotionVectorQuery {
    MotionVectorQuery {
        cur_world: cur,
        prev: PrevParticleState::record(prev),
    }
}

#[test]
fn empty_input_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let camera = camera_no_jitter(NdcConvention::TopLeftYDown);
    let out = gpu.eval(&ctx, camera, &[]);
    assert!(out.is_empty(), "an empty query slice yields no results");
}

#[test]
fn front_facing_particles_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let camera = camera_no_jitter(NdcConvention::TopLeftYDown);
    // Points with z < 0 sit in front of the camera (w = -z > 0).
    let queries = [
        moving(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.1, 0.0, -5.0)),
        moving(Vec3::new(1.0, -2.0, -8.0), Vec3::new(0.9, -1.8, -8.2)),
        moving(Vec3::new(-3.0, 4.0, -20.0), Vec3::new(-3.1, 3.6, -19.5)),
        moving(Vec3::new(2.5, 2.5, -2.0), Vec3::new(2.4, 2.6, -2.1)),
    ];
    let got = check(&ctx, &gpu, camera, &queries);
    assert!(
        got.iter().all(Option::is_some),
        "front-facing particles should all produce a motion vector"
    );
    assert!(
        got.iter()
            .any(|r| r.is_some_and(|m| m.uv_delta.x.abs() > EPS || m.uv_delta.y.abs() > EPS)),
        "at least one particle should have a non-trivial motion vector"
    );
}

#[test]
fn jitter_is_removed_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    // Non-zero, distinct per-frame jitter must be subtracted after the divide;
    // a port that forgets it (or swaps the two frames) diverges from the
    // reference here.
    let camera = CameraMotionState {
        cur_view_proj: perspective(1.5, 1.3, -0.1, -100.0),
        prev_view_proj: perspective(1.5, 1.3, -0.1, -100.0),
        cur_jitter: Vec2::new(0.013, -0.027),
        prev_jitter: Vec2::new(-0.009, 0.021),
        convention: NdcConvention::TopLeftYDown,
    };
    let queries = [
        moving(Vec3::new(0.3, -0.4, -3.0), Vec3::new(0.3, -0.4, -3.0)),
        moving(Vec3::new(1.2, 0.7, -6.0), Vec3::new(1.1, 0.8, -6.3)),
    ];
    check(&ctx, &gpu, camera, &queries);
}

#[test]
fn both_ndc_conventions_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let queries = [
        moving(Vec3::new(0.5, 1.5, -4.0), Vec3::new(0.4, 1.3, -4.1)),
        moving(Vec3::new(-2.0, -1.0, -9.0), Vec3::new(-2.2, -0.8, -8.7)),
        moving(Vec3::new(3.0, -3.0, -12.0), Vec3::new(2.8, -3.1, -12.4)),
    ];
    for convention in [NdcConvention::TopLeftYDown, NdcConvention::BottomLeftYUp] {
        let camera = camera_no_jitter(convention);
        check(&ctx, &gpu, camera, &queries);
    }
    // The two conventions must actually differ in V (sign-flipped), otherwise
    // the convention branch is dead and the test above would be vacuous.
    let top = gpu.eval(
        &ctx,
        camera_no_jitter(NdcConvention::TopLeftYDown),
        &queries,
    );
    let bottom = gpu.eval(
        &ctx,
        camera_no_jitter(NdcConvention::BottomLeftYUp),
        &queries,
    );
    assert!(
        top.iter()
            .zip(bottom.iter())
            .any(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => (a.uv_delta.y - b.uv_delta.y).abs() > EPS,
                _ => false,
            }),
        "the two conventions should differ in the V channel"
    );
}

#[test]
fn random_depths_and_random_camera_matrices_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let conventions = [NdcConvention::TopLeftYDown, NdcConvention::BottomLeftYUp];
    for round in 0..6_u32 {
        // Alternate between a physical perspective pair and a fully arbitrary
        // matrix pair so the reference, not any projection assumption, defines
        // the truth across a wide input distribution.
        let (cur_vp, prev_vp) = if round % 2 == 0 {
            (
                perspective(1.2 + lcg(&mut state) * 0.3, 1.4, -0.1, -500.0),
                perspective(1.2 + lcg(&mut state) * 0.3, 1.4, -0.1, -500.0),
            )
        } else {
            (
                random_matrix(&mut state, 4.0),
                random_matrix(&mut state, 4.0),
            )
        };
        let camera = CameraMotionState {
            cur_view_proj: cur_vp,
            prev_view_proj: prev_vp,
            cur_jitter: Vec2::new(lcg(&mut state) * 0.02, lcg(&mut state) * 0.02),
            prev_jitter: Vec2::new(lcg(&mut state) * 0.02, lcg(&mut state) * 0.02),
            convention: conventions[(round % 2) as usize],
        };
        let mut queries = Vec::with_capacity(64);
        for _ in 0..64 {
            // Random world positions spanning a range of depths, including some
            // behind the camera (positive z) that exercise the `w` guard.
            let cur = Vec3::new(
                lcg(&mut state) * 10.0,
                lcg(&mut state) * 10.0,
                lcg(&mut state) * 30.0,
            );
            let prev = Vec3::new(
                cur.x + lcg(&mut state) * 0.5,
                cur.y + lcg(&mut state) * 0.5,
                cur.z + lcg(&mut state) * 0.5,
            );
            queries.push(moving(cur, prev));
        }
        check(&ctx, &gpu, camera, &queries);
    }
}

#[test]
fn spawned_particles_have_no_history() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let camera = camera_no_jitter(NdcConvention::TopLeftYDown);
    // A particle spawned this frame carries no valid history; the reference
    // returns `None` even though its position projects validly.
    let queries = [
        MotionVectorQuery {
            cur_world: Vec3::new(0.0, 0.0, -5.0),
            prev: PrevParticleState::spawned(Vec3::new(0.0, 0.0, -5.0)),
        },
        moving(Vec3::new(1.0, 1.0, -5.0), Vec3::new(0.9, 1.0, -5.0)),
    ];
    let got = check(&ctx, &gpu, camera, &queries);
    assert!(got[0].is_none(), "a spawned particle produces no vector");
    assert!(got[1].is_some(), "a tracked particle still produces one");
}

#[test]
fn degenerate_w_is_rejected_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let camera = camera_no_jitter(NdcConvention::TopLeftYDown);
    // With `w = -z`, a point at `z = 0` has `w = 0 <= EPS_W` (on the camera
    // plane) and a point at `z > 0` has `w < 0` (behind it); both must be
    // rejected. The `cur`/`prev` degenerate cases are split so a port that
    // guards only one frame is caught.
    let queries = [
        // Current position on the camera plane.
        moving(Vec3::new(0.5, 0.5, 0.0), Vec3::new(0.5, 0.5, -5.0)),
        // Current position behind the camera.
        moving(Vec3::new(0.5, 0.5, 3.0), Vec3::new(0.5, 0.5, -5.0)),
        // Previous position behind the camera.
        moving(Vec3::new(0.5, 0.5, -5.0), Vec3::new(0.5, 0.5, 2.0)),
        // Both valid for contrast.
        moving(Vec3::new(0.5, 0.5, -5.0), Vec3::new(0.4, 0.5, -5.0)),
    ];
    let got = check(&ctx, &gpu, camera, &queries);
    assert!(got[0].is_none(), "a point on the camera plane is rejected");
    assert!(got[1].is_none(), "a point behind the camera is rejected");
    assert!(got[2].is_none(), "a prev point behind the camera is rejected");
    assert!(got[3].is_some(), "the fully valid particle still produces one");
}

#[test]
fn mixed_batch_matches_cpu_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionVectors::new(&ctx);
    let camera = CameraMotionState {
        cur_view_proj: perspective(1.6, 1.2, -0.1, -200.0),
        prev_view_proj: perspective(1.55, 1.2, -0.1, -200.0),
        cur_jitter: Vec2::new(0.004, -0.002),
        prev_jitter: Vec2::new(-0.003, 0.005),
        convention: NdcConvention::BottomLeftYUp,
    };
    // Interleave valid, spawned and degenerate particles so one dispatch
    // exercises every branch and lane indexing cannot be accidentally correct.
    let queries = [
        moving(Vec3::new(0.2, -0.3, -4.0), Vec3::new(0.1, -0.2, -4.2)),
        MotionVectorQuery {
            cur_world: Vec3::new(1.0, 1.0, -6.0),
            prev: PrevParticleState::spawned(Vec3::new(1.0, 1.0, -6.0)),
        },
        moving(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -5.0)),
        moving(Vec3::new(-1.5, 2.0, -10.0), Vec3::new(-1.6, 2.1, -9.8)),
        moving(Vec3::new(5.0, -5.0, -3.0), Vec3::new(5.0, -5.0, -3.0)),
    ];
    let got = check(&ctx, &gpu, camera, &queries);
    assert!(got[1].is_none(), "spawned lane is None");
    assert!(got[2].is_none(), "behind-camera lane is None");
    assert!(got[0].is_some() && got[3].is_some() && got[4].is_some());
}
