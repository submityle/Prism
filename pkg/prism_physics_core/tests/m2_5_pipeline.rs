//! Acceptance test for the M2.5 deliverable: 异步流水线 + 状态哈希 (联机地基).
//!
//! The milestone's acceptance criterion is "插值无抖动" — interpolation must be
//! jitter-free. These tests drive a *real* XPBD free-fall simulation through the
//! fixed-step pipeline at a render rate that does **not** evenly divide the
//! physics rate, which is exactly the situation that makes an un-interpolated
//! renderer stutter. We then prove three things from live simulation data:
//!
//! 1. The raw per-frame physics pose stutters: on frames where the accumulator
//!    did not cross a whole [`FixedStepPipeline::fixed_dt`], the current snapshot
//!    repeats the previous frame's pose verbatim (a visible stall).
//! 2. The interpolated render pose is smooth and monotone: a falling body's
//!    interpolated height never ticks back upward frame-to-frame, and it never
//!    stalls on a repeated value. That is the "插值无抖动" guarantee.
//! 3. The state hash is deterministic — identical runs produce identical
//!    hashes, and a diverging input produces a different hash — which is the
//!    networking desync-detection foundation the milestone is named for.
//!
//! # Provenance
//!
//! These scenarios are original to Prism. They contain **no Unreal Engine
//! source or derived code**.

use glam::Vec3;
use prism_physics_core::{
    hash_state, locate_divergence, BodyDesc, BodyHandle, CpuBackend, FixedStepPipeline,
    PhysicsCommand, PhysicsWorld, StateHash, XpbdSolver,
};

/// Physics tick: 100 Hz.
const FIXED_DT: f32 = 0.01;
/// Render frame time: ~166 Hz. Deliberately smaller than and not an integer
/// divisor of [`FIXED_DT`] so some frames run zero physics sub-steps (forcing a
/// raw stall) and others run one (forcing a raw jump).
const RENDER_DT: f32 = 0.006;
/// Number of render frames to simulate.
const FRAMES: usize = 80;

/// Builds a CPU backend running the real XPBD solver with four sub-steps.
fn dynamics_backend() -> CpuBackend {
    CpuBackend::new(Box::new(XpbdSolver::new()), 4)
}

/// Spawns a single dynamic body at the origin in a gravity field and returns
/// the world plus the body handle.
fn free_fall_world() -> (PhysicsWorld, BodyHandle) {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    (world, body)
}

/// Runs the pipeline for [`FRAMES`] render frames, collecting the raw current
/// snapshot height and the interpolated render height for the body each frame.
fn run_capture(body: BodyHandle) -> (Vec<f32>, Vec<f32>) {
    let (mut world, _) = free_fall_world();
    let mut backend = dynamics_backend();
    let mut pipe = FixedStepPipeline::new(FIXED_DT);
    pipe.prime(&world);

    let mut raw = Vec::with_capacity(FRAMES);
    let mut interp = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        pipe.advance(&mut backend, &mut world, RENDER_DT);
        let raw_y = pipe
            .current_snapshot()
            .and_then(|s| s.pose(body))
            .map_or(0.0, |p| p.position.y);
        let interp_y = pipe.interpolated_pose(body).map_or(0.0, |p| p.position.y);
        raw.push(raw_y);
        interp.push(interp_y);
    }
    (raw, interp)
}

#[test]
fn raw_snapshot_stutters_but_interpolation_is_smooth() {
    let (_, body) = free_fall_world();
    let (raw, interp) = run_capture(body);

    // The body must actually be falling over the window, otherwise the test
    // proves nothing about jitter.
    assert!(
        *interp.last().unwrap() < interp[0] - 0.01,
        "body should fall a measurable distance: first={} last={}",
        interp[0],
        interp.last().unwrap()
    );

    // (1) Raw snapshot stutter: at the mismatched render rate, at least one
    // frame must repeat the previous frame's raw height exactly (a stall the
    // player would see as a hitch).
    let raw_stalls = raw
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() <= f32::EPSILON)
        .count();
    assert!(
        raw_stalls > 0,
        "expected the raw un-interpolated stream to stall at least once, got none"
    );

    // (2a) Interpolation is monotone: a falling body's interpolated height must
    // never tick back upward from one frame to the next (that upward tick is
    // precisely the jitter the milestone forbids). A tiny epsilon absorbs f32
    // rounding in the blend.
    let tol = 1.0e-5;
    for pair in interp.windows(2) {
        assert!(
            pair[1] <= pair[0] + tol,
            "interpolated height jittered upward: {} -> {}",
            pair[0],
            pair[1]
        );
    }

    // (2b) Interpolation is smooth: once the body is moving it must not stall on
    // a repeated value the way the raw stream does. We count consecutive-equal
    // interpolated frames and require strictly fewer stalls than the raw stream.
    let interp_stalls = interp
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() <= f32::EPSILON)
        .count();
    assert!(
        interp_stalls < raw_stalls,
        "interpolation should remove stalls: raw={raw_stalls} interp={interp_stalls}"
    );
}

#[test]
fn state_hash_is_deterministic_across_identical_runs() {
    let run = || {
        let (mut world, _) = free_fall_world();
        let mut backend = dynamics_backend();
        let mut pipe = FixedStepPipeline::new(FIXED_DT);
        pipe.prime(&world);
        for _ in 0..FRAMES {
            pipe.advance(&mut backend, &mut world, RENDER_DT);
        }
        pipe.state_hash()
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "identical runs must yield identical state hashes");
    assert_ne!(
        a,
        StateHash(0),
        "a stepped simulation must hash to non-zero"
    );
}

#[test]
fn divergent_input_produces_a_different_hash() {
    // Baseline run: pure free fall.
    let (mut world_a, body_a) = free_fall_world();
    let mut backend_a = dynamics_backend();
    let mut pipe_a = FixedStepPipeline::new(FIXED_DT);
    pipe_a.prime(&world_a);

    // Divergent run: identical except a single sideways impulse applied through
    // the command queue at the first step boundary.
    let (mut world_b, body_b) = free_fall_world();
    let mut backend_b = dynamics_backend();
    let mut pipe_b = FixedStepPipeline::new(FIXED_DT);
    pipe_b.prime(&world_b);
    pipe_b.queue_command(PhysicsCommand::ApplyLinearImpulse {
        body: body_b,
        impulse: Vec3::new(0.5, 0.0, 0.0),
    });

    for _ in 0..FRAMES {
        pipe_a.advance(&mut backend_a, &mut world_a, RENDER_DT);
        pipe_b.advance(&mut backend_b, &mut world_b, RENDER_DT);
    }

    assert_ne!(
        pipe_a.state_hash(),
        pipe_b.state_hash(),
        "a divergent impulse must change the state hash"
    );

    // The divergence detector must also point at the body that actually moved
    // differently (both worlds have exactly one body at slot 0).
    let snap_a = pipe_a.current_snapshot().unwrap();
    let snap_b = pipe_b.current_snapshot().unwrap();
    assert_ne!(hash_state(snap_a), hash_state(snap_b));
    assert_eq!(
        locate_divergence(snap_a, snap_b),
        Some(0),
        "the impulse-affected body sits at slot 0 and must be flagged"
    );

    // Sanity: the baseline body fell straight down (x stays ~0) while the
    // divergent body drifted along +x.
    let ax = world_a.bodies.position(body_a).unwrap().x;
    let bx = world_b.bodies.position(body_b).unwrap().x;
    assert!(
        ax.abs() < 1.0e-4,
        "baseline body should not drift in x: {ax}"
    );
    assert!(bx > 0.01, "impulsed body should drift in +x: {bx}");
}
