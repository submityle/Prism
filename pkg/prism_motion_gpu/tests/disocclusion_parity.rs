//! Real-device parity for the per-pixel disocclusion (history-rejection) test.
//!
//! Each test builds `(current, history)` surface-sample pairs, runs the CPU
//! golden [`classify`], runs the `GPU` [`GpuDisocclusion`] kernel on a real
//! adapter, and asserts they agree. The discrete verdict -- the accepted flag
//! and the rejection reason bits that drive the `DISOCCLUDED` flag -- is
//! compared **exactly**; the graded confidence is compared within a tight
//! tolerance because a fast-math backend may contract the normal dot product to
//! an FMA. The depth signal uses no fused arithmetic, so its contribution is
//! bit-exact; the suite is constructed so the accepted flag and reason bits
//! never depend on an FMA-sensitive boundary. The suite skips gracefully when
//! no adapter is available.

use prism_motion_gpu::context::GpuContext;
use prism_motion_gpu::disocclusion::{DisocclusionResult, GpuDisocclusion};
use prism_render_architecture::motion::disocclusion::{
    classify, DisocclusionParams, RejectionReasons, SurfacePoint,
};

/// Confidence tolerance: well above the normal-dot FMA noise (a few ULP, scaled
/// by the `1 / (1 - threshold)` remap) yet far below any real formula error.
const CONF_TOL: f32 = 1e-5;

const UP: [f32; 3] = [0.0, 1.0, 0.0];
const RIGHT: [f32; 3] = [1.0, 0.0, 0.0];

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Deterministic LCG mapped to `f32`, no transcendentals.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// Uniform-ish `f32` in `[0, 1)` from the high bits.
    fn unit(&mut self) -> f32 {
        let v = self.next_u32() >> 8;
        (v as f32) / (16_777_216.0_f32)
    }
}

/// Asserts a device result matches the golden verdict: discrete fields exactly,
/// confidence within [`CONF_TOL`].
fn assert_matches(
    gpu: DisocclusionResult,
    golden_c: SurfacePoint,
    golden_h: SurfacePoint,
    params: DisocclusionParams,
    label: &str,
) {
    let golden = classify(golden_c, golden_h, params);
    assert_eq!(gpu.accepted, golden.accepted, "{label}: accepted mismatch");
    assert_eq!(
        gpu.reasons,
        golden.reasons.bits(),
        "{label}: reasons mismatch (gpu {:#x} golden {:#x})",
        gpu.reasons,
        golden.reasons.bits()
    );
    assert!(
        (gpu.confidence - golden.confidence).abs() <= CONF_TOL,
        "{label}: confidence {} vs golden {}",
        gpu.confidence,
        golden.confidence
    );
}

#[test]
fn coherent_surface_is_accepted() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = [SurfacePoint::new(1.0, UP, 42)];
    let history = [SurfacePoint::new(1.0, UP, 42)];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "coherent");
    // Anti-vacuous: a fully coherent sample is accepted with full confidence.
    assert!(out[0].accepted);
    assert!((out[0].confidence - 1.0).abs() <= CONF_TOL);
    assert_eq!(out[0].reasons, 0);
}

#[test]
fn surface_mismatch_is_rejected() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = [SurfacePoint::new(1.0, UP, 7)];
    let history = [SurfacePoint::new(1.0, UP, 8)];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "surface_mismatch");
    // Anti-vacuous: mismatch hard-rejects to zero confidence with the bit set.
    assert!(!out[0].accepted);
    assert_eq!(out[0].confidence, 0.0);
    assert_ne!(
        out[0].reasons & RejectionReasons::SURFACE_MISMATCH.bits(),
        0
    );
}

#[test]
fn surface_ignored_when_disabled() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::new(0.05, 0.9, false, 0.5);
    let current = [SurfacePoint::new(1.0, UP, 7)];
    let history = [SurfacePoint::new(1.0, UP, 8)];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "surface_disabled");
    // Anti-vacuous: with the check disabled the id gap no longer rejects.
    assert!(out[0].accepted);
    assert_eq!(
        out[0].reasons & RejectionReasons::SURFACE_MISMATCH.bits(),
        0
    );
}

#[test]
fn depth_discontinuity_is_rejected() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = [SurfacePoint::new(1.0, UP, 1)];
    let history = [SurfacePoint::new(5.0, UP, 1)];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "depth_gap");
    // Anti-vacuous: a 4x depth gap collapses the depth signal.
    assert!(!out[0].accepted);
    assert_ne!(
        out[0].reasons & RejectionReasons::DEPTH_DISCONTINUITY.bits(),
        0
    );
}

#[test]
fn normal_discontinuity_is_rejected() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = [SurfacePoint::new(1.0, UP, 1)];
    let history = [SurfacePoint::new(1.0, RIGHT, 1)];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "normal_crease");
    // Anti-vacuous: orthogonal normals drop below the cosine threshold.
    assert!(!out[0].accepted);
    assert_ne!(
        out[0].reasons & RejectionReasons::NORMAL_DISCONTINUITY.bits(),
        0
    );
}

#[test]
fn graded_depth_and_normal_confidence_match_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    // Tolerance 0.1 so a 5% relative depth gap grades to half confidence.
    let p = DisocclusionParams::new(0.1, 0.9, true, 0.25);
    // cos 0.95 is halfway between the 0.9 threshold and 1.0 -> 0.5 confidence.
    let half_normal = [0.312_249_8, 0.95, 0.0];
    let current = [
        SurfacePoint::new(1.0, UP, 5), // 5% depth gap, matching normal
        SurfacePoint::new(1.0, UP, 9), // perfect depth, graded normal
    ];
    let history = [
        SurfacePoint::new(0.95, UP, 5),
        SurfacePoint::new(1.0, half_normal, 9),
    ];
    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_matches(out[0], current[0], history[0], p, "graded_depth");
    assert_matches(out[1], current[1], history[1], p, "graded_normal");
    // Anti-vacuous: both grade to ~0.5, above the 0.25 accept threshold.
    assert!((out[0].confidence - 0.5).abs() <= 2e-3);
    assert!((out[1].confidence - 0.5).abs() <= 2e-3);
    assert!(out[0].accepted && out[1].accepted);
}

#[test]
fn length_mismatch_returns_none_and_empty_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = [SurfacePoint::new(1.0, UP, 1)];
    assert!(
        kernel.classify(&ctx, &current, &[], p).is_none(),
        "length mismatch must return None"
    );
    let empty = kernel
        .classify(&ctx, &[], &[], p)
        .expect("empty matched lengths");
    assert!(empty.is_empty(), "empty input must yield empty output");
}

#[test]
fn large_multi_workgroup_batch_matches_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDisocclusion::new(&ctx);
    let p = DisocclusionParams::default(); // tol 0.05, cos 0.9, require match, accept 0.5

    // 203 pixels spans several 64-wide workgroups (and a partial final group).
    // The depth signal (bit-exact arithmetic) and the discrete normal/surface
    // signals (identical vs orthogonal normal, matching vs mismatched id) keep
    // the accepted flag and reason bits off any FMA-sensitive boundary.
    let count = 203usize;
    let mut rng = Lcg::new(0x51ed_c0de);
    let mut current = Vec::with_capacity(count);
    let mut history = Vec::with_capacity(count);
    for i in 0..count {
        let base_depth = 0.5 + rng.unit() * 4.0;
        // Relative gap in [0, ~0.12): straddles the 0.05 tolerance both ways.
        let gap = (rng.unit() - 0.5) * 0.12 * base_depth;
        let id = u64::from(rng.next_u32());
        let normal_matches = rng.unit() < 0.6;
        let id_matches = rng.unit() < 0.7;
        let hist_normal = if normal_matches { UP } else { RIGHT };
        let hist_id = if id_matches { id } else { id ^ 0x9e37_79b9 };
        current.push(SurfacePoint::new(base_depth, UP, id.max(1)));
        history.push(SurfacePoint::new(
            base_depth + gap,
            hist_normal,
            hist_id.max(1),
        ));
        let _ = i;
    }

    let out = kernel
        .classify(&ctx, &current, &history, p)
        .expect("matched lengths");
    assert_eq!(out.len(), count);
    let mut any_accepted = false;
    let mut any_rejected = false;
    let mut any_depth = false;
    let mut any_normal = false;
    let mut any_surface = false;
    for i in 0..count {
        assert_matches(out[i], current[i], history[i], p, "batch");
        if out[i].accepted {
            any_accepted = true;
        } else {
            any_rejected = true;
        }
        if out[i].reasons & RejectionReasons::DEPTH_DISCONTINUITY.bits() != 0 {
            any_depth = true;
        }
        if out[i].reasons & RejectionReasons::NORMAL_DISCONTINUITY.bits() != 0 {
            any_normal = true;
        }
        if out[i].reasons & RejectionReasons::SURFACE_MISMATCH.bits() != 0 {
            any_surface = true;
        }
    }
    // Anti-vacuous: the batch exercises every verdict path.
    assert!(any_accepted, "no sample was accepted");
    assert!(any_rejected, "no sample was rejected");
    assert!(any_depth, "no depth discontinuity surfaced");
    assert!(any_normal, "no normal discontinuity surfaced");
    assert!(any_surface, "no surface mismatch surfaced");
}
