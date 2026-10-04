//! Real-device parity for the cloth half-space time-of-impact twin:
//! [`GpuClothHalfSpaceToi`](prism_volumetric_gpu::cloth_half_space_toi::GpuClothHalfSpaceToi)
//! must reproduce the `CPU` golden `half_space_toi` of
//! `prism_physics_core::soft::collision::ccd` (which the render-layer
//! `prism_render_architecture::cloth::ccd` wraps verbatim).
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the zero-normal guard, the `s0 <= 0` already-behind short-circuit, the
//! `ds >= -EPS_COEF` parallel/receding guard, the linear crossing
//! `t = -s0 / ds` and the `t <= 1` acceptance clamped by `t.max(0)` — written
//! out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! The fixtures cover the zero-length normal miss, the start-behind
//! (`s0 <= 0`) hit at `t = 0`, the parallel-to-plane miss, a clean crossing
//! with `0 < t < 1`, a crossing just past the segment end (`t > 1`) that
//! misses, and a crossing landing near the `t = 1` boundary that hits. A mixed
//! batch validates the `std430` stride, and a `512`-query `LCG` sweep follows,
//! plus an empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! `hit` and `valid` are discrete and are compared exactly with `assert_eq!`.
//! The time of impact `t` is the only continuous channel and is compared with
//! an absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`, floor
//! `1e-6`), because the division `-s0 / ds` is a single `f32` operation whose
//! last bit can differ between host and device. Fixtures and the sweep keep
//! samples clear of the `s0 = 0`, `ds = -EPS_COEF` and `t = 1` knees so the
//! discrete `hit` channel never flips on a rounding tie.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::ccd`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_half_space_toi::{
    ClothHalfSpaceToiQuery, ClothHalfSpaceToiResult, GpuClothHalfSpaceToi,
};
use prism_volumetric_gpu::GpuContext;

/// Squared-length floor below which the normal has no defined plane.
const EPS_LEN_SQ: f32 = 1e-12;
/// Along-normal delta floor below which the segment is parallel/receding.
const EPS_COEF: f32 = 1e-12;

/// Relative-tolerance floor so tiny magnitudes do not demand absurd precision.
const REL_FLOOR: f32 = 1e-6;

/// Independent host re-implementation of `ccd::half_space_toi`, flattened into
/// a `(hit, t, valid)` triple exactly as the twin encodes the `Option<f32>`.
fn oracle(q: &ClothHalfSpaceToiQuery) -> ClothHalfSpaceToiResult {
    let prev = [q.prev_x, q.prev_y, q.prev_z];
    let curr = [q.curr_x, q.curr_y, q.curr_z];
    let normal = [q.normal_x, q.normal_y, q.normal_z];

    let len_sq = dot(normal, normal);
    if len_sq <= EPS_LEN_SQ {
        return miss();
    }
    let s0 = dot(normal, prev) - q.offset;
    if s0 <= 0.0 {
        return hit_at(0.0);
    }
    let delta = [curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2]];
    let ds = dot(normal, delta);
    if ds >= -EPS_COEF {
        return miss();
    }
    let t = -s0 / ds;
    if t <= 1.0 {
        hit_at(t.max(0.0))
    } else {
        miss()
    }
}

/// Three-component dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A no-hit answer: `hit = 0`, `t = 0`, `valid = 1`.
fn miss() -> ClothHalfSpaceToiResult {
    ClothHalfSpaceToiResult {
        hit: 0,
        t: 0.0,
        valid: 1,
    }
}

/// A hit answer at time `t`.
fn hit_at(t: f32) -> ClothHalfSpaceToiResult {
    ClothHalfSpaceToiResult {
        hit: 1,
        t,
        valid: 1,
    }
}

/// Absolute-or-relative closeness for the one continuous channel.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1e-4 {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= 1e-3
}

/// Asserts the GPU result for one query matches the oracle: discrete channels
/// exactly, the time of impact within tolerance (and only when it hits).
fn assert_result(got: ClothHalfSpaceToiResult, want: ClothHalfSpaceToiResult, label: &str) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert_eq!(got.hit, want.hit, "hit mismatch: {label}");
    if want.hit == 1 {
        assert!(
            close(got.t, want.t),
            "t mismatch: {label} got={} want={}",
            got.t,
            want.t
        );
    }
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuClothHalfSpaceToi,
    q: ClothHalfSpaceToiQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn zero_length_normal_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // A zero normal has no defined plane -> miss.
    let q = ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(oracle(&q).hit, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q, "zero_length_normal");
}

#[test]
fn start_behind_plane_hits_at_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // prev is already behind the plane (s0 = -1 <= 0) -> hit at t = 0.
    let q = ClothHalfSpaceToiQuery::new(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let want = oracle(&q);
    assert_eq!(want.hit, 1, "fixture sanity");
    assert!(close(want.t, 0.0), "fixture sanity: t near 0");
    assert_parity(&ctx, &gpu, q, "start_behind");
}

#[test]
fn parallel_segment_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // prev in front (s0 = 2) and the move is parallel to the plane (ds = 0) ->
    // never crosses -> miss.
    let q = ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 5.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    assert_eq!(oracle(&q).hit, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q, "parallel_segment");
}

#[test]
fn receding_segment_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // prev in front (s0 = 1) and moving further in front (ds = +3) -> miss.
    let q = ClothHalfSpaceToiQuery::new(0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    assert_eq!(oracle(&q).hit, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q, "receding_segment");
}

#[test]
fn clean_crossing_mid_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // s0 = 1, ds = -4 -> t = 0.25, well inside 0..1.
    let q = ClothHalfSpaceToiQuery::new(0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let want = oracle(&q);
    assert_eq!(want.hit, 1, "fixture sanity");
    assert!(close(want.t, 0.25), "fixture sanity: t = 0.25");
    assert_parity(&ctx, &gpu, q, "clean_crossing");
}

#[test]
fn crossing_past_segment_end_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // s0 = 2, ds = -1 -> t = 2.0 > 1 -> the plane is reached only after the
    // segment ends -> miss. 2.0 is a comfortable margin past the t = 1 knee.
    let q = ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    assert_eq!(oracle(&q).hit, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q, "crossing_past_end");
}

#[test]
fn crossing_near_segment_end_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // s0 = 0.9, ds = -1 -> t = 0.9, clearly below the t = 1 knee -> hit.
    let q = ClothHalfSpaceToiQuery::new(0.0, 0.9, 0.0, 0.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0);
    let want = oracle(&q);
    assert_eq!(want.hit, 1, "fixture sanity");
    assert!(close(want.t, 0.9), "fixture sanity: t = 0.9");
    assert_parity(&ctx, &gpu, q, "crossing_near_end");
}

#[test]
fn offset_plane_oblique_normal_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // A non-axis plane with a non-zero offset to exercise the full dot products.
    // normal = (0.6, 0.8, 0), offset = 1.0.
    // prev = (2, 2, 0): s0 = 0.6*2 + 0.8*2 - 1 = 1.8 (in front).
    // curr = (-1, -1, 0): ds = 0.6*(-3) + 0.8*(-3) = -4.2 -> t = 1.8/4.2.
    let q = ClothHalfSpaceToiQuery::new(2.0, 2.0, 0.0, -1.0, -1.0, 0.0, 0.6, 0.8, 0.0, 1.0);
    let want = oracle(&q);
    assert_eq!(want.hit, 1, "fixture sanity");
    assert_parity(&ctx, &gpu, q, "offset_oblique");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    // A >=2-element batch mixing every branch validates the std430 stride end
    // to end: zero normal, start-behind, parallel, clean crossing, past-end,
    // and an oblique offset plane.
    let queries = vec![
        ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ClothHalfSpaceToiQuery::new(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 5.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ClothHalfSpaceToiQuery::new(0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ClothHalfSpaceToiQuery::new(0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ClothHalfSpaceToiQuery::new(2.0, 2.0, 0.0, -1.0, -1.0, 0.0, 0.6, 0.8, 0.0, 1.0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("mixed batch index {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothHalfSpaceToi::new(&ctx);
    let mut rng = Lcg::new(0x1E_7C_55_A3);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // A well-conditioned normal, well away from the zero-length knee.
        let nx = rng.next_range(-1.0, 1.0);
        let ny = rng.next_range(-1.0, 1.0);
        let nz = rng.next_range(-1.0, 1.0);
        let normal = [nx, ny, nz];
        let len_sq = nx * nx + ny * ny + nz * nz;
        if len_sq < 0.25 {
            // Reject near-degenerate normals so the plane is well defined.
            continue;
        }

        let prev = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let curr = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let offset = rng.next_range(-2.0, 2.0);

        // Reject samples sitting on any of the three knees so the discrete hit
        // channel cannot flip on a rounding tie between host and device.
        let s0 = dot(normal, prev) - offset;
        if s0.abs() < 1e-2 {
            continue;
        }
        let delta = [curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2]];
        let ds = dot(normal, delta);
        if ds.abs() < 1e-2 {
            continue;
        }
        if s0 > 0.0 && ds < 0.0 {
            let t = -s0 / ds;
            if (t - 1.0).abs() < 1e-2 {
                continue;
            }
        }

        queries.push(ClothHalfSpaceToiQuery::new(
            prev[0], prev[1], prev[2], curr[0], curr[1], curr[2], nx, ny, nz, offset,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
