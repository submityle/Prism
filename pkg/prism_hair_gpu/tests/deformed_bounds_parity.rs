//! Real-device parity for the deformed scene-bounds reduction twin:
//! [`GpuHairDeformedBounds`] must reproduce the `CPU` golden
//! [`deformed_bounds`](prism_render_architecture::hair::gpu_scene_handoff::deformed_bounds)
//! — the per-axis `min` / `max` over every finite deformed render point and the
//! derived `center`, `half_extents` and bounding `radius` the `SceneBounds` row
//! carries.
//!
//! Coverage: the empty groom (zero-extent origin box via the host early return,
//! no dispatch), a single point (degenerate zero-extent box at that point), a
//! handful of points spanning all three axes, a batch that mixes `NaN` / `+inf`
//! / `-inf` components the fold must skip, an all-non-finite batch that still
//! collapses to the origin box, negative-coordinate clouds, and a large batch of
//! more than `256` points so the grid-stride load and the shared-memory tree
//! fold are both exercised past one workgroup.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel uses only comparisons,
//! add / multiply, `sqrt` and workgroup shared memory in the portable
//! core-`WGSL` subset, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `min` / `max` fold selects existing finite coordinates with no
//! arithmetic, so `center` and `half_extents` are bit-exact up to the final
//! `(min +/- max) * 0.5`; the `radius` square root may differ by a few
//! low-mantissa `ULP`. Every field is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3` — tight enough to fail a genuinely wrong port (a dropped
//! point, a swapped axis, a missing non-finite skip), loose enough to admit the
//! square-root rounding. Every point is an explicit `[f32; 3]` literal (never
//! `sin`/`cos`), and the empty / all-non-finite reads are the exact origin box
//! so a no-op kernel could not pass.
//!
//! This is the crate's second **many-inputs-to-one-output** reduction twin
//! (after [`GpuHairAnalysisReduce`]) and its first per-axis `AABB` `min` / `max`
//! fold, validating the scene-bounds half of the sim → render-graph handoff the
//! golden [`deformed_bounds`] contract names.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::deformed_bounds::{reference_deformed_bounds, GpuHairDeformedBounds};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::gpu_scene::SceneBounds;

/// Builds a context or prints a skip notice and returns `None` on hosts without
/// a `wgpu` adapter.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping deformed bounds parity: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts one scalar matches the golden within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got} vs cpu {expected} (abs {abs_diff}, rel {rel_diff})",
    );
}

/// Asserts the whole bounds row matches the golden field-for-field.
fn assert_bounds(got: SceneBounds, expected: SceneBounds, label: &str) {
    assert_close(
        got.center[0],
        expected.center[0],
        &format!("{label}.center.x"),
    );
    assert_close(
        got.center[1],
        expected.center[1],
        &format!("{label}.center.y"),
    );
    assert_close(
        got.center[2],
        expected.center[2],
        &format!("{label}.center.z"),
    );
    assert_close(got.radius, expected.radius, &format!("{label}.radius"));
    assert_close(
        got.half_extents[0],
        expected.half_extents[0],
        &format!("{label}.half_extents.x"),
    );
    assert_close(
        got.half_extents[1],
        expected.half_extents[1],
        &format!("{label}.half_extents.y"),
    );
    assert_close(
        got.half_extents[2],
        expected.half_extents[2],
        &format!("{label}.half_extents.z"),
    );
}

/// An empty groom folds to the zero-extent origin box through the host
/// early-return path that never touches the device.
#[test]
fn gpu_empty_is_origin_box() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    let points: [[f32; 3]; 0] = [];
    let got = folder.eval(&ctx, &points);
    assert_bounds(got, reference_deformed_bounds(&points), "empty");
    assert_close(got.center[0], 0.0, "empty.center.x");
    assert_close(got.radius, 0.0, "empty.radius");
    assert_close(got.half_extents[0], 0.0, "empty.half_extents.x");
}

/// A single point is a degenerate zero-extent box centred at that point.
#[test]
fn gpu_single_point_is_degenerate() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    let points = [[2.5_f32, -3.0, 7.25]];
    let got = folder.eval(&ctx, &points);
    let expected = reference_deformed_bounds(&points);
    assert_bounds(got, expected, "single");
    assert_close(got.center[0], 2.5, "single.center.x");
    assert_close(got.center[1], -3.0, "single.center.y");
    assert_close(got.center[2], 7.25, "single.center.z");
    assert_close(got.radius, 0.0, "single.radius");
}

/// A spread of points over all three axes yields the enclosing box.
#[test]
fn gpu_matches_golden_small_cloud() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    let points = [
        [-1.0_f32, 2.0, 0.5],
        [3.0, -4.0, 1.5],
        [0.25, 0.75, -2.0],
        [5.0, 1.0, 4.0],
        [-2.5, 6.0, -1.0],
    ];
    let got = folder.eval(&ctx, &points);
    assert_bounds(got, reference_deformed_bounds(&points), "cloud");
}

/// Points with any non-finite component are skipped, so the box matches the
/// box of only the finite points.
#[test]
fn gpu_skips_non_finite_points() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    let points = [
        [1.0_f32, 1.0, 1.0],
        [f32::NAN, 10.0, 10.0],
        [2.0, 3.0, -1.0],
        [100.0, f32::INFINITY, 0.0],
        [-5.0, -2.0, 4.0],
        [0.0, 0.0, f32::NEG_INFINITY],
    ];
    let got = folder.eval(&ctx, &points);
    let expected = reference_deformed_bounds(&points);
    assert_bounds(got, expected, "mixed");
    // The surviving finite points are (1,1,1), (2,3,-1), (-5,-2,4): verify the
    // extents come only from those, so the skip really happened.
    assert_close(got.center[0], (-5.0 + 2.0) * 0.5, "mixed.center.x");
    assert_close(
        got.half_extents[0],
        (2.0 - -5.0) * 0.5,
        "mixed.half_extents.x",
    );
}

/// An all-non-finite batch still collapses to the zero-extent origin box.
#[test]
fn gpu_all_non_finite_is_origin_box() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    let points = [
        [f32::NAN, 0.0, 0.0],
        [0.0, f32::INFINITY, 0.0],
        [0.0, 0.0, f32::NEG_INFINITY],
    ];
    let got = folder.eval(&ctx, &points);
    assert_bounds(got, reference_deformed_bounds(&points), "all_bad");
    assert_close(got.center[0], 0.0, "all_bad.center.x");
    assert_close(got.radius, 0.0, "all_bad.radius");
    assert_close(got.half_extents[1], 0.0, "all_bad.half_extents.y");
}

/// A large batch of more than `256` points crosses the single workgroup, so the
/// grid-stride load and the shared-memory tree fold are both exercised.
#[test]
fn gpu_matches_golden_large_batch() {
    let Some(ctx) = context_or_skip() else {
        return;
    };
    let folder = GpuHairDeformedBounds::new(&ctx);
    // Deterministic integer-lattice spiral; explicit arithmetic, no transcendentals.
    let mut points: Vec<[f32; 3]> = Vec::with_capacity(1000);
    let mut i = 0_i32;
    while i < 1000 {
        let x = ((i * 7) % 211 - 105) as f32 * 0.5;
        let y = ((i * 13) % 307 - 153) as f32 * 0.25;
        let z = ((i * 5) % 97 - 48) as f32 * 0.75;
        points.push([x, y, z]);
        i += 1;
    }
    let got = folder.eval(&ctx, &points);
    assert_bounds(got, reference_deformed_bounds(&points), "large");
}
