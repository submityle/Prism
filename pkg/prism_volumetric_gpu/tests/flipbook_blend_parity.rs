//! Real-device parity for the flipbook sub-frame blend twin:
//! [`GpuFlipbookBlend`](prism_volumetric_gpu::flipbook_blend::GpuFlipbookBlend)
//! must reproduce the `CPU` golden
//! [`flipbook_blend`](prism_render_architecture::particle::flipbook_blend)
//! across all three input kinds (`phase`, `age`, `life`), both wrap modes
//! (`Clamp` holds the last frame, `Loop` wraps modulo the frame count), the
//! `frames == 0` static-cell degenerate case, negative inputs that clamp to a
//! still first frame, zero atlas dimensions and a batch of random samples —
//! each compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The frame indices are integer arithmetic (a `floor`, a `u32` cast, a modulo
//! or a clamp), so they reproduce bit for bit and are asserted with exact `==`.
//! The blend weight and the `UV` rectangle coordinates are a `floor`-based
//! fract and a few multiplies/divides by the integer grid dimensions; `CPU` and
//! `GPU` evaluate the same closed form, but a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! `ULP`. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` fields — loose enough to admit a legal fused
//! multiply-add contraction, yet tight enough to fail a genuinely wrong port (a
//! swapped wrap branch, a transposed `col`/`row`, a dropped `+ 1` frame).
//!
//! Every fixture places its phase comfortably in the *middle* of a frame
//! (fractional part well away from `0.0` and `1.0`) so a legal `ULP` wobble in
//! an `age * fps` or `life * frames` product never floors to a different frame
//! than the reference; the integer-exact assertion stays robust.
//!
//! Provenance: Prism flipbook sub-frame blend design (section 15); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::flipbook_blend::{
    blend_for_age, blend_for_life, blend_for_phase, sample_rects, AtlasLayout, WrapMode,
};
use prism_volumetric_gpu::flipbook_blend::{
    FlipbookQuery, FlipbookResult, FlipbookSample, GpuFlipbookBlend,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the `f32` blend and `UV` fields. A `GPU` may fuse
/// a multiply-add the scalar reference leaves separate; `1e-4` admits that
/// legal slack while still failing a genuinely wrong port.
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

/// Returns whether two `[u_min, v_min, u_max, v_max]` rectangles agree within
/// the `close` bound on every component.
fn rect_close(a: [f32; 4], b: [f32; 4]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]) && close(a[3], b[3])
}

/// Computes the reference [`FlipbookResult`] for one sample under the batch's
/// shared configuration, so the `GPU` output can be checked against it.
fn reference(
    layout: AtlasLayout,
    frames: u32,
    fps: f32,
    wrap: WrapMode,
    sample: FlipbookSample,
) -> FlipbookResult {
    let blend = match sample {
        FlipbookSample::Phase(p) => blend_for_phase(p, frames, wrap),
        FlipbookSample::Age(a) => blend_for_age(a, fps, frames, wrap),
        FlipbookSample::Life(l) => blend_for_life(l, frames, wrap),
    };
    let (rect_a, rect_b, _) = sample_rects(&layout, &blend);
    FlipbookResult {
        blend,
        rect_a,
        rect_b,
    }
}

/// Runs the `GPU` flipbook blend over `query` and asserts sample-for-sample
/// parity against the `CPU` golden, returning the `GPU` results for any extra
/// per-test assertions.
fn check(ctx: &GpuContext, gpu: &GpuFlipbookBlend, query: &FlipbookQuery) -> Vec<FlipbookResult> {
    let got = gpu.eval(ctx, query);
    assert_eq!(
        got.len(),
        query.samples.len(),
        "result count must match the sample count"
    );

    for (idx, (g, &sample)) in got.iter().zip(query.samples.iter()).enumerate() {
        let w = reference(query.layout, query.frames, query.fps, query.wrap, sample);
        assert_eq!(
            g.blend.frame_a, w.blend.frame_a,
            "sample {idx} frame_a: gpu {} vs cpu {} ({sample:?})",
            g.blend.frame_a, w.blend.frame_a
        );
        assert_eq!(
            g.blend.frame_b, w.blend.frame_b,
            "sample {idx} frame_b: gpu {} vs cpu {} ({sample:?})",
            g.blend.frame_b, w.blend.frame_b
        );
        assert!(
            close(g.blend.blend, w.blend.blend),
            "sample {idx} blend: gpu {} vs cpu {} ({sample:?})",
            g.blend.blend,
            w.blend.blend
        );
        assert!(
            rect_close(g.rect_a, w.rect_a),
            "sample {idx} rect_a: gpu {:?} vs cpu {:?} ({sample:?})",
            g.rect_a,
            w.rect_a
        );
        assert!(
            rect_close(g.rect_b, w.rect_b),
            "sample {idx} rect_b: gpu {:?} vs cpu {:?} ({sample:?})",
            g.rect_b,
            w.rect_b
        );
    }
    got
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

#[test]
fn phase_zero_half_and_mid_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 4,
            rows: 2,
        },
        frames: 8,
        fps: 12.0,
        wrap: WrapMode::Loop,
        // Phases sit mid-frame (0.0, 0.5, 0.3) so the floored index is robust.
        samples: vec![
            FlipbookSample::Phase(0.0),
            FlipbookSample::Phase(0.5),
            FlipbookSample::Phase(2.3),
        ],
    };
    let got = check(&ctx, &gpu, &query);
    assert_eq!((got[0].blend.frame_a, got[0].blend.frame_b), (0, 1));
    assert_eq!((got[2].blend.frame_a, got[2].blend.frame_b), (2, 3));
}

#[test]
fn loop_and_clamp_resolve_the_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    let layout = AtlasLayout {
        columns: 4,
        rows: 2,
    };
    // Phase 7.3 sits in the last frame of an 8-frame sheet; Loop wraps the next
    // cell to 0, Clamp holds it on the last cell.
    let loop_q = FlipbookQuery {
        layout,
        frames: 8,
        fps: 12.0,
        wrap: WrapMode::Loop,
        samples: vec![FlipbookSample::Phase(7.3)],
    };
    let got_loop = check(&ctx, &gpu, &loop_q);
    assert_eq!(
        (got_loop[0].blend.frame_a, got_loop[0].blend.frame_b),
        (7, 0)
    );

    let clamp_q = FlipbookQuery {
        layout,
        frames: 8,
        fps: 12.0,
        wrap: WrapMode::Clamp,
        samples: vec![FlipbookSample::Phase(7.3), FlipbookSample::Phase(20.3)],
    };
    let got_clamp = check(&ctx, &gpu, &clamp_q);
    assert_eq!(
        (got_clamp[0].blend.frame_a, got_clamp[0].blend.frame_b),
        (7, 7)
    );
    assert_eq!(
        (got_clamp[1].blend.frame_a, got_clamp[1].blend.frame_b),
        (7, 7)
    );
}

#[test]
fn age_scales_by_fps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // age 0.65s at 4fps -> phase 2.6 (mid-frame); negative age clamps to a
    // still first frame.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 3,
            rows: 3,
        },
        frames: 9,
        fps: 4.0,
        wrap: WrapMode::Loop,
        samples: vec![FlipbookSample::Age(0.65), FlipbookSample::Age(-1.0)],
    };
    let got = check(&ctx, &gpu, &query);
    assert_eq!((got[0].blend.frame_a, got[0].blend.frame_b), (2, 3));
    assert_eq!((got[1].blend.frame_a, got[1].blend.frame_b), (0, 1));
}

#[test]
fn life_maps_normalized_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // life 0.3 over 10 frames -> phase 3.0-ish; use 0.33 -> 3.3 to stay
    // mid-frame and avoid the integer boundary a legal ULP wobble could cross.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 5,
            rows: 2,
        },
        frames: 10,
        fps: 24.0,
        wrap: WrapMode::Loop,
        samples: vec![FlipbookSample::Life(0.33), FlipbookSample::Life(0.77)],
    };
    let got = check(&ctx, &gpu, &query);
    assert_eq!((got[0].blend.frame_a, got[0].blend.frame_b), (3, 4));
    assert_eq!((got[1].blend.frame_a, got[1].blend.frame_b), (7, 8));
}

#[test]
fn uv_rects_lay_out_the_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // A 4x4 sheet: frame 5 is col 1 row 1 (centre-ish), frame 15 is bottom
    // right; parity on the rects pins the col/row split and the V direction.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 4,
            rows: 4,
        },
        frames: 16,
        fps: 30.0,
        wrap: WrapMode::Clamp,
        samples: vec![FlipbookSample::Phase(5.4), FlipbookSample::Phase(14.5)],
    };
    let got = check(&ctx, &gpu, &query);
    assert!(
        rect_close(got[0].rect_a, [0.25, 0.25, 0.5, 0.5]),
        "frame 5 should be the col-1 row-1 cell, got {:?}",
        got[0].rect_a
    );
}

#[test]
fn zero_frames_is_static() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // With 0 frames the sheet is a single static cell: both frames 0, blend 0.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 4,
            rows: 4,
        },
        frames: 0,
        fps: 12.0,
        wrap: WrapMode::Loop,
        samples: vec![
            FlipbookSample::Phase(3.0),
            FlipbookSample::Age(2.0),
            FlipbookSample::Life(0.5),
        ],
    };
    let got = check(&ctx, &gpu, &query);
    for r in &got {
        assert_eq!((r.blend.frame_a, r.blend.frame_b), (0, 0));
        assert!(close(r.blend.blend, 0.0));
    }
}

#[test]
fn negative_phase_clamps_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 4,
            rows: 2,
        },
        frames: 8,
        fps: 12.0,
        wrap: WrapMode::Loop,
        samples: vec![FlipbookSample::Phase(-5.0)],
    };
    let got = check(&ctx, &gpu, &query);
    assert_eq!((got[0].blend.frame_a, got[0].blend.frame_b), (0, 1));
    assert!(close(got[0].blend.blend, 0.0));
}

#[test]
fn zero_dimensions_avoid_division_by_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // A 0x0 layout is treated as 1x1, so every frame maps to the full sheet.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 0,
            rows: 0,
        },
        frames: 4,
        fps: 12.0,
        wrap: WrapMode::Clamp,
        // Phase 0.5 lands on frame 0, whose cell is the whole sheet under a
        // 1x1 effective layout; higher frames walk off the single cell in V,
        // so frame 0 is the one that must map to the full `[0, 0, 1, 1]` rect.
        samples: vec![FlipbookSample::Phase(0.5)],
    };
    let got = check(&ctx, &gpu, &query);
    assert!(rect_close(got[0].rect_a, [0.0, 0.0, 1.0, 1.0]));
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    // An empty sample list issues no dispatch (a storage buffer cannot be
    // zero-sized) and must return an empty result.
    let query = FlipbookQuery {
        layout: AtlasLayout {
            columns: 4,
            rows: 4,
        },
        frames: 8,
        fps: 12.0,
        wrap: WrapMode::Loop,
        samples: Vec::new(),
    };
    let got = gpu.eval(&ctx, &query);
    assert!(got.is_empty(), "an empty batch stays empty");
}

#[test]
fn random_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFlipbookBlend::new(&ctx);
    let mut state = 0x5eed_f11b_00c0_ffee_u64;

    // Sweep a spread of atlas layouts, frame counts and wrap modes, each with a
    // mixed batch of phase/age/life samples whose phase is forced mid-frame
    // (base integer plus a 0.1..0.9 fract) so the integer-exact frame assertion
    // never straddles a boundary a legal ULP wobble could cross.
    let configs = [
        (
            AtlasLayout {
                columns: 4,
                rows: 4,
            },
            16u32,
            24.0f32,
            WrapMode::Loop,
        ),
        (
            AtlasLayout {
                columns: 8,
                rows: 1,
            },
            8,
            12.0,
            WrapMode::Clamp,
        ),
        (
            AtlasLayout {
                columns: 3,
                rows: 5,
            },
            15,
            30.0,
            WrapMode::Loop,
        ),
        (
            AtlasLayout {
                columns: 6,
                rows: 2,
            },
            11,
            48.0,
            WrapMode::Clamp,
        ),
    ];

    for &(layout, frames, fps, wrap) in &configs {
        let mut samples = Vec::new();
        for _ in 0..24 {
            // A mid-frame phase target in [0, frames): base cell + safe fract.
            let base = (lcg(&mut state) * frames as f32).floor();
            let fract = 0.1 + lcg(&mut state) * 0.8;
            let phase = base + fract;
            match (lcg(&mut state) * 3.0) as u32 {
                0 => samples.push(FlipbookSample::Phase(phase)),
                1 => samples.push(FlipbookSample::Age(phase / fps)),
                _ => samples.push(FlipbookSample::Life((phase / frames as f32).min(0.999))),
            }
        }
        let query = FlipbookQuery {
            layout,
            frames,
            fps,
            wrap,
            samples,
        };
        check(&ctx, &gpu, &query);
    }
}
