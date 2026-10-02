//! Real-device parity for the deterministic, closed-form core of the
//! sprite-atlas shelf rectangle packer:
//! [`GpuAtlasPacking`](prism_volumetric_gpu::GpuAtlasPacking) must reproduce the
//! `CPU` golden
//! [`atlas_packing`](prism_render_architecture::particle::atlas_packing) numeric
//! spine routine for routine: the saturating right/bottom edge
//! ([`PackedRect::right`](prism_render_architecture::particle::atlas_packing::PackedRect::right),
//! [`PackedRect::bottom`](prism_render_architecture::particle::atlas_packing::PackedRect::bottom)),
//! the normalized `UV` rectangle
//! ([`PackedRect::uv_rect`](prism_render_architecture::particle::atlas_packing::PackedRect::uv_rect)),
//! the half-open overlap verdict
//! ([`rects_overlap`](prism_render_architecture::particle::atlas_packing::rects_overlap)),
//! the single-rectangle coverage fraction
//! ([`occupancy`](prism_render_architecture::particle::atlas_packing::occupancy)),
//! the packer's inlined fits predicate, and the one-step shelf placement
//! transition the golden packer loop runs per rectangle.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer and boolean routines (the saturating edges, the overlap and fits
//! verdicts and the shelf placement) are bit-exact and are compared with `==`.
//! The continuous entries (the `UV` rectangle and the coverage fraction) thread
//! through a `u32`-to-`f32` widen and a divide, so `CPU` and `GPU` are not
//! guaranteed bit-exact: a `GPU` may round a widen or a divide a hair
//! differently. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every continuous quantity. Every
//! fixture is placed clear of each branch boundary: the fits and overlap
//! operands sit well inside or well outside their thresholds, the saturating
//! edges are either comfortably below `u32::MAX` or deliberately saturated (and
//! then compared with `==`), and every `atlas` dimension is positive (one
//! zero-dimension degenerate fixture aside) with the widened products kept well
//! below the `2^24` exact range. The fixtures use no transcendental method
//! (only unsigned arithmetic and widening divides), and the random batch draws
//! from a host-side integer `LCG` so it needs no external math library.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::atlas_packing::cpu_reference;
use prism_volumetric_gpu::{AtlasPackingQuery, AtlasPackingResult, GpuAtlasPacking, GpuContext};

/// Absolute parity bound on every continuous quantity. A `GPU` may round a
/// widen or a divide a hair differently from the scalar reference, perturbing
/// the low mantissa bits by a few units in the last place; `1e-4` admits that
/// legal slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (coverage fractions
/// near one, `UV` coordinates near one) where a few units in the last place
/// exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether a `GPU` result matches the `CPU` reference, matching variant
/// against variant: the integer and boolean lanes compare exactly, the scalar
/// and `UV` lanes apply the continuous-field bound.
fn results_match(got: &AtlasPackingResult, want: &AtlasPackingResult) -> bool {
    match (got, want) {
        (AtlasPackingResult::Edge(a), AtlasPackingResult::Edge(b)) => a == b,
        (AtlasPackingResult::Flag(a), AtlasPackingResult::Flag(b)) => a == b,
        (
            AtlasPackingResult::Placement {
                fits: fa,
                x: xa,
                y: ya,
                cursor_x: cxa,
                cursor_y: cya,
                shelf_height: sha,
            },
            AtlasPackingResult::Placement {
                fits: fb,
                x: xb,
                y: yb,
                cursor_x: cxb,
                cursor_y: cyb,
                shelf_height: shb,
            },
        ) => fa == fb && xa == xb && ya == yb && cxa == cxb && cya == cyb && sha == shb,
        (AtlasPackingResult::Scalar(a), AtlasPackingResult::Scalar(b)) => approx(*a, *b),
        (AtlasPackingResult::Uv(a), AtlasPackingResult::Uv(b)) => {
            a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y))
        }
        _ => false,
    }
}

/// Evaluates a single query on device and asserts it matches the `CPU` golden.
fn check(gpu: &GpuAtlasPacking, ctx: &GpuContext, query: AtlasPackingQuery) {
    let want = cpu_reference(&query);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one query yields one result");
    assert!(
        results_match(&got[0], &want),
        "GPU result {:?} must match CPU {:?} for {:?}",
        got[0],
        want,
        query
    );
}

/// A tiny integer linear-congruential generator; only integer and shift work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Advances the generator and returns a raw 32-bit word.
fn lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// Draws an unsigned texel value in `[lo, hi)` from the generator.
fn rand_u32(state: &mut u64, lo: u32, hi: u32) -> u32 {
    lo + (lcg_u32(state) % (hi - lo))
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn right_edge_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // Ordinary edges well below the saturation boundary.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::RightEdge { x: 100, width: 50 },
    );
    check(&gpu, &ctx, AtlasPackingQuery::RightEdge { x: 0, width: 0 });
    // Saturating edge: `x + width` overflows and pins at `u32::MAX`.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::RightEdge {
            x: u32::MAX,
            width: 5,
        },
    );
}

#[test]
fn bottom_edge_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::BottomEdge { y: 200, height: 64 },
    );
    // Saturating edge: `y + height` overflows and pins at `u32::MAX`.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::BottomEdge {
            y: u32::MAX - 2,
            height: 10,
        },
    );
}

#[test]
fn fits_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // Clear accept: both padded extents land comfortably inside the atlas.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Fits {
            cursor_x: 10,
            cursor_y: 5,
            padded_width: 20,
            padded_height: 15,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
    // Clear reject: the padded width overruns the atlas by a wide margin.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Fits {
            cursor_x: 200,
            cursor_y: 5,
            padded_width: 100,
            padded_height: 15,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
}

#[test]
fn occupancy_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // Small operands keep the widened product well below the exact range.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Occupancy {
            width: 100,
            height: 50,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
    // Zero-area atlas returns a zero fraction (no division by zero).
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Occupancy {
            width: 10,
            height: 10,
            atlas_width: 0,
            atlas_height: 64,
        },
    );
}

#[test]
fn overlap_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // Clear overlap: the second rectangle sits squarely inside the first's span.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Overlap {
            a: [0, 0, 10, 10],
            b: [5, 5, 10, 10],
        },
    );
    // Edge-touching rectangles are half-open, so they do not overlap.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::Overlap {
            a: [0, 0, 10, 10],
            b: [10, 0, 10, 10],
        },
    );
}

#[test]
fn uv_rect_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // Clean power-of-two normalization: [0.25, 0.125, 0.75, 0.375].
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::UvRect {
            x: 64,
            y: 32,
            width: 128,
            height: 64,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
    // Zero-dimension atlas returns an all-zero UV rectangle.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::UvRect {
            x: 10,
            y: 10,
            width: 20,
            height: 20,
            atlas_width: 128,
            atlas_height: 0,
        },
    );
}

#[test]
fn shelf_placement_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    // No wrap: the rectangle fits the current shelf; cursor advances in place.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::ShelfPlacement {
            cursor_x: 10,
            cursor_y: 5,
            shelf_height: 30,
            padded_width: 20,
            padded_height: 15,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
    // Wrap: the rectangle overruns the row, so the cursor drops to a new shelf
    // at `cursor_y + shelf_height` and restarts at `x = 0`.
    check(
        &gpu,
        &ctx,
        AtlasPackingQuery::ShelfPlacement {
            cursor_x: 250,
            cursor_y: 5,
            shelf_height: 30,
            padded_width: 20,
            padded_height: 15,
            atlas_width: 256,
            atlas_height: 256,
        },
    );
}

/// Appends one query of every routine, drawing fixtures clear of each branch
/// boundary. `flip` steers the fits, overlap and shelf-placement cases into
/// their opposite branches so the batch exercises both verdicts and both the
/// wrapping and non-wrapping placement transitions.
fn push_suite(state: &mut u64, flip: bool, queries: &mut Vec<AtlasPackingQuery>) {
    queries.push(AtlasPackingQuery::RightEdge {
        x: rand_u32(state, 0, 4096),
        width: rand_u32(state, 1, 256),
    });
    queries.push(AtlasPackingQuery::BottomEdge {
        y: rand_u32(state, 0, 4096),
        height: rand_u32(state, 1, 256),
    });

    // Fits: accept branch when `!flip` (both extents well inside), reject branch
    // when `flip` (the padded width overruns the atlas by a wide margin).
    if flip {
        queries.push(AtlasPackingQuery::Fits {
            cursor_x: 500,
            cursor_y: 10,
            padded_width: rand_u32(state, 100, 200),
            padded_height: rand_u32(state, 10, 40),
            atlas_width: 512,
            atlas_height: 512,
        });
    } else {
        queries.push(AtlasPackingQuery::Fits {
            cursor_x: rand_u32(state, 0, 64),
            cursor_y: rand_u32(state, 0, 64),
            padded_width: rand_u32(state, 10, 60),
            padded_height: rand_u32(state, 10, 60),
            atlas_width: 512,
            atlas_height: 512,
        });
    }

    // Occupancy: small positive rectangle inside a positive atlas.
    queries.push(AtlasPackingQuery::Occupancy {
        width: rand_u32(state, 1, 256),
        height: rand_u32(state, 1, 256),
        atlas_width: rand_u32(state, 256, 1024),
        atlas_height: rand_u32(state, 256, 1024),
    });

    // Overlap: disjoint when `!flip` (the second rectangle sits far to the
    // right), clearly overlapping when `flip` (nested inside the first).
    if flip {
        queries.push(AtlasPackingQuery::Overlap {
            a: [0, 0, 40, 40],
            b: [10, 10, 40, 40],
        });
    } else {
        queries.push(AtlasPackingQuery::Overlap {
            a: [0, 0, 20, 20],
            b: [100, 100, 20, 20],
        });
    }

    // Shelf placement: wrap branch when `flip` (cursor near the right edge),
    // non-wrapping branch otherwise.
    if flip {
        queries.push(AtlasPackingQuery::ShelfPlacement {
            cursor_x: 500,
            cursor_y: rand_u32(state, 0, 64),
            shelf_height: rand_u32(state, 10, 48),
            padded_width: rand_u32(state, 20, 60),
            padded_height: rand_u32(state, 10, 48),
            atlas_width: 512,
            atlas_height: 1024,
        });
    } else {
        queries.push(AtlasPackingQuery::ShelfPlacement {
            cursor_x: rand_u32(state, 0, 64),
            cursor_y: rand_u32(state, 0, 64),
            shelf_height: rand_u32(state, 10, 48),
            padded_width: rand_u32(state, 20, 60),
            padded_height: rand_u32(state, 10, 48),
            atlas_width: 512,
            atlas_height: 1024,
        });
    }

    // UV rect: positive atlas, edges comfortably inside the exact range.
    let aw = rand_u32(state, 256, 1024);
    let ah = rand_u32(state, 256, 1024);
    queries.push(AtlasPackingQuery::UvRect {
        x: rand_u32(state, 0, aw / 2),
        y: rand_u32(state, 0, ah / 2),
        width: rand_u32(state, 1, aw / 2),
        height: rand_u32(state, 1, ah / 2),
        atlas_width: aw,
        atlas_height: ah,
    });

    // Keep the continuous generator advancing so successive rounds differ even
    // where the integer draws dominate.
    let _ = lcg(state);
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAtlasPacking::new(&ctx);
    let mut state = 0x_a71a_5aac_0bad_0001_u64 ^ 0x_1234_5678_9abc_def0;
    let mut queries = Vec::new();
    for round in 0..24 {
        push_suite(&mut state, round % 2 == 0, &mut queries);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_flag_true = false;
    let mut saw_flag_false = false;
    let mut saw_wrap = false;
    let mut saw_no_wrap = false;
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_reference(q);
        if let AtlasPackingResult::Flag(flag) = want {
            if flag {
                saw_flag_true = true;
            } else {
                saw_flag_false = true;
            }
        }
        // A shelf placement wraps iff the padded rectangle overruns the row at
        // the current cursor; detect both transitions from the query itself.
        if let AtlasPackingQuery::ShelfPlacement {
            cursor_x,
            padded_width,
            atlas_width,
            ..
        } = q
        {
            if cursor_x.saturating_add(*padded_width) > *atlas_width {
                saw_wrap = true;
            } else {
                saw_no_wrap = true;
            }
        }
        assert!(
            results_match(g, &want),
            "GPU result {g:?} must match CPU {want:?} for {q:?}"
        );
    }
    assert!(
        saw_flag_true && saw_flag_false,
        "the batch must exercise both boolean verdicts"
    );
    assert!(
        saw_wrap && saw_no_wrap,
        "the batch must exercise both shelf-placement transitions"
    );
}
