//! Real-device parity for the `atlas` mip-padding twin:
//! [`GpuAtlasMipPadding`] must reproduce the `CPU` golden
//! [`atlas_mip_padding`](prism_render_architecture::particle::atlas_mip_padding)
//! pure-integer subset bit for bit across random batches, every border
//! [`PadMode`](prism_render_architecture::particle::atlas_mip_padding::PadMode)
//! with negative, out-of-range and edge coordinates, the mip-level and
//! safe-level counters at their shift/leading-zero edges, the saturating rect
//! accessors, a single query, and the degenerate empty batch.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned operation is exact integer arithmetic — a bounded left shift,
//! a leading-zero count, a signed clamp, a signed mirror fold and saturating
//! `u32` add/subtract — so the `CPU` and `GPU` agree bit for bit. The parity
//! test therefore asserts a strict `==` on every output word with no tolerance:
//! any mismatch is a genuine port defect.
//!
//! # Oracle
//!
//! The expected value for each query is the golden `pub` function evaluated on
//! the same operands, so the twin is pinned directly against the reference
//! rather than a re-transcribed copy. The scalar operations compare component
//! `0`; the pad-rect operation compares all four `Rect` fields.
//!
//! # Fixtures
//!
//! The deterministic `u64` `LCG` fixtures draw operands from regimes that
//! exercise each operation's edges: mip-level counts spanning the `31`-bit shift
//! clamp, dimensions spanning zero and the leading-zero boundary, gutter
//! coordinates reaching negative, in-range and past-the-edge values under each
//! border mode, and rect origins/sizes that drive both the saturating and
//! non-saturating `u32` paths. The fixtures use pure integer arithmetic with no
//! external math library and no transcendental method; integer parity has no
//! boundary fuzz, so no reject-sampling margin is needed.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_mip_padding`
//! 的纯 `u32`/`i32` 子集真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::atlas_mip_padding::{
    gutter_source_index, max_safe_mip_levels, required_padding, PadMode, Rect,
};
use prism_volumetric_gpu::atlas_mip_padding::{
    GpuAtlasMipPadding, GpuAtlasPadOp, GpuAtlasPadQuery, GpuPadMode,
};
use prism_volumetric_gpu::GpuContext;

/// Every operation the twin supports, in a fixed order so a random batch cycles
/// through all six with each dispatch.
const OP_TABLE: [GpuAtlasPadOp; 6] = [
    GpuAtlasPadOp::RequiredPadding,
    GpuAtlasPadOp::MaxSafeMipLevels,
    GpuAtlasPadOp::GutterSourceIndex,
    GpuAtlasPadOp::RectRight,
    GpuAtlasPadOp::RectBottom,
    GpuAtlasPadOp::PadRect,
];

/// Every border mode, in a fixed order so the gutter fixtures cycle through all
/// three.
const MODE_TABLE: [GpuPadMode; 3] = [
    GpuPadMode::ClampEdge,
    GpuPadMode::Mirror,
    GpuPadMode::Transparent,
];

/// Maps the twin's border mode onto the golden [`PadMode`] for the oracle.
fn golden_mode(mode: GpuPadMode) -> PadMode {
    match mode {
        GpuPadMode::ClampEdge => PadMode::ClampEdge,
        GpuPadMode::Mirror => PadMode::Mirror,
        GpuPadMode::Transparent => PadMode::Transparent,
    }
}

/// A tiny deterministic `64`-bit linear-congruential generator so the "random"
/// fixtures are reproducible run to run without pulling in an external crate.
/// The constants are the common `PCG`/`MMIX` `LCG` multiplier and increment.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// A reproducible value in `[0, bound)` for a non-zero `bound`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// The golden expected result for one query: the `pub` reference function
/// evaluated on the same operands. Scalar operations fill component `0`; the
/// pad-rect operation fills all four `Rect` fields.
fn cpu_expected(q: &GpuAtlasPadQuery) -> [u32; 4] {
    match q.op {
        GpuAtlasPadOp::RequiredPadding => [required_padding(q.x), 0, 0, 0],
        GpuAtlasPadOp::MaxSafeMipLevels => [max_safe_mip_levels(q.w, q.h), 0, 0, 0],
        GpuAtlasPadOp::GutterSourceIndex => [
            gutter_source_index(q.coord, q.w, golden_mode(q.mode)),
            0,
            0,
            0,
        ],
        GpuAtlasPadOp::RectRight => [Rect::new(q.x, q.y, q.w, q.h).right(), 0, 0, 0],
        GpuAtlasPadOp::RectBottom => [Rect::new(q.x, q.y, q.w, q.h).bottom(), 0, 0, 0],
        GpuAtlasPadOp::PadRect => {
            let r = Rect::new(q.x, q.y, q.w, q.h).pad_rect(q.pad);
            [r.x, r.y, r.w, r.h]
        }
    }
}

/// Runs one batch on the device and asserts every lane matches the golden
/// function exactly.
fn check_batch(
    label: &str,
    engine: &GpuAtlasMipPadding,
    ctx: &GpuContext,
    queries: &[GpuAtlasPadQuery],
) {
    let gpu = engine.run(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "{label}: result count");
    for (i, q) in queries.iter().enumerate() {
        let cpu = cpu_expected(q);
        assert_eq!(
            gpu[i], cpu,
            "{label}[{i}] op {op:?} mode {mode:?} x={x} y={y} w={w} h={h} coord={coord} pad={pad}: \
             cpu {cpu:?}, gpu {g:?}",
            op = q.op,
            mode = q.mode,
            x = q.x,
            y = q.y,
            w = q.w,
            h = q.h,
            coord = q.coord,
            pad = q.pad,
            g = gpu[i],
        );
    }
}

/// Builds a reproducible batch of `count` queries, cycling through all six
/// operations and drawing operands from edge-covering regimes per operation.
fn random_batch(rng: &mut Lcg, count: usize) -> Vec<GpuAtlasPadQuery> {
    let mut queries = Vec::with_capacity(count);
    for i in 0..count {
        let op = OP_TABLE[i % OP_TABLE.len()];
        let mut q = GpuAtlasPadQuery {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            coord: 0,
            pad: 0,
            mode: MODE_TABLE[(rng.below(3)) as usize],
            op,
        };
        match op {
            GpuAtlasPadOp::RequiredPadding => {
                // 0..=33 spans the no-gutter case and the 31-bit shift clamp.
                q.x = rng.below(34) as u32;
            }
            GpuAtlasPadOp::MaxSafeMipLevels => {
                // Include zero and the leading-zero boundaries up to a few k.
                q.w = rng.below(4_097) as u32;
                q.h = rng.below(4_097) as u32;
            }
            GpuAtlasPadOp::GutterSourceIndex => {
                let size = 1 + rng.below(64) as u32;
                q.w = size;
                // coord in [-2*size, 2*size): negative, in-range and past both
                // edges, so clamp/mirror/transparent all exercise their branches.
                let span = (4 * size) as i64;
                q.coord = (rng.next_u64() as i64 % span - (2 * size) as i64) as i32;
            }
            GpuAtlasPadOp::RectRight | GpuAtlasPadOp::RectBottom => {
                q.x = rng.below(100_000) as u32;
                q.y = rng.below(100_000) as u32;
                q.w = rng.below(100_000) as u32;
                q.h = rng.below(100_000) as u32;
            }
            GpuAtlasPadOp::PadRect => {
                q.x = rng.below(200) as u32;
                q.y = rng.below(200) as u32;
                q.w = rng.below(200) as u32;
                q.h = rng.below(200) as u32;
                q.pad = rng.below(32) as u32;
            }
        }
        queries.push(q);
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_batches() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping atlas-mip-padding parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    for (seed, count) in [(1usize, 72usize), (2, 128), (3, 257), (4, 66)] {
        let mut rng = Lcg::new(0xA715_0000_u64.wrapping_add(seed as u64));
        let queries = random_batch(&mut rng, count);
        check_batch(
            &format!("random batch {seed} (count={count})"),
            &engine,
            &ctx,
            &queries,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_gutter_modes_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    let mut queries = Vec::new();
    // Across each mode, sweep coordinates from well below zero to past the far
    // edge of several block sizes, including the zero-size degenerate case.
    for size in [0u32, 1, 2, 4, 5, 16] {
        for coord in -8i32..=20 {
            for &mode in &MODE_TABLE {
                queries.push(GpuAtlasPadQuery {
                    x: 0,
                    y: 0,
                    w: size,
                    h: 0,
                    coord,
                    pad: 0,
                    mode,
                    op: GpuAtlasPadOp::GutterSourceIndex,
                });
            }
        }
    }
    check_batch("gutter modes boundaries", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_required_and_max_levels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    let mut queries = Vec::new();
    // required_padding: 0 (no gutter), 1 (bilinear seam), the half-footprint
    // doublings, and absurd counts that hit the 31-bit shift clamp.
    for mip in [0u32, 1, 2, 3, 4, 5, 11, 31, 32, 33, 100] {
        queries.push(GpuAtlasPadQuery {
            x: mip,
            y: 0,
            w: 0,
            h: 0,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::RequiredPadding,
        });
    }
    // max_safe_mip_levels: zero (degenerate), the exact powers of two and the
    // one-less/one-more leading-zero boundaries, and a non-square block.
    for (w, h) in [
        (0u32, 0u32),
        (1, 1),
        (2, 2),
        (3, 4),
        (7, 8),
        (8, 8),
        (9, 16),
        (255, 256),
        (1024, 768),
        (4096, 4096),
    ] {
        queries.push(GpuAtlasPadQuery {
            x: 0,
            y: 0,
            w,
            h,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::MaxSafeMipLevels,
        });
    }
    check_batch("required and max levels", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_rect_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    let queries = vec![
        // right/bottom ordinary and saturating at u32::MAX.
        GpuAtlasPadQuery {
            x: 10,
            y: 20,
            w: 30,
            h: 40,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::RectRight,
        },
        GpuAtlasPadQuery {
            x: 10,
            y: 20,
            w: 30,
            h: 40,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::RectBottom,
        },
        GpuAtlasPadQuery {
            x: u32::MAX,
            y: u32::MAX,
            w: 5,
            h: 5,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::RectRight,
        },
        GpuAtlasPadQuery {
            x: u32::MAX,
            y: u32::MAX,
            w: 5,
            h: 5,
            coord: 0,
            pad: 0,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::RectBottom,
        },
        // pad_rect: interior block (full ring), top-left corner clamp, and a
        // saturating width/height growth.
        GpuAtlasPadQuery {
            x: 100,
            y: 100,
            w: 40,
            h: 50,
            coord: 0,
            pad: 8,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::PadRect,
        },
        GpuAtlasPadQuery {
            x: 1,
            y: 1,
            w: 3,
            h: 3,
            coord: 0,
            pad: 4,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::PadRect,
        },
        GpuAtlasPadQuery {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
            coord: 0,
            pad: 7,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::PadRect,
        },
        GpuAtlasPadQuery {
            x: 5,
            y: 5,
            w: u32::MAX - 2,
            h: u32::MAX - 2,
            coord: 0,
            pad: 10,
            mode: GpuPadMode::ClampEdge,
            op: GpuAtlasPadOp::PadRect,
        },
    ];
    check_batch("rect edges", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_single_query() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    let queries = vec![GpuAtlasPadQuery {
        x: 0,
        y: 0,
        w: 7,
        h: 0,
        coord: -3,
        pad: 0,
        mode: GpuPadMode::Mirror,
        op: GpuAtlasPadOp::GutterSourceIndex,
    }];
    check_batch("single query", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuAtlasMipPadding::new(&ctx);
    let out = engine.run(&ctx, &[]);
    assert!(
        out.is_empty(),
        "empty batch must short-circuit to an empty vec"
    );
}
