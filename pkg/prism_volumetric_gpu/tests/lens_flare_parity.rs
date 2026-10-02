//! Real-device parity for the screen-space lens-flare twin:
//! [`GpuLensFlare`](prism_volumetric_gpu::lens_flare::GpuLensFlare) must
//! reproduce the `CPU` golden
//! [`lens_flare`](prism_render_architecture::particle::lens_flare) numeric
//! surface across an empty batch, every per-op fixture, a mixed tagged batch and
//! a large pseudo-random batch compared lane for lane.
//!
//! The twin reproduces the module's pure numeric pieces one thread per query:
//! the private `clamp01` / `smoothstep01` / `distance` primitives, the public
//! [`luminance`](prism_render_architecture::particle::lens_flare::luminance), and
//! the [`LensFlareParams`](prism_render_architecture::particle::lens_flare::LensFlareParams)
//! methods `threshold_weight`, `axis_dir`, `ghost_uv`, `ghost_chroma_uvs`,
//! `halo_weight`, `radial_attenuation`, `screen_fade` and `sample_ghost`. The
//! variable-length ghost-chain walk
//! [`LensFlareParams::sample_ghosts`](prism_render_architecture::particle::lens_flare::LensFlareParams::sample_ghosts)
//! stays host-side and is exercised by issuing one `SampleGhost` query per index,
//! so the suite twins only the per-ghost closed form.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, guarded
//! divides and at most two `sqrt` calls, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the `f32` lanes. The
//! fixtures stay clear of the `axis_dir` degeneracy guard (except the explicit
//! degenerate cases) by keeping the bright point well off the center, so the
//! comparison exercises the live solve.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`；
//! 无需外部数学库，无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::lens_flare::{luminance, LensFlareParams};
use prism_volumetric_gpu::lens_flare::{GpuLensFlare, LensFlareQuery, LensFlareResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the `f32` lanes. A `GPU` may fuse a multiply-add the
/// scalar reference leaves separate, perturbing the low mantissa bits by a few
/// units in the last place; `1e-4` admits that legal slack while still failing a
/// genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Returns whether two `UV` pairs agree component-wise within tolerance.
fn approx_vec2(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Returns whether two chromatic `UV` triples agree channel-wise within
/// tolerance.
fn approx_chroma(a: [[f32; 2]; 3], b: [[f32; 2]; 3]) -> bool {
    approx_vec2(a[0], b[0]) && approx_vec2(a[1], b[1]) && approx_vec2(a[2], b[2])
}

// ---------------------------------------------------------------------------
// Local reimplementation of the reference's private free functions.
// ---------------------------------------------------------------------------

/// Clamps a scalar into the closed unit interval, mirroring the reference
/// private `clamp01`.
fn clamp01_ref(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The multiply-only `smoothstep` `t^2 (3 - 2 t)` after clamping, mirroring the
/// reference private `smoothstep01`.
fn smoothstep01_ref(t: f32) -> f32 {
    let c = clamp01_ref(t);
    c * c * (3.0 - 2.0 * c)
}

/// Euclidean `UV` distance, mirroring the reference private `distance`. `sqrt`
/// is a permitted non-transcendental primitive.
fn distance_ref(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Builds a [`LensFlareParams`] from the fields a query carries, with
/// `ghost_count` fixed at `1` (the twin is one-shot per ghost). Non-negative
/// shaping coefficients pass through the reference clamp unchanged, so the twin
/// and the reference see identical inputs.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the flat optical knobs the golden LensFlareParams::new takes"
)]
fn mk(
    center: [f32; 2],
    threshold: f32,
    knee: f32,
    ghost_spacing: f32,
    chroma_offset: f32,
    halo_radius: f32,
    halo_width: f32,
    halo_intensity: f32,
    radial_falloff: f32,
    edge_fade: f32,
) -> LensFlareParams {
    LensFlareParams::new(
        center,
        threshold,
        knee,
        ghost_spacing,
        1,
        chroma_offset,
        halo_radius,
        halo_width,
        halo_intensity,
        radial_falloff,
        edge_fade,
    )
}

// ---------------------------------------------------------------------------
// Golden evaluation and comparison.
// ---------------------------------------------------------------------------

/// Evaluates the `CPU` golden for one query, returning the expected
/// [`LensFlareResult`].
fn expected(query: &LensFlareQuery) -> LensFlareResult {
    match query {
        LensFlareQuery::Clamp01 { x } => LensFlareResult::Scalar(clamp01_ref(*x)),
        LensFlareQuery::Smoothstep01 { t } => LensFlareResult::Scalar(smoothstep01_ref(*t)),
        LensFlareQuery::Distance { a, b } => LensFlareResult::Scalar(distance_ref(*a, *b)),
        LensFlareQuery::Luminance { rgb } => LensFlareResult::Scalar(luminance(*rgb)),
        LensFlareQuery::ThresholdWeight {
            threshold,
            knee,
            lum,
        } => {
            let p = mk(
                [0.5, 0.5],
                *threshold,
                *knee,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            LensFlareResult::Scalar(p.threshold_weight(*lum))
        }
        LensFlareQuery::AxisDir { center, bright_uv } => {
            let p = mk(*center, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            LensFlareResult::Vec2(p.axis_dir(*bright_uv))
        }
        LensFlareQuery::GhostUv {
            center,
            ghost_spacing,
            bright_uv,
            index,
        } => {
            let p = mk(
                *center,
                0.0,
                0.0,
                *ghost_spacing,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            LensFlareResult::Vec2(p.ghost_uv(*bright_uv, *index))
        }
        LensFlareQuery::GhostChromaUvs {
            center,
            ghost_spacing,
            chroma_offset,
            bright_uv,
            index,
        } => {
            let p = mk(
                *center,
                0.0,
                0.0,
                *ghost_spacing,
                *chroma_offset,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            LensFlareResult::ChromaUvs(p.ghost_chroma_uvs(*bright_uv, *index))
        }
        LensFlareQuery::HaloWeight {
            center,
            halo_radius,
            halo_width,
            halo_intensity,
            uv,
        } => {
            let p = mk(
                *center,
                0.0,
                0.0,
                0.0,
                0.0,
                *halo_radius,
                *halo_width,
                *halo_intensity,
                0.0,
                0.0,
            );
            LensFlareResult::Scalar(p.halo_weight(*uv))
        }
        LensFlareQuery::RadialAttenuation { radial_falloff, r } => {
            let p = mk(
                [0.5, 0.5],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                *radial_falloff,
                0.0,
            );
            LensFlareResult::Scalar(p.radial_attenuation(*r))
        }
        LensFlareQuery::ScreenFade { edge_fade, uv } => {
            let p = mk(
                [0.5, 0.5],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                *edge_fade,
            );
            LensFlareResult::Scalar(p.screen_fade(*uv))
        }
        LensFlareQuery::SampleGhost {
            center,
            ghost_spacing,
            chroma_offset,
            radial_falloff,
            edge_fade,
            bright_uv,
            bright_weight,
            index,
        } => {
            let p = mk(
                *center,
                0.0,
                0.0,
                *ghost_spacing,
                *chroma_offset,
                0.0,
                0.0,
                0.0,
                *radial_falloff,
                *edge_fade,
            );
            let ghost = p.sample_ghost(*bright_uv, *bright_weight, *index);
            LensFlareResult::Ghost {
                uv_rgb: ghost.uv_rgb,
                weight: ghost.weight,
            }
        }
    }
}

/// Asserts the `GPU` result matches the `CPU` golden for one lane, applying the
/// tolerance on every `f32` lane.
fn compare(lane: usize, query: &LensFlareQuery, got: &LensFlareResult) {
    let want = expected(query);
    match (got, &want) {
        (LensFlareResult::Scalar(g), LensFlareResult::Scalar(w)) => {
            assert!(approx(*g, *w), "lane {lane}: scalar gpu {g} vs cpu {w}");
        }
        (LensFlareResult::Vec2(g), LensFlareResult::Vec2(w)) => {
            assert!(
                approx_vec2(*g, *w),
                "lane {lane}: vec2 gpu {g:?} vs cpu {w:?}"
            );
        }
        (LensFlareResult::ChromaUvs(g), LensFlareResult::ChromaUvs(w)) => {
            assert!(
                approx_chroma(*g, *w),
                "lane {lane}: chroma gpu {g:?} vs cpu {w:?}"
            );
        }
        (
            LensFlareResult::Ghost {
                uv_rgb: g_uv,
                weight: g_w,
            },
            LensFlareResult::Ghost {
                uv_rgb: w_uv,
                weight: w_w,
            },
        ) => {
            assert!(
                approx_chroma(*g_uv, *w_uv),
                "lane {lane}: ghost uv gpu {g_uv:?} vs cpu {w_uv:?}"
            );
            assert!(
                approx(*g_w, *w_w),
                "lane {lane}: ghost weight gpu {g_w} vs cpu {w_w}"
            );
        }
        (g, w) => panic!("lane {lane}: result kind mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches a single query and asserts it matches the golden.
fn check_one(ctx: &GpuContext, gpu: &GpuLensFlare, query: LensFlareQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one result per query");
    compare(0, &query, &got[0]);
}

// ---------------------------------------------------------------------------
// Per-op fixtures.
// ---------------------------------------------------------------------------

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn clamp01_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Interior, clearly below, clearly above: all away from the clamp kinks.
    check_one(&ctx, &gpu, LensFlareQuery::Clamp01 { x: 0.37 });
    check_one(&ctx, &gpu, LensFlareQuery::Clamp01 { x: -0.8 });
    check_one(&ctx, &gpu, LensFlareQuery::Clamp01 { x: 1.6 });
}

#[test]
fn smoothstep01_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(&ctx, &gpu, LensFlareQuery::Smoothstep01 { t: 0.3 });
    check_one(&ctx, &gpu, LensFlareQuery::Smoothstep01 { t: 0.72 });
    check_one(&ctx, &gpu, LensFlareQuery::Smoothstep01 { t: -0.4 });
    check_one(&ctx, &gpu, LensFlareQuery::Smoothstep01 { t: 1.5 });
}

#[test]
fn distance_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::Distance {
            a: [0.2, 0.1],
            b: [0.8, 0.6],
        },
    );
}

#[test]
fn luminance_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::Luminance {
            rgb: [0.25, 0.6, 0.15],
        },
    );
}

#[test]
fn threshold_weight_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Below band, mid band and above band, all away from the smoothstep kinks.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ThresholdWeight {
            threshold: 1.0,
            knee: 0.25,
            lum: 0.5,
        },
    );
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ThresholdWeight {
            threshold: 1.0,
            knee: 0.25,
            lum: 1.05,
        },
    );
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ThresholdWeight {
            threshold: 1.0,
            knee: 0.25,
            lum: 1.6,
        },
    );
}

#[test]
fn axis_dir_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Bright point well off center, so the live normalization runs.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::AxisDir {
            center: [0.5, 0.5],
            bright_uv: [0.85, 0.72],
        },
    );
}

#[test]
fn axis_dir_degenerate_center() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Bright point exactly on the center: both sides return the zero vector.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::AxisDir {
            center: [0.5, 0.5],
            bright_uv: [0.5, 0.5],
        },
    );
}

#[test]
fn ghost_uv_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::GhostUv {
            center: [0.5, 0.5],
            ghost_spacing: 0.35,
            bright_uv: [0.82, 0.6],
            index: 2,
        },
    );
}

#[test]
fn ghost_chroma_uvs_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::GhostChromaUvs {
            center: [0.5, 0.5],
            ghost_spacing: 0.35,
            chroma_offset: 0.02,
            bright_uv: [0.82, 0.6],
            index: 2,
        },
    );
}

#[test]
fn ghost_chroma_uvs_degenerate_center() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Bright point on center: ghosts collapse and the chromatic split vanishes.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::GhostChromaUvs {
            center: [0.5, 0.5],
            ghost_spacing: 0.35,
            chroma_offset: 0.02,
            bright_uv: [0.5, 0.5],
            index: 2,
        },
    );
}

#[test]
fn halo_weight_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Inside the band and well outside it, away from the smoothstep kinks.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::HaloWeight {
            center: [0.5, 0.5],
            halo_radius: 0.3,
            halo_width: 0.08,
            halo_intensity: 0.7,
            uv: [0.5, 0.78],
        },
    );
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::HaloWeight {
            center: [0.5, 0.5],
            halo_radius: 0.3,
            halo_width: 0.08,
            halo_intensity: 0.7,
            uv: [0.5, 0.95],
        },
    );
}

#[test]
fn radial_attenuation_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::RadialAttenuation {
            radial_falloff: 2.0,
            r: 0.0,
        },
    );
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::RadialAttenuation {
            radial_falloff: 2.0,
            r: 0.45,
        },
    );
}

#[test]
fn screen_fade_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // Deep interior saturates to one; a clearly off-screen UV fades to zero.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ScreenFade {
            edge_fade: 0.05,
            uv: [0.5, 0.5],
        },
    );
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ScreenFade {
            edge_fade: 0.05,
            uv: [1.4, 0.5],
        },
    );
    // A UV inside the fade margin exercises the live smoothstep band.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::ScreenFade {
            edge_fade: 0.2,
            uv: [0.08, 0.5],
        },
    );
}

#[test]
fn sample_ghost_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::SampleGhost {
            center: [0.5, 0.5],
            ghost_spacing: 0.35,
            chroma_offset: 0.01,
            radial_falloff: 2.0,
            edge_fade: 0.05,
            bright_uv: [0.7, 0.62],
            bright_weight: 0.8,
            index: 1,
        },
    );
}

#[test]
fn sample_ghost_offscreen_zero_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    // A bright point near an edge with a large spacing throws ghost 3 well
    // outside the frame; the folded weight vanishes on both sides.
    check_one(
        &ctx,
        &gpu,
        LensFlareQuery::SampleGhost {
            center: [0.5, 0.5],
            ghost_spacing: 0.9,
            chroma_offset: 0.01,
            radial_falloff: 2.0,
            edge_fade: 0.05,
            bright_uv: [0.98, 0.5],
            bright_weight: 1.0,
            index: 3,
        },
    );
}

// ---------------------------------------------------------------------------
// Mixed batch.
// ---------------------------------------------------------------------------

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    let queries = vec![
        LensFlareQuery::Clamp01 { x: 0.42 },
        LensFlareQuery::Smoothstep01 { t: 0.6 },
        LensFlareQuery::Distance {
            a: [0.1, 0.2],
            b: [0.7, 0.9],
        },
        LensFlareQuery::Luminance {
            rgb: [0.5, 0.4, 0.1],
        },
        LensFlareQuery::ThresholdWeight {
            threshold: 1.2,
            knee: 0.3,
            lum: 1.1,
        },
        LensFlareQuery::AxisDir {
            center: [0.5, 0.5],
            bright_uv: [0.9, 0.65],
        },
        LensFlareQuery::GhostUv {
            center: [0.5, 0.5],
            ghost_spacing: 0.4,
            bright_uv: [0.78, 0.58],
            index: 3,
        },
        LensFlareQuery::GhostChromaUvs {
            center: [0.5, 0.5],
            ghost_spacing: 0.4,
            chroma_offset: 0.015,
            bright_uv: [0.78, 0.58],
            index: 3,
        },
        LensFlareQuery::HaloWeight {
            center: [0.5, 0.5],
            halo_radius: 0.35,
            halo_width: 0.09,
            halo_intensity: 0.6,
            uv: [0.5, 0.82],
        },
        LensFlareQuery::RadialAttenuation {
            radial_falloff: 1.5,
            r: 0.3,
        },
        LensFlareQuery::ScreenFade {
            edge_fade: 0.04,
            uv: [0.5, 0.5],
        },
        LensFlareQuery::SampleGhost {
            center: [0.5, 0.5],
            ghost_spacing: 0.3,
            chroma_offset: 0.02,
            radial_falloff: 1.5,
            edge_fade: 0.05,
            bright_uv: [0.72, 0.6],
            bright_weight: 0.9,
            index: 2,
        },
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random batch.
// ---------------------------------------------------------------------------

/// A tiny host-side `u64` linear congruential generator, so the fixtures use no
/// `f32` transcendental method. The multiplier and increment are the well-known
/// `PCG` / `Knuth` constants.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A float in `[0, 1)` with 24 bits of entropy.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A float in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// A `UV` whose offset from `[0.5, 0.5]` has squared length above `0.01`, so
    /// the optical axis is well clear of the degeneracy guard.
    fn bright_uv(&mut self) -> [f32; 2] {
        loop {
            let bx = 0.5 + self.signed() * 0.45;
            let by = 0.5 + self.signed() * 0.45;
            let dx = bx - 0.5;
            let dy = by - 0.5;
            if dx * dx + dy * dy > 0.01 {
                return [bx, by];
            }
        }
    }
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLensFlare::new(&ctx);
    let mut rng = Lcg::new(0x1ec5_f1a2_u64);
    let mut queries: Vec<LensFlareQuery> = Vec::new();
    for _ in 0..180 {
        let op = rng.next_u32() % 12;
        let query = match op {
            0 => LensFlareQuery::Clamp01 {
                x: rng.signed() * 2.0,
            },
            1 => LensFlareQuery::Smoothstep01 {
                t: rng.signed() * 1.5,
            },
            2 => LensFlareQuery::Distance {
                a: [rng.unit(), rng.unit()],
                b: [rng.unit(), rng.unit()],
            },
            3 => LensFlareQuery::Luminance {
                rgb: [rng.unit(), rng.unit(), rng.unit()],
            },
            4 => LensFlareQuery::ThresholdWeight {
                threshold: 0.5 + rng.unit(),
                knee: 0.15 + rng.unit() * 0.4,
                lum: rng.unit() * 2.0,
            },
            5 => LensFlareQuery::AxisDir {
                center: [0.5, 0.5],
                bright_uv: rng.bright_uv(),
            },
            6 => LensFlareQuery::GhostUv {
                center: [0.5, 0.5],
                ghost_spacing: 0.2 + rng.unit() * 0.4,
                bright_uv: rng.bright_uv(),
                index: 1 + rng.next_u32() % 4,
            },
            7 => LensFlareQuery::GhostChromaUvs {
                center: [0.5, 0.5],
                ghost_spacing: 0.2 + rng.unit() * 0.4,
                chroma_offset: rng.unit() * 0.03,
                bright_uv: rng.bright_uv(),
                index: 1 + rng.next_u32() % 4,
            },
            8 => LensFlareQuery::HaloWeight {
                center: [0.5, 0.5],
                halo_radius: 0.2 + rng.unit() * 0.2,
                halo_width: 0.05 + rng.unit() * 0.1,
                halo_intensity: 0.3 + rng.unit() * 0.6,
                uv: [rng.unit(), rng.unit()],
            },
            9 => LensFlareQuery::RadialAttenuation {
                radial_falloff: rng.unit() * 3.0,
                r: rng.unit(),
            },
            10 => LensFlareQuery::ScreenFade {
                edge_fade: 0.03 + rng.unit() * 0.2,
                uv: [rng.unit(), rng.unit()],
            },
            _ => LensFlareQuery::SampleGhost {
                center: [0.5, 0.5],
                ghost_spacing: 0.2 + rng.unit() * 0.4,
                chroma_offset: rng.unit() * 0.03,
                radial_falloff: rng.unit() * 3.0,
                edge_fade: 0.03 + rng.unit() * 0.2,
                bright_uv: rng.bright_uv(),
                bright_weight: rng.unit(),
                index: 1 + rng.next_u32() % 4,
            },
        };
        queries.push(query);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}
