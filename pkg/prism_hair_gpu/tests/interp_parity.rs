//! Real-device parity for the guide-to-render interpolation twin:
//! [`GpuHairInterp`] must reproduce the `CPU` golden
//! [`interpolate_render_strand`](prism_render_architecture::hair::interpolation::interpolate_render_strand)
//! for a batch of render-strand bindings, covering the weighted blend, the
//! length jitter, the clump pull toward the representative guide, the
//! seed-stable curl helix, the per-point position jitter, the degenerate
//! guards (out-of-range index, empty guide, zero weight), and the multi-strand
//! batch walk.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The blend/clump/jitter are closed-form and the hash/`sin_turns` are
//! integer/polynomial, so the `CPU` and `GPU` evaluate the same expressions and
//! diverge only through legal fused-multiply-add contraction. Parity is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a swapped transform, a wrong salt, a broken
//! `splitmix64` emulation), loose enough to admit the contraction. Guides are
//! built from straight lines and kinks (never `sin`/`cos`), and the shaping
//! cases additionally assert a non-trivial displacement so a no-op kernel could
//! not pass.
//!
//! Provenance: standard guide-to-render interpolation plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::interp::GpuHairInterp;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::interpolation::{
    interpolate_render_strand, InterpolationParams, RenderStrandBinding, Vec3,
};

/// Asserts a single component matches within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Neutral parameters: no clump, no curl, no jitter. Individual tests turn on
/// one transform at a time so a failure localizes to that stage.
fn neutral_params() -> InterpolationParams {
    InterpolationParams {
        clump_count: 1,
        clump_strength: 0.0,
        curl_frequency: 0.0,
        curl_amplitude: 0.0,
        position_jitter: 0.0,
        length_jitter: 0.0,
    }
}

/// A straight polyline from `root` stepping by `step` for `n` points.
fn line(root: Vec3, step: Vec3, n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            Vec3::new(
                root.x + step.x * i as f32,
                root.y + step.y * i as f32,
                root.z + step.z * i as f32,
            )
        })
        .collect()
}

/// Runs the golden per binding and asserts the `GPU` result matches per
/// component, returning the `GPU` output for further non-trivial assertions.
fn assert_parity(
    ctx: &GpuContext,
    interp: &GpuHairInterp,
    guides: &[Vec<Vec3>],
    bindings: &[RenderStrandBinding],
    params: InterpolationParams,
) -> Vec<Vec<Vec3>> {
    let guide_refs: Vec<&[Vec3]> = guides.iter().map(Vec::as_slice).collect();
    let gpu = interp.eval(ctx, &guide_refs, bindings, params);
    assert_eq!(gpu.len(), bindings.len(), "one list per binding");
    for (b, binding) in bindings.iter().enumerate() {
        let mut cpu = Vec::new();
        interpolate_render_strand(&guide_refs, binding, params, &mut cpu);
        assert_eq!(
            gpu[b].len(),
            cpu.len(),
            "strand {b}: gpu {} points, cpu {} points",
            gpu[b].len(),
            cpu.len()
        );
        for (i, (g, c)) in gpu[b].iter().zip(cpu.iter()).enumerate() {
            assert_close(g.x, c.x, &format!("strand {b} point {i} x"));
            assert_close(g.y, c.y, &format!("strand {b} point {i} y"));
            assert_close(g.z, c.z, &format!("strand {b} point {i} z"));
        }
    }
    gpu
}

/// Single guide, no transforms: the render strand equals the guide exactly.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_guide_passthrough_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 8)];
    let bindings = [RenderStrandBinding {
        guides: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
        root_uv: (0.25, 0.75),
        seed: 1,
    }];

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, neutral_params());
    assert_eq!(gpu[0].len(), 8, "output length follows the guide");
    // No transform: the strand must equal the guide, so a kernel that dropped
    // the blend could not pass.
    assert_close(gpu[0][7].y, 0.7, "passthrough tip y");
}

/// Weighted blend of two offset guides lands between them.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_weighted_blend_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![
        line(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 10),
        line(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 10),
    ];
    let bindings = [RenderStrandBinding {
        guides: [0, 1, 0, 0],
        weights: [0.75, 0.25, 0.0, 0.0],
        root_uv: (0.5, 0.5),
        seed: 7,
    }];

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, neutral_params());
    // 0.75*(-1) + 0.25*(1) = -0.5 along x, so the blend must actually mix.
    assert_close(gpu[0][0].x, -0.5, "blend root x");
}

/// Clump pull drags the strand toward the representative guide at the tip.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_clump_pull_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![
        line(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 12),
        line(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 12),
    ];
    // Guide 0 is the representative (largest weight); clump pulls the tip
    // toward it, more strongly than the root.
    let bindings = [RenderStrandBinding {
        guides: [0, 1, 0, 0],
        weights: [0.6, 0.4, 0.0, 0.0],
        root_uv: (0.5, 0.5),
        seed: 3,
    }];
    let mut params = neutral_params();
    params.clump_strength = 0.8;

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, params);
    // The tip must sit closer to guide 0's x (-1.0) than the un-clumped blend
    // (0.6*-1 + 0.4*1 = -0.2), confirming the clump actually pulled.
    assert!(
        gpu[0][11].x < -0.2,
        "clumped tip should be pulled toward rep guide: {}",
        gpu[0][11].x
    );
}

/// Curl adds a helix off the clumped tangent; the strand leaves its axis.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_curl_helix_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 16)];
    let bindings = [RenderStrandBinding {
        guides: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
        root_uv: (0.1, 0.2),
        seed: 42,
    }];
    let mut params = neutral_params();
    params.curl_frequency = 3.0;
    params.curl_amplitude = 0.2;

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, params);
    // The straight guide runs along +y; curl must push some interior point off
    // the x==0 && z==0 axis, so a kernel that skipped curl could not pass.
    let off_axis = gpu[0].iter().any(|p| p.x.abs() > 1e-3 || p.z.abs() > 1e-3);
    assert!(off_axis, "curl should displace the strand off its axis");
}

/// Position jitter perturbs interior points deterministically.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_position_jitter_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 16)];
    let bindings = [RenderStrandBinding {
        guides: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
        root_uv: (0.3, 0.4),
        seed: 123,
    }];
    let mut params = neutral_params();
    params.position_jitter = 0.5;

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, params);
    let off_axis = gpu[0].iter().any(|p| p.x.abs() > 1e-4 || p.z.abs() > 1e-4);
    assert!(off_axis, "position jitter should perturb interior points");
}

/// Length jitter rescales the strand relative to the root.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_length_jitter_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 10)];
    // Two seeds so the jitter direction differs; parity is per binding.
    let bindings = [
        RenderStrandBinding {
            guides: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 11,
        },
        RenderStrandBinding {
            guides: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 99,
        },
    ];
    let mut params = neutral_params();
    params.length_jitter = 0.5;

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, params);
    // At least one strand's tip must differ from the un-jittered 0.9 length.
    let jittered = (gpu[0][9].y - 0.9).abs() > 1e-3 || (gpu[1][9].y - 0.9).abs() > 1e-3;
    assert!(jittered, "length jitter should rescale at least one strand");
}

/// Mismatched guide resolutions: output length follows the shortest guide.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_shortest_guide_length_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![
        line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 12),
        line(Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 5),
    ];
    let bindings = [RenderStrandBinding {
        guides: [0, 1, 0, 0],
        weights: [0.5, 0.5, 0.0, 0.0],
        root_uv: (0.5, 0.5),
        seed: 5,
    }];

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, neutral_params());
    assert_eq!(gpu[0].len(), 5, "output clamps to the shortest guide");
}

/// Degenerate bindings (out-of-range index, empty guide, all-zero weights)
/// yield empty lists, and a healthy strand in the same batch still expands.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_bindings_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![
        line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 8),
        Vec::new(), // empty guide
    ];
    let bindings = [
        // Out-of-range guide index.
        RenderStrandBinding {
            guides: [9, 9, 9, 9],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 1,
        },
        // Only an empty guide contributes.
        RenderStrandBinding {
            guides: [1, 1, 1, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 2,
        },
        // All-zero weights.
        RenderStrandBinding {
            guides: [0, 0, 0, 0],
            weights: [0.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 3,
        },
        // Healthy strand in the same batch.
        RenderStrandBinding {
            guides: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed: 4,
        },
    ];

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, neutral_params());
    assert!(gpu[0].is_empty(), "out-of-range binding yields empty list");
    assert!(gpu[1].is_empty(), "empty-guide binding yields empty list");
    assert!(gpu[2].is_empty(), "zero-weight binding yields empty list");
    assert_eq!(gpu[3].len(), 8, "healthy binding still expands");
}

/// A fully empty batch returns no lists without dispatching.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    let guides = vec![line(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), 8)];
    let bindings: [RenderStrandBinding; 0] = [];

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, neutral_params());
    assert!(gpu.is_empty(), "empty batch returns no lists");
}

/// A larger multi-strand batch with all transforms on, kinked guides, exercises
/// the full chain and the per-strand seed independence at once.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_full_chain_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping interp parity: no wgpu adapter on this host");
        return;
    };
    let interp = GpuHairInterp::new(&ctx);

    // Kinked guides (two straight segments) exercise the tangent framing.
    let mut g0 = line(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.02, 0.1, 0.0), 10);
    g0.extend(line(
        Vec3::new(-0.3, 1.0, 0.0),
        Vec3::new(0.1, 0.1, 0.05),
        10,
    ));
    let mut g1 = line(Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.02, 0.1, 0.0), 10);
    g1.extend(line(
        Vec3::new(0.3, 1.0, 0.0),
        Vec3::new(-0.1, 0.1, 0.05),
        10,
    ));
    let g2 = line(Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.0, 0.12, -0.01), 20);
    let guides = vec![g0, g1, g2];

    let mut bindings = Vec::new();
    for s in 0..24u32 {
        let w0 = 0.2 + (s % 5) as f32 * 0.1;
        let w1 = 0.5 - (s % 3) as f32 * 0.1;
        let w2 = 0.3;
        bindings.push(RenderStrandBinding {
            guides: [0, 1, 2, 7],
            weights: [w0, w1, w2, 0.0],
            root_uv: (s as f32 / 24.0, 0.5),
            seed: 1000 + s * 17,
        });
    }

    let params = InterpolationParams {
        clump_count: 4,
        clump_strength: 0.6,
        curl_frequency: 2.5,
        curl_amplitude: 0.15,
        position_jitter: 0.3,
        length_jitter: 0.4,
    };

    let gpu = assert_parity(&ctx, &interp, &guides, &bindings, params);
    assert_eq!(gpu.len(), 24, "one list per strand");
    assert!(
        gpu.iter().all(|s| s.len() == 20),
        "every strand clamps to the shortest guide (20 points)"
    );
}
