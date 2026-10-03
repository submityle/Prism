//! Real-device parity: the `GPU` opacity-micromap baker must match the
//! `prism_micromap` `CPU` golden element for element.
//!
//! Each test acquires a headless device best-effort and skips cleanly when the
//! host has no usable adapter, so the suite is green on a `CI` image without a
//! `GPU` while running the full dispatch on any machine with a real device.

use prism_micromap::omm::{
    bake_triangle, BakedOmm, OmmBakeInput, OmmFormat, SampleStrategy, SubdivisionLevel,
    TextureAlphaMask, WrapMode,
};
use prism_micromap_gpu::{GpuBakedOmm, GpuContext, GpuOmmBaker};

/// Runs `body` with a best-effort headless device, skipping if none exists.
fn with_gpu(body: impl FnOnce(&GpuContext, &GpuOmmBaker)) {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic so a GPU-less host records why parity was skipped"
        )]
        {
            eprintln!("skipping OMM GPU parity: no usable adapter on this host");
        }
        return;
    };
    let baker = GpuOmmBaker::new(&ctx);
    body(&ctx, &baker);
}

/// Bakes `uv` on the `CPU` golden with the uniform sampling strategy.
fn cpu_bake(
    uv: [[f32; 2]; 3],
    level: SubdivisionLevel,
    format: OmmFormat,
    samples: u32,
    mask: &TextureAlphaMask,
) -> BakedOmm {
    bake_triangle(
        &OmmBakeInput {
            uv,
            level,
            format,
            strategy: SampleStrategy::Uniform {
                samples_per_edge: samples,
            },
        },
        mask,
    )
}

/// Asserts the `GPU` bake equals the `CPU` golden state codes and packed bytes.
fn assert_exact(cpu: &BakedOmm, gpu: &GpuBakedOmm) {
    let cpu_states: Vec<u8> = cpu.states().iter().map(|s| s.as_u8()).collect();
    assert_eq!(gpu.states, cpu_states, "state codes diverged");
    assert_eq!(gpu.data, cpu.data(), "packed bytes diverged");
}

#[test]
fn all_opaque_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        let mask = TextureAlphaMask::with_wrap(4, 4, vec![1.0; 16], 0.5, WrapMode::Clamp);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for level in [0u8, 1, 2, 3] {
            let level = SubdivisionLevel::new(level).expect("level in range");
            for format in [OmmFormat::TwoState, OmmFormat::FourState] {
                let cpu = cpu_bake(uv, level, format, 4, &mask);
                let gpu = baker.bake(ctx, uv, level, format, 4, &mask);
                assert_exact(&cpu, &gpu);
            }
        }
    });
}

#[test]
fn all_transparent_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        let mask = TextureAlphaMask::with_wrap(4, 4, vec![0.0; 16], 0.5, WrapMode::Clamp);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for level in [0u8, 2, 3] {
            let level = SubdivisionLevel::new(level).expect("level in range");
            for format in [OmmFormat::TwoState, OmmFormat::FourState] {
                let cpu = cpu_bake(uv, level, format, 4, &mask);
                let gpu = baker.bake(ctx, uv, level, format, 4, &mask);
                assert_exact(&cpu, &gpu);
            }
        }
    });
}

#[test]
fn half_split_mask_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // Left half opaque, right half transparent on an 8-wide grid; the
        // boundary at u = 0.5 lands on a texel edge, exercising the mixed
        // tri-state path and the unknown classifications.
        let mut alpha = vec![0.0f32; 8 * 8];
        for row in 0..8 {
            for col in 0..4 {
                alpha[row * 8 + col] = 1.0;
            }
        }
        let mask = TextureAlphaMask::with_wrap(8, 8, alpha, 0.5, WrapMode::Clamp);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for level in [1u8, 2, 3] {
            let level = SubdivisionLevel::new(level).expect("level in range");
            for format in [OmmFormat::TwoState, OmmFormat::FourState] {
                let cpu = cpu_bake(uv, level, format, 4, &mask);
                let gpu = baker.bake(ctx, uv, level, format, 4, &mask);
                assert_exact(&cpu, &gpu);
            }
        }
    });
}

#[test]
fn blocky_mask_repeat_wrap_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // A 4x4 block pattern with repeat wrapping and a non-axis-aligned
        // triangle, so classification mixes several texels per micro-triangle.
        let alpha = vec![
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
        ];
        let mask = TextureAlphaMask::with_wrap(4, 4, alpha, 0.5, WrapMode::Repeat);
        let uv = [[0.1, 0.2], [0.9, 0.3], [0.4, 0.8]];
        for level in [1u8, 2, 3] {
            let level = SubdivisionLevel::new(level).expect("level in range");
            for format in [OmmFormat::TwoState, OmmFormat::FourState] {
                let cpu = cpu_bake(uv, level, format, 4, &mask);
                let gpu = baker.bake(ctx, uv, level, format, 4, &mask);
                assert_exact(&cpu, &gpu);
            }
        }
    });
}
