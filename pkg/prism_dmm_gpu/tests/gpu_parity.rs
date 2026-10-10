//! Real-device parity: the `GPU` displaced-micro-map baker must match the
//! `prism_dmm` `CPU` golden element for element.
//!
//! Each test acquires a headless device best-effort and skips cleanly when the
//! host has no usable adapter, so the suite is green on a `CI` image without a
//! `GPU` while running the full dispatch on any machine with a real device.

use prism_dmm::{
    bake_triangle, BakedDmm, DisplacementScaleBias, DmmBakeInput, DmmSubdivisionLevel,
    ScaleBiasMode, TextureDisplacementMap, WrapMode,
};
use prism_dmm_gpu::{GpuBakedDmm, GpuContext, GpuDmmBaker};

/// Runs `body` with a best-effort headless device, skipping if none exists.
fn with_gpu(body: impl FnOnce(&GpuContext, &GpuDmmBaker)) {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic so a GPU-less host records why parity was skipped"
        )]
        {
            eprintln!("skipping DMM GPU parity: no usable adapter on this host");
        }
        return;
    };
    let baker = GpuDmmBaker::new(&ctx);
    body(&ctx, &baker);
}

/// Bakes `uv` on the `CPU` golden.
fn cpu_bake(
    uv: [[f32; 2]; 3],
    level: DmmSubdivisionLevel,
    scale_bias_mode: ScaleBiasMode,
    map: &TextureDisplacementMap,
) -> BakedDmm {
    bake_triangle(
        &DmmBakeInput {
            uv,
            level,
            scale_bias_mode,
        },
        map,
    )
}

/// Asserts the `GPU` bake equals the `CPU` golden codes and packed bytes.
fn assert_exact(cpu: &BakedDmm, gpu: &GpuBakedDmm) {
    assert_eq!(gpu.codes.as_slice(), cpu.codes(), "codes diverged");
    assert_eq!(gpu.data.as_slice(), cpu.data(), "packed bytes diverged");
}

#[test]
fn flat_map_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // A constant height field is a degenerate [min, max] range, so every
        // micro-vertex must bake to code 0 on both backends.
        let map = TextureDisplacementMap::new(2, 2, vec![0.5; 4], WrapMode::Clamp);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for level in [0u8, 1, 2, 3] {
            let level = DmmSubdivisionLevel::new(level).expect("level in range");
            let cpu = cpu_bake(uv, level, ScaleBiasMode::PerTriangle, &map);
            let gpu = baker.bake(ctx, uv, level, ScaleBiasMode::PerTriangle, &map);
            assert!(
                gpu.codes.iter().all(|&c| c == 0),
                "flat map must bake to code 0"
            );
            assert_exact(&cpu, &gpu);
        }
    });
}

#[test]
fn ramp_map_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // A ramp along the U axis exercises bilinear interpolation and a
        // non-degenerate per-triangle range.
        let map = TextureDisplacementMap::new(2, 2, vec![0.0, 1.0, 0.0, 1.0], WrapMode::Clamp);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for level in [0u8, 1, 2, 3, 4] {
            let level = DmmSubdivisionLevel::new(level).expect("level in range");
            let cpu = cpu_bake(uv, level, ScaleBiasMode::PerTriangle, &map);
            let gpu = baker.bake(ctx, uv, level, ScaleBiasMode::PerTriangle, &map);
            assert_exact(&cpu, &gpu);
        }
    });
}

#[test]
fn non_axis_triangle_repeat_wrap_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // A varied 4x4 field with repeat wrapping and a non-axis-aligned
        // triangle, so sampling mixes several texels per micro-vertex.
        let map = TextureDisplacementMap::new(
            4,
            4,
            vec![
                0.0, 0.3, 0.7, 1.0, //
                0.2, 0.9, 0.4, 0.1, //
                0.6, 0.5, 0.8, 0.35, //
                0.95, 0.15, 0.55, 0.25, //
            ],
            WrapMode::Repeat,
        );
        let uv = [[0.1, 0.2], [0.9, 0.3], [0.4, 0.8]];
        for level in [1u8, 2, 3, 4] {
            let level = DmmSubdivisionLevel::new(level).expect("level in range");
            let cpu = cpu_bake(uv, level, ScaleBiasMode::PerTriangle, &map);
            let gpu = baker.bake(ctx, uv, level, ScaleBiasMode::PerTriangle, &map);
            assert_exact(&cpu, &gpu);
        }
    });
}

#[test]
fn fixed_scale_bias_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // A caller-provided fixed range must drive both backends identically,
        // including clamping of out-of-range heights.
        let map = TextureDisplacementMap::new(
            4,
            4,
            vec![
                0.0, 0.3, 0.7, 1.0, //
                0.2, 0.9, 0.4, 0.1, //
                0.6, 0.5, 0.8, 0.35, //
                0.95, 0.15, 0.55, 0.25, //
            ],
            WrapMode::Clamp,
        );
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let mode = ScaleBiasMode::Fixed(DisplacementScaleBias::new(0.1, 0.8));
        for level in [0u8, 1, 2, 3, 4] {
            let level = DmmSubdivisionLevel::new(level).expect("level in range");
            let cpu = cpu_bake(uv, level, mode, &map);
            let gpu = baker.bake(ctx, uv, level, mode, &map);
            assert_exact(&cpu, &gpu);
        }
    });
}

#[test]
fn deepest_level_matches_cpu_golden() {
    with_gpu(|ctx, baker| {
        // Level 5 is the maximum: 561 micro-vertices, 772 packed bytes.
        let map = TextureDisplacementMap::new(
            4,
            4,
            vec![
                0.0, 0.3, 0.7, 1.0, //
                0.2, 0.9, 0.4, 0.1, //
                0.6, 0.5, 0.8, 0.35, //
                0.95, 0.15, 0.55, 0.25, //
            ],
            WrapMode::Repeat,
        );
        let uv = [[0.05, 0.1], [0.85, 0.2], [0.3, 0.95]];
        let level = DmmSubdivisionLevel::new(5).expect("level in range");
        let cpu = cpu_bake(uv, level, ScaleBiasMode::PerTriangle, &map);
        let gpu = baker.bake(ctx, uv, level, ScaleBiasMode::PerTriangle, &map);
        assert_exact(&cpu, &gpu);
    });
}
