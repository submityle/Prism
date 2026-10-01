//! Real-device parity for the `RGB` ↔ `YCoCg` / `YCoCg-R` transform twin:
//! [`GpuRgbYCoCg`](prism_volumetric_gpu::rgb_ycocg::GpuRgbYCoCg) must reproduce
//! the `CPU` golden
//! [`rgb_ycocg`](prism_render_architecture::particle::rgb_ycocg) across black,
//! white, the `RGB` primaries, greys and a batch of random fixtures, for all
//! four transforms.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! The lossy `f32` transform touches only `+ - * /` with power-of-two
//! coefficients in the same order the reference uses, so `CPU` and `GPU`
//! evaluate the same closed form; they are not guaranteed bit-exact because a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The
//! lossless `YCoCg-R` path is pure integer arithmetic (subtract, add,
//! arithmetic right shift), so it is bit-exact and is compared with `==`.
//!
//! Provenance: standard `YCoCg` / `YCoCg-R` reversible colour transform
//! (`H.264` lossless mode); no third-party engine source or derived code.

use prism_render_architecture::particle::rgb_ycocg::{
    rgb_to_ycocg, rgb_to_ycocg_r, ycocg_r_to_rgb, ycocg_to_rgb, Rgb, YCoCg,
};
use prism_volumetric_gpu::rgb_ycocg::GpuRgbYCoCg;
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the lossy `f32` transform. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws the next pseudo-random byte from `state`.
fn lcg_u8(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) & 0xff) as u8
}

/// The canonical lossy `f32` fixtures: black, white, the three `RGB` primaries,
/// a spread of greys and assorted off-axis colours.
fn f32_fixtures() -> Vec<Rgb> {
    let mut v = vec![
        Rgb::new(0.0, 0.0, 0.0),
        Rgb::new(1.0, 1.0, 1.0),
        Rgb::new(1.0, 0.0, 0.0),
        Rgb::new(0.0, 1.0, 0.0),
        Rgb::new(0.0, 0.0, 1.0),
        Rgb::new(0.2, 0.4, 0.6),
        Rgb::new(0.9, 0.1, 0.3),
        Rgb::new(0.05, 0.95, 0.5),
        Rgb::new(0.7, 0.7, 0.2),
    ];
    for &g in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        v.push(Rgb::new(g, g, g));
    }
    v
}

/// The canonical lossless integer fixtures: black, white, the three primaries,
/// greys and edge channels.
fn u8_fixtures() -> Vec<[u8; 3]> {
    vec![
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [120, 120, 120],
        [1, 254, 3],
        [254, 1, 252],
        [128, 127, 129],
        [200, 50, 75],
    ]
}

#[test]
fn rgb_to_ycocg_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let mut inputs = f32_fixtures();
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    for _ in 0..256 {
        inputs.push(Rgb::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)));
    }

    let got = gpu.rgb_to_ycocg(&ctx, &inputs);
    assert_eq!(got.len(), inputs.len());
    for (idx, (g, c)) in got.iter().zip(inputs.iter()).enumerate() {
        let want = rgb_to_ycocg(*c);
        assert!(
            close(g.y, want.y),
            "sample {idx} Y: gpu {} vs cpu {}",
            g.y,
            want.y
        );
        assert!(
            close(g.co, want.co),
            "sample {idx} Co: gpu {} vs cpu {}",
            g.co,
            want.co
        );
        assert!(
            close(g.cg, want.cg),
            "sample {idx} Cg: gpu {} vs cpu {}",
            g.cg,
            want.cg
        );
    }
}

#[test]
fn ycocg_to_rgb_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    // Build YCoCg inputs by transforming the RGB fixtures plus random colours on
    // the CPU, then invert them on the GPU and compare against the CPU inverse.
    let mut rgbs = f32_fixtures();
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    for _ in 0..256 {
        rgbs.push(Rgb::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)));
    }
    let inputs: Vec<YCoCg> = rgbs.iter().map(|&c| rgb_to_ycocg(c)).collect();

    let got = gpu.ycocg_to_rgb(&ctx, &inputs);
    assert_eq!(got.len(), inputs.len());
    for (idx, (g, c)) in got.iter().zip(inputs.iter()).enumerate() {
        let want = ycocg_to_rgb(*c);
        assert!(
            close(g.r, want.r),
            "sample {idx} R: gpu {} vs cpu {}",
            g.r,
            want.r
        );
        assert!(
            close(g.g, want.g),
            "sample {idx} G: gpu {} vs cpu {}",
            g.g,
            want.g
        );
        assert!(
            close(g.b, want.b),
            "sample {idx} B: gpu {} vs cpu {}",
            g.b,
            want.b
        );
    }
}

#[test]
fn lossy_round_trip_recovers_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let mut inputs = f32_fixtures();
    let mut state = 0x3243_f6a8_885a_308d_u64;
    for _ in 0..128 {
        inputs.push(Rgb::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)));
    }

    let ycocg = gpu.rgb_to_ycocg(&ctx, &inputs);
    let back = gpu.ycocg_to_rgb(&ctx, &ycocg);
    assert_eq!(back.len(), inputs.len());
    for (idx, (b, c)) in back.iter().zip(inputs.iter()).enumerate() {
        assert!(
            close(b.r, c.r),
            "sample {idx} R round trip: {} vs {}",
            b.r,
            c.r
        );
        assert!(
            close(b.g, c.g),
            "sample {idx} G round trip: {} vs {}",
            b.g,
            c.g
        );
        assert!(
            close(b.b, c.b),
            "sample {idx} B round trip: {} vs {}",
            b.b,
            c.b
        );
    }
}

#[test]
fn grey_has_zero_chroma() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let greys: Vec<Rgb> = [0.0f32, 0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|&v| Rgb::new(v, v, v))
        .collect();
    let got = gpu.rgb_to_ycocg(&ctx, &greys);
    for (g, src) in got.iter().zip(greys.iter()) {
        assert!(close(g.co, 0.0), "grey Co must vanish, got {}", g.co);
        assert!(close(g.cg, 0.0), "grey Cg must vanish, got {}", g.cg);
        assert!(
            close(g.y, src.r),
            "grey luma must equal the value, got {}",
            g.y
        );
    }
}

#[test]
fn rgb_to_ycocg_r_matches_reference_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let mut inputs = u8_fixtures();
    // A coarse full-range sweep so every channel visits its extremes.
    for r in (0u16..=255).step_by(17) {
        for g in (0u16..=255).step_by(29) {
            for b in (0u16..=255).step_by(43) {
                inputs.push([r as u8, g as u8, b as u8]);
            }
        }
    }
    let mut state = 0xa409_3822_299f_31d0_u64;
    for _ in 0..256 {
        inputs.push([lcg_u8(&mut state), lcg_u8(&mut state), lcg_u8(&mut state)]);
    }

    let got = gpu.rgb_to_ycocg_r(&ctx, &inputs);
    assert_eq!(got.len(), inputs.len());
    for (idx, (g, c)) in got.iter().zip(inputs.iter()).enumerate() {
        assert_eq!(
            *g,
            rgb_to_ycocg_r(*c),
            "sample {idx} {c:?} forward mismatch"
        );
    }
}

#[test]
fn ycocg_r_to_rgb_matches_reference_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let mut rgbs = u8_fixtures();
    for r in (0u16..=255).step_by(23) {
        for g in (0u16..=255).step_by(31) {
            for b in (0u16..=255).step_by(47) {
                rgbs.push([r as u8, g as u8, b as u8]);
            }
        }
    }
    // The reference forward transform generates the YCoCg-R fixtures the GPU
    // inverse consumes, and the CPU inverse is the oracle.
    let inputs: Vec<_> = rgbs.iter().map(|&c| rgb_to_ycocg_r(c)).collect();

    let got = gpu.ycocg_r_to_rgb(&ctx, &inputs);
    assert_eq!(got.len(), inputs.len());
    for (idx, (g, c)) in got.iter().zip(inputs.iter()).enumerate() {
        assert_eq!(*g, ycocg_r_to_rgb(*c), "sample {idx} inverse mismatch");
    }
}

#[test]
fn ycocg_r_round_trip_is_lossless() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let mut inputs = u8_fixtures();
    for r in (0u16..=255).step_by(19) {
        for g in (0u16..=255).step_by(37) {
            for b in (0u16..=255).step_by(53) {
                inputs.push([r as u8, g as u8, b as u8]);
            }
        }
    }

    // Forward on the GPU, inverse on the GPU: the lifting is bit-exact, so the
    // chain must recover every source triple exactly.
    let ycocg_r = gpu.rgb_to_ycocg_r(&ctx, &inputs);
    let back = gpu.ycocg_r_to_rgb(&ctx, &ycocg_r);
    assert_eq!(back.len(), inputs.len());
    for (idx, (b, c)) in back.iter().zip(inputs.iter()).enumerate() {
        assert_eq!(b, c, "sample {idx} {c:?} is not recovered losslessly");
    }
}

#[test]
fn grey_has_zero_integer_chroma() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);

    let greys = vec![[0u8, 0, 0], [64, 64, 64], [120, 120, 120], [255, 255, 255]];
    let got = gpu.rgb_to_ycocg_r(&ctx, &greys);
    for (g, src) in got.iter().zip(greys.iter()) {
        assert_eq!(g.co, 0, "grey Co must be zero");
        assert_eq!(g.cg, 0, "grey Cg must be zero");
        assert_eq!(g.y, i32::from(src[0]), "grey luma must equal the value");
    }
}

#[test]
fn empty_inputs_return_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbYCoCg::new(&ctx);
    assert!(gpu.rgb_to_ycocg(&ctx, &[]).is_empty());
    assert!(gpu.ycocg_to_rgb(&ctx, &[]).is_empty());
    assert!(gpu.rgb_to_ycocg_r(&ctx, &[]).is_empty());
    assert!(gpu.ycocg_r_to_rgb(&ctx, &[]).is_empty());
}
