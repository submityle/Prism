//! Real-device parity for the `HSL` / `HSV` color-conversion twin:
//! [`GpuHslHsv`](prism_volumetric_gpu::hsl_hsv::GpuHslHsv) must reproduce the
//! `CPU` golden `prism_math::color::hsl` conversions — `Hsla::from_srgb`,
//! `Hsla::to_srgb`, `Hsva::from_srgb`, `Hsva::to_srgb` — together with the
//! private `rgb_to_hue` / `hue_to_rgb` helpers, selected per query by a
//! `dir_id`.
//!
//! Both color models are defined over non-linear `sRGB` components (the color
//! picker convention): hue is in degrees `[0, 360)` and the remaining axes in
//! `[0, 1]`. The four stateless directions are `0` `rgb->hsl`, `1` `hsl->rgb`,
//! `2` `rgb->hsv`, `3` `hsv->rgb`; any other `dir_id` is invalid.
//!
//! The oracle below is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core`, `prism_math` or `glam`. The golden is pure `f32`, so
//! the oracle is pure `f32` too and mirrors the same operator order, the same
//! `rem_euclid` reconstruction and the same channel / sector branches.
//!
//! The fixtures cover each direction with hand-chosen colors — strict
//! per-channel maxima (one for each `rgb_to_hue` branch), a grayscale input
//! (zero chroma, hue `0`), and `hsl` / `hsv` inputs whose hue sits mid-sector,
//! away from the `60`-degree knees. A round-trip spot-check feeds `rgb -> hsl`
//! back through `hsl -> rgb`. A mixed batch (including an out-of-range
//! `dir_id`) validates the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A `512`-step sweep follows, all
//! master-valid, rejection-sampling off the `60`-degree sector knees for hue
//! inputs and off channel ties for `rgb` inputs so the branch choice is stable.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both sides evaluate the same pure-`f32` closed form, so `CPU` and `GPU` need
//! not be bit-exact under reassociation. Every continuous output is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the discrete
//! `valid` word is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::color::hsl`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hsl_hsv::{GpuHslHsv, HslHsvQuery, HslHsvResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The resolved oracle outputs, in the same encoding as the device result.
struct Oracle {
    out: [f32; 4],
    valid: bool,
}

/// Euclidean remainder, matching the `WGSL` reconstruction `a - b * floor(a/b)`
/// and the golden's `f32::rem_euclid`.
fn rem_euclid(a: f32, b: f32) -> f32 {
    a.rem_euclid(b)
}

/// Independent host oracle for `rgb_to_hue`: returns `(max, min, chroma,
/// hue-in-degrees)` in the golden operator order. Grayscale (`chroma == 0`)
/// yields hue `0`; the tie-break order is `max == r`, then `max == g`, else
/// blue. The fixtures and sweep keep inputs off channel ties so this matches
/// the `GPU`'s ordered compares.
fn rgb_to_hue(r: f32, g: f32, b: f32) -> (f32, f32, f32, f32) {
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let chroma = mx - mn;
    let hue = if chroma == 0.0 {
        0.0
    } else if mx == r {
        60.0 * rem_euclid((g - b) / chroma, 6.0)
    } else if mx == g {
        60.0 * ((b - r) / chroma + 2.0)
    } else {
        60.0 * ((r - g) / chroma + 4.0)
    };
    (mx, mn, chroma, hue)
}

/// Independent host oracle for `hue_to_rgb`: reconstructs `(r, g, b, alpha)`
/// from hue / chroma plus a per-channel offset `m`, in the golden operator
/// order with the `h as u32` sector selection.
fn hue_to_rgb(hue: f32, chroma: f32, m: f32, alpha: f32) -> [f32; 4] {
    let h = rem_euclid(hue, 360.0) / 60.0;
    let x = chroma * (1.0 - (rem_euclid(h, 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    [r1 + m, g1 + m, b1 + m, alpha]
}

/// Independent host oracle: reproduces the golden conversion for the query's
/// direction, pure `f32`, with the master validity gate `dir_id <= 3`.
fn oracle(q: &HslHsvQuery) -> Oracle {
    let (c0, c1, c2, c3) = (q.c0, q.c1, q.c2, q.c3);
    let out = match q.dir_id {
        0 => {
            // rgb -> hsl.
            let (mx, mn, chroma, hue) = rgb_to_hue(c0, c1, c2);
            let lightness = 0.5 * (mx + mn);
            let saturation = if lightness <= 0.0 || lightness >= 1.0 {
                0.0
            } else {
                chroma / (1.0 - (2.0 * lightness - 1.0).abs())
            };
            [hue, saturation, lightness, c3]
        }
        1 => {
            // hsl -> rgb.
            let chroma = (1.0 - (2.0 * c2 - 1.0).abs()) * c1;
            let m = c2 - 0.5 * chroma;
            hue_to_rgb(c0, chroma, m, c3)
        }
        2 => {
            // rgb -> hsv.
            let (mx, _mn, chroma, hue) = rgb_to_hue(c0, c1, c2);
            let value = mx;
            let saturation = if value <= 0.0 { 0.0 } else { chroma / value };
            [hue, saturation, value, c3]
        }
        3 => {
            // hsv -> rgb.
            let chroma = c2 * c1;
            let m = c2 - chroma;
            hue_to_rgb(c0, chroma, m, c3)
        }
        _ => [0.0, 0.0, 0.0, 0.0],
    };
    Oracle {
        out,
        valid: q.dir_id <= 3,
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the `valid`
/// word matches exactly, each output matches to tolerance when valid, and every
/// output is zero when the master flag is cleared.
fn assert_parity(gpu: &HslHsvResult, q: &HslHsvQuery, label: &str) {
    let o = oracle(q);
    let gpu_valid = gpu.valid == 1;
    assert_eq!(gpu_valid, o.valid, "{label}: master valid flag");

    if !o.valid {
        assert_eq!(gpu.out0, 0.0, "{label}: out0 zeroed");
        assert_eq!(gpu.out1, 0.0, "{label}: out1 zeroed");
        assert_eq!(gpu.out2, 0.0, "{label}: out2 zeroed");
        assert_eq!(gpu.out3, 0.0, "{label}: out3 zeroed");
        return;
    }

    let gpu_out = [gpu.out0, gpu.out1, gpu.out2, gpu.out3];
    for (k, (&g, &e)) in gpu_out.iter().zip(o.out.iter()).enumerate() {
        assert!(close(g, e), "{label}: out{k} gpu={g} oracle={e}");
    }
}

#[test]
fn rgb_to_hsl_each_channel_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    // One strict-max input per rgb_to_hue branch (red, green, blue), plus a
    // grayscale (zero chroma, hue 0) input.
    let queries = vec![
        HslHsvQuery::new(0, 0.80, 0.30, 0.10, 1.0),
        HslHsvQuery::new(0, 0.20, 0.70, 0.30, 0.5),
        HslHsvQuery::new(0, 0.10, 0.40, 0.90, 0.25),
        HslHsvQuery::new(0, 0.50, 0.50, 0.50, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "rgb->hsl[{i}] should be valid");
        assert_parity(res, q, &format!("rgb_to_hsl[{i}]"));
    }
    // Grayscale has zero chroma, so hue and saturation are both zero.
    assert!(close(out[3].out0, 0.0), "grayscale hue is 0");
    assert!(close(out[3].out1, 0.0), "grayscale saturation is 0");
    assert!(close(out[3].out2, 0.5), "grayscale lightness is 0.5");
}

#[test]
fn rgb_to_hsv_each_channel_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    let queries = vec![
        HslHsvQuery::new(2, 0.80, 0.30, 0.10, 1.0),
        HslHsvQuery::new(2, 0.20, 0.70, 0.30, 0.5),
        HslHsvQuery::new(2, 0.10, 0.40, 0.90, 0.25),
        HslHsvQuery::new(2, 0.50, 0.50, 0.50, 1.0),
        // Pure black: value 0, saturation gate hits the value <= 0 branch.
        HslHsvQuery::new(2, 0.0, 0.0, 0.0, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "rgb->hsv[{i}] should be valid");
        assert_parity(res, q, &format!("rgb_to_hsv[{i}]"));
    }
    assert!(close(out[4].out1, 0.0), "black saturation is 0");
    assert!(close(out[4].out2, 0.0), "black value is 0");
}

#[test]
fn hsl_to_rgb_mid_sector_hues() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    // Hues at sector midpoints (30, 90, 150, 210, 270, 330) keep `h as u32`
    // stable; saturation and lightness are well inside (0, 1).
    let queries = vec![
        HslHsvQuery::new(1, 30.0, 0.60, 0.50, 1.0),
        HslHsvQuery::new(1, 90.0, 0.75, 0.40, 0.5),
        HslHsvQuery::new(1, 150.0, 0.50, 0.55, 1.0),
        HslHsvQuery::new(1, 210.0, 0.90, 0.45, 0.25),
        HslHsvQuery::new(1, 270.0, 0.40, 0.60, 1.0),
        HslHsvQuery::new(1, 330.0, 0.80, 0.50, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "hsl->rgb[{i}] should be valid");
        assert_parity(res, q, &format!("hsl_to_rgb[{i}]"));
    }
}

#[test]
fn hsv_to_rgb_mid_sector_hues() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    let queries = vec![
        HslHsvQuery::new(3, 30.0, 0.60, 0.80, 1.0),
        HslHsvQuery::new(3, 90.0, 0.75, 0.70, 0.5),
        HslHsvQuery::new(3, 150.0, 0.50, 0.90, 1.0),
        HslHsvQuery::new(3, 210.0, 0.90, 0.65, 0.25),
        HslHsvQuery::new(3, 270.0, 0.40, 0.85, 1.0),
        HslHsvQuery::new(3, 330.0, 0.80, 0.75, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "hsv->rgb[{i}] should be valid");
        assert_parity(res, q, &format!("hsv_to_rgb[{i}]"));
    }
}

#[test]
fn round_trip_rgb_hsl_rgb() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    // Mid-sector-hue colors so the inverse `h as u32` floor is stable.
    let originals = [
        [0.80f32, 0.35, 0.15, 1.0],
        [0.25, 0.70, 0.30, 0.5],
        [0.15, 0.40, 0.85, 0.75],
    ];
    for (i, rgb) in originals.iter().enumerate() {
        let fwd = HslHsvQuery::new(0, rgb[0], rgb[1], rgb[2], rgb[3]);
        let hsl = gpu.evaluate(&ctx, std::slice::from_ref(&fwd));
        assert_eq!(hsl.len(), 1);
        assert_eq!(hsl[0].valid, 1);
        let back = HslHsvQuery::new(1, hsl[0].out0, hsl[0].out1, hsl[0].out2, hsl[0].out3);
        let rgb_again = gpu.evaluate(&ctx, std::slice::from_ref(&back));
        assert_eq!(rgb_again.len(), 1);
        assert_eq!(rgb_again[0].valid, 1);
        assert!(close(rgb_again[0].out0, rgb[0]), "round-trip[{i}] r");
        assert!(close(rgb_again[0].out1, rgb[1]), "round-trip[{i}] g");
        assert!(close(rgb_again[0].out2, rgb[2]), "round-trip[{i}] b");
        assert!(close(rgb_again[0].out3, rgb[3]), "round-trip[{i}] alpha");
    }
}

#[test]
fn out_of_range_direction_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    let queries = vec![
        HslHsvQuery::new(4, 0.5, 0.5, 0.5, 1.0),
        HslHsvQuery::new(7, 0.1, 0.2, 0.3, 0.4),
        HslHsvQuery::new(u32::MAX, 1.0, 1.0, 1.0, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "dir_id[{i}] out of range must be invalid");
        assert_parity(res, q, &format!("out_of_range[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    // One of each direction plus an out-of-range selector, to exercise the
    // 32-byte std430 stride across heterogeneous slots.
    let queries = vec![
        HslHsvQuery::new(0, 0.80, 0.30, 0.10, 1.0),
        HslHsvQuery::new(1, 90.0, 0.75, 0.40, 0.5),
        HslHsvQuery::new(2, 0.10, 0.40, 0.90, 0.25),
        HslHsvQuery::new(3, 270.0, 0.40, 0.85, 1.0),
        HslHsvQuery::new(7, 0.0, 0.0, 0.0, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    assert_eq!(out[0].valid, 1, "slot 0 rgb->hsl valid");
    assert_eq!(out[1].valid, 1, "slot 1 hsl->rgb valid");
    assert_eq!(out[2].valid, 1, "slot 2 rgb->hsv valid");
    assert_eq!(out[3].valid, 1, "slot 3 hsv->rgb valid");
    assert_eq!(out[4].valid, 0, "slot 4 out-of-range invalid");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHslHsv::new(&ctx);
    let mut lcg = Lcg::new(0x2B7E_1591);
    let mut queries = Vec::with_capacity(512);
    let mut dir_counts = [0u32; 4];
    while queries.len() < 512 {
        let dir_id = lcg.next_u32() % 4;
        let q = match dir_id {
            0 | 2 => {
                // rgb inputs: reject channel ties so the max branch is stable.
                let r = lcg.next_range(0.0, 1.0);
                let g = lcg.next_range(0.0, 1.0);
                let b = lcg.next_range(0.0, 1.0);
                let mx = r.max(g).max(b);
                let mn = r.min(g).min(b);
                // Second-largest gap: distance from the max to the next
                // channel. Reject if too small so the branch cannot flip.
                let second = [r, g, b]
                    .iter()
                    .copied()
                    .filter(|&v| v < mx - f32::EPSILON)
                    .fold(f32::MIN, f32::max);
                let second = if second == f32::MIN { mn } else { second };
                if (mx - second) < 0.03 || (mx - mn) < 0.03 {
                    continue;
                }
                let alpha = lcg.next_range(0.0, 1.0);
                HslHsvQuery::new(dir_id, r, g, b, alpha)
            }
            _ => {
                // hsl / hsv inputs: reject hues near 60-degree sector knees so
                // `h as u32` cannot flip between CPU and GPU.
                let hue = lcg.next_range(0.0, 360.0);
                let in_sector = hue.rem_euclid(60.0);
                if in_sector < 3.0 || in_sector > 57.0 {
                    continue;
                }
                let sat = lcg.next_range(0.05, 1.0);
                let third = lcg.next_range(0.1, 0.9);
                let alpha = lcg.next_range(0.0, 1.0);
                HslHsvQuery::new(dir_id, hue, sat, third, alpha)
            }
        };
        dir_counts[dir_id as usize] += 1;
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be master-valid");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
    for (dir, count) in dir_counts.iter().enumerate() {
        assert!(*count > 0, "sweep should cover direction {dir}");
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
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
