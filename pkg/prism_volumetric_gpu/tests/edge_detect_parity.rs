//! Real-device parity for the screen-space edge-detection twin:
//! [`GpuEdgeDetect`](prism_volumetric_gpu::edge_detect::GpuEdgeDetect) must
//! reproduce the `CPU` golden
//! [`edge_detect`](prism_render_architecture::particle::edge_detect) responses
//! across a flat (edge-free) frame, a vertical step edge, a diagonal gradient,
//! an isolated depth discontinuity, a rotating normal field and a batch of
//! random frames at several resolutions, compared pixel-for-pixel and
//! channel-for-channel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each response is a fixed, non-reorderable sequence of multiplies, adds and
//! one `sqrt` (plus the cubic `smoothstep` for the mask), so `CPU` and `GPU`
//! evaluate the same closed form. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — tight enough to fail a
//! genuinely wrong port (a swapped kernel sign, a dropped tap, a missing edge
//! clamp, a wrong `luminance` weight) yet loose enough to admit a legal fused
//! multiply-add contraction. The `smoothstep` mask is driven only with a
//! non-degenerate `knee` (or well-separated magnitudes for the hard-step case)
//! so the comparison never straddles the discontinuous hard-step boundary.
//!
//! Provenance: standard `Sobel`/`Roberts` edge detection; no third-party engine
//! source or derived code.

use prism_render_architecture::particle::edge_detect::{
    depth_edge, luminance, normal_edge, roberts_magnitude, sobel_magnitude, EdgeParams,
};
use prism_volumetric_gpu::edge_detect::{EdgeDetectQuery, EdgeFrame, GpuEdgeDetect};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound; admits a few units-in-the-last-place of fused
/// multiply-add slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound for larger magnitudes where a few `ULP` exceed the
/// absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Whether `a` and `b` agree within the absolute or relative bound.
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

/// Clamp-to-edge of a signed coordinate into `[0, extent)`, matching the
/// kernel's `clamp_coord` and the reference host gather.
fn clamp_coord(coord: i32, extent: usize) -> usize {
    if extent == 0 {
        return 0;
    }
    if coord < 0 {
        return 0;
    }
    let c = coord as usize;
    c.min(extent - 1)
}

/// Reference `luminance` of the clamped color pixel at `(x, y)`.
fn load_luma(frame: &EdgeFrame, x: i32, y: i32) -> f32 {
    let cx = clamp_coord(x, frame.width);
    let cy = clamp_coord(y, frame.height);
    luminance(frame.color[cy * frame.width + cx])
}

/// Clamped depth sample at `(x, y)`.
fn load_depth(frame: &EdgeFrame, x: i32, y: i32) -> f32 {
    let cx = clamp_coord(x, frame.width);
    let cy = clamp_coord(y, frame.height);
    frame.depth[cy * frame.width + cx]
}

/// Clamped normal at `(x, y)`.
fn load_normal(frame: &EdgeFrame, x: i32, y: i32) -> [f32; 3] {
    let cx = clamp_coord(x, frame.width);
    let cy = clamp_coord(y, frame.height);
    frame.normal[cy * frame.width + cx]
}

/// Gathers the clamped row-major `3x3` window of a scalar sampler around
/// `(x, y)`: index `row * 3 + col`, column mapping to the `x` offset and row to
/// the `y` offset — the exact layout the kernel uses.
fn gather3x3(
    frame: &EdgeFrame,
    x: i32,
    y: i32,
    sample: impl Fn(&EdgeFrame, i32, i32) -> f32,
) -> [f32; 9] {
    let mut w = [0.0f32; 9];
    let mut i = 0usize;
    for dy in -1..=1 {
        for dx in -1..=1 {
            w[i] = sample(frame, x + dx, y + dy);
            i += 1;
        }
    }
    w
}

/// The eight `3x3` normal neighbors (center excluded) in row-major order.
fn normal_neighbors(frame: &EdgeFrame, x: i32, y: i32) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            out.push(load_normal(frame, x + dx, y + dy));
        }
    }
    out
}

/// The `CPU` golden five responses for pixel `(x, y)` under `params`.
fn reference_pixel(frame: &EdgeFrame, params: EdgeParams, x: i32, y: i32) -> [f32; 5] {
    let luma_win = gather3x3(frame, x, y, load_luma);
    let luma_sobel = sobel_magnitude(&luma_win);

    let a = load_luma(frame, x, y);
    let b = load_luma(frame, x + 1, y);
    let c = load_luma(frame, x, y + 1);
    let d = load_luma(frame, x + 1, y + 1);
    let luma_rob = roberts_magnitude(a, b, c, d);

    let depth_win = gather3x3(frame, x, y, load_depth);
    let dep_edge = depth_edge(&depth_win);

    let center = load_normal(frame, x, y);
    let neighbors = normal_neighbors(frame, x, y);
    let nrm_edge = normal_edge(center, &neighbors);

    let mask = params.edge_mask(luma_sobel);
    [luma_sobel, luma_rob, dep_edge, nrm_edge, mask]
}

/// Runs the `GPU` edge detector and asserts pixel- and channel-for-channel
/// parity against the `CPU` golden, returning the `GPU` output for extra
/// per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuEdgeDetect,
    frame: &EdgeFrame,
    params: EdgeParams,
) -> prism_volumetric_gpu::edge_detect::EdgeDetectOutput {
    let query = EdgeDetectQuery {
        frame: frame.clone(),
        params,
    };
    let got = gpu.eval(ctx, &query);

    assert_eq!(got.width, frame.width, "width must match the frame");
    assert_eq!(got.height, frame.height, "height must match the frame");
    assert_eq!(
        got.pixels.len(),
        frame.width * frame.height,
        "pixel count must match the frame"
    );

    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = y * frame.width + x;
            let want = reference_pixel(frame, params, x as i32, y as i32);
            let g = got.pixels[idx];
            let gpu_vals = [
                g.luma_sobel,
                g.luma_roberts,
                g.depth_edge,
                g.normal_edge,
                g.mask,
            ];
            for (channel, (gv, wv)) in gpu_vals.iter().zip(want.iter()).enumerate() {
                assert!(
                    close(*gv, *wv),
                    "pixel ({x}, {y}) channel {channel}: gpu {gv} vs cpu {wv}"
                );
            }
        }
    }
    got
}

/// A flat `[r, g, b]` color plane.
fn flat_color(width: usize, height: usize, rgb: [f32; 3]) -> Vec<[f32; 3]> {
    vec![rgb; width * height]
}

/// A flat depth plane.
fn flat_depth(width: usize, height: usize, d: f32) -> Vec<f32> {
    vec![d; width * height]
}

/// A flat normal plane.
fn flat_normal(width: usize, height: usize, n: [f32; 3]) -> Vec<[f32; 3]> {
    vec![n; width * height]
}

/// Normalizes a 3-vector, defaulting to `+Z` for a degenerate input so the
/// uploaded normals stay well defined.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1.0e-6 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[test]
fn flat_frame_has_no_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A uniform frame: every Sobel/Roberts response is 0 and every normal
    // crease is 0, so the mask is 0 below the band. This pins the no-edge case.
    let (w, h) = (8, 6);
    let frame = EdgeFrame::new(
        w,
        h,
        flat_color(w, h, [0.4, 0.6, 0.8]),
        flat_depth(w, h, 0.5),
        flat_normal(w, h, [0.0, 0.0, 1.0]),
    );
    let params = EdgeParams::new(0.5, 0.25, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    for p in &got.pixels {
        assert!(close(p.luma_sobel, 0.0), "flat luma edge must be zero");
        assert!(close(p.depth_edge, 0.0), "flat depth edge must be zero");
        assert!(close(p.normal_edge, 0.0), "flat normal edge must be zero");
        assert!(close(p.mask, 0.0), "flat mask must be zero below the band");
    }
}

#[test]
fn vertical_step_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // Left half dark, right half bright: a purely vertical luminance and depth
    // break, with a normal flip across the seam.
    let (w, h) = (10, 7);
    let mut color = Vec::with_capacity(w * h);
    let mut depth = Vec::with_capacity(w * h);
    let mut normal = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                color.push([0.0, 0.0, 0.0]);
                depth.push(0.1);
                normal.push([1.0, 0.0, 0.0]);
            } else {
                color.push([1.0, 1.0, 1.0]);
                depth.push(0.9);
                normal.push([-1.0, 0.0, 0.0]);
            }
        }
    }
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    let params = EdgeParams::new(2.0, 1.0, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    // The seam must light up on at least one pixel so the test is not vacuous.
    let max_luma = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.luma_sobel));
    assert!(max_luma > EPS, "the step seam must produce a luma edge");
    let max_depth = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.depth_edge));
    assert!(max_depth > EPS, "the step seam must produce a depth edge");
    let max_normal = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.normal_edge));
    assert!(max_normal > EPS, "the normal flip must produce a crease");
}

#[test]
fn diagonal_gradient_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A diagonal ramp drives both Sobel axes and the Roberts cross at once.
    let (w, h) = (9, 9);
    let mut color = Vec::with_capacity(w * h);
    let mut normal = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x + y) as f32) * 0.1;
            color.push([v, v, v]);
            // A normal that rotates with the diagonal so creases appear too.
            normal.push(normalize([v, 1.0, 0.5]));
        }
    }
    let depth = flat_depth(w, h, 0.3);
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    let params = EdgeParams::new(0.3, 0.2, 1.5);
    let got = check(&ctx, &gpu, &frame, params);
    let max_rob = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.luma_roberts));
    assert!(
        max_rob > EPS,
        "the diagonal ramp must drive the Roberts cross"
    );
}

#[test]
fn isolated_depth_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // Flat color and normals, a horizontal depth cliff: only the depth channel
    // should respond, isolating `depth_edge`.
    let (w, h) = (8, 8);
    let color = flat_color(w, h, [0.5, 0.5, 0.5]);
    let normal = flat_normal(w, h, [0.0, 0.0, 1.0]);
    let mut depth = Vec::with_capacity(w * h);
    for y in 0..h {
        for _x in 0..w {
            depth.push(if y < h / 2 { 0.2 } else { 0.95 });
        }
    }
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    let params = EdgeParams::new(0.5, 0.25, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    for p in &got.pixels {
        assert!(close(p.luma_sobel, 0.0), "flat color implies no luma edge");
        assert!(close(p.normal_edge, 0.0), "flat normals imply no crease");
    }
    let max_depth = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.depth_edge));
    assert!(
        max_depth > EPS,
        "the depth cliff must light the depth channel"
    );
}

#[test]
fn rotating_normal_field_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // Flat color and depth, normals that swing column to column so the
    // `1 - dot` crease is the only non-trivial response.
    let (w, h) = (12, 5);
    let color = flat_color(w, h, [0.7, 0.7, 0.7]);
    let depth = flat_depth(w, h, 0.4);
    let mut normal = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            // Alternating tilt direction between neighbors maximizes the crease.
            let tilt = if x % 2 == 0 { 0.9 } else { -0.9 };
            normal.push(normalize([tilt, 0.2, 1.0]));
        }
    }
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    let params = EdgeParams::new(0.5, 0.25, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    for p in &got.pixels {
        assert!(close(p.luma_sobel, 0.0), "flat color implies no luma edge");
        assert!(close(p.depth_edge, 0.0), "flat depth implies no depth edge");
    }
    let max_normal = got
        .pixels
        .iter()
        .fold(0.0f32, |acc, p| acc.max(p.normal_edge));
    assert!(max_normal > EPS, "swinging normals must produce a crease");
}

#[test]
fn edge_mask_spans_the_smoothstep_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A gentle luminance ramp produces a spread of Sobel magnitudes; with a
    // wide knee the mask must land strictly inside `(0, 1)` on some pixel,
    // exercising the cubic knee rather than only its saturated ends.
    let (w, h) = (16, 4);
    let mut color = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            let v = (x as f32) * 0.05;
            color.push([v, v, v]);
        }
    }
    let depth = flat_depth(w, h, 0.3);
    let normal = flat_normal(w, h, [0.0, 0.0, 1.0]);
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    // luma_sobel for a slope of 0.05/pixel is 4 * 0.05 = 0.2; center the band
    // there with a wide knee so interior pixels sit on the knee's slope.
    let params = EdgeParams::new(0.2, 0.18, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    let partial = got
        .pixels
        .iter()
        .any(|p| p.mask > EPS && p.mask < 1.0 - EPS);
    assert!(partial, "some pixel must land inside the smoothstep knee");
}

#[test]
fn hard_step_mask_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A zero knee is a hard step; drive it with well-separated magnitudes (flat
    // interior at 0, step seam at 4) so neither side of the comparison straddles
    // the discontinuity.
    let (w, h) = (8, 6);
    let mut color = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 0.0 } else { 1.0 };
            color.push([v, v, v]);
        }
    }
    let depth = flat_depth(w, h, 0.3);
    let normal = flat_normal(w, h, [0.0, 0.0, 1.0]);
    let frame = EdgeFrame::new(w, h, color, depth, normal);
    let params = EdgeParams::new(2.0, 0.0, 1.0);
    check(&ctx, &gpu, &frame, params);
}

#[test]
fn single_pixel_frame_round_trips() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A 1x1 frame: every tap clamps onto the single pixel, so all gradients are
    // zero. This stresses the clamp-to-edge path at its extreme.
    let frame = EdgeFrame::new(
        1,
        1,
        vec![[0.3, 0.7, 0.9]],
        vec![0.42],
        vec![normalize([0.1, 0.2, 1.0])],
    );
    let params = EdgeParams::new(0.5, 0.25, 1.0);
    let got = check(&ctx, &gpu, &frame, params);
    assert_eq!(got.pixels.len(), 1);
    assert!(close(got.pixels[0].luma_sobel, 0.0));
    assert!(close(got.pixels[0].normal_edge, 0.0));
}

#[test]
fn empty_frame_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    // A zero-dimension frame yields an empty output and issues no dispatch.
    let frame = EdgeFrame::new(0, 4, Vec::new(), Vec::new(), Vec::new());
    let query = EdgeDetectQuery {
        frame,
        params: EdgeParams::new(0.5, 0.25, 1.0),
    };
    let got = gpu.eval(&ctx, &query);
    assert!(got.pixels.is_empty(), "an empty frame stays empty");
}

#[test]
fn random_frames_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEdgeDetect::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep even, odd, tall, wide and prime resolutions so the clamp-to-edge
    // gather is exercised on every boundary, each compared pixel-for-pixel.
    let resolutions = [
        (2usize, 2usize),
        (3, 7),
        (5, 5),
        (6, 10),
        (9, 4),
        (13, 11),
        (16, 9),
    ];
    for &(w, h) in &resolutions {
        let mut color = Vec::with_capacity(w * h);
        let mut depth = Vec::with_capacity(w * h);
        let mut normal = Vec::with_capacity(w * h);
        for _ in 0..(w * h) {
            color.push([
                lcg(&mut state) * 2.0,
                lcg(&mut state) * 2.0,
                lcg(&mut state) * 2.0,
            ]);
            depth.push(lcg(&mut state));
            // Random normals spanning the sphere, normalized so the crease is
            // well scaled; the component shift keeps them off the origin.
            normal.push(normalize([
                lcg(&mut state) * 2.0 - 1.0,
                lcg(&mut state) * 2.0 - 1.0,
                lcg(&mut state) * 2.0 - 1.0,
            ]));
        }
        let frame = EdgeFrame::new(w, h, color, depth, normal);
        let threshold = 0.2 + lcg(&mut state) * 0.6;
        let knee = 0.1 + lcg(&mut state) * 0.3;
        let scale = 0.5 + lcg(&mut state) * 1.5;
        let params = EdgeParams::new(threshold, knee, scale);
        check(&ctx, &gpu, &frame, params);
    }
}
