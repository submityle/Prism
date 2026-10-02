//! BC1 (DXT1) block encoder: compress a 4x4 `RGBA8` tile to an 8-byte block.
//!
//! This is the inverse of [`decode_bc1`](crate::decode_bc1) and the first
//! member of the texture *compression* path. AAA pipelines bake albedo and
//! other colour maps to BC1 (4 bpp, 1-bit optional alpha) offline, so a
//! correct, high-quality CPU encoder is a prerequisite for a self-contained
//! texture tool. The algorithm is the classic one used by `squish`,
//! `DirectXTex` and NVTT:
//!
//! 1. Fit the dominant colour axis with a few iterations of power iteration on
//!    the 3x3 colour covariance (PCA), avoiding the axis-aligned bounding-box
//!    bias of naive min/max range fitting.
//! 2. Project the tile onto that axis and take the extreme projections as the
//!    initial `RGB565` endpoints.
//! 3. Assign each texel the nearest of the four decoded palette colours, then
//!    refine the endpoints by least squares against those fixed indices and
//!    re-quantise, keeping whichever candidate has the lower sum-of-squared
//!    error. A couple of refinement passes recover most of the quality a full
//!    cluster fit would.
//!
//! Transparency: if any texel is below the 1-bit alpha threshold the block is
//! emitted in BC1's 3-colour punch-through mode (`color0 <= color1`), with the
//! transparent texels assigned index 3; otherwise the opaque 4-colour mode is
//! forced (`color0 > color1`). Either way the output round-trips through
//! [`decode_bc1`](crate::decode_bc1) exactly as encoded.
//!
//! Pure analytic/integer arithmetic -- no AI/ML -- so the result is
//! deterministic and a GPU twin encoder reproduces it bit-for-bit given the
//! same tie-breaks.
//!
//! # References
//! * S. Brown, "`squish`" DXT compression library (PCA + least-squares fit).
//! * J.M.P. van Waveren & I. Castano, "Real-Time `YCoCg`-DXT Compression"
//!   (`NVIDIA`, endpoint fitting background).
//! * Khronos Data Format Spec 1.3, S3TC/BC1 block layout.

use super::super::color_block::rgb565_to_rgb888;

/// Alpha below this (0-255) encodes as BC1 punch-through transparent.
const ALPHA_THRESHOLD: u8 = 128;

/// Truncating `(2a + b) / 3`, matching the BC1 decoder palette.
#[inline]
fn third(a: u8, b: u8) -> u8 {
    ((2 * u16::from(a) + u16::from(b)) / 3) as u8
}

/// Round-half-up quantise a `[0, 255]` channel to an `n`-bit code.
#[inline]
fn quantize(value: f32, max_code: f32) -> u16 {
    let v = (value / 255.0).clamp(0.0, 1.0);
    (v * max_code + 0.5).floor() as u16
}

/// Pack an `[0, 255]` RGB endpoint into `RGB565`.
#[inline]
fn pack_565(rgb: [f32; 3]) -> u16 {
    let r = quantize(rgb[0], 31.0);
    let g = quantize(rgb[1], 63.0);
    let b = quantize(rgb[2], 31.0);
    (r << 11) | (g << 5) | b
}

#[inline]
fn to_f32(c: [u8; 3]) -> [f32; 3] {
    [f32::from(c[0]), f32::from(c[1]), f32::from(c[2])]
}

/// Squared RGB distance between two colours.
#[inline]
fn dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    let d = |x: u8, y: u8| {
        let v = i32::from(x) - i32::from(y);
        (v * v) as u32
    };
    d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
}

/// Principal colour axis of the opaque texels via power iteration on the 3x3
/// covariance. Returns a (not necessarily normalised) direction; falls back to
/// the luma-ish grey axis when the cluster is degenerate.
fn principal_axis(points: &[[f32; 3]]) -> [f32; 3] {
    let n = points.len() as f32;
    let mut mean = [0.0f32; 3];
    for p in points {
        for c in 0..3 {
            mean[c] += p[c];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    // Symmetric covariance (upper triangle).
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let dx = p[0] - mean[0];
        let dy = p[1] - mean[1];
        let dz = p[2] - mean[2];
        xx += dx * dx;
        xy += dx * dy;
        xz += dx * dz;
        yy += dy * dy;
        yz += dy * dz;
        zz += dz * dz;
    }
    // Power iteration from a stable non-degenerate seed.
    let mut v = [1.0f32, 1.0, 1.0];
    for _ in 0..8 {
        let nv = [
            xx * v[0] + xy * v[1] + xz * v[2],
            xy * v[0] + yy * v[1] + yz * v[2],
            xz * v[0] + yz * v[1] + zz * v[2],
        ];
        let len2 = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
        if len2 <= 1.0e-12 {
            // Covariance annihilates the vector: cluster is a single point.
            return [1.0, 1.0, 1.0];
        }
        let inv = 1.0 / len2.sqrt();
        v = [nv[0] * inv, nv[1] * inv, nv[2] * inv];
    }
    v
}

/// Build the decoded 4-colour palette from two `RGB565` endpoints (opaque mode).
fn palette4(c0: u16, c1: u16) -> [[u8; 3]; 4] {
    let e0 = rgb565_to_rgb888(c0);
    let e1 = rgb565_to_rgb888(c1);
    [
        e0,
        e1,
        [third(e0[0], e1[0]), third(e0[1], e1[1]), third(e0[2], e1[2])],
        [third(e1[0], e0[0]), third(e1[1], e0[1]), third(e1[2], e0[2])],
    ]
}

/// Interpolation weight (fraction toward endpoint 1) for each opaque palette
/// index: `0 -> 0`, `2 -> 1/3`, `3 -> 2/3`, `1 -> 1`.
const WEIGHT: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];

/// Assign nearest opaque palette indices and return `(indices, total_error)`.
fn assign4(colors: &[[u8; 3]; 16], palette: &[[u8; 3]; 4]) -> ([u8; 16], u64) {
    let mut idx = [0u8; 16];
    let mut err = 0u64;
    for (t, &c) in colors.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (i, &p) in palette.iter().enumerate() {
            let d = dist2(c, p);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        idx[t] = best as u8;
        err += u64::from(best_d);
    }
    (idx, err)
}

/// Least-squares re-fit of the two endpoints given fixed opaque indices.
fn refit(colors: &[[u8; 3]; 16], idx: &[u8; 16]) -> Option<([f32; 3], [f32; 3])> {
    let (mut a, mut b, mut c) = (0.0f32, 0.0, 0.0);
    let mut p = [0.0f32; 3];
    let mut q = [0.0f32; 3];
    for (t, &c8) in colors.iter().enumerate() {
        let w = WEIGHT[idx[t] as usize];
        let iw = 1.0 - w;
        a += iw * iw;
        b += iw * w;
        c += w * w;
        let cf = to_f32(c8);
        for ch in 0..3 {
            p[ch] += iw * cf[ch];
            q[ch] += w * cf[ch];
        }
    }
    let det = a * c - b * b;
    if det.abs() <= 1.0e-6 {
        return None;
    }
    let inv = 1.0 / det;
    let mut e0 = [0.0f32; 3];
    let mut e1 = [0.0f32; 3];
    for ch in 0..3 {
        e0[ch] = (c * p[ch] - b * q[ch]) * inv;
        e1[ch] = (a * q[ch] - b * p[ch]) * inv;
    }
    Some((e0, e1))
}

fn pack_block(c0: u16, c1: u16, idx: &[u8; 16]) -> [u8; 8] {
    let mut bits = 0u32;
    for (t, &i) in idx.iter().enumerate() {
        bits |= u32::from(i & 0x3) << (2 * t);
    }
    let mut out = [0u8; 8];
    out[0] = (c0 & 0xFF) as u8;
    out[1] = (c0 >> 8) as u8;
    out[2] = (c1 & 0xFF) as u8;
    out[3] = (c1 >> 8) as u8;
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

/// Encode the opaque 4-colour mode. Returns the packed block and its error.
fn encode_opaque(colors: &[[u8; 3]; 16]) -> ([u8; 8], u64) {
    let points: Vec<[f32; 3]> = colors.iter().map(|&c| to_f32(c)).collect();
    let axis = principal_axis(&points);

    // Project onto the axis; the extreme projections seed the endpoints.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut lo_c = points[0];
    let mut hi_c = points[0];
    for &pt in &points {
        let d = pt[0] * axis[0] + pt[1] * axis[1] + pt[2] * axis[2];
        if d < lo {
            lo = d;
            lo_c = pt;
        }
        if d > hi {
            hi = d;
            hi_c = pt;
        }
    }

    let mut ep0 = hi_c;
    let mut ep1 = lo_c;
    let mut best: Option<([u8; 8], u64)> = None;

    // Initial fit plus two least-squares refinement passes.
    for _ in 0..3 {
        let (c0q, c1q, idx, err) = fit_endpoints(colors, ep0, ep1);
        let improved = best.as_ref().map(|(_, e)| err < *e).unwrap_or(true);
        if improved {
            best = Some((pack_block(c0q, c1q, &idx), err));
        }
        match refit(colors, &idx) {
            Some((n0, n1)) => {
                ep0 = [
                    n0[0].clamp(0.0, 255.0),
                    n0[1].clamp(0.0, 255.0),
                    n0[2].clamp(0.0, 255.0),
                ];
                ep1 = [
                    n1[0].clamp(0.0, 255.0),
                    n1[1].clamp(0.0, 255.0),
                    n1[2].clamp(0.0, 255.0),
                ];
            }
            None => break,
        }
    }
    best.unwrap_or_else(|| (pack_block(0, 0, &[0u8; 16]), u64::MAX))
}

/// Quantise a pair of float endpoints, force 4-colour ordering (`c0 > c1`), and
/// assign nearest indices. Returns `(c0, c1, indices, error)`.
fn fit_endpoints(colors: &[[u8; 3]; 16], ep0: [f32; 3], ep1: [f32; 3]) -> (u16, u16, [u8; 16], u64) {
    let mut c0 = pack_565(ep0);
    let mut c1 = pack_565(ep1);
    // 4-colour opaque mode requires c0 > c1. If equal, nudge so the mode and
    // palette stay well-defined (single-colour block still decodes correctly).
    if c0 == c1 {
        if c1 > 0 {
            c1 -= 1;
        } else {
            c0 += 1;
        }
    }
    let mut swapped = false;
    if c0 < c1 {
        core::mem::swap(&mut c0, &mut c1);
        swapped = true;
    }
    let palette = palette4(c0, c1);
    let (mut idx, err) = assign4(colors, &palette);
    // The swap relabels endpoints 0<->1; interpolated 2<->3 follow.
    if swapped {
        for i in idx.iter_mut() {
            *i = match *i {
                0 => 1,
                1 => 0,
                2 => 3,
                _ => 2,
            };
        }
    }
    (c0, c1, idx, err)
}

/// Encode the 3-colour punch-through mode: `color0 <= color1`, index 3 is
/// transparent black, indices 0/1/2 are the two endpoints and their midpoint.
fn encode_punchthrough(colors: &[[u8; 3]; 16], alpha: &[u8; 16]) -> [u8; 8] {
    // Fit endpoints over only the opaque texels.
    let opaque: Vec<[f32; 3]> = colors
        .iter()
        .zip(alpha.iter())
        .filter(|&(_, &a)| a >= ALPHA_THRESHOLD)
        .map(|(&c, _)| to_f32(c))
        .collect();

    let (mut c0, mut c1) = if opaque.is_empty() {
        (0u16, 0u16)
    } else {
        let axis = principal_axis(&opaque);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let (mut lo_c, mut hi_c) = (opaque[0], opaque[0]);
        for &pt in &opaque {
            let d = pt[0] * axis[0] + pt[1] * axis[1] + pt[2] * axis[2];
            if d < lo {
                lo = d;
                lo_c = pt;
            }
            if d > hi {
                hi = d;
                hi_c = pt;
            }
        }
        (pack_565(lo_c), pack_565(hi_c))
    };
    // 3-colour mode requires c0 <= c1.
    if c0 > c1 {
        core::mem::swap(&mut c0, &mut c1);
    }

    let e0 = rgb565_to_rgb888(c0);
    let e1 = rgb565_to_rgb888(c1);
    let mid = [
        ((u16::from(e0[0]) + u16::from(e1[0])) / 2) as u8,
        ((u16::from(e0[1]) + u16::from(e1[1])) / 2) as u8,
        ((u16::from(e0[2]) + u16::from(e1[2])) / 2) as u8,
    ];
    let palette3 = [e0, e1, mid];

    let mut idx = [0u8; 16];
    for (t, &c) in colors.iter().enumerate() {
        if alpha[t] < ALPHA_THRESHOLD {
            idx[t] = 3; // transparent black
            continue;
        }
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (i, &p) in palette3.iter().enumerate() {
            let d = dist2(c, p);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        idx[t] = best as u8;
    }
    pack_block(c0, c1, &idx)
}

/// Encode one 4x4 `RGBA8` tile (row-major, texel `t = y*4 + x`) to an 8-byte
/// BC1 block.
///
/// Any texel with `alpha < 128` triggers BC1's 3-colour punch-through mode and
/// that texel becomes transparent black; otherwise the opaque 4-colour mode is
/// used. The output round-trips through [`decode_bc1`](crate::decode_bc1).
#[must_use]
pub fn encode_bc1(tile: &[[u8; 4]; 16]) -> [u8; 8] {
    let mut colors = [[0u8; 3]; 16];
    let mut alpha = [0u8; 16];
    for (t, texel) in tile.iter().enumerate() {
        colors[t] = [texel[0], texel[1], texel[2]];
        alpha[t] = texel[3];
    }
    if alpha.iter().any(|&a| a < ALPHA_THRESHOLD) {
        encode_punchthrough(&colors, &alpha)
    } else {
        encode_opaque(&colors).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_bc1;

    fn tile_from(colors: [[u8; 4]; 16]) -> [[u8; 4]; 16] {
        colors
    }

    fn ssd(a: [[u8; 4]; 16], b: [[u8; 4]; 16]) -> u64 {
        let mut e = 0u64;
        for t in 0..16 {
            for c in 0..3 {
                let d = i32::from(a[t][c]) - i32::from(b[t][c]);
                e += (d * d) as u64;
            }
        }
        e
    }

    #[test]
    fn solid_colour_round_trips_closely() {
        let col = [120u8, 200, 60, 255];
        let tile = tile_from([col; 16]);
        let block = encode_bc1(&tile);
        let decoded = decode_bc1(&block);
        // Every texel should decode to the same colour, within 565 quantisation.
        for texel in decoded {
            assert!(i32::from(texel[0]) - 120 <= 8 && 120 - i32::from(texel[0]) <= 8);
            assert_eq!(texel[3], 255);
        }
        assert!(ssd(tile, decoded) < 16 * 3 * 64, "solid ssd too high");
    }

    #[test]
    fn black_white_endpoints_are_preserved() {
        let mut tile = [[0u8, 0, 0, 255]; 16];
        for t in (0..16).step_by(2) {
            tile[t] = [255, 255, 255, 255];
        }
        let decoded = decode_bc1(&encode_bc1(&tile));
        // Pure black and white must survive as near-exact endpoints.
        let has_white = decoded.iter().any(|t| t[0] > 240 && t[1] > 240 && t[2] > 240);
        let has_black = decoded.iter().any(|t| t[0] < 16 && t[1] < 16 && t[2] < 16);
        assert!(has_white && has_black, "endpoints lost: {decoded:?}");
    }

    #[test]
    fn smooth_gradient_error_is_bounded() {
        // A horizontal grey ramp sits on the colour axis, so BC1 should encode
        // it with small error.
        let mut tile = [[0u8; 4]; 16];
        for y in 0..4 {
            for x in 0..4 {
                let v = (x * 85) as u8; // 0, 85, 170, 255
                tile[y * 4 + x] = [v, v, v, 255];
            }
        }
        let decoded = decode_bc1(&encode_bc1(&tile));
        // 16 texels * 3 channels; a well-fit ramp stays well under ~20 LSB RMS.
        assert!(ssd(tile, decoded) < 16 * 3 * 400, "gradient ssd {}", ssd(tile, decoded));
    }

    #[test]
    fn opaque_block_decodes_fully_opaque() {
        let tile = tile_from([[10, 150, 240, 255]; 16]);
        for texel in decode_bc1(&encode_bc1(&tile)) {
            assert_eq!(texel[3], 255);
        }
    }

    #[test]
    fn transparent_texels_round_trip_as_transparent() {
        let mut tile = [[200u8, 50, 50, 255]; 16];
        tile[0] = [0, 0, 0, 0];
        tile[5] = [0, 0, 0, 10];
        let decoded = decode_bc1(&encode_bc1(&tile));
        assert_eq!(decoded[0][3], 0, "texel 0 should be transparent");
        assert_eq!(decoded[5][3], 0, "texel 5 should be transparent");
        // An opaque texel keeps full coverage.
        assert_eq!(decoded[1][3], 255);
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            *texel = [(t * 17) as u8, (255 - t * 15) as u8, (t * 9) as u8, 255];
        }
        assert_eq!(encode_bc1(&tile), encode_bc1(&tile));
    }

    #[test]
    fn refit_beats_or_matches_initial_on_two_tone() {
        // Two clusters along a diagonal: the least-squares pass must not make
        // the error worse than the seed fit.
        let mut tile = [[20u8, 30, 200, 255]; 16];
        for texel in tile.iter_mut().skip(8) {
            *texel = [220, 180, 40, 255];
        }
        let colors: [[u8; 3]; 16] = core::array::from_fn(|t| [tile[t][0], tile[t][1], tile[t][2]]);
        let (_, err) = encode_opaque(&colors);
        let decoded = decode_bc1(&encode_bc1(&tile));
        assert_eq!(err, ssd(tile, decoded));
    }

    #[test]
    fn single_colour_axis_is_stable() {
        // Degenerate cluster (all identical) must not divide by zero or panic.
        let tile = tile_from([[77, 77, 77, 255]; 16]);
        let decoded = decode_bc1(&encode_bc1(&tile));
        for texel in decoded {
            assert!((i32::from(texel[0]) - 77).abs() <= 8);
        }
    }
}
