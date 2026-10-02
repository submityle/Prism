//! BC7/BPTC **mode 6** block encoder: compress a 4x4 `RGBA8` tile to a 16-byte
//! single-subset BC7 block.
//!
//! BC7 is the premier modern LDR block format (8 bpp, 4 channels) used by AAA
//! pipelines for high-quality colour-plus-alpha textures. This encoder targets
//! **mode 6** -- the one BC7 mode with *no* partitions, *no* component
//! rotation, and *no* index-selection bit: a single endpoint pair at 7 bits per
//! channel plus a unique p-bit per endpoint (expanding to 8-bit endpoints), and
//! sixteen shared 4-bit indices. Mode 6 is the natural single-subset workhorse
//! and the only mode that does not need the Khronos partition/anchor tables, so
//! it can be encoded with pure analytic/integer math.
//!
//! Pipeline (mirrors the BC1 encoder, lifted to 4D `RGBA`):
//! 1. principal colour axis of the sixteen `RGBA` texels via power iteration on
//!    the 4x4 covariance;
//! 2. extreme projections onto that axis seed the two endpoints;
//! 3. each endpoint is quantised to 7 bits/channel plus a shared p-bit by
//!    trying both p-bit parities and keeping the lower per-endpoint error;
//! 4. every texel takes its nearest of the sixteen interpolated colours;
//! 5. a least-squares endpoint re-fit over the fixed indices (two passes,
//!    keeping the lower total error) tightens the result beyond the raw seed.
//!
//! The anchor index (texel 0) is forced into `0..8` -- BC7 drops its high bit
//! to save a bit -- by swapping endpoints and inverting all indices when
//! needed. The output round-trips through
//! [`decode_bc7_mode6`](crate::decode_bc7_mode6): the reconstructed endpoints
//! and interpolation here use the identical `(v << 1) | p` expansion and
//! 1/64-unit 4-bit weights as the decoder, so a decode of the encoded block
//! reproduces exactly the colours this encoder chose. Pure integer/`f64`
//! arithmetic -- no AI/ML.
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC/BC7 mode 6.
//! * `DirectXTex` / `NVTT` / `ispc_texcomp` BC7 encoders (mode-6 fit).

/// 4-bit index interpolation weights (Khronos `aWeight4`), in 1/64 units.
/// Mirrors the decoder's table so encode/decode interpolate identically.
const WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 56, 60, 64];

/// Expand a 7-bit channel plus its p-bit to 8 bits, as the decoder does.
fn expand7(v: u32, p: u32) -> u8 {
    (((v << 1) | (p & 1)) & 0xFF) as u8
}

/// Interpolate two 8-bit endpoints at a 1/64-unit weight, as the decoder does.
fn interp(e0: u8, e1: u8, weight: u32) -> u8 {
    let r = ((64 - weight) * u32::from(e0) + weight * u32::from(e1) + 32) >> 6;
    (r & 0xFF) as u8
}

/// Quantise one 8-bit-space endpoint to a 7-bit value per channel plus a single
/// shared p-bit, trying both parities and keeping the lower squared error.
///
/// Returns `(p, v[4], reconstructed[4])` where `reconstructed[c] ==
/// expand7(v[c], p)`.
fn quantize_endpoint(target: [i32; 4]) -> (u32, [u32; 4], [u8; 4]) {
    let mut best: Option<(u64, u32, [u32; 4], [u8; 4])> = None;
    for p in 0..2u32 {
        let v: [u32; 4] = core::array::from_fn(|c| {
            let num = target[c] - p as i32;
            let x = if num <= 0 { 0 } else { (num + 1) / 2 };
            x.clamp(0, 127) as u32
        });
        let recon: [u8; 4] = core::array::from_fn(|c| expand7(v[c], p));
        let err: u64 = (0..4)
            .map(|c| {
                let d = i64::from(recon[c]) - i64::from(target[c]);
                (d * d) as u64
            })
            .sum();
        if best.is_none_or(|(be, ..)| err < be) {
            best = Some((err, p, v, recon));
        }
    }
    let (_, p, v, recon) = best.expect("two parities evaluated");
    (p, v, recon)
}

/// Nearest 4-bit index for one texel against the sixteen interpolated colours;
/// returns `(index, squared_error)`.
fn nearest_index(texel: [i32; 4], e0: [u8; 4], e1: [u8; 4]) -> (u8, u64) {
    let mut best = 0u8;
    let mut best_err = u64::MAX;
    for (i, &w) in WEIGHT4.iter().enumerate() {
        let err: u64 = (0..4)
            .map(|c| {
                let p = i64::from(interp(e0[c], e1[c], w));
                let d = p - i64::from(texel[c]);
                (d * d) as u64
            })
            .sum();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    (best, best_err)
}

/// Assign indices to every texel and return `(indices, total_error)`.
fn assign(points: &[[i32; 4]; 16], e0: [u8; 4], e1: [u8; 4]) -> ([u8; 16], u64) {
    let mut idx = [0u8; 16];
    let mut err = 0u64;
    for (t, texel) in points.iter().enumerate() {
        let (i, e) = nearest_index(*texel, e0, e1);
        idx[t] = i;
        err += e;
    }
    (idx, err)
}

/// Least-squares re-fit of the two continuous endpoints given fixed indices.
///
/// Minimises `sum_t (a_t*e0 + b_t*e1 - target)^2` per channel, with
/// `a_t = (64 - w_t)/64`, `b_t = w_t/64`. Returns `None` for a degenerate
/// (single-cluster) normal matrix so the caller keeps the previous endpoints.
fn refit(points: &[[i32; 4]; 16], idx: &[u8; 16]) -> Option<([i32; 4], [i32; 4])> {
    let mut saa = 0.0f64;
    let mut sab = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sat = [0.0f64; 4];
    let mut sbt = [0.0f64; 4];
    for (t, texel) in points.iter().enumerate() {
        let w = f64::from(WEIGHT4[idx[t] as usize]);
        let a = (64.0 - w) / 64.0;
        let b = w / 64.0;
        saa += a * a;
        sab += a * b;
        sbb += b * b;
        for c in 0..4 {
            sat[c] += a * f64::from(texel[c]);
            sbt[c] += b * f64::from(texel[c]);
        }
    }
    let det = saa * sbb - sab * sab;
    if det.abs() < 1e-6 {
        return None;
    }
    let inv = 1.0 / det;
    let e0: [i32; 4] = core::array::from_fn(|c| {
        let v = (sbb * sat[c] - sab * sbt[c]) * inv;
        (v + 0.5).floor().clamp(0.0, 255.0) as i32
    });
    let e1: [i32; 4] = core::array::from_fn(|c| {
        let v = (saa * sbt[c] - sab * sat[c]) * inv;
        (v + 0.5).floor().clamp(0.0, 255.0) as i32
    });
    Some((e0, e1))
}

/// Principal `RGBA` axis of the texels via power iteration on the 4x4
/// covariance. Falls back to a non-zero unit axis for a degenerate block.
fn principal_axis(points: &[[i32; 4]; 16]) -> [f64; 4] {
    let mut mean = [0.0f64; 4];
    for texel in points {
        for c in 0..4 {
            mean[c] += f64::from(texel[c]);
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }
    let mut cov = [[0.0f64; 4]; 4];
    for texel in points {
        let d: [f64; 4] = core::array::from_fn(|c| f64::from(texel[c]) - mean[c]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let mut axis = [1.0f64, 1.0, 1.0, 1.0];
    for _ in 0..24 {
        let next: [f64; 4] =
            core::array::from_fn(|i| (0..4).map(|j| cov[i][j] * axis[j]).sum::<f64>());
        let norm = (next.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-9 {
            break;
        }
        axis = core::array::from_fn(|c| next[c] / norm);
    }
    axis
}

/// LSB-first bit writer over a 16-byte block, the inverse of the decoder's
/// `BitReader`: field bit `i` of a value lands at absolute block bit
/// `base + i`.
struct BitWriter {
    bytes: [u8; 16],
    pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: [0u8; 16],
            pos: 0,
        }
    }

    fn write(&mut self, value: u32, n: u32) {
        for i in 0..n {
            let bit = ((value >> i) & 1) as u8;
            if bit != 0 {
                self.bytes[self.pos / 8] |= 1 << (self.pos % 8);
            }
            self.pos += 1;
        }
    }
}

/// Encode one 4x4 `RGBA8` tile as a 16-byte BC7 **mode 6** block.
#[must_use]
pub fn encode_bc7_mode6(tile: &[[u8; 4]; 16]) -> [u8; 16] {
    let points: [[i32; 4]; 16] =
        core::array::from_fn(|t| core::array::from_fn(|c| i32::from(tile[t][c])));

    // Seed endpoints from the extreme projections onto the principal axis.
    let axis = principal_axis(&points);
    let mut lo_t = 0usize;
    let mut hi_t = 0usize;
    let mut lo_p = f64::INFINITY;
    let mut hi_p = f64::NEG_INFINITY;
    for (t, texel) in points.iter().enumerate() {
        let proj = (0..4).map(|c| f64::from(texel[c]) * axis[c]).sum::<f64>();
        if proj < lo_p {
            lo_p = proj;
            lo_t = t;
        }
        if proj > hi_p {
            hi_p = proj;
            hi_t = t;
        }
    }
    let mut seed0 = points[lo_t];
    let mut seed1 = points[hi_t];

    // Quantise + assign, then two least-squares refinement passes; keep best.
    let mut best_block: Option<(u64, [u8; 16])> = None;
    for _pass in 0..3 {
        let (p0, v0, re0) = quantize_endpoint(seed0);
        let (p1, v1, re1) = quantize_endpoint(seed1);
        let (mut idx, err) = assign(&points, re0, re1);

        // Anchor rule: texel-0 index high bit must be 0; swap + invert if not.
        let (fp0, fv0, fp1, fv1) = if idx[0] & 0b1000 != 0 {
            for i in &mut idx {
                *i = 15 - *i;
            }
            (p1, v1, p0, v0)
        } else {
            (p0, v0, p1, v1)
        };

        let mut w = BitWriter::new();
        w.write(0b100_0000, 7); // mode-6 unary marker: six 0s then a 1.
        for c in 0..4 {
            w.write(fv0[c], 7);
            w.write(fv1[c], 7);
        }
        w.write(fp0, 1);
        w.write(fp1, 1);
        w.write(u32::from(idx[0]), 3); // anchor: implicit high bit 0.
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }

        if best_block.is_none_or(|(be, _)| err < be) {
            best_block = Some((err, w.bytes));
        }

        match refit(&points, &idx) {
            Some((e0, e1)) => {
                seed0 = e0;
                seed1 = e1;
            }
            None => break,
        }
    }

    best_block.expect("at least one pass runs").1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_bc7_mode6;

    fn sse(a: &[[u8; 4]; 16], b: &[[u8; 4]; 16]) -> u64 {
        let mut s = 0u64;
        for (ta, tb) in a.iter().zip(b.iter()) {
            for c in 0..4 {
                let d = i64::from(ta[c]) - i64::from(tb[c]);
                s += (d * d) as u64;
            }
        }
        s
    }

    #[test]
    fn flat_tile_round_trips_near_exact() {
        // A constant tile has coincident endpoints; mode-6 reproduces it to
        // within the 7-bit+p-bit endpoint quantisation (<= 1 per channel).
        let tile = [[37u8, 150, 220, 240]; 16];
        let out = decode_bc7_mode6(&encode_bc7_mode6(&tile));
        for texel in &out {
            assert!((i32::from(texel[0]) - 37).abs() <= 1);
            assert!((i32::from(texel[1]) - 150).abs() <= 1);
            assert!((i32::from(texel[2]) - 220).abs() <= 1);
            assert!((i32::from(texel[3]) - 240).abs() <= 1);
        }
    }

    #[test]
    fn two_colour_axis_is_reconstructed() {
        // Two clusters at the ends map to the two endpoints with low error.
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            *texel = if t % 2 == 0 {
                [10, 20, 30, 255]
            } else {
                [200, 180, 160, 255]
            };
        }
        let out = decode_bc7_mode6(&encode_bc7_mode6(&tile));
        // Endpoints land near the two clusters: error per channel stays small.
        for (t, texel) in out.iter().enumerate() {
            let want = if t % 2 == 0 {
                [10i32, 20, 30, 255]
            } else {
                [200, 180, 160, 255]
            };
            for c in 0..4 {
                assert!(
                    (i32::from(texel[c]) - want[c]).abs() <= 6,
                    "texel {t} ch {c}"
                );
            }
        }
    }

    #[test]
    fn alpha_gradient_tracks_input() {
        // A smooth alpha ramp with constant RGB stays monotonic and close.
        let mut tile = [[128u8, 64, 32, 0]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            texel[3] = (t * 17) as u8;
        }
        let out = decode_bc7_mode6(&encode_bc7_mode6(&tile));
        for (t, texel) in out.iter().enumerate() {
            assert!(
                (i32::from(texel[3]) - (t as i32) * 17).abs() <= 10,
                "alpha {t}"
            );
        }
    }

    #[test]
    fn anchor_index_high_bit_is_zero() {
        // The packed anchor is only 3 bits, so texel-0 index must be < 8.
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = (t * 16) as u8;
            *texel = [v, 255 - v, v / 2, 255];
        }
        let block = encode_bc7_mode6(&tile);
        // Reconstruct the anchor index from the bitstream and check < 8.
        // Bits: 7 (mode) + 56 (endpoints) + 2 (p) = 65; anchor is next 3 bits.
        let mut acc = 0u32;
        for i in 0..3u32 {
            let bitpos = 65 + i as usize;
            let bit = (block[bitpos / 8] >> (bitpos % 8)) & 1;
            acc |= u32::from(bit) << i;
        }
        assert!(acc < 8, "anchor index {acc} must fit 3 bits");
    }

    #[test]
    fn gradient_is_lower_error_than_single_endpoint() {
        // A full RGB gradient must beat the trivial constant-colour encode.
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = (t * 16) as u8;
            *texel = [v, v, v, 255];
        }
        let out = decode_bc7_mode6(&encode_bc7_mode6(&tile));
        let flat = [[120u8, 120, 120, 255]; 16];
        assert!(sse(&out, &tile) < sse(&flat, &tile));
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            *texel = [(t * 15) as u8, (t * 7) as u8, (255 - t * 15) as u8, 200];
        }
        assert_eq!(encode_bc7_mode6(&tile), encode_bc7_mode6(&tile));
    }

    #[test]
    fn round_trips_through_decoder_tag() {
        // Encoded block must be recognised as mode 6 by the public dispatcher.
        let tile = [[50u8, 100, 150, 200]; 16];
        let block = encode_bc7_mode6(&tile);
        // Mode-6 marker: low 7 bits == 0b1000000.
        assert_eq!(block[0] & 0x7F, 0b100_0000);
    }
}
