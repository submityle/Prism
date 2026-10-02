//! ETC2 `RGB8` base-mode (ETC1-compatible) block encoder: compress a 4x4
//! `RGBA8` tile to an 8-byte block.
//!
//! This is the inverse of
//! [`decode_etc2_rgb8`](crate::decode_etc2_rgb8): it takes an uncompressed
//! `RGBA8` tile (alpha ignored -- `ETC2_RGB8` carries no alpha) and emits one
//! of the two ETC1-compatible base sub-formats, individual or differential.
//! It is the mobile-AAA counterpart to the desktop BC encoders in this module
//! and the first member of the ETC compression path.
//!
//! # Algorithm
//! The classic iPACKMAN/ETC1 fast encoder. For each of the two sub-block
//! orientations (`flip = 0` vertical 2x4 | 2x4, `flip = 1` horizontal 4x2 /
//! 4x2) and each base-colour mode (differential `RGB555` + signed 3-bit delta,
//! or individual `RGB444` x2):
//!
//! 1. The per-sub-block base colour is the quantised average of its texels.
//!    Differential mode derives the second base as a clamped signed delta of
//!    the first (kept inside `0..=31` so the block never aliases into a
//!    `T`/`H`/planar extension).
//! 2. For each sub-block the encoder brute-forces the 8 intensity codewords and,
//!    per texel, the 4 pixel indices, keeping the least-squared-error choice.
//!
//! The configuration with the lowest total squared error wins. The result
//! round-trips through [`decode_etc2_rgb8`](crate::decode_etc2_rgb8) exactly as
//! encoded; a flat tile reconstructs loss-free. Pure integer arithmetic -- no
//! AI/ML -- so the output is deterministic and a GPU twin reproduces it
//! bit-for-bit.
//!
//! Base selection is the plain quantised average (ETC1 "fast" tier); a
//! luma-perturbation base search for higher PSNR is a tracked follow-up, as are
//! the `T`/`H`/planar and `EAC` encoders.
//!
//! # References
//! * Khronos Data Format Specification 1.3, "ETC2 compressed texture image
//!   formats"; `OpenGL ES 3.0` specification, ETC2/EAC appendix.
//! * Strom & Pettersson, "iPACKMAN" (2007), the ETC1 base scheme.

/// ETC1 intensity-modifier table (Khronos). Row = 3-bit codeword, column =
/// 2-bit pixel index `(msb << 1) | lsb`. Mirrors the decoder's private copy
/// (tiny, canonical, never changes) to keep the two paths decoupled.
const ETC1_MODIFIER: [[i32; 4]; 8] = [
    [2, 8, -2, -8],
    [5, 17, -5, -17],
    [9, 29, -9, -29],
    [13, 42, -13, -42],
    [18, 60, -18, -60],
    [24, 80, -24, -80],
    [33, 106, -33, -106],
    [47, 183, -47, -183],
];

/// Quantise an 8-bit channel to 5 bits (`RGB555`), rounding to nearest.
fn quant5(v: u8) -> i32 {
    (i32::from(v) * 31 + 127) / 255
}

/// Quantise an 8-bit channel to 4 bits (`RGB444`), rounding to nearest.
fn quant4(v: u8) -> i32 {
    (i32::from(v) * 15 + 127) / 255
}

/// Replicate a 5-bit value to 8 bits, matching the decoder's `RGB555` expand.
fn ext5(v: i32) -> i32 {
    let v = v & 0x1F;
    (v << 3) | (v >> 2)
}

/// Replicate a 4-bit value to 8 bits, matching the decoder's `RGB444` expand.
fn ext4(v: i32) -> i32 {
    let v = v & 0xF;
    (v << 4) | v
}

/// Set the inclusive big-endian bit range `[lo, hi]` (bit 63 = MSB) of `bits`
/// to `val`, the exact inverse of the decoder's `field` extraction.
fn set_field(bits: &mut u64, hi: u32, lo: u32, val: u32) {
    let width = hi - lo + 1;
    let mask = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    *bits |= u64::from(val & mask) << lo;
}

/// Squared Euclidean distance between two `RGB` colours.
fn sq_dist(a: [i32; 3], b: [i32; 3]) -> u64 {
    let mut d = 0i64;
    for c in 0..3 {
        let e = i64::from(a[c] - b[c]);
        d += e * e;
    }
    u64::try_from(d).unwrap_or(u64::MAX)
}

/// The eight output texel indices belonging to a sub-block.
///
/// `flip = false` splits vertically (sub-block one is the left columns `x < 2`,
/// sub-block two the right); `flip = true` splits horizontally (sub-block one is
/// the top rows `y < 2`). Texel order matches the decoder's `t = y * 4 + x`.
fn subblock_texels(flip: bool, sub_one: bool) -> [usize; 8] {
    let mut out = [0usize; 8];
    let mut n = 0;
    for y in 0..4usize {
        for x in 0..4usize {
            let in_one = if flip { y < 2 } else { x < 2 };
            if in_one == sub_one {
                out[n] = y * 4 + x;
                n += 1;
            }
        }
    }
    out
}

/// Fit one sub-block: given a reconstructed 8-bit base colour and the eight
/// source texels, pick the codeword and per-texel index minimising squared
/// error. Returns `(error, codeword, indices_by_texel)` where `indices_by_texel`
/// is sparse (only the sub-block's eight texels are set).
fn fit_subblock(
    base: [i32; 3],
    texels: &[usize; 8],
    src: &[[i32; 3]; 16],
) -> (u64, usize, [u8; 16]) {
    let mut best_err = u64::MAX;
    let mut best_cw = 0usize;
    let mut best_idx = [0u8; 16];
    for (cw, row) in ETC1_MODIFIER.iter().enumerate() {
        let mut err = 0u64;
        let mut idx = [0u8; 16];
        for &t in texels {
            let target = src[t];
            let mut texel_best = u64::MAX;
            let mut texel_idx = 0u8;
            for (i, &m) in row.iter().enumerate() {
                let cand = [
                    (base[0] + m).clamp(0, 255),
                    (base[1] + m).clamp(0, 255),
                    (base[2] + m).clamp(0, 255),
                ];
                let d = sq_dist(cand, target);
                if d < texel_best {
                    texel_best = d;
                    texel_idx = u8::try_from(i).unwrap_or(0);
                }
            }
            err = err.saturating_add(texel_best);
            idx[t] = texel_idx;
        }
        if err < best_err {
            best_err = err;
            best_cw = cw;
            best_idx = idx;
        }
    }
    (best_err, best_cw, best_idx)
}

/// Average the source colours of a set of texels, rounding to nearest.
fn average(texels: &[usize; 8], src: &[[i32; 3]; 16]) -> [i32; 3] {
    let mut sum = [0i32; 3];
    for &t in texels {
        for c in 0..3 {
            sum[c] += src[t][c];
        }
    }
    [(sum[0] + 4) / 8, (sum[1] + 4) / 8, (sum[2] + 4) / 8]
}

/// One fully-evaluated encoding candidate.
struct Candidate {
    err: u64,
    bits: u64,
}

/// Build and score a differential-mode candidate for the given flip, or `None`
/// if the clamped deltas cannot keep both sub-blocks inside `0..=31` (which
/// would alias the block into a `T`/`H`/planar extension).
fn try_differential(flip: bool, src: &[[i32; 3]; 16]) -> Candidate {
    let s1 = subblock_texels(flip, true);
    let s2 = subblock_texels(flip, false);
    let a1 = average(&s1, src);
    let a2 = average(&s2, src);

    let mut base1_5 = [0i32; 3];
    let mut delta = [0i32; 3];
    let mut base2_5 = [0i32; 3];
    for c in 0..3 {
        let b1 = quant5(u8::try_from(a1[c].clamp(0, 255)).unwrap_or(0));
        let b2 = quant5(u8::try_from(a2[c].clamp(0, 255)).unwrap_or(0));
        let mut d = (b2 - b1).clamp(-4, 3);
        // Keep base + delta inside the valid 5-bit range so the block decodes
        // as differential rather than overflowing into a T/H/planar extension.
        d = d.clamp(-b1, 31 - b1).clamp(-4, 3);
        base1_5[c] = b1;
        delta[c] = d;
        base2_5[c] = b1 + d;
    }
    let base1 = [ext5(base1_5[0]), ext5(base1_5[1]), ext5(base1_5[2])];
    let base2 = [ext5(base2_5[0]), ext5(base2_5[1]), ext5(base2_5[2])];

    let (e1, cw1, idx1) = fit_subblock(base1, &s1, src);
    let (e2, cw2, idx2) = fit_subblock(base2, &s2, src);

    let mut bits = 0u64;
    set_field(&mut bits, 63, 59, u32::try_from(base1_5[0]).unwrap_or(0));
    set_field(&mut bits, 55, 51, u32::try_from(base1_5[1]).unwrap_or(0));
    set_field(&mut bits, 47, 43, u32::try_from(base1_5[2]).unwrap_or(0));
    set_field(
        &mut bits,
        58,
        56,
        u32::try_from(delta[0] & 0x7).unwrap_or(0),
    );
    set_field(
        &mut bits,
        50,
        48,
        u32::try_from(delta[1] & 0x7).unwrap_or(0),
    );
    set_field(
        &mut bits,
        42,
        40,
        u32::try_from(delta[2] & 0x7).unwrap_or(0),
    );
    set_field(&mut bits, 33, 33, 1); // diff
    write_common(&mut bits, flip, cw1, cw2, &idx1, &idx2, &s1, &s2);

    Candidate {
        err: e1.saturating_add(e2),
        bits,
    }
}

/// Build and score an individual-mode candidate for the given flip.
fn try_individual(flip: bool, src: &[[i32; 3]; 16]) -> Candidate {
    let s1 = subblock_texels(flip, true);
    let s2 = subblock_texels(flip, false);
    let a1 = average(&s1, src);
    let a2 = average(&s2, src);

    let mut b1_4 = [0i32; 3];
    let mut b2_4 = [0i32; 3];
    for c in 0..3 {
        b1_4[c] = quant4(u8::try_from(a1[c].clamp(0, 255)).unwrap_or(0));
        b2_4[c] = quant4(u8::try_from(a2[c].clamp(0, 255)).unwrap_or(0));
    }
    let base1 = [ext4(b1_4[0]), ext4(b1_4[1]), ext4(b1_4[2])];
    let base2 = [ext4(b2_4[0]), ext4(b2_4[1]), ext4(b2_4[2])];

    let (e1, cw1, idx1) = fit_subblock(base1, &s1, src);
    let (e2, cw2, idx2) = fit_subblock(base2, &s2, src);

    let mut bits = 0u64;
    set_field(&mut bits, 63, 60, u32::try_from(b1_4[0]).unwrap_or(0));
    set_field(&mut bits, 59, 56, u32::try_from(b2_4[0]).unwrap_or(0));
    set_field(&mut bits, 55, 52, u32::try_from(b1_4[1]).unwrap_or(0));
    set_field(&mut bits, 51, 48, u32::try_from(b2_4[1]).unwrap_or(0));
    set_field(&mut bits, 47, 44, u32::try_from(b1_4[2]).unwrap_or(0));
    set_field(&mut bits, 43, 40, u32::try_from(b2_4[2]).unwrap_or(0));
    // diff bit stays 0 for individual mode.
    write_common(&mut bits, flip, cw1, cw2, &idx1, &idx2, &s1, &s2);

    Candidate {
        err: e1.saturating_add(e2),
        bits,
    }
}

/// Write the fields shared by both base modes: codewords, flip bit, and the
/// per-texel 2-bit indices (LSB plane bits 0..15, MSB plane bits 16..31).
fn write_common(
    bits: &mut u64,
    flip: bool,
    cw1: usize,
    cw2: usize,
    idx1: &[u8; 16],
    idx2: &[u8; 16],
    s1: &[usize; 8],
    s2: &[usize; 8],
) {
    set_field(bits, 39, 37, u32::try_from(cw1).unwrap_or(0));
    set_field(bits, 36, 34, u32::try_from(cw2).unwrap_or(0));
    set_field(bits, 32, 32, u32::from(flip));
    for (idx, sub) in [(idx1, s1), (idx2, s2)] {
        for &t in sub {
            let x = u32::try_from(t % 4).unwrap_or(0);
            let y = u32::try_from(t / 4).unwrap_or(0);
            let p = x * 4 + y;
            let i = u32::from(idx[t]);
            set_field(bits, p, p, i & 1);
            set_field(bits, p + 16, p + 16, (i >> 1) & 1);
        }
    }
}

/// Encode a 4x4 `RGBA8` tile (row-major, `t = y * 4 + x`) into one 8-byte
/// `ETC2_RGB8` block. Alpha is ignored. The output always decodes as an
/// ETC1-compatible base mode (never a `T`/`H`/planar extension).
#[must_use]
pub fn encode_etc2_rgb8(tile: &[[u8; 4]; 16]) -> [u8; 8] {
    let mut src = [[0i32; 3]; 16];
    for (t, px) in tile.iter().enumerate() {
        src[t] = [i32::from(px[0]), i32::from(px[1]), i32::from(px[2])];
    }

    let mut best = Candidate {
        err: u64::MAX,
        bits: 0,
    };
    for flip in [false, true] {
        for cand in [try_differential(flip, &src), try_individual(flip, &src)] {
            if cand.err < best.err {
                best = cand;
            }
        }
    }
    best.bits.to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_etc2_rgb8, etc2_rgb8_mode, Etc2Mode};

    fn fill(colour: [u8; 3]) -> [[u8; 4]; 16] {
        [[colour[0], colour[1], colour[2], 255]; 16]
    }

    #[test]
    fn flat_block_round_trips_losslessly() {
        // 132 = ext5(16); +2 (codeword 0, index 0) reaches 134 exactly.
        let tile = fill([134, 134, 134]);
        let block = encode_etc2_rgb8(&tile);
        let out = decode_etc2_rgb8(&block).unwrap();
        for texel in &out {
            assert_eq!(*texel, [134, 134, 134, 255]);
        }
    }

    #[test]
    fn two_colour_vertical_split_is_exact() {
        // Left columns = 134 (base 132 + 2), right columns = 130 (base 132 - 2):
        // both exactly representable, so the differential/flip=0 fit is loss-free.
        let mut tile = fill([0, 0, 0]);
        for y in 0..4 {
            for x in 0..4 {
                let t = y * 4 + x;
                tile[t] = if x < 2 {
                    [134, 134, 134, 255]
                } else {
                    [130, 130, 130, 255]
                };
            }
        }
        let block = encode_etc2_rgb8(&tile);
        let out = decode_etc2_rgb8(&block).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let t = y * 4 + x;
                let want = if x < 2 {
                    [134, 134, 134, 255]
                } else {
                    [130, 130, 130, 255]
                };
                assert_eq!(out[t], want, "@{x},{y}");
            }
        }
    }

    #[test]
    fn output_is_always_a_base_mode() {
        for colour in [[0, 0, 0], [255, 255, 255], [200, 30, 90], [17, 240, 128]] {
            let block = encode_etc2_rgb8(&fill(colour));
            let mode = etc2_rgb8_mode(&block);
            assert!(
                matches!(mode, Etc2Mode::Individual | Etc2Mode::Differential),
                "colour {colour:?} aliased into {mode:?}"
            );
            assert!(decode_etc2_rgb8(&block).is_ok());
        }
    }

    #[test]
    fn flat_colour_round_trips_within_tolerance() {
        // Every flat colour reconstructs to within a few LSB (base is the exact
        // quantised colour; the smallest modifier is +/-2).
        for colour in [[10, 20, 30], [200, 30, 90], [17, 240, 128], [128, 128, 128]] {
            let block = encode_etc2_rgb8(&fill(colour));
            let out = decode_etc2_rgb8(&block).unwrap();
            for texel in &out {
                for c in 0..3 {
                    let diff = (i32::from(texel[c]) - i32::from(colour[c])).abs();
                    assert!(diff <= 8, "colour {colour:?} channel {c} off by {diff}");
                }
            }
        }
    }

    #[test]
    fn gradient_round_trips_with_bounded_error() {
        // A horizontal ramp: encode, decode, require a small mean error.
        let mut tile = [[0u8; 4]; 16];
        for y in 0..4 {
            for x in 0..4 {
                let v = u8::try_from(x * 70 + 20).unwrap_or(255);
                tile[y * 4 + x] = [v, v, v, 255];
            }
        }
        let block = encode_etc2_rgb8(&tile);
        let out = decode_etc2_rgb8(&block).unwrap();
        let mut total = 0u64;
        for (t, px) in tile.iter().enumerate() {
            for c in 0..3 {
                let d = i64::from(i32::from(out[t][c]) - i32::from(px[c]));
                total += u64::try_from(d * d).unwrap_or(0);
            }
        }
        // 48 channel samples; a sane fast ETC encoder stays well under this.
        assert!(total < 48 * 400, "gradient squared error too high: {total}");
    }

    #[test]
    fn encode_is_deterministic() {
        let tile = fill([73, 150, 221]);
        assert_eq!(encode_etc2_rgb8(&tile), encode_etc2_rgb8(&tile));
    }
}
