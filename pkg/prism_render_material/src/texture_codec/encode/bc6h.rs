//! BC6H/BPTC **mode 11** HDR block encoder: compress a 4x4 tile of
//! non-negative IEEE half-float `RGB` texels to a 16-byte single-subset BC6H
//! block (unsigned profile).
//!
//! BC6H is the GPU HDR block format (8 bpp, three channels, no alpha) used by
//! AAA pipelines for lightmaps, reflection probes, and HDR source textures.
//! This encoder targets the **unsigned mode 11** layout -- the single-subset
//! mode with direct 10-bit endpoints, no partitions, no delta coding, and a
//! single sixteen-entry 4-bit index set. Like BC7 mode 6, mode 11 needs *none*
//! of the Khronos partition/anchor tables, so it can be encoded with pure
//! integer/`f64` analytic math.
//!
//! Input is a tile of half-float **bit patterns** (`u16`), the native BC6H
//! storage domain: the decoder's finish stage emits exactly these patterns, so
//! encoding/round-tripping stays in the quasi-logarithmic half space the
//! format was designed around. Negative, infinite, and `NaN` inputs are
//! clamped to the representable `[0, 0x7BFF]` positive-finite range (the
//! unsigned profile cannot store a sign).
//!
//! Pipeline (mirrors the BC7 mode-6 encoder, lifted to the BC6H finish chain):
//! 1. map each target half to the pre-finish intermediate domain
//!    `T = round(h * 64 / 31)` (the inverse of the decoder's `(q*31) >> 6`
//!    finish scale), giving a monotonic 16-bit target per channel;
//! 2. principal `RGB` axis of the sixteen targets via power iteration;
//! 3. extreme projections seed the two 16-bit endpoints;
//! 4. each endpoint is quantised to a 10-bit code (`unquantize_unsigned`
//!    inverse) and reconstructed exactly as the decoder will;
//! 5. every texel takes its nearest of the sixteen *finished* half values
//!    (exact decoder forward), and a least-squares endpoint re-fit over the
//!    fixed indices (two passes, keep lower error) tightens the result.
//!
//! The anchor index (texel 0) is forced into `0..8` by swapping endpoints and
//! inverting all indices when needed. The output round-trips through
//! [`decode_bc6h_mode11_unsigned`](crate::decode_bc6h_mode11_unsigned) and is
//! tagged mode 11 for [`decode_bc6h_unsigned`](crate::decode_bc6h_unsigned).
//! Pure integer/`f64` arithmetic -- no AI/ML.
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC/BC6H mode 11 (unsigned).
//! * `DirectXTex` / `NVTT` / `ispc_texcomp` BC6H encoders (single-subset fit).

/// 4-bit index interpolation weights (Khronos `aWeight4`), in 1/64 units.
/// Mirrors the decoder's table so encode/decode interpolate identically.
const WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 56, 60, 64];

/// Largest half-float bit pattern that the unsigned finish stage can emit
/// (`65504.0`, the maximum finite half). Infinities and `NaN`s clamp here.
const MAX_HALF: u32 = 0x7BFF;

/// 10-bit endpoint precision for mode 11.
const PREC: u32 = 10;

/// Unquantise a 10-bit unsigned endpoint code to its 16-bit intermediate, as
/// [`decode_bc6h_mode11_unsigned`] does. Interior codes expand to `q*64 + 32`.
fn unquantize_unsigned(code: u32) -> u32 {
    if code == 0 {
        0
    } else if code == (1 << PREC) - 1 {
        0xFFFF
    } else {
        ((code << 16) + 0x8000) >> PREC
    }
}

/// Interpolate two 16-bit intermediates at a 1/64-unit weight and apply the
/// unsigned finish scale, reproducing the decoder's half-bit output exactly.
fn interp_finish(e0: u32, e1: u32, weight: u32) -> u16 {
    let q = ((64 - weight) * e0 + weight * e1 + 32) >> 6;
    (((q * 31) >> 6) & 0xFFFF) as u16
}

/// Clamp an arbitrary half-bit pattern to the unsigned-representable range and
/// return its pre-finish intermediate target `T = round(h * 64 / 31)`.
fn target_intermediate(half: u16) -> u32 {
    let h = u32::from(half);
    let hv = if h & 0x8000 != 0 {
        0 // negatives are unrepresentable in the unsigned profile
    } else {
        (h & 0x7FFF).min(MAX_HALF) // collapse inf/NaN/over-max to the ceiling
    };
    ((hv * 64) + 15) / 31
}

/// Quantise one 16-bit intermediate to a 10-bit code, returning
/// `(code, reconstructed)` where `reconstructed == unquantize_unsigned(code)`.
fn quantize_endpoint(value: i32) -> (u32, u32) {
    let v = value.clamp(0, 0xFFFF);
    // Nearest interior inverse of `q*64 + 32` is `round((v-32)/64) == v/64`;
    // the clamp snaps the saturated extremes (code 0 -> 0, code 1023 -> 0xFFFF).
    let code = (v / 64).clamp(0, (1 << PREC) - 1) as u32;
    (code, unquantize_unsigned(code))
}

/// Nearest 4-bit index for one texel channel-triple against the sixteen
/// finished colours; returns `(index, squared_error)` in half-bit space.
fn nearest_index(target: [u32; 3], e0: [u32; 3], e1: [u32; 3]) -> (u8, u64) {
    let mut best = 0u8;
    let mut best_err = u64::MAX;
    for (i, &w) in WEIGHT4.iter().enumerate() {
        let err: u64 = (0..3)
            .map(|c| {
                let got = i64::from(interp_finish(e0[c], e1[c], w));
                let want = i64::from(target_from_intermediate(target[c]));
                let d = got - want;
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

/// Recover the clamped target half from its intermediate (inverse of
/// [`target_intermediate`] up to rounding), for error comparison against the
/// finished output.
fn target_from_intermediate(t: u32) -> u16 {
    (((t * 31) >> 6) & 0xFFFF) as u16
}

/// Assign indices to every texel and return `(indices, total_error)`.
fn assign(targets: &[[u32; 3]; 16], e0: [u32; 3], e1: [u32; 3]) -> ([u8; 16], u64) {
    let mut idx = [0u8; 16];
    let mut err = 0u64;
    for (t, target) in targets.iter().enumerate() {
        let (i, e) = nearest_index(*target, e0, e1);
        idx[t] = i;
        err += e;
    }
    (idx, err)
}

/// Principal `RGB` axis of the intermediate targets via power iteration on the
/// 3x3 covariance. Falls back to a non-zero axis for a degenerate block.
fn principal_axis(points: &[[u32; 3]; 16]) -> [f64; 3] {
    let mut mean = [0.0f64; 3];
    for p in points {
        for c in 0..3 {
            mean[c] += f64::from(p[c]);
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }
    let mut cov = [[0.0f64; 3]; 3];
    for p in points {
        let d: [f64; 3] = core::array::from_fn(|c| f64::from(p[c]) - mean[c]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let mut axis = [1.0f64, 1.0, 1.0];
    for _ in 0..24 {
        let next: [f64; 3] = core::array::from_fn(|i| (0..3).map(|j| cov[i][j] * axis[j]).sum());
        let norm = next.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-9 {
            break;
        }
        axis = core::array::from_fn(|c| next[c] / norm);
    }
    axis
}

/// Least-squares re-fit of the two continuous 16-bit endpoints given fixed
/// indices, minimising `sum_t (a_t*e0 + b_t*e1 - target)^2` per channel with
/// `a_t = (64 - w_t)/64`, `b_t = w_t/64`. Returns `None` for a degenerate
/// (single-cluster) normal matrix so the caller keeps the prior endpoints.
fn refit(targets: &[[u32; 3]; 16], idx: &[u8; 16]) -> Option<([i32; 3], [i32; 3])> {
    let mut saa = 0.0f64;
    let mut sab = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sat = [0.0f64; 3];
    let mut sbt = [0.0f64; 3];
    for (t, target) in targets.iter().enumerate() {
        let w = f64::from(WEIGHT4[idx[t] as usize]);
        let a = (64.0 - w) / 64.0;
        let b = w / 64.0;
        saa += a * a;
        sab += a * b;
        sbb += b * b;
        for c in 0..3 {
            sat[c] += a * f64::from(target[c]);
            sbt[c] += b * f64::from(target[c]);
        }
    }
    let det = saa * sbb - sab * sab;
    if det.abs() < 1e-6 {
        return None;
    }
    let inv = 1.0 / det;
    let e0: [i32; 3] = core::array::from_fn(|c| {
        let v = (sbb * sat[c] - sab * sbt[c]) * inv;
        (v + 0.5).floor().clamp(0.0, 65535.0) as i32
    });
    let e1: [i32; 3] = core::array::from_fn(|c| {
        let v = (saa * sbt[c] - sab * sat[c]) * inv;
        (v + 0.5).floor().clamp(0.0, 65535.0) as i32
    });
    Some((e0, e1))
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
            if (value >> i) & 1 != 0 {
                self.bytes[self.pos / 8] |= 1 << (self.pos % 8);
            }
            self.pos += 1;
        }
    }
}

/// Encode one 4x4 tile of non-negative half-float `RGB` bit patterns as a
/// 16-byte BC6H **unsigned mode 11** block.
#[must_use]
pub fn encode_bc6h_mode11_unsigned(tile: &[[u16; 3]; 16]) -> [u8; 16] {
    let targets: [[u32; 3]; 16] =
        core::array::from_fn(|t| core::array::from_fn(|c| target_intermediate(tile[t][c])));

    // Seed endpoints from the extreme projections onto the principal axis.
    let axis = principal_axis(&targets);
    let mut lo_t = 0usize;
    let mut hi_t = 0usize;
    let mut lo_p = f64::INFINITY;
    let mut hi_p = f64::NEG_INFINITY;
    for (t, target) in targets.iter().enumerate() {
        let proj = (0..3).map(|c| f64::from(target[c]) * axis[c]).sum::<f64>();
        if proj < lo_p {
            lo_p = proj;
            lo_t = t;
        }
        if proj > hi_p {
            hi_p = proj;
            hi_t = t;
        }
    }
    let mut seed0: [i32; 3] = core::array::from_fn(|c| targets[lo_t][c] as i32);
    let mut seed1: [i32; 3] = core::array::from_fn(|c| targets[hi_t][c] as i32);

    let mut best_block: Option<(u64, [u8; 16])> = None;
    for _pass in 0..3 {
        let q0: [(u32, u32); 3] = core::array::from_fn(|c| quantize_endpoint(seed0[c]));
        let q1: [(u32, u32); 3] = core::array::from_fn(|c| quantize_endpoint(seed1[c]));
        let re0: [u32; 3] = core::array::from_fn(|c| q0[c].1);
        let re1: [u32; 3] = core::array::from_fn(|c| q1[c].1);
        let (mut idx, err) = assign(&targets, re0, re1);

        // Anchor rule: texel-0 index high bit must be 0; swap + invert if not.
        let (c0, c1) = if idx[0] & 0b1000 != 0 {
            for i in &mut idx {
                *i = 15 - *i;
            }
            ([q1[0].0, q1[1].0, q1[2].0], [q0[0].0, q0[1].0, q0[2].0])
        } else {
            ([q0[0].0, q0[1].0, q0[2].0], [q1[0].0, q1[1].0, q1[2].0])
        };

        let mut w = BitWriter::new();
        w.write(0b00011, 5); // mode-11 marker.
        for &v in &c0 {
            w.write(v, PREC);
        }
        for &v in &c1 {
            w.write(v, PREC);
        }
        w.write(u32::from(idx[0]), 3); // anchor: implicit high bit 0.
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }

        if best_block.is_none_or(|(be, _)| err < be) {
            best_block = Some((err, w.bytes));
        }

        match refit(&targets, &idx) {
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
    use crate::{decode_bc6h_mode11_unsigned, decode_bc6h_unsigned};

    /// IEEE round-to-nearest-even f32 -> half bit pattern, for building inputs.
    fn f32_to_half(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = bits & 0x7F_FFFF;
        if exp <= 0 {
            return sign; // flush tiny magnitudes to signed zero (inputs stay >= 2^-14)
        }
        if exp >= 0x1F {
            return sign | 0x7C00; // overflow to inf (clamped by the encoder)
        }
        let half_mant = (mant >> 13) as u16;
        sign | ((exp as u16) << 10) | half_mant
    }

    fn sse(a: &[[f32; 3]; 16], b: &[[f32; 3]; 16]) -> f64 {
        let mut s = 0.0;
        for (ta, tb) in a.iter().zip(b.iter()) {
            for c in 0..3 {
                let d = f64::from(ta[c]) - f64::from(tb[c]);
                s += d * d;
            }
        }
        s
    }

    #[test]
    fn flat_tile_round_trips_close() {
        let h = f32_to_half(12.5);
        let tile = [[h; 3]; 16];
        let out = decode_bc6h_mode11_unsigned(&encode_bc6h_mode11_unsigned(&tile));
        for texel in &out {
            for (c, &got) in texel.iter().enumerate() {
                assert!((got - 12.5).abs() <= 0.5, "channel {c} = {got}");
            }
        }
    }

    #[test]
    fn hdr_gradient_tracks_input() {
        let mut tile = [[0u16; 3]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = 1.0 + t as f32; // 1.0 .. 16.0
            *texel = [f32_to_half(v); 3];
        }
        let out = decode_bc6h_mode11_unsigned(&encode_bc6h_mode11_unsigned(&tile));
        for (t, texel) in out.iter().enumerate() {
            let want = 1.0 + t as f32;
            assert!((texel[0] - want).abs() <= 1.5, "texel {t} = {}", texel[0]);
        }
    }

    #[test]
    fn two_cluster_axis_is_reconstructed() {
        let lo = f32_to_half(2.0);
        let hi = f32_to_half(40.0);
        let mut tile = [[0u16; 3]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            *texel = if t % 2 == 0 { [lo; 3] } else { [hi; 3] };
        }
        let out = decode_bc6h_mode11_unsigned(&encode_bc6h_mode11_unsigned(&tile));
        for (t, texel) in out.iter().enumerate() {
            let want = if t % 2 == 0 { 2.0 } else { 40.0 };
            assert!((texel[0] - want).abs() <= 2.0, "texel {t} = {}", texel[0]);
        }
    }

    #[test]
    fn gradient_beats_flat_encode() {
        let mut tile = [[0u16; 3]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = 1.0 + (t as f32) * 2.0;
            *texel = [f32_to_half(v); 3];
        }
        let want: [[f32; 3]; 16] =
            core::array::from_fn(|t| core::array::from_fn(|_| 1.0 + (t as f32) * 2.0));
        let out = decode_bc6h_mode11_unsigned(&encode_bc6h_mode11_unsigned(&tile));
        let flat = [[16.0f32; 3]; 16];
        assert!(sse(&out, &want) < sse(&flat, &want));
    }

    #[test]
    fn negative_input_clamps_to_zero() {
        // Unsigned profile cannot store a sign: negatives become non-negative.
        let neg = f32_to_half(-5.0);
        let tile = [[neg; 3]; 16];
        let out = decode_bc6h_mode11_unsigned(&encode_bc6h_mode11_unsigned(&tile));
        for texel in &out {
            for (c, &got) in texel.iter().enumerate() {
                assert!(got >= 0.0, "channel {c} = {got}");
                assert!(got <= 0.5, "channel {c} = {got}");
            }
        }
    }

    #[test]
    fn anchor_index_high_bit_is_zero() {
        let mut tile = [[0u16; 3]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = 0.5 + (t as f32);
            *texel = [f32_to_half(v), f32_to_half(v * 0.5), f32_to_half(16.0 - v)];
        }
        let block = encode_bc6h_mode11_unsigned(&tile);
        // Bits: 5 (mode) + 60 (6 x 10-bit endpoints) = 65; anchor is next 3.
        let mut acc = 0u32;
        for i in 0..3u32 {
            let bitpos = 65 + i as usize;
            acc |= u32::from((block[bitpos / 8] >> (bitpos % 8)) & 1) << i;
        }
        assert!(acc < 8, "anchor index {acc} must fit 3 bits");
    }

    #[test]
    fn block_is_tagged_mode_11() {
        let tile = [[f32_to_half(7.0); 3]; 16];
        let block = encode_bc6h_mode11_unsigned(&tile);
        assert_eq!(block[0] & 0b1_1111, 0b00011);
        // Dispatcher must accept it as unsigned mode 11.
        assert!(decode_bc6h_unsigned(&block).is_ok());
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut tile = [[0u16; 3]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = 1.0 + (t as f32) * 1.3;
            *texel = [f32_to_half(v), f32_to_half(v + 2.0), f32_to_half(v * 0.7)];
        }
        assert_eq!(
            encode_bc6h_mode11_unsigned(&tile),
            encode_bc6h_mode11_unsigned(&tile)
        );
    }
}
