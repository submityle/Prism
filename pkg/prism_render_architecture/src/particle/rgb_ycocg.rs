//! `RGB` ↔ `YCoCg` / `YCoCg-R` reversible colour-space transforms for the
//! particle temporal-antialiasing (`TAA`) and checkerboard chroma path
//! (design §21).
//!
//! `YCoCg` splits a colour into one *luma* channel (`Y`) and two *chroma*
//! channels (`Co`, orange–blue; `Cg`, green–magenta). Production temporal
//! resolvers (`Unreal`'s `TAA`, `Frostbite`'s checkerboard rebuild) work in
//! `YCoCg` because the neighbourhood colour-clamp that kills ghosting is far
//! tighter in luma/chroma than in raw `RGB`, and because chroma can be
//! subsampled 2×2 with little visible loss. This module owns the
//! `CPU`-verifiable maths of that transform and packs a `YCoCg` triple into the
//! `std430` block a `GPU` resolve pass binds.
//!
//! # Two transforms
//!
//! 1. **Lossy `YCoCg`** — [`rgb_to_ycocg`] / [`ycocg_to_rgb`] use the classic
//!    quarter/half weights. The round trip is algebraically exact but, in
//!    finite `f32`, only reproduces the input to a small tolerance; it is the
//!    variant a `TAA` history clamp runs in.
//! 2. **Lossless `YCoCg-R`** — [`rgb_to_ycocg_r`] / [`ycocg_r_to_rgb`] use the
//!    integer *lifting* scheme (the `YCoCg-R` of `H.264` lossless mode). It is
//!    **bit-exact** reversible on integer channels, so it is the variant a
//!    lossless framebuffer compressor uses.
//!
//! # Relationship to sibling modules
//!
//! This module is *not* [`super::checkerboard_resolve`]: that module weaves a
//! half-density sample pattern back into a full image in *space*, whereas this
//! module only changes the *colour basis* a resolver operates in and never
//! looks at pixel parity. It is also *not* [`super::tonemap`]: `tonemap`
//! compresses open-ended `HDR` radiance into display `sRGB`, applying a
//! `gamma`-style non-linear curve, whereas the `YCoCg` transforms here are
//! purely linear (plus an integer lifting variant) and reuse neither module's
//! types.
//!
//! # Determinism
//!
//! Every routine touches only `+ - * /`, integer shifts, [`f32::clamp`], and
//! the `smoothstep` polynomial `t*t*(3-2t)`. No transcendental function
//! (`sin`/`cos`/`exp`/`ln`/`powf`) and no `f32::round`/`f32::ceil` is ever
//! called, matching the determinism contract of [`super::tonemap`] so a future
//! `GPU` evaluation reproduces the `CPU` result bit for bit.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Absolute tolerance for the internal `f32` guards that avoid a direct `==`.
const CMP_EPS: f32 = 1e-6;

/// Scales a luma gap into the `smoothstep` edge-strength input; the larger the
/// value, the sooner a luma difference counts as a hard edge.
const EDGE_SCALE: f32 = 8.0;

/// Byte size of one `YCoCg` sample in the `std430` layout: a single `vec4`
/// slot (three populated floats plus one zero-padding float).
pub const YCOCG_STD430_SIZE: usize = VEC4_STRIDE;

/// A linear `RGB` colour triple.
///
/// Values are conventionally in `[0, 1]` for the lossy transform but the maths
/// is defined for any `f32`.
#[derive(Clone, Copy, Debug)]
pub struct Rgb {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

/// A lossy `YCoCg` colour triple (luma plus orange–blue and green–magenta
/// chroma).
#[derive(Clone, Copy, Debug)]
pub struct YCoCg {
    /// Luma channel.
    pub y: f32,
    /// Orange–blue chroma channel.
    pub co: f32,
    /// Green–magenta chroma channel.
    pub cg: f32,
}

/// A lossless integer `YCoCg-R` triple produced by the lifting transform.
///
/// The channels are stored as `i32` because `Co`/`Cg` can exceed the `u8`
/// range of the source `RGB` and can be negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YCoCgR {
    /// Luma channel.
    pub y: i32,
    /// Orange–blue chroma channel.
    pub co: i32,
    /// Green–magenta chroma channel.
    pub cg: i32,
}

impl Rgb {
    /// Builds an `RGB` triple.
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl YCoCg {
    /// Builds a `YCoCg` triple.
    #[must_use]
    pub fn new(y: f32, co: f32, cg: f32) -> Self {
        Self { y, co, cg }
    }

    /// Packs this sample into its `std430` `vec4` slot ([`YCOCG_STD430_SIZE`]
    /// bytes): `Y`, `Co`, `Cg` as little-endian `f32`, then a zero-padding
    /// float so the next sample starts on a `vec4` boundary.
    #[must_use]
    pub fn to_std430(&self) -> [u8; YCOCG_STD430_SIZE] {
        let mut bytes = [0u8; YCOCG_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.y.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.co.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.cg.to_le_bytes());
        // bytes[12..16] stay zero: the `vec4` padding slot.
        bytes
    }

    /// Reads a `YCoCg` sample back out of a `std430` `vec4` slot, ignoring the
    /// trailing padding float.
    #[must_use]
    pub fn from_std430(bytes: &[u8; YCOCG_STD430_SIZE]) -> Self {
        Self {
            y: read_le_f32(bytes, 0),
            co: read_le_f32(bytes, 4),
            cg: read_le_f32(bytes, 8),
        }
    }

    /// Byte size of a `count`-element `GPU` storage buffer of `YCoCg` samples,
    /// reusing the shared [`storage_bytes`] helper (non-empty and saturating).
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(YCOCG_STD430_SIZE, count)
    }
}

/// Reads a little-endian `f32` out of `bytes` at `offset`.
fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[offset..offset + 4]);
    f32::from_le_bytes(buf)
}

/// Lossy forward transform: `Y = R/4 + G/2 + B/4`, `Co = R/2 − B/2`,
/// `Cg = −R/4 + G/2 − B/4`.
#[must_use]
pub fn rgb_to_ycocg(c: Rgb) -> YCoCg {
    YCoCg {
        y: c.r * 0.25 + c.g * 0.5 + c.b * 0.25,
        co: c.r * 0.5 - c.b * 0.5,
        cg: -c.r * 0.25 + c.g * 0.5 - c.b * 0.25,
    }
}

/// Lossy inverse transform: `R = Y + Co − Cg`, `G = Y + Cg`, `B = Y − Co − Cg`.
#[must_use]
pub fn ycocg_to_rgb(c: YCoCg) -> Rgb {
    Rgb {
        r: c.y + c.co - c.cg,
        g: c.y + c.cg,
        b: c.y - c.co - c.cg,
    }
}

/// Lossless integer forward transform (`YCoCg-R` lifting):
/// `Co = R − B`, `t = B + (Co >> 1)`, `Cg = G − t`, `Y = t + (Cg >> 1)`.
///
/// The `>> 1` are arithmetic shifts on `i32`, i.e. floor-division by two, which
/// is exactly what makes the scheme bit-exact reversible for negative chroma.
#[must_use]
pub fn rgb_to_ycocg_r(rgb: [u8; 3]) -> YCoCgR {
    let r = i32::from(rgb[0]);
    let g = i32::from(rgb[1]);
    let b = i32::from(rgb[2]);
    let co = r - b;
    let t = b + (co >> 1);
    let cg = g - t;
    let y = t + (cg >> 1);
    YCoCgR { y, co, cg }
}

/// Lossless integer inverse transform (`YCoCg-R` lifting):
/// `t = Y − (Cg >> 1)`, `G = Cg + t`, `B = t − (Co >> 1)`, `R = B + Co`.
///
/// Recovers the original `[u8; 3]` bit-for-bit for any triple produced by
/// [`rgb_to_ycocg_r`]. The final channels are clamped into `[0, 255]` purely
/// as a defensive guard against out-of-contract inputs.
#[must_use]
pub fn ycocg_r_to_rgb(c: YCoCgR) -> [u8; 3] {
    let t = c.y - (c.cg >> 1);
    let g = c.cg + t;
    let b = t - (c.co >> 1);
    let r = b + c.co;
    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

/// Clamps an `i32` into the `u8` range without a lossy cast.
fn clamp_u8(v: i32) -> u8 {
    let clamped = v.clamp(0, 255);
    u8::try_from(clamped).unwrap_or(0)
}

/// A 2×2 chroma-subsampled block: four full-resolution luma taps sharing one
/// averaged `Co`/`Cg` chroma pair.
#[derive(Clone, Copy, Debug)]
pub struct Chroma2x2 {
    /// Per-subpixel luma taps, in row-major `[top-left, top-right,
    /// bottom-left, bottom-right]` order.
    pub luma: [f32; 4],
    /// Shared orange–blue chroma for the whole block.
    pub co: f32,
    /// Shared green–magenta chroma for the whole block.
    pub cg: f32,
}

/// Subsamples a 2×2 block of `YCoCg` pixels: keeps all four luma taps but
/// replaces the four chroma pairs with their average, halving chroma bandwidth.
#[must_use]
pub fn pack_2x2_chroma(block: [YCoCg; 4]) -> Chroma2x2 {
    let luma = [block[0].y, block[1].y, block[2].y, block[3].y];
    let co = (block[0].co + block[1].co + block[2].co + block[3].co) * 0.25;
    let cg = (block[0].cg + block[1].cg + block[2].cg + block[3].cg) * 0.25;
    Chroma2x2 { luma, co, cg }
}

/// Rebuilds four full-resolution `YCoCg` pixels from a subsampled block by
/// re-attaching the shared chroma to each preserved luma tap.
#[must_use]
pub fn unpack_2x2_chroma(block: &Chroma2x2) -> [YCoCg; 4] {
    let mut out = [YCoCg::new(0.0, 0.0, 0.0); 4];
    for (dst, &y) in out.iter_mut().zip(block.luma.iter()) {
        dst.y = y;
        dst.co = block.co;
        dst.cg = block.cg;
    }
    out
}

/// A luma-tagged chroma sample used by the edge-oriented reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct ChromaSample {
    /// Full-resolution luma at this tap.
    pub luma: f32,
    /// Orange–blue chroma at this tap.
    pub co: f32,
    /// Green–magenta chroma at this tap.
    pub cg: f32,
}

impl ChromaSample {
    /// Builds a chroma sample.
    #[must_use]
    pub fn new(luma: f32, co: f32, cg: f32) -> Self {
        Self { luma, co, cg }
    }
}

/// `smoothstep` on `[0, 1]`: clamps `t` then evaluates `t*t*(3 − 2t)`.
fn smoothstep01(t: f32) -> f32 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// Reconstructs the chroma at a target pixel from two neighbouring chroma
/// samples `a` and `b`, using the target's own luma to steer the blend.
///
/// When the two neighbours share a luma the result is a plain bilinear average
/// (each weighted `0.5`); as their luma gap widens (a hard edge), a
/// `smoothstep` factor pulls the blend toward inverse-luma-distance weighting
/// so chroma is interpolated *along* the edge rather than bled *across* it.
/// Returns the reconstructed `(Co, Cg)` pair.
#[must_use]
pub fn reconstruct_chroma(target_luma: f32, a: ChromaSample, b: ChromaSample) -> (f32, f32) {
    let luma_gap = (a.luma - b.luma).abs();
    let edge = smoothstep01(luma_gap * EDGE_SCALE);

    let da = (target_luma - a.luma).abs();
    let db = (target_luma - b.luma).abs();
    let denom = da + db;
    // Inverse-distance weight of `b`: the farther the target is from `a`
    // (large `da`), the more `b` contributes.
    let idw_b = if denom < CMP_EPS { 0.5 } else { da / denom };

    let wb = 0.5 * (1.0 - edge) + idw_b * edge;
    let wa = 1.0 - wb;

    (wa * a.co + wb * b.co, wa * a.cg + wb * b.cg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn roundtrip_close(c: Rgb) -> bool {
        let back = ycocg_to_rgb(rgb_to_ycocg(c));
        (back.r - c.r).abs() < 1e-5 && (back.g - c.g).abs() < 1e-5 && (back.b - c.b).abs() < 1e-5
    }

    #[test]
    fn lossy_roundtrip_black() {
        assert!(roundtrip_close(Rgb::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn lossy_roundtrip_white() {
        assert!(roundtrip_close(Rgb::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn lossy_roundtrip_assorted_colours() {
        let colours = [
            Rgb::new(0.2, 0.4, 0.6),
            Rgb::new(0.9, 0.1, 0.3),
            Rgb::new(0.05, 0.95, 0.5),
            Rgb::new(0.7, 0.7, 0.2),
        ];
        for &c in &colours {
            assert!(roundtrip_close(c));
        }
    }

    #[test]
    fn grey_has_zero_chroma() {
        for &v in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let yc = rgb_to_ycocg(Rgb::new(v, v, v));
            assert!(approx(yc.co, 0.0));
            assert!(approx(yc.cg, 0.0));
        }
    }

    #[test]
    fn grey_luma_equals_value() {
        for &v in &[0.1_f32, 0.5, 0.9] {
            let yc = rgb_to_ycocg(Rgb::new(v, v, v));
            assert!(approx(yc.y, v));
        }
    }

    #[test]
    fn red_primary_values() {
        let yc = rgb_to_ycocg(Rgb::new(1.0, 0.0, 0.0));
        assert!(approx(yc.y, 0.25));
        assert!(approx(yc.co, 0.5));
        assert!(approx(yc.cg, -0.25));
    }

    #[test]
    fn green_primary_values() {
        let yc = rgb_to_ycocg(Rgb::new(0.0, 1.0, 0.0));
        assert!(approx(yc.y, 0.5));
        assert!(approx(yc.co, 0.0));
        assert!(approx(yc.cg, 0.5));
    }

    #[test]
    fn blue_primary_values() {
        let yc = rgb_to_ycocg(Rgb::new(0.0, 0.0, 1.0));
        assert!(approx(yc.y, 0.25));
        assert!(approx(yc.co, -0.5));
        assert!(approx(yc.cg, -0.25));
    }

    #[test]
    fn inverse_reproduces_known_triple() {
        let c = ycocg_to_rgb(YCoCg::new(0.25, 0.5, -0.25));
        assert!(approx(c.r, 1.0));
        assert!(approx(c.g, 0.0));
        assert!(approx(c.b, 0.0));
    }

    #[test]
    fn ycocg_r_roundtrip_black_and_white() {
        for &rgb in &[[0u8, 0, 0], [255, 255, 255]] {
            assert_eq!(ycocg_r_to_rgb(rgb_to_ycocg_r(rgb)), rgb);
        }
    }

    #[test]
    fn ycocg_r_roundtrip_primaries() {
        for &rgb in &[[255u8, 0, 0], [0, 255, 0], [0, 0, 255]] {
            assert_eq!(ycocg_r_to_rgb(rgb_to_ycocg_r(rgb)), rgb);
        }
    }

    #[test]
    fn ycocg_r_roundtrip_bit_exact_sweep() {
        for r in (0u8..=255).step_by(17) {
            for g in (0u8..=255).step_by(29) {
                for b in (0u8..=255).step_by(43) {
                    let rgb = [r, g, b];
                    assert_eq!(ycocg_r_to_rgb(rgb_to_ycocg_r(rgb)), rgb);
                }
            }
        }
    }

    #[test]
    fn ycocg_r_roundtrip_edge_channels() {
        for &rgb in &[[1u8, 254, 3], [254, 1, 252], [128, 127, 129], [200, 50, 75]] {
            assert_eq!(ycocg_r_to_rgb(rgb_to_ycocg_r(rgb)), rgb);
        }
    }

    #[test]
    fn ycocg_r_grey_has_zero_chroma() {
        let yc = rgb_to_ycocg_r([120, 120, 120]);
        assert_eq!(yc.co, 0);
        assert_eq!(yc.cg, 0);
        assert_eq!(yc.y, 120);
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(YCOCG_STD430_SIZE, 16);
        assert_eq!(YCOCG_STD430_SIZE, VEC4_STRIDE);
        let bytes = YCoCg::new(0.5, 0.1, -0.2).to_std430();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn std430_roundtrips_the_triple() {
        let yc = YCoCg::new(0.42, -0.3, 0.17);
        let bytes = yc.to_std430();
        let back = YCoCg::from_std430(&bytes);
        assert!(approx(back.y, yc.y));
        assert!(approx(back.co, yc.co));
        assert!(approx(back.cg, yc.cg));
    }

    #[test]
    fn std430_padding_slot_is_zero() {
        let bytes = YCoCg::new(0.9, 0.4, -0.5).to_std430();
        assert!(approx(read_le_f32(&bytes, 12), 0.0));
    }

    #[test]
    fn std430_channel_layout_is_little_endian() {
        let yc = YCoCg::new(1.0, 2.0, 3.0);
        let bytes = yc.to_std430();
        assert!(approx(read_le_f32(&bytes, 0), 1.0));
        assert!(approx(read_le_f32(&bytes, 4), 2.0));
        assert!(approx(read_le_f32(&bytes, 8), 3.0));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(YCoCg::gpu_storage_bytes(0), 16);
        assert_eq!(YCoCg::gpu_storage_bytes(1), 16);
        assert_eq!(YCoCg::gpu_storage_bytes(10), 160);
    }

    #[test]
    fn pack_2x2_averages_chroma_and_keeps_luma() {
        let block = [
            YCoCg::new(0.1, 0.4, -0.2),
            YCoCg::new(0.2, 0.0, 0.2),
            YCoCg::new(0.3, -0.4, 0.6),
            YCoCg::new(0.4, 0.8, -0.6),
        ];
        let packed = pack_2x2_chroma(block);
        assert!(approx(packed.luma[0], 0.1));
        assert!(approx(packed.luma[3], 0.4));
        assert!(approx(packed.co, (0.4 + 0.0 - 0.4 + 0.8) * 0.25));
        assert!(approx(packed.cg, (-0.2 + 0.2 + 0.6 - 0.6) * 0.25));
    }

    #[test]
    fn unpack_reattaches_shared_chroma() {
        let block = [
            YCoCg::new(0.1, 0.4, -0.2),
            YCoCg::new(0.2, 0.0, 0.2),
            YCoCg::new(0.3, -0.4, 0.6),
            YCoCg::new(0.4, 0.8, -0.6),
        ];
        let packed = pack_2x2_chroma(block);
        let out = unpack_2x2_chroma(&packed);
        for (dst, src) in out.iter().zip(block.iter()) {
            assert!(approx(dst.y, src.y));
            assert!(approx(dst.co, packed.co));
            assert!(approx(dst.cg, packed.cg));
        }
    }

    #[test]
    fn reconstruct_equal_luma_is_plain_average() {
        let a = ChromaSample::new(0.5, 0.2, -0.4);
        let b = ChromaSample::new(0.5, 0.6, 0.4);
        let (co, cg) = reconstruct_chroma(0.5, a, b);
        assert!(approx(co, 0.4));
        assert!(approx(cg, 0.0));
    }

    #[test]
    fn reconstruct_edge_biases_toward_nearer_luma() {
        // A hard luma edge: target luma matches `a`, so `a`'s chroma dominates.
        let a = ChromaSample::new(0.0, 1.0, 0.0);
        let b = ChromaSample::new(1.0, -1.0, 0.0);
        let (co, _cg) = reconstruct_chroma(0.0, a, b);
        assert!(co > 0.5);
    }

    #[test]
    fn reconstruct_edge_symmetry() {
        let a = ChromaSample::new(0.0, 1.0, 0.0);
        let b = ChromaSample::new(1.0, -1.0, 0.0);
        let (co_near_a, _) = reconstruct_chroma(0.0, a, b);
        let (co_near_b, _) = reconstruct_chroma(1.0, a, b);
        assert!(approx(co_near_a, -co_near_b));
    }

    #[test]
    fn reconstruct_result_within_sample_range() {
        let a = ChromaSample::new(0.2, -0.3, 0.5);
        let b = ChromaSample::new(0.8, 0.7, -0.1);
        let (co, cg) = reconstruct_chroma(0.5, a, b);
        assert!((-0.3 - CMP_EPS..=0.7 + CMP_EPS).contains(&co));
        assert!((-0.1 - CMP_EPS..=0.5 + CMP_EPS).contains(&cg));
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!(approx(smoothstep01(-1.0), 0.0));
        assert!(approx(smoothstep01(0.0), 0.0));
        assert!(approx(smoothstep01(0.5), 0.5));
        assert!(approx(smoothstep01(1.0), 1.0));
        assert!(approx(smoothstep01(2.0), 1.0));
    }

    #[test]
    fn clamp_u8_saturates_out_of_range() {
        assert_eq!(clamp_u8(-5), 0);
        assert_eq!(clamp_u8(300), 255);
        assert_eq!(clamp_u8(128), 128);
    }
}
