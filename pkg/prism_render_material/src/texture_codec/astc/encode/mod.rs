//! ASTC LDR encoder (CPU, pure integer/`f64` arithmetic -- no AI/ML path).
//!
//! Built up in GPU-validatable milestones mirroring the decoder. The first
//! landed stage targets the cleanest provable configuration:
//!
//! * **block mode 83** (`0b1010011`): a 4x4 weight grid, single plane, weight
//!   range QUANT_8 (3 bit, bit-only -- no trit/quint);
//! * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
//! * **QUANT_256 colour**: with `color_bits = 111 - 48 = 63` and six CEM-8
//!   integers the colour quant level is QUANT_256, i.e. 8-bit identity, so
//!   endpoint bytes are written straight into the block.
//!
//! Every endpoint and weight choice uses the exact decode-side arithmetic, and
//! the output is round-tripped through [`super::decode_astc_4x4_ldr`] in tests
//! and parity-checked against the platform ASTC hardware decoder.
//!
//! Later milestones add weight-mode search, trit/quint colour quantisation,
//! CEM 12 (RGBA), larger footprints, dual-plane, multi-partition, and HDR.

mod bits;
mod color_quant;
mod endpoint_fit;
mod weight_fit;

/// Encode sixteen `RGBA8` texels (row-major, `texel = y * 4 + x`) into a single
/// 4x4 ASTC LDR block using block mode 83, a single partition, CEM 8 (RGB
/// direct) with QUANT_256 colour, and QUANT_8 (3-bit) weights.
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored. The result is a valid LDR block
/// that [`super::decode_astc_4x4_ldr`] decodes back to the fitted colours.
#[must_use]
pub fn encode_astc_single_partition_4x4_ldr(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 83;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 3; // QUANT_8

    let (e0, e1) = endpoint_fit::fit_rgb_endpoints(texels);
    let raw = weight_fit::quantize_weights_bits(texels, e0, e1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11 (partition-count field at bits
    // 11,12 and the CEM field below are left/overwritten explicitly).
    w.write_bits(0, 11, BLOCK_MODE);
    // Single partition => partition-count field (bits 11,12) stays 0.
    // CEM field: 4 bits starting at block bit 13 (bits 13,14,15 in byte 1 and
    // bit 16 = byte 2 bit 0). CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six 8-bit colour values at block bit 17, LSB-first, in the decoder's
    // read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let vals = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]];
    for (i, v) in vals.iter().enumerate() {
        w.write_bits(17 + i as u32 * 8, 8, u32::from(*v));
    }
    // Sixteen 3-bit weights packed bit-reversed from the top of the block.
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

/// Encode sixteen `RGBA8` texels (row-major, `texel = y * 4 + x`) into a single
/// 4x4 ASTC LDR block using block mode 578, a single partition, CEM 8 (RGB
/// direct) with **QUANT_192** colour (trit BISE), and QUANT_16 (4-bit, bit-only)
/// weights.
///
/// Compared with [`encode_astc_single_partition_4x4_ldr`] (mode 83, 3-bit
/// weights, 8-bit identity colour), this doubles the weight resolution to
/// sixteen levels at the cost of slightly coarser colour endpoints (QUANT_192
/// reconstructs any 8-bit target within 2 LSB). On smooth gradients the finer
/// weights win decisively; the endpoint rounding is negligible.
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_4x4_ldr_q192(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 578;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 4; // QUANT_16, bit-only
    const COLOR_LEVEL: usize = 15; // QUANT_192

    // Fit the principal-axis endpoints, then quantize each channel into the
    // QUANT_192 packed representation.
    let (e0_raw, e1_raw) = endpoint_fit::fit_rgb_endpoints(texels);
    let mut p0: [u8; 3] =
        core::array::from_fn(|c| color_quant::quantize_color_channel(COLOR_LEVEL, e0_raw[c]));
    let mut p1: [u8; 3] =
        core::array::from_fn(|c| color_quant::quantize_color_channel(COLOR_LEVEL, e1_raw[c]));

    // Reconstruct the decoded endpoints the hardware interpolates between.
    let unq = |p: [u8; 3]| -> [u8; 3] {
        core::array::from_fn(|c| super::color_unquant::unquant_color(COLOR_LEVEL, p[c]))
    };
    let mut d0 = unq(p0);
    let mut d1 = unq(p1);

    // CEM 8 applies blue-contraction + endpoint swap when `hadd(e0) > hadd(e1)`
    // on the *unquantized* colours. Pre-swap the packed endpoints so the
    // decoder takes the plain path and interpolates d0..d1 directly; weights
    // are fitted after the swap so the texel mapping stays correct.
    let hadd = |c: [u8; 3]| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    if hadd(d0) > hadd(d1) {
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut d0, &mut d1);
    }

    // Fit 4-bit weights against the *decoded* endpoints (not the raw fit) so
    // the chosen levels reconstruct the intended colours.
    let raw = weight_fit::quantize_weights_bits(texels, d0, d1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    w.write_bits(0, 11, BLOCK_MODE);
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    let mut block = w.into_block();

    // Six QUANT_192 colour integers as a trit BISE at block bit 17, in the
    // decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b]. Each packed
    // value is `low(6 bits) | (trit << 6)`, exactly the table index.
    let packed = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]];
    super::trit_quint::encode_trit_sequence(&mut block, 17, 6, &packed);
    block
}

#[cfg(test)]
mod tests {
    use super::super::decode_astc_4x4_ldr;
    use super::encode_astc_single_partition_4x4_ldr;
    use super::encode_astc_single_partition_4x4_ldr_q192;

    /// Max per-channel RGB error over the sixteen texels after a round trip.
    fn max_rgb_err(src: &[[u8; 4]; 16], dec: &[[u8; 4]; 16]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn constant_block_round_trips_exactly() {
        let src = [[73u8, 150, 211, 255]; 16];
        let blk = encode_astc_single_partition_4x4_ldr(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode constant block");
        // A constant block has coincident endpoints; QUANT_256 is identity, so
        // the colour is reproduced exactly (alpha is forced to 255).
        assert_eq!(max_rgb_err(&src, &dec), 0, "constant block RGB mismatch");
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn gray_ramp_round_trips_within_tolerance() {
        // A smooth gray ramp lies on a single colour axis, so the 3-bit weights
        // track it closely.
        let src: [[u8; 4]; 16] =
            core::array::from_fn(|t| [(t * 17) as u8, (t * 17) as u8, (t * 17) as u8, 255]);
        let blk = encode_astc_single_partition_4x4_ldr(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode gray ramp");
        // Three bits give eight weight levels, so a 16-step ramp cannot be
        // reproduced exactly; the worst texel lands ~17 LSB from its target.
        assert!(max_rgb_err(&src, &dec) <= 18, "gray ramp error too large");
    }

    #[test]
    fn rgb_gradient_round_trips_within_tolerance() {
        // A colour gradient along the R/G/B diagonal: endpoints capture the
        // extremes and the weights interpolate between them.
        let src: [[u8; 4]; 16] = core::array::from_fn(|t| {
            let f = t as u8 * 16;
            [f, 255 - f, (f / 2) + 40, 255]
        });
        let blk = encode_astc_single_partition_4x4_ldr(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode rgb gradient");
        // Three-bit weights on a single axis: a modest tolerance covers the
        // quantisation of texels that stray off the principal axis.
        assert!(
            max_rgb_err(&src, &dec) <= 24,
            "rgb gradient error too large"
        );
    }

    #[test]
    fn q192_constant_block_round_trips_within_two_lsb() {
        let src = [[73u8, 150, 211, 255]; 16];
        let blk = encode_astc_single_partition_4x4_ldr_q192(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode QUANT_192 constant block");
        // Constant block => coincident endpoints; QUANT_192 reconstructs each
        // channel within 2 LSB.
        assert!(
            max_rgb_err(&src, &dec) <= 2,
            "QUANT_192 constant block error too large"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn q192_gray_ramp_beats_mode83() {
        // Sixteen-step gray ramp: the mode-578 encoder has 16 weight levels, so
        // it reproduces the ramp far more accurately than the 8-level mode 83.
        let src: [[u8; 4]; 16] =
            core::array::from_fn(|t| [(t * 17) as u8, (t * 17) as u8, (t * 17) as u8, 255]);
        let blk83 = encode_astc_single_partition_4x4_ldr(&src);
        let blk578 = encode_astc_single_partition_4x4_ldr_q192(&src);
        let dec83 = decode_astc_4x4_ldr(&blk83).expect("decode mode 83 gray ramp");
        let dec578 = decode_astc_4x4_ldr(&blk578).expect("decode mode 578 gray ramp");
        let e83 = max_rgb_err(&src, &dec83);
        let e578 = max_rgb_err(&src, &dec578);
        assert!(
            e578 < e83,
            "mode 578 ({e578}) should beat mode 83 ({e83}) on a gray ramp"
        );
        // Sixteen levels over a sixteen-step ramp: near-exact.
        assert!(e578 <= 4, "mode 578 gray ramp error too large: {e578}");
    }

    #[test]
    fn q192_rgb_gradient_round_trips_within_tolerance() {
        let src: [[u8; 4]; 16] = core::array::from_fn(|t| {
            let f = t as u8 * 16;
            [f, 255 - f, (f / 2) + 40, 255]
        });
        let blk = encode_astc_single_partition_4x4_ldr_q192(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode QUANT_192 rgb gradient");
        // 16 weight levels on a single axis + QUANT_192 endpoints: tight.
        assert!(
            max_rgb_err(&src, &dec) <= 8,
            "QUANT_192 rgb gradient error too large"
        );
    }
}
