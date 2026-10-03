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

/// Encode sixteen `RGBA8` texels into a single 4x4 ASTC LDR block, choosing the
/// block configuration that minimises reconstruction error for *this* block.
///
/// Both landed single-partition encoders are tried --
/// [`encode_astc_single_partition_4x4_ldr`] (mode 83: 8-bit identity colour,
/// 3-bit weights) and [`encode_astc_single_partition_4x4_ldr_q192`] (mode 578:
/// QUANT_192 colour, 4-bit weights) -- each candidate is decoded with the exact
/// hardware decode path ([`super::decode_astc_4x4_ldr`]), and the candidate with
/// the smallest sum-of-squared RGB error against the source is returned.
///
/// This is an honest per-block quality search: smooth gradients favour the
/// finer 4-bit weights of mode 578, while blocks that need exact 8-bit endpoints
/// (few distinct colours, hard edges) favour mode 83's identity colour. The
/// returned block always decodes to error no worse than either individual mode,
/// so callers get the better of the two with no quality regression.
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_4x4_ldr_quality(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let candidates = [
        encode_astc_single_partition_4x4_ldr(texels),
        encode_astc_single_partition_4x4_ldr_q192(texels),
    ];

    // Sum of squared RGB error of a decoded block against the source texels.
    // Alpha is forced to 255 by CEM 8, so it is excluded from the metric.
    let ssd = |dec: &[[u8; 4]; 16]| -> u64 {
        let mut e = 0u64;
        for (s, d) in texels.iter().zip(dec.iter()) {
            for c in 0..3 {
                let v = i64::from(s[c]) - i64::from(d[c]);
                e += (v * v) as u64;
            }
        }
        e
    };

    let mut best = candidates[0];
    let mut best_err = u64::MAX;
    for blk in candidates {
        // Every candidate is produced by a proven encoder, so the decode always
        // succeeds; skip any that somehow fail rather than panicking.
        if let Ok(dec) = super::decode_astc_4x4_ldr(&blk) {
            let err = ssd(&dec);
            if err < best_err {
                best_err = err;
                best = blk;
            }
        }
    }
    best
}

/// Encode sixteen `RGBA8` texels (row-major, `texel = y * 4 + x`) into a single
/// 4x4 ASTC LDR block that carries a **real per-endpoint alpha** via CEM 12
/// (RGBA direct), unlike the CEM-8 encoders above which force alpha to 255.
///
/// Configuration:
/// * **block mode 66**: 4x4 weight grid, single plane, weight range QUANT_4
///   (2-bit, bit-only) -- 32 weight bits;
/// * **single partition**, **CEM 12** (RGBA direct, eight colour integers);
/// * **QUANT_256 colour**: `color_bits = 111 - 32 = 79` with eight CEM-12
///   integers gives colour quant level QUANT_256 (8-bit identity), so the eight
///   endpoint bytes -- including both alphas -- are written straight into the
///   block and decode back bit-for-bit.
///
/// The two bits of weight resolution give only four interpolation levels, so
/// smooth gradients reconstruct coarsely; the *endpoints* (and therefore the
/// alpha extremes) are exact. This is the first honest alpha-carrying ASTC
/// milestone -- finer-weight CEM-12 variants land later.
#[must_use]
pub fn encode_astc_single_partition_4x4_ldr_rgba(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 66;
    const CEM_RGBA_DIRECT: u32 = 12;
    const WEIGHT_BITS: u32 = 2; // QUANT_4, bit-only

    // Fit RGBA endpoints ordered so hadd_rgb(e0) <= hadd_rgb(e1); the CEM-12
    // decoder then takes its plain (no blue-contraction swap) path and, with
    // QUANT_256 identity colour, reconstructs these bytes exactly. Weights are
    // fitted against the same endpoints over all four channels.
    let (e0, e1) = endpoint_fit::fit_rgba_endpoints(texels);
    let raw = weight_fit::quantize_weights_bits_rgba(texels, e0, e1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field (4 bits at block bit 13): value 12 sets block bits 15 and 16.
    w.write_bits(13, 4, CEM_RGBA_DIRECT);
    // Eight 8-bit colour values at block bit 17, LSB-first, in the decoder's
    // read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b, e0.a, e1.a].
    let vals = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]];
    for (i, v) in vals.iter().enumerate() {
        w.write_bits(17 + i as u32 * 8, 8, u32::from(*v));
    }
    // Sixteen 2-bit weights packed bit-reversed from the top of the block.
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

/// Encode sixteen `RGBA8` texels into a single 4x4 ASTC LDR block using a
/// **finer-weight CEM-12** configuration:
///
/// * **block mode 67** (`0b1000011`): a 4x4 weight grid, single plane, weight
///   range QUANT_6 (one trit + one low bit -> **six** interpolation levels) --
///   42 weight bits;
/// * **single partition**, **CEM 12** (RGBA direct, eight colour integers);
/// * **QUANT_256 colour**: `color_bits = 111 - 42 = 69` with eight CEM-12
///   integers gives colour quant level QUANT_256 (8-bit identity), so the eight
///   endpoint bytes -- including both alphas -- are written straight into the
///   block and decode back bit-for-bit.
///
/// Six weight levels (vs the four of [`encode_astc_single_partition_4x4_ldr_rgba`])
/// reconstruct smooth gradients more tightly while still carrying real alpha.
/// The weight stream is a trit BISE sequence, so it is packed with
/// `trit_quint::encode_trit_sequence` + `bits::BlockWriter::mirror_weight_stream`
/// (the exact inverse of the decoder's reversed-ISE weight path) rather than the
/// bit-only `write_weights_reversed`.
#[must_use]
pub fn encode_astc_single_partition_4x4_ldr_rgba_q6(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 67;
    const CEM_RGBA_DIRECT: u32 = 12;
    const WEIGHT_LEVELS: u32 = 6; // QUANT_6: one trit + one low bit
    const WEIGHT_LOW_BITS: u32 = 1; // trit range low bits (bits==1)

    // Fit RGBA endpoints ordered so hadd_rgb(e0) <= hadd_rgb(e1); the CEM-12
    // decoder then takes its plain (no blue-contraction swap) path and, with
    // QUANT_256 identity colour, reconstructs these bytes exactly. Weights are
    // fitted against the same endpoints over all four channels using the full
    // trit-aware unquantiser.
    let (e0, e1) = endpoint_fit::fit_rgba_endpoints(texels);
    let raw = weight_fit::quantize_weights_ise_rgba(texels, e0, e1, WEIGHT_LEVELS);

    let mut w = bits::BlockWriter::new();
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field (4 bits at block bit 13): value 12.
    w.write_bits(13, 4, CEM_RGBA_DIRECT);
    // Eight 8-bit colour values at block bit 17, LSB-first, in the decoder's
    // read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b, e0.a, e1.a].
    let vals = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]];
    for (i, v) in vals.iter().enumerate() {
        w.write_bits(17 + i as u32 * 8, 8, u32::from(*v));
    }
    // Encode the sixteen trit weights into a scratch block LSB-first from bit 0,
    // then mirror the 42-bit stream into the top of the block (bit p -> 127-p),
    // which is the exact inverse of the decoder's reversed-ISE weight read.
    let mut scratch = [0u8; 16];
    super::trit_quint::encode_trit_sequence(&mut scratch, 0, WEIGHT_LOW_BITS, &raw);
    w.mirror_weight_stream(&scratch);
    w.into_block()
}

/// Encode twenty-five `RGBA8` texels (row-major, `texel = y * 5 + x`) into a
/// single **5x5** ASTC LDR block -- the first larger-footprint encoder.
///
/// Configuration (verified against the authoritative block-mode scan):
/// * **block mode 243** (`0b011110011`): a 5x5 weight grid, single plane,
///   weight range QUANT_8 (3-bit, bit-only) -- 25 weights x 3 = 75 weight bits;
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_64 colour** (`color_bits = 111 - 75 = 36`, six CEM-8 integers ->
///   colour quant level QUANT_64, a 6-bit bit-only range): each endpoint
///   channel is quantised to 6 bits and the decoded (unquantised) endpoints are
///   fed to the weight fit so the levels reconstruct the intended colours.
///
/// A 5x5 full weight grid needs **no bilinear infill** -- weight `t` maps 1:1 to
/// texel `t` in row-major order -- so the chosen levels decode back exactly up
/// to the QUANT_64 endpoint rounding (<= 2 LSB) and the 3-bit weight step. This
/// mirrors [`encode_astc_single_partition_4x4_ldr`] but on the larger footprint
/// where the colour range necessarily drops from identity to QUANT_64.
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_5x5_ldr(texels: &[[u8; 4]; 25]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 243;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 3; // QUANT_8, bit-only
    const COLOR_LEVEL: usize = 10; // QUANT_64 (6-bit, bit-only)
    const COLOR_BITS: u32 = 6;

    // Fit the principal-axis RGB endpoints over all twenty-five texels, then
    // quantise each channel into the QUANT_64 packed representation.
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
    // on the *decoded* colours. Pre-swap the packed endpoints so the decoder
    // takes the plain path and interpolates d0..d1 directly; weights are fitted
    // after the swap so the texel mapping stays correct.
    let hadd = |c: [u8; 3]| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    if hadd(d0) > hadd(d1) {
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut d0, &mut d1);
    }

    // Fit the twenty-five 3-bit weights against the *decoded* endpoints so the
    // chosen levels reconstruct the intended colours (no infill on a full grid).
    let raw = weight_fit::quantize_weights_bits(texels, d0, d1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six QUANT_64 colour integers (6-bit bit-only) at block bit 17, LSB-first,
    // in the decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let packed = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]];
    for (i, &v) in packed.iter().enumerate() {
        w.write_bits(17 + i as u32 * COLOR_BITS, COLOR_BITS, u32::from(v));
    }
    // Twenty-five 3-bit weights packed bit-reversed from the top of the block.
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

/// Encode thirty-six `RGBA8` texels (row-major, `texel = y * 6 + x`) into a
/// single **6x6** ASTC LDR block.
///
/// Configuration (verified against the authoritative block-mode scan):
/// * **block mode 276** (`0b100010100`): a 6x6 weight grid, single plane,
///   weight range **QUANT_3** (one trit, zero low bits -> three interpolation
///   levels, 58 weight bits);
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_256 colour**: with `color_bits = 111 - 58 = 53` and six CEM-8
///   integers the colour quant level is QUANT_256 (8-bit identity), so the six
///   endpoint bytes are written straight into the block and decode back
///   bit-for-bit -- the endpoints are *exact*, unlike the QUANT_64 rounding of
///   the 5x5 encoder.
///
/// This is the mirror-image trade of [`encode_astc_single_partition_5x5_ldr`]:
/// a 6x6 full grid leaves only enough colour bits for identity endpoints but
/// forces the weight range down to three trit levels. The endpoints therefore
/// reconstruct exactly while smooth gradients are quantised to three steps.
/// Because the 6x6 weight grid equals the footprint there is **no bilinear
/// infill** -- weight `t` maps 1:1 to texel `t` in row-major order.
///
/// The weight stream is a trit BISE sequence, so it is packed with
/// `trit_quint::encode_trit_sequence` + `bits::BlockWriter::mirror_weight_stream`
/// (the exact inverse of the decoder's reversed-ISE weight read) rather than
/// the bit-only `write_weights_reversed`.
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_6x6_ldr(texels: &[[u8; 4]; 36]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 276;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_LEVELS: u32 = 3; // QUANT_3: one trit, zero low bits
    const WEIGHT_LOW_BITS: u32 = 0; // trit range low bits (bits == 0)

    // Fit the principal-axis RGB endpoints over all thirty-six texels. The fit
    // orders them so `hadd(e0) <= hadd(e1)`, which with QUANT_256 identity
    // colour means the CEM-8 decoder reproduces them without its
    // blue-contraction swap -- so the raw bytes are also the decoded endpoints.
    let (e0, e1) = endpoint_fit::fit_rgb_endpoints(texels);

    // Fit the thirty-six trit weights against the exact endpoints over RGB (no
    // infill on a full grid).
    let raw = weight_fit::quantize_weights_ise(texels, e0, e1, WEIGHT_LEVELS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six 8-bit identity colour values at block bit 17, LSB-first, in the
    // decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let vals = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]];
    for (i, v) in vals.iter().enumerate() {
        w.write_bits(17 + i as u32 * 8, 8, u32::from(*v));
    }
    // Encode the thirty-six trit weights into a scratch block LSB-first from
    // bit 0, then mirror the 58-bit stream into the top of the block
    // (bit p -> 127-p), the exact inverse of the decoder's reversed-ISE read.
    let mut scratch = [0u8; 16];
    super::trit_quint::encode_trit_sequence(&mut scratch, 0, WEIGHT_LOW_BITS, &raw);
    w.mirror_weight_stream(&scratch);
    w.into_block()
}

/// Encode sixty-four `RGBA8` texels (row-major, `texel = y * 8 + x`) into a
/// single **8x8** ASTC LDR block.
///
/// Configuration (verified against the authoritative block-mode scan -- 8x8 is
/// the *only* single-plane CEM-8 full-grid mode on this footprint):
/// * **block mode 1348** (`0b10101000100`): an 8x8 weight grid, single plane,
///   weight range **QUANT_2** (one bit -> two interpolation levels, 64 weight
///   bits);
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_192 trit colour**: with `color_bits = 111 - 64 = 47` the six CEM-8
///   integers are a QUANT_192 trit BISE sequence (46 bits), not identity
///   colour, so the endpoints are quantised to QUANT_192 and pre-swapped to
///   dodge the decoder's blue-contraction branch (mirrors the mode-578 4x4
///   encoder's colour path).
///
/// 8x8 is a legal hardware footprint (`AstcBlock::B8x8`), and the 8x8 weight
/// grid equals the footprint so there is **no bilinear infill** -- weight `t`
/// maps 1:1 to texel `t` in row-major order. The single-bit weight range means
/// each texel snaps to the nearer of the two QUANT_192 endpoints.
///
/// The weight stream is a plain 1-bit range, packed with the bit-only
/// `bits::BlockWriter::write_weights_reversed`; the colour is a trit BISE
/// written with `trit_quint::encode_trit_sequence` at block bit 17 (the two
/// regions, colour 17..63 and weights 64..128, do not overlap).
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_8x8_ldr(texels: &[[u8; 4]; 64]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 1348;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 1; // QUANT_2, bit-only (two levels)
    const COLOR_LEVEL: usize = 15; // QUANT_192 (trit + 6 low bits)
    const COLOR_LOW_BITS: u32 = 6; // QUANT_192 trit range low bits

    // Fit the principal-axis RGB endpoints over all sixty-four texels, then
    // quantise each channel into the QUANT_192 packed representation.
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
    // on the *decoded* colours. Pre-swap the packed endpoints so the decoder
    // takes the plain path and interpolates d0..d1 directly; weights are fitted
    // after the swap so the texel mapping stays correct.
    let hadd = |c: [u8; 3]| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    if hadd(d0) > hadd(d1) {
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut d0, &mut d1);
    }

    // Fit the sixty-four 1-bit weights against the *decoded* endpoints (no
    // infill on a full grid): each texel snaps to the nearer endpoint.
    let raw = weight_fit::quantize_weights_bits(texels, d0, d1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Sixty-four 1-bit weights packed bit-reversed from the top of the block
    // (bits 64..128).
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    let mut block = w.into_block();

    // Six QUANT_192 colour integers as a trit BISE at block bit 17, in the
    // decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b]. Each packed
    // value is `low(6 bits) | (trit << 6)`, exactly the table index. Colour
    // (17..63) and the weights (64..128) do not overlap.
    let packed = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]];
    super::trit_quint::encode_trit_sequence(&mut block, 17, COLOR_LOW_BITS, &packed);
    block
}

/// Encode twenty `RGBA8` texels (row-major, `texel = y * 5 + x`, 5 wide x 4
/// tall) into a single **5x4** ASTC LDR block -- the first *non-square* legal
/// footprint encoder (`AstcBlock::B5x4`).
///
/// Configuration (verified against the authoritative block-mode scan):
/// * **block mode 706** (`0b1011000010`): a 5x4 weight grid, single plane,
///   weight range **QUANT_16** (four bit-only bits -> sixteen interpolation
///   levels, 20*4 = 80 weight bits);
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_32 colour**: with `color_bits = 111 - 80 = 31` the six CEM-8
///   integers are a QUANT_32 bit-only BISE (6*5 = 30 bits), so each endpoint
///   channel is quantised to one of 32 levels and the packed value is written
///   straight into the block (no trit/quint interleave).
///
/// 5x4 is a legal hardware footprint and the 5x4 weight grid equals the
/// footprint, so there is **no bilinear infill** -- weight `t` maps 1:1 to
/// texel `t` in row-major order. Sixteen weight levels give smooth gradients
/// while QUANT_32 endpoints stay within the bit-only quantisation budget. This
/// reuses the const-generic endpoint/weight primitives (`N = 20`) proven on the
/// square footprints, extending the encoder to the non-square footprint family.
///
/// The weight stream is a plain 4-bit range packed with the bit-only
/// `bits::BlockWriter::write_weights_reversed`; the colour is written as six
/// direct 5-bit values at block bit 17 (colour 17..47 and weights 48..128 do
/// not overlap).
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_5x4_ldr(texels: &[[u8; 4]; 20]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 706;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 4; // QUANT_16, bit-only (sixteen levels)
    const COLOR_LEVEL: usize = 7; // QUANT_32 (5-bit, bit-only)
    const COLOR_BITS: u32 = 5;

    // Fit the principal-axis RGB endpoints over all twenty texels, then
    // quantise each channel into the QUANT_32 packed representation.
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
    // on the *decoded* colours. Pre-swap the packed endpoints so the decoder
    // takes the plain path and interpolates d0..d1 directly; weights are fitted
    // after the swap so the texel mapping stays correct.
    let hadd = |c: [u8; 3]| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    if hadd(d0) > hadd(d1) {
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut d0, &mut d1);
    }

    // Fit the twenty 4-bit weights against the *decoded* endpoints (no infill
    // on a full grid): each texel picks the nearest of sixteen levels.
    let raw = weight_fit::quantize_weights_bits(texels, d0, d1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six QUANT_32 colour integers (5-bit bit-only) at block bit 17, LSB-first,
    // in the decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let packed = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]];
    for (i, &v) in packed.iter().enumerate() {
        w.write_bits(17 + i as u32 * COLOR_BITS, COLOR_BITS, u32::from(v));
    }
    // Twenty 4-bit weights packed bit-reversed from the top of the block
    // (bits 48..128).
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

/// Encode forty `RGBA8` texels (row-major, `texel = y * 8 + x`, 8 wide x 5
/// tall) into a single **8x5** ASTC LDR block (`AstcBlock::B8x5`), extending
/// the non-square footprint family.
///
/// Configuration (verified against the authoritative block-mode scan,
/// footprint (8, 5)):
/// * **block mode 102** (`0b0001100110`): an 8x5 weight grid, single plane,
///   weight range **QUANT_4** (two bit-only bits -> four interpolation levels,
///   40*2 = 80 weight bits);
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_32 colour**: with `color_bits = 111 - 80 = 31` the six CEM-8
///   integers are a QUANT_32 bit-only BISE (6*5 = 30 bits), identical to the
///   5x4 mode-706 colour recipe.
///
/// 8x5 is a legal hardware footprint and the 8x5 weight grid equals the
/// footprint, so there is **no bilinear infill** -- weight `t` maps 1:1 to
/// texel `t` in row-major order. Mode 102 is the richest single-plane CEM-8
/// full-grid mode on 8x5 that stays within the bit-only colour budget: four
/// weight levels plus 32-level endpoints. This reuses the const-generic
/// endpoint/weight primitives (`N = 40`) proven on the smaller footprints.
///
/// The weight stream is a plain 2-bit range packed with the bit-only
/// `bits::BlockWriter::write_weights_reversed`; the colour is written as six
/// direct 5-bit values at block bit 17 (colour 17..47 and weights 48..128 do
/// not overlap).
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_8x5_ldr(texels: &[[u8; 4]; 40]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 102;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 2; // QUANT_4, bit-only (four levels)
    const COLOR_LEVEL: usize = 7; // QUANT_32 (5-bit, bit-only)
    const COLOR_BITS: u32 = 5;

    // Fit the principal-axis RGB endpoints over all forty texels, then quantise
    // each channel into the QUANT_32 packed representation.
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
    // on the *decoded* colours. Pre-swap the packed endpoints so the decoder
    // takes the plain path and interpolates d0..d1 directly; weights are fitted
    // after the swap so the texel mapping stays correct.
    let hadd = |c: [u8; 3]| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    if hadd(d0) > hadd(d1) {
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut d0, &mut d1);
    }

    // Fit the forty 2-bit weights against the *decoded* endpoints (no infill on
    // a full grid): each texel picks the nearest of four levels.
    let raw = weight_fit::quantize_weights_bits(texels, d0, d1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six QUANT_32 colour integers (5-bit bit-only) at block bit 17, LSB-first,
    // in the decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let packed = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]];
    for (i, &v) in packed.iter().enumerate() {
        w.write_bits(17 + i as u32 * COLOR_BITS, COLOR_BITS, u32::from(v));
    }
    // Forty 2-bit weights packed bit-reversed from the top of the block
    // (bits 48..128).
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

/// Encode forty-eight `RGBA8` texels (row-major, `texel = y * 8 + x`, 8 wide x
/// 6 tall) into a single **8x6** ASTC LDR block (`AstcBlock::B8x6`), extending
/// the non-square footprint family with an identity-colour recipe.
///
/// Configuration (verified against the authoritative block-mode scan,
/// footprint (8, 6)):
/// * **block mode 324** (`0b0101000100`): an 8x6 weight grid, single plane,
///   weight range **QUANT_2** (one bit -> two interpolation levels, 48*1 = 48
///   weight bits);
/// * **single partition**, **CEM 8** (RGB direct, alpha forced to 255);
/// * **QUANT_256 colour**: with `color_bits = 111 - 48 = 63` and six CEM-8
///   integers the colour quant level is QUANT_256 (8-bit identity), so the six
///   endpoint bytes are written straight into the block and decode back
///   bit-for-bit -- the endpoints are *exact*.
///
/// This is the identity-colour trade mirrored from the 6x6 mode-276 encoder,
/// but with a single-bit weight range instead of trit weights: a larger 8x6
/// footprint leaves 63 colour bits (enough for identity endpoints) yet only one
/// weight bit per texel, so each texel snaps to the nearer of the two exact
/// endpoints. Because the 8x6 weight grid equals the footprint there is **no
/// bilinear infill** -- weight `t` maps 1:1 to texel `t` in row-major order.
///
/// The weight stream is a plain 1-bit range packed with the bit-only
/// `bits::BlockWriter::write_weights_reversed`; the colour is six direct 8-bit
/// values at block bit 17 (colour 17..65 and weights 80..128 do not overlap).
///
/// CEM 8 carries no alpha, so the decoded block has alpha 255 for every texel
/// and the input alpha channel is ignored.
#[must_use]
pub fn encode_astc_single_partition_8x6_ldr(texels: &[[u8; 4]; 48]) -> [u8; 16] {
    const BLOCK_MODE: u32 = 324;
    const CEM_RGB_DIRECT: u32 = 8;
    const WEIGHT_BITS: u32 = 1; // QUANT_2, bit-only (two levels)

    // Fit the principal-axis RGB endpoints over all forty-eight texels. The fit
    // orders them so `hadd(e0) <= hadd(e1)`, which with QUANT_256 identity
    // colour means the CEM-8 decoder reproduces them without its
    // blue-contraction swap -- so the raw bytes are also the decoded endpoints.
    let (e0, e1) = endpoint_fit::fit_rgb_endpoints(texels);

    // Fit the forty-eight 1-bit weights against the exact endpoints (no infill
    // on a full grid): each texel snaps to the nearer endpoint.
    let raw = weight_fit::quantize_weights_bits(texels, e0, e1, WEIGHT_BITS);

    let mut w = bits::BlockWriter::new();
    // Block mode occupies block bits 0..11; single partition leaves the
    // partition-count field (bits 11,12) at 0.
    w.write_bits(0, 11, BLOCK_MODE);
    // CEM field: 4 bits at block bit 13. CEM 8 sets only block bit 16.
    w.write_bits(13, 4, CEM_RGB_DIRECT);
    // Six 8-bit identity colour values at block bit 17, LSB-first, in the
    // decoder's read order [e0.r, e1.r, e0.g, e1.g, e0.b, e1.b].
    let vals = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]];
    for (i, v) in vals.iter().enumerate() {
        w.write_bits(17 + i as u32 * 8, 8, u32::from(*v));
    }
    // Forty-eight 1-bit weights packed bit-reversed from the top of the block
    // (bits 80..128).
    w.write_weights_reversed(&raw, WEIGHT_BITS);
    w.into_block()
}

#[cfg(test)]
mod tests {
    use super::super::decode_astc_4x4_ldr;
    use super::super::decode_astc_ldr;
    use super::encode_astc_single_partition_4x4_ldr;
    use super::encode_astc_single_partition_4x4_ldr_q192;
    use super::encode_astc_single_partition_4x4_ldr_quality;
    use super::encode_astc_single_partition_4x4_ldr_rgba;
    use super::encode_astc_single_partition_4x4_ldr_rgba_q6;
    use super::encode_astc_single_partition_5x4_ldr;
    use super::encode_astc_single_partition_5x5_ldr;
    use super::encode_astc_single_partition_6x6_ldr;
    use super::encode_astc_single_partition_8x5_ldr;
    use super::encode_astc_single_partition_8x6_ldr;
    use super::encode_astc_single_partition_8x8_ldr;

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

    /// Sum of squared RGB error over the sixteen texels (alpha excluded),
    /// matching the metric the quality encoder minimises.
    fn ssd_rgb(src: &[[u8; 4]; 16], dec: &[[u8; 4]; 16]) -> u64 {
        let mut e = 0u64;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                let v = i64::from(s[c]) - i64::from(d[c]);
                e += (v * v) as u64;
            }
        }
        e
    }

    #[test]
    fn quality_never_worse_than_either_mode() {
        // Diverse tiles so the two modes trade wins: constant, gray ramp, rgb
        // gradient, two-colour checker, and a pseudo-random noise block.
        let tiles: [[[u8; 4]; 16]; 5] = [
            [[73, 150, 211, 255]; 16],
            core::array::from_fn(|t| [(t * 17) as u8, (t * 17) as u8, (t * 17) as u8, 255]),
            core::array::from_fn(|t| {
                let f = t as u8 * 16;
                [f, 255 - f, (f / 2) + 40, 255]
            }),
            core::array::from_fn(|t| {
                if (t / 4 + t % 4) % 2 == 0 {
                    [240, 20, 30, 255]
                } else {
                    [10, 200, 60, 255]
                }
            }),
            core::array::from_fn(|t| {
                let r = (t.wrapping_mul(97).wrapping_add(13) & 0xff) as u8;
                let g = (t.wrapping_mul(53).wrapping_add(7) & 0xff) as u8;
                let b = (t.wrapping_mul(29).wrapping_add(1) & 0xff) as u8;
                [r, g, b, 255]
            }),
        ];

        for (i, src) in tiles.iter().enumerate() {
            let blk83 = encode_astc_single_partition_4x4_ldr(src);
            let blk578 = encode_astc_single_partition_4x4_ldr_q192(src);
            let blkq = encode_astc_single_partition_4x4_ldr_quality(src);

            let e83 = ssd_rgb(src, &decode_astc_4x4_ldr(&blk83).expect("decode mode 83"));
            let e578 = ssd_rgb(src, &decode_astc_4x4_ldr(&blk578).expect("decode mode 578"));
            let decq = decode_astc_4x4_ldr(&blkq).expect("decode quality block");
            let eq = ssd_rgb(src, &decq);

            // The chosen block must be no worse than the better individual mode.
            assert!(
                eq <= e83.min(e578),
                "tile {i}: quality SSD {eq} worse than min(mode83 {e83}, mode578 {e578})"
            );
            // And it must be exactly one of the two candidates.
            assert!(
                blkq == blk83 || blkq == blk578,
                "tile {i}: quality block is not one of the candidates"
            );
            for d in &decq {
                assert_eq!(d[3], 255, "tile {i}: CEM 8 forces alpha 255");
            }
        }
    }

    #[test]
    fn quality_picks_mode578_on_gray_ramp() {
        // Smooth ramp: the 4-bit weights of mode 578 win, so quality selects it.
        let src: [[u8; 4]; 16] =
            core::array::from_fn(|t| [(t * 17) as u8, (t * 17) as u8, (t * 17) as u8, 255]);
        let blk578 = encode_astc_single_partition_4x4_ldr_q192(&src);
        let blkq = encode_astc_single_partition_4x4_ldr_quality(&src);
        assert_eq!(
            blkq, blk578,
            "gray ramp should select the mode-578 candidate"
        );
    }

    /// Max per-channel error over *all four* channels (RGBA) after a round trip.
    fn max_rgba_err(src: &[[u8; 4]; 16], dec: &[[u8; 4]; 16]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..4 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn rgba_constant_block_round_trips_exactly_incl_alpha() {
        // Coincident endpoints => QUANT_256 identity reproduces the colour and
        // the real alpha exactly, proving CEM 12 carries alpha (not forced 255).
        let src = [[73u8, 150, 211, 128]; 16];
        let blk = encode_astc_single_partition_4x4_ldr_rgba(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode constant RGBA block");
        assert_eq!(max_rgba_err(&src, &dec), 0, "constant RGBA block mismatch");
        for d in &dec {
            assert_eq!(d[3], 128, "CEM 12 must carry the real alpha, not 255");
        }
    }

    #[test]
    fn rgba_alpha_gradient_is_carried_not_forced() {
        // Fixed RGB with a linear alpha ramp 0..255. The two 2-bit weight levels
        // only reach four alpha steps, but the decoded alpha must *vary* and
        // track the source -- it must never collapse to a constant 255.
        let src: [[u8; 4]; 16] = core::array::from_fn(|t| [40, 90, 160, (t * 17) as u8]);
        let blk = encode_astc_single_partition_4x4_ldr_rgba(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode alpha gradient");

        // Endpoints bracket the alpha range, so texel 0 (~0) and texel 15 (255)
        // land near the extremes.
        assert!(dec[0][3] <= 32, "low alpha not carried: {}", dec[0][3]);
        assert!(dec[15][3] >= 223, "high alpha not carried: {}", dec[15][3]);

        // Alpha must genuinely vary across the block.
        let mut amin = 255i32;
        let mut amax = 0i32;
        for d in &dec {
            amin = amin.min(i32::from(d[3]));
            amax = amax.max(i32::from(d[3]));
        }
        assert!(
            amax - amin >= 128,
            "alpha did not vary across the block (min {amin}, max {amax})"
        );

        // The low end must land well below 255: a CEM-8 encoder would force
        // every texel to 255, so a sub-255 texel proves alpha is really stored.
        // (Mid/high texels may legitimately snap to the 255 weight level under
        // 2-bit quantisation, so only the low end is asserted here; the
        // constant-128 test above is the decisive "not forced" proof.)
        assert!(
            dec.iter().any(|d| d[3] < 200),
            "every decoded alpha was >= 200 -- alpha looks forced, not carried"
        );
    }

    #[test]
    fn rgba_q6_constant_block_round_trips_exactly_incl_alpha() {
        // Coincident endpoints => QUANT_256 identity reproduces the colour and
        // the real alpha exactly, exactly as the four-level CEM-12 encoder does.
        let src = [[73u8, 150, 211, 128]; 16];
        let blk = encode_astc_single_partition_4x4_ldr_rgba_q6(&src);
        let dec = decode_astc_4x4_ldr(&blk).expect("decode constant RGBA q6 block");
        assert_eq!(
            max_rgba_err(&src, &dec),
            0,
            "constant RGBA q6 block mismatch"
        );
        for d in &dec {
            assert_eq!(d[3], 128, "CEM 12 q6 must carry the real alpha, not 255");
        }
    }

    #[test]
    fn rgba_q6_reconstructs_gradient_tighter_than_four_levels() {
        // A smooth on-axis grayscale ramp: both encoders fit identical RGBA
        // endpoints (same `fit_rgba_endpoints`), so the only difference is weight
        // resolution. Six trit levels must bracket the ramp more tightly than the
        // four bit-only levels of the M4 CEM-12 encoder.
        let src: [[u8; 4]; 16] = core::array::from_fn(|t| {
            let v = (t * 17) as u8; // 0..255 across the 16 texels
            [v, v, v, 255]
        });

        let blk_m4 = encode_astc_single_partition_4x4_ldr_rgba(&src);
        let blk_q6 = encode_astc_single_partition_4x4_ldr_rgba_q6(&src);

        let dec_m4 = decode_astc_4x4_ldr(&blk_m4).expect("decode M4 gradient");
        let dec_q6 = decode_astc_4x4_ldr(&blk_q6).expect("decode q6 gradient");

        let err_m4 = max_rgba_err(&src, &dec_m4);
        let err_q6 = max_rgba_err(&src, &dec_q6);

        assert!(
            err_q6 < err_m4,
            "six-level weights ({err_q6}) did not beat four-level weights ({err_m4})"
        );
    }
    /// Max per-channel RGB error over twenty-five texels after a round trip.
    fn max_rgb_err_25(src: &[[u8; 4]; 25], dec: &[[u8; 4]; 25]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn five_by_five_constant_block_round_trips_within_two_lsb() {
        let src = [[73u8, 150, 211, 255]; 25];
        let blk = encode_astc_single_partition_5x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 5).expect("decode 5x5 constant");
        assert_eq!(count, 25, "5x5 footprint must decode 25 texels");
        let dec: [[u8; 4]; 25] = core::array::from_fn(|t| dec144[t]);
        // Coincident endpoints quantised to QUANT_64: reconstructed within the
        // 6-bit colour step (<= 2 LSB on the evenly spaced 64-level ramp).
        assert!(
            max_rgb_err_25(&src, &dec) <= 2,
            "5x5 constant block exceeds QUANT_64 colour step"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn five_by_five_gray_ramp_round_trips_within_tolerance() {
        // A smooth gray ramp lies on a single colour axis; the 3-bit weights
        // (eight levels) track the twenty-five-step ramp with a bounded error.
        let src: [[u8; 4]; 25] =
            core::array::from_fn(|t| [(t * 10) as u8, (t * 10) as u8, (t * 10) as u8, 255]);
        let blk = encode_astc_single_partition_5x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 5).expect("decode 5x5 gray ramp");
        assert_eq!(count, 25);
        let dec: [[u8; 4]; 25] = core::array::from_fn(|t| dec144[t]);
        // Eight weight levels across a 25-step ramp cannot be exact; the worst
        // texel lands within one weight step (~1/7 of the range) plus the
        // QUANT_64 endpoint rounding.
        assert!(
            max_rgb_err_25(&src, &dec) <= 20,
            "5x5 gray ramp error too large"
        );
    }

    #[test]
    fn five_by_five_rgb_gradient_round_trips_within_tolerance() {
        // A colour gradient along the R/G/B diagonal: endpoints capture the
        // extremes and the 3-bit weights interpolate between them.
        let src: [[u8; 4]; 25] = core::array::from_fn(|t| {
            let f = (t * 10) as u8;
            [f, 255 - f, (f / 2) + 20, 255]
        });
        let blk = encode_astc_single_partition_5x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 5).expect("decode 5x5 rgb gradient");
        assert_eq!(count, 25);
        let dec: [[u8; 4]; 25] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_25(&src, &dec) <= 28,
            "5x5 rgb gradient error too large"
        );
    }

    /// Max per-channel RGB error over thirty-six texels after a round trip.
    fn max_rgb_err_36(src: &[[u8; 4]; 36], dec: &[[u8; 4]; 36]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn six_by_six_constant_block_round_trips_exactly() {
        // Coincident endpoints with QUANT_256 identity colour: unlike the 5x5
        // encoder's QUANT_64 rounding, the 6x6 encoder stores the endpoints
        // exactly, so a constant block round-trips bit-for-bit on RGB.
        let src = [[73u8, 150, 211, 255]; 36];
        let blk = encode_astc_single_partition_6x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 6, 6).expect("decode 6x6 constant");
        assert_eq!(count, 36, "6x6 footprint must decode 36 texels");
        let dec: [[u8; 4]; 36] = core::array::from_fn(|t| dec144[t]);
        assert_eq!(
            max_rgb_err_36(&src, &dec),
            0,
            "6x6 constant block must round-trip exactly with identity colour"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn six_by_six_two_colour_endpoints_are_exact() {
        // A hard split between two colours: every texel sits on one of the two
        // exact endpoints, so three trit weight levels (which include both
        // extremes) reconstruct each texel exactly despite the coarse range.
        let a = [20u8, 40, 60, 255];
        let b = [200u8, 180, 160, 255];
        let src: [[u8; 4]; 36] = core::array::from_fn(|t| if t % 2 == 0 { a } else { b });
        let blk = encode_astc_single_partition_6x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 6, 6).expect("decode 6x6 two-colour");
        assert_eq!(count, 36);
        let dec: [[u8; 4]; 36] = core::array::from_fn(|t| dec144[t]);
        // Endpoints are stored exactly (identity colour) and the weight range
        // includes the 0 and max levels, so both colours reconstruct exactly.
        assert_eq!(
            max_rgb_err_36(&src, &dec),
            0,
            "6x6 two-colour block must hit both exact endpoints"
        );
    }

    #[test]
    fn six_by_six_gray_ramp_round_trips_within_tolerance() {
        // A smooth gray ramp on a single axis: only three trit weight levels are
        // available, so the mid-ramp texels snap to the nearest of three steps.
        // Endpoints are exact, so the error is bounded by half the weight step.
        let src: [[u8; 4]; 36] =
            core::array::from_fn(|t| [(t * 7) as u8, (t * 7) as u8, (t * 7) as u8, 255]);
        let blk = encode_astc_single_partition_6x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 6, 6).expect("decode 6x6 gray ramp");
        assert_eq!(count, 36);
        let dec: [[u8; 4]; 36] = core::array::from_fn(|t| dec144[t]);
        // Three levels across a 36-step ramp: worst case is about a quarter of
        // the full range between adjacent levels.
        assert!(
            max_rgb_err_36(&src, &dec) <= 70,
            "6x6 gray ramp error too large for three trit levels"
        );
    }

    fn max_rgb_err_64(src: &[[u8; 4]; 64], dec: &[[u8; 4]; 64]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn eight_by_eight_constant_block_round_trips_within_quant192() {
        // Mode 1348 uses QUANT_192 trit colour (not identity), so a constant
        // block reconstructs within the QUANT_192 quantisation budget (<= 2
        // LSB) rather than bit-exactly, regardless of the single-bit weights.
        let src = [[41u8, 173, 98, 255]; 64];
        let blk = encode_astc_single_partition_8x8_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 8).expect("decode 8x8 constant");
        assert_eq!(count, 64, "8x8 footprint must decode 64 texels");
        let dec: [[u8; 4]; 64] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_64(&src, &dec) <= 2,
            "8x8 constant block must round-trip within the QUANT_192 budget"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn eight_by_eight_two_colour_endpoints_within_quant192() {
        // A hard split between two colours: every texel sits on one of the two
        // endpoints. With exact snapping the only error is the QUANT_192
        // endpoint quantisation (<= 2 LSB), not weight interpolation.
        let a = [15u8, 35, 55, 255];
        let b = [210u8, 190, 170, 255];
        let src: [[u8; 4]; 64] = core::array::from_fn(|t| if t % 2 == 0 { a } else { b });
        let blk = encode_astc_single_partition_8x8_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 8).expect("decode 8x8 two-colour");
        assert_eq!(count, 64);
        let dec: [[u8; 4]; 64] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_64(&src, &dec) <= 2,
            "8x8 two-colour block must hit both endpoints within QUANT_192"
        );
    }

    #[test]
    fn eight_by_eight_gray_ramp_snaps_to_nearer_endpoint() {
        // A smooth gray ramp with only two weight levels: every texel snaps to
        // whichever endpoint is closer, so the worst case is about half the
        // endpoint span plus the QUANT_192 endpoint budget.
        let src: [[u8; 4]; 64] =
            core::array::from_fn(|t| [(t * 4) as u8, (t * 4) as u8, (t * 4) as u8, 255]);
        let blk = encode_astc_single_partition_8x8_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 8).expect("decode 8x8 gray ramp");
        assert_eq!(count, 64);
        let dec: [[u8; 4]; 64] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_64(&src, &dec) <= 128,
            "8x8 gray ramp error exceeds the two-level half-span bound"
        );
    }

    fn max_rgb_err_20(src: &[[u8; 4]; 20], dec: &[[u8; 4]; 20]) -> i32 {
        let mut m = 0i32;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn five_by_four_constant_block_round_trips_within_quant32() {
        // Mode 706 uses QUANT_32 (5-bit) colour, so a constant block
        // reconstructs within the QUANT_32 quantisation budget (<= 5 LSB).
        let src = [[41u8, 173, 98, 255]; 20];
        let blk = encode_astc_single_partition_5x4_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 4).expect("decode 5x4 constant");
        assert_eq!(count, 20, "5x4 footprint must decode 20 texels");
        let dec: [[u8; 4]; 20] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_20(&src, &dec) <= 5,
            "5x4 constant block must round-trip within the QUANT_32 budget"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn five_by_four_two_colour_endpoints_within_quant32() {
        // A hard split between two colours: every texel sits on one of the two
        // endpoints, reached exactly by the sixteen weight levels (0 / max), so
        // the only error is the QUANT_32 endpoint quantisation (<= 5 LSB).
        let a = [15u8, 35, 55, 255];
        let b = [210u8, 190, 170, 255];
        let src: [[u8; 4]; 20] = core::array::from_fn(|t| if t % 2 == 0 { a } else { b });
        let blk = encode_astc_single_partition_5x4_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 4).expect("decode 5x4 two-colour");
        assert_eq!(count, 20);
        let dec: [[u8; 4]; 20] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_20(&src, &dec) <= 5,
            "5x4 two-colour block must hit both endpoints within QUANT_32"
        );
    }

    #[test]
    fn five_by_four_rgb_gradient_round_trips_within_tolerance() {
        // A smooth RGB gradient: sixteen weight levels interpolate between the
        // QUANT_32 endpoints, so the reconstruction stays within the combined
        // weight-step and endpoint budget.
        let src: [[u8; 4]; 20] = core::array::from_fn(|t| {
            let v = (t * 12) as u8;
            [v, 255 - v, (v / 2).wrapping_add(40), 255]
        });
        let blk = encode_astc_single_partition_5x4_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 5, 4).expect("decode 5x4 gradient");
        assert_eq!(count, 20);
        let dec: [[u8; 4]; 20] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_20(&src, &dec) <= 24,
            "5x4 gradient error exceeds the sixteen-level + QUANT_32 budget"
        );
    }

    fn max_rgb_err_40(src: &[[u8; 4]; 40], dec: &[[u8; 4]; 40]) -> i32 {
        let mut m = 0;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn eight_by_five_constant_block_round_trips_within_quant32() {
        // Mode 102 uses QUANT_32 (5-bit) colour, so a constant block
        // reconstructs within the QUANT_32 quantisation budget (<= 5 LSB).
        let src = [[41u8, 173, 98, 255]; 40];
        let blk = encode_astc_single_partition_8x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 5).expect("decode 8x5 constant");
        assert_eq!(count, 40, "8x5 footprint must decode 40 texels");
        let dec: [[u8; 4]; 40] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_40(&src, &dec) <= 5,
            "8x5 constant block must round-trip within the QUANT_32 budget"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn eight_by_five_two_colour_endpoints_within_quant32() {
        // A hard split between two colours: every texel sits on one of the two
        // endpoints, reached exactly by the four weight levels (0 / max), so
        // the only error is the QUANT_32 endpoint quantisation (<= 5 LSB).
        let a = [15u8, 35, 55, 255];
        let b = [210u8, 190, 170, 255];
        let src: [[u8; 4]; 40] = core::array::from_fn(|t| if t % 2 == 0 { a } else { b });
        let blk = encode_astc_single_partition_8x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 5).expect("decode 8x5 two-colour");
        assert_eq!(count, 40);
        let dec: [[u8; 4]; 40] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_40(&src, &dec) <= 5,
            "8x5 two-colour block must hit both endpoints within QUANT_32"
        );
    }

    #[test]
    fn eight_by_five_rgb_gradient_round_trips_within_tolerance() {
        // A smooth RGB gradient: four weight levels interpolate between the
        // QUANT_32 endpoints, so the reconstruction stays within the combined
        // weight-step and endpoint budget (coarser than 5x4's sixteen levels).
        let src: [[u8; 4]; 40] = core::array::from_fn(|t| {
            let v = (t * 6) as u8;
            [v, 255 - v, (v / 2).wrapping_add(40), 255]
        });
        let blk = encode_astc_single_partition_8x5_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 5).expect("decode 8x5 gradient");
        assert_eq!(count, 40);
        let dec: [[u8; 4]; 40] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_40(&src, &dec) <= 70,
            "8x5 gradient error exceeds the four-level + QUANT_32 budget"
        );
    }

    fn max_rgb_err_48(src: &[[u8; 4]; 48], dec: &[[u8; 4]; 48]) -> i32 {
        let mut m = 0;
        for (s, d) in src.iter().zip(dec.iter()) {
            for c in 0..3 {
                m = m.max((i32::from(s[c]) - i32::from(d[c])).abs());
            }
        }
        m
    }

    #[test]
    fn eight_by_six_constant_block_round_trips_exactly() {
        // Mode 324 uses QUANT_256 identity colour, so a constant block
        // reconstructs the endpoints exactly (err == 0).
        let src = [[41u8, 173, 98, 255]; 48];
        let blk = encode_astc_single_partition_8x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 6).expect("decode 8x6 constant");
        assert_eq!(count, 48, "8x6 footprint must decode 48 texels");
        let dec: [[u8; 4]; 48] = core::array::from_fn(|t| dec144[t]);
        assert_eq!(
            max_rgb_err_48(&src, &dec),
            0,
            "8x6 constant block must round-trip exactly with identity colour"
        );
        for d in &dec {
            assert_eq!(d[3], 255, "CEM 8 forces alpha 255");
        }
    }

    #[test]
    fn eight_by_six_two_colour_endpoints_are_exact() {
        // A hard split between two colours: identity endpoints plus the two
        // weight levels (0 / max) hit both colours exactly (err == 0).
        let a = [15u8, 35, 55, 255];
        let b = [210u8, 190, 170, 255];
        let src: [[u8; 4]; 48] = core::array::from_fn(|t| if t % 2 == 0 { a } else { b });
        let blk = encode_astc_single_partition_8x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 6).expect("decode 8x6 two-colour");
        assert_eq!(count, 48);
        let dec: [[u8; 4]; 48] = core::array::from_fn(|t| dec144[t]);
        assert_eq!(
            max_rgb_err_48(&src, &dec),
            0,
            "8x6 two-colour block must hit both exact endpoints"
        );
    }

    #[test]
    fn eight_by_six_rgb_gradient_round_trips_within_tolerance() {
        // A smooth RGB gradient: only two weight levels, so the midrange is
        // quantised coarsely (<= 128 LSB), but endpoints stay exact.
        let src: [[u8; 4]; 48] = core::array::from_fn(|t| {
            let v = (t * 5) as u8;
            [v, 255 - v, (v / 2).wrapping_add(40), 255]
        });
        let blk = encode_astc_single_partition_8x6_ldr(&src);
        let (dec144, count) = decode_astc_ldr(&blk, 8, 6).expect("decode 8x6 gradient");
        assert_eq!(count, 48);
        let dec: [[u8; 4]; 48] = core::array::from_fn(|t| dec144[t]);
        assert!(
            max_rgb_err_48(&src, &dec) <= 128,
            "8x6 gradient error exceeds the two-level budget"
        );
    }
}
