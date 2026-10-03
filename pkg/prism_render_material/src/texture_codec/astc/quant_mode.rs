//! ASTC colour quantisation-mode selection.
//!
//! Once the colour bit budget is known (what remains of the block after the
//! weight stream), the decoder must recover the quantisation level the encoder
//! used for the colour endpoints. The ARM `astcenc` reference decoder does this
//! with a static `quant_mode_table[integer_count / 2][color_bits]` lookup
//! (`astcenc_quantization.cpp`, Apache-2.0): the row is picked by the number of
//! colour integers the Colour Endpoint Mode needs and the column by the colour
//! bit budget. The entry is an index into the `astcenc` `QUANT_*` enum, or a
//! `-1` sentinel meaning the budget is too small to encode the endpoints.
//!
//! Rows are transcribed verbatim; row three (six integers, the CEM 8 RGB case)
//! is the one already GPU-proven on the single-partition path.
//!
//! Pure integer lookup -- no AI/ML path.

/// `astcenc` `QUANT_6`: the smallest colour quant level the reference accepts
/// for endpoints. Any entry below this (including the `-1` "budget too small"
/// sentinel) is treated as an error block. The colour unquant table index is
/// `level - QUANT_6`.
pub(super) const QUANT_6: i8 = 4;

/// `astcenc` `QUANT_256`: the 8-bit identity colour quant level.
pub(super) const QUANT_256: i8 = 20;

/// Row 0 (0/1 colour integers): never a valid endpoint configuration.
const ROW0: [i8; 128] = [-1; 128];

/// Row 1 (2 colour integers, e.g. CEM 0/1 luminance).
const ROW1: [i8; 128] = [
    -1, -1, 0, 0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
];

/// Row 2 (4 colour integers, e.g. CEM 4/5/6 luminance+alpha / RGB scale).
const ROW2: [i8; 128] = [
    -1, -1, -1, -1, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, //
    8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
];

/// Row 3 (6 colour integers, e.g. CEM 8/9/10 RGB / RGB delta / RGB scale+A).
const ROW3: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, //
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, //
    12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
];

/// Row 4 (8 colour integers, e.g. CEM 12/13 RGBA / RGBA delta).
const ROW4: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, //
    2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, //
    8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, //
    14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
];

/// Reference `quant_mode_table[integer_count / 2][color_bits]`: given the
/// number of colour integers the CEM needs and the colour bit budget, return
/// the `astcenc` colour quant level, or a value below [`QUANT_6`] (the `-1`
/// sentinel, or an under-budget level) meaning "error block".
///
/// `integer_count` is one of `2, 4, 6, 8` (two per endpoint-class step). A
/// colour-bits value past the end of the table saturates at [`QUANT_256`],
/// matching the reference's trailing fill.
#[inline]
pub(super) fn color_quant_level(integer_count: u32, color_bits: usize) -> i8 {
    let row: &[i8; 128] = match integer_count >> 1 {
        1 => &ROW1,
        2 => &ROW2,
        3 => &ROW3,
        4 => &ROW4,
        _ => &ROW0,
    };
    *row.get(color_bits).unwrap_or(&QUANT_256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row3_matches_the_gpu_proven_cem8_values() {
        // color_bits 69 (mode 67 weight budget) -> QUANT_256 identity.
        assert_eq!(color_quant_level(6, 69), QUANT_256);
        // color_bits 47 (mode 578) -> level 19 = QUANT_192.
        assert_eq!(color_quant_level(6, 47), 19);
        // color_bits 15 -> QUANT_5 (3), below QUANT_6: error block.
        assert_eq!(color_quant_level(6, 15), 3);
        // color_bits 5 -> -1 sentinel.
        assert_eq!(color_quant_level(6, 5), -1);
    }

    #[test]
    fn out_of_range_bits_saturate_at_quant256() {
        assert_eq!(color_quant_level(6, 200), QUANT_256);
        assert_eq!(color_quant_level(2, 999), QUANT_256);
    }

    #[test]
    fn each_row_is_monotonic_nondecreasing() {
        for ic in [2u32, 4, 6, 8] {
            let mut prev = i8::MIN;
            for bits in 0..128usize {
                let v = color_quant_level(ic, bits);
                assert!(v >= prev, "row {ic} not monotonic at {bits}");
                prev = v;
            }
        }
    }

    #[test]
    fn fewer_integers_never_need_more_bits() {
        // For any budget, a configuration with fewer colour integers can encode
        // at least as high a quant level as one with more integers.
        for bits in 0..128usize {
            let r1 = color_quant_level(2, bits);
            let r2 = color_quant_level(4, bits);
            let r3 = color_quant_level(6, bits);
            let r4 = color_quant_level(8, bits);
            assert!(r1 >= r2 && r2 >= r3 && r3 >= r4, "ordering at {bits}");
        }
    }
}
