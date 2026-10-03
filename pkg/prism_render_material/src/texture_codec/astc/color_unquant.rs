//! ASTC colour-endpoint unquantization tables.
//!
//! Once the colour endpoints are read out of the block as a Bounded Integer
//! Sequence (BISE), each decoded "scrambled packed quant" integer is mapped to
//! an 8-bit colour component through a per-quant-level lookup table. The ARM
//! `astcenc` reference decoder calls these
//! `color_scrambled_pquant_to_uquant_tables` (`astcenc_quantization.cpp`,
//! Apache-2.0): they fold *unscramble* and *unquantize* into a single indexed
//! read, so the raw `decode_ise` output (`low | (digit << bits)`) indexes them
//! directly -- no separate unscramble step is required. This is the same packed
//! representation already consumed, and GPU-proven, on the weight path.
//!
//! The tables are indexed by a *colour quant level index* in `0..=16`, where
//! index 0 corresponds to the `astcenc` enum value `QUANT_6` and index 16 to
//! `QUANT_256`. `QUANT_256` is the 8-bit identity and is handled in code rather
//! than stored as a 256-entry table.
//!
//! Pure integer lookup -- no AI/ML path.

/// Number of quantisation levels for each colour quant level index `0..=16`
/// (`QUANT_6 ..= QUANT_256`). Fed to `IseRange::from_num_levels` to classify
/// the colour integer sequence.
const NUM_LEVELS: [u32; 17] = [
    6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256,
];

// `color_scrambled_pquant_to_uquant_q*` tables, verbatim from the astcenc
// decoder (indices 0..=15 -> QUANT_6 ..= QUANT_192). QUANT_256 is identity.

const Q6: [u8; 6] = [0, 255, 51, 204, 102, 153];

const Q8: [u8; 8] = [0, 36, 73, 109, 146, 182, 219, 255];

const Q10: [u8; 10] = [0, 255, 28, 227, 56, 199, 84, 171, 113, 142];

const Q12: [u8; 12] = [0, 255, 69, 186, 23, 232, 92, 163, 46, 209, 116, 139];

const Q16: [u8; 16] = [
    0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255,
];

const Q20: [u8; 20] = [
    0, 255, 67, 188, 13, 242, 80, 175, 27, 228, 94, 161, 40, 215, 107, 148, 54, 201, 121, 134,
];

const Q24: [u8; 24] = [
    0, 255, 33, 222, 66, 189, 99, 156, 11, 244, 44, 211, 77, 178, 110, 145, 22, 233, 55, 200, 88,
    167, 121, 134,
];

const Q32: [u8; 32] = [
    0, 8, 16, 24, 33, 41, 49, 57, 66, 74, 82, 90, 99, 107, 115, 123, 132, 140, 148, 156, 165, 173,
    181, 189, 198, 206, 214, 222, 231, 239, 247, 255,
];

const Q40: [u8; 40] = [
    0, 255, 32, 223, 65, 190, 97, 158, 6, 249, 39, 216, 71, 184, 104, 151, 13, 242, 45, 210, 78,
    177, 110, 145, 19, 236, 52, 203, 84, 171, 117, 138, 26, 229, 58, 197, 91, 164, 123, 132,
];

const Q48: [u8; 48] = [
    0, 255, 16, 239, 32, 223, 48, 207, 65, 190, 81, 174, 97, 158, 113, 142, 5, 250, 21, 234, 38,
    217, 54, 201, 70, 185, 86, 169, 103, 152, 119, 136, 11, 244, 27, 228, 43, 212, 59, 196, 76,
    179, 92, 163, 108, 147, 124, 131,
];

const Q64: [u8; 64] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 65, 69, 73, 77, 81, 85, 89, 93,
    97, 101, 105, 109, 113, 117, 121, 125, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170,
    174, 178, 182, 186, 190, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247,
    251, 255,
];

const Q80: [u8; 80] = [
    0, 255, 16, 239, 32, 223, 48, 207, 64, 191, 80, 175, 96, 159, 112, 143, 3, 252, 19, 236, 35,
    220, 51, 204, 67, 188, 83, 172, 100, 155, 116, 139, 6, 249, 22, 233, 38, 217, 54, 201, 71, 184,
    87, 168, 103, 152, 119, 136, 9, 246, 25, 230, 42, 213, 58, 197, 74, 181, 90, 165, 106, 149,
    122, 133, 13, 242, 29, 226, 45, 210, 61, 194, 77, 178, 93, 162, 109, 146, 125, 130,
];

const Q96: [u8; 96] = [
    0, 255, 8, 247, 16, 239, 24, 231, 32, 223, 40, 215, 48, 207, 56, 199, 64, 191, 72, 183, 80,
    175, 88, 167, 96, 159, 104, 151, 112, 143, 120, 135, 2, 253, 10, 245, 18, 237, 26, 229, 35,
    220, 43, 212, 51, 204, 59, 196, 67, 188, 75, 180, 83, 172, 91, 164, 99, 156, 107, 148, 115,
    140, 123, 132, 5, 250, 13, 242, 21, 234, 29, 226, 37, 218, 45, 210, 53, 202, 61, 194, 70, 185,
    78, 177, 86, 169, 94, 161, 102, 153, 110, 145, 118, 137, 126, 129,
];

const Q128: [u8; 128] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
    50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96,
    98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 129, 131, 133, 135,
    137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173,
    175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211,
    213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249,
    251, 253, 255,
];

const Q160: [u8; 160] = [
    0, 255, 8, 247, 16, 239, 24, 231, 32, 223, 40, 215, 48, 207, 56, 199, 64, 191, 72, 183, 80,
    175, 88, 167, 96, 159, 104, 151, 112, 143, 120, 135, 1, 254, 9, 246, 17, 238, 25, 230, 33, 222,
    41, 214, 49, 206, 57, 198, 65, 190, 73, 182, 81, 174, 89, 166, 97, 158, 105, 150, 113, 142,
    121, 134, 3, 252, 11, 244, 19, 236, 27, 228, 35, 220, 43, 212, 51, 204, 59, 196, 67, 188, 75,
    180, 83, 172, 91, 164, 99, 156, 107, 148, 115, 140, 123, 132, 4, 251, 12, 243, 20, 235, 28,
    227, 36, 219, 44, 211, 52, 203, 60, 195, 68, 187, 76, 179, 84, 171, 92, 163, 100, 155, 108,
    147, 116, 139, 124, 131, 6, 249, 14, 241, 22, 233, 30, 225, 38, 217, 46, 209, 54, 201, 62, 193,
    70, 185, 78, 177, 86, 169, 94, 161, 102, 153, 110, 145, 118, 137, 126, 129,
];

const Q192: [u8; 192] = [
    0, 255, 4, 251, 8, 247, 12, 243, 16, 239, 20, 235, 24, 231, 28, 227, 32, 223, 36, 219, 40, 215,
    44, 211, 48, 207, 52, 203, 56, 199, 60, 195, 64, 191, 68, 187, 72, 183, 76, 179, 80, 175, 84,
    171, 88, 167, 92, 163, 96, 159, 100, 155, 104, 151, 108, 147, 112, 143, 116, 139, 120, 135,
    124, 131, 1, 254, 5, 250, 9, 246, 13, 242, 17, 238, 21, 234, 25, 230, 29, 226, 33, 222, 37,
    218, 41, 214, 45, 210, 49, 206, 53, 202, 57, 198, 61, 194, 65, 190, 69, 186, 73, 182, 77, 178,
    81, 174, 85, 170, 89, 166, 93, 162, 97, 158, 101, 154, 105, 150, 109, 146, 113, 142, 117, 138,
    121, 134, 125, 130, 2, 253, 6, 249, 10, 245, 14, 241, 18, 237, 22, 233, 26, 229, 30, 225, 34,
    221, 38, 217, 42, 213, 46, 209, 50, 205, 54, 201, 58, 197, 62, 193, 66, 189, 70, 185, 74, 181,
    78, 177, 82, 173, 86, 169, 90, 165, 94, 161, 98, 157, 102, 153, 106, 149, 110, 145, 114, 141,
    118, 137, 122, 133, 126, 129,
];

/// The sixteen non-identity colour unquant tables, indexed `0..=15`
/// (`QUANT_6 ..= QUANT_192`). Index 16 (`QUANT_256`) is identity, handled in
/// [`unquant_color`].
const TABLES: [&[u8]; 16] = [
    &Q6, &Q8, &Q10, &Q12, &Q16, &Q20, &Q24, &Q32, &Q40, &Q48, &Q64, &Q80, &Q96, &Q128, &Q160, &Q192,
];

/// Number of quantisation levels for colour quant level `level_index`
/// (`0..=16`, i.e. `QUANT_* - QUANT_6`). Panics in debug on out-of-range input;
/// callers derive the index from the reference quant-mode table so it is always
/// within `0..=16`.
#[inline]
pub(super) fn color_quant_num_levels(level_index: usize) -> u32 {
    NUM_LEVELS[level_index]
}

/// Unquantize one "scrambled packed quant" colour integer `packed` at colour
/// quant level `level_index` (`0..=16`, `QUANT_6 ..= QUANT_256`) to an 8-bit
/// colour component.
///
/// `QUANT_256` (index 16) is the identity; all other levels index the
/// corresponding `color_scrambled_pquant_to_uquant` table. `packed` is bounded
/// below the table length by construction of the matching `IseRange`.
#[inline]
pub(super) fn unquant_color(level_index: usize, packed: u8) -> u8 {
    if level_index >= 16 {
        // QUANT_256 (and any higher index) is the 8-bit identity.
        return packed;
    }
    TABLES[level_index][packed as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lengths_match_level_counts() {
        for (idx, table) in TABLES.iter().enumerate() {
            assert_eq!(
                table.len() as u32,
                NUM_LEVELS[idx],
                "table {idx} length must equal its quant level count"
            );
        }
        // QUANT_256 identity reports 256 levels without storing a table.
        assert_eq!(NUM_LEVELS[16], 256);
    }

    #[test]
    fn quant256_is_the_identity() {
        for v in 0..=255u8 {
            assert_eq!(unquant_color(16, v), v);
        }
    }

    #[test]
    fn quant16_is_even_17_step_replication() {
        // QUANT_16 (index 4) replicates the 4-bit value into both nibbles:
        // out = v * 17.
        for v in 0..16u8 {
            assert_eq!(unquant_color(4, v), v.wrapping_mul(17));
        }
    }

    #[test]
    fn endpoints_hit_the_full_01_scale() {
        // Every table must contain the exact black and white endpoints so a
        // fully saturated encode decodes to 0 / 255.
        for table in TABLES {
            assert!(table.contains(&0), "table missing 0 endpoint");
            assert!(table.contains(&255), "table missing 255 endpoint");
        }
    }

    #[test]
    fn num_levels_lookup_matches_table() {
        assert_eq!(color_quant_num_levels(0), 6);
        assert_eq!(color_quant_num_levels(15), 192);
        assert_eq!(color_quant_num_levels(16), 256);
    }
}
