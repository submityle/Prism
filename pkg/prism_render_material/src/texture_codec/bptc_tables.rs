//! Shared BPTC (block-partitioned texture compression) constant tables.
//!
//! These partition / anchor / interpolation-weight tables are defined once by
//! the Khronos Data Format Specification and are reused across BC7 (LDR) and
//! BC6H (HDR) block decoders. BC7 historically carried private copies in
//! `bc7.rs`; the BC6H two-subset decoder needs the identical two-subset
//! partition and anchor tables plus the 3-bit interpolation weights, so they
//! live here as `pub(crate)` to avoid divergent duplicates.
//!
//! Texel layout is row-major: texel index `t = y * 4 + x` for `x, y in 0..4`.

/// 2-bit index interpolation weights (Khronos `aWeight2`), in 1/64 units.
pub(crate) const WEIGHT2: [u32; 4] = [0, 21, 43, 64];

/// 3-bit index interpolation weights (Khronos `aWeight3`), in 1/64 units.
/// BC6H two-subset modes and BC7 mode 0/2 index with these.
pub(crate) const WEIGHT3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];

/// 4-bit index interpolation weights (Khronos `aWeight4`), in 1/64 units.
pub(crate) const WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// BPTC 2-subset partition table (Khronos Data Format Spec, Table "Partition
/// Table for 2 Subsets"). `BPTC_PARTITIONS_2[p][t]` is the subset (`0` or `1`)
/// of texel `t = y*4 + x` under partition `p in 0..64`. Shared by BC7 modes
/// 1/3/7 and BC6H's two-region modes 1-10. Every row has `partition[0] == 0`,
/// so texel 0 is always subset 0's anchor.
#[rustfmt::skip]
pub(crate) const BPTC_PARTITIONS_2: [[u8; 16]; 64] = [
    [0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1],
    [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
    [0,1,1,1, 0,1,1,1, 0,1,1,1, 0,1,1,1],
    [0,0,0,1, 0,0,1,1, 0,0,1,1, 0,1,1,1],
    [0,0,0,0, 0,0,0,1, 0,0,0,1, 0,0,1,1],
    [0,0,1,1, 0,1,1,1, 0,1,1,1, 1,1,1,1],
    [0,0,0,1, 0,0,1,1, 0,1,1,1, 1,1,1,1],
    [0,0,0,0, 0,0,0,1, 0,0,1,1, 0,1,1,1],
    [0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0,1,1],
    [0,0,1,1, 0,1,1,1, 1,1,1,1, 1,1,1,1],
    [0,0,0,0, 0,0,0,1, 0,1,1,1, 1,1,1,1],
    [0,0,0,0, 0,0,0,0, 0,0,0,1, 0,1,1,1],
    [0,0,0,1, 0,1,1,1, 1,1,1,1, 1,1,1,1],
    [0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1],
    [0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1],
    [0,0,0,0, 1,0,0,0, 1,1,1,0, 1,1,1,1],
    [0,1,1,1, 0,0,0,1, 0,0,0,0, 0,0,0,0],
    [0,0,0,0, 0,0,0,0, 1,0,0,0, 1,1,1,0],
    [0,1,1,1, 0,0,1,1, 0,0,0,1, 0,0,0,0],
    [0,0,1,1, 0,0,0,1, 0,0,0,0, 0,0,0,0],
    [0,0,0,0, 1,0,0,0, 1,1,0,0, 1,1,1,0],
    [0,0,0,0, 0,0,0,0, 1,0,0,0, 1,1,0,0],
    [0,1,1,1, 0,0,1,1, 0,0,1,1, 0,0,0,1],
    [0,0,1,1, 0,0,0,1, 0,0,0,1, 0,0,0,0],
    [0,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0],
    [0,1,1,0, 0,1,1,0, 0,1,1,0, 0,1,1,0],
    [0,0,1,1, 0,1,1,0, 0,1,1,0, 1,1,0,0],
    [0,0,0,1, 0,1,1,1, 1,1,1,0, 1,0,0,0],
    [0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0],
    [0,1,1,1, 0,0,0,1, 1,0,0,0, 1,1,1,0],
    [0,0,1,1, 1,0,0,1, 1,0,0,1, 1,1,0,0],
    [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1],
    [0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1],
    [0,1,0,1, 1,0,1,0, 0,1,0,1, 1,0,1,0],
    [0,0,1,1, 0,0,1,1, 1,1,0,0, 1,1,0,0],
    [0,0,1,1, 1,1,0,0, 0,0,1,1, 1,1,0,0],
    [0,1,0,1, 0,1,0,1, 1,0,1,0, 1,0,1,0],
    [0,1,1,0, 1,0,0,1, 0,1,1,0, 1,0,0,1],
    [0,1,0,1, 1,0,1,0, 1,0,1,0, 0,1,0,1],
    [0,1,1,1, 0,0,1,1, 1,1,0,0, 1,1,1,0],
    [0,0,0,1, 0,0,1,1, 1,1,0,0, 1,0,0,0],
    [0,0,1,1, 0,0,1,0, 0,1,0,0, 1,1,0,0],
    [0,0,1,1, 1,0,1,1, 1,1,0,1, 1,1,0,0],
    [0,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,0],
    [0,0,1,1, 1,1,0,0, 1,1,0,0, 0,0,1,1],
    [0,1,1,0, 0,1,1,0, 1,0,0,1, 1,0,0,1],
    [0,0,0,0, 0,1,1,0, 0,1,1,0, 0,0,0,0],
    [0,1,0,0, 1,1,1,0, 0,1,0,0, 0,0,0,0],
    [0,0,1,0, 0,1,1,1, 0,0,1,0, 0,0,0,0],
    [0,0,0,0, 0,0,1,0, 0,1,1,1, 0,0,1,0],
    [0,0,0,0, 0,1,0,0, 1,1,1,0, 0,1,0,0],
    [0,1,1,0, 1,1,0,0, 1,0,0,1, 0,0,1,1],
    [0,0,1,1, 0,1,1,0, 1,1,0,0, 1,0,0,1],
    [0,1,1,0, 0,0,1,1, 1,0,0,1, 1,1,0,0],
    [0,0,1,1, 1,0,0,1, 1,1,0,0, 0,1,1,0],
    [0,1,1,0, 1,1,0,0, 1,1,0,0, 1,0,0,1],
    [0,1,1,0, 0,0,1,1, 0,0,1,1, 1,0,0,1],
    [0,1,1,1, 1,1,1,0, 1,0,0,0, 0,0,0,1],
    [0,0,0,1, 1,0,0,0, 1,1,1,0, 0,1,1,1],
    [0,0,0,0, 1,1,1,1, 0,0,1,1, 0,0,1,1],
    [0,0,1,1, 0,0,1,1, 1,1,1,1, 0,0,0,0],
    [0,0,1,0, 0,0,1,0, 1,1,1,0, 1,1,1,0],
    [0,1,0,0, 0,1,0,0, 0,1,1,1, 0,1,1,1],
];

/// BPTC 2-subset anchor table: `BPTC_ANCHORS_2[p]` is the texel index that
/// holds subset 1's anchor (fixed high-bit-zero) index under partition `p`.
/// Subset 0's anchor is always texel 0. (Khronos Data Format Spec, "Fixup"
/// / anchor index tables for 2 subsets.)
#[rustfmt::skip]
pub(crate) const BPTC_ANCHORS_2: [usize; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,
     2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,
     2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2,
    15, 15, 15, 15, 15,  2,  2, 15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_endpoints_are_exact() {
        assert_eq!(WEIGHT2[0], 0);
        assert_eq!(WEIGHT2[3], 64);
        assert_eq!(WEIGHT3[0], 0);
        assert_eq!(WEIGHT3[7], 64);
        assert_eq!(WEIGHT4[0], 0);
        assert_eq!(WEIGHT4[15], 64);
    }

    #[test]
    fn partition2_rows_start_at_subset_zero_and_are_binary() {
        for row in &BPTC_PARTITIONS_2 {
            assert_eq!(row[0], 0, "texel 0 must be subset 0 anchor");
            assert!(row.iter().all(|&s| s <= 1), "2-subset entries are 0 or 1");
            assert!(row.iter().any(|&s| s == 1), "subset 1 must appear");
        }
    }

    #[test]
    fn anchor2_indices_are_in_range_and_nonzero() {
        for (p, &a) in BPTC_ANCHORS_2.iter().enumerate() {
            assert!(a < 16, "anchor must be a valid texel index");
            assert_ne!(a, 0, "subset 1 anchor never collides with texel 0");
            // The anchor texel must actually belong to subset 1.
            assert_eq!(BPTC_PARTITIONS_2[p][a], 1, "anchor must sit in subset 1");
        }
    }
}
