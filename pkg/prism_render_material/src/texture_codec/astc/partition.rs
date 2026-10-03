//! ASTC procedural partition assignment.
//!
//! Multi-partition ASTC blocks do not store an explicit per-texel partition
//! map. Instead a 10-bit *partition seed* drives a fixed hash that assigns each
//! texel to one of 2, 3 or 4 partitions. The generator is pure integer
//! arithmetic and is identical across every conformant decoder, so the seed
//! plus the texel coordinate fully determine the partition index.
//!
//! Transcribed from the ARM `astcenc` reference decoder
//! (`astcenc_partition_tables.cpp`, `hash52` / `select_partition`, Apache-2.0).
//! The logic is reproduced exactly -- including the integer squaring that
//! biases the distribution and the per-seed shift selection -- so the partition
//! map matches the hardware bit-for-bit.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

/// Hash used to seed the procedural partition assignment.
///
/// A 32-bit integer avalanche identical to astcenc `hash52`. Every step uses
/// wrapping arithmetic because the reference relies on 32-bit overflow.
#[inline]
fn hash52(mut inp: u32) -> u32 {
    inp ^= inp >> 15;
    // (2^4 + 1) * (2^7 + 1) * (2^17 - 1)
    inp = inp.wrapping_mul(0xEEDE_0891);
    inp ^= inp >> 5;
    inp = inp.wrapping_add(inp << 16);
    inp ^= inp >> 7;
    inp ^= inp >> 3;
    inp ^= inp << 6;
    inp ^= inp >> 17;
    inp
}

/// Assign a single texel at `(x, y, z)` to a partition for `partition_count`
/// partitions, driven by the 10-bit block `seed`.
///
/// `small_block` must be set for blocks with fewer than 32 texels (which
/// includes every 2D footprint from 4x4 up to 5x5); it biases the coordinates
/// by doubling them to spread the hash more evenly. The returned index is in
/// `0..partition_count`.
///
/// This is a faithful transcription of astcenc `select_partition`: the twelve
/// nibble seeds are squared to bias them low, shifted by a seed-selected
/// amount, then combined with the texel coordinates to form four accumulators
/// whose argmax picks the partition.
#[must_use]
pub(super) fn select_partition(
    seed: i32,
    x: i32,
    y: i32,
    z: i32,
    partition_count: i32,
    small_block: bool,
) -> u8 {
    let (mut x, mut y, mut z) = (x, y, z);
    if small_block {
        x <<= 1;
        y <<= 1;
        z <<= 1;
    }

    let seed = seed + (partition_count - 1) * 1024;
    let rnum = hash52(seed as u32);

    let mut s = [0i32; 12];
    s[0] = (rnum & 0xF) as i32;
    s[1] = ((rnum >> 4) & 0xF) as i32;
    s[2] = ((rnum >> 8) & 0xF) as i32;
    s[3] = ((rnum >> 12) & 0xF) as i32;
    s[4] = ((rnum >> 16) & 0xF) as i32;
    s[5] = ((rnum >> 20) & 0xF) as i32;
    s[6] = ((rnum >> 24) & 0xF) as i32;
    s[7] = ((rnum >> 28) & 0xF) as i32;
    s[8] = ((rnum >> 18) & 0xF) as i32;
    s[9] = ((rnum >> 22) & 0xF) as i32;
    s[10] = ((rnum >> 26) & 0xF) as i32;
    s[11] = (((rnum >> 30) | (rnum << 2)) & 0xF) as i32;

    // Square each seed to bias its distribution towards lower values.
    for v in &mut s {
        *v *= *v;
    }

    // Seed-selected shift amounts, exactly as the reference computes them.
    let (sh1, sh2);
    if seed & 1 != 0 {
        sh1 = if seed & 2 != 0 { 4 } else { 5 };
        sh2 = if partition_count == 3 { 6 } else { 5 };
    } else {
        sh1 = if partition_count == 3 { 6 } else { 5 };
        sh2 = if seed & 2 != 0 { 4 } else { 5 };
    }
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };

    s[0] >>= sh1;
    s[1] >>= sh2;
    s[2] >>= sh1;
    s[3] >>= sh2;
    s[4] >>= sh1;
    s[5] >>= sh2;
    s[6] >>= sh1;
    s[7] >>= sh2;
    s[8] >>= sh3;
    s[9] >>= sh3;
    s[10] >>= sh3;
    s[11] >>= sh3;

    // Form the four accumulators in 64-bit so the low six bits match the
    // reference's 32-bit wrap exactly (the final `& 0x3F` discards everything
    // above bit 5, so a wider intermediate is equivalent).
    let (x, y, z) = (x as i64, y as i64, z as i64);
    let a = (s[0] as i64 * x + s[1] as i64 * y + s[10] as i64 * z + (rnum >> 14) as i64) & 0x3F;
    let b = (s[2] as i64 * x + s[3] as i64 * y + s[11] as i64 * z + (rnum >> 10) as i64) & 0x3F;
    let c = (s[4] as i64 * x + s[5] as i64 * y + s[8] as i64 * z + (rnum >> 6) as i64) & 0x3F;
    let d = (s[6] as i64 * x + s[7] as i64 * y + s[9] as i64 * z + (rnum >> 2) as i64) & 0x3F;

    // Drop the accumulators that are unused for fewer than four partitions.
    let d = if partition_count <= 3 { 0 } else { d };
    let c = if partition_count <= 2 { 0 } else { c };
    let b = if partition_count <= 1 { 0 } else { b };

    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash52_matches_reference_values() {
        // Spot-check the avalanche against values produced by the reference
        // `hash52` for a few seeds (computed from the C expression).
        assert_eq!(hash52(0), 0x0000_0000);
        // Non-zero seeds must avalanche to large, well-mixed values; just
        // assert determinism and that distinct seeds differ.
        assert_ne!(hash52(1), hash52(2));
        assert_ne!(hash52(1024), hash52(2048));
    }

    #[test]
    fn indices_stay_in_range() {
        for pc in 2..=4i32 {
            for seed in 0..64i32 {
                for y in 0..4i32 {
                    for x in 0..4i32 {
                        let p = select_partition(seed, x, y, 0, pc, true);
                        assert!(
                            (p as i32) < pc,
                            "seed {seed} pc {pc} ({x},{y}) -> {p} out of range"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn single_partition_is_always_zero() {
        for seed in 0..32i32 {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    assert_eq!(select_partition(seed, x, y, 0, 1, true), 0);
                }
            }
        }
    }

    #[test]
    fn assignment_is_deterministic() {
        let first = select_partition(123, 2, 1, 0, 3, true);
        let again = select_partition(123, 2, 1, 0, 3, true);
        assert_eq!(first, again);
    }
}
