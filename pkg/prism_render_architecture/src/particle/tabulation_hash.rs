//! Tabulation hashing for `u64` keys.
//!
//! This module implements a simple tabulation (table) hash. A `u64` key is
//! split into its 8 constituent bytes; each byte indexes a dedicated lookup
//! table of 256 entries, and the selected entries are combined with `XOR`.
//!
//! The lookup tables are filled deterministically from a fixed seed using the
//! `splitmix64` pseudo-random step. Because generation is purely integer-based
//! (`wrapping` arithmetic, shifts, and `XOR`), the result is identical on any
//! `CPU` or `GPU` host and across repeated runs with the same seed.
//!
//! Tabulation hashing offers strong statistical quality (3-independence) while
//! remaining branch-free and cheap to evaluate, which makes it well suited for
//! hashing particle keys in rendering pipelines.

/// Fixed seed used by [`TabulationHasher::new_default`].
pub const FIXED_SEED: u64 = 0x243F_6A88_85A3_08D3;

/// Number of lookup tables, one per byte of a `u64` key.
const TABLE_COUNT: usize = 8;

/// Number of entries in each lookup table, one per possible byte value.
const TABLE_SIZE: usize = 256;

/// `splitmix64` increment constant (the fractional part of the golden ratio).
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First `splitmix64` mixing multiplier.
const SPLITMIX_MUL_A: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second `splitmix64` mixing multiplier.
const SPLITMIX_MUL_B: u64 = 0x94D0_49BB_1331_11EB;

/// Advance a `splitmix64` state in place and return the next output value.
///
/// The state is first advanced by a fixed additive constant, then the advanced
/// state is passed through two multiply-xorshift mixing rounds. All arithmetic
/// uses `wrapping` operations and bit shifts, so there is no overflow or
/// floating-point behavior.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL_A);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL_B);
    z ^ (z >> 31)
}

/// A tabulation hasher over `u64` keys.
///
/// The hasher owns 8 lookup tables of 256 `u64` entries each. Each byte of a
/// key selects one entry from the corresponding table, and all selected entries
/// are combined with `XOR` to produce the final hash.
pub struct TabulationHasher {
    tables: [[u64; TABLE_SIZE]; TABLE_COUNT],
}

impl TabulationHasher {
    /// Build a hasher whose tables are filled deterministically from `seed`.
    ///
    /// The tables are populated in order: table 0 entry 0, table 0 entry 1,
    /// ..., table 0 entry 255, table 1 entry 0, and so on, each value being the
    /// next output of `splitmix64` seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut tables = [[0u64; TABLE_SIZE]; TABLE_COUNT];
        for table in tables.iter_mut() {
            for slot in table.iter_mut() {
                *slot = splitmix64_next(&mut state);
            }
        }
        Self { tables }
    }

    /// Build a hasher using the module-wide [`FIXED_SEED`].
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(FIXED_SEED)
    }

    /// Hash a `u64` key by looking up each byte in its table and `XOR`-ing the
    /// selected entries together.
    #[must_use]
    pub fn hash(&self, key: u64) -> u64 {
        let mut acc = 0u64;
        for (i, table) in self.tables.iter().enumerate() {
            let byte = ((key >> (8 * i)) & 0xFF) as usize;
            acc ^= table[byte];
        }
        acc
    }
}

/// Convenience function hashing `key` with a [`TabulationHasher::new_default`].
#[must_use]
pub fn tabulation_hash(key: u64) -> u64 {
    TabulationHasher::new_default().hash(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn fixed_seed_constant_matches_spec() {
        const { assert!(FIXED_SEED == 0x243F_6A88_85A3_08D3) };
    }

    #[test]
    fn table_dimensions_are_correct() {
        let h = TabulationHasher::new_default();
        assert!(h.tables.len() == 8);
        assert!(h.tables[0].len() == 256);
    }

    #[test]
    fn reference_table_0_0() {
        let h = TabulationHasher::new_default();
        assert!(h.tables[0][0] == 0x2cb0_f69f_4abe_a221);
    }

    #[test]
    fn reference_table_0_1() {
        let h = TabulationHasher::new_default();
        assert!(h.tables[0][1] == 0x9417_0347_2314_8989);
    }

    #[test]
    fn reference_table_7_255() {
        let h = TabulationHasher::new_default();
        assert!(h.tables[7][255] == 0x71ef_020b_b71b_64ac);
    }

    #[test]
    fn reference_hash_0() {
        assert!(tabulation_hash(0) == 0x0bd7_67b6_7ee9_057c);
    }

    #[test]
    fn reference_hash_1() {
        assert!(tabulation_hash(1) == 0xb370_926e_1743_2ed4);
    }

    #[test]
    fn reference_hash_deadbeef() {
        assert!(tabulation_hash(0xdead_beef) == 0x8184_f48e_9879_a4bc);
    }

    #[test]
    fn reference_hash_all_ones() {
        assert!(tabulation_hash(0xffff_ffff_ffff_ffff) == 0x2975_fbff_ace4_4a34);
    }

    #[test]
    fn reference_hash_byte_ramp() {
        assert!(tabulation_hash(0x0102_0304_0506_0708) == 0x096f_afb1_8214_85b1);
    }

    #[test]
    fn hash_zero_equals_xor_of_zero_entries() {
        let h = TabulationHasher::new_default();
        let mut expected = 0u64;
        for table in h.tables.iter() {
            expected ^= table[0];
        }
        assert!(h.hash(0) == expected);
    }

    #[test]
    fn free_function_matches_method() {
        let h = TabulationHasher::new_default();
        for key in [0u64, 1, 2, 42, 0xdead_beef, u64::MAX] {
            assert!(tabulation_hash(key) == h.hash(key));
        }
    }

    #[test]
    fn deterministic_tables_same_seed() {
        let a = TabulationHasher::new(FIXED_SEED);
        let b = TabulationHasher::new(FIXED_SEED);
        assert!(a.tables == b.tables);
    }

    #[test]
    fn deterministic_tables_arbitrary_seed() {
        let a = TabulationHasher::new(0x1234_5678_9abc_def0);
        let b = TabulationHasher::new(0x1234_5678_9abc_def0);
        assert!(a.tables == b.tables);
    }

    #[test]
    fn deterministic_hash_repeated() {
        let h = TabulationHasher::new_default();
        let first = h.hash(0x0bad_c0de);
        let second = h.hash(0x0bad_c0de);
        assert!(first == second);
    }

    #[test]
    fn new_default_equals_new_fixed_seed() {
        let a = TabulationHasher::new_default();
        let b = TabulationHasher::new(FIXED_SEED);
        assert!(a.tables == b.tables);
        for key in [0u64, 1, 7, 999, u64::MAX] {
            assert!(a.hash(key) == b.hash(key));
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = TabulationHasher::new(1);
        let b = TabulationHasher::new(2);
        assert!(a.tables != b.tables);
    }

    #[test]
    fn splitmix64_first_output_matches() {
        let mut state = FIXED_SEED;
        let first = splitmix64_next(&mut state);
        assert!(first == 0x2cb0_f69f_4abe_a221);
    }

    #[test]
    fn splitmix64_second_output_matches() {
        let mut state = FIXED_SEED;
        let _ = splitmix64_next(&mut state);
        let second = splitmix64_next(&mut state);
        assert!(second == 0x9417_0347_2314_8989);
    }

    #[test]
    fn splitmix64_advances_state() {
        let mut state = 0u64;
        let _ = splitmix64_next(&mut state);
        assert!(state == SPLITMIX_GAMMA);
    }

    #[test]
    fn splitmix64_is_pure_for_same_state() {
        let mut s1 = 0x5555_5555_5555_5555;
        let mut s2 = 0x5555_5555_5555_5555;
        assert!(splitmix64_next(&mut s1) == splitmix64_next(&mut s2));
    }

    #[test]
    fn tables_fill_order_is_table_major() {
        // Entry (t, j) is produced at stream index (t * 256 + j).
        let h = TabulationHasher::new_default();
        let mut state = FIXED_SEED;
        let mut stream = Vec::new();
        for _ in 0..(TABLE_COUNT * TABLE_SIZE) {
            stream.push(splitmix64_next(&mut state));
        }
        for t in 0..TABLE_COUNT {
            for j in 0..TABLE_SIZE {
                assert!(h.tables[t][j] == stream[(t * TABLE_SIZE) + j]);
            }
        }
    }

    #[test]
    fn single_byte_keys_select_first_table() {
        let h = TabulationHasher::new_default();
        for byte in 0u64..256 {
            let mut expected = h.tables[0][byte as usize];
            for table in h.tables.iter().skip(1) {
                expected ^= table[0];
            }
            assert!(h.hash(byte) == expected);
        }
    }

    #[test]
    fn second_byte_keys_select_second_table() {
        let h = TabulationHasher::new_default();
        for byte in 0u64..256 {
            let key = byte << 8;
            let mut expected = h.tables[1][byte as usize];
            for (i, table) in h.tables.iter().enumerate() {
                if i != 1 {
                    expected ^= table[0];
                }
            }
            assert!(h.hash(key) == expected);
        }
    }

    #[test]
    fn top_byte_keys_select_last_table() {
        let h = TabulationHasher::new_default();
        for byte in 0u64..256 {
            let key = byte << 56;
            let mut expected = h.tables[7][byte as usize];
            for (i, table) in h.tables.iter().enumerate() {
                if i != 7 {
                    expected ^= table[0];
                }
            }
            assert!(h.hash(key) == expected);
        }
    }

    #[test]
    fn hash_is_full_byte_xor_decomposition() {
        let h = TabulationHasher::new_default();
        let keys = [
            0x0102_0304_0506_0708u64,
            0xfeed_face_dead_beef,
            0x1357_9bdf_0246_8ace,
        ];
        for &key in keys.iter() {
            let mut expected = 0u64;
            for i in 0..TABLE_COUNT {
                let byte = ((key >> (8 * i)) & 0xFF) as usize;
                expected ^= h.tables[i][byte];
            }
            assert!(h.hash(key) == expected);
        }
    }

    #[test]
    fn xor_linearity_over_byte_components() {
        // The hash of a key equals the XOR of hashing each isolated byte lane
        // combined with the zero-key contribution of the other lanes.
        let h = TabulationHasher::new_default();
        let key = 0xdead_beef_cafe_babeu64;
        let mut expected = 0u64;
        for i in 0..TABLE_COUNT {
            let masked = key & (0xFFu64 << (8 * i));
            // Only lane i contributes a non-zero byte; combine lane entry.
            let byte = ((masked >> (8 * i)) & 0xFF) as usize;
            expected ^= h.tables[i][byte];
        }
        assert!(h.hash(key) == expected);
    }

    #[test]
    fn distinct_small_keys_mostly_distinct() {
        let h = TabulationHasher::new_default();
        let mut seen = Vec::new();
        for key in 0u64..4096 {
            seen.push(h.hash(key));
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert!(seen.len() == before);
    }

    #[test]
    fn no_collisions_on_large_sample() {
        let h = TabulationHasher::new_default();
        let mut seen = Vec::new();
        let mut key = 0x1u64;
        for _ in 0..20_000 {
            seen.push(h.hash(key));
            key = key.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert!(seen.len() == before);
    }

    #[test]
    fn sequential_keys_no_collisions() {
        let h = TabulationHasher::new_default();
        let mut seen = Vec::new();
        for key in 0u64..10_000 {
            seen.push(h.hash(key));
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert!(seen.len() == before);
    }

    #[test]
    fn high_bit_keys_no_collisions() {
        let h = TabulationHasher::new_default();
        let mut seen = Vec::new();
        for i in 0u64..10_000 {
            let key = (i << 32) | i;
            seen.push(h.hash(key));
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert!(seen.len() == before);
    }

    #[test]
    fn adjacent_keys_differ() {
        let h = TabulationHasher::new_default();
        for key in 0u64..5000 {
            assert!(h.hash(key) != h.hash(key + 1));
        }
    }

    #[test]
    fn single_bit_flips_change_hash() {
        let h = TabulationHasher::new_default();
        let base = 0x0u64;
        for bit in 0..64u32 {
            let flipped = base ^ (1u64 << bit);
            assert!(h.hash(base) != h.hash(flipped));
        }
    }

    #[test]
    fn independent_hashers_agree() {
        let a = TabulationHasher::new(777);
        let b = TabulationHasher::new(777);
        for key in [0u64, 1, 100, 0xabc_def, u64::MAX, 0x8000_0000_0000_0000] {
            assert!(a.hash(key) == b.hash(key));
        }
    }

    #[test]
    fn hash_covers_all_tables_for_max_key() {
        let h = TabulationHasher::new_default();
        let mut expected = 0u64;
        for table in h.tables.iter() {
            expected ^= table[255];
        }
        assert!(h.hash(u64::MAX) == expected);
    }

    #[test]
    fn zero_seed_is_deterministic() {
        let a = TabulationHasher::new(0);
        let b = TabulationHasher::new(0);
        assert!(a.tables == b.tables);
        assert!(a.hash(123) == b.hash(123));
    }

    #[test]
    fn max_seed_is_deterministic() {
        let a = TabulationHasher::new(u64::MAX);
        let b = TabulationHasher::new(u64::MAX);
        assert!(a.tables == b.tables);
        assert!(a.hash(456) == b.hash(456));
    }

    #[test]
    fn table_entries_not_all_zero() {
        let h = TabulationHasher::new_default();
        let mut any_nonzero = false;
        for table in h.tables.iter() {
            for &entry in table.iter() {
                if entry != 0 {
                    any_nonzero = true;
                }
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn each_table_has_distinct_entries() {
        let h = TabulationHasher::new_default();
        for table in h.tables.iter() {
            let mut entries: Vec<u64> = table.to_vec();
            entries.sort_unstable();
            let before = entries.len();
            entries.dedup();
            assert!(entries.len() == before);
        }
    }

    #[test]
    fn free_function_reference_vectors() {
        assert!(tabulation_hash(0) == 0x0bd7_67b6_7ee9_057c);
        assert!(tabulation_hash(1) == 0xb370_926e_1743_2ed4);
        assert!(tabulation_hash(0x0102_0304_0506_0708) == 0x096f_afb1_8214_85b1);
    }

    #[test]
    fn hashing_is_idempotent_across_instances() {
        let a = TabulationHasher::new_default();
        let b = TabulationHasher::new_default();
        for key in [7u64, 8, 9, 0xffff, 0x1_0000_0000] {
            assert!(a.hash(key) == b.hash(key));
        }
    }
}
