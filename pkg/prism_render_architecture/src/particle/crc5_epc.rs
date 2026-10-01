//! `CRC-5/EPC`: the 5-bit cyclic redundancy check used by the EPC Gen2 (`RFID`)
//! air-interface protocol, implemented here as a pure-integer, forward
//! (`MSB`-first) checksum for cheaply validating short control fields and small
//! particle metadata records (design § integrity checks).
//!
//! The algorithm follows the standard `CRC-5/EPC` parameter set: `width = 5`,
//! polynomial `poly = 0x09`, initial value `init = 0x09`, non-reflected input
//! (`refin = false`), non-reflected output (`refout = false`), and final xor
//! `xorout = 0x00`. Because neither the input nor the output is reflected, the
//! computation runs most-significant-bit first using the nominal polynomial.
//! Each input byte is folded into the running remainder across all eight of its
//! bits, and the final remainder is combined with `xorout` and masked to five
//! bits.
//!
//! Every operation is an integer xor, shift, or mask on a [`u32`] accumulator;
//! there are no floating-point, heap, or unsafe operations, so the routine is
//! usable in a `no_std` context. The reference check value for this variant is
//! `crc5_epc(b"123456789") == 0x00`, and the empty input yields the initial
//! value `0x09` (there is no final xor to cancel it).
//!
//! Scope: this is a short error-detection code, not a hash and not a message
//! authentication code. `CRC-5/EPC` is *not* cryptographically secure and
//! collisions are trivial to construct on purpose, so it must never be used to
//! authenticate data or guard against a malicious adversary. It is meant only
//! for catching accidental corruption in short fields.

/// The bit width of the `CRC-5/EPC` checksum.
const WIDTH: u32 = 5;
/// The nominal (`MSB`-first) generator polynomial, `0x09`.
const POLY: u32 = 0x9;
/// The initial remainder value, `0x09`.
const INIT: u32 = 0x9;
/// The final xor applied to the finished remainder, `0x00`.
const XOROUT: u32 = 0x0;
/// Whether each input byte is bit-reflected before being folded in.
const REFLECT_IN: bool = false;
/// Whether the finished remainder is bit-reflected before the final xor.
const REFLECT_OUT: bool = false;
/// The five-bit mask applied to the running and finished remainder.
const MASK: u32 = 0x1f;
/// The most-significant bit of the five-bit remainder register.
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflects the low `bits` of `value`, reversing their order.
const fn reflect(value: u32, bits: u32) -> u32 {
    let mut out = 0u32;
    let mut i = 0u32;
    while i < bits {
        if ((value >> i) & 1) != 0 {
            out |= 1u32 << (bits - 1 - i);
        }
        i += 1;
    }
    out
}

/// Computes the `CRC-5/EPC` checksum of `data` in a single call.
///
/// The empty slice yields the initial value `0x09`, and `crc5_epc(b"123456789")`
/// is the reference check value `0x00` for this variant. The result is always
/// within the five-bit range `0..=0x1F`.
#[must_use]
pub fn crc5_epc(data: &[u8]) -> u8 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(byte as u32, 8)
        } else {
            byte as u32
        };
        let mut i = 0u32;
        while i < 8 {
            let bit = (b >> (7 - i)) & 1;
            let msb = u32::from((reg & TOPBIT) != 0);
            reg = (reg << 1) & MASK;
            if (msb ^ bit) != 0 {
                reg ^= POLY;
            }
            i += 1;
        }
        idx += 1;
    }
    if REFLECT_OUT {
        reg = reflect(reg, WIDTH);
    }
    ((reg ^ XOROUT) & MASK) as u8
}

#[cfg(test)]
mod tests {
    use super::crc5_epc;

    // --- 24 anchor vectors (reference truth) ---

    #[test]
    fn anchor_80() {
        assert!(crc5_epc(&[0x80]) == 0x0b);
    }

    #[test]
    fn anchor_12345678() {
        assert!(crc5_epc(&[0x12, 0x34, 0x56, 0x78]) == 0x0b);
    }

    #[test]
    fn anchor_empty() {
        assert!(crc5_epc(b"") == 0x09);
    }

    #[test]
    fn anchor_z00() {
        assert!(crc5_epc(&[0x00]) == 0x15);
    }

    #[test]
    fn anchor_ff() {
        assert!(crc5_epc(&[0xff]) == 0x06);
    }

    #[test]
    fn anchor_a() {
        assert!(crc5_epc(b"a") == 0x00);
    }

    #[test]
    fn anchor_ab() {
        assert!(crc5_epc(b"ab") == 0x0e);
    }

    #[test]
    fn anchor_abc() {
        assert!(crc5_epc(b"abc") == 0x06);
    }

    #[test]
    fn anchor_two_zeros() {
        assert!(crc5_epc(&[0, 0]) == 0x17);
    }

    #[test]
    fn anchor_01() {
        assert!(crc5_epc(&[0x01]) == 0x1c);
    }

    #[test]
    fn anchor_02() {
        assert!(crc5_epc(&[0x02]) == 0x07);
    }

    #[test]
    fn anchor_7f() {
        assert!(crc5_epc(&[0x7f]) == 0x18);
    }

    #[test]
    fn anchor_aa_55() {
        assert!(crc5_epc(&[0xaa, 0x55]) == 0x08);
    }

    #[test]
    fn anchor_55_aa() {
        assert!(crc5_epc(&[0x55, 0xaa]) == 0x02);
    }

    #[test]
    fn anchor_deadbeef() {
        assert!(crc5_epc(&[0xde, 0xad, 0xbe, 0xef]) == 0x0a);
    }

    #[test]
    fn anchor_hello() {
        assert!(crc5_epc(b"Hello") == 0x04);
    }

    #[test]
    fn anchor_fox() {
        assert!(crc5_epc(b"The quick brown fox") == 0x05);
    }

    #[test]
    fn anchor_four_zeros() {
        assert!(crc5_epc(&[0u8; 4]) == 0x12);
    }

    #[test]
    fn anchor_four_ff() {
        assert!(crc5_epc(&[0xffu8; 4]) == 0x1b);
    }

    #[test]
    fn anchor_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc5_epc(&data) == 0x16);
    }

    #[test]
    fn anchor_check() {
        assert!(crc5_epc(b"123456789") == 0x00);
    }

    #[test]
    fn anchor_a5_1000() {
        assert!(crc5_epc(&[0xA5u8; 1000]) == 0x0e);
    }

    #[test]
    fn anchor_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc5_epc(&data) == 0x02);
    }

    #[test]
    fn anchor_b() {
        assert!(crc5_epc(b"b") == 0x1b);
    }

    // --- structural / property tests ---

    #[test]
    fn determinism_repeated_calls() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        let first = crc5_epc(&data);
        let second = crc5_epc(&data);
        let third = crc5_epc(&data);
        assert!(first == second);
        assert!(second == third);
    }

    #[test]
    fn order_sensitivity() {
        let forward = crc5_epc(&[0x01, 0x02, 0x03]);
        let reversed = crc5_epc(&[0x03, 0x02, 0x01]);
        assert!(forward != reversed);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc5_epc(b"a") != crc5_epc(b"b"));
    }

    #[test]
    fn prefix_differs_from_full() {
        let prefix = crc5_epc(b"ab");
        let full = crc5_epc(b"abc");
        assert!(prefix != full);
    }

    #[test]
    fn length_sensitive_zeros() {
        let one = crc5_epc(&[0x00]);
        let two = crc5_epc(&[0x00, 0x00]);
        let three = crc5_epc(&[0x00, 0x00, 0x00]);
        assert!(one != two);
        assert!(two != three);
    }

    #[test]
    fn empty_equals_init() {
        assert!(crc5_epc(b"") == 0x09);
    }

    #[test]
    fn check_constant_is_zero() {
        assert!(crc5_epc(b"123456789") == 0x00);
    }

    #[test]
    fn all_results_in_five_bit_range() {
        let mut byte = 0u16;
        while byte < 256 {
            let r = crc5_epc(&[byte as u8]);
            assert!((0..=0x1f).contains(&r));
            byte += 1;
        }
    }

    #[test]
    fn empty_in_range() {
        let r = crc5_epc(b"");
        assert!((0..=0x1f).contains(&r));
    }

    #[test]
    fn long_input_in_range() {
        let r = crc5_epc(&[0xA5u8; 1000]);
        assert!((0..=0x1f).contains(&r));
    }

    #[test]
    fn single_byte_256_values_distinguishable() {
        // Any two distinct single bytes that happen to collide are allowed for a
        // 5-bit code, but the full set must span a sensible spread of values and
        // adjacent values should not all be identical. Here we assert that there
        // exist at least two distinct checksums across all 256 single bytes.
        let first = crc5_epc(&[0u8]);
        let mut found_distinct = false;
        let mut byte = 1u16;
        while byte < 256 {
            if crc5_epc(&[byte as u8]) != first {
                found_distinct = true;
            }
            byte += 1;
        }
        assert!(found_distinct);
    }

    #[test]
    fn pairwise_known_distinct_bytes() {
        // From the anchors: 0x00 -> 0x15, 0x01 -> 0x1c, 0x02 -> 0x07, these are
        // pairwise distinct, demonstrating single-byte separability.
        let r00 = crc5_epc(&[0x00]);
        let r01 = crc5_epc(&[0x01]);
        let r02 = crc5_epc(&[0x02]);
        assert!(r00 != r01);
        assert!(r01 != r02);
        assert!(r00 != r02);
    }

    #[test]
    fn chunking_does_not_apply_but_full_is_stable() {
        // The one-shot routine is stable regardless of how the caller assembled
        // the slice; verify two independently built identical arrays agree.
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        let mut i = 0usize;
        while i < 8 {
            a[i] = (i * 3) as u8;
            b[i] = (i * 3) as u8;
            i += 1;
        }
        assert!(crc5_epc(&a) == crc5_epc(&b));
    }

    #[test]
    fn even_length_zeros_pattern() {
        // Exercise is-multiple-of style reasoning without heap allocation.
        let data = [0u8; 6];
        assert!(data.len().is_multiple_of(2));
        let r = crc5_epc(&data);
        assert!((0..=0x1f).contains(&r));
    }

    #[test]
    fn single_bit_flip_changes_output() {
        let base = crc5_epc(&[0x00]);
        let flipped = crc5_epc(&[0x01]);
        assert!(base != flipped);
    }

    #[test]
    fn leading_zero_affects_result() {
        let without = crc5_epc(&[0x01]);
        let with = crc5_epc(&[0x00, 0x01]);
        assert!(without != with);
    }

    #[test]
    fn hello_differs_from_hell() {
        assert!(crc5_epc(b"Hello") != crc5_epc(b"Hell"));
    }
}
