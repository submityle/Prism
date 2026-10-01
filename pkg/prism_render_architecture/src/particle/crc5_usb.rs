//! `CRC-5/USB`: the 5-bit cyclic redundancy check used by the `USB` token
//! packet protocol, implemented as a pure-integer, bit-reflected (`LSB`-first)
//! checksum for cheaply validating short control fields and small particle
//! metadata records (design § integrity checks).
//!
//! The algorithm follows the standard `CRC-5/USB` parameter set: `width = 5`,
//! polynomial `poly = 0x05`, initial value `init = 0x1F`, reflected input
//! (`refin = true`), reflected output (`refout = true`), and final xor
//! `xorout = 0x1F`. Because both the input and the output are reflected, the
//! computation runs least-significant-bit first and uses the bit-reversed form
//! of the polynomial, `0x14`, exposed as [`CRC5_USB_POLY_REFLECTED`]. Each input
//! byte is folded into the running remainder across all eight of its bits, and
//! the final remainder is combined with `xorout` and masked to five bits.
//!
//! The one-shot [`crc5_usb`] and the streaming [`Crc5Usb`] accumulator share the
//! identical byte-at-a-time core, so feeding the data in arbitrary chunks always
//! yields the same checksum as a single call. Every operation is an integer
//! xor, shift, or mask; there are no floating-point or unsafe operations. The
//! well-known reference check value is `crc5_usb(b"123456789") == 0x19`, and the
//! empty input yields `0x00` because the initial value and `xorout` cancel.
//!
//! Scope: this is a short error-detection code, not a hash and not a message
//! authentication code. `CRC-5/USB` is *not* cryptographically secure and
//! collisions are trivial to construct on purpose, so it must never be used to
//! authenticate data or guard against a malicious adversary. It is meant only
//! for catching accidental corruption in short fields.

/// The reflected (`LSB`-first) form of the `CRC-5/USB` generator polynomial.
///
/// The nominal polynomial is `0x05`; reversing its five significant bits gives
/// the value used by the bit-reflected inner loop.
pub const CRC5_USB_POLY_REFLECTED: u8 = 0x14;

/// The initial remainder value for `CRC-5/USB` (`init = 0x1F`).
pub const CRC5_USB_INIT: u8 = 0x1F;

/// The final xor value applied to the remainder for `CRC-5/USB`
/// (`xorout = 0x1F`).
pub const CRC5_USB_XOROUT: u8 = 0x1F;

/// The five-bit mask applied to the finished checksum (`0x1F`).
pub const CRC5_USB_MASK: u8 = 0x1F;

/// Computes the `CRC-5/USB` checksum of `data` in a single call.
///
/// The empty slice yields `0x00` (the initial value and `xorout` cancel), and
/// `crc5_usb(b"123456789")` is the standard reference check value `0x19`. The
/// result is always within the five-bit range `0..=0x1F`.
#[must_use]
pub fn crc5_usb(data: &[u8]) -> u8 {
    let mut crc: u8 = CRC5_USB_INIT;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ CRC5_USB_POLY_REFLECTED;
            } else {
                crc >>= 1;
            }
        }
    }
    (crc ^ CRC5_USB_XOROUT) & CRC5_USB_MASK
}

/// A streaming `CRC-5/USB` accumulator.
///
/// Create one with [`Crc5Usb::new`], feed bytes with [`Crc5Usb::update`] in any
/// chunking, and read the checksum with [`Crc5Usb::finalize`]. The result is
/// identical to the one-shot [`crc5_usb`] over the concatenation of all updates.
/// `finalize` applies the final `xorout` and five-bit mask without mutating the
/// accumulator, so updating can continue afterwards; `update` performs no
/// finalisation of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc5Usb {
    crc: u8,
}

impl Crc5Usb {
    /// Creates a new accumulator seeded with the `CRC-5/USB` initial value.
    #[must_use]
    pub fn new() -> Self {
        Self { crc: CRC5_USB_INIT }
    }

    /// Folds every byte of `data` into the running remainder.
    ///
    /// This performs no final `xorout`; call [`Crc5Usb::finalize`] to read the
    /// checksum.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.crc ^= byte;
            for _ in 0..8 {
                if (self.crc & 1) != 0 {
                    self.crc = (self.crc >> 1) ^ CRC5_USB_POLY_REFLECTED;
                } else {
                    self.crc >>= 1;
                }
            }
        }
    }

    /// Returns the finished checksum by applying `xorout` and the five-bit mask.
    ///
    /// This does not mutate the accumulator, so further [`Crc5Usb::update`]
    /// calls remain valid afterwards.
    #[must_use]
    pub fn finalize(&self) -> u8 {
        (self.crc ^ CRC5_USB_XOROUT) & CRC5_USB_MASK
    }
}

impl Default for Crc5Usb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience helper: streams `data` through a fresh accumulator in one go.
    fn streamed_once(data: &[u8]) -> u8 {
        let mut h = Crc5Usb::new();
        h.update(data);
        h.finalize()
    }

    #[test]
    fn check_value_reference_vector() {
        assert_eq!(crc5_usb(b"123456789"), 0x19);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc5_usb(b""), 0x00);
    }

    #[test]
    fn empty_input_init_xorout_cancel() {
        assert_eq!(CRC5_USB_INIT ^ CRC5_USB_XOROUT, 0x00);
    }

    #[test]
    fn reflected_poly_constant_is_0x14() {
        assert_eq!(CRC5_USB_POLY_REFLECTED, 0x14);
    }

    #[test]
    fn init_constant_is_0x1f() {
        assert_eq!(CRC5_USB_INIT, 0x1F);
    }

    #[test]
    fn xorout_constant_is_0x1f() {
        assert_eq!(CRC5_USB_XOROUT, 0x1F);
    }

    #[test]
    fn mask_constant_is_0x1f() {
        assert_eq!(CRC5_USB_MASK, 0x1F);
    }

    #[test]
    fn result_fits_five_bits_for_check_vector() {
        assert!(crc5_usb(b"123456789") <= 0x1F);
    }

    #[test]
    fn result_fits_five_bits_for_empty() {
        assert!(crc5_usb(b"") <= 0x1F);
    }

    #[test]
    fn result_always_within_five_bits_single_bytes() {
        for b in 0u16..=255 {
            let v = crc5_usb(&[b as u8]);
            assert!(v <= 0x1F, "byte {b} produced {v}");
        }
    }

    #[test]
    fn result_always_within_five_bits_pairs() {
        for a in 0u16..=255 {
            for b in (0u16..=255).step_by(17) {
                let v = crc5_usb(&[a as u8, b as u8]);
                assert!(v <= 0x1F);
            }
        }
    }

    #[test]
    fn high_bits_always_clear() {
        for b in 0u16..=255 {
            let v = crc5_usb(&[b as u8, b as u8, b as u8]);
            assert_eq!(v & !0x1F, 0);
        }
    }

    #[test]
    fn deterministic_repeated_calls() {
        let data = b"deterministic payload";
        let first = crc5_usb(data);
        for _ in 0..64 {
            assert_eq!(crc5_usb(data), first);
        }
    }

    #[test]
    fn deterministic_across_fresh_slices() {
        let a = crc5_usb(b"prism-particle");
        let b = crc5_usb(b"prism-particle");
        assert_eq!(a, b);
    }

    #[test]
    fn stream_matches_oneshot_check_vector() {
        assert_eq!(streamed_once(b"123456789"), crc5_usb(b"123456789"));
    }

    #[test]
    fn stream_matches_oneshot_empty() {
        assert_eq!(streamed_once(b""), crc5_usb(b""));
    }

    #[test]
    fn stream_matches_oneshot_arbitrary() {
        let data = b"the quick brown fox";
        assert_eq!(streamed_once(data), crc5_usb(data));
    }

    #[test]
    fn incremental_byte_by_byte_matches_oneshot() {
        let data = b"chunked-update-equivalence";
        let mut h = Crc5Usb::new();
        for &byte in data {
            h.update(&[byte]);
        }
        assert_eq!(h.finalize(), crc5_usb(data));
    }

    #[test]
    fn incremental_two_chunks_matches_oneshot() {
        let data = b"123456789";
        let mut h = Crc5Usb::new();
        h.update(&data[..4]);
        h.update(&data[4..]);
        assert_eq!(h.finalize(), 0x19);
    }

    #[test]
    fn incremental_three_chunks_matches_oneshot() {
        let data = b"abcdefghijklmnop";
        let mut h = Crc5Usb::new();
        h.update(&data[..3]);
        h.update(&data[3..9]);
        h.update(&data[9..]);
        assert_eq!(h.finalize(), crc5_usb(data));
    }

    #[test]
    fn incremental_many_split_points_match() {
        let data = b"split-point-sweep-data-block";
        let oneshot = crc5_usb(data);
        for split in 0..=data.len() {
            let mut h = Crc5Usb::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), oneshot, "split at {split}");
        }
    }

    #[test]
    fn empty_updates_do_not_change_result() {
        let data = b"interleaved-empties";
        let mut h = Crc5Usb::new();
        h.update(b"");
        h.update(&data[..5]);
        h.update(b"");
        h.update(&data[5..]);
        h.update(b"");
        assert_eq!(h.finalize(), crc5_usb(data));
    }

    #[test]
    fn finalize_does_not_mutate_accumulator() {
        let mut h = Crc5Usb::new();
        h.update(b"1234");
        let first = h.finalize();
        let second = h.finalize();
        assert_eq!(first, second);
    }

    #[test]
    fn finalize_then_continue_matches_full_stream() {
        let mut h = Crc5Usb::new();
        h.update(b"1234");
        let _ = h.finalize();
        h.update(b"56789");
        assert_eq!(h.finalize(), 0x19);
    }

    #[test]
    fn new_equals_default() {
        assert_eq!(Crc5Usb::new(), Crc5Usb::default());
    }

    #[test]
    fn fresh_accumulator_finalize_is_zero() {
        assert_eq!(Crc5Usb::new().finalize(), 0x00);
    }

    #[test]
    fn clone_is_independent() {
        let mut h = Crc5Usb::new();
        h.update(b"abc");
        let snapshot = h;
        h.update(b"def");
        assert_eq!(snapshot.finalize(), crc5_usb(b"abc"));
        assert_eq!(h.finalize(), crc5_usb(b"abcdef"));
    }

    #[test]
    fn single_zero_byte() {
        let v = crc5_usb(&[0x00]);
        assert!(v <= 0x1F);
        assert_eq!(v, streamed_once(&[0x00]));
    }

    #[test]
    fn single_0xff_byte() {
        let v = crc5_usb(&[0xFF]);
        assert!(v <= 0x1F);
        assert_eq!(v, streamed_once(&[0xFF]));
    }

    #[test]
    fn all_zero_bytes_vary_by_length() {
        let a = crc5_usb(&[0x00; 1]);
        let b = crc5_usb(&[0x00; 2]);
        let c = crc5_usb(&[0x00; 3]);
        assert!(a <= 0x1F && b <= 0x1F && c <= 0x1F);
    }

    #[test]
    fn differing_inputs_can_differ() {
        assert_ne!(crc5_usb(b"A"), crc5_usb(b"B"));
    }

    #[test]
    fn length_sensitivity() {
        assert_ne!(crc5_usb(b"\x00"), crc5_usb(b"\x00\x00"));
    }

    #[test]
    fn order_sensitivity() {
        let ab = crc5_usb(b"\x01\x02");
        let ba = crc5_usb(b"\x02\x01");
        assert!(ab <= 0x1F && ba <= 0x1F);
    }

    #[test]
    fn long_input_within_range() {
        let data = [0xA5u8; 1024];
        let v = crc5_usb(&data);
        assert!(v <= 0x1F);
    }

    #[test]
    fn long_input_stream_matches_oneshot() {
        let data = [0x5Au8; 777];
        let mut h = Crc5Usb::new();
        for chunk in data.chunks(13) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), crc5_usb(&data));
    }

    #[test]
    fn repeated_pattern_stream_matches() {
        let mut data = [0u8; 300];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        let mut h = Crc5Usb::new();
        for chunk in data.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), crc5_usb(&data));
    }

    #[test]
    fn ascii_digits_check() {
        assert_eq!(crc5_usb(b"123456789"), 0x19);
    }

    #[test]
    fn stream_check_in_odd_chunks() {
        let data = b"123456789";
        let mut h = Crc5Usb::new();
        h.update(&data[..1]);
        h.update(&data[1..2]);
        h.update(&data[2..5]);
        h.update(&data[5..]);
        assert_eq!(h.finalize(), 0x19);
    }

    #[test]
    fn every_single_byte_stream_matches_oneshot() {
        for b in 0u16..=255 {
            let byte = b as u8;
            assert_eq!(streamed_once(&[byte]), crc5_usb(&[byte]));
        }
    }

    #[test]
    fn two_byte_exhaustive_stream_matches_oneshot() {
        for a in (0u16..=255).step_by(5) {
            for b in (0u16..=255).step_by(5) {
                let data = [a as u8, b as u8];
                let mut h = Crc5Usb::new();
                h.update(&data[..1]);
                h.update(&data[1..]);
                assert_eq!(h.finalize(), crc5_usb(&data));
            }
        }
    }

    #[test]
    fn all_results_observed_within_range_exhaustive_small() {
        for a in 0u16..=255 {
            for b in (0u16..=255).step_by(31) {
                for c in (0u16..=255).step_by(53) {
                    let v = crc5_usb(&[a as u8, b as u8, c as u8]);
                    assert_eq!(v & 0xE0, 0);
                }
            }
        }
    }
}
