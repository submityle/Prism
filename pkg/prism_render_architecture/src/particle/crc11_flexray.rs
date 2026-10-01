//! `CRC-11`/`FlexRay` checksum, implemented bit-at-a-time (MSB-first) with pure integers.
//!
//! Parameters: width=11, poly=0x385, init=0x01A, refin=false, refout=false, xorout=0x000.
//!
//! The running register is held in a `u16` whose low 11 bits are significant.
//! `mask` keeps the register inside 11 bits and `top` selects the most-significant
//! bit (bit 10) that is compared against each incoming message bit.

/// Low 11-bit mask applied to the register after every shift.
const MASK: u16 = 0x7FF;
/// Most-significant bit (bit 10) of the 11-bit register.
const TOP: u16 = 0x400;
/// Generator polynomial (`CRC-11`/`FlexRay`).
const POLY: u16 = 0x385;
/// Initial register value.
const INIT: u16 = 0x01A;
/// Final XOR value.
const XOROUT: u16 = 0x000;

/// Compute the `CRC-11`/`FlexRay` checksum of `data` in a single call.
///
/// Returns the 11-bit checksum in the low bits of a `u16`.
#[must_use]
pub fn checksum(data: &[u8]) -> u16 {
    let mut crc = INIT;
    for &byte in data {
        crc = update_byte(crc, byte);
    }
    crc ^ XOROUT
}

/// Fold a single input byte into the running register (MSB-first, refin=false).
#[must_use]
fn update_byte(mut crc: u16, byte: u8) -> u16 {
    for i in (0..8).rev() {
        let inbit = u16::from((byte >> i) & 1);
        let msb = if (crc & TOP) != 0 { 1u16 } else { 0u16 };
        crc = (crc << 1) & MASK;
        if (msb ^ inbit) == 1 {
            crc ^= POLY;
        }
    }
    crc
}

/// Incremental (streaming) `CRC-11`/`FlexRay` computation.
///
/// Feed bytes with [`Crc11FlexRay::update`] and read the result with
/// [`Crc11FlexRay::finalize`]. The result matches [`checksum`] over the same
/// concatenated input.
#[derive(Clone, Copy, Debug)]
pub struct Crc11FlexRay {
    crc: u16,
}

impl Crc11FlexRay {
    /// Create a new streaming instance seeded with the init value.
    #[must_use]
    pub fn new() -> Self {
        Self { crc: INIT }
    }

    /// Fold a chunk of bytes into the running register.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.crc = update_byte(self.crc, byte);
        }
    }

    /// Produce the final checksum (applies xorout).
    #[must_use]
    pub fn finalize(&self) -> u16 {
        self.crc ^ XOROUT
    }
}

impl Default for Crc11FlexRay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::checksum;
    use super::Crc11FlexRay;

    // --- Hard reference vectors (enshrined) ---

    #[test]
    fn hard_vector_check_string() {
        assert_eq!(checksum(b"123456789"), 0x5A3);
    }

    #[test]
    fn hard_vector_empty_equals_init() {
        assert_eq!(checksum(&[]), 0x01A);
    }

    #[test]
    fn hard_vector_single_0x41() {
        assert_eq!(checksum(&[0x41]), 0x032);
    }

    #[test]
    fn hard_vector_single_0x00() {
        assert_eq!(checksum(&[0x00]), 0x68F);
    }

    // --- Empty input properties ---

    #[test]
    fn empty_slice_is_init() {
        assert_eq!(checksum(&[]), 0x01A);
    }

    #[test]
    fn empty_result_fits_11_bits() {
        assert!(checksum(&[]) <= 0x7FF);
    }

    // --- Range / masking properties ---

    #[test]
    fn result_always_within_11_bits_single_bytes() {
        for b in 0u16..=255 {
            let v = checksum(&[b as u8]);
            assert!(v <= 0x7FF);
        }
    }

    #[test]
    fn result_within_11_bits_for_check_string() {
        assert!(checksum(b"123456789") <= 0x7FF);
    }

    // --- Single-byte spot checks ---

    #[test]
    fn single_byte_0xff_within_range() {
        assert!(checksum(&[0xFF]) <= 0x7FF);
    }

    #[test]
    fn single_byte_0x01_within_range() {
        assert!(checksum(&[0x01]) <= 0x7FF);
    }

    #[test]
    fn single_byte_0x80_within_range() {
        assert!(checksum(&[0x80]) <= 0x7FF);
    }

    // --- Streaming vs one-shot consistency ---

    #[test]
    fn streaming_matches_oneshot_check_string() {
        let mut s = Crc11FlexRay::new();
        s.update(b"123456789");
        assert_eq!(s.finalize(), checksum(b"123456789"));
    }

    #[test]
    fn streaming_matches_oneshot_empty() {
        let s = Crc11FlexRay::new();
        assert_eq!(s.finalize(), checksum(&[]));
    }

    #[test]
    fn streaming_matches_oneshot_single_0x41() {
        let mut s = Crc11FlexRay::new();
        s.update(&[0x41]);
        assert_eq!(s.finalize(), checksum(&[0x41]));
    }

    #[test]
    fn streaming_matches_oneshot_single_0x00() {
        let mut s = Crc11FlexRay::new();
        s.update(&[0x00]);
        assert_eq!(s.finalize(), checksum(&[0x00]));
    }

    #[test]
    fn streaming_chunked_matches_oneshot() {
        let mut s = Crc11FlexRay::new();
        s.update(b"1234");
        s.update(b"56789");
        assert_eq!(s.finalize(), checksum(b"123456789"));
    }

    #[test]
    fn streaming_byte_by_byte_matches_oneshot() {
        let mut s = Crc11FlexRay::new();
        for &b in b"123456789" {
            s.update(&[b]);
        }
        assert_eq!(s.finalize(), checksum(b"123456789"));
    }

    #[test]
    fn streaming_default_matches_new() {
        let a = Crc11FlexRay::new();
        let b = Crc11FlexRay::default();
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn streaming_empty_updates_are_noop() {
        let mut s = Crc11FlexRay::new();
        s.update(&[]);
        s.update(b"123456789");
        s.update(&[]);
        assert_eq!(s.finalize(), checksum(b"123456789"));
    }

    #[test]
    fn streaming_split_at_each_boundary() {
        let msg = b"123456789";
        for split in 0..=msg.len() {
            let (head, tail) = msg.split_at(split);
            let mut s = Crc11FlexRay::new();
            s.update(head);
            s.update(tail);
            assert_eq!(s.finalize(), checksum(msg));
        }
    }

    // --- Different data yields (generally) different crc ---

    #[test]
    fn different_single_bytes_differ() {
        assert_ne!(checksum(&[0x00]), checksum(&[0x01]));
    }

    #[test]
    fn different_strings_differ() {
        assert_ne!(checksum(b"hello"), checksum(b"world"));
    }

    #[test]
    fn append_changes_crc() {
        assert_ne!(checksum(b"1234"), checksum(b"12345"));
    }

    #[test]
    fn order_matters() {
        assert_ne!(checksum(&[0x01, 0x02]), checksum(&[0x02, 0x01]));
    }

    #[test]
    fn single_vs_double_zero_differ() {
        assert_ne!(checksum(&[0x00]), checksum(&[0x00, 0x00]));
    }

    // --- Determinism ---

    #[test]
    fn deterministic_repeated_calls() {
        let a = checksum(b"123456789");
        let b = checksum(b"123456789");
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_empty() {
        assert_eq!(checksum(&[]), checksum(&[]));
    }

    // --- Multi-byte spot checks within range ---

    #[test]
    fn multibyte_abc_within_range() {
        assert!(checksum(b"ABC") <= 0x7FF);
    }

    #[test]
    fn multibyte_all_ff_within_range() {
        assert!(checksum(&[0xFF, 0xFF, 0xFF, 0xFF]) <= 0x7FF);
    }

    #[test]
    fn multibyte_all_zero_within_range() {
        assert!(checksum(&[0x00, 0x00, 0x00, 0x00]) <= 0x7FF);
    }

    #[test]
    fn multibyte_mixed_within_range() {
        assert!(checksum(&[0xDE, 0xAD, 0xBE, 0xEF]) <= 0x7FF);
    }

    // --- Prefix independence of streaming reuse ---

    #[test]
    fn two_independent_streams_match() {
        let mut a = Crc11FlexRay::new();
        let mut b = Crc11FlexRay::new();
        a.update(b"abc");
        b.update(b"abc");
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn finalize_does_not_mutate_state() {
        let mut s = Crc11FlexRay::new();
        s.update(b"1234");
        let first = s.finalize();
        let second = s.finalize();
        assert_eq!(first, second);
    }

    #[test]
    fn finalize_then_continue() {
        let mut s = Crc11FlexRay::new();
        s.update(b"1234");
        let _ = s.finalize();
        s.update(b"56789");
        assert_eq!(s.finalize(), checksum(b"123456789"));
    }

    // --- Exhaustive-ish single byte streaming consistency ---

    #[test]
    fn all_single_bytes_streaming_matches_oneshot() {
        for b in 0u16..=255 {
            let byte = b as u8;
            let mut s = Crc11FlexRay::new();
            s.update(&[byte]);
            assert_eq!(s.finalize(), checksum(&[byte]));
        }
    }

    #[test]
    fn copy_semantics_preserve_value() {
        let mut s = Crc11FlexRay::new();
        s.update(b"12");
        let snapshot = s;
        s.update(b"3456789");
        assert_eq!(s.finalize(), checksum(b"123456789"));
        assert_eq!(snapshot.finalize(), checksum(b"12"));
    }

    #[test]
    fn long_input_within_range() {
        let data = [0x5Au8; 64];
        assert!(checksum(&data) <= 0x7FF);
    }

    #[test]
    fn long_input_streaming_matches_oneshot() {
        let data = [0xA5u8; 64];
        let mut s = Crc11FlexRay::new();
        s.update(&data);
        assert_eq!(s.finalize(), checksum(&data));
    }
}
