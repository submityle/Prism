//! Deterministic content hashing for cache keys and ABI versioning.
//!
//! We deliberately avoid [`std::collections::hash_map::DefaultHasher`] here:
//! its output is only guaranteed stable within a single program run, whereas
//! ABI versions and on-disk cache keys must be reproducible across builds and
//! machines. FNV-1a is tiny, dependency-free, and fully deterministic, which
//! is all this crate needs (it is not used for anything security sensitive).

/// 64-bit FNV-1a offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// 64-bit FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A streaming FNV-1a 64-bit hasher with a stable, portable result.
#[derive(Debug, Clone)]
pub struct Fnv1a {
    state: u64,
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }
}

impl Fnv1a {
    /// Start a fresh hasher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold a byte slice into the running hash.
    pub fn write(&mut self, bytes: &[u8]) {
        let mut state = self.state;
        for &byte in bytes {
            state ^= u64::from(byte);
            state = state.wrapping_mul(FNV_PRIME);
        }
        self.state = state;
    }

    /// Fold a string, framed with its length so that concatenation is
    /// unambiguous (`["ab", "c"]` hashes differently from `["a", "bc"]`).
    pub fn write_framed(&mut self, text: &str) {
        self.write(&(text.len() as u64).to_le_bytes());
        self.write(text.as_bytes());
    }

    /// Fold a `u32`, little-endian.
    pub fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    /// Fold a `u64`, little-endian.
    pub fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// Finish and return the 64-bit digest.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

/// Convenience: hash a single byte slice in one call.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matches_offset_basis() {
        assert_eq!(Fnv1a::new().finish(), FNV_OFFSET_BASIS);
    }

    #[test]
    fn known_vector_a() {
        // FNV-1a("a") is a well-known published test vector.
        assert_eq!(hash_bytes(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn known_vector_foobar() {
        assert_eq!(hash_bytes(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn framing_disambiguates_concatenation() {
        let mut a = Fnv1a::new();
        a.write_framed("ab");
        a.write_framed("c");

        let mut b = Fnv1a::new();
        b.write_framed("a");
        b.write_framed("bc");

        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn is_deterministic_across_instances() {
        let mut a = Fnv1a::new();
        a.write_framed("prism");
        a.write_u32(7);
        let mut b = Fnv1a::new();
        b.write_framed("prism");
        b.write_u32(7);
        assert_eq!(a.finish(), b.finish());
    }
}
