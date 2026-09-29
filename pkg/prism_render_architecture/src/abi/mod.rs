//! Versioned CPU, HLSL, WESL, and SPIR-V data contracts.

/// Identifies a version of a generated render ABI.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AbiVersion(pub u32);

/// A stable, generational index shared across CPU and GPU code.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GenerationalHandle {
    pub index: u32,
    pub generation: u32,
}

impl GenerationalHandle {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    pub const fn is_valid(self) -> bool {
        self.index != u32::MAX
    }
}

impl Default for GenerationalHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Hashes all inputs that affect a generated ABI package.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AbiHash(pub [u8; 32]);

// ----------------------------------------------------------------------------
// GenerationalHandle: packing helpers and generation lifecycle (additive).
// ----------------------------------------------------------------------------

impl GenerationalHandle {
    /// Builds a handle from an explicit `index` and `generation`.
    ///
    /// This never validates `index`; `u32::MAX` is reserved for [`Self::INVALID`].
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Packs the handle into a single `u64` for compact `CPU`/`GPU` transfer.
    ///
    /// The layout places `generation` in the high 32 bits and `index` in the
    /// low 32 bits, so [`Self::from_u64`] is an exact inverse.
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        ((self.generation as u64) << 32) | (self.index as u64)
    }

    /// Unpacks a handle previously produced by [`Self::to_u64`].
    #[must_use]
    pub const fn from_u64(bits: u64) -> Self {
        Self {
            index: (bits & 0xFFFF_FFFF) as u32,
            generation: (bits >> 32) as u32,
        }
    }

    /// Returns a copy of this handle rebound to `generation`.
    #[must_use]
    pub const fn with_generation(self, generation: u32) -> Self {
        Self {
            index: self.index,
            generation,
        }
    }

    /// Returns a copy with the generation advanced by one (wrapping).
    ///
    /// Recycling a slot bumps its generation so old handles never resolve; the
    /// wrap keeps the operation total without an overflow panic.
    #[must_use]
    pub const fn bumped(self) -> Self {
        Self {
            index: self.index,
            generation: self.generation.wrapping_add(1),
        }
    }
}

// ----------------------------------------------------------------------------
// AbiVersion: semantic-version-style compatibility (additive).
// ----------------------------------------------------------------------------

impl AbiVersion {
    /// Composes a version from a `major` and `minor` component.
    ///
    /// The `major` component occupies the high 16 bits and `minor` the low 16,
    /// matching the decoding performed by [`Self::major`]/[`Self::minor`].
    #[must_use]
    pub const fn from_parts(major: u16, minor: u16) -> Self {
        Self(((major as u32) << 16) | (minor as u32))
    }

    /// Returns the raw packed representation.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns the major component (high 16 bits).
    #[must_use]
    pub const fn major(self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// Returns the minor component (low 16 bits).
    #[must_use]
    pub const fn minor(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// Reports whether `self` satisfies the `required` contract version.
    ///
    /// Compatibility follows a backward-compatible rule: the major components
    /// must match exactly, and `self`'s minor must be at least `required`'s.
    /// A newer minor can serve an older consumer, but never the reverse.
    #[must_use]
    pub const fn is_compatible_with(self, required: Self) -> bool {
        self.major() == required.major() && self.minor() >= required.minor()
    }
}

// ----------------------------------------------------------------------------
// AbiHash: deterministic FNV-1a content hashing (additive, no external crates).
// ----------------------------------------------------------------------------

/// 64-bit `FNV`-1a offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// 64-bit `FNV`-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Odd mixing constant (fractional bits of the golden ratio) used to
/// de-correlate the four hashing lanes.
const LANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

impl AbiHash {
    /// The all-zero hash, used as an "unset" sentinel.
    pub const ZERO: Self = Self([0u8; 32]);

    /// Hashes a single byte slice into a deterministic 32-byte digest.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut builder = AbiHashBuilder::new();
        builder.write(bytes);
        builder.finish()
    }

    /// Hashes an ordered list of byte slices with length framing.
    ///
    /// Each part is prefixed with its length, so `[b"ab", b"c"]` and
    /// `[b"a", b"bc"]` produce distinct digests despite equal concatenations.
    #[must_use]
    pub fn combine(parts: &[&[u8]]) -> Self {
        let mut builder = AbiHashBuilder::new();
        for part in parts {
            builder.write_framed(part);
        }
        builder.finish()
    }
}

/// Incremental builder for [`AbiHash`] digests.
///
/// Four independent `FNV`-1a lanes are seeded with distinct salts and fed the
/// same byte stream; their little-endian outputs are concatenated to fill the
/// 32-byte digest. This is a deterministic content fingerprint for detecting
/// `ABI` drift, not a cryptographic hash.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AbiHashBuilder {
    lanes: [u64; 4],
}

impl AbiHashBuilder {
    /// Creates a builder with the four lanes seeded to distinct states.
    #[must_use]
    pub const fn new() -> Self {
        let mut lanes = [0u64; 4];
        let mut i = 0usize;
        while i < 4 {
            lanes[i] = FNV_OFFSET_BASIS ^ LANE_SALT.wrapping_mul(i as u64 + 1);
            i += 1;
        }
        Self { lanes }
    }

    /// Absorbs raw bytes into every lane using `FNV`-1a.
    pub fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            let mut lane = 0usize;
            while lane < 4 {
                self.lanes[lane] = (self.lanes[lane] ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
                lane += 1;
            }
        }
    }

    /// Absorbs a length-framed byte slice, so slice boundaries are significant.
    pub fn write_framed(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }

    /// Finalizes the digest with a per-lane avalanche mix.
    #[must_use]
    pub fn finish(self) -> AbiHash {
        let mut out = [0u8; 32];
        let mut lane = 0usize;
        while lane < 4 {
            // xorshift-multiply finisher improves bit diffusion of the lane.
            let mut v = self.lanes[lane];
            v ^= v >> 33;
            v = v.wrapping_mul(FNV_PRIME);
            v ^= v >> 29;
            let bytes = v.to_le_bytes();
            let base = lane * 8;
            let mut b = 0usize;
            while b < 8 {
                out[base + b] = bytes[b];
                b += 1;
            }
            lane += 1;
        }
        AbiHash(out)
    }
}

impl Default for AbiHashBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_pack_round_trips() {
        let handle = GenerationalHandle::new(42, 7);
        let bits = handle.to_u64();
        assert_eq!(GenerationalHandle::from_u64(bits), handle);
        assert_eq!(bits, (7u64 << 32) | 42u64);
    }

    #[test]
    fn invalid_handle_round_trips_and_stays_invalid() {
        let bits = GenerationalHandle::INVALID.to_u64();
        let restored = GenerationalHandle::from_u64(bits);
        assert_eq!(restored, GenerationalHandle::INVALID);
        assert!(!restored.is_valid());
    }

    #[test]
    fn bumped_advances_generation_and_wraps() {
        let handle = GenerationalHandle::new(3, 0);
        assert_eq!(handle.bumped().generation, 1);
        assert_eq!(handle.bumped().index, 3);
        let maxed = GenerationalHandle::new(3, u32::MAX);
        assert_eq!(maxed.bumped().generation, 0);
    }

    #[test]
    fn with_generation_preserves_index() {
        let handle = GenerationalHandle::new(9, 1).with_generation(5);
        assert_eq!(handle.index, 9);
        assert_eq!(handle.generation, 5);
    }

    #[test]
    fn version_parts_round_trip() {
        let version = AbiVersion::from_parts(3, 17);
        assert_eq!(version.major(), 3);
        assert_eq!(version.minor(), 17);
        assert_eq!(version.raw(), (3u32 << 16) | 17u32);
    }

    #[test]
    fn version_compatibility_follows_semver_rule() {
        let required = AbiVersion::from_parts(2, 4);
        // Same major, higher or equal minor: compatible.
        assert!(AbiVersion::from_parts(2, 4).is_compatible_with(required));
        assert!(AbiVersion::from_parts(2, 9).is_compatible_with(required));
        // Same major, lower minor: incompatible (missing newer additions).
        assert!(!AbiVersion::from_parts(2, 3).is_compatible_with(required));
        // Different major: never compatible.
        assert!(!AbiVersion::from_parts(3, 4).is_compatible_with(required));
        assert!(!AbiVersion::from_parts(1, 9).is_compatible_with(required));
    }

    #[test]
    fn hash_is_deterministic() {
        let a = AbiHash::from_bytes(b"prism-render-abi");
        let b = AbiHash::from_bytes(b"prism-render-abi");
        assert_eq!(a, b);
        assert_ne!(a, AbiHash::ZERO);
    }

    #[test]
    fn hash_distinguishes_different_inputs() {
        let a = AbiHash::from_bytes(b"layout-v1");
        let b = AbiHash::from_bytes(b"layout-v2");
        assert_ne!(a, b);
    }

    #[test]
    fn hash_lanes_are_decorrelated() {
        // A well-seeded digest should not collapse to four identical lanes.
        let digest = AbiHash::from_bytes(b"decorrelation-check").0;
        let lane0 = &digest[0..8];
        let lane1 = &digest[8..16];
        let lane2 = &digest[16..24];
        let lane3 = &digest[24..32];
        assert_ne!(lane0, lane1);
        assert_ne!(lane1, lane2);
        assert_ne!(lane2, lane3);
    }

    #[test]
    fn combine_is_framing_sensitive() {
        let framed = AbiHash::combine(&[b"ab", b"c"]);
        let regrouped = AbiHash::combine(&[b"a", b"bc"]);
        // Equal concatenation, different framing => different digest.
        assert_ne!(framed, regrouped);
        // Determinism holds for the framed form.
        assert_eq!(framed, AbiHash::combine(&[b"ab", b"c"]));
    }

    #[test]
    fn builder_matches_from_bytes() {
        let mut builder = AbiHashBuilder::new();
        builder.write(b"chunk-one");
        builder.write(b"chunk-two");
        assert_eq!(builder.finish(), AbiHash::from_bytes(b"chunk-onechunk-two"));
    }
}
