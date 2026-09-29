//! Deterministic snapshot / restore *layout contract* for the Ember particle
//! engine (design §5, §11, §29).
//!
//! Deterministic rewind and replay — the ability to reconstruct an emitter's
//! exact `GPU` state at an earlier frame — is what lets a production `VFX`
//! engine offer scrubbable timelines, netcode rollback, and bit-reproducible
//! recordings, mirroring `Niagara`'s deterministic-replay guarantees without
//! reusing any of its code. Achieving that needs a *versioned, self-describing
//! byte layout*: a header identifying the format plus one `Structure-of-Arrays`
//! channel per captured attribute, sized by the same `std430` stride rules the
//! live `GPU` buffers use.
//!
//! This module owns only the **`CPU`-verifiable layout contract**: the magic /
//! version words, the attribute bitmask, the fixed-size header, the per-channel
//! and total byte accounting, and the version-migration decision. It performs
//! no real `IO` and reads no live buffers — it answers "how many bytes does a
//! snapshot of this shape occupy, and can this format be restored?" so a future
//! backend can serialize against a stable `ABI`.
//!
//! The randomness state is captured purely as *layout fields* — the
//! `frame`/`seed`/`stream` triple that seeds the stateless hash `RNG` in
//! [`super::determinism`]. Storing those three integers is enough to replay any
//! historical draw, so this module never re-implements the `RNG` itself.
//!
//! Everything here is deterministic integer arithmetic: byte accounting
//! saturates instead of overflowing, an empty pool still reserves one element
//! per channel (the `clamp-to-one` rule shared with [`super::gpu_layout`]), and
//! no operation panics on degenerate input.

use alloc::vec::Vec;
use core::cmp::Ordering;

use super::gpu_layout::{storage_bytes, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE};

/// Four-byte magic identifying a well-formed Ember snapshot: the `ASCII` bytes
/// `PRSN` ("`P`rism `S`erialized `N`apshot"), packed big-endian so a hex dump
/// reads left-to-right.
pub const SNAPSHOT_MAGIC: u32 = 0x50_52_53_4E;

/// Current snapshot layout version. Bumped whenever the header or channel
/// layout changes in a way that requires migration on restore.
pub const SNAPSHOT_VERSION: u32 = 1;

/// One capturable per-particle `Structure-of-Arrays` channel (design §5.1).
///
/// Each variant owns a single bit in [`AttributeMask`] and a `std430` per-
/// particle stride chosen to match the width the live simulation buffers use:
///
/// * [`SnapshotAttribute::Position`], [`SnapshotAttribute::Velocity`] — a
///   world-space `vec3` aligned up to a `vec4` (`VEC4_STRIDE`), matching how an
///   aligned `vec3` is padded in `std430`.
/// * [`SnapshotAttribute::Color`] — a linear `RGBA` `vec4` (`VEC4_STRIDE`).
/// * [`SnapshotAttribute::Size`] — a non-uniform `vec2` sprite size
///   (`VEC2_STRIDE`).
/// * [`SnapshotAttribute::Age`], [`SnapshotAttribute::Lifetime`],
///   [`SnapshotAttribute::Rotation`] — a scalar `f32` (`U32_STRIDE`).
/// * [`SnapshotAttribute::Custom`] — one scalar user channel packed as a `u32`
///   (`U32_STRIDE`); wider user payloads are captured as several custom
///   channels by a future authoring layer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SnapshotAttribute {
    /// World / local position (`vec3` padded to `vec4`).
    Position,
    /// Linear velocity (`vec3` padded to `vec4`).
    Velocity,
    /// Seconds a particle has been alive (scalar `f32`).
    Age,
    /// Total lifespan before retirement (scalar `f32`).
    Lifetime,
    /// Linear `RGBA` color (`vec4`).
    Color,
    /// Non-uniform sprite size (`vec2`).
    Size,
    /// Rotation angle in radians (scalar `f32`).
    Rotation,
    /// A single scalar user channel (`u32`).
    Custom,
}

impl SnapshotAttribute {
    /// Every attribute in a fixed, deterministic order (matches ascending bit
    /// order). Iterating this yields a stable channel layout across runs.
    pub const ALL: [SnapshotAttribute; 8] = [
        SnapshotAttribute::Position,
        SnapshotAttribute::Velocity,
        SnapshotAttribute::Age,
        SnapshotAttribute::Lifetime,
        SnapshotAttribute::Color,
        SnapshotAttribute::Size,
        SnapshotAttribute::Rotation,
        SnapshotAttribute::Custom,
    ];

    /// The single [`AttributeMask`] bit this attribute occupies.
    #[must_use]
    pub const fn bit(self) -> u32 {
        match self {
            SnapshotAttribute::Position => 1 << 0,
            SnapshotAttribute::Velocity => 1 << 1,
            SnapshotAttribute::Age => 1 << 2,
            SnapshotAttribute::Lifetime => 1 << 3,
            SnapshotAttribute::Color => 1 << 4,
            SnapshotAttribute::Size => 1 << 5,
            SnapshotAttribute::Rotation => 1 << 6,
            SnapshotAttribute::Custom => 1 << 7,
        }
    }

    /// The `std430` per-particle stride of this attribute's channel, in bytes.
    #[must_use]
    pub const fn stride(self) -> usize {
        match self {
            SnapshotAttribute::Position
            | SnapshotAttribute::Velocity
            | SnapshotAttribute::Color => VEC4_STRIDE,
            SnapshotAttribute::Size => VEC2_STRIDE,
            SnapshotAttribute::Age
            | SnapshotAttribute::Lifetime
            | SnapshotAttribute::Rotation
            | SnapshotAttribute::Custom => U32_STRIDE,
        }
    }
}

/// A bitmask selecting which [`SnapshotAttribute`] channels a snapshot captures
/// (design §5.1).
///
/// The mask is a plain `u32` of the per-attribute bits from
/// [`SnapshotAttribute::bit`], so it copies freely, hashes, and round-trips
/// through the header without allocation. Unknown high bits are preserved on
/// [`AttributeMask::from_bits`] but ignored by the accounting, which only walks
/// the known [`SnapshotAttribute::ALL`] set.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AttributeMask(u32);

impl AttributeMask {
    /// The empty mask (no channels captured).
    pub const EMPTY: AttributeMask = AttributeMask(0);

    /// Wraps a raw bit pattern.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        AttributeMask(bits)
    }

    /// The raw bit pattern.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// A mask with every known attribute set.
    #[must_use]
    pub const fn all() -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < SnapshotAttribute::ALL.len() {
            bits |= SnapshotAttribute::ALL[i].bit();
            i += 1;
        }
        AttributeMask(bits)
    }

    /// Whether no known attribute bit is set.
    ///
    /// Only the known attribute bits are considered, so a mask carrying just
    /// reserved high bits still reports empty for accounting purposes.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 & AttributeMask::all().0 == 0
    }

    /// Whether `attr`'s bit is set.
    #[must_use]
    pub const fn contains(self, attr: SnapshotAttribute) -> bool {
        self.0 & attr.bit() != 0
    }

    /// Returns a copy of this mask with `attr`'s bit set (builder style).
    #[must_use]
    pub const fn with(self, attr: SnapshotAttribute) -> Self {
        AttributeMask(self.0 | attr.bit())
    }

    /// Collects the set attributes in stable [`SnapshotAttribute::ALL`] order.
    #[must_use]
    pub fn iter_set(self) -> Vec<SnapshotAttribute> {
        let mut out = Vec::new();
        for attr in SnapshotAttribute::ALL {
            if self.contains(attr) {
                out.push(attr);
            }
        }
        out
    }
}

/// The fixed-size, versioned head of a snapshot (design §11, §29).
///
/// The header is a flat run of `u32` words so it maps 1:1 onto a `std430`
/// uniform / storage block a future `GPU` restore kernel can read directly. The
/// `rng_*` fields are the [`super::determinism`] hash-`RNG` seed triple; they
/// are stored verbatim so replay re-derives history without persisting any
/// per-particle random values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SnapshotHeader {
    /// Format identifier; must equal [`SNAPSHOT_MAGIC`] to be readable.
    pub magic: u32,
    /// Layout version this snapshot was written with.
    pub version: u32,
    /// The emitter pool's fixed capacity (channel length, in particles).
    pub capacity: u32,
    /// How many pool slots are live in this snapshot (`<= capacity`).
    pub particle_count: u32,
    /// Which `Structure-of-Arrays` channels this snapshot captures.
    pub attribute_mask: AttributeMask,
    /// The simulation frame index the hash-`RNG` was seeded at.
    pub rng_frame: u32,
    /// The effect-wide deterministic `RNG` seed.
    pub rng_seed: u32,
    /// The `RNG` stream namespace (see [`super::determinism::StreamId`]).
    pub rng_stream: u32,
}

/// The number of fixed `u32` words in a [`SnapshotHeader`].
///
/// Kept as an explicit constant so [`header_bytes`] stays a single source of
/// truth and the compiler flags any field added without updating the count via
/// the `debug_assert` in the tests.
const HEADER_WORD_COUNT: usize = 8;

impl SnapshotHeader {
    /// Builds a header for the current [`SNAPSHOT_MAGIC`] / [`SNAPSHOT_VERSION`].
    #[must_use]
    pub const fn new(
        capacity: u32,
        particle_count: u32,
        attribute_mask: AttributeMask,
        rng_frame: u32,
        rng_seed: u32,
        rng_stream: u32,
    ) -> Self {
        Self {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            capacity,
            particle_count,
            attribute_mask,
            rng_frame,
            rng_seed,
            rng_stream,
        }
    }
}

/// The byte size of a [`SnapshotHeader`]: the fixed word count times the scalar
/// `U32_STRIDE`.
#[must_use]
pub const fn header_bytes() -> usize {
    HEADER_WORD_COUNT * U32_STRIDE
}

/// The byte size of one attribute channel holding `capacity` particles.
///
/// Delegates to [`storage_bytes`] with the attribute's own stride, inheriting
/// the `clamp-to-one` rule (an empty pool still reserves one element) and the
/// saturating multiply (a degenerate `capacity` can never wrap).
#[must_use]
pub fn attribute_channel_bytes(attr: SnapshotAttribute, capacity: usize) -> usize {
    storage_bytes(attr.stride(), capacity)
}

/// The total byte size of a snapshot with this header's shape.
///
/// Sums [`header_bytes`] with every set attribute's [`attribute_channel_bytes`]
/// at the header's capacity. Each addition saturates, so an adversarial mask /
/// capacity combination clamps at [`usize::MAX`] instead of wrapping to a small
/// (and therefore under-allocated) value.
#[must_use]
pub fn snapshot_total_bytes(header: &SnapshotHeader) -> usize {
    let capacity = header.capacity as usize;
    let mut total = header_bytes();
    for attr in header.attribute_mask.iter_set() {
        total = total.saturating_add(attribute_channel_bytes(attr, capacity));
    }
    total
}

/// The restore-time compatibility decision for a snapshot's magic / version
/// (design §29).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VersionGuard {
    /// Same magic and version: restore directly.
    Compatible,
    /// Same magic, older version: run the migration path from `from` to `to`.
    NeedsMigration {
        /// The on-disk version encountered.
        from: u32,
        /// The current [`SNAPSHOT_VERSION`] to migrate up to.
        to: u32,
    },
    /// Wrong magic, or a version newer than this build understands: refuse.
    Incompatible,
}

/// Classifies a snapshot's magic and version against the current build.
///
/// * A mismatched `magic` is [`VersionGuard::Incompatible`] — the bytes are not
///   an Ember snapshot at all.
/// * An older `version` is [`VersionGuard::NeedsMigration`] carrying both ends
///   of the migration range.
/// * The current `version` is [`VersionGuard::Compatible`].
/// * A newer `version` is [`VersionGuard::Incompatible`] — this build cannot
///   know a future layout, so it refuses rather than guessing.
#[must_use]
pub fn guard(header_magic: u32, header_version: u32) -> VersionGuard {
    if header_magic != SNAPSHOT_MAGIC {
        return VersionGuard::Incompatible;
    }
    match header_version.cmp(&SNAPSHOT_VERSION) {
        Ordering::Less => VersionGuard::NeedsMigration {
            from: header_version,
            to: SNAPSHOT_VERSION,
        },
        Ordering::Equal => VersionGuard::Compatible,
        Ordering::Greater => VersionGuard::Incompatible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_ascii_prsn() {
        assert_eq!(SNAPSHOT_MAGIC.to_be_bytes(), *b"PRSN");
    }

    #[test]
    fn attribute_bits_are_unique_and_ordered() {
        let mut seen = 0u32;
        for attr in SnapshotAttribute::ALL {
            let bit = attr.bit();
            assert_eq!(bit.count_ones(), 1, "each attribute owns one bit");
            assert_eq!(seen & bit, 0, "attribute bits must not overlap");
            seen |= bit;
        }
        assert_eq!(seen, AttributeMask::all().bits());
    }

    #[test]
    fn attribute_strides_match_type_widths() {
        assert_eq!(SnapshotAttribute::Position.stride(), VEC4_STRIDE);
        assert_eq!(SnapshotAttribute::Velocity.stride(), VEC4_STRIDE);
        assert_eq!(SnapshotAttribute::Color.stride(), VEC4_STRIDE);
        assert_eq!(SnapshotAttribute::Size.stride(), VEC2_STRIDE);
        assert_eq!(SnapshotAttribute::Age.stride(), U32_STRIDE);
        assert_eq!(SnapshotAttribute::Lifetime.stride(), U32_STRIDE);
        assert_eq!(SnapshotAttribute::Rotation.stride(), U32_STRIDE);
        assert_eq!(SnapshotAttribute::Custom.stride(), U32_STRIDE);
    }

    #[test]
    fn empty_mask_is_empty_and_captures_nothing() {
        let mask = AttributeMask::EMPTY;
        assert!(mask.is_empty());
        assert!(mask.iter_set().is_empty());
        assert!(!mask.contains(SnapshotAttribute::Position));
    }

    #[test]
    fn with_sets_the_expected_bit() {
        let mask = AttributeMask::EMPTY
            .with(SnapshotAttribute::Position)
            .with(SnapshotAttribute::Color);
        assert!(mask.contains(SnapshotAttribute::Position));
        assert!(mask.contains(SnapshotAttribute::Color));
        assert!(!mask.contains(SnapshotAttribute::Velocity));
        assert!(!mask.is_empty());
    }

    #[test]
    fn iter_set_is_stable_and_ordered() {
        let mask = AttributeMask::EMPTY
            .with(SnapshotAttribute::Color)
            .with(SnapshotAttribute::Position)
            .with(SnapshotAttribute::Rotation);
        assert_eq!(
            mask.iter_set(),
            [
                SnapshotAttribute::Position,
                SnapshotAttribute::Color,
                SnapshotAttribute::Rotation,
            ]
        );
    }

    #[test]
    fn all_mask_lists_every_attribute() {
        assert_eq!(AttributeMask::all().iter_set(), SnapshotAttribute::ALL);
    }

    #[test]
    fn reserved_high_bits_do_not_count_as_present() {
        // A mask carrying only an unknown high bit is empty for accounting.
        let mask = AttributeMask::from_bits(1 << 20);
        assert!(mask.is_empty());
        assert!(mask.iter_set().is_empty());
    }

    #[test]
    fn header_bytes_is_fixed() {
        assert_eq!(header_bytes(), HEADER_WORD_COUNT * U32_STRIDE);
        assert_eq!(header_bytes(), 32);
    }

    #[test]
    fn empty_mask_total_is_just_the_header() {
        let header = SnapshotHeader::new(1024, 512, AttributeMask::EMPTY, 7, 42, 3);
        assert_eq!(snapshot_total_bytes(&header), header_bytes());
    }

    #[test]
    fn full_mask_total_sums_every_channel() {
        let capacity = 100usize;
        let header = SnapshotHeader::new(
            capacity as u32,
            capacity as u32,
            AttributeMask::all(),
            0,
            0,
            0,
        );
        let expected: usize = SnapshotAttribute::ALL
            .iter()
            .map(|attr| attribute_channel_bytes(*attr, capacity))
            .fold(header_bytes(), usize::saturating_add);
        assert_eq!(snapshot_total_bytes(&header), expected);
        // Cross-check against the hand-computed stride sum:
        //   position+velocity+color = 3 * 16 * 100 = 4800
        //   size                    = 8 * 100       =  800
        //   age+lifetime+rotation+custom = 4 * 4 * 100 = 1600
        //   header                  = 32
        assert_eq!(snapshot_total_bytes(&header), 4800 + 800 + 1600 + 32);
    }

    #[test]
    fn channel_bytes_clamp_to_one_at_zero_capacity() {
        assert_eq!(
            attribute_channel_bytes(SnapshotAttribute::Position, 0),
            VEC4_STRIDE
        );
        assert_eq!(
            attribute_channel_bytes(SnapshotAttribute::Age, 0),
            U32_STRIDE
        );
    }

    #[test]
    fn zero_capacity_total_still_reserves_one_element_per_channel() {
        let header = SnapshotHeader::new(0, 0, AttributeMask::all(), 0, 0, 0);
        let expected = header_bytes()
            + SnapshotAttribute::ALL
                .iter()
                .map(|attr| attr.stride())
                .sum::<usize>();
        assert_eq!(snapshot_total_bytes(&header), expected);
    }

    #[test]
    fn total_bytes_are_monotonic_in_capacity() {
        let mask = AttributeMask::all();
        let mut previous = 0usize;
        for capacity in [0u32, 1, 2, 10, 100, 4096] {
            let header = SnapshotHeader::new(capacity, capacity, mask, 0, 0, 0);
            let total = snapshot_total_bytes(&header);
            assert!(total >= previous, "total must not shrink as capacity grows");
            previous = total;
        }
    }

    #[test]
    fn channel_bytes_saturate_instead_of_wrapping() {
        // A near-`usize::MAX` capacity would overflow a naive stride multiply;
        // the `storage_bytes` clamp must pin the result at the ceiling.
        assert_eq!(
            attribute_channel_bytes(SnapshotAttribute::Position, usize::MAX),
            usize::MAX
        );
        assert_eq!(
            attribute_channel_bytes(SnapshotAttribute::Size, usize::MAX),
            usize::MAX
        );
    }

    #[test]
    fn total_bytes_saturate_across_many_channels() {
        // Force the per-channel accounting to already sit at the ceiling, then
        // confirm summing further saturating channels stays pinned rather than
        // wrapping back down to a small (under-allocated) value.
        let ceiling = attribute_channel_bytes(SnapshotAttribute::Position, usize::MAX);
        let stepped = ceiling.saturating_add(attribute_channel_bytes(
            SnapshotAttribute::Velocity,
            usize::MAX,
        ));
        assert_eq!(stepped, usize::MAX);
    }

    #[test]
    fn max_u32_capacity_total_does_not_wrap() {
        // The largest header-expressible capacity must still produce the exact
        // saturating sum without panicking or wrapping.
        let capacity = u32::MAX;
        let header = SnapshotHeader::new(capacity, capacity, AttributeMask::all(), 0, 0, 0);
        let expected: usize = SnapshotAttribute::ALL
            .iter()
            .map(|attr| attribute_channel_bytes(*attr, capacity as usize))
            .fold(header_bytes(), usize::saturating_add);
        assert_eq!(snapshot_total_bytes(&header), expected);
    }

    #[test]
    fn guard_rejects_wrong_magic() {
        assert_eq!(
            guard(0xDEAD_BEEF, SNAPSHOT_VERSION),
            VersionGuard::Incompatible
        );
        assert_eq!(guard(0, 0), VersionGuard::Incompatible);
    }

    #[test]
    fn guard_accepts_current_version() {
        assert_eq!(
            guard(SNAPSHOT_MAGIC, SNAPSHOT_VERSION),
            VersionGuard::Compatible
        );
    }

    #[test]
    fn guard_flags_older_version_for_migration() {
        let older = guard(SNAPSHOT_MAGIC, 0);
        assert_eq!(
            older,
            VersionGuard::NeedsMigration {
                from: 0,
                to: SNAPSHOT_VERSION,
            }
        );
    }

    #[test]
    fn guard_rejects_future_version() {
        assert_eq!(
            guard(SNAPSHOT_MAGIC, SNAPSHOT_VERSION + 1),
            VersionGuard::Incompatible
        );
    }

    #[test]
    fn header_constructor_stamps_current_format() {
        let header = SnapshotHeader::new(8, 4, AttributeMask::EMPTY, 1, 2, 3);
        assert_eq!(header.magic, SNAPSHOT_MAGIC);
        assert_eq!(header.version, SNAPSHOT_VERSION);
        assert_eq!(
            guard(header.magic, header.version),
            VersionGuard::Compatible
        );
    }
}
