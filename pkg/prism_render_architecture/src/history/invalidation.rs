//! History invalidation reasons as a compact bitmask.
//!
//! Temporal techniques (`TAA`, temporal upscalers, screen-space denoisers,
//! reprojected shadows and reflections) reuse the previous frame's result to
//! amortize cost across frames. That reuse is only valid while the reprojection
//! assumptions hold. When something breaks those assumptions the history must
//! be *invalidated* so the consumer falls back to a fresh, non-reprojected
//! estimate for one or more frames.
//!
//! [`InvalidationMask`] packs the distinct invalidation reasons into a single
//! `u32` so producers can accumulate reasons cheaply and consumers can test
//! them without branching per reason. It is a pure-integer newtype, so it is
//! `Eq`/`Hash`/`Ord` and every operation is a deterministic bit manipulation.

/// A named history-invalidation reason, one bit in an [`InvalidationMask`].
///
/// The discriminant is the *bit index*, not the mask value, so a reason maps to
/// its mask through [`InvalidationReason::mask`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InvalidationReason {
    /// The camera teleported (cut, respawn, cinematic jump); no pixel from the
    /// previous frame reprojects to this one.
    CameraCut,
    /// The render resolution or output size changed, so history texels no
    /// longer align with the new grid.
    Resolution,
    /// Auto-exposure moved enough that reprojected radiance is mis-scaled.
    Exposure,
    /// Scene topology changed (instances created or destroyed) beyond what
    /// motion vectors can reproject.
    Scene,
    /// Lighting environment changed (a light toggled, sky relit) so cached
    /// shading is stale.
    Lighting,
    /// Material parameters changed, invalidating cached shaded history.
    Material,
    /// Newly streamed-in geometry or textures revealed surfaces that had no
    /// valid history.
    StreamingReveal,
    /// A shader or pipeline recompiled with different math, so old results are
    /// not bit-comparable.
    ShaderVersion,
}

impl InvalidationReason {
    /// Every reason in bit order, for iteration and completeness checks.
    pub const ALL: [InvalidationReason; 8] = [
        InvalidationReason::CameraCut,
        InvalidationReason::Resolution,
        InvalidationReason::Exposure,
        InvalidationReason::Scene,
        InvalidationReason::Lighting,
        InvalidationReason::Material,
        InvalidationReason::StreamingReveal,
        InvalidationReason::ShaderVersion,
    ];

    /// The zero-based bit index this reason occupies.
    #[must_use]
    pub const fn bit_index(self) -> u32 {
        match self {
            InvalidationReason::CameraCut => 0,
            InvalidationReason::Resolution => 1,
            InvalidationReason::Exposure => 2,
            InvalidationReason::Scene => 3,
            InvalidationReason::Lighting => 4,
            InvalidationReason::Material => 5,
            InvalidationReason::StreamingReveal => 6,
            InvalidationReason::ShaderVersion => 7,
        }
    }

    /// The single-bit [`InvalidationMask`] for this reason.
    #[must_use]
    pub const fn mask(self) -> InvalidationMask {
        InvalidationMask(1 << self.bit_index())
    }
}

/// A set of [`InvalidationReason`]s packed into a `u32`.
///
/// Construction, union, intersection, and difference are all pure bit
/// operations, so the type is deterministic and cheap to accumulate across a
/// frame. Bits above the highest defined reason are never set by this module's
/// constructors; [`InvalidationMask::from_bits_truncate`] masks them off.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InvalidationMask(pub u32);

impl InvalidationMask {
    /// The empty set: nothing is invalidated.
    pub const EMPTY: Self = Self(0);

    /// A hard camera cut: no history reprojects.
    pub const CAMERA_CUT: Self = InvalidationReason::CameraCut.mask();
    /// The render resolution or output size changed.
    pub const RESOLUTION: Self = InvalidationReason::Resolution.mask();
    /// Auto-exposure moved beyond the reuse tolerance.
    pub const EXPOSURE: Self = InvalidationReason::Exposure.mask();
    /// Scene topology changed beyond motion-vector reprojection.
    pub const SCENE: Self = InvalidationReason::Scene.mask();
    /// The lighting environment changed.
    pub const LIGHTING: Self = InvalidationReason::Lighting.mask();
    /// Material parameters changed.
    pub const MATERIAL: Self = InvalidationReason::Material.mask();
    /// Newly streamed content revealed history-less surfaces.
    pub const STREAMING_REVEAL: Self = InvalidationReason::StreamingReveal.mask();
    /// A shader or pipeline recompiled with different math.
    pub const SHADER_VERSION: Self = InvalidationReason::ShaderVersion.mask();

    /// The set of every defined reason.
    pub const ALL: Self = Self(
        Self::CAMERA_CUT.0
            | Self::RESOLUTION.0
            | Self::EXPOSURE.0
            | Self::SCENE.0
            | Self::LIGHTING.0
            | Self::MATERIAL.0
            | Self::STREAMING_REVEAL.0
            | Self::SHADER_VERSION.0,
    );

    /// Wraps a raw bit pattern without validating that only defined bits are
    /// set. Prefer [`Self::from_bits_truncate`] for untrusted input.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Wraps a raw bit pattern, clearing any bit that does not correspond to a
    /// defined [`InvalidationReason`].
    #[must_use]
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Builds a mask from a single reason.
    #[must_use]
    pub const fn from_reason(reason: InvalidationReason) -> Self {
        reason.mask()
    }

    /// The raw bit pattern.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when no reason is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` when every defined reason is set.
    #[must_use]
    pub const fn is_all(self) -> bool {
        self.0 == Self::ALL.0
    }

    /// The number of distinct reasons currently set.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` when `reason` is present.
    #[must_use]
    pub const fn contains_reason(self, reason: InvalidationReason) -> bool {
        self.contains(reason.mask())
    }

    /// Returns `true` when every reason in `other` is also present in `self`.
    ///
    /// The empty set is contained by every mask.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` when `self` and `other` share at least one reason.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The union of the two reason sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The reasons present in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// The reasons in `self` that are not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// The reasons in exactly one of the two sets.
    #[must_use]
    pub const fn symmetric_difference(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Inserts every reason in `other` (in place).
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes every reason in `other` (in place).
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Collects the set reasons in bit order.
    #[must_use]
    pub fn reasons(self) -> Vec<InvalidationReason> {
        let mut out = Vec::new();
        for reason in InvalidationReason::ALL {
            if self.contains_reason(reason) {
                out.push(reason);
            }
        }
        out
    }

    /// Returns `true` when *any* invalidation reason forces a full history
    /// reset rather than a partial, category-scoped refresh.
    ///
    /// A camera cut, resolution change, or shader-version change breaks
    /// reprojection globally, so no reprojected texel is trustworthy.
    #[must_use]
    pub const fn forces_full_reset(self) -> bool {
        self.intersects(Self(
            Self::CAMERA_CUT.0 | Self::RESOLUTION.0 | Self::SHADER_VERSION.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_bits_are_unique_and_dense() {
        let mut seen = 0u32;
        for (i, reason) in InvalidationReason::ALL.iter().enumerate() {
            assert_eq!(reason.bit_index() as usize, i);
            let bit = reason.mask().bits();
            assert_eq!(bit.count_ones(), 1);
            assert_eq!(seen & bit, 0, "duplicate bit for {reason:?}");
            seen |= bit;
        }
        assert_eq!(seen, InvalidationMask::ALL.bits());
    }

    #[test]
    fn all_is_union_of_every_reason() {
        let mut acc = InvalidationMask::EMPTY;
        for reason in InvalidationReason::ALL {
            acc = acc.union(reason.mask());
        }
        assert_eq!(acc, InvalidationMask::ALL);
        assert!(InvalidationMask::ALL.is_all());
        assert_eq!(InvalidationMask::ALL.count(), 8);
    }

    #[test]
    fn empty_set_contains_nothing_but_is_contained() {
        let empty = InvalidationMask::EMPTY;
        assert!(empty.is_empty());
        assert!(!empty.intersects(InvalidationMask::ALL));
        assert!(InvalidationMask::ALL.contains(empty));
        assert!(empty.contains(empty));
        assert!(!empty.contains(InvalidationMask::SCENE));
        assert!(empty.reasons().is_empty());
    }

    #[test]
    fn set_algebra_is_consistent() {
        let a = InvalidationMask::SCENE.union(InvalidationMask::LIGHTING);
        let b = InvalidationMask::LIGHTING.union(InvalidationMask::MATERIAL);
        assert_eq!(a.union(b).count(), 3);
        assert_eq!(a.intersection(b), InvalidationMask::LIGHTING);
        assert_eq!(a.difference(b), InvalidationMask::SCENE);
        assert_eq!(
            a.symmetric_difference(b),
            InvalidationMask::SCENE.union(InvalidationMask::MATERIAL)
        );
    }

    #[test]
    fn insert_and_remove_mutate_in_place() {
        let mut m = InvalidationMask::EMPTY;
        m.insert(InvalidationMask::EXPOSURE);
        m.insert(InvalidationMask::SCENE);
        assert!(m.contains(InvalidationMask::EXPOSURE));
        assert!(m.contains_reason(InvalidationReason::Scene));
        m.remove(InvalidationMask::EXPOSURE);
        assert!(!m.contains_reason(InvalidationReason::Exposure));
        assert_eq!(m, InvalidationMask::SCENE);
    }

    #[test]
    fn from_bits_truncate_drops_undefined_bits() {
        let raw = 0xFFFF_FFFF;
        let masked = InvalidationMask::from_bits_truncate(raw);
        assert_eq!(masked, InvalidationMask::ALL);
        // `from_bits` keeps the raw pattern verbatim.
        assert_eq!(InvalidationMask::from_bits(raw).bits(), raw);
    }

    #[test]
    fn reasons_listed_in_bit_order() {
        let m = InvalidationMask::SHADER_VERSION
            .union(InvalidationMask::CAMERA_CUT)
            .union(InvalidationMask::MATERIAL);
        assert_eq!(
            m.reasons(),
            alloc::vec![
                InvalidationReason::CameraCut,
                InvalidationReason::Material,
                InvalidationReason::ShaderVersion,
            ]
        );
    }

    #[test]
    fn full_reset_reasons_are_recognized() {
        assert!(InvalidationMask::CAMERA_CUT.forces_full_reset());
        assert!(InvalidationMask::RESOLUTION.forces_full_reset());
        assert!(InvalidationMask::SHADER_VERSION.forces_full_reset());
        // Category-scoped reasons do not force a global reset on their own.
        assert!(!InvalidationMask::SCENE.forces_full_reset());
        assert!(!InvalidationMask::LIGHTING.forces_full_reset());
        assert!(InvalidationMask::LIGHTING
            .union(InvalidationMask::CAMERA_CUT)
            .forces_full_reset());
    }
}
