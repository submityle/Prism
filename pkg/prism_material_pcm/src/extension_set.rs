//! The fixed-width extension digest carried by the material header ABI.
//!
//! `extensions_used` records every registry slot a material's live response
//! touches; `extensions_required` records the subset the material *cannot*
//! shade without. Both are 64-bit and split into two `u32` header words, so the
//! digest is 4 words total (`used[lohi]`, `required[lohi]`). The digest is the
//! hot-path projection of the registry: `lobe_mask` is its `kind = Lobe`
//! subset, and it is never dispatched on.

use crate::HEADER_DIGEST_SLOTS;
use core::fmt;

/// A registry slot cannot be represented in the 64-bit header digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensionSetError {
    /// The rejected slot.
    pub slot: u32,
    /// The digest width.
    pub width: u32,
}

impl fmt::Display for ExtensionSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "registry slot {} exceeds the {}-slot header digest",
            self.slot, self.width
        )
    }
}

impl core::error::Error for ExtensionSetError {}

/// A material's `used` / `required` extension slots.
///
/// Invariant: `required ⊆ used`, upheld by every mutator.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ExtensionSet {
    used: u64,
    required: u64,
}

impl ExtensionSet {
    /// The empty digest (a pure specification-core surface).
    pub const EMPTY: Self = Self {
        used: 0,
        required: 0,
    };

    /// A digest with every representable slot marked `used` — the "gate off"
    /// value used by consumers that have no restriction to apply.
    pub const SATURATED: Self = Self {
        used: u64::MAX,
        required: 0,
    };

    /// A set containing just `slot`, as `used`.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionSetError`] when `slot` is outside the digest.
    pub fn from_used_slot(slot: u32) -> Result<Self, ExtensionSetError> {
        let mut set = Self::EMPTY;
        set.insert_used(slot)?;
        Ok(set)
    }

    /// Build from the four header words.
    pub const fn from_words(
        used_low: u32,
        used_high: u32,
        required_low: u32,
        required_high: u32,
    ) -> Self {
        let used = (used_low as u64) | ((used_high as u64) << 32);
        let required = (required_low as u64) | ((required_high as u64) << 32);
        Self {
            used,
            required: required & used,
        }
    }

    /// The low 32 bits of the `used` mask (header word 0).
    pub const fn used_low(self) -> u32 {
        self.used as u32
    }

    /// The high 32 bits of the `used` mask (header word 1).
    pub const fn used_high(self) -> u32 {
        (self.used >> 32) as u32
    }

    /// The low 32 bits of the `required` mask (header word 2).
    pub const fn required_low(self) -> u32 {
        self.required as u32
    }

    /// The high 32 bits of the `required` mask (header word 3).
    pub const fn required_high(self) -> u32 {
        (self.required >> 32) as u32
    }

    /// The full `used` mask.
    pub const fn used(self) -> u64 {
        self.used
    }

    /// The full `required` mask.
    pub const fn required(self) -> u64 {
        self.required
    }

    /// Mark `slot` as used.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionSetError`] when `slot` is outside the digest.
    pub fn insert_used(&mut self, slot: u32) -> Result<(), ExtensionSetError> {
        let bit = Self::bit(slot)?;
        self.used |= bit;
        Ok(())
    }

    /// Mark `slot` as required (which also marks it used).
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionSetError`] when `slot` is outside the digest.
    pub fn insert_required(&mut self, slot: u32) -> Result<(), ExtensionSetError> {
        let bit = Self::bit(slot)?;
        self.used |= bit;
        self.required |= bit;
        Ok(())
    }

    /// Whether `slot` is marked used.
    pub const fn contains_used(self, slot: u32) -> bool {
        slot < HEADER_DIGEST_SLOTS && self.used & (1u64 << slot) != 0
    }

    /// Whether `slot` is marked required.
    pub const fn contains_required(self, slot: u32) -> bool {
        slot < HEADER_DIGEST_SLOTS && self.required & (1u64 << slot) != 0
    }

    /// Number of used slots.
    pub const fn used_count(self) -> u32 {
        self.used.count_ones()
    }

    /// The union of the `used` masks (the `required` masks intersect).
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            used: self.used | other.used,
            required: (self.required | other.required) & (self.used | other.used),
        }
    }

    /// Whether every required slot here is also required in `other`.
    pub const fn required_is_subset_of(self, other: Self) -> bool {
        self.required & !other.required == 0
    }

    /// Whether every used slot here is also used in `other`.
    pub const fn used_is_subset_of(self, other: Self) -> bool {
        self.used & !other.used == 0
    }

    /// Keep only the used slots whose bit is set in `keep`.
    #[must_use]
    pub const fn intersect_used(self, keep: u64) -> Self {
        let used = self.used & keep;
        Self {
            used,
            required: self.required & used,
        }
    }

    /// Iterate the used slots, lowest first.
    pub fn iter_used(self) -> impl Iterator<Item = u32> {
        (0..HEADER_DIGEST_SLOTS).filter(move |slot| self.contains_used(*slot))
    }

    /// Iterate the required slots, lowest first.
    pub fn iter_required(self) -> impl Iterator<Item = u32> {
        (0..HEADER_DIGEST_SLOTS).filter(move |slot| self.contains_required(*slot))
    }

    const fn bit(slot: u32) -> Result<u64, ExtensionSetError> {
        if slot < HEADER_DIGEST_SLOTS {
            Ok(1u64 << slot)
        } else {
            Err(ExtensionSetError {
                slot,
                width: HEADER_DIGEST_SLOTS,
            })
        }
    }
}

impl FromIterator<u32> for ExtensionSet {
    /// Collects slots as `used`; a slot outside the digest is dropped, so
    /// prefer the fallible mutators where a slot is not already validated.
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut set = Self::EMPTY;
        for slot in iter {
            let _ = set.insert_used(slot);
        }
        set
    }
}
