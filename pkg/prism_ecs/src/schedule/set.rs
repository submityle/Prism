//! System sets: named, hashable labels that group systems for ordering and
//! run-condition configuration (design §8.2).
//!
//! A [`SystemSet`] is any `'static` label that can report a stable
//! [`SystemSetId`]. Ordering edges (`before`/`after`) and shared run-conditions
//! are expressed against sets rather than individual systems, which keeps the
//! schedule configuration decoupled from concrete function identities.

use core::any::{type_name, TypeId};
use core::fmt;

/// A stable identity for a [`SystemSet`].
///
/// Combines the label type's [`TypeId`] with a caller-supplied discriminant so
/// a single enum type can expose one distinct set per variant while unit-struct
/// labels collapse to discriminant `0`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemSetId {
    type_id: TypeId,
    discriminant: u64,
    name: &'static str,
}

impl SystemSetId {
    /// The id for a unit label type `T` (discriminant `0`).
    #[inline]
    #[must_use]
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            discriminant: 0,
            name: type_name::<T>(),
        }
    }

    /// The id for label type `T` with an explicit `discriminant` (use the enum
    /// variant index for multi-variant labels).
    #[inline]
    #[must_use]
    pub fn with<T: 'static>(discriminant: u64) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            discriminant,
            name: type_name::<T>(),
        }
    }

    /// The label type's name, for diagnostics.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The caller-supplied discriminant.
    #[inline]
    #[must_use]
    pub fn discriminant(&self) -> u64 {
        self.discriminant
    }
}

impl fmt::Debug for SystemSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.discriminant == 0 {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}#{}", self.name, self.discriminant)
        }
    }
}

/// A label that groups systems. Implement for your own unit structs or enums to
/// create ordering/condition anchors.
///
/// For a unit struct the derived-style implementation is a one-liner:
///
/// ```
/// use prism_ecs::schedule::{SystemSet, SystemSetId};
///
/// struct Physics;
/// impl SystemSet for Physics {
///     fn set_id(&self) -> SystemSetId {
///         SystemSetId::of::<Self>()
///     }
/// }
/// ```
pub trait SystemSet: Send + Sync + 'static {
    /// This label's stable identity.
    fn set_id(&self) -> SystemSetId;
}
