//! Change events emitted by [`Assets`](crate::Assets) storage.

use crate::error::AssetErrorId;
use crate::id::AssetId;
use core::fmt;
use core::mem;

/// A change notification for assets of type `A`, drained from
/// [`Assets`](crate::Assets) via [`Assets::drain_events`](crate::Assets::drain_events).
pub enum AssetEvent<A: ?Sized> {
    /// A new asset was inserted.
    Added {
        /// The id of the newly inserted asset.
        id: AssetId<A>,
    },
    /// An existing asset was mutated (handed out via `get_mut`).
    Modified {
        /// The id of the mutated asset.
        id: AssetId<A>,
    },
    /// An asset was removed and its slot freed.
    Removed {
        /// The id of the removed asset.
        id: AssetId<A>,
    },
    /// An asset's load failed. The `error` resolves to the recorded reason in
    /// the owning [`ErrorRegistry`](crate::ErrorRegistry).
    Failed {
        /// The id of the asset whose load failed.
        id: AssetId<A>,
        /// A compact reference to the recorded failure detail.
        error: AssetErrorId,
    },
    /// An asset and its full dependency closure finished loading.
    LoadedWithDependencies {
        /// The id whose dependency closure is now fully ready.
        id: AssetId<A>,
    },
}

impl<A: ?Sized> AssetEvent<A> {
    /// The asset id this event refers to.
    #[must_use]
    pub fn id(&self) -> AssetId<A> {
        match self {
            Self::Added { id }
            | Self::Modified { id }
            | Self::Removed { id }
            | Self::Failed { id, .. }
            | Self::LoadedWithDependencies { id } => *id,
        }
    }

    /// The failure detail id if this is an [`AssetEvent::Failed`], else `None`.
    #[must_use]
    pub fn error(&self) -> Option<AssetErrorId> {
        match self {
            Self::Failed { error, .. } => Some(*error),
            _ => None,
        }
    }

    /// Whether this is an [`AssetEvent::Added`].
    #[must_use]
    pub fn is_added(&self) -> bool {
        matches!(self, Self::Added { .. })
    }

    /// Whether this is an [`AssetEvent::Modified`].
    #[must_use]
    pub fn is_modified(&self) -> bool {
        matches!(self, Self::Modified { .. })
    }

    /// Whether this is an [`AssetEvent::Removed`].
    #[must_use]
    pub fn is_removed(&self) -> bool {
        matches!(self, Self::Removed { .. })
    }

    /// Whether this is an [`AssetEvent::Failed`].
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// Whether this is an [`AssetEvent::LoadedWithDependencies`].
    #[must_use]
    pub fn is_loaded_with_dependencies(&self) -> bool {
        matches!(self, Self::LoadedWithDependencies { .. })
    }
}

// Manual trait impls so `A` need not implement these traits: the event only
// stores `Copy` payloads (an `AssetId<A>` whose type tag is a function pointer,
// plus an optional `Copy` error id).
impl<A: ?Sized> Clone for AssetEvent<A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: ?Sized> Copy for AssetEvent<A> {}

impl<A: ?Sized> PartialEq for AssetEvent<A> {
    fn eq(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
            && self.id() == other.id()
            && self.error() == other.error()
    }
}

impl<A: ?Sized> Eq for AssetEvent<A> {}

impl<A: ?Sized> fmt::Debug for AssetEvent<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed { id, error } => f
                .debug_struct("Failed")
                .field("id", id)
                .field("error", error)
                .finish(),
            other => {
                let name = match other {
                    Self::Added { .. } => "Added",
                    Self::Modified { .. } => "Modified",
                    Self::Removed { .. } => "Removed",
                    Self::LoadedWithDependencies { .. } => "LoadedWithDependencies",
                    Self::Failed { .. } => unreachable!("handled above"),
                };
                f.debug_struct(name).field("id", &other.id()).finish()
            }
        }
    }
}
