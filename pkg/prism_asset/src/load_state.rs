//! Per-asset and recursive dependency load progress.
//!
//! Both states are `Copy`: a failure is referenced by a compact
//! [`AssetErrorId`] into an [`ErrorRegistry`](crate::ErrorRegistry) rather than
//! an inline `String`, so the hot success path stays small and these states can
//! be stored by value in components and copied across the main/render split.
//! The human-readable reason lives once in the registry and is rendered on
//! demand by diagnostics (§15 of the design).

use crate::error::AssetErrorId;

/// The load progress of a single asset's own data, independent of its
/// dependencies.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum LoadState {
    /// No load has been requested yet.
    #[default]
    NotLoaded,
    /// A load is in flight.
    Loading,
    /// The asset's own data finished loading successfully.
    Loaded,
    /// The load failed; the id resolves to the recorded reason in the owning
    /// [`ErrorRegistry`](crate::ErrorRegistry).
    Failed(AssetErrorId),
}

impl LoadState {
    /// Whether the asset's own data has finished loading.
    #[must_use]
    pub const fn is_loaded(self) -> bool {
        matches!(self, Self::Loaded)
    }

    /// Whether a load is currently in flight.
    #[must_use]
    pub const fn is_loading(self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Whether the load failed.
    #[must_use]
    pub const fn is_failed(self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// The error id if this state is [`LoadState::Failed`], else `None`.
    #[must_use]
    pub const fn error(self) -> Option<AssetErrorId> {
        match self {
            Self::Failed(id) => Some(id),
            _ => None,
        }
    }
}

/// The aggregate load state of an asset together with its full transitive
/// dependency closure, answering "is everything this asset needs ready?".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum RecursiveDependencyLoadState {
    /// At least one asset in the closure has not started loading.
    #[default]
    NotLoaded,
    /// The closure is still loading and nothing has failed.
    Loading,
    /// Every asset in the closure loaded successfully.
    Loaded,
    /// At least one asset in the closure failed; carries the first failure's
    /// error id.
    Failed(AssetErrorId),
}

impl RecursiveDependencyLoadState {
    /// Folds another closure state into this one using worst-wins precedence:
    /// `Failed` dominates, then `Loading`, then `NotLoaded`, and only an
    /// all-`Loaded` closure stays `Loaded`. When both sides failed, the
    /// left-hand (earlier-observed) failure is kept so propagation is stable.
    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        match (self, other) {
            (failed @ Self::Failed(_), _) | (_, failed @ Self::Failed(_)) => failed,
            (Self::Loading, _) | (_, Self::Loading) => Self::Loading,
            (Self::NotLoaded, _) | (_, Self::NotLoaded) => Self::NotLoaded,
            (Self::Loaded, Self::Loaded) => Self::Loaded,
        }
    }

    /// Whether the entire dependency closure is ready.
    #[must_use]
    pub const fn is_loaded(self) -> bool {
        matches!(self, Self::Loaded)
    }

    /// Whether a load is still in flight anywhere in the closure.
    #[must_use]
    pub const fn is_loading(self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Whether any asset in the closure failed.
    #[must_use]
    pub const fn is_failed(self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// The first error id if this closure state is
    /// [`RecursiveDependencyLoadState::Failed`], else `None`.
    #[must_use]
    pub const fn error(self) -> Option<AssetErrorId> {
        match self {
            Self::Failed(id) => Some(id),
            _ => None,
        }
    }
}

impl From<LoadState> for RecursiveDependencyLoadState {
    fn from(state: LoadState) -> Self {
        match state {
            LoadState::NotLoaded => Self::NotLoaded,
            LoadState::Loading => Self::Loading,
            LoadState::Loaded => Self::Loaded,
            LoadState::Failed(id) => Self::Failed(id),
        }
    }
}
