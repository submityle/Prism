//! Per-asset and recursive dependency load progress.

use alloc::string::String;

/// The load progress of a single asset's own data, independent of its
/// dependencies.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum LoadState {
    /// No load has been requested yet.
    #[default]
    NotLoaded,
    /// A load is in flight.
    Loading,
    /// The asset's own data finished loading successfully.
    Loaded,
    /// The load failed; the string carries a human-readable reason.
    Failed(String),
}

impl LoadState {
    /// Whether the asset's own data has finished loading.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded)
    }

    /// Whether a load is currently in flight.
    #[must_use]
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Whether the load failed.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// The aggregate load state of an asset together with its full transitive
/// dependency closure, answering "is everything this asset needs ready?".
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum RecursiveDependencyLoadState {
    /// At least one asset in the closure has not started loading.
    #[default]
    NotLoaded,
    /// The closure is still loading and nothing has failed.
    Loading,
    /// Every asset in the closure loaded successfully.
    Loaded,
    /// At least one asset in the closure failed; carries the first reason.
    Failed(String),
}

impl RecursiveDependencyLoadState {
    /// Folds another closure state into this one using worst-wins precedence:
    /// `Failed` dominates, then `Loading`, then `NotLoaded`, and only an
    /// all-`Loaded` closure stays `Loaded`.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (failed @ Self::Failed(_), _) | (_, failed @ Self::Failed(_)) => failed,
            (Self::Loading, _) | (_, Self::Loading) => Self::Loading,
            (Self::NotLoaded, _) | (_, Self::NotLoaded) => Self::NotLoaded,
            (Self::Loaded, Self::Loaded) => Self::Loaded,
        }
    }

    /// Whether the entire dependency closure is ready.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded)
    }

    /// Whether any asset in the closure failed.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

impl From<&LoadState> for RecursiveDependencyLoadState {
    fn from(state: &LoadState) -> Self {
        match state {
            LoadState::NotLoaded => Self::NotLoaded,
            LoadState::Loading => Self::Loading,
            LoadState::Loaded => Self::Loaded,
            LoadState::Failed(reason) => Self::Failed(reason.clone()),
        }
    }
}
