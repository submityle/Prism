//! Compact, interned load-error identities.
//!
//! A failed load must carry enough context to diagnose it — which path, which
//! source, why, and who depended on it — but storing a `String` reason inline
//! on every asset's hot state bloats the common success path and makes
//! [`LoadState`](crate::LoadState) non-`Copy`. Instead, failures are recorded
//! once in an [`ErrorRegistry`] and referenced everywhere by a `Copy`
//! [`AssetErrorId`] (a 32-bit index). Load state stays small; the human-
//! readable detail lives in one place and can be rendered by a diagnostics
//! panel, aggregated by root cause (§15 of the design).

use crate::id::UntypedAssetId;
use alloc::string::String;
use alloc::vec::Vec;

/// A compact, `Copy` reference to an error record held by an [`ErrorRegistry`].
///
/// Meaningful only together with the registry that minted it; an id from one
/// registry must not be resolved against another.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AssetErrorId(u32);

impl AssetErrorId {
    /// The raw index value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A recorded load failure with full diagnostic context.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AssetError {
    /// The asset path (or best-known locator) that failed to load.
    pub path: String,
    /// A human-readable description of what went wrong.
    pub reason: String,
    /// The dependent asset that triggered this load, if the failure surfaced
    /// while resolving a dependency closure rather than a direct request.
    pub dependent: Option<UntypedAssetId>,
}

impl AssetError {
    /// Creates an error record for a direct load failure (no dependent).
    #[must_use]
    pub fn new(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            reason: reason.into(),
            dependent: None,
        }
    }

    /// Sets the dependent asset whose closure surfaced this failure.
    #[must_use]
    pub fn with_dependent(mut self, dependent: UntypedAssetId) -> Self {
        self.dependent = Some(dependent);
        self
    }
}

/// An append-only registry mapping [`AssetErrorId`]s to [`AssetError`] records.
///
/// Records are never moved or removed while ids referencing them may exist, so
/// an [`AssetErrorId`] stays valid for the registry's lifetime. The registry is
/// intentionally simple and single-owner; cross-thread error reporting funnels
/// through the load scheduler (a later milestone) rather than sharing this
/// directly.
#[derive(Default)]
pub struct ErrorRegistry {
    errors: Vec<AssetError>,
}

impl ErrorRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Records `error`, returning a `Copy` id that resolves back to it.
    pub fn record(&mut self, error: AssetError) -> AssetErrorId {
        let id = AssetErrorId(u32::try_from(self.errors.len()).expect("error count fits u32"));
        self.errors.push(error);
        id
    }

    /// Resolves an id to its record, or `None` if the id is out of range (for
    /// example minted by a different registry).
    #[must_use]
    pub fn get(&self, id: AssetErrorId) -> Option<&AssetError> {
        self.errors.get(id.0 as usize)
    }

    /// The number of recorded errors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Whether no errors have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Iterates over `(id, &error)` for every recorded failure.
    pub fn iter(&self) -> impl Iterator<Item = (AssetErrorId, &AssetError)> {
        self.errors
            .iter()
            .enumerate()
            .map(|(i, e)| (AssetErrorId(i as u32), e))
    }
}
