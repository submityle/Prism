//! Type-owned presentation substitutes, independent of asset identity/readiness.
//!
//! Fallback bytes are a separate, explicit reservation, not an unaccounted
//! pinned arena asset. The caller supplies the full resident byte charge
//! (including owned allocations); the runtime cannot infer a generic `A`'s
//! heap footprint. Rejected replacements leave the previous registration intact.

use crate::asset::Asset;
use crate::load_state::LoadState;
use core::fmt;

/// Which presentation substitute to register for an asset type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackKind {
    /// A temporary representation while the requested value is loading.
    Placeholder,
    /// A missing/error representation; also used while loading if no placeholder exists.
    Failure,
}

/// The origin of a value returned by a fallback-aware read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackSource {
    /// The requested asset itself is ready.
    Asset,
    /// A type-registered loading representation.
    Placeholder,
    /// A type-registered missing/error representation.
    Fallback,
}

/// A presentation value paired with the requested asset's **real** load state.
///
/// Substitution never makes dependency readiness succeed or clears a failure.
/// Consumers that require real payloads should continue using `Assets::get`.
#[derive(Debug)]
pub struct AssetView<'a, A> {
    /// The ready value or the registered type-correct substitute.
    pub value: &'a A,
    /// The requested slot's state, not the substitute's state.
    pub load_state: LoadState,
    /// Whether this value is the requested payload or a substitute.
    pub source: FallbackSource,
}

/// A fallback reservation would exceed its explicit byte limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FallbackBudgetError {
    /// Combined charge of both substitutes after the proposed operation.
    pub required_bytes: u128,
    /// Configured upper bound, including both substitutes.
    pub budget_bytes: u64,
}

// Manual error implementations keep this no_std core dependency-free; adding
// an error-derive dependency belongs to the crate manifest owner.
impl fmt::Display for FallbackBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fallback needs {} bytes; budget is {}",
            self.required_bytes, self.budget_bytes
        )
    }
}

impl core::error::Error for FallbackBudgetError {}

struct Substitute<A> {
    value: A,
    bytes: u64,
}

/// At most two typed substitutes, subject to a shared hard byte reservation.
///
/// No built-in texture, geometry, or audio policy is imposed. Values live
/// outside arena slots, so replacing one cannot alias a stale asset generation,
/// mint a fake ready event, or resurrect an expired weak handle.
pub struct AssetFallbacks<A> {
    budget_bytes: u64,
    placeholder: Option<Substitute<A>>,
    failure: Option<Substitute<A>>,
}

impl<A: Asset> AssetFallbacks<A> {
    /// Creates an empty registry with an explicit resident byte limit.
    #[must_use]
    pub const fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            placeholder: None,
            failure: None,
        }
    }

    /// The shared upper bound for both substitute values.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Total declared bytes currently reserved by registered values.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        // Every insertion is checked in u128 before either value is replaced.
        self.placeholder.as_ref().map_or(0, |s| s.bytes)
            + self.failure.as_ref().map_or(0, |s| s.bytes)
    }

    /// Changes the byte limit atomically; registered substitutes are never evicted implicitly.
    pub fn set_budget_bytes(&mut self, budget_bytes: u64) -> Result<(), FallbackBudgetError> {
        let required_bytes = u128::from(self.used_bytes());
        if required_bytes > u128::from(budget_bytes) {
            return Err(FallbackBudgetError {
                required_bytes,
                budget_bytes,
            });
        }
        self.budget_bytes = budget_bytes;
        Ok(())
    }

    /// Registers a substitute, returning the previous value on success.
    ///
    /// `bytes` must cover all resident allocations owned by `value`. Values are
    /// only exposed immutably, so changing their accounted size requires another
    /// checked registration. Even a zero-byte value occupies one of just two slots.
    pub fn register(
        &mut self,
        kind: FallbackKind,
        value: A,
        bytes: u64,
    ) -> Result<Option<A>, FallbackBudgetError> {
        let old_bytes = self.slot(kind).as_ref().map_or(0, |s| s.bytes);
        let required_bytes = u128::from(self.used_bytes() - old_bytes) + u128::from(bytes);
        if required_bytes > u128::from(self.budget_bytes) {
            return Err(FallbackBudgetError {
                required_bytes,
                budget_bytes: self.budget_bytes,
            });
        }
        Ok(self
            .slot_mut(kind)
            .replace(Substitute { value, bytes })
            .map(|s| s.value))
    }

    /// Unregisters one substitute and immediately releases its byte reservation.
    pub fn clear(&mut self, kind: FallbackKind) -> Option<A> {
        self.slot_mut(kind).take().map(|s| s.value)
    }

    /// Reads a registered substitute without pretending that an arena slot is ready.
    #[must_use]
    pub fn get(&self, kind: FallbackKind) -> Option<&A> {
        self.slot(kind).as_ref().map(|s| &s.value)
    }

    pub(crate) fn resolve(&self, state: LoadState) -> Option<AssetView<'_, A>> {
        let (value, source) = match state {
            LoadState::Loading => match self.get(FallbackKind::Placeholder) {
                Some(value) => (value, FallbackSource::Placeholder),
                None => (self.get(FallbackKind::Failure)?, FallbackSource::Fallback),
            },
            LoadState::Failed(_) => (self.get(FallbackKind::Failure)?, FallbackSource::Fallback),
            LoadState::NotLoaded | LoadState::Loaded => return None,
        };
        Some(AssetView {
            value,
            load_state: state,
            source,
        })
    }

    fn slot(&self, kind: FallbackKind) -> &Option<Substitute<A>> {
        match kind {
            FallbackKind::Placeholder => &self.placeholder,
            FallbackKind::Failure => &self.failure,
        }
    }

    fn slot_mut(&mut self, kind: FallbackKind) -> &mut Option<Substitute<A>> {
        match kind {
            FallbackKind::Placeholder => &mut self.placeholder,
            FallbackKind::Failure => &mut self.failure,
        }
    }
}

impl<A: Asset> Default for AssetFallbacks<A> {
    fn default() -> Self {
        Self::new(0)
    }
}
