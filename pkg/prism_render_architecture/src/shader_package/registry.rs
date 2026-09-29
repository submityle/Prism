//! Registry indexing shader packages by [`ShaderPackageId`].
//!
//! The registry owns each package's [`ShaderPackageManifest`] and its current
//! [`PackageState`]. Newly registered packages start in `Validating`; callers
//! then drive them to `Ready` or `Rejected`, either directly through the
//! checked state machine or via [`ShaderPackageRegistry::validate`], which runs
//! manifest validation and applies the resulting transition.
//!
//! A [`alloc::collections::BTreeMap`] backs the index so iteration order is
//! deterministic across runs.

use alloc::collections::BTreeMap;

use super::manifest::{ManifestExpectation, ValidationOutcome};
use super::state::TransitionError;
use super::{PackageState, ShaderPackageId, ShaderPackageManifest};

/// Why a package could not be registered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegisterError {
    /// A package with this identifier is already registered.
    Duplicate(ShaderPackageId),
}

/// A stored manifest paired with its lifecycle state.
#[derive(Clone, Debug)]
pub struct RegistryEntry {
    /// The registered manifest.
    pub manifest: ShaderPackageManifest,
    /// The package's current lifecycle state.
    pub state: PackageState,
}

/// Deterministic registry of shader packages keyed by identifier.
#[derive(Clone, Debug, Default)]
pub struct ShaderPackageRegistry {
    entries: BTreeMap<ShaderPackageId, RegistryEntry>,
}

impl ShaderPackageRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Registers `manifest`, starting the package in [`PackageState::Validating`].
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError::Duplicate`] if the identifier is already present.
    pub fn register(&mut self, manifest: ShaderPackageManifest) -> Result<(), RegisterError> {
        if self.entries.contains_key(&manifest.id) {
            return Err(RegisterError::Duplicate(manifest.id.clone()));
        }
        let id = manifest.id.clone();
        self.entries.insert(
            id,
            RegistryEntry {
                manifest,
                state: PackageState::Validating,
            },
        );
        Ok(())
    }

    /// Returns the stored entry for `id`, if any.
    #[must_use]
    pub fn get(&self, id: &ShaderPackageId) -> Option<&RegistryEntry> {
        self.entries.get(id)
    }

    /// Returns the current lifecycle state for `id`, if registered.
    #[must_use]
    pub fn state_of(&self, id: &ShaderPackageId) -> Option<PackageState> {
        self.entries.get(id).map(|entry| entry.state)
    }

    /// Reports whether `id` is registered.
    #[must_use]
    pub fn contains(&self, id: &ShaderPackageId) -> bool {
        self.entries.contains_key(id)
    }

    /// Number of registered packages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Reports whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates entries in deterministic identifier order.
    pub fn iter(&self) -> impl Iterator<Item = (&ShaderPackageId, &RegistryEntry)> {
        self.entries.iter()
    }

    /// Applies a checked state transition to a registered package.
    ///
    /// # Errors
    ///
    /// Returns `Some(Err(..))` when the transition is illegal, `None` when the
    /// package is not registered, and `Some(Ok(new_state))` on success.
    pub fn transition(
        &mut self,
        id: &ShaderPackageId,
        next: PackageState,
    ) -> Option<Result<PackageState, TransitionError>> {
        let entry = self.entries.get_mut(id)?;
        Some(match entry.state.try_transition(next) {
            Ok(state) => {
                entry.state = state;
                Ok(state)
            }
            Err(err) => Err(err),
        })
    }

    /// Validates a registered package and drives it to `Ready`/`Rejected`.
    ///
    /// Returns `None` when `id` is not registered. On success the entry's state
    /// reflects the outcome and the [`ValidationOutcome`] is returned.
    pub fn validate(
        &mut self,
        id: &ShaderPackageId,
        expectation: &ManifestExpectation,
    ) -> Option<ValidationOutcome> {
        let entry = self.entries.get_mut(id)?;
        // Ensure we are in a validating state before recording a verdict.
        if !matches!(entry.state, PackageState::Validating) {
            entry.state = PackageState::Validating;
        }
        let outcome = entry.manifest.validate(expectation);
        entry.state = if outcome.is_accepted() {
            PackageState::Ready
        } else {
            PackageState::Rejected
        };
        Some(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{AbiHash, AbiVersion};
    use alloc::string::ToString;
    use alloc::vec::Vec;

    fn manifest(id: &str, hash: AbiHash) -> ShaderPackageManifest {
        ShaderPackageManifest {
            id: ShaderPackageId::new(id),
            source_revision: "rev".to_string(),
            entry_point: "main".to_string(),
            abi_version: AbiVersion::from_parts(2, 4),
            abi_hash: hash,
            permutation_count: 2,
        }
    }

    #[test]
    fn register_and_query() {
        let mut registry = ShaderPackageRegistry::new();
        let hash = AbiHash::from_bytes(b"h");
        registry.register(manifest("a", hash)).unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(&ShaderPackageId::new("a")));
        assert_eq!(
            registry.state_of(&ShaderPackageId::new("a")),
            Some(PackageState::Validating)
        );
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut registry = ShaderPackageRegistry::new();
        let hash = AbiHash::from_bytes(b"h");
        registry.register(manifest("a", hash)).unwrap();
        assert_eq!(
            registry.register(manifest("a", hash)),
            Err(RegisterError::Duplicate(ShaderPackageId::new("a")))
        );
    }

    #[test]
    fn validate_drives_to_ready() {
        let mut registry = ShaderPackageRegistry::new();
        let hash = AbiHash::from_bytes(b"h");
        registry.register(manifest("a", hash)).unwrap();
        let expect = ManifestExpectation::new(AbiVersion::from_parts(2, 0), hash);
        let outcome = registry
            .validate(&ShaderPackageId::new("a"), &expect)
            .unwrap();
        assert!(outcome.is_accepted());
        assert_eq!(
            registry.state_of(&ShaderPackageId::new("a")),
            Some(PackageState::Ready)
        );
    }

    #[test]
    fn validate_drives_to_rejected_on_mismatch() {
        let mut registry = ShaderPackageRegistry::new();
        registry
            .register(manifest("a", AbiHash::from_bytes(b"good")))
            .unwrap();
        let expect =
            ManifestExpectation::new(AbiVersion::from_parts(2, 0), AbiHash::from_bytes(b"bad"));
        let outcome = registry
            .validate(&ShaderPackageId::new("a"), &expect)
            .unwrap();
        assert!(!outcome.is_accepted());
        assert_eq!(
            registry.state_of(&ShaderPackageId::new("a")),
            Some(PackageState::Rejected)
        );
    }

    #[test]
    fn transition_reports_illegal_moves() {
        let mut registry = ShaderPackageRegistry::new();
        registry
            .register(manifest("a", AbiHash::from_bytes(b"h")))
            .unwrap();
        // Validating -> Missing is illegal.
        let result = registry.transition(&ShaderPackageId::new("a"), PackageState::Missing);
        assert!(matches!(result, Some(Err(_))));
        // Unknown id yields None.
        assert!(registry
            .transition(&ShaderPackageId::new("missing"), PackageState::Ready)
            .is_none());
    }

    #[test]
    fn iteration_is_ordered_by_id() {
        let mut registry = ShaderPackageRegistry::new();
        let hash = AbiHash::from_bytes(b"h");
        registry.register(manifest("zeta", hash)).unwrap();
        registry.register(manifest("alpha", hash)).unwrap();
        let ids: Vec<&str> = registry.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["alpha", "zeta"]);
    }
}
