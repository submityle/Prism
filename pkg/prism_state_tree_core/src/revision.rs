//! Plan revision identity and host-owned hot-reload compatibility decisions.
//!
//! This module deliberately does not inspect a [`Plan`](crate::Plan), assign
//! identities to its nodes, or translate [`StateId`](crate::StateId) values.
//! A host supplies a stable semantic fingerprint and, when a revision changes,
//! records its own decisions using opaque stable keys.  This keeps asset
//! identity, instance storage, and runtime state ownership outside the core
//! contract while still giving every host the same bounded and deterministic
//! hand-off protocol.

use alloc::vec::Vec;

/// Maximum number of per-revision migration records retained by one decision
/// set.
///
/// The limit bounds the memory that this contract can request from a host. A
/// host may select any smaller capacity, including zero when it only needs a
/// whole-instance reset or rejection decision.
pub const MAX_MIGRATION_DECISIONS: usize = 4096;

/// Stable semantic identity of a compiled plan revision.
///
/// The value is supplied by the host from a stable fingerprint. It is not a
/// version counter and it does not expose or infer any `StateId` mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlanRevision(u128);

impl PlanRevision {
    /// Creates a revision from a stable semantic fingerprint.
    #[must_use]
    pub const fn from_fingerprint(fingerprint: u128) -> Self {
        Self(fingerprint)
    }

    /// Alias for [`PlanRevision::from_fingerprint`].
    #[must_use]
    pub const fn new(fingerprint: u128) -> Self {
        Self::from_fingerprint(fingerprint)
    }

    /// Returns the stable fingerprint represented by this revision.
    #[must_use]
    pub const fn fingerprint(self) -> u128 {
        self.0
    }

    /// Classifies a proposed hot-reload before any host migration decisions
    /// are supplied.
    #[must_use]
    pub const fn classify_against(self, next: Self) -> CompatibilityClass {
        if self.0 == next.0 {
            CompatibilityClass::Exact
        } else {
            CompatibilityClass::RequiresMigration
        }
    }
}

/// Compatibility classification for a plan revision hand-off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompatibilityClass {
    /// The source and target fingerprints are identical; no migration is
    /// needed.
    Exact,
    /// The fingerprints differ and a host decision set has not accepted the
    /// hand-off yet.
    RequiresMigration,
    /// The host has explicitly accepted the changed revision after recording
    /// the decisions it owns.
    Migrated,
    /// The host has explicitly rejected the changed revision.
    Incompatible,
}

/// Opaque stable identity used by a host's migration records.
///
/// This key is intentionally a separate type from [`StateId`](crate::StateId)
/// and cannot be used as a state index. It may represent an asset node,
/// instance slot, gameplay object, or any other identity whose lifetime is
/// governed by the host.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MigrationKey(u128);

impl MigrationKey {
    /// Creates an opaque migration key from a host-owned stable identity.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Returns the host-owned identity value.
    #[must_use]
    pub const fn value(self) -> u128 {
        self.0
    }
}

/// Host decision for one stable identity during a revision hand-off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationAction {
    /// Keep the host-owned instance data and semantic identity alive. The
    /// host remains responsible for validating that preservation is safe.
    Preserve,
    /// Keep the identity but reconstruct its host-owned data from the target
    /// revision's defaults.
    Reinitialize,
    /// Remove the host-owned data associated with the identity.
    Remove,
}

/// One bounded, host-owned migration decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationDecision {
    key: MigrationKey,
    action: MigrationAction,
}

impl MigrationDecision {
    /// Creates a decision for one opaque stable identity.
    #[must_use]
    pub const fn new(key: MigrationKey, action: MigrationAction) -> Self {
        Self { key, action }
    }

    /// Returns the opaque stable identity covered by this decision.
    #[must_use]
    pub const fn key(self) -> MigrationKey {
        self.key
    }

    /// Returns the host action for this identity.
    #[must_use]
    pub const fn action(self) -> MigrationAction {
        self.action
    }
}

/// Structured reason why a host rejected a changed plan revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationRejection {
    /// The changed plan alters semantics that the host cannot preserve.
    SemanticChange,
    /// A required stable identity is absent from the target revision.
    RequiredIdentityMissing(MigrationKey),
    /// The host policy rejects this hot reload independently of individual
    /// identity decisions.
    HostPolicy,
}

/// Structured failure from the revision compatibility contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RevisionError {
    /// The requested record capacity exceeds the contract's hard bound.
    CapacityLimit {
        /// Requested record capacity.
        requested: usize,
        /// Maximum supported record capacity.
        maximum: usize,
    },
    /// A new record would exceed the decision set's selected capacity.
    DecisionCapacityExceeded {
        /// Configured capacity.
        capacity: usize,
    },
    /// The same stable identity was recorded more than once.
    DuplicateDecision(MigrationKey),
    /// A decision set for an unchanged revision cannot contain migration
    /// records or a migration verdict.
    UnnecessaryMigration,
    /// The decision set has already received an accept/reject verdict.
    AlreadyFinalized,
    /// A changed revision must be finalized before its compatibility can be
    /// classified as accepted or rejected.
    NotFinalized,
}

/// Final host verdict for a changed plan revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationVerdict {
    /// The host accepts the target revision after applying its decisions.
    Accepted,
    /// The host rejects the target revision for the supplied reason.
    Rejected(MigrationRejection),
}

/// Bounded host decisions for one source-to-target plan revision hand-off.
///
/// The set is an accounting and compatibility contract only. It never
/// rewrites an active configuration, converts a [`MigrationKey`] to a
/// [`StateId`](crate::StateId), or claims that a host's `Preserve` action is
/// semantically valid. Those operations remain explicit host responsibilities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationDecisionSet {
    source: PlanRevision,
    target: PlanRevision,
    capacity: usize,
    decisions: Vec<MigrationDecision>,
    verdict: Option<MigrationVerdict>,
}

impl MigrationDecisionSet {
    /// Creates an empty bounded decision set.
    pub fn new(
        source: PlanRevision,
        target: PlanRevision,
        capacity: usize,
    ) -> Result<Self, RevisionError> {
        if capacity > MAX_MIGRATION_DECISIONS {
            return Err(RevisionError::CapacityLimit {
                requested: capacity,
                maximum: MAX_MIGRATION_DECISIONS,
            });
        }
        Ok(Self {
            source,
            target,
            capacity,
            decisions: Vec::with_capacity(capacity),
            verdict: None,
        })
    }

    /// Returns the source revision.
    #[must_use]
    pub const fn source(&self) -> PlanRevision {
        self.source
    }

    /// Returns the target revision.
    #[must_use]
    pub const fn target(&self) -> PlanRevision {
        self.target
    }

    /// Returns the configured record capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of recorded decisions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Returns whether no identity decisions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Returns decisions in the order supplied by the host.
    #[must_use]
    pub fn decisions(&self) -> &[MigrationDecision] {
        &self.decisions
    }

    /// Returns the first decision for a stable identity, if present.
    #[must_use]
    pub fn decision_for(&self, key: MigrationKey) -> Option<MigrationDecision> {
        self.decisions
            .iter()
            .copied()
            .find(|decision| decision.key == key)
    }

    /// Returns the verdict, if this changed revision has been finalized.
    #[must_use]
    pub const fn verdict(&self) -> Option<MigrationVerdict> {
        self.verdict
    }

    /// Adds one decision without allowing duplicate keys or unbounded growth.
    pub fn try_push(&mut self, decision: MigrationDecision) -> Result<(), RevisionError> {
        if self.source == self.target {
            return Err(RevisionError::UnnecessaryMigration);
        }
        if self.verdict.is_some() {
            return Err(RevisionError::AlreadyFinalized);
        }
        if self.decision_for(decision.key).is_some() {
            return Err(RevisionError::DuplicateDecision(decision.key));
        }
        if self.decisions.len() >= self.capacity {
            return Err(RevisionError::DecisionCapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.decisions.push(decision);
        Ok(())
    }

    /// Accepts the changed revision after the host has completed its own
    /// decision application.
    pub fn accept(&mut self) -> Result<(), RevisionError> {
        self.finalize(MigrationVerdict::Accepted)
    }

    /// Rejects the changed revision with a structured host reason.
    pub fn reject(&mut self, reason: MigrationRejection) -> Result<(), RevisionError> {
        self.finalize(MigrationVerdict::Rejected(reason))
    }

    /// Classifies the hand-off using revision equality and the host verdict.
    pub fn classify(&self) -> Result<CompatibilityClass, RevisionError> {
        if self.source == self.target {
            return Ok(CompatibilityClass::Exact);
        }
        match self.verdict {
            None => Err(RevisionError::NotFinalized),
            Some(MigrationVerdict::Accepted) => Ok(CompatibilityClass::Migrated),
            Some(MigrationVerdict::Rejected(_)) => Ok(CompatibilityClass::Incompatible),
        }
    }

    fn finalize(&mut self, verdict: MigrationVerdict) -> Result<(), RevisionError> {
        if self.source == self.target {
            return Err(RevisionError::UnnecessaryMigration);
        }
        if self.verdict.is_some() {
            return Err(RevisionError::AlreadyFinalized);
        }
        self.verdict = Some(verdict);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompatibilityClass, MigrationAction, MigrationDecision, MigrationDecisionSet, MigrationKey,
        MigrationRejection, PlanRevision, RevisionError, MAX_MIGRATION_DECISIONS,
    };

    #[test]
    fn fingerprint_identity_is_stable_and_classifies_changes() {
        let first = PlanRevision::from_fingerprint(0x11);
        let same = PlanRevision::new(0x11);
        let next = PlanRevision::new(0x22);

        assert_eq!(first, same);
        assert_eq!(first.fingerprint(), 0x11);
        assert_eq!(first.classify_against(same), CompatibilityClass::Exact);
        assert_eq!(
            first.classify_against(next),
            CompatibilityClass::RequiresMigration
        );
    }

    #[test]
    fn changed_revision_requires_explicit_host_verdict() {
        let source = PlanRevision::new(1);
        let target = PlanRevision::new(2);
        let mut set = MigrationDecisionSet::new(source, target, 2).unwrap();

        assert_eq!(set.classify(), Err(RevisionError::NotFinalized));
        set.try_push(MigrationDecision::new(
            MigrationKey::new(7),
            MigrationAction::Preserve,
        ))
        .unwrap();
        set.accept().unwrap();
        assert_eq!(set.classify(), Ok(CompatibilityClass::Migrated));
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.decision_for(MigrationKey::new(7)).unwrap().action(),
            MigrationAction::Preserve
        );
    }

    #[test]
    fn bounded_records_reject_duplicates_and_overflow() {
        let mut set =
            MigrationDecisionSet::new(PlanRevision::new(1), PlanRevision::new(2), 1).unwrap();
        let decision = MigrationDecision::new(MigrationKey::new(9), MigrationAction::Remove);
        set.try_push(decision).unwrap();
        assert_eq!(
            set.try_push(decision),
            Err(RevisionError::DuplicateDecision(MigrationKey::new(9)))
        );
        assert_eq!(
            set.try_push(MigrationDecision::new(
                MigrationKey::new(10),
                MigrationAction::Reinitialize,
            )),
            Err(RevisionError::DecisionCapacityExceeded { capacity: 1 })
        );
    }

    #[test]
    fn rejection_is_structured_and_finalization_is_one_shot() {
        let key = MigrationKey::new(17);
        let mut set =
            MigrationDecisionSet::new(PlanRevision::new(3), PlanRevision::new(4), 0).unwrap();
        set.reject(MigrationRejection::RequiredIdentityMissing(key))
            .unwrap();
        assert_eq!(set.classify(), Ok(CompatibilityClass::Incompatible));
        assert_eq!(
            set.try_push(MigrationDecision::new(key, MigrationAction::Preserve)),
            Err(RevisionError::AlreadyFinalized)
        );
        assert_eq!(set.accept(), Err(RevisionError::AlreadyFinalized));
    }

    #[test]
    fn exact_revision_never_claims_migration() {
        let revision = PlanRevision::new(5);
        let mut set = MigrationDecisionSet::new(revision, revision, 1).unwrap();

        assert_eq!(set.classify(), Ok(CompatibilityClass::Exact));
        assert_eq!(
            set.try_push(MigrationDecision::new(
                MigrationKey::new(1),
                MigrationAction::Preserve,
            )),
            Err(RevisionError::UnnecessaryMigration)
        );
        assert_eq!(set.accept(), Err(RevisionError::UnnecessaryMigration));
    }

    #[test]
    fn capacity_has_a_hard_bound() {
        assert_eq!(
            MigrationDecisionSet::new(
                PlanRevision::new(1),
                PlanRevision::new(2),
                MAX_MIGRATION_DECISIONS + 1,
            ),
            Err(RevisionError::CapacityLimit {
                requested: MAX_MIGRATION_DECISIONS + 1,
                maximum: MAX_MIGRATION_DECISIONS,
            })
        );
    }
}
