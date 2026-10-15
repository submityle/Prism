//! Behavior-pack contracts, compatibility checks, and bounded rollout control.
//!
//! A behavior pack is a versioned provider of stable event and slot contracts.
//! This module keeps loading and rollout decisions separate from the runtime:
//! it validates immutable metadata, selects instances deterministically, and
//! exposes a small state machine that a host can drive once per control tick.
//! No package code is executed here and no host-owned state is rewritten.

use alloc::vec::Vec;

use crate::{EventId, PlanRevision, SlotGuid};

/// Maximum number of dependency requirements in one behavior pack or load
/// context.
pub const MAX_PACK_DEPENDENCIES: usize = 128;
/// Maximum number of provided or required slot contracts.
pub const MAX_PACK_SLOTS: usize = 4096;
/// Maximum number of provided or required event contracts.
pub const MAX_PACK_EVENTS: usize = 4096;
/// Maximum number of plan revisions from which a pack can migrate.
pub const MAX_PACK_MIGRATIONS: usize = 256;
/// Maximum number of deterministic instance groups in a selector.
pub const MAX_INSTANCE_GROUPS: u16 = 4096;
/// One hundred percent expressed in basis points.
pub const BASIS_POINTS: u16 = 10_000;
/// Maximum number of rollout or rollback steps retained by the controller.
pub const MAX_ROLLOUT_STEPS: u16 = 1024;

/// Semantic version used by package and stable-contract metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SemVer {
    major: u32,
    minor: u32,
    patch: u32,
}

impl SemVer {
    /// Creates a semantic version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns the major component.
    #[must_use]
    pub const fn major(self) -> u32 {
        self.major
    }

    /// Returns the minor component.
    #[must_use]
    pub const fn minor(self) -> u32 {
        self.minor
    }

    /// Returns the patch component.
    #[must_use]
    pub const fn patch(self) -> u32 {
        self.patch
    }

    /// Returns whether `provided` is a backward-compatible upgrade of `self`.
    ///
    /// For a non-zero major version, the major component must remain equal and
    /// the provided version must not be older. For `0.x`, the minor component
    /// is treated as the compatibility boundary, matching the usual caret
    /// interpretation for pre-1.0 contracts.
    #[must_use]
    pub const fn compatible_with(self, provided: Self) -> bool {
        if self.major != provided.major || !provided.is_at_least(self) {
            return false;
        }
        if self.major == 0 {
            self.minor == provided.minor
        } else {
            true
        }
    }

    const fn is_at_least(self, required: Self) -> bool {
        self.major > required.major
            || (self.major == required.major
                && (self.minor > required.minor
                    || (self.minor == required.minor && self.patch >= required.patch)))
    }
}

/// Version constraint used by a package dependency requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VersionRequirement {
    /// Requires exactly this version.
    Exact(SemVer),
    /// Requires a compatible version at least as new as this version.
    Compatible(SemVer),
    /// Requires this version or any newer major version.
    AtLeast(SemVer),
}

impl VersionRequirement {
    /// Creates an exact version requirement.
    #[must_use]
    pub const fn exact(version: SemVer) -> Self {
        Self::Exact(version)
    }

    /// Creates a same-major compatible version requirement.
    #[must_use]
    pub const fn compatible(version: SemVer) -> Self {
        Self::Compatible(version)
    }

    /// Creates a lower-bound version requirement.
    #[must_use]
    pub const fn at_least(version: SemVer) -> Self {
        Self::AtLeast(version)
    }

    /// Returns whether an installed version satisfies this requirement.
    #[must_use]
    pub const fn matches(self, installed: SemVer) -> bool {
        match self {
            Self::Exact(required) => {
                required.major == installed.major
                    && required.minor == installed.minor
                    && required.patch == installed.patch
            }
            Self::Compatible(required) => required.compatible_with(installed),
            Self::AtLeast(required) => installed.is_at_least(required),
        }
    }
}

/// Stable identity of a dependency contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DependencyId(pub u128);

/// A package dependency and the version it accepts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DependencyRequirement {
    /// Stable dependency identity.
    pub id: DependencyId,
    /// Accepted dependency versions.
    pub version: VersionRequirement,
}

/// A dependency installed by the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstalledDependency {
    /// Stable dependency identity.
    pub id: DependencyId,
    /// Installed semantic version.
    pub version: SemVer,
}

/// Versioned public slot contract supplied by a behavior pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProvidedSlot {
    /// Stable slot identity. It must never be recycled for another meaning.
    pub guid: SlotGuid,
    /// Semantic version of the slot meaning and encoding.
    pub version: SemVer,
}

impl ProvidedSlot {
    /// Creates a slot contract.
    #[must_use]
    pub const fn new(guid: SlotGuid, version: SemVer) -> Self {
        Self { guid, version }
    }
}

/// Versioned public event contract supplied by a behavior pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProvidedEvent {
    /// Stable event identity. It must never be recycled for another meaning.
    pub id: EventId,
    /// Semantic version of the event payload and meaning.
    pub version: SemVer,
}

impl ProvidedEvent {
    /// Creates an event contract.
    #[must_use]
    pub const fn new(id: EventId, version: SemVer) -> Self {
        Self { id, version }
    }
}

/// Contract category used by bounded definition errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractKind {
    /// Dependency requirement or installation record.
    Dependency,
    /// Slot contract.
    Slot,
    /// Event contract.
    Event,
    /// Plan-revision migration source.
    PlanMigration,
}

/// Error produced while constructing a bounded contract collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackDefinitionError {
    /// The requested collection would exceed its hard bound.
    CapacityExceeded {
        /// Collection category.
        kind: ContractKind,
        /// Requested number of entries after insertion.
        requested: usize,
        /// Maximum number of entries allowed.
        maximum: usize,
    },
    /// A dependency identity occurs more than once.
    DuplicateDependency(DependencyId),
    /// A slot identity occurs more than once.
    DuplicateSlot(SlotGuid),
    /// An event identity occurs more than once.
    DuplicateEvent(EventId),
    /// A migration source revision occurs more than once.
    DuplicatePlanMigration(PlanRevision),
}

/// A validated behavior package and its stable public contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BehaviorPack {
    id: u128,
    version: SemVer,
    plan_revision: PlanRevision,
    engine_requirement: Option<VersionRequirement>,
    requires: Vec<DependencyRequirement>,
    slots: Vec<ProvidedSlot>,
    events: Vec<ProvidedEvent>,
    migrations: Vec<PlanRevision>,
}

impl BehaviorPack {
    /// Creates an empty package contract.
    #[must_use]
    pub const fn new(id: u128, version: SemVer, plan_revision: PlanRevision) -> Self {
        Self {
            id,
            version,
            plan_revision,
            engine_requirement: None,
            requires: Vec::new(),
            slots: Vec::new(),
            events: Vec::new(),
            migrations: Vec::new(),
        }
    }

    /// Returns the stable package identity.
    #[must_use]
    pub const fn id(&self) -> u128 {
        self.id
    }

    /// Returns the package semantic version.
    #[must_use]
    pub const fn version(&self) -> SemVer {
        self.version
    }

    /// Returns the plan revision provided by this package.
    #[must_use]
    pub const fn plan_revision(&self) -> PlanRevision {
        self.plan_revision
    }

    /// Returns the engine version requirement, if one was declared.
    #[must_use]
    pub const fn engine_requirement(&self) -> Option<VersionRequirement> {
        self.engine_requirement
    }

    /// Declares the engine version requirement for this package.
    pub const fn require_engine(&mut self, requirement: VersionRequirement) {
        self.engine_requirement = Some(requirement);
    }

    /// Returns dependency requirements in insertion order.
    #[must_use]
    pub fn requires(&self) -> &[DependencyRequirement] {
        &self.requires
    }

    /// Returns public slot contracts in insertion order.
    #[must_use]
    pub fn slots(&self) -> &[ProvidedSlot] {
        &self.slots
    }

    /// Returns public event contracts in insertion order.
    #[must_use]
    pub fn events(&self) -> &[ProvidedEvent] {
        &self.events
    }

    /// Returns accepted source plan revisions in insertion order.
    #[must_use]
    pub fn migrations(&self) -> &[PlanRevision] {
        &self.migrations
    }

    /// Adds a dependency requirement with duplicate and capacity checks.
    pub fn require_dependency(
        &mut self,
        requirement: DependencyRequirement,
    ) -> Result<(), PackDefinitionError> {
        if self
            .requires
            .iter()
            .any(|existing| existing.id == requirement.id)
        {
            return Err(PackDefinitionError::DuplicateDependency(requirement.id));
        }
        ensure_capacity(
            self.requires.len(),
            MAX_PACK_DEPENDENCIES,
            ContractKind::Dependency,
        )?;
        self.requires.push(requirement);
        Ok(())
    }

    /// Adds a public slot contract with duplicate and capacity checks.
    pub fn provide_slot(&mut self, slot: ProvidedSlot) -> Result<(), PackDefinitionError> {
        if self.slots.iter().any(|existing| existing.guid == slot.guid) {
            return Err(PackDefinitionError::DuplicateSlot(slot.guid));
        }
        ensure_capacity(self.slots.len(), MAX_PACK_SLOTS, ContractKind::Slot)?;
        self.slots.push(slot);
        Ok(())
    }

    /// Adds a public event contract with duplicate and capacity checks.
    pub fn provide_event(&mut self, event: ProvidedEvent) -> Result<(), PackDefinitionError> {
        if self.events.iter().any(|existing| existing.id == event.id) {
            return Err(PackDefinitionError::DuplicateEvent(event.id));
        }
        ensure_capacity(self.events.len(), MAX_PACK_EVENTS, ContractKind::Event)?;
        self.events.push(event);
        Ok(())
    }

    /// Records that this package can migrate from one source plan revision.
    pub fn allow_migration_from(
        &mut self,
        source: PlanRevision,
    ) -> Result<(), PackDefinitionError> {
        if self.migrations.contains(&source) {
            return Err(PackDefinitionError::DuplicatePlanMigration(source));
        }
        ensure_capacity(
            self.migrations.len(),
            MAX_PACK_MIGRATIONS,
            ContractKind::PlanMigration,
        )?;
        self.migrations.push(source);
        Ok(())
    }

    /// Validates all package requirements against a host load context.
    ///
    /// The check is read-only and bounded by the package and context limits.
    /// A successful report is sufficient for the host to admit metadata; it
    /// does not execute package code or apply a migration to an instance.
    pub fn validate_load(
        &self,
        context: &PackLoadContext,
    ) -> Result<PackLoadReport, PackLoadError> {
        if let Some(required) = self.engine_requirement
            && !required.matches(context.engine_version)
        {
            return Err(PackLoadError::EngineVersionMismatch {
                required,
                installed: context.engine_version,
            });
        }
        for requirement in &self.requires {
            let installed = context
                .dependencies
                .iter()
                .find(|dependency| dependency.id == requirement.id)
                .ok_or(PackLoadError::MissingDependency {
                    id: requirement.id,
                    required: requirement.version,
                })?;
            if !requirement.version.matches(installed.version) {
                return Err(PackLoadError::DependencyVersionMismatch {
                    id: requirement.id,
                    required: requirement.version,
                    installed: installed.version,
                });
            }
        }

        for required in &context.required_slots {
            let provided = self
                .slots
                .iter()
                .find(|slot| slot.guid == required.guid)
                .ok_or(PackLoadError::MissingSlot(required.guid))?;
            if !required.version.compatible_with(provided.version) {
                return Err(PackLoadError::SlotVersionMismatch {
                    guid: required.guid,
                    required: required.version,
                    provided: provided.version,
                });
            }
        }

        for required in &context.required_events {
            let provided = self
                .events
                .iter()
                .find(|event| event.id == required.id)
                .ok_or(PackLoadError::MissingEvent(required.id))?;
            if !required.version.compatible_with(provided.version) {
                return Err(PackLoadError::EventVersionMismatch {
                    id: required.id,
                    required: required.version,
                    provided: provided.version,
                });
            }
        }

        let plan = match context.active_plan_revision {
            None => PlanCompatibility::Unpinned,
            Some(active) if active == self.plan_revision => PlanCompatibility::Exact,
            Some(active) if self.migrations.contains(&active) => PlanCompatibility::Migrated,
            Some(active) => {
                return Err(PackLoadError::PlanRevisionNotMigratable {
                    active,
                    provided: self.plan_revision,
                });
            }
        };

        Ok(PackLoadReport {
            package_id: self.id,
            package_version: self.version,
            engine_checked: self.engine_requirement.is_some(),
            plan,
            dependencies_checked: self.requires.len(),
            slots_checked: context.required_slots.len(),
            events_checked: context.required_events.len(),
        })
    }
}

fn ensure_capacity(
    current: usize,
    maximum: usize,
    kind: ContractKind,
) -> Result<(), PackDefinitionError> {
    let requested = current.saturating_add(1);
    if requested > maximum {
        Err(PackDefinitionError::CapacityExceeded {
            kind,
            requested,
            maximum,
        })
    } else {
        Ok(())
    }
}

/// Host-side contract requirements checked when a behavior pack is loaded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackLoadContext {
    engine_version: SemVer,
    dependencies: Vec<InstalledDependency>,
    required_slots: Vec<ProvidedSlot>,
    required_events: Vec<ProvidedEvent>,
    active_plan_revision: Option<PlanRevision>,
}

impl PackLoadContext {
    /// Creates an empty load context for one engine version.
    #[must_use]
    pub const fn new(engine_version: SemVer) -> Self {
        Self {
            engine_version,
            dependencies: Vec::new(),
            required_slots: Vec::new(),
            required_events: Vec::new(),
            active_plan_revision: None,
        }
    }

    /// Returns the engine version supplied to the context.
    #[must_use]
    pub const fn engine_version(&self) -> SemVer {
        self.engine_version
    }

    /// Sets the active plan revision used for migration compatibility.
    pub const fn with_active_plan_revision(mut self, revision: PlanRevision) -> Self {
        self.active_plan_revision = Some(revision);
        self
    }

    /// Adds one installed dependency.
    pub fn install_dependency(
        &mut self,
        dependency: InstalledDependency,
    ) -> Result<(), PackDefinitionError> {
        if self
            .dependencies
            .iter()
            .any(|existing| existing.id == dependency.id)
        {
            return Err(PackDefinitionError::DuplicateDependency(dependency.id));
        }
        ensure_capacity(
            self.dependencies.len(),
            MAX_PACK_DEPENDENCIES,
            ContractKind::Dependency,
        )?;
        self.dependencies.push(dependency);
        Ok(())
    }

    /// Requires one stable slot from the package.
    pub fn require_slot(&mut self, slot: ProvidedSlot) -> Result<(), PackDefinitionError> {
        if self
            .required_slots
            .iter()
            .any(|existing| existing.guid == slot.guid)
        {
            return Err(PackDefinitionError::DuplicateSlot(slot.guid));
        }
        ensure_capacity(
            self.required_slots.len(),
            MAX_PACK_SLOTS,
            ContractKind::Slot,
        )?;
        self.required_slots.push(slot);
        Ok(())
    }

    /// Requires one stable event from the package.
    pub fn require_event(&mut self, event: ProvidedEvent) -> Result<(), PackDefinitionError> {
        if self
            .required_events
            .iter()
            .any(|existing| existing.id == event.id)
        {
            return Err(PackDefinitionError::DuplicateEvent(event.id));
        }
        ensure_capacity(
            self.required_events.len(),
            MAX_PACK_EVENTS,
            ContractKind::Event,
        )?;
        self.required_events.push(event);
        Ok(())
    }
}

/// Plan-revision result included in a successful load report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanCompatibility {
    /// No active revision was supplied by the host.
    Unpinned,
    /// The package and active revision are identical.
    Exact,
    /// The package declares a migration path from the active revision.
    Migrated,
}

/// Evidence returned after a package passes all load checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackLoadReport {
    /// Stable package identity.
    pub package_id: u128,
    /// Package semantic version.
    pub package_version: SemVer,
    /// Whether the package declared and passed an engine requirement.
    pub engine_checked: bool,
    /// Plan-revision compatibility result.
    pub plan: PlanCompatibility,
    /// Number of dependency requirements checked.
    pub dependencies_checked: usize,
    /// Number of required slots checked.
    pub slots_checked: usize,
    /// Number of required events checked.
    pub events_checked: usize,
}

/// Structured refusal to load a behavior package.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackLoadError {
    /// The host engine does not satisfy the package requirement.
    EngineVersionMismatch {
        /// Required engine version constraint.
        required: VersionRequirement,
        /// Installed engine version.
        installed: SemVer,
    },
    /// A required dependency is absent from the host.
    MissingDependency {
        /// Stable dependency identity.
        id: DependencyId,
        /// Required version constraint.
        required: VersionRequirement,
    },
    /// An installed dependency does not satisfy the requirement.
    DependencyVersionMismatch {
        /// Stable dependency identity.
        id: DependencyId,
        /// Required version constraint.
        required: VersionRequirement,
        /// Installed version.
        installed: SemVer,
    },
    /// A required public slot is absent.
    MissingSlot(SlotGuid),
    /// A public slot exists but its version is incompatible.
    SlotVersionMismatch {
        /// Stable slot identity.
        guid: SlotGuid,
        /// Minimum compatible consumer version.
        required: SemVer,
        /// Package-provided version.
        provided: SemVer,
    },
    /// A required public event is absent.
    MissingEvent(EventId),
    /// A public event exists but its version is incompatible.
    EventVersionMismatch {
        /// Stable event identity.
        id: EventId,
        /// Minimum compatible consumer version.
        required: SemVer,
        /// Package-provided version.
        provided: SemVer,
    },
    /// The active plan revision has no declared migration path.
    PlanRevisionNotMigratable {
        /// Host-active plan revision.
        active: PlanRevision,
        /// Package-provided target revision.
        provided: PlanRevision,
    },
}

/// Stable identity of an instance used for deterministic rollout selection.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InstanceId(pub u128);

/// Deterministic group identity assigned to an instance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InstanceGroupId(pub u16);

/// Error from deterministic instance-group selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupSelectionError {
    /// A selector cannot contain zero groups.
    ZeroGroups,
    /// The requested group count exceeds the fixed bound.
    TooManyGroups {
        /// Requested number of groups.
        requested: u16,
        /// Maximum supported group count.
        maximum: u16,
    },
    /// A percentage was outside the inclusive `[0, 10000]` range.
    InvalidBasisPoints(u16),
}

/// Deterministic, allocation-free instance group selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstanceGroupSelector {
    group_count: u16,
    seed: u64,
}

impl InstanceGroupSelector {
    /// Creates a selector with a fixed group count and stable seed.
    pub const fn new(group_count: u16, seed: u64) -> Result<Self, GroupSelectionError> {
        if group_count == 0 {
            return Err(GroupSelectionError::ZeroGroups);
        }
        if group_count > MAX_INSTANCE_GROUPS {
            return Err(GroupSelectionError::TooManyGroups {
                requested: group_count,
                maximum: MAX_INSTANCE_GROUPS,
            });
        }
        Ok(Self { group_count, seed })
    }

    /// Returns the configured number of groups.
    #[must_use]
    pub const fn group_count(self) -> u16 {
        self.group_count
    }

    /// Returns the configured deterministic seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Assigns an instance to a stable group.
    #[must_use]
    pub fn group_for(self, instance: InstanceId) -> InstanceGroupId {
        InstanceGroupId((hash_instance(instance, self.seed) % u64::from(self.group_count)) as u16)
    }

    /// Returns a stable basis-point bucket in `[0, 10000)` for an instance.
    #[must_use]
    pub fn bucket_for(self, instance: InstanceId) -> u16 {
        self.group_bucket(self.group_for(instance))
    }

    /// Returns whether an instance belongs to the selected rollout fraction.
    pub fn selected(
        self,
        instance: InstanceId,
        basis_points: u16,
    ) -> Result<bool, GroupSelectionError> {
        self.selected_group(self.group_for(instance), basis_points)
    }

    /// Returns whether a complete deterministic group is selected.
    pub fn selected_group(
        self,
        group: InstanceGroupId,
        basis_points: u16,
    ) -> Result<bool, GroupSelectionError> {
        if basis_points > BASIS_POINTS {
            return Err(GroupSelectionError::InvalidBasisPoints(basis_points));
        }
        if group.0 >= self.group_count {
            return Ok(false);
        }
        Ok(self.group_bucket(group) < basis_points)
    }

    fn group_bucket(self, group: InstanceGroupId) -> u16 {
        (hash_group(group, self.seed) % u64::from(BASIS_POINTS)) as u16
    }
}

fn hash_instance(instance: InstanceId, seed: u64) -> u64 {
    let folded = (instance.0 as u64) ^ ((instance.0 >> 64) as u64).rotate_left(17);
    let mut value = folded ^ seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash_group(group: InstanceGroupId, seed: u64) -> u64 {
    hash_instance(
        InstanceId(u128::from(group.0)),
        seed ^ 0xd6e8_feb8_6659_fd93,
    )
}

/// Configuration for one bounded rollout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutConfig {
    /// Final candidate percentage in basis points.
    pub target_basis_points: u16,
    /// Percentage moved on each controller step.
    pub step_basis_points: u16,
    /// Maximum number of steps in either the rollout or rollback phase.
    pub max_steps: u16,
    /// Consecutive recorded failures needed to trigger rollback.
    pub failure_threshold: u16,
}

impl RolloutConfig {
    /// Validates and creates a rollout configuration.
    pub const fn new(
        target_basis_points: u16,
        step_basis_points: u16,
        max_steps: u16,
        failure_threshold: u16,
    ) -> Result<Self, RolloutError> {
        if target_basis_points > BASIS_POINTS || target_basis_points == 0 {
            return Err(RolloutError::InvalidTarget(target_basis_points));
        }
        if step_basis_points == 0 || step_basis_points > BASIS_POINTS {
            return Err(RolloutError::InvalidStep(step_basis_points));
        }
        if max_steps == 0 || max_steps > MAX_ROLLOUT_STEPS {
            return Err(RolloutError::InvalidStepBudget(max_steps));
        }
        if failure_threshold == 0 {
            return Err(RolloutError::InvalidFailureThreshold);
        }
        let required_steps = target_basis_points.div_ceil(step_basis_points);
        if required_steps > max_steps {
            return Err(RolloutError::InsufficientStepBudget {
                required: required_steps,
                maximum: max_steps,
            });
        }
        Ok(Self {
            target_basis_points,
            step_basis_points,
            max_steps,
            failure_threshold,
        })
    }
}

/// Lifecycle phase of a rollout controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutState {
    /// Only the stable revision is selected.
    Stable,
    /// The candidate is selected for a bounded fraction of instances.
    RollingOut,
    /// The candidate reached its configured target fraction.
    Active,
    /// The candidate fraction is being reduced toward zero.
    RollingBack,
    /// Rollback completed and the previous stable revision is selected.
    RolledBack,
}

/// Structured rollout-controller failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutError {
    /// A rollout is already active and must be promoted or rolled back first.
    AlreadyInProgress,
    /// The candidate revision equals the stable revision.
    SameRevision,
    /// The target percentage is invalid.
    InvalidTarget(u16),
    /// The step percentage is invalid.
    InvalidStep(u16),
    /// The step budget is invalid or exceeds the hard bound.
    InvalidStepBudget(u16),
    /// The failure threshold cannot be zero.
    InvalidFailureThreshold,
    /// The configured step budget cannot reach the target percentage.
    InsufficientStepBudget {
        /// Minimum number of steps required.
        required: u16,
        /// Configured maximum number of steps.
        maximum: u16,
    },
    /// The requested operation requires an active rollout.
    NotRollingOut,
    /// The requested operation requires a candidate rollout.
    NoCandidate,
    /// The controller has no further transition to perform.
    NoTransition,
}

/// Observable result of one bounded controller transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutEvent {
    /// A candidate rollout was initialized.
    Started {
        /// Candidate plan revision.
        candidate: PlanRevision,
        /// Configured final percentage.
        target_basis_points: u16,
    },
    /// The candidate fraction moved upward.
    Advanced {
        /// Candidate plan revision.
        candidate: PlanRevision,
        /// New candidate percentage.
        basis_points: u16,
    },
    /// The candidate reached its target fraction.
    Activated {
        /// Candidate plan revision.
        candidate: PlanRevision,
    },
    /// Rollback started after a request or failure threshold.
    RollbackStarted {
        /// Candidate plan revision.
        candidate: PlanRevision,
        /// Candidate percentage at rollback start.
        basis_points: u16,
    },
    /// Rollback reached zero candidate instances.
    RolledBack {
        /// Candidate plan revision that was removed.
        candidate: PlanRevision,
    },
    /// The active candidate became the new stable revision.
    Promoted {
        /// Newly stable plan revision.
        revision: PlanRevision,
    },
}

/// Bounded, deterministic rollout and rollback state machine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RolloutController {
    stable_revision: PlanRevision,
    candidate_revision: Option<PlanRevision>,
    selector: InstanceGroupSelector,
    state: RolloutState,
    target_basis_points: u16,
    current_basis_points: u16,
    step_basis_points: u16,
    max_steps: u16,
    steps: u16,
    failure_threshold: u16,
    failures: u16,
}

impl RolloutController {
    /// Creates a controller that selects the supplied stable revision.
    #[must_use]
    pub const fn new(stable_revision: PlanRevision, selector: InstanceGroupSelector) -> Self {
        Self {
            stable_revision,
            candidate_revision: None,
            selector,
            state: RolloutState::Stable,
            target_basis_points: 0,
            current_basis_points: 0,
            step_basis_points: 0,
            max_steps: 0,
            steps: 0,
            failure_threshold: 0,
            failures: 0,
        }
    }

    /// Returns the currently stable revision.
    #[must_use]
    pub const fn stable_revision(&self) -> PlanRevision {
        self.stable_revision
    }

    /// Returns the candidate revision, if a rollout exists.
    #[must_use]
    pub const fn candidate_revision(&self) -> Option<PlanRevision> {
        self.candidate_revision
    }

    /// Returns the controller state.
    #[must_use]
    pub const fn state(&self) -> RolloutState {
        self.state
    }

    /// Returns the current candidate fraction in basis points.
    #[must_use]
    pub const fn basis_points(&self) -> u16 {
        self.current_basis_points
    }

    /// Returns the number of transitions already consumed by this operation.
    #[must_use]
    pub const fn steps(&self) -> u16 {
        self.steps
    }

    /// Returns the number of recorded failures for the current candidate.
    #[must_use]
    pub const fn failures(&self) -> u16 {
        self.failures
    }

    /// Starts a bounded rollout from the stable revision to `candidate`.
    pub fn start_rollout(
        &mut self,
        candidate: PlanRevision,
        config: RolloutConfig,
    ) -> Result<RolloutEvent, RolloutError> {
        if self.candidate_revision.is_some() {
            return Err(RolloutError::AlreadyInProgress);
        }
        if candidate == self.stable_revision {
            return Err(RolloutError::SameRevision);
        }
        self.candidate_revision = Some(candidate);
        self.state = RolloutState::RollingOut;
        self.target_basis_points = config.target_basis_points;
        self.current_basis_points = 0;
        self.step_basis_points = config.step_basis_points;
        self.max_steps = config.max_steps;
        self.steps = 0;
        self.failure_threshold = config.failure_threshold;
        self.failures = 0;
        Ok(RolloutEvent::Started {
            candidate,
            target_basis_points: config.target_basis_points,
        })
    }

    /// Advances the rollout or rollback by at most one configured step.
    pub fn advance(&mut self) -> Result<RolloutEvent, RolloutError> {
        let candidate = self.candidate_revision.ok_or(RolloutError::NoCandidate)?;
        match self.state {
            RolloutState::RollingOut => {
                if self.steps >= self.max_steps {
                    return Err(RolloutError::NoTransition);
                }
                self.steps = self.steps.saturating_add(1);
                self.current_basis_points = self
                    .current_basis_points
                    .saturating_add(self.step_basis_points)
                    .min(self.target_basis_points);
                if self.current_basis_points == self.target_basis_points {
                    self.state = RolloutState::Active;
                    Ok(RolloutEvent::Activated { candidate })
                } else {
                    Ok(RolloutEvent::Advanced {
                        candidate,
                        basis_points: self.current_basis_points,
                    })
                }
            }
            RolloutState::RollingBack => {
                if self.steps >= self.max_steps {
                    return Err(RolloutError::NoTransition);
                }
                self.steps = self.steps.saturating_add(1);
                self.current_basis_points = self
                    .current_basis_points
                    .saturating_sub(self.step_basis_points);
                if self.current_basis_points == 0 {
                    self.state = RolloutState::RolledBack;
                    self.candidate_revision = None;
                    Ok(RolloutEvent::RolledBack { candidate })
                } else {
                    Ok(RolloutEvent::Advanced {
                        candidate,
                        basis_points: self.current_basis_points,
                    })
                }
            }
            RolloutState::Stable | RolloutState::Active | RolloutState::RolledBack => {
                Err(RolloutError::NoTransition)
            }
        }
    }

    /// Records one failure and starts rollback at the configured threshold.
    pub fn record_failure(&mut self) -> Result<Option<RolloutEvent>, RolloutError> {
        if self.candidate_revision.is_none() {
            return Err(RolloutError::NoCandidate);
        }
        if !matches!(self.state, RolloutState::RollingOut | RolloutState::Active) {
            return Err(RolloutError::NotRollingOut);
        }
        self.failures = self.failures.saturating_add(1);
        if self.failures < self.failure_threshold {
            return Ok(None);
        }
        Ok(Some(self.begin_rollback()?))
    }

    /// Requests rollback without waiting for the failure threshold.
    pub fn request_rollback(&mut self) -> Result<RolloutEvent, RolloutError> {
        self.begin_rollback()
    }

    fn begin_rollback(&mut self) -> Result<RolloutEvent, RolloutError> {
        let candidate = self.candidate_revision.ok_or(RolloutError::NoCandidate)?;
        if !matches!(self.state, RolloutState::RollingOut | RolloutState::Active) {
            return Err(RolloutError::NotRollingOut);
        }
        self.state = RolloutState::RollingBack;
        self.steps = 0;
        Ok(RolloutEvent::RollbackStarted {
            candidate,
            basis_points: self.current_basis_points,
        })
    }

    /// Promotes an active candidate and clears the rollout state.
    pub fn promote(&mut self) -> Result<RolloutEvent, RolloutError> {
        let candidate = self.candidate_revision.ok_or(RolloutError::NoCandidate)?;
        if self.state != RolloutState::Active {
            return Err(RolloutError::NotRollingOut);
        }
        self.stable_revision = candidate;
        self.candidate_revision = None;
        self.state = RolloutState::Stable;
        self.current_basis_points = 0;
        self.target_basis_points = 0;
        self.step_basis_points = 0;
        self.max_steps = 0;
        self.steps = 0;
        self.failure_threshold = 0;
        self.failures = 0;
        Ok(RolloutEvent::Promoted {
            revision: candidate,
        })
    }

    /// Selects the revision for one instance using the current rollout state.
    #[must_use]
    pub fn revision_for(&self, instance: InstanceId) -> PlanRevision {
        let Some(candidate) = self.candidate_revision else {
            return self.stable_revision;
        };
        let candidate_selected = match self.state {
            RolloutState::RollingOut | RolloutState::RollingBack | RolloutState::Active => self
                .selector
                .selected(instance, self.current_basis_points)
                .unwrap_or(false),
            RolloutState::Stable | RolloutState::RolledBack => false,
        };
        if candidate_selected {
            candidate
        } else {
            self.stable_revision
        }
    }

    /// Returns the deterministic group assigned to one instance.
    #[must_use]
    pub fn group_for(&self, instance: InstanceId) -> InstanceGroupId {
        self.selector.group_for(instance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revision(value: u128) -> PlanRevision {
        PlanRevision::new(value)
    }

    fn config() -> RolloutConfig {
        RolloutConfig::new(5_000, 2_500, 2, 2).unwrap()
    }

    #[test]
    fn semver_compatibility_respects_pre_one_major_boundary() {
        let one = SemVer::new(1, 2, 0);
        assert!(one.compatible_with(SemVer::new(1, 4, 1)));
        assert!(!one.compatible_with(SemVer::new(2, 0, 0)));
        let zero = SemVer::new(0, 2, 0);
        assert!(zero.compatible_with(SemVer::new(0, 2, 4)));
        assert!(!zero.compatible_with(SemVer::new(0, 3, 0)));
    }

    #[test]
    fn package_rejects_missing_or_incompatible_contracts() {
        let mut pack = BehaviorPack::new(7, SemVer::new(1, 0, 0), revision(2));
        pack.require_engine(VersionRequirement::compatible(SemVer::new(1, 2, 0)));
        pack.require_dependency(DependencyRequirement {
            id: DependencyId(11),
            version: VersionRequirement::compatible(SemVer::new(1, 2, 0)),
        })
        .unwrap();
        pack.provide_slot(ProvidedSlot::new(SlotGuid(3), SemVer::new(1, 1, 0)))
            .unwrap();
        pack.provide_event(ProvidedEvent::new(EventId(5), SemVer::new(1, 0, 0)))
            .unwrap();

        let mut context = PackLoadContext::new(SemVer::new(1, 0, 0));
        context
            .install_dependency(InstalledDependency {
                id: DependencyId(11),
                version: SemVer::new(2, 0, 0),
            })
            .unwrap();
        context
            .require_slot(ProvidedSlot::new(SlotGuid(3), SemVer::new(1, 0, 0)))
            .unwrap();
        context
            .require_event(ProvidedEvent::new(EventId(5), SemVer::new(1, 0, 0)))
            .unwrap();

        assert_eq!(
            pack.validate_load(&context),
            Err(PackLoadError::EngineVersionMismatch {
                required: VersionRequirement::Compatible(SemVer::new(1, 2, 0)),
                installed: SemVer::new(1, 0, 0),
            })
        );

        let mut context = PackLoadContext::new(SemVer::new(1, 2, 1));
        context
            .install_dependency(InstalledDependency {
                id: DependencyId(11),
                version: SemVer::new(2, 0, 0),
            })
            .unwrap();
        context
            .require_slot(ProvidedSlot::new(SlotGuid(3), SemVer::new(1, 0, 0)))
            .unwrap();
        context
            .require_event(ProvidedEvent::new(EventId(5), SemVer::new(1, 0, 0)))
            .unwrap();
        assert_eq!(
            pack.validate_load(&context),
            Err(PackLoadError::DependencyVersionMismatch {
                id: DependencyId(11),
                required: VersionRequirement::Compatible(SemVer::new(1, 2, 0)),
                installed: SemVer::new(2, 0, 0),
            })
        );
    }

    #[test]
    fn package_accepts_stable_ids_and_declared_plan_migration() {
        let source = revision(1);
        let target = revision(2);
        let mut pack = BehaviorPack::new(7, SemVer::new(1, 0, 0), target);
        pack.provide_slot(ProvidedSlot::new(SlotGuid(3), SemVer::new(1, 1, 0)))
            .unwrap();
        pack.provide_event(ProvidedEvent::new(EventId(5), SemVer::new(1, 0, 0)))
            .unwrap();
        pack.allow_migration_from(source).unwrap();

        let mut context =
            PackLoadContext::new(SemVer::new(1, 0, 0)).with_active_plan_revision(source);
        context
            .require_slot(ProvidedSlot::new(SlotGuid(3), SemVer::new(1, 0, 0)))
            .unwrap();
        context
            .require_event(ProvidedEvent::new(EventId(5), SemVer::new(1, 0, 0)))
            .unwrap();

        let report = pack.validate_load(&context).unwrap();
        assert!(!report.engine_checked);
        assert_eq!(report.plan, PlanCompatibility::Migrated);
        assert_eq!(report.slots_checked, 1);
        assert_eq!(report.events_checked, 1);
    }

    #[test]
    fn duplicate_contracts_are_rejected_before_growth() {
        let mut pack = BehaviorPack::new(1, SemVer::new(1, 0, 0), revision(1));
        let slot = ProvidedSlot::new(SlotGuid(9), SemVer::new(1, 0, 0));
        pack.provide_slot(slot).unwrap();
        assert_eq!(
            pack.provide_slot(slot),
            Err(PackDefinitionError::DuplicateSlot(SlotGuid(9)))
        );
        let event = ProvidedEvent::new(EventId(9), SemVer::new(1, 0, 0));
        pack.provide_event(event).unwrap();
        assert_eq!(
            pack.provide_event(event),
            Err(PackDefinitionError::DuplicateEvent(EventId(9)))
        );
    }

    #[test]
    fn instance_selection_is_deterministic_and_bounded() {
        let selector = InstanceGroupSelector::new(16, 99).unwrap();
        let instance = InstanceId(0xfeed_beef);
        assert_eq!(selector.group_for(instance), selector.group_for(instance));
        assert!(selector.group_for(instance).0 < 16);
        assert_eq!(
            selector.selected(instance, BASIS_POINTS + 1),
            Err(GroupSelectionError::InvalidBasisPoints(BASIS_POINTS + 1))
        );
        assert!(selector.selected(instance, BASIS_POINTS).unwrap());
        assert!(!selector.selected(instance, 0).unwrap());

        let group = selector.group_for(instance);
        let mut same_group = None;
        for value in 0..1_024u128 {
            let candidate = InstanceId(value);
            if candidate != instance && selector.group_for(candidate) == group {
                same_group = Some(candidate);
                break;
            }
        }
        let same_group = same_group.expect("a bounded selector has a repeated group");
        assert_eq!(
            selector.selected(instance, 2_500),
            selector.selected(same_group, 2_500)
        );
    }

    #[test]
    fn rollout_advances_in_bounded_steps_and_selects_groups() {
        let selector = InstanceGroupSelector::new(32, 17).unwrap();
        let mut controller = RolloutController::new(revision(1), selector);
        assert_eq!(controller.revision_for(InstanceId(4)), revision(1));
        assert_eq!(
            controller.start_rollout(revision(2), config()),
            Ok(RolloutEvent::Started {
                candidate: revision(2),
                target_basis_points: 5_000,
            })
        );
        assert_eq!(
            controller.advance(),
            Ok(RolloutEvent::Advanced {
                candidate: revision(2),
                basis_points: 2_500
            })
        );
        assert_eq!(
            controller.advance(),
            Ok(RolloutEvent::Activated {
                candidate: revision(2)
            })
        );
        assert_eq!(controller.state(), RolloutState::Active);
        assert_eq!(controller.basis_points(), 5_000);
        let mut saw_stable = false;
        let mut saw_candidate = false;
        for value in 0..1_024u128 {
            let instance = InstanceId(value);
            let expected = if selector.selected(instance, 5_000).unwrap() {
                saw_candidate = true;
                revision(2)
            } else {
                saw_stable = true;
                revision(1)
            };
            assert_eq!(controller.revision_for(instance), expected);
        }
        assert!(saw_stable && saw_candidate);
    }

    #[test]
    fn failure_threshold_triggers_complete_rollback_without_unbounded_ticks() {
        let selector = InstanceGroupSelector::new(8, 7).unwrap();
        let mut controller = RolloutController::new(revision(10), selector);
        controller.start_rollout(revision(11), config()).unwrap();
        controller.advance().unwrap();
        assert_eq!(controller.record_failure().unwrap(), None);
        assert_eq!(
            controller.record_failure().unwrap(),
            Some(RolloutEvent::RollbackStarted {
                candidate: revision(11),
                basis_points: 2_500
            })
        );
        assert_eq!(
            controller.advance(),
            Ok(RolloutEvent::RolledBack {
                candidate: revision(11)
            })
        );
        assert_eq!(controller.state(), RolloutState::RolledBack);
        assert_eq!(controller.revision_for(InstanceId(2)), revision(10));
        assert_eq!(controller.steps(), 1);
    }

    #[test]
    fn active_candidate_can_be_promoted_once() {
        let selector = InstanceGroupSelector::new(2, 0).unwrap();
        let mut controller = RolloutController::new(revision(1), selector);
        controller.start_rollout(revision(2), config()).unwrap();
        controller.advance().unwrap();
        controller.advance().unwrap();
        assert_eq!(
            controller.promote(),
            Ok(RolloutEvent::Promoted {
                revision: revision(2)
            })
        );
        assert_eq!(controller.state(), RolloutState::Stable);
        assert_eq!(controller.stable_revision(), revision(2));
        assert_eq!(controller.candidate_revision(), None);
    }
}
