//! Deterministic domain lifecycle orchestration.

use std::collections::BTreeMap;
use std::fmt;

use crate::{DiagnosticReport, DomainId, DomainRegistry};

/// Lifecycle state tracked for one registered domain.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DomainState {
    /// Registered but not started.
    Registered,
    /// The start transition is being applied.
    Starting,
    /// Started and available to dependents.
    Active,
    /// The stop transition is being applied.
    Stopping,
    /// Previously active, now stopped.
    Stopped,
}

/// Direction of a lifecycle transition.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LifecycleAction {
    /// Start a domain.
    Start,
    /// Stop a domain.
    Stop,
}

/// One state transition emitted by the lifecycle manager.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleStep {
    /// Domain being transitioned.
    pub domain: DomainId,
    /// Transition direction.
    pub action: LifecycleAction,
    /// State before this transition.
    pub from: DomainState,
    /// State after this transition.
    pub to: DomainState,
}

/// Ordered lifecycle transitions resulting from one operation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LifecycleReport {
    steps: Vec<LifecycleStep>,
}

impl LifecycleReport {
    /// Creates an empty report.
    #[must_use]
    pub const fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Returns the emitted transitions in application order.
    #[must_use]
    pub fn steps(&self) -> &[LifecycleStep] {
        &self.steps
    }

    fn push(&mut self, step: LifecycleStep) {
        self.steps.push(step);
    }
}

/// Errors raised by lifecycle orchestration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleError {
    /// The registry is not valid enough to start or stop domains.
    InvalidRegistry(DiagnosticReport),
    /// An operation referred to an unregistered domain.
    UnknownDomain(DomainId),
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegistry(report) => write!(formatter, "invalid domain registry: {report}"),
            Self::UnknownDomain(domain) => write!(formatter, "unknown domain {domain}"),
        }
    }
}

impl std::error::Error for LifecycleError {}

/// In-memory lifecycle state machine for assembled domains.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DomainRuntime {
    states: BTreeMap<DomainId, DomainState>,
}

impl DomainRuntime {
    /// Creates an empty runtime state store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            states: BTreeMap::new(),
        }
    }

    /// Synchronizes state entries with the current registry.
    pub fn sync(&mut self, registry: &DomainRegistry) {
        self.states
            .retain(|domain, _| registry.get(domain).is_some());
        for descriptor in registry.domains() {
            self.states
                .entry(descriptor.id.clone())
                .or_insert(DomainState::Registered);
        }
    }

    /// Returns a domain's current state, if registered in this runtime.
    #[must_use]
    pub fn state(&self, domain: &DomainId) -> Option<DomainState> {
        self.states.get(domain).copied()
    }

    /// Returns all tracked states in stable domain order.
    #[must_use]
    pub fn states(&self) -> &BTreeMap<DomainId, DomainState> {
        &self.states
    }

    /// Starts every domain in dependency-first order.
    pub fn start_all(
        &mut self,
        registry: &DomainRegistry,
    ) -> Result<LifecycleReport, LifecycleError> {
        let order = registry
            .start_order()
            .map_err(LifecycleError::InvalidRegistry)?;
        self.sync(registry);
        let mut report = LifecycleReport::new();
        for domain in order {
            self.start_one(&domain, &mut report);
        }
        Ok(report)
    }

    /// Starts a domain and every inactive dependency required by it.
    pub fn start_domain(
        &mut self,
        registry: &DomainRegistry,
        target: &DomainId,
    ) -> Result<LifecycleReport, LifecycleError> {
        let order = registry
            .start_order()
            .map_err(LifecycleError::InvalidRegistry)?;
        if registry.get(target).is_none() {
            return Err(LifecycleError::UnknownDomain(target.clone()));
        }
        self.sync(registry);
        let required = dependency_closure(registry, target, false);
        let mut report = LifecycleReport::new();
        for domain in order {
            if required.contains(&domain) {
                self.start_one(&domain, &mut report);
            }
        }
        Ok(report)
    }

    /// Stops every active domain in reverse dependency order.
    pub fn stop_all(
        &mut self,
        registry: &DomainRegistry,
    ) -> Result<LifecycleReport, LifecycleError> {
        let order = registry
            .stop_order()
            .map_err(LifecycleError::InvalidRegistry)?;
        self.sync(registry);
        let mut report = LifecycleReport::new();
        for domain in order {
            self.stop_one(&domain, &mut report);
        }
        Ok(report)
    }

    /// Stops a domain and active dependents before stopping the target.
    pub fn stop_domain(
        &mut self,
        registry: &DomainRegistry,
        target: &DomainId,
    ) -> Result<LifecycleReport, LifecycleError> {
        let order = registry
            .stop_order()
            .map_err(LifecycleError::InvalidRegistry)?;
        if registry.get(target).is_none() {
            return Err(LifecycleError::UnknownDomain(target.clone()));
        }
        self.sync(registry);
        let affected = dependency_closure(registry, target, true);
        let mut report = LifecycleReport::new();
        for domain in order {
            if affected.contains(&domain) {
                self.stop_one(&domain, &mut report);
            }
        }
        Ok(report)
    }

    fn start_one(&mut self, domain: &DomainId, report: &mut LifecycleReport) {
        let current = self
            .states
            .get(domain)
            .copied()
            .unwrap_or(DomainState::Registered);
        if current == DomainState::Active {
            return;
        }
        self.states.insert(domain.clone(), DomainState::Starting);
        report.push(LifecycleStep {
            domain: domain.clone(),
            action: LifecycleAction::Start,
            from: current,
            to: DomainState::Starting,
        });
        self.states.insert(domain.clone(), DomainState::Active);
        report.push(LifecycleStep {
            domain: domain.clone(),
            action: LifecycleAction::Start,
            from: DomainState::Starting,
            to: DomainState::Active,
        });
    }

    fn stop_one(&mut self, domain: &DomainId, report: &mut LifecycleReport) {
        let current = self
            .states
            .get(domain)
            .copied()
            .unwrap_or(DomainState::Registered);
        if current != DomainState::Active {
            return;
        }
        self.states.insert(domain.clone(), DomainState::Stopping);
        report.push(LifecycleStep {
            domain: domain.clone(),
            action: LifecycleAction::Stop,
            from: current,
            to: DomainState::Stopping,
        });
        self.states.insert(domain.clone(), DomainState::Stopped);
        report.push(LifecycleStep {
            domain: domain.clone(),
            action: LifecycleAction::Stop,
            from: DomainState::Stopping,
            to: DomainState::Stopped,
        });
    }
}

fn dependency_closure(
    registry: &DomainRegistry,
    target: &DomainId,
    include_dependents: bool,
) -> std::collections::BTreeSet<DomainId> {
    let mut closure = std::collections::BTreeSet::from([target.clone()]);
    let mut changed = true;
    while changed {
        changed = false;
        for descriptor in registry.domains() {
            let matches = if include_dependents {
                descriptor
                    .dependencies
                    .keys()
                    .any(|dependency| closure.contains(dependency))
            } else {
                closure.contains(&descriptor.id)
            };
            if matches && closure.insert(descriptor.id.clone()) {
                changed = true;
            }
            if !include_dependents && closure.contains(&descriptor.id) {
                for dependency in descriptor.dependencies.keys() {
                    if closure.insert(dependency.clone()) {
                        changed = true;
                    }
                }
            }
        }
    }
    closure
}
