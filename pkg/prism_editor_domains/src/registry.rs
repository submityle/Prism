//! Deterministic domain registration, validation, and dependency ordering.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::model::{ContributionId, DomainDescriptor, DomainId, DomainProfile, VersionConstraint};
use crate::ContributionKind;

/// Stable diagnostic code emitted by domain assembly validation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum DiagnosticCode {
    /// Two domains claim the same contribution surface and identity.
    ContributionConflict,
    /// A dependency refers to an unregistered domain.
    MissingDependency,
    /// A dependency's domain version is outside its accepted range.
    DependencyVersionMismatch,
    /// A required capability is not exposed by a dependency.
    MissingCapability,
    /// A required capability version is outside its accepted range.
    CapabilityVersionMismatch,
    /// The dependency graph contains a cycle.
    DependencyCycle,
    /// Two profile declarations conflict.
    ProfileConflict,
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::ContributionConflict => "contribution-conflict",
            Self::MissingDependency => "missing-dependency",
            Self::DependencyVersionMismatch => "dependency-version-mismatch",
            Self::MissingCapability => "missing-capability",
            Self::CapabilityVersionMismatch => "capability-version-mismatch",
            Self::DependencyCycle => "dependency-cycle",
            Self::ProfileConflict => "profile-conflict",
        };
        formatter.write_str(value)
    }
}

/// One deterministic assembly diagnostic.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Diagnostic {
    /// Diagnostic severity.
    pub severity: crate::DiagnosticSeverity,
    /// Stable diagnostic code.
    pub code: DiagnosticCode,
    /// Domain most directly associated with the issue.
    pub domain: Option<DomainId>,
    /// Stable subject key used for sorting and UI navigation.
    pub subject: String,
    /// Human-readable diagnostic message.
    pub message: String,
}

impl Diagnostic {
    pub(crate) fn error(
        code: DiagnosticCode,
        domain: Option<DomainId>,
        subject: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity: crate::DiagnosticSeverity::Error,
            code,
            domain,
            subject: subject.into(),
            message: message.into(),
        }
    }
}

/// A sorted, immutable-style collection of assembly diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticReport {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticReport {
    /// Creates a report and applies its canonical ordering.
    #[must_use]
    pub fn new(mut diagnostics: Vec<Diagnostic>) -> Self {
        diagnostics.sort();
        diagnostics.dedup();
        Self { diagnostics }
    }

    /// Returns whether no diagnostics were emitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Returns whether the report contains no blocking errors.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == crate::DiagnosticSeverity::Error)
    }

    /// Returns diagnostics in canonical order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Consumes the report and returns its diagnostics.
    #[must_use]
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics
    }
}

impl fmt::Display for DiagnosticReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index != 0 {
                formatter.write_str("\n")?;
            }
            write!(
                formatter,
                "{} [{}] {}: {}",
                match diagnostic.severity {
                    crate::DiagnosticSeverity::Info => "info",
                    crate::DiagnosticSeverity::Warning => "warning",
                    crate::DiagnosticSeverity::Error => "error",
                },
                diagnostic.code,
                diagnostic.subject,
                diagnostic.message
            )?;
        }
        Ok(())
    }
}

/// A successful domain registration result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainRegistration {
    /// Registered domain identity.
    pub domain: DomainId,
    /// Monotonic registry revision after the registration.
    pub revision: u64,
}

/// A successful profile registration result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileRegistration {
    /// Registered profile identity.
    pub profile: crate::ProfileId,
    /// Monotonic registry revision after the registration.
    pub revision: u64,
}

/// Errors raised while changing the domain registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegistryError {
    /// A descriptor failed local structural validation.
    InvalidDescriptor(crate::ModelError),
    /// A domain identity is already registered.
    DuplicateDomain(DomainId),
    /// A profile references a domain that is not registered.
    UnknownDomain(DomainId),
    /// A profile identity is already registered with different contents.
    ProfileConflict(crate::ProfileId),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(error) => {
                write!(formatter, "invalid domain descriptor: {error}")
            }
            Self::DuplicateDomain(domain) => {
                write!(formatter, "domain {domain} is already registered")
            }
            Self::UnknownDomain(domain) => write!(formatter, "unknown domain {domain}"),
            Self::ProfileConflict(profile) => write!(
                formatter,
                "profile {profile} conflicts with an existing registration"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Deterministic registry of domain descriptors and optional profiles.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DomainRegistry {
    domains: BTreeMap<DomainId, DomainDescriptor>,
    profiles: BTreeMap<crate::ProfileId, DomainProfile>,
    revision: u64,
}

impl DomainRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            domains: BTreeMap::new(),
            profiles: BTreeMap::new(),
            revision: 0,
        }
    }

    /// Registers a domain descriptor.
    pub fn register_domain(
        &mut self,
        descriptor: DomainDescriptor,
    ) -> Result<DomainRegistration, RegistryError> {
        descriptor
            .validate()
            .map_err(RegistryError::InvalidDescriptor)?;
        if self.domains.contains_key(&descriptor.id) {
            return Err(RegistryError::DuplicateDomain(descriptor.id));
        }
        let domain = descriptor.id.clone();
        self.domains.insert(domain.clone(), descriptor);
        self.revision = self.revision.saturating_add(1);
        Ok(DomainRegistration {
            domain,
            revision: self.revision,
        })
    }

    /// Registers an optional profile independently of a descriptor profile.
    pub fn register_profile(
        &mut self,
        profile: DomainProfile,
    ) -> Result<ProfileRegistration, RegistryError> {
        if !self.domains.contains_key(&profile.domain) {
            return Err(RegistryError::UnknownDomain(profile.domain));
        }
        if let Some(existing) = self.profiles.get(&profile.id) {
            if existing != &profile {
                return Err(RegistryError::ProfileConflict(profile.id));
            }
            return Ok(ProfileRegistration {
                profile: profile.id,
                revision: self.revision,
            });
        }
        let profile_id = profile.id.clone();
        self.profiles.insert(profile_id.clone(), profile);
        self.revision = self.revision.saturating_add(1);
        Ok(ProfileRegistration {
            profile: profile_id,
            revision: self.revision,
        })
    }

    /// Returns a descriptor by stable domain identity.
    #[must_use]
    pub fn get(&self, domain: &DomainId) -> Option<&DomainDescriptor> {
        self.domains.get(domain)
    }

    /// Returns an independently registered profile by identity.
    #[must_use]
    pub fn profile(&self, profile: &crate::ProfileId) -> Option<&DomainProfile> {
        self.profiles.get(profile)
    }

    /// Returns domains in stable identity order.
    pub fn domains(&self) -> impl Iterator<Item = &DomainDescriptor> {
        self.domains.values()
    }

    /// Returns the current registration revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Validates dependencies, contribution collisions, profiles, and cycles.
    #[must_use]
    pub fn validate(&self) -> DiagnosticReport {
        let mut diagnostics = Vec::new();
        self.validate_dependencies(&mut diagnostics);
        self.validate_contributions(&mut diagnostics);
        self.validate_profiles(&mut diagnostics);
        self.validate_cycles(&mut diagnostics);
        DiagnosticReport::new(diagnostics)
    }

    /// Computes a deterministic dependency-first order.
    pub fn start_order(&self) -> Result<Vec<DomainId>, DiagnosticReport> {
        let report = self.validate();
        if !report.is_valid() {
            return Err(report);
        }
        Ok(self.topological_order())
    }

    /// Computes the reverse of the deterministic dependency-first order.
    pub fn stop_order(&self) -> Result<Vec<DomainId>, DiagnosticReport> {
        let mut order = self.start_order()?;
        order.reverse();
        Ok(order)
    }

    fn validate_dependencies(&self, diagnostics: &mut Vec<Diagnostic>) {
        for descriptor in self.domains.values() {
            for dependency in descriptor.dependencies.values() {
                let Some(required) = self.domains.get(&dependency.domain) else {
                    diagnostics.push(Diagnostic::error(
                        DiagnosticCode::MissingDependency,
                        Some(descriptor.id.clone()),
                        dependency.domain.to_string(),
                        format!("domain {} is not registered", dependency.domain),
                    ));
                    continue;
                };
                if !dependency.version.matches(required.version) {
                    diagnostics.push(Diagnostic::error(
                        DiagnosticCode::DependencyVersionMismatch,
                        Some(descriptor.id.clone()),
                        dependency.domain.to_string(),
                        format!(
                            "requires {} in {}, registered {}",
                            dependency.domain,
                            format_range(dependency.version),
                            required.version
                        ),
                    ));
                }
                for (capability, required_range) in &dependency.capabilities {
                    let Some(exposed) = required.capabilities.get(capability) else {
                        diagnostics.push(Diagnostic::error(
                            DiagnosticCode::MissingCapability,
                            Some(descriptor.id.clone()),
                            format!("{}:{capability}", dependency.domain),
                            format!(
                                "domain {} does not expose capability {capability}",
                                dependency.domain
                            ),
                        ));
                        continue;
                    };
                    if !required_range.matches(exposed.version) {
                        diagnostics.push(Diagnostic::error(
                            DiagnosticCode::CapabilityVersionMismatch,
                            Some(descriptor.id.clone()),
                            format!("{}:{capability}", dependency.domain),
                            format!(
                                "requires capability {capability} in {}, registered {}",
                                format_range(*required_range),
                                exposed.version
                            ),
                        ));
                    }
                }
            }
        }
    }

    fn validate_contributions(&self, diagnostics: &mut Vec<Diagnostic>) {
        let mut owners: BTreeMap<(ContributionKind, ContributionId), DomainId> = BTreeMap::new();
        for descriptor in self.domains.values() {
            for contribution in &descriptor.contributions {
                let key = (contribution.kind(), contribution.id().clone());
                if let Some(owner) = owners.insert(key.clone(), descriptor.id.clone()) {
                    diagnostics.push(Diagnostic::error(
                        DiagnosticCode::ContributionConflict,
                        Some(descriptor.id.clone()),
                        format!("{}:{}", key.0, key.1),
                        format!(
                            "{}:{} is claimed by both {} and {}",
                            key.0, key.1, owner, descriptor.id
                        ),
                    ));
                }
            }
        }
    }

    fn validate_profiles(&self, diagnostics: &mut Vec<Diagnostic>) {
        let mut profiles: BTreeMap<crate::ProfileId, BTreeSet<DomainProfile>> = BTreeMap::new();
        for descriptor in self.domains.values() {
            if let Some(profile) = &descriptor.profile {
                profiles
                    .entry(profile.id.clone())
                    .or_default()
                    .insert(profile.clone());
            }
        }
        for profile in self.profiles.values() {
            profiles
                .entry(profile.id.clone())
                .or_default()
                .insert(profile.clone());
        }
        for (id, variants) in profiles {
            if variants.len() > 1 {
                let owners = variants
                    .iter()
                    .map(|profile| profile.domain.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                diagnostics.push(Diagnostic::error(
                    DiagnosticCode::ProfileConflict,
                    None,
                    id.to_string(),
                    format!("profile {id} has conflicting declarations from {owners}"),
                ));
            }
        }
    }

    fn validate_cycles(&self, diagnostics: &mut Vec<Diagnostic>) {
        let (order, remaining) = self.topological_order_with_remaining();
        if order.len() == self.domains.len() {
            return;
        }
        for domain in remaining {
            diagnostics.push(Diagnostic::error(
                DiagnosticCode::DependencyCycle,
                Some(domain.clone()),
                domain.to_string(),
                format!("domain {domain} participates in a dependency cycle"),
            ));
        }
    }

    fn topological_order(&self) -> Vec<DomainId> {
        self.topological_order_with_remaining().0
    }

    fn topological_order_with_remaining(&self) -> (Vec<DomainId>, Vec<DomainId>) {
        let mut indegree: BTreeMap<DomainId, usize> = self
            .domains
            .keys()
            .cloned()
            .map(|domain| (domain, 0))
            .collect();
        let mut outgoing: BTreeMap<DomainId, BTreeSet<DomainId>> = self
            .domains
            .keys()
            .cloned()
            .map(|domain| (domain, BTreeSet::new()))
            .collect();
        for descriptor in self.domains.values() {
            for dependency in descriptor.dependencies.values() {
                if !self.domains.contains_key(&dependency.domain) {
                    continue;
                }
                *indegree
                    .get_mut(&descriptor.id)
                    .expect("descriptor is indexed") += 1;
                outgoing
                    .get_mut(&dependency.domain)
                    .expect("dependency is indexed")
                    .insert(descriptor.id.clone());
            }
        }

        let mut ready = indegree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(domain, _)| domain.clone())
            .collect::<BTreeSet<_>>();
        let mut order = Vec::with_capacity(self.domains.len());
        while let Some(domain) = ready.pop_first() {
            order.push(domain.clone());
            for dependent in outgoing.get(&domain).expect("domain is indexed") {
                let degree = indegree.get_mut(dependent).expect("dependent is indexed");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(dependent.clone());
                }
            }
        }
        let remaining = indegree
            .into_iter()
            .filter_map(|(domain, degree)| (degree != 0).then_some(domain))
            .collect();
        (order, remaining)
    }
}

fn format_range(range: VersionConstraint) -> String {
    match (range.min_inclusive, range.max_exclusive) {
        (None, None) => "any version".to_owned(),
        (Some(min), None) => format!(">={min}"),
        (None, Some(max)) => format!("<{max}"),
        (Some(min), Some(max)) => format!(">={min}, <{max}"),
    }
}
