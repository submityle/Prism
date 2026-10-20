//! Generic assembly contracts for Prism editor domains.
//!
//! This crate describes how a domain announces identity, versions, capabilities,
//! editor contributions, dependencies, optional profiles, and lifecycle state.
//! It intentionally does not implement VFX, animation, rendering, graph
//! evaluation, or timeline algorithms. Adapters can consume these descriptors
//! and provide those domain-specific behaviors elsewhere.
//!
//! All registries use ordered collections and all reports use explicit sorting,
//! so registration order does not affect dependency plans or diagnostics.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod lifecycle;
mod model;
mod registry;

pub use lifecycle::{
    DomainRuntime, DomainState, LifecycleAction, LifecycleError, LifecycleReport, LifecycleStep,
};
pub use model::{
    CapabilityDescriptor, CapabilityId, CommandContribution, ContributionId, ContributionKind,
    DiagnosticSeverity, DomainDependency, DomainDescriptor, DomainId, DomainProfile, DomainVersion,
    EditorContribution, GraphNodeContribution, GraphPort, ModelError, PanelContribution,
    PanelSurface, PortDirection, ProfileId, ProfileValue, TimelineContribution,
    ValidatorContribution, VersionConstraint,
};
pub use registry::{
    Diagnostic, DiagnosticCode, DiagnosticReport, DomainRegistration, DomainRegistry,
    ProfileRegistration, RegistryError,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(name: &str) -> DomainDescriptor {
        DomainDescriptor::new(
            DomainId::new(name).expect("valid domain ID"),
            DomainVersion::new(1, 0, 0),
            name,
        )
        .expect("valid descriptor")
    }

    fn contribution(name: &str) -> EditorContribution {
        EditorContribution::Command(
            CommandContribution::new(
                ContributionId::new(name).expect("valid contribution ID"),
                name,
                "test",
            )
            .expect("valid command"),
        )
    }

    #[test]
    fn identifiers_and_versions_are_canonical() {
        assert!(DomainId::new("VFX").is_err());
        assert!(DomainId::new("vfx/").is_err());
        let id = DomainId::new("vfx.core").expect("valid ID");
        assert_eq!(id.to_string(), "vfx.core");
        assert_eq!(
            "1.2.3".parse::<DomainVersion>().expect("valid version"),
            DomainVersion::new(1, 2, 3)
        );
        assert!("01.2.3".parse::<DomainVersion>().is_err());
    }

    #[test]
    fn version_constraints_match_without_string_parsing() {
        let range = VersionConstraint::new(
            Some(DomainVersion::new(1, 2, 0)),
            Some(DomainVersion::new(2, 0, 0)),
        )
        .expect("valid range");
        assert!(range.matches(DomainVersion::new(1, 9, 9)));
        assert!(!range.matches(DomainVersion::new(2, 0, 0)));
        assert!(VersionConstraint::exact(DomainVersion::new(3, 0, 0))
            .matches(DomainVersion::new(3, 0, 0)));
    }

    #[test]
    fn descriptor_sorts_contributions_and_rejects_duplicates() {
        let mut descriptor = domain("editor");
        descriptor
            .add_contribution(contribution("z.command"))
            .expect("first contribution");
        descriptor
            .add_contribution(contribution("a.command"))
            .expect("second contribution");
        assert_eq!(descriptor.contributions[0].id().as_str(), "a.command");
        assert!(descriptor
            .add_contribution(contribution("a.command"))
            .is_err());
    }

    #[test]
    fn graph_ports_are_sorted_and_unique() {
        let mut node = GraphNodeContribution::new(
            ContributionId::new("node").expect("valid ID"),
            "Node",
            "math",
        )
        .expect("valid node");
        let output = GraphPort::new(
            ContributionId::new("z.output").expect("valid ID"),
            "Output",
            PortDirection::Output,
            "scalar",
        )
        .expect("valid port");
        let input = GraphPort::new(
            ContributionId::new("a.input").expect("valid ID"),
            "Input",
            PortDirection::Input,
            "scalar",
        )
        .expect("valid port");
        node.add_port(output.clone()).expect("first port");
        node.add_port(input).expect("second port");
        assert_eq!(node.ports[0].id.as_str(), "a.input");
        assert!(node.add_port(output).is_err());
    }

    #[test]
    fn dependency_order_is_stable_and_reversed_for_stop() {
        let mut registry = DomainRegistry::new();
        let base = domain("base");
        let mut feature = domain("feature");
        feature
            .add_dependency(DomainDependency::new(base.id.clone()))
            .expect("dependency");
        registry.register_domain(feature).expect("feature");
        registry.register_domain(base).expect("base");
        assert_eq!(
            registry.start_order().expect("valid order"),
            vec![
                DomainId::new("base").expect("valid ID"),
                DomainId::new("feature").expect("valid ID")
            ]
        );
        assert_eq!(
            registry.stop_order().expect("valid order"),
            vec![
                DomainId::new("feature").expect("valid ID"),
                DomainId::new("base").expect("valid ID")
            ]
        );
    }

    #[test]
    fn dependency_and_capability_diagnostics_are_explicit() {
        let mut provider = domain("provider");
        provider
            .add_capability(
                CapabilityDescriptor::new(
                    CapabilityId::new("scene").expect("valid ID"),
                    DomainVersion::new(1, 0, 0),
                    "scene access",
                )
                .expect("capability"),
            )
            .expect("capability registration");
        let mut consumer = domain("consumer");
        consumer
            .add_dependency(
                DomainDependency::new(provider.id.clone())
                    .with_version(VersionConstraint::exact(DomainVersion::new(2, 0, 0)))
                    .requiring_capability(
                        CapabilityId::new("missing").expect("valid ID"),
                        VersionConstraint::any(),
                    ),
            )
            .expect("dependency");
        let mut registry = DomainRegistry::new();
        registry.register_domain(consumer).expect("consumer");
        registry.register_domain(provider).expect("provider");
        let report = registry.validate();
        let codes = report
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.code)
            .collect::<Vec<_>>();
        assert!(codes.contains(&DiagnosticCode::DependencyVersionMismatch));
        assert!(codes.contains(&DiagnosticCode::MissingCapability));
        assert!(!report.is_valid());
    }

    #[test]
    fn cycles_produce_sorted_diagnostics_and_block_plans() {
        let mut left = domain("left");
        let mut right = domain("right");
        left.add_dependency(DomainDependency::new(right.id.clone()))
            .expect("dependency");
        right
            .add_dependency(DomainDependency::new(left.id.clone()))
            .expect("dependency");
        let mut registry = DomainRegistry::new();
        registry.register_domain(right).expect("right");
        registry.register_domain(left).expect("left");
        let report = registry.validate();
        assert_eq!(
            report
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.code == DiagnosticCode::DependencyCycle)
                .count(),
            2
        );
        assert!(registry.start_order().is_err());
    }

    #[test]
    fn contribution_collisions_are_reported_across_domains() {
        let mut first = domain("first");
        let mut second = domain("second");
        first
            .add_contribution(contribution("shared.command"))
            .expect("contribution");
        second
            .add_contribution(contribution("shared.command"))
            .expect("contribution");
        let mut registry = DomainRegistry::new();
        registry.register_domain(second).expect("second");
        registry.register_domain(first).expect("first");
        let report = registry.validate();
        assert!(report
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::ContributionConflict));
    }

    #[test]
    fn profile_conflicts_are_reported_and_exact_registration_is_idempotent() {
        let mut first = domain("first");
        let mut second = domain("second");
        let profile_id = ProfileId::new("default").expect("valid profile ID");
        first
            .set_profile(DomainProfile::new(
                profile_id.clone(),
                first.id.clone(),
                DomainVersion::new(1, 0, 0),
            ))
            .expect("profile");
        second
            .set_profile(DomainProfile::new(
                profile_id,
                second.id.clone(),
                DomainVersion::new(1, 0, 0),
            ))
            .expect("profile");
        let mut registry = DomainRegistry::new();
        registry.register_domain(first).expect("first");
        registry.register_domain(second).expect("second");
        assert!(registry
            .validate()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::ProfileConflict));

        let mut clean = DomainRegistry::new();
        clean.register_domain(domain("owner")).expect("owner");
        let profile = DomainProfile::new(
            ProfileId::new("owner.profile").expect("valid profile ID"),
            DomainId::new("owner").expect("valid domain ID"),
            DomainVersion::new(1, 0, 0),
        );
        clean.register_profile(profile.clone()).expect("profile");
        clean.register_profile(profile).expect("idempotent profile");
    }

    #[test]
    fn runtime_starts_dependencies_and_stops_dependents_safely() {
        let mut registry = DomainRegistry::new();
        let base = domain("base");
        let mut feature = domain("feature");
        feature
            .add_dependency(DomainDependency::new(base.id.clone()))
            .expect("dependency");
        registry.register_domain(feature).expect("feature");
        registry.register_domain(base).expect("base");
        let mut runtime = DomainRuntime::new();
        let start = runtime
            .start_domain(&registry, &DomainId::new("feature").expect("valid ID"))
            .expect("start");
        assert_eq!(
            runtime.state(&DomainId::new("base").expect("valid ID")),
            Some(DomainState::Active)
        );
        assert_eq!(
            runtime.state(&DomainId::new("feature").expect("valid ID")),
            Some(DomainState::Active)
        );
        assert_eq!(
            start.steps().first().expect("start step").domain.as_str(),
            "base"
        );

        let stop = runtime
            .stop_domain(&registry, &DomainId::new("base").expect("valid ID"))
            .expect("stop");
        assert_eq!(
            stop.steps().first().expect("stop step").domain.as_str(),
            "feature"
        );
        assert_eq!(
            runtime.state(&DomainId::new("base").expect("valid ID")),
            Some(DomainState::Stopped)
        );
        assert_eq!(
            runtime.state(&DomainId::new("feature").expect("valid ID")),
            Some(DomainState::Stopped)
        );
    }

    #[test]
    fn diagnostics_are_independent_of_registration_order() {
        let mut first = domain("first");
        let mut second = domain("second");
        first
            .add_dependency(DomainDependency::new(
                DomainId::new("missing").expect("valid ID"),
            ))
            .expect("dependency");
        second
            .add_dependency(DomainDependency::new(
                DomainId::new("missing").expect("valid ID"),
            ))
            .expect("dependency");

        let mut left = DomainRegistry::new();
        left.register_domain(first.clone()).expect("first");
        left.register_domain(second.clone()).expect("second");
        let mut right = DomainRegistry::new();
        right.register_domain(second).expect("second");
        right.register_domain(first).expect("first");
        assert_eq!(left.validate(), right.validate());
    }

    #[test]
    fn profile_settings_are_ordered_and_validated() {
        let mut profile = DomainProfile::new(
            ProfileId::new("profile").expect("valid profile ID"),
            DomainId::new("domain").expect("valid domain ID"),
            DomainVersion::new(1, 0, 0),
        );
        profile
            .set("quality", ProfileValue::Integer(3))
            .expect("setting");
        profile
            .set("enabled", ProfileValue::Boolean(true))
            .expect("setting");
        assert_eq!(
            profile.settings.keys().next().expect("setting key"),
            "enabled"
        );
        assert!(profile
            .set("Invalid", ProfileValue::Text("x".to_owned()))
            .is_err());
    }

    #[test]
    fn registry_validates_publicly_assembled_descriptors() {
        let descriptor = DomainDescriptor {
            id: DomainId::new("domain").expect("valid domain ID"),
            version: DomainVersion::new(1, 0, 0),
            display_name: String::new(),
            capabilities: Default::default(),
            dependencies: Default::default(),
            contributions: Vec::new(),
            profile: None,
        };
        let mut registry = DomainRegistry::new();
        assert!(matches!(
            registry.register_domain(descriptor),
            Err(RegistryError::InvalidDescriptor(
                ModelError::EmptyText { .. }
            ))
        ));
    }
}
