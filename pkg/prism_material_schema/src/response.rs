//! Response declarations are data contracts, not shader mathematics.

use alloc::collections::{BTreeMap, BTreeSet};
use serde::Deserialize;

use crate::{ConsumedStep, Field, FieldType, Namespace, SchemaError, Stability, SurfaceSchema};

/// Closed engine-provided programs. Project programs are registered separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinResponse {
    /// Physically based surface response.
    Lit,
    /// Non-photorealistic surface response.
    Npr,
    /// Base color and emission without scene lighting.
    Unlit,
}

impl BuiltinResponse {
    /// Canonical schema ID.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lit => "lit",
            Self::Npr => "npr",
            Self::Unlit => "unlit",
        }
    }
    /// Generated runtime variant, checked by the Rust compiler.
    pub const fn rust_variant(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::Npr => "Npr",
            Self::Unlit => "Unlit",
        }
    }
    /// Associated `consumed_by` program marker.
    pub const fn consumed_step(self) -> ConsumedStep {
        match self {
            Self::Lit => ConsumedStep::Lit,
            Self::Npr => ConsumedStep::Npr,
            Self::Unlit => ConsumedStep::Unlit,
        }
    }
}

/// A field is either semantically interpreted with evidence or explicitly ignored.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum FieldResponse {
    /// Real response interpretation; a decoder assignment is not evidence.
    Mapped {
        /// Concrete behavior affected by the field.
        semantic: String,
        /// Workspace-relative Rust source and executable test name (`path::test`).
        test: String,
    },
    /// An intentional omission from this program's interpretation.
    IntentionallyUnmapped {
        /// Explanation specific to the program/field, not an empty exemption.
        reason: String,
    },
}

/// One registered schema program with all five contracts declared explicitly.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseProgram {
    /// Stable human-readable ID.
    pub id: String,
    /// Globally unique registry identity.
    pub uid: String,
    /// Engine-provided executable family; not a raw project ProgramId.
    pub builtin: BuiltinResponse,
    /// PCM extension digest slot, independent from the dense program index.
    pub registry_slot: u32,
    /// Governance attribution.
    pub namespace: Namespace,
    /// Vendor attribution.
    pub vendor: String,
    /// Stability level.
    pub stability: Stability,
    /// Required capabilities inherited by materials using this program.
    pub requires_caps: Vec<String>,
    /// Whether the result changes with view direction.
    pub view_dependent: bool,
    /// Lobe coverage; field coverage is authored directly on each field.
    pub surface: ProgramSurfaceContract,
    /// Compile-time lighting inputs.
    pub lighting: ProgramLightingContract,
    /// Produced and blendable output channels.
    pub output: ProgramOutputContract,
    /// Permitted execution paths and cost class.
    pub execution: ProgramExecutionContract,
    /// Cache dependencies and temporal behavior.
    pub cache: ProgramCacheContract,
}

/// Exhaustive lobe interpretation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramSurfaceContract {
    /// Lobes whose fields affect this response.
    pub supported_lobes: Vec<String>,
    /// Every other lobe with a substantive reason.
    pub ignored_lobes: BTreeMap<String, String>,
}

/// Lighting terms read by executable code, never per-frame quality switches.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramLightingContract {
    /// Analytic direct lighting.
    pub direct_light: bool,
    /// Shared shadow visibility.
    pub shadows: bool,
    /// Diffuse indirect lighting.
    pub indirect_diffuse: bool,
    /// Reflection lighting.
    pub reflection: bool,
    /// Transmission lighting.
    pub transmission: bool,
    /// Temporal history input.
    pub temporal_history: bool,
}

/// Fixed response channel names, shared with generated runtime manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramChannel {
    /// Direct radiance.
    Direct,
    /// Indirect diffuse radiance.
    IndirectDiffuse,
    /// Reflection radiance.
    Reflection,
    /// Transmission radiance.
    Transmission,
    /// Emission radiance.
    Emission,
    /// Shared geometric alpha.
    Alpha,
    /// Typed auxiliary data.
    Auxiliary,
}

impl ProgramChannel {
    /// Canonical schema spelling used by generated runtime metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::IndirectDiffuse => "indirect_diffuse",
            Self::Reflection => "reflection",
            Self::Transmission => "transmission",
            Self::Emission => "emission",
            Self::Alpha => "alpha",
            Self::Auxiliary => "auxiliary",
        }
    }

    /// Runtime variant spelling.
    pub const fn rust_variant(self) -> &'static str {
        match self {
            Self::Direct => "Direct",
            Self::IndirectDiffuse => "IndirectDiffuse",
            Self::Reflection => "Reflection",
            Self::Transmission => "Transmission",
            Self::Emission => "Emission",
            Self::Alpha => "Alpha",
            Self::Auxiliary => "Auxiliary",
        }
    }
}

/// Output channel contract.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramOutputContract {
    /// Channels the program initializes.
    pub produced_channels: Vec<ProgramChannel>,
    /// Radiance channels permitted in a blend.
    pub blendable_channels: Vec<ProgramChannel>,
}

/// Cost used for scheduling, not pipeline identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ProgramCostClass {
    /// Low-cost response.
    Low,
    /// Medium-cost response.
    Medium,
    /// High-cost response.
    High,
}

impl ProgramCostClass {
    /// Canonical schema spelling used by generated runtime metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Permitted execution paths.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionContract {
    /// Fused built-in pairs are allowed.
    pub supports_fused: bool,
    /// Independent response passes are allowed.
    pub supports_split: bool,
    /// Scheduling cost.
    pub max_cost_class: ProgramCostClass,
}

/// Required temporal behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ProgramTemporalRequirement {
    /// No history requirement.
    None,
    /// History improves quality but is not required.
    Optional,
    /// History is required.
    Required,
}

impl ProgramTemporalRequirement {
    /// Canonical schema spelling used by generated runtime metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Optional => "optional",
            Self::Required => "required",
        }
    }
}

/// Cache dependencies. Built-ins use fixed response channels; project manifests
/// may additionally register typed surface auxiliary channels at runtime.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramCacheContract {
    /// Read response channels.
    pub reads: Vec<ProgramChannel>,
    /// Written response channels.
    pub writes: Vec<ProgramChannel>,
    /// Temporal policy.
    pub temporal: ProgramTemporalRequirement,
}

fn invalid(message: impl Into<String>) -> SchemaError {
    SchemaError::ResponseContract(message.into())
}

impl SurfaceSchema {
    /// Non-padding fields in deterministic core then canonical lobe order.
    pub fn response_fields(&self) -> Vec<(Option<&str>, &Field)> {
        self.core
            .fields
            .iter()
            .filter(|f| f.ty != FieldType::Pad)
            .map(|f| (None, f))
            .chain(self.canonical_lobes().into_iter().flat_map(|lobe| {
                lobe.fields
                    .iter()
                    .filter(|f| f.ty != FieldType::Pad)
                    .map(move |f| (Some(lobe.id.as_str()), f))
            }))
            .collect()
    }

    pub(crate) fn validate_responses(&self) -> Result<(), SchemaError> {
        let mut identities = BTreeSet::new();
        let mut builtins = BTreeSet::new();
        let lobe_ids: BTreeSet<&str> = self.lobes.iter().map(|l| l.id.as_str()).collect();
        for program in &self.response_programs {
            if program.id != program.builtin.as_str()
                || !identities.insert(program.id.as_str())
                || !builtins.insert(program.builtin)
            {
                return Err(invalid(format!(
                    "invalid or duplicate response program {}",
                    program.id
                )));
            }
            let supported: BTreeSet<&str> = program
                .surface
                .supported_lobes
                .iter()
                .map(String::as_str)
                .collect();
            let ignored: BTreeSet<&str> = program
                .surface
                .ignored_lobes
                .keys()
                .map(String::as_str)
                .collect();
            if supported.len() != program.surface.supported_lobes.len()
                || !supported.is_disjoint(&ignored)
                || supported.union(&ignored).copied().collect::<BTreeSet<_>>() != lobe_ids
                || program
                    .surface
                    .ignored_lobes
                    .values()
                    .any(|r| r.trim().is_empty())
            {
                return Err(invalid(format!(
                    "{} must classify every lobe exactly once with nonempty ignore reasons",
                    program.id
                )));
            }
            let produced: BTreeSet<_> = program.output.produced_channels.iter().copied().collect();
            let blendable: BTreeSet<_> =
                program.output.blendable_channels.iter().copied().collect();
            if produced.is_empty()
                || produced.len() != program.output.produced_channels.len()
                || blendable.len() != program.output.blendable_channels.len()
                || !blendable.is_subset(&produced)
                || blendable.contains(&ProgramChannel::Alpha)
                || blendable.contains(&ProgramChannel::Auxiliary)
                || (!program.execution.supports_fused && !program.execution.supports_split)
            {
                return Err(invalid(format!(
                    "{} has invalid output or execution contract",
                    program.id
                )));
            }
            for channels in [&program.cache.reads, &program.cache.writes] {
                if channels.iter().copied().collect::<BTreeSet<_>>().len() != channels.len() {
                    return Err(invalid(format!(
                        "{} has duplicate cache channels",
                        program.id
                    )));
                }
            }
            if !program.cache.writes.iter().all(|c| produced.contains(c)) {
                return Err(invalid(format!(
                    "{} caches a channel it does not produce",
                    program.id
                )));
            }
            if matches!(program.cache.temporal, ProgramTemporalRequirement::Required)
                && !program.lighting.temporal_history
            {
                return Err(invalid(format!(
                    "{} requires undeclared temporal lighting",
                    program.id
                )));
            }
            let mut mapped_lobes = BTreeSet::new();
            for (owner, field) in self.response_fields() {
                let key = format!("{}.{}", owner.unwrap_or("core"), field.name);
                let Some(response) = field.responses.get(&program.id) else {
                    return Err(invalid(format!(
                        "{key} has no {} response declaration",
                        program.id
                    )));
                };
                let mapped = matches!(response, FieldResponse::Mapped { .. });
                if field.consumed_by.contains(&program.builtin.consumed_step()) != mapped {
                    return Err(invalid(format!(
                        "{key} consumed_by disagrees with {} semantic coverage",
                        program.id
                    )));
                }
                match response {
                    FieldResponse::Mapped { semantic, test } => {
                        if semantic.trim().is_empty()
                            || !test.split_once("::").is_some_and(|(path, name)| {
                                path.starts_with("pkg/")
                                    && path.ends_with(".rs")
                                    && !name.trim().is_empty()
                            })
                        {
                            return Err(invalid(format!(
                                "{key}/{} needs semantic and executable test evidence",
                                program.id
                            )));
                        }
                        if let Some(lobe) = owner {
                            if !supported.contains(lobe) {
                                return Err(invalid(format!(
                                    "{key} maps an ignored lobe in {}",
                                    program.id
                                )));
                            }
                            mapped_lobes.insert(lobe);
                        }
                    }
                    FieldResponse::IntentionallyUnmapped { reason } if reason.trim().is_empty() => {
                        return Err(invalid(format!(
                            "{key}/{} has an empty ignore reason",
                            program.id
                        )));
                    }
                    FieldResponse::IntentionallyUnmapped { .. } => {}
                }
            }
            if mapped_lobes != supported {
                return Err(invalid(format!(
                    "{} claims a supported lobe without mapped fields",
                    program.id
                )));
            }
        }
        for (owner, field) in self.response_fields() {
            if field
                .responses
                .keys()
                .any(|key| !identities.contains(key.as_str()))
            {
                return Err(invalid(format!(
                    "{}.{} refers to an unknown response",
                    owner.unwrap_or("core"),
                    field.name
                )));
            }
            let unique: BTreeSet<_> = field.consumed_by.iter().collect();
            if unique.len() != field.consumed_by.len() {
                return Err(invalid(format!("{} has duplicate consumers", field.name)));
            }
        }
        Ok(())
    }
}
